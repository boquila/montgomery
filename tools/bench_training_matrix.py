"""Run the reproducible Montgomery/Burn vs Ultralytics/PyTorch training matrix.

Run from the repository root with the locked CUDA benchmark environment:

    uv run --project tools tools/bench_training_matrix.py

Generated checkpoints and logs stay under target/. The compact JSON result is consumed by
tools/plot_training_comparison.py and is suitable for copying into docs/assets/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmark_resources import ResourceMonitor


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target" / "release" / ("montgomery.exe" if os.name == "nt" else "montgomery")
ULTRA_SCRIPT = ROOT / "tools" / "bench_ultralytics_train.py"
ULTRA_VALIDATION_SCRIPT = ROOT / "tools" / "validate_ultralytics_training.py"
DATA_SCRIPT = ROOT / "tools" / "prepare_training_benchmark_data.py"
DEFAULT_OUTPUT = ROOT / "target" / "performance-comparison" / "results.json"


class ObservedCommandError(RuntimeError):
    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


def repository_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        metadata["revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        metadata["dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        metadata["revision"] = None
        metadata["dirty"] = None
    if NATIVE.is_file():
        metadata["native_binary"] = str(NATIVE)
        digest = hashlib.sha256()
        with NATIVE.open("rb") as binary:
            for chunk in iter(lambda: binary.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata["native_binary_sha256"] = digest.hexdigest()
    return metadata


@dataclass(frozen=True)
class Scenario:
    id: str
    group: str
    model: str
    task: str
    imgsz: int
    batch: int
    epochs: int

    @property
    def native_weights(self) -> Path:
        return ROOT / "target" / f"{self.model}.bpk"

    @property
    def ultralytics_weights(self) -> Path:
        return ROOT / "target" / f"{self.model}.pt"

    @property
    def native_data(self) -> Path:
        names = {
            "classify": "imagenet10.yaml",
            "detect": "coco8.yaml",
            "segment": "coco8-seg.yaml",
        }
        return ROOT / "target" / "performance-comparison" / "data" / names[self.task]

    @property
    def ultralytics_data(self) -> Path:
        if self.task == "classify":
            return ROOT / "target" / "performance-comparison" / "data" / "imagenet10"
        return self.native_data


# Each axis has an explicit baseline, so plots do not silently compare several variables at once.
# Classification retains its architecture-defined 224 px input. Detection and segmentation use a
# real-world 640 px baseline, with the complete nano-family matrix repeated at 1280 px.
SCENARIOS = [
    # Family x task coverage.
    *(Scenario(f"{model}-{task}", "family-task", f"{model}-{suffix}" if suffix else model, task,
               224 if task == "classify" else 640, 2, 3)
      for model in ("yolov8n", "yolo11n", "yolo26n")
      for task, suffix in (("classify", "cls"), ("detect", ""), ("segment", "seg"))),
    # Detect-only families at the same settings as the other nano detectors.
    *(Scenario(f"{model}-detect", "family-task", model, "detect", 640, 2, 3)
      for model in ("yolov3-tinyu", "yolov10n", "yolo12n")),
    # Repeat every nano detection and segmentation comparison at 1280 px.
    *(Scenario(f"{model}-{task}-1280px", "resolution", f"{model}-{suffix}" if suffix else model,
               task, 1280, 2, 3)
      for model in ("yolov8n", "yolo11n", "yolo26n")
      for task, suffix in (("detect", ""), ("segment", "seg"))),
    *(Scenario(f"{model}-detect-1280px", "resolution", model, "detect", 1280, 2, 3)
      for model in ("yolov3-tinyu", "yolov10n", "yolo12n")),
    # YOLO26 scale coverage at fixed task settings.
    *(Scenario(f"yolo26{scale}-{task}", "scale", f"yolo26{scale}-{suffix}" if suffix else f"yolo26{scale}",
               task, 224 if task == "classify" else 640, 2, 3)
      for scale in ("s", "m")
      for task, suffix in (("classify", "cls"), ("detect", ""), ("segment", "seg"))),
    # Batch scaling around the family/task YOLO26n baseline.
    *(Scenario(f"yolo26n-{task}-batch{batch}", "batch", f"yolo26n-{suffix}" if suffix else "yolo26n",
               task, 224 if task == "classify" else 640, batch, 3)
      for task, suffix in (("classify", "cls"), ("detect", ""), ("segment", "seg"))
      for batch in (1, 4)),
    # Longer convergence sanity checks.
    *(Scenario(f"yolo26n-{task}-10epochs", "convergence", f"yolo26n-{suffix}" if suffix else "yolo26n",
               task, 224 if task == "classify" else 640, 2, 10)
      for task, suffix in (("classify", "cls"), ("detect", ""), ("segment", "seg"))),
]


def command_for(framework: str, scenario: Scenario, project: Path, name: str) -> list[str]:
    if framework == "native":
        return [
            str(NATIVE),
            "train",
            "--model",
            str(scenario.native_weights),
            "--data",
            str(scenario.native_data),
            "--epochs",
            str(scenario.epochs),
            "--batch",
            str(scenario.batch),
            "--imgsz",
            str(scenario.imgsz),
            "--workers",
            "4",
            "--prefetch",
            "2",
            "--seed",
            "0",
            "--project",
            str(project),
            "--name",
            name,
            "--no-val",
            "--no-export",
            "--save-period",
            "1",
        ]
    return [
        sys.executable,
        str(ULTRA_SCRIPT),
        str(scenario.ultralytics_weights),
        str(scenario.ultralytics_data),
        str(project),
        "--name",
        name,
        "--task",
        scenario.task,
        "--epochs",
        str(scenario.epochs),
        "--batch",
        str(scenario.batch),
        "--imgsz",
        str(scenario.imgsz),
        "--workers",
        "4",
        "--seed",
        "0",
    ]


def read_curve(path: Path, framework: str, task: str) -> list[float]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if framework == "native":
        return [float(row["train/loss"]) for row in rows]
    if task == "classify":
        return [float(row["train/loss"]) for row in rows]
    keys = ["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss"]
    return [sum(float(row[key]) for key in keys if key in row and row[key]) for row in rows]


def find_native_run(stdout: str) -> Path:
    match = re.search(r"Training run:\s*(.+)", stdout)
    if not match:
        raise RuntimeError("native command did not print its run directory")
    path = Path(match.group(1).strip())
    return path if path.is_absolute() else ROOT / path


def run_observed(
    command: list[str], resource_interval_seconds: float | None
) -> tuple[subprocess.CompletedProcess[str], float, dict[str, Any] | None]:
    """Run a command while observing its complete process tree."""
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    monitor = (
        ResourceMonitor(process.pid, resource_interval_seconds)
        if resource_interval_seconds is not None
        else None
    )
    if monitor is not None:
        monitor.start()
    try:
        stdout, stderr = process.communicate()
        wall_seconds = time.perf_counter() - started
    finally:
        resources = monitor.stop() if monitor is not None else None
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return completed, wall_seconds, resources


def run_once(
    framework: str,
    scenario: Scenario,
    output_root: Path,
    label: str,
    keep_logs: bool = True,
    keep_run: bool = False,
    validate: bool = False,
    resource_interval_seconds: float | None = 0.1,
) -> dict[str, Any]:
    project = output_root / "runs" / framework / scenario.id
    project.mkdir(parents=True, exist_ok=True)
    name = label
    command = command_for(framework, scenario, project, name)
    completed, wall_seconds, resources = run_observed(command, resource_interval_seconds)
    if keep_logs:
        logs = output_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"{scenario.id}-{framework}-{label}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (logs / f"{scenario.id}-{framework}-{label}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if resources is not None:
            (logs / f"{scenario.id}-{framework}-{label}.resources.json").write_text(
                json.dumps(resources, indent=2) + "\n", encoding="utf-8"
            )
    if completed.returncode:
        message = (
            f"{scenario.id} {framework} failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout[-3000:]}\nstderr:\n{completed.stderr[-3000:]}"
        )
        raise ObservedCommandError(
            message,
            {
                "phase": "training",
                "scenario": asdict(scenario),
                "framework": framework,
                "label": label,
                "returncode": completed.returncode,
                "wall_seconds": wall_seconds,
                "command": [part.replace(str(ROOT), "<repo>") for part in command],
                "resources": resources,
            },
        )
    normalized_command = [part.replace(str(ROOT), "<repo>") for part in command]
    try:
        if framework == "native":
            run_dir = find_native_run(completed.stdout + completed.stderr)
            metrics = run_dir / "results.csv"
            internal_seconds = None
            metadata_path = run_dir / "environment.json"
        else:
            run_dir = project / name
            metrics = run_dir / "results.csv"
            metadata_path = run_dir / "benchmark.json"
            benchmark = json.loads(metadata_path.read_text(encoding="utf-8"))
            internal_seconds = float(benchmark["seconds"])
        framework_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        curve = read_curve(metrics, framework, scenario.task)
    except Exception as error:
        raise ObservedCommandError(
            f"{scenario.id} {framework} output parsing failed: {error}",
            {
                "phase": "training-output",
                "scenario": asdict(scenario),
                "framework": framework,
                "label": label,
                "returncode": completed.returncode,
                "wall_seconds": wall_seconds,
                "command": normalized_command,
                "resources": resources,
                "error": f"{type(error).__name__}: {error}",
            },
        ) from error
    result = {
        "framework": framework,
        "label": label,
        "wall_seconds": wall_seconds,
        "internal_seconds": internal_seconds,
        "command": normalized_command,
        "framework_metadata": framework_metadata,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "loss_curve": curve,
        "final_loss": curve[-1] if curve else None,
        "resources": resources,
    }
    if validate:
        if framework == "native":
            validation_command = [
                str(NATIVE),
                "val",
                "--checkpoint",
                str(run_dir / "checkpoints" / "last"),
                "--json",
            ]
        else:
            validation_command = [
                sys.executable,
                str(ULTRA_VALIDATION_SCRIPT),
                str(run_dir / "weights" / "last.pt"),
                str(scenario.ultralytics_data),
                "--task",
                scenario.task,
                "--imgsz",
                str(scenario.imgsz),
                "--batch",
                str(scenario.batch),
            ]
        validation, validation_seconds, validation_resources = run_observed(
            validation_command, resource_interval_seconds
        )
        if keep_logs:
            validation_prefix = f"{scenario.id}-{framework}-{label}.validation"
            (logs / f"{validation_prefix}.stdout.log").write_text(
                validation.stdout, encoding="utf-8"
            )
            (logs / f"{validation_prefix}.stderr.log").write_text(
                validation.stderr, encoding="utf-8"
            )
            if validation_resources is not None:
                (logs / f"{validation_prefix}.resources.json").write_text(
                    json.dumps(validation_resources, indent=2) + "\n", encoding="utf-8"
                )
        if validation.returncode:
            message = (
                f"{scenario.id} {framework} validation failed ({validation.returncode})\n"
                f"stdout:\n{validation.stdout[-3000:]}\nstderr:\n{validation.stderr[-3000:]}"
            )
            raise ObservedCommandError(
                message,
                {
                    "phase": "validation",
                    "scenario": asdict(scenario),
                    "framework": framework,
                    "label": label,
                    "returncode": validation.returncode,
                    "wall_seconds": validation_seconds,
                    "command": [
                        part.replace(str(ROOT), "<repo>") for part in validation_command
                    ],
                    "resources": validation_resources,
                    "completed_training": result,
                },
            )
        try:
            if framework == "native":
                json_start = validation.stdout.find("{")
                validation_metrics = json.loads(validation.stdout[json_start:])
            else:
                marker = "VALIDATION_JSON="
                line = next(
                    line
                    for line in validation.stdout.splitlines()
                    if line.startswith(marker)
                )
                validation_metrics = json.loads(line.removeprefix(marker))
        except Exception as error:
            raise ObservedCommandError(
                f"{scenario.id} {framework} validation output parsing failed: {error}",
                {
                    "phase": "validation-output",
                    "scenario": asdict(scenario),
                    "framework": framework,
                    "label": label,
                    "returncode": validation.returncode,
                    "wall_seconds": validation_seconds,
                    "command": [
                        part.replace(str(ROOT), "<repo>") for part in validation_command
                    ],
                    "resources": validation_resources,
                    "error": f"{type(error).__name__}: {error}",
                    "completed_training": result,
                },
            ) from error
        result["validation"] = {
            "seconds": validation_seconds,
            "metrics": validation_metrics,
            "resources": validation_resources,
        }
    if not keep_run:
        resolved_run = run_dir.resolve()
        resolved_output = output_root.resolve()
        if not resolved_run.is_relative_to(resolved_output):
            raise RuntimeError(f"refusing to remove run outside benchmark output: {resolved_run}")
        shutil.rmtree(resolved_run)
        result["run_dir_retained"] = False
    else:
        result["run_dir_retained"] = True
    return result


def persist_failure(
    result: dict[str, Any], output: Path, error: ObservedCommandError
) -> None:
    result.setdefault("failures", []).append(error.record)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def validate_assets(scenarios: list[Scenario]) -> None:
    if not NATIVE.exists():
        raise FileNotFoundError(f"release training binary not found: {NATIVE}")
    for scenario in scenarios:
        for path in (
            scenario.native_weights,
            scenario.ultralytics_weights,
            scenario.native_data,
            scenario.ultralytics_data,
        ):
            if not path.exists():
                raise FileNotFoundError(path)


def summarize_values(values: list[float | int]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "sample_count": len(values),
    }


def resource_values(
    trials: list[dict[str, Any]], section: str, metric: str
) -> list[float | int]:
    values = []
    for trial in trials:
        resources = trial.get("resources")
        value = resources.get(section, {}).get(metric) if resources else None
        if value is not None:
            values.append(value)
    return values


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    seconds = [trial["wall_seconds"] for trial in trials]
    return {
        "median_seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "mean_seconds": statistics.mean(seconds),
        "stdev_seconds": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
        "resources": {
            "peak_process_count": summarize_values(
                resource_values(trials, "process_tree", "peak_process_count")
            ),
            "peak_rss_bytes": summarize_values(
                resource_values(trials, "process_tree", "peak_rss_bytes")
            ),
            "peak_private_bytes": summarize_values(
                resource_values(trials, "process_tree", "peak_private_bytes")
            ),
            "peak_gpu_dedicated_bytes": summarize_values(
                resource_values(trials, "gpu", "peak_dedicated_bytes")
            ),
            "peak_gpu_shared_bytes": summarize_values(
                resource_values(trials, "gpu", "peak_shared_bytes")
            ),
        },
    }


def selected_scenarios(names: list[str]) -> list[Scenario]:
    if not names:
        return SCENARIOS
    by_id = {scenario.id: scenario for scenario in SCENARIOS}
    missing = [name for name in names if name not in by_id]
    if missing:
        raise ValueError(f"unknown scenarios: {', '.join(missing)}")
    return [by_id[name] for name in names]


def validate_resume_compatibility(
    existing: dict[str, Any], requested: dict[str, Any]
) -> None:
    if existing.get("schema") != requested["schema"]:
        raise ValueError("cannot resume a result with a different schema")
    if existing.get("methodology") != requested["methodology"]:
        raise ValueError("cannot resume a result with different benchmark settings")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--segmentation-repeats",
        type=int,
        default=2,
        help="repeats for the substantially slower segmentation cells",
    )
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument(
        "--native-binary",
        type=Path,
        help="benchmark this native executable instead of target/release/montgomery",
    )
    parser.add_argument("--skip-prime", action="store_true")
    parser.add_argument("--keep-runs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resource-sample-ms",
        type=float,
        default=100.0,
        help="process-tree RAM and GPU-memory sampling interval (default: 100 ms)",
    )
    parser.add_argument(
        "--no-resource-monitor",
        action="store_true",
        help="disable per-trial RAM and GPU-memory telemetry",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.native_binary is not None:
        global NATIVE
        NATIVE = args.native_binary.resolve()
    if not NATIVE.is_file():
        parser.error(f"native executable does not exist: {NATIVE}")
    if args.list:
        for scenario in SCENARIOS:
            print(scenario.id)
        return
    if args.repeats < 1 or args.segmentation_repeats < 1:
        parser.error("repeat counts must be positive")
    if args.resource_sample_ms <= 0:
        parser.error("--resource-sample-ms must be positive")
    resource_interval_seconds = (
        None if args.no_resource_monitor else args.resource_sample_ms / 1000.0
    )

    scenarios = selected_scenarios(args.scenario)
    subprocess.run([sys.executable, str(DATA_SCRIPT)], cwd=ROOT, check=True)
    validate_assets(scenarios)
    output = args.output.resolve()
    output_root = output.parent
    output_root.mkdir(parents=True, exist_ok=True)
    new_result: dict[str, Any] = {
        "schema": "montgomery-training-comparison-v3",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "methodology": {
            "timer": "external process wall clock",
            "repeats": args.repeats,
            "segmentation_repeats": args.segmentation_repeats,
            "native_prime_before_measurement": not args.skip_prime,
            "alternating_framework_order": True,
            "workers": 4,
            "native_prefetch": 2,
            "seed": 0,
            "precision": "FP32",
            "optimizer": "AdamW",
            "validation_in_timed_region": False,
            "post_training_validation": "first trial of each ten-epoch convergence scenario",
            "checkpoint_each_epoch": True,
            "classification_imgsz": 224,
            "vision_baseline_imgsz": 640,
            "vision_resolution_sweep": [640, 1280],
            "resource_monitoring": {
                "enabled": not args.no_resource_monitor,
                "target_sample_interval_ms": args.resource_sample_ms,
                "scope": "training/validation root process plus recursive descendants",
                "ram": "simultaneous process-tree RSS and private/USS peaks",
                "gpu": (
                    "per-process dedicated and shared counters where the operating system or "
                    "driver exposes them; never substituted with whole-device usage"
                ),
                "timing_effect": (
                    "final observer shutdown is outside command wall time; concurrent sampling "
                    "work is not corrected for"
                ),
            },
            "data_loading": {
                "workers_requested_each": 4,
                "native": {
                    "worker_model": "threads",
                    "prefetch_scope": "global prepared-batch queue",
                    "prefetch_batches": 2,
                },
                "ultralytics": {
                    "worker_model": "PyTorch processes",
                    "pin_memory": True,
                    "prefetch_scope": "per worker",
                    "prefetch_tasks_per_worker": 4,
                    "maximum_queued_tasks": 16,
                },
                "interpretation": (
                    "end-to-end operational comparison; process model and buffering are measured "
                    "implementation differences, not normalized internals"
                ),
            },
        },
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "source": repository_metadata(),
        "scenarios": [],
        "failures": [],
    }
    if args.resume and output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        validate_resume_compatibility(result, new_result)
        result.setdefault("failures", [])
    else:
        result = new_result
    completed_ids = {entry["scenario"]["id"] for entry in result["scenarios"]}

    for index, scenario in enumerate(scenarios, start=1):
        if scenario.id in completed_ids:
            print(f"[{index}/{len(scenarios)}] {scenario.id}: already complete", flush=True)
            continue
        print(f"[{index}/{len(scenarios)}] {scenario.id}", flush=True)
        entry: dict[str, Any] = {"scenario": asdict(scenario), "trials": {}}
        if not args.skip_prime:
            print("  priming native WGPU kernels", flush=True)
            try:
                entry["native_prime"] = run_once(
                    "native",
                    scenario,
                    output_root,
                    "prime",
                    keep_logs=True,
                    keep_run=args.keep_runs,
                    resource_interval_seconds=resource_interval_seconds,
                )
            except ObservedCommandError as error:
                persist_failure(result, output, error)
                raise
        repeats = args.segmentation_repeats if scenario.task == "segment" else args.repeats
        entry["repeat_count"] = repeats
        for repeat in range(repeats):
            order = ("native", "ultralytics") if repeat % 2 == 0 else ("ultralytics", "native")
            for framework in order:
                print(f"  trial {repeat + 1}/{repeats}: {framework}", flush=True)
                try:
                    trial = run_once(
                        framework,
                        scenario,
                        output_root,
                        f"trial-{repeat + 1}",
                        keep_run=args.keep_runs,
                        validate=scenario.group == "convergence" and repeat == 0,
                        resource_interval_seconds=resource_interval_seconds,
                    )
                except ObservedCommandError as error:
                    persist_failure(result, output, error)
                    raise
                entry["trials"].setdefault(framework, []).append(trial)
                usage = trial.get("resources")
                if usage:
                    ram = usage["process_tree"]["peak_rss_bytes"]
                    vram = usage["gpu"]["peak_dedicated_bytes"]
                    details = []
                    if ram is not None:
                        details.append(f"RAM {ram / 1024**3:.2f} GiB")
                    if vram is not None:
                        details.append(f"VRAM {vram / 1024**3:.2f} GiB")
                    elif not usage["gpu"]["available"]:
                        details.append("VRAM unavailable")
                    if usage["sampling"]["monitoring_errors"] or usage["gpu"][
                        "sampling_errors"
                    ]:
                        details.append("telemetry warning")
                    suffix = f" | {', '.join(details)}" if details else ""
                else:
                    suffix = ""
                print(f"    {trial['wall_seconds']:.3f}s{suffix}", flush=True)
        entry["summary"] = {
            framework: summarize(entry["trials"][framework])
            for framework in ("native", "ultralytics")
        }
        native = entry["summary"]["native"]["median_seconds"]
        ultra = entry["summary"]["ultralytics"]["median_seconds"]
        entry["summary"]["native_over_ultralytics"] = native / ultra
        result["scenarios"].append(entry)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
