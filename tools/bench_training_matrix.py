"""Run the reproducible boquilens/Burn vs Ultralytics/PyTorch training matrix.

Run from the repository root with the locked CUDA benchmark environment:

    uv run --locked tools/bench_training_matrix.py

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


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "target" / "release" / ("boquilens.exe" if os.name == "nt" else "boquilens")
ULTRA_SCRIPT = ROOT / "tools" / "bench_ultralytics_train.py"
ULTRA_VALIDATION_SCRIPT = ROOT / "tools" / "validate_ultralytics_training.py"
DATA_SCRIPT = ROOT / "tools" / "prepare_training_benchmark_data.py"
DEFAULT_OUTPUT = ROOT / "target" / "performance-comparison" / "results.json"


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
        return ROOT / "target" / f"{self.model}-state.pt"

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
SCENARIOS = [
    # Family x task coverage.
    *(Scenario(f"{model}-{task}", "family-task", f"{model}-{suffix}" if suffix else model, task,
               224 if task == "classify" else 320, 2, 3)
      for model in ("yolov8n", "yolo11n", "yolo26n")
      for task, suffix in (("classify", "cls"), ("detect", ""), ("segment", "seg"))),
    # YOLO26 scale coverage at fixed task settings.
    *(Scenario(f"yolo26{scale}-{task}", "scale", f"yolo26{scale}-{suffix}" if suffix else f"yolo26{scale}",
               task, 224 if task == "classify" else 320, 2, 3)
      for scale in ("s", "m")
      for task, suffix in (("classify", "cls"), ("detect", ""), ("segment", "seg"))),
    # Batch scaling around the family/task YOLO26n baseline.
    *(Scenario(f"yolo26n-{task}-batch{batch}", "batch", f"yolo26n-{suffix}" if suffix else "yolo26n",
               task, 224 if task == "classify" else 320, batch, 3)
      for task, suffix in (("classify", "cls"), ("detect", ""), ("segment", "seg"))
      for batch in (1, 4)),
    # Detection resolution scaling around the 320 px baseline.
    *(Scenario(f"yolo26n-detect-{imgsz}px", "resolution", "yolo26n", "detect", imgsz, 2, 3)
      for imgsz in (64, 128, 640)),
    # Longer convergence sanity checks.
    *(Scenario(f"yolo26n-{task}-10epochs", "convergence", f"yolo26n-{suffix}" if suffix else "yolo26n",
               task, 224 if task == "classify" else 320, 2, 10)
      for task, suffix in (("classify", "cls"), ("detect", ""), ("segment", "seg"))),
]


def command_for(framework: str, scenario: Scenario, project: Path, name: str) -> list[str]:
    if framework == "native":
        return [
            str(NATIVE),
            "train",
            "--model",
            scenario.model,
            "--weights",
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
        return [float(row["loss"]) for row in rows]
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


def run_once(
    framework: str,
    scenario: Scenario,
    output_root: Path,
    label: str,
    keep_logs: bool = True,
    keep_run: bool = False,
    validate: bool = False,
) -> dict[str, Any]:
    project = output_root / "runs" / framework / scenario.id
    project.mkdir(parents=True, exist_ok=True)
    name = label
    command = command_for(framework, scenario, project, name)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    if keep_logs:
        logs = output_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"{scenario.id}-{framework}-{label}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (logs / f"{scenario.id}-{framework}-{label}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
    if completed.returncode:
        raise RuntimeError(
            f"{scenario.id} {framework} failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout[-3000:]}\nstderr:\n{completed.stderr[-3000:]}"
        )
    if framework == "native":
        run_dir = find_native_run(completed.stdout + completed.stderr)
        metrics = run_dir / "metrics.csv"
        internal_seconds = None
        metadata_path = run_dir / "environment.json"
    else:
        run_dir = project / name
        metrics = run_dir / "results.csv"
        metadata_path = run_dir / "benchmark.json"
        benchmark = json.loads(metadata_path.read_text(encoding="utf-8"))
        internal_seconds = float(benchmark["seconds"])
    framework_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    curve = read_curve(metrics, framework, scenario.task)
    normalized_command = [part.replace(str(ROOT), "<repo>") for part in command]
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
        validation_started = time.perf_counter()
        validation = subprocess.run(
            validation_command,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if validation.returncode:
            raise RuntimeError(
                f"{scenario.id} {framework} validation failed\n"
                f"stdout:\n{validation.stdout[-3000:]}\nstderr:\n{validation.stderr[-3000:]}"
            )
        if framework == "native":
            json_start = validation.stdout.find("{")
            validation_metrics = json.loads(validation.stdout[json_start:])
        else:
            marker = "VALIDATION_JSON="
            line = next(line for line in validation.stdout.splitlines() if line.startswith(marker))
            validation_metrics = json.loads(line.removeprefix(marker))
        result["validation"] = {
            "seconds": time.perf_counter() - validation_started,
            "metrics": validation_metrics,
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


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    seconds = [trial["wall_seconds"] for trial in trials]
    return {
        "median_seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "mean_seconds": statistics.mean(seconds),
        "stdev_seconds": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
    }


def selected_scenarios(names: list[str]) -> list[Scenario]:
    if not names:
        return SCENARIOS
    by_id = {scenario.id: scenario for scenario in SCENARIOS}
    missing = [name for name in names if name not in by_id]
    if missing:
        raise ValueError(f"unknown scenarios: {', '.join(missing)}")
    return [by_id[name] for name in names]


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
    parser.add_argument("--skip-prime", action="store_true")
    parser.add_argument("--keep-runs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for scenario in SCENARIOS:
            print(scenario.id)
        return
    if args.repeats < 1 or args.segmentation_repeats < 1:
        parser.error("repeat counts must be positive")

    scenarios = selected_scenarios(args.scenario)
    subprocess.run([sys.executable, str(DATA_SCRIPT)], cwd=ROOT, check=True)
    validate_assets(scenarios)
    output = args.output.resolve()
    output_root = output.parent
    output_root.mkdir(parents=True, exist_ok=True)
    new_result: dict[str, Any] = {
        "schema": "boquilens-training-comparison-v1",
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
        },
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "source": repository_metadata(),
        "scenarios": [],
    }
    if args.resume and output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("schema") != new_result["schema"]:
            raise ValueError("cannot resume a result with a different schema")
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
            entry["native_prime"] = run_once(
                "native", scenario, output_root, "prime", keep_logs=True, keep_run=args.keep_runs
            )
        repeats = args.segmentation_repeats if scenario.task == "segment" else args.repeats
        entry["repeat_count"] = repeats
        for repeat in range(repeats):
            order = ("native", "ultralytics") if repeat % 2 == 0 else ("ultralytics", "native")
            for framework in order:
                print(f"  trial {repeat + 1}/{repeats}: {framework}", flush=True)
                trial = run_once(
                    framework,
                    scenario,
                    output_root,
                    f"trial-{repeat + 1}",
                    keep_run=args.keep_runs,
                    validate=scenario.group == "convergence" and repeat == 0,
                )
                entry["trials"].setdefault(framework, []).append(trial)
                print(f"    {trial['wall_seconds']:.3f}s", flush=True)
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
