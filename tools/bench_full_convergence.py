#!/usr/bin/env python3
"""Run matched full-COCO convergence training with Montgomery and Ultralytics.

This is intentionally separate from the fast performance matrix. It refuses partial datasets and
stores every command, log, CSV, and model under ``target/full-convergence``.

    uv run --project tools --locked tools/bench_full_convergence.py --download --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ultralytics.data.utils import check_det_dataset


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "release" / ("montgomery.exe" if os.name == "nt" else "montgomery")
EXPECTED_TRAIN = 118_287
EXPECTED_VAL = 5_000
AUTO_WORKERS = max(1, min(8, ((os.cpu_count() or 2) + 1) // 2))


def image_count(path: Path) -> int:
    if path.is_file() and path.suffix.lower() == ".txt":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def resolve_coco(download: bool) -> tuple[Path, Path, dict[int, str]]:
    resolved = check_det_dataset("coco.yaml", autodownload=download)
    train = Path(resolved["train"])
    val = Path(resolved["val"])
    if not train.is_absolute():
        train = Path(resolved["path"]) / train
    if not val.is_absolute():
        val = Path(resolved["path"]) / val
    train_count, val_count = image_count(train), image_count(val)
    if train_count != EXPECTED_TRAIN or val_count != EXPECTED_VAL:
        raise RuntimeError(
            f"full COCO 2017 required: found train={train_count}, val={val_count}; "
            f"expected {EXPECTED_TRAIN}/{EXPECTED_VAL}. Pass --download to fetch it."
        )
    return train.resolve(), val.resolve(), {int(key): value for key, value in resolved["names"].items()}


def run(command: list[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as output:
        completed = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")
    return elapsed


def final_results(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"training completed without results CSV: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise RuntimeError(f"training produced an empty results CSV: {path}")
    result: dict[str, object] = {}
    for name, value in rows[-1].items():
        if value is None or not value.strip():
            continue
        try:
            result[name] = float(value)
        except ValueError:
            result[name] = value
    return result


def write_report(output: Path, args: argparse.Namespace, results: list[dict[str, object]]) -> None:
    report = {
        "format": "montgomery-full-convergence-v1",
        "dataset": {
            "name": "COCO 2017",
            "train_images": EXPECTED_TRAIN,
            "val_images": EXPECTED_VAL,
        },
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "results": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=AUTO_WORKERS)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--task", choices=("detect", "segment"), action="append")
    parser.add_argument("--framework", choices=("native", "ultralytics"), action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "target" / "full-convergence")
    args = parser.parse_args()
    if not BINARY.is_file():
        raise SystemExit("build first: cargo build --locked --release --features training")
    tasks = args.task or ["detect", "segment"]
    frameworks = args.framework or ["native", "ultralytics"]
    train, val, names = resolve_coco(args.download)
    args.output.mkdir(parents=True, exist_ok=True)
    native_data = args.output / "coco2017.yaml"
    native_data.write_text(
        json.dumps(
            {
                "train": str(train),
                "val": str(val),
                "names": [names[index] for index in range(len(names))],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.prepare_only:
        print(native_data)
        return

    results: list[dict[str, object]] = []
    for task in tasks:
        model = "yolo26n-seg" if task == "segment" else "yolo26n"
        pt = ROOT / "target" / f"{model}.pt"
        state = ROOT / "target" / f"{model}-state.pt"
        if "ultralytics" in frameworks and not pt.exists():
            subprocess.run([sys.executable, "-c", f"from ultralytics import YOLO; YOLO('{model}.pt')"], cwd=ROOT, check=True)
            shutil.move(ROOT / f"{model}.pt", pt)
        if "native" in frameworks and not state.exists():
            if not pt.exists():
                subprocess.run([sys.executable, "-c", f"from ultralytics import YOLO; YOLO('{model}.pt')"], cwd=ROOT, check=True)
                shutil.move(ROOT / f"{model}.pt", pt)
            subprocess.run([sys.executable, str(ROOT / "tools" / "export_ultralytics_state.py"), str(pt), str(state)], cwd=ROOT, check=True)

        if "native" in frameworks:
            command = [
                str(BINARY), "train", "--model", model, "--weights", str(state),
                "--data", str(native_data), "--epochs", str(args.epochs), "--batch", str(args.batch),
                "--imgsz", str(args.imgsz), "--workers", str(args.workers), "--prefetch", "2",
                "--save-period", str(args.save_period), "--project", str(args.output / "native"),
                "--name", f"yolo26n-{task}",
            ]
            command = [str(item) for item in command]
            log = args.output / "logs" / f"native-{task}.log"
            seconds = run(command, log)
            match = re.search(r"Training run:\s*(.+)", log.read_text(encoding="utf-8"))
            if match is None:
                raise RuntimeError(f"native run path is absent from {log}")
            run_dir = Path(match.group(1).strip())
            if not run_dir.is_absolute():
                run_dir = ROOT / run_dir
            csv_path = run_dir / "results.csv"
            results.append(
                {
                    "framework": "native",
                    "task": task,
                    "seconds": seconds,
                    "command": command,
                    "run": str(run_dir),
                    "results_csv": str(csv_path),
                    "final": final_results(csv_path),
                }
            )
            write_report(args.output, args, results)

        if "ultralytics" in frameworks:
            command = [
                sys.executable, str(ROOT / "tools" / "bench_ultralytics_train.py"), str(pt), native_data,
                str(args.output / "ultralytics"), "--name", f"yolo26n-{task}", "--task", task,
                "--epochs", str(args.epochs), "--batch", str(args.batch), "--imgsz", str(args.imgsz),
                "--workers", str(args.workers), "--save-period", str(args.save_period), "--val", "--plots",
            ]
            command = [str(item) for item in command]
            seconds = run(command, args.output / "logs" / f"ultralytics-{task}.log")
            run_dir = args.output / "ultralytics" / f"yolo26n-{task}"
            csv_path = run_dir / "results.csv"
            results.append(
                {
                    "framework": "ultralytics",
                    "task": task,
                    "seconds": seconds,
                    "command": command,
                    "run": str(run_dir),
                    "results_csv": str(csv_path),
                    "final": final_results(csv_path),
                }
            )
            write_report(args.output, args, results)

    print(args.output / "report.json")


if __name__ == "__main__":
    main()
