#!/usr/bin/env python3
"""Profile WGPU command aggregation on representative YOLO26 training workloads.

Run from the repository root:

    uv run --project tools tools/profile_wgpu_training.py

The script uses one release binary and isolated processes because CubeCL reads
``CUBECL_WGPU_MAX_TASKS`` when the WGPU device is first initialized.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target" / "release" / ("montgomery.exe" if os.name == "nt" else "montgomery")


@dataclass(frozen=True)
class Workload:
    name: str
    model: str
    data: str
    weights: str
    imgsz: int
    batch: int


WORKLOADS = (
    Workload("batch1-detect", "yolo26n", "coco8.yaml", "yolo26n.bpk", 320, 1),
    Workload("batch1-segment", "yolo26n-seg", "coco8-seg.yaml", "yolo26n-seg.bpk", 320, 1),
    Workload("highres-detect", "yolo26n", "coco8.yaml", "yolo26n.bpk", 640, 2),
    Workload("medium-segment", "yolo26m-seg", "coco8-seg.yaml", "yolo26m-seg.bpk", 320, 2),
)


PROFILE = re.compile(
    r"training-profile epoch=(?P<epoch>\d+) batches=(?P<batches>\d+) "
    r"data_ms=(?P<data>[\d.]+) forward_ms=(?P<forward>[\d.]+) "
    r"backward_ms=(?P<backward>[\d.]+) optimizer_ms=(?P<optimizer>[\d.]+) "
    r"after_step_ms=(?P<after_step>[\d.]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--tasks", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workload", action="append", choices=[item.name for item in WORKLOADS])
    parser.add_argument("--output", type=Path, default=ROOT / "target" / "wgpu-training-profile.json")
    args = parser.parse_args()
    selected = [item for item in WORKLOADS if not args.workload or item.name in args.workload]
    if not args.binary.is_file():
        raise SystemExit(f"release binary is missing: {args.binary}")

    output_root = args.output.parent / "wgpu-training-profile-runs"
    results: list[dict[str, object]] = []
    for workload in selected:
        for tasks_max in args.tasks:
            for repeat in range(args.repeats):
                command = [
                str(args.binary), "train", "--model", str(ROOT / "target" / workload.weights),
                "--data", str(ROOT / "target" / "performance-comparison" / "data" / workload.data),
                "--epochs", "1", "--batch", str(workload.batch), "--imgsz", str(workload.imgsz),
                "--workers", "4", "--prefetch", "2", "--seed", "0",
                "--project", str(output_root), "--name", f"{workload.name}-tasks{tasks_max}-r{repeat + 1}",
                "--no-val", "--no-export",
                ]
                environment = os.environ.copy()
                environment["CUBECL_WGPU_MAX_TASKS"] = str(tasks_max)
                environment["MONTGOMERY_PROFILE_TRAINING"] = "1"
                started = time.perf_counter()
                completed = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True, encoding="utf-8",
                    errors="replace", capture_output=True, check=False,
                )
                wall = time.perf_counter() - started
                combined = completed.stdout + "\n" + completed.stderr
                match = PROFILE.search(combined)
                result: dict[str, object] = {
                "workload": workload.name,
                "model": workload.model,
                "imgsz": workload.imgsz,
                "batch": workload.batch,
                "tasks_max": tasks_max,
                "repeat": repeat + 1,
                "wall_seconds": wall,
                "returncode": completed.returncode,
                }
                if match:
                    result.update({key: float(value) for key, value in match.groupdict().items()})
                if completed.returncode or not match:
                    result["tail"] = combined[-3000:]
                results.append(result)
                print(f"{workload.name:18} tasks={tasks_max:3} repeat={repeat + 1} wall={wall:8.3f}s", flush=True)
                if completed.returncode:
                    raise SystemExit(completed.returncode)

    payload = {
        "format": "montgomery-wgpu-training-profile-v1",
        "binary": str(args.binary),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
