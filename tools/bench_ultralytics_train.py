"""Benchmark an Ultralytics training command used by the performance comparison.

The timer covers model construction, checkpoint loading, training, and epoch checkpoint writes.
Python import time is intentionally outside the internal timer; the matrix harness also records
process wall time, which includes imports for both framework commands.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO, __version__ as ultralytics_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("project", type=Path)
    parser.add_argument("--name", default="imagenet10-loss")
    parser.add_argument("--task", choices=("classify", "detect", "segment"), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--save-period", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU training comparison")

    project = args.project.resolve()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model = YOLO(args.checkpoint)
    model.train(
        data=args.dataset,
        epochs=args.epochs,
        batch=args.batch,
        nbs=args.batch,
        imgsz=args.imgsz,
        device=0,
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=0.0,
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        amp=False,
        auto_augment="randaugment",
        erasing=0.4,
        fliplr=0.5,
        scale=0.5,
        val=args.val,
        save=True,
        save_period=args.save_period,
        plots=args.plots,
        project=project,
        name=args.name,
        exist_ok=True,
        verbose=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = project / args.name / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "ultralytics": ultralytics_version,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
                "task": args.task,
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "workers": args.workers,
                "seed": args.seed,
                "seconds": elapsed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"benchmark_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
