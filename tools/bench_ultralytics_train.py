"""Benchmark the Ultralytics classification training loop used by the README comparison."""

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
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
        imgsz=224,
        device=0,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=0.0,
        cos_lr=True,
        seed=0,
        deterministic=True,
        pretrained=True,
        amp=False,
        auto_augment="randaugment",
        erasing=0.4,
        fliplr=0.5,
        scale=0.5,
        val=False,
        save=True,
        save_period=1,
        plots=False,
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
                "epochs": args.epochs,
                "batch": args.batch,
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
