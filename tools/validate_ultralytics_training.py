"""Validate an Ultralytics training checkpoint and emit a compact JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--task", choices=("classify", "detect", "segment"), required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    metrics = YOLO(args.checkpoint).val(
        data=args.dataset,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        amp=False,
        plots=False,
        verbose=False,
    )
    summary = {key: float(value) for key, value in metrics.results_dict.items()}
    print("VALIDATION_JSON=" + json.dumps({"task": args.task, "metrics": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
