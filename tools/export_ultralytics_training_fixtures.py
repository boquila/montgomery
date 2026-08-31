"""Export a deterministic Ultralytics loss/gradient fixture (8.4.117, commit 461196cf0).

The input is a torch batch dictionary captured immediately before ``model.loss``. This avoids
cross-language augmentation RNG assumptions and lets one batch cover empty/crowded/segmentation
cases. Output belongs under ``target/`` and is development-only AGPL-derived parity evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO, __version__ as ultralytics_version


def summary(tensor: torch.Tensor) -> dict[str, object]:
    flat = tensor.detach().float().cpu().reshape(-1)
    count = min(64, flat.numel())
    indexes = torch.linspace(0, max(flat.numel() - 1, 0), count).round().long().unique() if count else []
    return {
        "shape": list(tensor.shape),
        "mean": float(flat.double().mean()) if flat.numel() else 0.0,
        "rms": float(flat.double().square().mean().sqrt()) if flat.numel() else 0.0,
        "samples": [[int(i), float(flat[i])] for i in indexes],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("batch", type=Path, help="torch-saved batch dict consumed by model.loss")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = YOLO(args.checkpoint).model.train().float()
    batch = torch.load(args.batch, map_location="cpu", weights_only=False)
    batch = {key: value.float() if key == "img" else value for key, value in batch.items()}
    model.zero_grad(set_to_none=True)
    total, items = model.loss(batch)
    total.sum().backward()
    gradients = {
        name: summary(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    fixture = {
        "format": "montgomery-ultralytics-training-v1",
        "reference": {"ultralytics": ultralytics_version, "commit": "461196cf0", "torch": torch.__version__},
        "model": args.model,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "batch_sha256": hashlib.sha256(args.batch.read_bytes()).hexdigest(),
        "total_loss": float(total.detach().sum()),
        "loss_items": summary(items),
        "gradients": gradients,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
