#!/usr/bin/env python3
"""Convert a trusted upstream YOLO checkpoint into tensor-only state for Burn import.

The same bridge handles official Ultralytics ``.pt`` and YOLOX ``.pth`` checkpoints. Python and
PyTorch are development-time dependencies only; Montgomery runtime commands consume ``.bpk``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


def tensor_state(value: object) -> Mapping[str, torch.Tensor]:
    if hasattr(value, "state_dict"):
        value = value.state_dict()
    if not isinstance(value, Mapping) or not value:
        raise TypeError("checkpoint model payload is not a non-empty state dict or module")
    invalid = [name for name, tensor in value.items() if not isinstance(name, str) or not torch.is_tensor(tensor)]
    if invalid:
        raise TypeError("checkpoint model payload contains non-tensor state entries")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="trusted upstream .pt or .pth checkpoint")
    parser.add_argument("output", type=Path, help="tensor-only .pt state for pack-weights")
    args = parser.parse_args()

    # Official checkpoints contain pickled model classes. Only run this converter on trusted files.
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root is not a mapping")
    payload = checkpoint.get("ema") or checkpoint.get("model")
    if payload is None:
        raise KeyError("checkpoint has neither an 'ema' nor a 'model' payload")
    state = {
        name: tensor.detach().float().cpu()
        for name, tensor in tensor_state(payload).items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, args.output)
    parameters = sum(tensor.numel() for tensor in state.values())
    print(f"wrote {len(state)} tensors / {parameters:,} values to {args.output}")


if __name__ == "__main__":
    main()
