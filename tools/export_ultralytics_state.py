"""Convert a full Ultralytics checkpoint into a tensor-only state dict for Burn import.

This is a development/build-time bridge. Python and PyTorch are not runtime dependencies of
Montgomery. The output preserves the original parameter keys and contains no optimizer or model
object metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics.nn.tasks import torch_safe_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint, _ = torch_safe_load(args.input)
    model = checkpoint.get("ema") or checkpoint["model"]
    state = {name: tensor.detach().float().cpu() for name, tensor in model.state_dict().items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, args.output)
    parameters = sum(tensor.numel() for tensor in state.values())
    print(f"wrote {len(state)} tensors / {parameters:,} values to {args.output}")
    for name in state:
        print(name)


if __name__ == "__main__":
    main()
