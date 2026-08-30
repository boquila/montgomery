"""Export official YOLOX 0.1.1rc0 one-batch loss and gradient fixtures.

``batch`` is a torch file containing ``images`` ([B,3,H,W], raw 0..255 RGB) and ``targets``
([B,M,5], class/cx/cy/w/h pixels). The official source checkout is assembled through the same
Apache-2.0 shim as ``export_yolox_fixtures.py``. Generated output belongs under ``target/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from export_yolox_fixtures import SCALE_CONFIGS, build_shim, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("batch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="yolox-nano", choices=sorted(SCALE_CONFIGS))
    parser.add_argument("--yolox-repo", type=Path, default=Path("target/yolox-ref/YOLOX-0.1.1rc0"))
    args = parser.parse_args()
    shim = args.output.parent / "yolox-training-ref-shim"
    build_shim(args.yolox_repo, shim)
    sys.path.insert(0, str(shim))
    from yolox.models.yolo_head import YOLOXHead
    from yolox.models.yolo_pafpn import YOLOPAFPN
    from yolox.models.yolox import YOLOX

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    scale = SCALE_CONFIGS[args.model]
    model = YOLOX(
        YOLOPAFPN(scale["depth"], scale["width"], in_channels=[256, 512, 1024],
                  depthwise=scale["depthwise"], act="silu"),
        YOLOXHead(80, scale["width"], in_channels=[256, 512, 1024],
                  depthwise=scale["depthwise"], act="silu"),
    ).train().float()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    batch = torch.load(args.batch, map_location="cpu", weights_only=False)
    model.zero_grad(set_to_none=True)
    losses = model(batch["images"].float(), batch["targets"].float())
    total = losses["total_loss"] if isinstance(losses, dict) else losses[0]
    total.backward()
    fixture = {
        "format": "boquilens-yolox-training-v1",
        "reference": {"tag": "0.1.1rc0", "torch": torch.__version__},
        "model": args.model,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "batch_sha256": hashlib.sha256(args.batch.read_bytes()).hexdigest(),
        "losses": {key: float(value.detach()) for key, value in losses.items()},
        "gradients": {
            name: summarize(parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
