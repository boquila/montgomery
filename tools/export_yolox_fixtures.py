"""Export reproducible YOLOX preprocessing and tensor parity fixtures.

Covers every YOLOX scale (nano/tiny/s/m/l/x): pass ``--model yolox-s`` with the matching official
checkpoint from the Megvii-BaseDetection/YOLOX release. Unlike the Ultralytics families, the
reference forward runs the *official* YOLOX PyTorch module sources (Apache-2.0): the script
assembles a small importable package from a plain YOLOX repository checkout (see ``--yolox-repo``)
under ``target/`` instead of pip-installing the project, which would require building its C++
extensions.

The generated files belong in ``target/`` and are intentionally not committed: they are derived
from an external checkpoint. The Rust ignored tests consume them to verify Burn's graph numerics
against the official implementation at the same preprocessed input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Official per-scale hyperparameters, mirroring exps/default/*.py at tag 0.1.1rc0 (the release the
# official checkpoints were trained with). Only nano uses depthwise convolutions.
SCALE_CONFIGS = {
    "yolox-nano": {"depth": 0.33, "width": 0.25, "depthwise": True},
    "yolox-tiny": {"depth": 0.33, "width": 0.375, "depthwise": False},
    "yolox-s": {"depth": 0.33, "width": 0.50, "depthwise": False},
    "yolox-m": {"depth": 0.67, "width": 0.75, "depthwise": False},
    "yolox-l": {"depth": 1.00, "width": 1.00, "depthwise": False},
    "yolox-x": {"depth": 1.33, "width": 1.25, "depthwise": False},
}

# Copied verbatim (except for this docstring) from the YOLOX checkout's yolox/utils/boxes.py so
# the shim does not need torchvision; bboxes_iou is only used by the head's training path.
SHIM_BBOXES_IOU = '''"""Dev-time shim: bboxes_iou copied from the official YOLOX checkout
(yolox/utils/boxes.py, Apache-2.0) so yolox.models.yolo_head imports without the full package."""

import torch


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)
'''

# yolo_head.py imports `from loguru import logger` at module level; the logger is only used in the
# training loss path, so a no-op stub keeps the shim offline.
SHIM_LOGURU = '''"""Dev-time shim standing in for the loguru package (unused at inference time)."""


class _Logger:
    def _noop(self, *args, **kwargs):
        pass

    error = warning = info = debug = success = _noop


logger = _Logger()
'''

SHIM_YOLOX_INIT = ""
SHIM_MODELS_INIT = ""
SHIM_UTILS_INIT = "from .boxes import bboxes_iou\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(tensor: torch.Tensor) -> dict[str, object]:
    flat = tensor.detach().float().cpu().contiguous().view(-1)
    sample_count = min(128, flat.numel())
    indices = torch.linspace(0, flat.numel() - 1, sample_count).round().long().unique()
    values = flat.double()
    return {
        "shape": list(tensor.shape),
        "mean": values.mean().item(),
        "rms": values.square().mean().sqrt().item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "samples": [[int(index), float(flat[index])] for index in indices],
    }


def build_shim(yolox_repo: Path, shim_dir: Path) -> None:
    """Assemble the importable package from the official checkout plus tiny shims."""
    models_src = yolox_repo / "yolox" / "models"
    models_dst = shim_dir / "yolox" / "models"
    utils_dst = shim_dir / "yolox" / "utils"
    models_dst.mkdir(parents=True, exist_ok=True)
    utils_dst.mkdir(parents=True, exist_ok=True)

    for name in ("network_blocks.py", "darknet.py", "yolo_pafpn.py", "yolo_head.py", "yolox.py",
                 "losses.py"):
        shutil.copyfile(models_src / name, models_dst / name)
    (shim_dir / "yolox" / "__init__.py").write_text(SHIM_YOLOX_INIT, encoding="utf-8")
    (models_dst / "__init__.py").write_text(SHIM_MODELS_INIT, encoding="utf-8")
    (utils_dst / "__init__.py").write_text(SHIM_UTILS_INIT, encoding="utf-8")
    (utils_dst / "boxes.py").write_text(SHIM_BBOXES_IOU, encoding="utf-8")
    (shim_dir / "loguru.py").write_text(SHIM_LOGURU, encoding="utf-8")


def yolox_letterbox(image_bgr: np.ndarray, size: int = 640) -> np.ndarray:
    """YOLOX's inference transform: resize to fit, anchor top-left, pad with 114."""
    height, width = image_bgr.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = int(width * scale), int(height * scale)
    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:resized_h, :resized_w] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="yolox-nano", help="YOLOX scale being exported")
    parser.add_argument(
        "--yolox-repo",
        type=Path,
        default=Path("target/yolox-ref/YOLOX-0.1.1rc0"),
        help="plain checkout of the official YOLOX repository matching the checkpoint",
    )
    args = parser.parse_args()
    if args.model not in SCALE_CONFIGS:
        raise ValueError(f"{args.model} is not a YOLOX scale: {sorted(SCALE_CONFIGS)}")
    if not args.yolox_repo.exists():
        raise FileNotFoundError(
            f"{args.yolox_repo} is missing; download "
            "https://github.com/Megvii-BaseDetection/YOLOX/archive/refs/tags/0.1.1rc0.zip "
            "and extract it there"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shim_dir = args.output_dir / "yolox-ref-shim"
    build_shim(args.yolox_repo, shim_dir)
    import sys

    sys.path.insert(0, str(shim_dir))
    from yolox.models.yolo_head import YOLOXHead  # noqa: E402
    from yolox.models.yolo_pafpn import YOLOPAFPN  # noqa: E402
    from yolox.models.yolox import YOLOX  # noqa: E402

    scale = SCALE_CONFIGS[args.model]
    model = YOLOX(
        YOLOPAFPN(
            scale["depth"],
            scale["width"],
            in_channels=[256, 512, 1024],
            depthwise=scale["depthwise"],
            act="silu",
        ),
        YOLOXHead(80, scale["width"], in_channels=[256, 512, 1024],
                  depthwise=scale["depthwise"], act="silu"),
    )

    checkpoint_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_state["model"] if "model" in checkpoint_state else checkpoint_state
    model.load_state_dict(state, strict=True)
    model = model.eval().float()

    source_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise FileNotFoundError(args.image)
    source_rgb = source_bgr[..., ::-1]
    prepared_rgb = yolox_letterbox(source_bgr)[..., ::-1].copy()

    source_path = args.output_dir / f"{args.model}-source-reference.png"
    input_path = args.output_dir / f"{args.model}-preprocessed-reference.png"
    Image.fromarray(source_rgb).save(source_path)
    Image.fromarray(prepared_rgb).save(input_path)

    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                # YOLOPAFPN returns (p3, p4, p5); suffix with the feature-map stage.
                for index, tensor in enumerate(output):
                    captured[f"{name}_p{index + 3}"] = tensor.detach()
            else:
                captured[name] = output.detach()

        return hook

    backbone = model.backbone.backbone  # YOLOPAFPN -> CSPDarknet
    hooks = [
        backbone.dark3.register_forward_hook(capture("backbone_dark3")),
        backbone.dark4.register_forward_hook(capture("backbone_dark4")),
        backbone.dark5.register_forward_hook(capture("backbone_dark5")),
        model.backbone.register_forward_hook(capture("pafpn")),
    ]
    input_tensor = torch.from_numpy(prepared_rgb).permute(2, 0, 1).unsqueeze(0).float()
    with torch.inference_mode():
        decoded = model(input_tensor)
    for hook in hooks:
        hook.remove()
    captured["head_decoded"] = decoded

    tensors = {
        "backbone_dark3": captured["backbone_dark3"],
        "backbone_dark4": captured["backbone_dark4"],
        "backbone_dark5": captured["backbone_dark5"],
        "pafpn_p3": captured["pafpn_p3"],
        "pafpn_p4": captured["pafpn_p4"],
        "pafpn_p5": captured["pafpn_p5"],
        "head_decoded": captured["head_decoded"],
    }
    fixture = {
        "format": "boquilens-yolox-golden-v1",
        "model": args.model,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "input_sha256": file_sha256(input_path),
        "tensors": {name: summarize(tensor) for name, tensor in tensors.items()},
    }
    fixture_path = args.output_dir / f"{args.model}-golden-v1.json"
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {fixture_path}")
    for name, tensor in tensors.items():
        print(f"{name}: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
