"""Export reproducible YOLO11 preprocessing and tensor parity fixtures.

Covers every detect and segmentation scale (n/s/m/l/x): pass ``--model yolo11s`` or
``--model yolo11s-seg`` with the matching checkpoint. Generated files belong in ``target/`` and
are not committed. The Rust ignored tests use them to separate preprocessing drift from graph
drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="yolo11n", help="detect or seg scale being exported")
    args = parser.parse_args()
    if not args.model.startswith("yolo11"):
        raise ValueError(f"{args.model} is not a YOLO11 detect/seg scale")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise FileNotFoundError(args.image)
    source_rgb = source_bgr[..., ::-1]
    prepared_bgr = LetterBox((640, 640), auto=True, stride=32)(image=source_bgr)
    prepared_rgb = prepared_bgr[..., ::-1].copy()

    source_path = args.output_dir / f"{args.model}-source-reference.png"
    input_path = args.output_dir / f"{args.model}-preprocessed-reference.png"
    Image.fromarray(source_rgb).save(source_path)
    Image.fromarray(prepared_rgb).save(input_path)

    model = YOLO(args.checkpoint).model.eval().float()
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        return hook

    hooks = [
        model.model[16].register_forward_hook(capture("body_p3")),
        model.model[19].register_forward_hook(capture("body_p4")),
        model.model[22].register_forward_hook(capture("body_p5")),
    ]
    input_tensor = torch.from_numpy(prepared_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    is_seg = args.model.endswith("-seg")
    with torch.inference_mode():
        result = model(input_tensor)
    if is_seg:
        # Segment.forward returns ((y, proto), preds): y already carries the 32 raw mask
        # coefficients after the class scores, and preds["mask_coefficient"] is the same tower
        # output before the concatenation.
        (y, protos), preds = result
    else:
        y, preds = result
    for hook in hooks:
        hook.remove()

    # The classic YOLO11 head concatenates DFL-decoded center-size boxes with sigmoid scores into
    # y ([batch, 4 + nc (+ nm), anchors]); the raw one2many towers are exposed through preds.
    decoded_boxes = y[:, :4, :].permute(0, 2, 1).contiguous()
    decoded_scores = y[:, 4 : 4 + model.nc, :].permute(0, 2, 1).contiguous()
    tensors = {
        **captured,
        "raw_boxes": preds["boxes"],
        "raw_scores": preds["scores"],
        "decoded_boxes_cxcywh": decoded_boxes,
        "decoded_scores": decoded_scores,
    }
    if is_seg:
        tensors["protos"] = protos
        tensors["mask_coeffs"] = preds["mask_coefficient"]
    fixture = {
        "format": "boquilens-ultralytics-golden-v1",
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
