"""Export reproducible YOLOv3-Tiny-U preprocessing and tensor parity fixtures.

The generated files belong in ``target/`` and are intentionally not committed: they are derived
from an external Ultralytics checkpoint. The Rust ignored tests consume them to distinguish image
preprocessing drift from model-graph drift.
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
from ultralytics.utils.ops import xywh2xyxy


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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise FileNotFoundError(args.image)
    source_rgb = source_bgr[..., ::-1]
    prepared_bgr = LetterBox((640, 640), auto=True, stride=32)(image=source_bgr)
    prepared_rgb = prepared_bgr[..., ::-1].copy()

    source_path = args.output_dir / "yolov3-tinyu-source-reference.png"
    input_path = args.output_dir / "yolov3-tinyu-preprocessed-reference.png"
    Image.fromarray(source_rgb).save(source_path)
    Image.fromarray(prepared_rgb).save(input_path)

    model = YOLO(args.checkpoint).model.eval().float()
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        return hook

    hooks = [
        model.model[15].register_forward_hook(capture("body_p5")),
        model.model[19].register_forward_hook(capture("body_p4")),
    ]
    input_tensor = torch.from_numpy(prepared_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.inference_mode():
        decoded, raw = model(input_tensor)
    for hook in hooks:
        hook.remove()

    decoded = decoded.permute(0, 2, 1).contiguous()
    tensors = {
        **captured,
        "raw_boxes": raw["boxes"],
        "raw_scores": raw["scores"],
        "decoded_boxes_xyxy": xywh2xyxy(decoded[..., :4]),
        "decoded_scores": decoded[..., 4:],
    }
    fixture = {
        "format": "montgomery-ultralytics-golden-v1",
        "model": "yolov3-tinyu",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "input_sha256": file_sha256(input_path),
        "tensors": {name: summarize(tensor) for name, tensor in tensors.items()},
    }
    fixture_path = args.output_dir / "yolov3-tinyu-golden-v1.json"
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {fixture_path}")
    for name, tensor in tensors.items():
        print(f"{name}: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
