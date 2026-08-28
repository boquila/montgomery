"""Export reproducible YOLOv8-cls preprocessing and tensor parity fixtures.

Covers every classify scale (n/s/m/l/x): pass ``--model yolov8n-cls`` with the matching official
checkpoint. Also records the official Ultralytics top-5 prediction for the reference image as the
end-to-end expectation. The generated files belong in ``target/`` and are intentionally not
committed: they are derived from an external Ultralytics checkpoint. The Rust ignored tests
consume them to distinguish image preprocessing drift from model-graph drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms


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
    parser.add_argument("--model", default="yolov8n-cls", help="classify scale being exported")
    args = parser.parse_args()
    if not args.model.startswith("yolov8") or not args.model.endswith("-cls"):
        raise ValueError(f"{args.model} is not a YOLOv8 classify scale")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics' classification inference transform: shortest-edge resize to 224 (anti-aliased
    # bilinear), centered 224x224 crop, ToTensor, identity normalization. PIL is the exact
    # torchvision backend for a .pt model's pickled transforms.
    transforms = classify_transforms(224)
    input_tensor = transforms(Image.open(args.image).convert("RGB"))
    # The composed transform ends in ToTensor ([0, 1] CHW float); store the identical pixels as
    # the uint8 RGB image the Rust tests read back.
    input_image = Image.fromarray(
        (input_tensor.permute(1, 2, 0) * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
    )
    input_path = args.output_dir / f"{args.model}-preprocessed-reference.png"
    input_image.save(input_path)

    model = YOLO(args.checkpoint).model.eval().float()
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        return hook

    # The YOLOv8-cls backbone is layers 0-8 (a pure C2f chain); the Classify head is model.9.
    hooks = [model.model[8].register_forward_hook(capture("backbone_p5"))]
    input_tensor = transforms(Image.open(args.image).convert("RGB")).unsqueeze(0).float()
    with torch.inference_mode():
        probs, logits = model(input_tensor)
    for hook in hooks:
        hook.remove()

    tensors = {
        **captured,
        "logits": logits,
        "probs": probs,
    }
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

    # Official end-to-end expectation for the source image.
    yolo = YOLO(args.checkpoint)
    result = yolo.predict(str(args.image), verbose=False)[0]
    top5 = result.probs.top5
    expectation = {
        "model": args.model,
        "source": str(args.image),
        "top5": [
            {
                "class_id": int(index),
                "name": result.names[index],
                "confidence": float(result.probs.data[index]),
            }
            for index in top5
        ],
    }
    expectation_path = args.output_dir / f"{args.model}-e2e-expected.json"
    expectation_path.write_text(json.dumps(expectation, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {expectation_path}: top1={expectation['top5'][0]}")


if __name__ == "__main__":
    main()
