"""Export official Ultralytics segmentation results as end-to-end parity fixtures.

Runs the official ``<model>.pt`` checkpoint on the reference image with Ultralytics' default
predict settings (conf 0.25, IoU 0.45, CPU) and writes:

- ``target/<model>-e2e-expected.json``: boxes, confidences, and classes in source-image pixels.
- ``target/<model>-e2e-mask-<i>.png``: each instance mask resampled onto the source-image grid
  with the same letterbox-geometry mapping the Rust runtime uses, so the ignored Rust test can
  compare boolean coverage directly (mask IoU).

Development-time only: Python, PyTorch, and the Ultralytics package are conversion dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


def letterbox_geometry(source_width: int, source_height: int, size: int = 640, stride: int = 32):
    """Replicate Ultralytics' LetterBox(auto=True) geometry: gain and centered padding."""
    gain = min(size / source_height, size / source_width)
    resized_width = round(source_width * gain)
    resized_height = round(source_height * gain)
    pad_x = (size - resized_width) % stride / 2
    pad_y = (size - resized_height) % stride / 2
    return gain, pad_x, pad_y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", required=True, help="seg model id, e.g. yolo11n-seg")
    args = parser.parse_args()
    if not args.model.endswith("-seg"):
        raise ValueError(f"{args.model} is not a -seg model id")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.checkpoint))
    results = model.predict(str(args.image), conf=0.25, iou=0.45, device="cpu", verbose=False)
    result = results[0]

    source_height, source_width = result.orig_shape
    gain, pad_x, pad_y = letterbox_geometry(source_width, source_height)

    detections = []
    masks = result.masks.data.cpu().numpy() if result.masks is not None else []
    if len(masks):
        canvas_height, canvas_width = masks.shape[1], masks.shape[2]
    else:
        canvas_height = canvas_width = 0
    for index in range(len(result.boxes)):
        box = result.boxes.xyxy[index].tolist()
        class_id = int(result.boxes.cls[index])
        canvas_mask = masks[index]
        # Source-pixel sampling of the canvas mask with the shared letterbox mapping.
        source_mask = np.zeros((source_height, source_width), dtype=np.uint8)
        source_x = np.clip(
            (np.arange(source_width) * gain + pad_x + 0.5).astype(np.int64), 0, canvas_width - 1
        )
        source_y = np.clip(
            (np.arange(source_height) * gain + pad_y + 0.5).astype(np.int64), 0, canvas_height - 1
        )
        source_mask = canvas_mask[np.ix_(source_y, source_x)].astype(np.uint8)
        mask_path = args.output_dir / f"{args.model}-e2e-mask-{index}.png"
        Image.fromarray(source_mask * 255).save(mask_path)
        detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": float(result.boxes.conf[index]),
                "box_xyxy_px": box,
                "mask_pixels": int(source_mask.sum()),
                "mask_file": mask_path.name,
            }
        )

    fixture = {
        "format": "boquilens-ultralytics-e2e-seg-v1",
        "model": args.model,
        "source_image": str(args.image),
        "image_size": [source_width, source_height],
        "detections": detections,
    }
    fixture_path = args.output_dir / f"{args.model}-e2e-expected.json"
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {fixture_path}")
    for detection in detections:
        print(
            f"{detection['class_name']:<14} {detection['confidence'] * 100:5.1f}%  "
            f"xyxy_px={[round(value, 1) for value in detection['box_xyxy_px']]}  "
            f"mask_px={detection['mask_pixels']}"
        )


if __name__ == "__main__":
    main()
