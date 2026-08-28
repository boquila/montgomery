#!/usr/bin/env python3
"""Export parameter-injected augmentation fixtures from the pinned sibling Ultralytics tree.

Generated data belongs under target/ and must not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "461196cf09175b64c9b9bd8babebf081c0540520"
PINNED_ENVIRONMENT = {
    "python": "3.11.15",
    "torch": "2.13.0+cpu",
    "torchvision": "0.28.0+cpu",
    "pillow": "12.3.0",
    "opencv": "5.0.0",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sibling_tree(repo: Path) -> Path:
    tree = (repo.parent / "ultralytics").resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise SystemExit(f"sibling Ultralytics is {commit}, expected {PINNED_COMMIT}")
    sys.path.insert(0, str(tree))
    return tree


def synthetic_image(np):
    yy, xx = np.mgrid[:73, :109]
    return np.stack(((xx * 3) % 256, (yy * 5) % 256, (xx + yy) % 256), axis=2).astype(np.uint8)


def write_fixture(args) -> None:
    repo = Path(__file__).resolve().parents[1]
    sibling_tree(repo)
    import cv2
    import numpy as np
    import torch
    import torchvision
    from ultralytics.data.augment import LetterBox, RandomHSV
    from ultralytics.utils.instance import Instances

    actual_environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": package_version("Pillow"),
        "opencv": cv2.__version__,
    }
    if actual_environment != PINNED_ENVIRONMENT:
        raise SystemExit(
            f"fixture environment {actual_environment!r} does not match pin {PINNED_ENVIRONMENT!r}"
        )

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR) if args.input else synthetic_image(np)
    if image is None:
        raise SystemExit(f"could not decode {args.input}")
    output = args.output / args.fixture
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "input-primary.png"
    cv2.imwrite(str(input_path), image)
    labels = {
        "img": image.copy(),
        "cls": np.array([[0.0]], dtype=np.float32),
        "instances": Instances(
            bboxes=np.array([[5.0, 7.0, image.shape[1] - 3.0, image.shape[0] - 4.0]], dtype=np.float32),
            segments=np.array([], dtype=np.float32).reshape(0, 1000, 2),
            bbox_format="xyxy",
            normalized=False,
        ),
    }
    if args.transform == "letterbox":
        params = {"new_shape": [args.size, args.size], "scaleup": False}
        result = LetterBox(new_shape=(args.size, args.size), scaleup=False)(labels)
        tolerance = "bounded-pixels"
    elif args.transform == "hsv":
        np.random.seed(args.seed)
        params = {"hgain": 0.015, "sgain": 0.7, "vgain": 0.4, "numpy_seed": args.seed}
        result = RandomHSV(**{k: params[k] for k in ("hgain", "sgain", "vgain")})(labels)
        tolerance = "exact"
    else:
        raise SystemExit(f"unsupported fixture transform {args.transform}")
    output_path = output / "output.png"
    cv2.imwrite(str(output_path), result["img"])
    instances = result["instances"]
    annotations = {
        "classes": result["cls"].reshape(-1).tolist(),
        "boxes": instances.bboxes.tolist(),
        "bbox_format": instances._bboxes.format,
        "normalized": instances.normalized,
    }
    (output / "input-annotations.json").write_text(
        json.dumps({"classes": [0], "boxes": [[5, 7, image.shape[1] - 3, image.shape[0] - 4]]}, indent=2)
    )
    (output / "output-annotations.json").write_text(json.dumps(annotations, indent=2))
    (output / "params.json").write_text(json.dumps(params, indent=2))
    manifest = {
        "ultralytics_commit": PINNED_COMMIT,
        "python": platform.python_version(),
        "packages": {
            "ultralytics": package_version("ultralytics"),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": package_version("Pillow"),
        },
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "fixture_generator": sha256(Path(__file__)),
        "transform": args.transform,
        "source_hashes": {"input-primary.png": sha256(input_path)},
        "output_hash": sha256(output_path),
        "dtype": "uint8",
        "layout": "HWC",
        "color_order": "BGR",
        "dimensions": list(result["img"].shape),
        "parameters": params,
        "partner_indexes": [],
        "tolerance": tolerance,
        "host_pid": os.getpid(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="synthetic-letterbox")
    parser.add_argument("--transform", choices=("letterbox", "hsv"), default="letterbox")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("target/augmentation-fixtures"))
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    write_fixture(parser.parse_args())


if __name__ == "__main__":
    main()
