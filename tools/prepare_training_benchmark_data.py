"""Prepare deterministic, matched local datasets for the training benchmark matrix.

The source fixtures are intentionally small. This script repeats their real images and labels with
hard links (copy fallback) so each timed run performs enough optimizer steps to measure training
rather than only process startup. Generated data belongs under target/ and is not committed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "target" / "training-quality"
OUTPUT = ROOT / "target" / "performance-comparison" / "data"


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_classification() -> None:
    source = SOURCE / "imagenet10-data"
    destination = OUTPUT / "imagenet10"
    for split, repetitions in (("train", 4), ("val", 1)):
        for class_dir in sorted((source / split).iterdir()):
            if not class_dir.is_dir():
                continue
            for image in sorted(path for path in class_dir.iterdir() if path.is_file()):
                for repetition in range(repetitions):
                    name = f"{image.stem}-r{repetition:02d}{image.suffix.lower()}"
                    link_or_copy(image, destination / split / class_dir.name / name)
    manifest = OUTPUT / "imagenet10.yaml"
    manifest.write_text(
        "path: imagenet10\n"
        "train: train\n"
        "val: val\n"
        "format: classification-folders\n",
        encoding="utf-8",
    )


def prepare_yolo(task: str, source_name: str, output_name: str) -> None:
    source = SOURCE / source_name
    destination = OUTPUT / output_name
    for split, repetitions in (("train", 8), ("val", 1)):
        for image in sorted(path for path in (source / "images" / split).iterdir() if path.is_file()):
            label = source / "labels" / split / f"{image.stem}.txt"
            for repetition in range(repetitions):
                stem = f"{image.stem}-r{repetition:02d}"
                link_or_copy(image, destination / "images" / split / f"{stem}{image.suffix.lower()}")
                if label.exists():
                    link_or_copy(label, destination / "labels" / split / f"{stem}.txt")
    names_line = next(
        line
        for line in (SOURCE / ("coco8-seg-local.yaml" if task == "segment" else "coco8-local.yaml"))
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("names:")
    )
    manifest = OUTPUT / f"{output_name}.yaml"
    manifest.write_text(
        f"path: {destination.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "format: yolo\n"
        f"{names_line}\n",
        encoding="utf-8",
    )


def main() -> None:
    prepare_classification()
    prepare_yolo("detect", "detect-data/coco8", "coco8")
    prepare_yolo("segment", "segment-data/coco8-seg", "coco8-seg")
    train_counts = {
        "classify": len(list((OUTPUT / "imagenet10" / "train").glob("*/*"))),
        "detect": len(list((OUTPUT / "coco8" / "images" / "train").glob("*"))),
        "segment": len(list((OUTPUT / "coco8-seg" / "images" / "train").glob("*"))),
    }
    print(", ".join(f"{task}={count}" for task, count in train_counts.items()))


if __name__ == "__main__":
    main()
