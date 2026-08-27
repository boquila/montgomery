# boquilens agent guide

This directory is the active Rust/Burn object-detection crate. Work from this directory unless a
task explicitly targets the sibling vendored projects.

## What is here

- `src/models/yolox/`: stable YOLOX-Nano implementation and Apache-2.0 checkpoint path.
- `src/models/yolov3_tiny/`: experimental YOLOv3-Tiny-Ultralytics implementation, including the
  body, split DFL head, native Burnpack loading, and version metadata.
- `src/models/yolov10/`: experimental YOLOv10n implementation, including the C2f/SCDown/SPPF/PSA/
  C2fCIB body, NMS-free one2one head, native Burnpack loading, and version metadata.
- `src/data/letterbox.rs`: model-specific preprocessing and reversible source-image geometry.
- `src/lib.rs`: `ModelId`, `Predictor`, detection results, NMS integration, and weight packing API.
- `src/main.rs`: the `predict` and `pack-weights` CLI commands.
- `tools/`: development-only Ultralytics checkpoint conversion and golden-fixture generators.

## Fast path

Stable YOLOX inference:

```console
cargo run --release -- predict --model yolox-nano --source assets/dog_bike_man.jpg
```

YOLOv3-Tiny-U and YOLOv10n require a tensor-only state and then a native artifact. The complete
workflow is documented in `README.md`; the short form is:

```console
python tools/export_ultralytics_state.py yolov3-tinyu.pt target/yolov3-tinyu-state.pt
cargo run --release -- pack-weights --model yolov3-tinyu --input target/yolov3-tinyu-state.pt --output target/yolov3-tinyu-coco-ultralytics-v8.4-boquilens-v1.bpk
cargo run --release -- predict --model yolov3-tinyu --weights target/yolov3-tinyu-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg

python tools/export_ultralytics_state.py yolov10n.pt target/yolov10n-state.pt
cargo run --release -- pack-weights --model yolov10n --input target/yolov10n-state.pt --output target/yolov10n-coco-ultralytics-v8.4-boquilens-v1.bpk
cargo run --release -- predict --model yolov10n --weights target/yolov10n-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg
```

Python/PyTorch is a development-time conversion dependency only. Normal inference is Rust/Burn.

## Verification

Run before handing off changes:

```console
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --no-default-features --lib
```

When the external checkpoint and fixtures are present, also run the ignored parity tests:

```console
cargo test --locked -- --ignored
```

Regenerate those fixtures with `tools/export_ultralytics_fixtures.py`. Generated checkpoints,
fixtures, images, and build output belong under `target/` and must not be committed.

## Invariants

- Public detections are continuous, unnormalized source-image `XYXY` box edges in pixels. They are
  not `XYWH` and not normalized; `xmax == width` and `ymax == height` are valid.
- YOLOX uses its existing top-left/raw-pixel transform. The Ultralytics-family models use
  Ultralytics-style stride-aligned rectangular letterboxing and RGB values normalized to `[0, 1]`.
- YOLOv10n is NMS-free: its one2one head output is top-300 selected and confidence-filtered like
  Ultralytics' end-to-end postprocess, not passed through non-maximum suppression.
- Keep model graph code independent of CLI, filesystem, rendering, and image decoding.
- Keep `ModelId` and CLI model names synchronized when adding a model.
- Preserve the explicit stable/experimental distinction in user-facing docs.

## Licensing boundary

boquilens is AGPL-3.0 (decided 2026-08). The YOLOX path is a derivative of Apache-2.0 code and uses
the Apache-2.0 option; YOLOX and its official weights are Apache-2.0. Ultralytics architectures and
official trained weights are AGPL-3.0 by default, so they are license-compatible with the project.
Artifacts derived from them inherit AGPL-3.0. Keep provenance and license statements current in
`NOTICE` and `README.md` whenever a checkpoint or artifact is redistributed.

## Scope direction

The next product work is a native weight distribution channel, `max_detections`, batching, and
YOLOX training/loss parity. Do not jump to YOLO26 until the current model/engine abstractions and
licensing decision are settled.
