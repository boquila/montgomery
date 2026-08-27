# boquilens agent guide

This directory is the active Rust/Burn object-detection crate. Work from this directory unless a
task explicitly targets the sibling vendored projects.

To bring up a new model family or scale variant, follow [MODEL_BRINGUP.md](MODEL_BRINGUP.md) end to
end.

## What is here

- `src/models/yolox/`: stable YOLOX-Nano implementation and Apache-2.0 checkpoint path.
- `src/models/yolov3_tiny/`: experimental YOLOv3-Tiny-Ultralytics implementation, including the
  body, split DFL head, native Burnpack loading, and version metadata.
- `src/models/yolov10/`: experimental YOLOv10 (n/s/m/b/l/x) implementation, including the C2f/
  SCDown/SPPF/PSA/C2fCIB bodies (the C2fCIB flavor varies per scale), NMS-free one2one head, native
  Burnpack loading, and version metadata.
- `src/models/yolo26/`: experimental YOLO26 (n/s/m/l/x) implementation, including the
  C3k2/residual-SPPF/C2PSA bodies (m/l/x force the C3k chain onto the early backbone stages) with an
  attention P5 stage, DFL-free NMS-free end-to-end head, native Burnpack loading, and version
  metadata.
- `src/data/letterbox.rs`: model-specific preprocessing and reversible source-image geometry.
- `src/lib.rs`: `ModelId`, `Predictor`, detection results, NMS integration, and weight packing API.
- `src/main.rs`: the `predict` and `pack-weights` CLI commands.
- `tools/`: development-only Ultralytics checkpoint conversion and golden-fixture generators.

## Fast path

Stable YOLOX inference:

```console
cargo run --release -- predict --model yolox-nano --source assets/dog_bike_man.jpg
```

YOLOv3-Tiny-U and the YOLOv10/YOLO26 scales require a tensor-only state and then a native artifact.
The complete workflow is documented in `README.md`; the short form (substitute any
`yolov10n/s/m/b/l/x` or `yolo26n/s/m/l/x` name):

```console
python tools/export_ultralytics_state.py yolov10n.pt target/yolov10n-state.pt
cargo run --release -- pack-weights --model yolov10n --input target/yolov10n-state.pt --output target/yolov10n-coco-ultralytics-v8.4-boquilens-v1.bpk
cargo run --release -- predict --model yolov10n --weights target/yolov10n-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg

python tools/export_ultralytics_state.py yolo26n.pt target/yolo26n-state.pt
cargo run --release -- pack-weights --model yolo26n --input target/yolo26n-state.pt --output target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk
cargo run --release -- predict --model yolo26n --weights target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg
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

Regenerate those fixtures with the per-family `tools/export_{ultralytics,yolov10,yolo26}_fixtures.py`
scripts, passing `--model <id>` to select the detect scale. Generated checkpoints, fixtures, images,
and build output belong under `target/` and must not be committed.

## Invariants

- Public detections are continuous, unnormalized source-image `XYXY` box edges in pixels. They are
  not `XYWH` and not normalized; `xmax == width` and `ymax == height` are valid.
- YOLOX uses its existing top-left/raw-pixel transform. The Ultralytics-family models use
  Ultralytics-style stride-aligned rectangular letterboxing and RGB values normalized to `[0, 1]`.
- YOLOv10 (all scales) is NMS-free: its one2one head output is top-300 selected and
  confidence-filtered like Ultralytics' end-to-end postprocess, not passed through non-maximum
  suppression. YOLO26 (all scales) shares that postprocess and is additionally DFL-free: the box
  tower emits the four XYXY side distances directly, decoded anchor-relative and scaled by the
  feature strides.
- The YOLOv10 and YOLO26 scale variants are not pure width/depth rescalings. The official
  per-scale YAMLs swap module flavors: YOLOv10s uses large-kernel C2fCIB towers, YOLOv10
  m/b/l/x use the plain depth-wise C2fCIB flavor (and x converts backbone layer 6), and YOLO26
  m/l/x force `c3k=True` on the early backbone stages at 0.25 expansion. Each variant's body
  declares this explicitly; keep them aligned with the vendored YAMLs and `parse_model`.
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
YOLOX training/loss parity.
