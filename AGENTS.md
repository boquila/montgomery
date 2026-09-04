# Montgomery agent guide

This directory is the active Rust/Burn crate. Work from the repository root unless a task
explicitly targets a sibling vendored project. Development checkpoints, fixtures, datasets,
images, reports, and build output belong under `target/` and must not be committed. User-facing
Burnpacks use the short `<model>.bpk` name in the repository root and are gitignored.

For a new family, scale, or task, follow [docs/MODEL_BRINGUP.md](docs/MODEL_BRINGUP.md) end to end.

## Repository map

- `src/models/yolox/`: stable YOLOX nano/tiny/s/m/l/x implementation and tensor-state import.
- `src/models/yolov3_tiny/`, `yolov8/`, `yolov10/`, `yolo11/`, `yolo12/`, `yolo26/`:
  experimental Ultralytics-family graphs and native Burnpack loaders.
- YOLOv8, YOLO11, and YOLO26 also provide `-seg` variants; YOLOv8, YOLO11, and YOLO26 provide
  `-cls` variants.
- `src/data/letterbox.rs`: inference preprocessing and reversible source-image geometry.
- `src/data/augmentation/`: feature-gated, traceable detect/segment/classify augmentation pinned
  to Ultralytics `v8.4.117-2-g461196cf0`. Parity lives in `tests/augmentation_parity.rs`.
- `src/training/`: WGPU-only native training, validation, checkpointing, and reporting.
- `src/lib.rs`: `ModelId`, `Predictor`, result types, postprocessing, masks, and weight packing.
- `src/main.rs`: CLI dispatch.
- `tools/`: Python conversion, fixture, ONNX, and benchmark utilities. Its Python environment is
  defined by `tools/pyproject.toml`, `tools/uv.lock`, and `tools/.python-version`.

Keep graph code independent of CLI, filesystems, rendering, and image decoding. Keep `ModelId`, CLI
names, loaders, and documentation synchronized. Preserve the stable/experimental distinction in
user-facing text.

## Common workflows

Run Python tools from the repository root with the tools project selected:

```console
uv run --project tools tools/export_checkpoint_state.py yolo26n.pt target/yolo26n-state.pt
```

Stable YOLOX and Ultralytics-family models both run from native Burnpacks:

```console
uv run --project tools tools/export_checkpoint_state.py target/yolox_nano.pth target/yolox-nano-state.pt
montgomery pack-weights --architecture yolox-nano --state target/yolox-nano-state.pt
montgomery predict --model yolox-nano.bpk --source docs/dog_bike_man.jpg

montgomery pack-weights --architecture yolo26n --state target/yolo26n-state.pt
montgomery predict --model yolo26n.bpk --source docs/dog_bike_man.jpg
```

Python/PyTorch is conversion- and development-time only; normal inference is Rust/Burn.

## Verification

Run before handing off changes:

```console
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
cargo check --no-default-features --lib
```

When external checkpoints and fixtures are available:

```console
cargo test -- --ignored
```

Training is opt-in. For training changes, run:

```console
cargo test --features training training
cargo clippy --features training --all-targets -- -D warnings
```

Real training, hardware smoke tests, and latency measurements must use `--release`. Single-image
latency tests must use `--test-threads 1` to avoid CPU contention. When touching a runtime/backend
boundary, compare CPU and GPU JSON detections on the reference image.

## Inference contracts

- Public detections are continuous, unnormalized source-image `XYXY` edges in pixels. They are not
  `XYWH`; `xmax == width` and `ymax == height` are valid.
- YOLOX uses top-left letterboxing and raw RGB pixels. Ultralytics detection/segmentation models use
  stride-aligned rectangular letterboxing and RGB values in `[0, 1]`.
- YOLOX batch norm uses eps `1e-3`, momentum `0.03`. Classification checkpoints use plain PyTorch
  defaults (eps `1e-5`, momentum `0.1`) through `BnFlavor::Pytorch`.
- YOLOv10 and YOLO26 are NMS-free end-to-end heads with top-300 selection. YOLO26 is also DFL-free.
- YOLOv8, YOLO11, and YOLO12 use classic DFL (`reg_max = 16`) plus class-aware NMS. YOLOv8 uses
  legacy full-3x3 `cv3` classification towers; YOLO11/12 use the light DWConv flavor.
- End-to-end heads intentionally retain near-duplicates. Do not apply classic NMS to them.
- Runtime artifacts are native `.bpk` files. Upstream `.pth`/`.pt` files are conversion and parity
  inputs only.

Checkpoint behavior wins over current upstream source. Important known examples: YOLOv8/YOLO11
SPPF input projections retain SiLU; YOLO26 SPPF has no projection activation and adds its residual;
YOLO12 attention checkpoints include the positional-convolution bias. Scale-specific YAML module
choices are explicit for YOLOv10, YOLO12, and YOLO26 and must not be reduced to blind width/depth
rescaling. Golden tensor tests are the authority for these quirks.

## Segmentation and classification

- Segmentation mask coefficients are 32 raw values per anchor. Carry them through the family's
  existing postprocess: NMS for YOLOv8/11, end-to-end top-k for YOLO26.
- Assemble masks as `coefficients @ prototypes`, bilinear-upsample to the letterboxed canvas with
  `align_corners = false`, threshold logits at `> 0`, crop to the box, and drop empty masks.
- `InstanceMask` is boolean source-image coverage. Map pixels through the same retained letterbox
  geometry used for boxes. `predict()` returns the box branch; `predict_segmentation()` returns
  masks.
- Classification uses the Ultralytics 224 px anti-aliased shortest-edge resize, centered crop, RGB
  `[0, 1]`, and a 1000-way softmax. End-to-end parity compares the top-5 set and probabilities,
  since near-tied class order is resize-rounding sensitive.

## Augmentation and training

- Detection/segmentation augmentation stays HWC BGR `u8` until Format; default Format emits CHW
  RGB `u8`. Classification converts to RGB before policy transforms and emits normalized CHW
  `f32` after RandomErasing.
- Native seeded output is a Montgomery contract. Cross-language parity uses injected parameters or
  traces, not equal seed values.
- Training is opt-in through `--features training` and WGPU-only; inference graphs and prediction
  output must remain unchanged.
- Losses consume raw logits. YOLOX uses objectness; modern heads use TAL. YOLOv10/YOLO26 training
  retains one-to-many plus detached-feature one-to-one branches; YOLO26 remains DFL-free.
- Assignment may synchronize detached values to the host, but loss totals must remain connected to
  the model graph and empty batches must stay finite.
- Resume checkpoints are full precision and include model, optimizer, EMA, schedules, progress,
  model specification, ordered class names, and payload hashes. Lossy inference Burnpacks are never
  resume inputs.

## Licensing

Montgomery is AGPL-3.0. Keep the repository `LICENSE` and user-facing licensing statements current
when redistributing checkpoints or derived artifacts.
