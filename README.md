# boquilens

Object detection, instance segmentation, and image classification in Rust with
[Burn](https://burn.dev).

Inference is native Rust: model execution, preprocessing, decoding, and postprocessing do not
require Python, PyTorch, or ONNX Runtime.

![Instance segmentation produced by YOLO11n-seg](assets/dog_bike_man-segmentation.png)

## Quick start

Prerequisites: Rust and [uv](https://docs.astral.sh/uv/).

Every model runs from a boquilens Burnpack (`.bpk`). Upstream `.pth` and `.pt` checkpoints are
one-time conversion inputs; they are not accepted by `predict`.

### 1. Prepare weights

For YOLOX, download a checkpoint from the
[official release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) and pack it
directly:

```console
cargo run --locked --release -- pack-weights --model yolox-nano --input target/yolox_nano.pth --output target/yolox-nano-coco-official-v0.1.1rc0-boquilens-v1.bpk
```

For Ultralytics models, first export a tensor-only state, then pack it:

```console
uv run --locked python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
uv run --locked tools/export_ultralytics_state.py yolo26n.pt target/yolo26n-state.pt
cargo run --locked --release -- pack-weights --model yolo26n --input target/yolo26n-state.pt --output target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk
```

`uv` creates the locked Python environment automatically. Python is needed only for conversion.

### 2. Run inference

```console
cargo run --locked --release -- predict --model yolo26n --weights target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg
```

Detection and segmentation commands print results and write an annotated PNG. Classification
commands print the top five classes. Useful options:

```console
--json                 Print machine-readable output
--confidence 0.30      Set the confidence threshold
--output result.png    Choose the annotated-image path
--device gpu           Use WGPU; build with --features gpu
--masks                Render masks for a -seg model
```

## Supported models

| Model | Variants | Tasks |
| --- | --- | --- |
| YOLOX | `nano, tiny, s, m, l, x` | Detect |
| YOLOv3 | `tinyu` | Detect |
| YOLOv8 | `n, s, m, l, x` | Detect, segment, classify |
| YOLOv10 | `n, s, m, b, l, x` | Detect |
| YOLO11 | `n, s, m, l, x` | Detect, segment, classify |
| YOLO12 | `n, s, m, l, x` | Detect |
| YOLO26 | `n, s, m, l, x` | Detect, segment, classify |

YOLOX is stable. All other model families are experimental.

Detect and segment models use COCO-80 classes. Classification models use ImageNet-1k and return
the top five classes. YOLOX Nano/Tiny use 416 px inputs, classifiers use 224 px, and the remaining
models use 640 px.

Task examples:

```console
# Detection
cargo run --locked --release -- predict --model yolox-nano --weights target/yolox-nano-coco-official-v0.1.1rc0-boquilens-v1.bpk --source image.jpg

# Instance segmentation
cargo run --locked --release -- predict --model yolo11n-seg --weights target/yolo11n-seg-coco-ultralytics-v8.4-boquilens-v1.bpk --source image.jpg --masks

# Classification
cargo run --locked --release -- predict --model yolo26s-cls --weights target/yolo26s-cls-imagenet1k-ultralytics-v8.4-boquilens-v1.bpk --source image.jpg --json
```

## Rust API

```rust,no_run
use boquilens::{ModelId, PredictOptions, Predictor};
use burn_flex::Flex;

fn main() -> boquilens::Result<()> {
    let predictor = Predictor::<Flex>::from_checkpoint(
        ModelId::Yolo26N,
        "target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk",
        PredictOptions::default(),
    )?;

    let (_image, detections) = predictor.predict_path("image.jpg")?;
    for detection in detections {
        println!("{}: {:.1}%", detection.class_name, detection.confidence * 100.0);
    }
    Ok(())
}
```

Use `predict_segmentation_path` for `-seg` models and `predict_classification_path` for `-cls`
models.

Detection boxes are source-image `XYXY` pixel edges. Segmentation results add a boolean mask in
source-image coordinates.

## GPU

Build with the `gpu` feature and select the device at runtime:

```console
cargo run --locked --release --features gpu -- predict --model yolo26n --device gpu --weights target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk --source image.jpg
```

The default CPU backend is Burn Flex. Benchmark methodology and recorded CPU/GPU results live in
[PERF_NOTES.md](PERF_NOTES.md).

## ONNX export

ONNX export is an offline development workflow. Create its pinned Python environment once:

```powershell
uv venv --python 3.13 target/.venv
uv pip sync --python target/.venv/Scripts/python.exe tools/onnx/requirements.lock.txt
```

Then export a Burnpack:

```powershell
cargo run --locked --release -- export-onnx `
  --model yolo26n `
  --weights target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk `
  --output target/yolo26n.onnx
```

The exporter validates the graph with ONNX Runtime and writes an `.onnx.json` contract beside the
model. Source checkout requirements and validation details are in
[tools/onnx/README.md](tools/onnx/README.md).

## Training

Native training is experimental, WGPU-only, and disabled by default. Always use a release build:

```console
cargo run --locked --release --features training -- train --model yolo26n --data dataset.yaml --weights target/yolo26n-state.pt --epochs 100
```

Each run contains:

- `results.csv`, `results.svg`, and `validation.jsonl` with losses, precision, recall, mAP, accuracy,
  fitness, and learning rate as applicable.
- `exports/best.bpk` and `exports/last.bpk`, ready for `predict`.
- `checkpoints/best` and `checkpoints/last`, the only two full resumable states retained.

Validation selects `best` automatically. `--save-period` controls crash-recovery frequency without
keeping epoch archives; preprocessing workers are chosen from available CPU parallelism unless
`--workers` is set. Use `--no-val --no-export` only for throughput measurements.

Training, validation, resume, and export are smoke-tested. Full COCO convergence can be reproduced
with `uv run --locked tools/bench_full_convergence.py --download`; quality parity is not claimed
until those long runs complete.

The reproducible RTX 5080 comparison covers 30 matched Ultralytics workloads across tasks, families,
scales, batches, resolutions, and ten-epoch smoke runs. Native wins 22 of 30 workloads and 11 of 12
smallest-variant family/task cells; YOLO26n segmentation is effectively tied but 0.7% behind.
Medium segmentation, batch-1 detection/segmentation, 640 px, and longer detect/segment runs remain
slower. See the
[full performance report](docs/performance-comparison.MD) for every result, chart, limitation, and the
one-command benchmark harness.

The supported augmentation contract is documented in
[AUGMENTATION_COMPATIBILITY.md](AUGMENTATION_COMPATIBILITY.md).

## Development

```console
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --no-default-features --lib
```

Model bring-up, ignored parity tests, fixture generation, and implementation invariants are in
[MODEL_BRINGUP.md](MODEL_BRINGUP.md) and [AGENTS.md](AGENTS.md).

## License

boquilens is [AGPL-3.0](LICENSE). YOLOX code and official weights are Apache-2.0; see
[LICENSE-APACHE](LICENSE-APACHE). Ultralytics architectures and checkpoints are AGPL-3.0. Full
provenance is recorded in [NOTICE](NOTICE).
