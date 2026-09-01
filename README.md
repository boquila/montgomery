<div align="center">

<picture>
  <img alt="Montgomery" src="/docs/logo.svg" width="58%">
</picture>

Native object detection, instance segmentation, and image classification in Rust with [Burn](https://burn.dev).

<h3>

[Repository](https://github.com/boquila/montgomery) | [Performance](https://github.com/boquila/montgomery/blob/main/docs/performance-comparison.MD) | [Fat LTO study](https://github.com/boquila/montgomery/blob/main/docs/lto-comparison.MD) | [Model support](#supported-models)

</h3>

[![CI](https://github.com/boquila/montgomery/actions/workflows/ci.yml/badge.svg)](https://github.com/boquila/montgomery/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-33da72)](LICENSE)

</div>

---

Montgomery is a native Rust computer-vision stack:

- YOLO inference on CPU or GPU
- WGPU training with validation, resumable checkpoints, and ready-to-use exports
- Detection, instance segmentation, and classification
- Burnpack and ONNX export

Normal inference needs no Python, PyTorch, or ONNX Runtime.

![Instance segmentation produced by YOLO11n-seg](assets/dog_bike_man-segmentation.png)

## Quick start

Install Rust and [uv](https://docs.astral.sh/uv/), then convert an upstream checkpoint once:

```console
uv run --locked python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
uv run --locked tools/export_ultralytics_state.py yolo26n.pt target/yolo26n-state.pt
cargo run --locked --release -- pack-weights --model yolo26n --input target/yolo26n-state.pt --output target/yolo26n-coco-ultralytics-v8.4-montgomery-v1.bpk
```

Run inference:

```console
cargo run --locked --release -- predict --model yolo26n --weights target/yolo26n-coco-ultralytics-v8.4-montgomery-v1.bpk --source image.jpg
```

Useful options: `--json`, `--confidence 0.30`, `--output result.png`, `--masks`, and `--device gpu`.
GPU inference requires `--features gpu`.

YOLOX accepts its official `.pth` directly through `pack-weights`. Other families use the
tensor-only conversion shown above. Prediction always uses a Montgomery `.bpk` Burnpack.

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

YOLOX is stable. The other families and native training are experimental.

## Rust API

```rust,no_run
use burn_flex::Flex;
use montgomery::{ModelId, PredictOptions, Predictor};

fn main() -> montgomery::Result<()> {
    let predictor = Predictor::<Flex>::from_checkpoint(
        ModelId::Yolo26N,
        "target/yolo26n-coco-ultralytics-v8.4-montgomery-v1.bpk",
        PredictOptions::default(),
    )?;

    let (_, detections) = predictor.predict_path("image.jpg")?;
    for detection in detections {
        println!("{}: {:.1}%", detection.class_name, detection.confidence * 100.0);
    }
    Ok(())
}
```

Use `predict_segmentation_path` for `-seg` models and `predict_classification_path` for `-cls`
models.

## Train

Training is WGPU-only and must use a release build:

```console
cargo run --locked --release --features training -- train --model yolo26n --data dataset.yaml --weights target/yolo26n-state.pt --epochs 100
```

Every run contains:

- `results.csv`, `results.svg`, and `validation.jsonl`
- `exports/best.bpk` and `exports/last.bpk`
- `checkpoints/best` and `checkpoints/last`

Only the best and latest resumable models are retained. Validation selects `best` automatically.
Use `--save-period` to control recovery checkpoints and `--workers` to override automatic CPU
worker selection.

Montgomery won 22 of 30 matched short training benchmarks against Ultralytics on an RTX 5080.
Classification and small batched workloads are strongest; batch-1, 640 px, medium segmentation,
and longer detection/segmentation runs still need work. Full methodology and limitations are in the
[performance report](docs/performance-comparison.MD). Full-dataset quality parity is not yet claimed.

## Export ONNX

```console
cargo run --locked --release -- export-onnx --model yolo26n --weights target/yolo26n-coco-ultralytics-v8.4-montgomery-v1.bpk --output target/yolo26n.onnx
```

The offline exporter validates the graph with ONNX Runtime. Setup details are in
[tools/onnx/README.md](tools/onnx/README.md).

## Develop

Stable Rust is the only requirement for the native crate. A fresh checkout is ready in two commands:

```console
git clone https://github.com/boquila/montgomery.git && cd montgomery
cargo test --locked
```

Cargo downloads and builds every Rust dependency automatically. Python is optional and only used
for checkpoint conversion, benchmark generation, and ONNX development; `uv run --locked <command>`
creates that environment from the committed `pyproject.toml` and `uv.lock` when needed.

The same checks used by CI are:

```console
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --no-default-features --lib
```

See [MODEL_BRINGUP.md](MODEL_BRINGUP.md) for new model families and [AGENTS.md](AGENTS.md) for
implementation invariants.

## License

Montgomery is [AGPL-3.0](LICENSE). YOLOX code and official weights are Apache-2.0; see
[LICENSE-APACHE](LICENSE-APACHE). Ultralytics architectures and checkpoints are AGPL-3.0. Full
provenance is recorded in [NOTICE](NOTICE).
