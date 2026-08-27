# boquilens

Object detection in pure Rust, powered by [Burn](https://burn.dev). The full inference path—model
graph, preprocessing, decoding, and post-processing—runs natively in Rust with no Python, PyTorch,
or ONNX Runtime.

![Detections drawn by boquilens on the bundled sample image](assets/dog_bike_man-detections.png)

## Models

Every model runs **object detection** on COCO-80 classes at 640 px input. The only runtime mode is
**Predict** (via the CLI and the Rust API); training and validation are out of scope. Experimental
models additionally support a one-time `pack-weights` conversion into boquilens' native Burnpack
format.

| Model    | Status       | Task   | Modes   | Variants | Weights                          |
| -------- | ------------ | ------ | ------- | -------- | -------------------------------- |
| YOLOX    | stable       | Detect | Predict | nano     | official `.pth`, auto-downloaded |
| YOLOv3   | experimental | Detect | Predict | tiny-u   | one-time `.bpk` pack             |
| YOLOv10  | experimental | Detect | Predict | n        | one-time `.bpk` pack             |
| YOLO26   | experimental | Detect | Predict | n        | one-time `.bpk` pack             |

Variant naming follows each family (YOLOX scales as nano/tiny/s/m/l/x, v3 ships a tiny model, and
the modern families use n/s/m/l/x). One variant per model today; the rest are future work. Pass the
variant-suffixed CLI name: `yolox-nano`, `yolov3-tinyu`, `yolov10n`, `yolo26n`. YOLOX ships
Apache-2.0 weights downloaded from the official release. Ultralytics-family weights are AGPL-3.0,
and the native artifacts derived from them inherit that license (see [NOTICE](NOTICE)).

Verified v1 artifacts:

| Model         | Bytes      | SHA-256                                                            |
| ------------- | ---------: | ------------------------------------------------------------------ |
| yolov3-tinyu  | 24,411,296 | `52AD28C04D234F500387E9C874A52447F6A107490968BF9A23C653DDCB14DBBA` |
| yolov10n      |  4,779,424 | `8A672F4924F52E89F7DF95C689C66CF157A96674CE1ADF3C2CF6A025D5C9C44B` |
| yolo26n       |  5,016,992 | `5FB09D89850E2ECB75C0580893239DEF9BB130E95A228FB319675F267B5B24C6` |

### Preparing experimental weights

One-time conversion of an official Ultralytics checkpoint (Python is a conversion-time dependency
only); substitute the model name for `yolov3-tinyu` or `yolov10n`:

```console
python -m pip install torch ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
python tools/export_ultralytics_state.py yolo26n.pt target/yolo26n-state.pt
cargo run --release -- pack-weights --model yolo26n --input target/yolo26n-state.pt --output target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk
```

After that, inference only needs the `.bpk` artifact and Rust.

## Usage

```console
cargo run --release -- predict --model yolox-nano --source assets/dog_bike_man.jpg
cargo run --release -- predict --model yolo26n --weights target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk --source image.jpg --json --confidence 0.30
cargo run -- --help
```

Detections are printed as a table or JSON and an annotated PNG is written next to the input
(`--output` overrides). Boxes are unnormalized, continuous `XYXY` pixel edges in the original
source image, clipped to its bounds and sorted by confidence; the JSON output carries matching
coordinate metadata.

### Rust API

```rust,no_run
use boquilens::{PredictOptions, Predictor};

fn main() -> boquilens::Result<()> {
    let predictor = Predictor::yolox_nano(PredictOptions::default())?;
    let (_image, detections) = predictor.predict_path("image.jpg")?;
    for detection in detections {
        println!("{}: {:.1}%", detection.class_name, detection.confidence * 100.0);
    }
    Ok(())
}
```

## Performance

- **CPU today.** Single-image, batch-1, 640 px inference through Burn's Flex backend on the CPU.
  This is the supported and tested path.
- **GPU not yet.** Burn ships GPU backends, but boquilens has not wired, benchmarked, or packaged
  them; the CLI currently runs wherever the Flex backend runs.
- **Numbers at release.** Latency and throughput benchmarks (CPU and GPU, per model) will be
  published with the first release; treat nothing in this README as a benchmark.

## Development

```console
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --no-default-features --lib
```

Golden parity fixtures are generated per model against the official Ultralytics checkpoints and
consumed by the ignored tests:

```console
python tools/export_yolo26_fixtures.py target/yolo26n.pt assets/dog_bike_man.jpg target
cargo test --locked -- --ignored
```

See [AGENTS.md](AGENTS.md) for the full model-porting workflow and invariants.

## License

boquilens is [AGPL-3.0](LICENSE). The YOLOX path derives from Apache-2.0 code
([LICENSE-APACHE](LICENSE-APACHE)) and uses official Apache-2.0 weights; Ultralytics architectures
and checkpoints are AGPL-3.0. Full provenance in [NOTICE](NOTICE).
