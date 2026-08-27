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

| Model    | Status       | Task   | Modes   | Variants       | Weights                          |
| -------- | ------------ | ------ | ------- | -------------- | -------------------------------- |
| YOLOX    | stable       | Detect | Predict | nano           | official `.pth`, auto-downloaded |
| YOLOv3   | experimental | Detect | Predict | tiny-u         | one-time `.bpk` pack             |
| YOLOv10  | experimental | Detect | Predict | n, s, m, b, l, x | one-time `.bpk` pack           |
| YOLO26   | experimental | Detect | Predict | n, s, m, l, x  | one-time `.bpk` pack             |

Variant naming follows each family (YOLOX scales as nano/tiny/s/m/l/x, v3 ships a tiny model, and
the modern families use n/s/m/l/x; YOLOv10 replaces the xl scale with b). Pass the variant-suffixed
CLI name: `yolox-nano`, `yolov3-tinyu`, `yolov10n`, `yolov10s`, ..., `yolo26x`. YOLOX ships
Apache-2.0 weights downloaded from the official release. Ultralytics-family weights are AGPL-3.0,
and the native artifacts derived from them inherit that license (see [NOTICE](NOTICE)).

Verified v1 artifacts:

| Model        | Bytes       | SHA-256                                                            |
| ------------ | ----------: | ------------------------------------------------------------------ |
| yolov3-tinyu |  24,411,296 | `52AD28C04D234F500387E9C874A52447F6A107490968BF9A23C653DDCB14DBBA` |
| yolov10n     |   4,779,424 | `8A672F4924F52E89F7DF95C689C66CF157A96674CE1ADF3C2CF6A025D5C9C44B` |
| yolov10s     |  14,822,560 | `6E6427357A25CFA6FE96D5BA0130808B16A699B0E282EF37A80366497BAC351F` |
| yolov10m     |  31,221,920 | `65C579B005413714F8E935316EA84FE12B689049621A9B15ED6EFF64B536F84C` |
| yolov10b     |  38,692,768 | `15EB91359D74E3A48D98356CF0410D9566B20AE70705C940A17C29078C73B906` |
| yolov10l     |  49,446,304 | `D0412F77DDE5E9ED53324687551FDF29AEF6F34387B9EBC838322891CA90C260` |
| yolov10x     |  59,978,400 | `B47954C6647A8298C2A0444CA53EEB2E498A4D4B52FDB14A934A4FD2AB6A39C2` |
| yolo26n      |   5,016,992 | `5FB09D89850E2ECB75C0580893239DEF9BB130E95A228FB319675F267B5B24C6` |
| yolo26s      |  19,283,872 | `DD287F71998783596CBF5204F29D589D215EBF910987623CBCB1DD8F0AD91855` |
| yolo26m      |  41,216,928 | `50A0BE494BA93D5663084161999B3D2B2C9ABB6DABB163D0AED2DB6F37591249` |
| yolo26l      |  50,140,064 | `19D2C802F3266571FC7298DB9C3AB0E912D4DD6004B1D37510124F92A428A171` |
| yolo26x      | 112,210,080 | `D1B1B94FC28423CC4FFD4EA04DEEAE3FE4A352B7E0D8F442D6CE9FA616C813A9` |

### Preparing experimental weights

One-time conversion of an official Ultralytics checkpoint (Python is a conversion-time dependency
only); substitute the model name for `yolov3-tinyu`, `yolov10n/s/m/b/l/x`, or `yolo26n/s/m/l/x`:

```console
python -m pip install torch ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo26m.pt')"
python tools/export_ultralytics_state.py yolo26m.pt target/yolo26m-state.pt
cargo run --release -- pack-weights --model yolo26m --input target/yolo26m-state.pt --output target/yolo26m-coco-ultralytics-v8.4-boquilens-v1.bpk
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
consumed by the ignored tests (the fixture exporters cover every detect scale via `--model`):

```console
python tools/export_yolo26_fixtures.py target/yolo26m.pt assets/dog_bike_man.jpg target --model yolo26m
cargo test --locked yolo26m -- --ignored
```

See [AGENTS.md](AGENTS.md) for the full model-porting workflow and invariants.

## License

boquilens is [AGPL-3.0](LICENSE). The YOLOX path derives from Apache-2.0 code
([LICENSE-APACHE](LICENSE-APACHE)) and uses official Apache-2.0 weights; Ultralytics architectures
and checkpoints are AGPL-3.0. Full provenance in [NOTICE](NOTICE).
