# boquilens

Object detection in pure Rust, powered by [Burn](https://burn.dev). The full inference path—model
graph, preprocessing, decoding, and post-processing—runs natively in Rust with no Python, PyTorch,
or ONNX Runtime.

![Detections drawn by boquilens on the bundled sample image](assets/dog_bike_man-detections.png)

## Models

Every model runs **object detection** on COCO-80 classes at 640 px input; the YOLO11-seg variants
additionally run **instance segmentation**. The only runtime mode is **Predict** (via the CLI and
the Rust API); training and validation are out of scope. Experimental models additionally support a
one-time `pack-weights` conversion into boquilens' native Burnpack format.

| Model    | Status       | Task             | Modes   | Variants               | Weights                          |
| -------- | ------------ | ---------------- | ------- | ---------------------- | -------------------------------- |
| YOLOX    | stable       | Detect           | Predict | nano, tiny, s, m, l, x | official `.pth`, auto-downloaded |
| YOLOv3   | experimental | Detect           | Predict | tiny-u                 | one-time `.bpk` pack             |
| YOLOv10  | experimental | Detect           | Predict | n, s, m, b, l, x       | one-time `.bpk` pack             |
| YOLO11   | experimental | Detect, Segment  | Predict | n, s, m, l, x (+n/s -seg) | one-time `.bpk` pack          |
| YOLO26   | experimental | Detect           | Predict | n, s, m, l, x          | one-time `.bpk` pack             |

Variant naming follows each family (YOLOX scales as nano/tiny/s/m/l/x, v3 ships a tiny model, and
the modern families use n/s/m/l/x; YOLOv10 replaces the xl scale with b). Pass the variant-suffixed
CLI name: `yolox-nano`, `yolox-tiny`, `yolox-s`, `yolox-m`, `yolox-l`, `yolox-x`, `yolov3-tinyu`,
`yolov10n`, ..., `yolo11n`, ..., `yolo11n-seg`, `yolo11s-seg`, ..., `yolo26x`. YOLOX ships
Apache-2.0 weights downloaded from the official release. Ultralytics-family weights are AGPL-3.0,
and the native artifacts derived from them inherit that license (see [NOTICE](NOTICE)). Every model
runs at 640 px input; note that YOLOX-Tiny's official evaluation resolution is 416 px, so its
published mAP (32.8) does not transfer one-to-one. YOLO11 is the only modern family whose
predictions pass through classic class-aware non-maximum suppression (the others are NMS-free
end-to-end). The YOLO11-seg models share that NMS path and add Ultralytics' Segment head: 32 mask
prototypes at stride 4 plus per-detection mask coefficients; instance masks are returned as
boolean coverage over the source image.

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
| yolo11n      |   5,399,968 | `36ACCB9BCEF72CD1DD3D534F54BE845C9EE4EE1697AD65C731FE028028E68BDF` |
| yolo11s      |  19,140,768 | `4277237339A0975D1E86FBFB7787D861982F9B64B857C458E0D998671AA63DB9` |
| yolo11m      |  40,561,568 | `ACFE957B42A17D81C9988772E2A1576592B3DB293DC8D52AFC91BCECB5595073` |
| yolo11l      |  51,208,352 | `84FE90D17143FB894CEFE6557D3619F000E1602BDC331905FE56E6AC996F953F` |
| yolo11x      | 114,597,280 | `1AC48B4A48165632F7B54A7B2E8471C9FB782CE436DE795C3155BCEF848C156E` |
| yolo11n-seg  |   5,919,808 | `A29FF611095F39E3875A22B03B93DC1FDCD5AE40A1310AA5DF4D3813E17B1FF4` |
| yolo11s-seg  |  20,465,216 | `FD9841F96748BD32A50EF508340F86A161B331D44F3D16678A96BED1A76342BE` |

### Preparing experimental weights

One-time conversion of an official Ultralytics checkpoint (Python is a conversion-time dependency
only); substitute the model name for `yolov3-tinyu`, `yolov10n/s/m/b/l/x`, `yolo11n/s/m/l/x`,
`yolo11n/s-seg`, or `yolo26n/s/m/l/x`:

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
cargo run --release -- predict --model yolo11n-seg --weights target/yolo11n-seg-coco-ultralytics-v8.4-boquilens-v1.bpk --source image.jpg --masks
cargo run -- --help
```

Detections are printed as a table or JSON and an annotated PNG is written next to the input
(`--output` overrides). Boxes are unnormalized, continuous `XYXY` pixel edges in the original
source image, clipped to its bounds and sorted by confidence; the JSON output carries matching
coordinate metadata. Segmentation models accept `--masks` to stroke the instance-mask outlines on
the annotated image and report per-detection covered-pixel counts (JSON gains a mask summary; the
full bitmask is available through the Rust API).

### Rust API

```rust,no_run
use boquilens::{PredictOptions, Predictor};

fn main() -> boquilens::Result<()> {
    // Every YOLOX scale has a dedicated helper (yolox_nano, yolox_tiny, yolox_s, yolox_m,
    // yolox_l, yolox_x); other models load through Predictor::new with a ModelId.
    let predictor = Predictor::yolox_x(PredictOptions::default())?;
    let (_image, detections) = predictor.predict_path("image.jpg")?;
    for detection in detections {
        println!("{}: {:.1}%", detection.class_name, detection.confidence * 100.0);
    }
    Ok(())
}
```

Instance segmentation is available for the YOLO11-seg variants through
`Predictor::predict_segmentation` / `predict_segmentation_path`, which return
`SegmentationDetection` values: the same box fields as `Detection` plus a boolean source-image
coverage mask (`InstanceMask`, `width * height` bytes):

```rust,no_run
use boquilens::{ModelId, PredictOptions, Predictor};

fn main() -> boquilens::Result<()> {
    let predictor = Predictor::new(ModelId::Yolo11NSeg, PredictOptions::default())?;
    let (_image, detections) = predictor.predict_segmentation_path("image.jpg")?;
    for detection in detections {
        let covered = detection.mask.data.iter().filter(|pixel| **pixel).count();
        println!("{}: {:.1}% mask_px={}", detection.class_name, detection.confidence * 100.0, covered);
    }
    Ok(())
}
```

## Performance

Single-image, batch-1 inference at 640 px input: model forward, head decode, and result sync, on
Burn's Flex CPU backend in release mode. Artifacts store f16 weights that are upcast and computed in
f32. Each number is the median of 10 timed runs after 3 warmup runs, measured sequentially (one test
at a time) via:

```console
cargo test --locked --release measures_single_inference_latency -- --ignored --nocapture --test-threads 1
```

The Ultralytics column is the official PyTorch runtime on the same machine and methodology
(batch 1, 640 px, fp32 CPU, fused conv+bn, 16 torch threads, 3 warmups + 10 timed runs), measured
with the development-only tool:

```console
& target\.venv\Scripts\python.exe tools\bench_ultralytics_cpu.py target\yolov10n.pt ...
```

The GPU column runs the same harness on Burn's Wgpu backend (built with `--features gpu`), on an
NVIDIA GeForce RTX 5080 selected through Vulkan (driver 610.47), f16 artifacts computed in f32:

```console
cargo test --locked --release --features gpu measures_single_inference_latency_gpu -- --ignored --nocapture --test-threads 1
```

Reference machine: AMD Ryzen 9 9950X3D (16C/32T), 32 GB RAM, Windows 11. Absolute numbers move with
hardware and library releases; treat them as a relative scale across variants, not a benchmark claim.
YOLOX scale rows are not measured yet: the latency harness lives with the per-scale model tests that
the Ultralytics families have and YOLOX does not.

| Model    | boquilens CPU (ms) | boquilens GPU (ms) | GPU vs CPU | Ultralytics PyTorch CPU (ms) |
| -------- | -----------------: | -----------------: | ---------: | ---------------------------: |
| yolov10n |              129.1 |               10.6 |      12.2x |                         19.0 |
| yolov10s |              241.0 |               19.8 |      12.2x |                         32.1 |
| yolov10m |              454.7 |               43.5 |      10.5x |                         62.8 |
| yolov10b |              594.0 |               64.1 |       9.3x |                         88.4 |
| yolov10l |              735.7 |               86.9 |       8.5x |                        122.2 |
| yolov10x |              939.6 |               94.8 |       9.9x |                        169.8 |
| yolo26n  |              116.2 |                9.3 |      12.5x |                         22.5 |
| yolo26s  |              237.1 |               21.0 |      11.3x |                         42.9 |
| yolo26m  |              478.0 |               53.5 |       8.9x |                         85.2 |
| yolo26l  |              619.3 |               66.1 |       9.4x |                        109.7 |
| yolo26x  |              975.2 |              124.8 |       7.8x |                        196.8 |
| yolo11n  |              130.1 |                8.5 |      15.3x |                         17.8 |
| yolo11s  |              243.8 |               17.3 |      14.1x |                         31.7 |
| yolo11m  |              486.7 |               45.6 |      10.7x |                         68.9 |
| yolo11l  |              634.6 |               56.5 |      11.2x |                         92.8 |
| yolo11x  |              991.6 |              115.6 |       8.6x |                        179.1 |
| yolo11n-seg |            173.9 |               13.5 |      12.9x |                         22.8 |
| yolo11s-seg |            307.4 |               29.9 |      10.3x |                         44.8 |

- **CPU.** The CPU columns are the always-available path: Burn's Flex backend. Preprocessing and
  top-k postprocessing are not included and add a small constant per image. Flex currently trails
  fused PyTorch CPU inference by roughly 5-7.5x; closing that gap (kernel fusion, threading) is CPU
  engineering work on top of Burn, independent of the model graphs.
- **GPU.** The `gpu` feature (burn-wgpu over Vulkan/DX12 on Windows and Linux, Metal on macOS)
  enables `--device gpu` in the CLI and the GPU latency tests above. Detections are numerically
  identical to the CPU path on the reference image (f32 compute on both). The first GPU forward
  compiles kernels and autotunes, so cold start is much slower than steady state; the harness
  excludes it via warmup runs.

## Development

```console
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --no-default-features --lib
```

Golden parity fixtures are generated per model against the official checkpoints and consumed by the
ignored tests (the fixture exporters cover every detect and seg scale via `--model`). YOLOX uses its
own exporter, which runs the official YOLOX PyTorch sources instead of the Ultralytics package:

```console
python tools/export_yolo26_fixtures.py target/yolo26m.pt assets/dog_bike_man.jpg target --model yolo26m
cargo test --locked yolo26m -- --ignored

python tools/export_yolo11_fixtures.py target/yolo11n.pt assets/dog_bike_man.jpg target --model yolo11n
cargo test --locked yolo11n -- --ignored

python tools/export_yolo11_fixtures.py target/yolo11n-seg.pt assets/dog_bike_man.jpg target --model yolo11n-seg
cargo test --locked yolo11n-seg -- --ignored

& target\.venv\Scripts\python.exe tools\export_yolox_fixtures.py target\checkpoints\yolox_tiny.pth assets\dog_bike_man.jpg target --model yolox-tiny
cargo test --locked yolox_tiny -- --ignored
```

The segmentation variants additionally compare the full runtime end to end against the official
Ultralytics predict (boxes plus per-detection mask IoU in source-image space); their expectation is
generated with:

```console
python tools/export_yolo11_seg_e2e.py target/yolo11n-seg.pt assets/dog_bike_man.jpg target --model yolo11n-seg
cargo test --locked yolo11n_seg_matches_ultralytics_end_to_end -- --ignored --nocapture
```

See [AGENTS.md](AGENTS.md) for the full model-porting workflow and invariants.

## License

boquilens is [AGPL-3.0](LICENSE). The YOLOX path derives from Apache-2.0 code
([LICENSE-APACHE](LICENSE-APACHE)) and uses official Apache-2.0 weights; Ultralytics architectures
and checkpoints are AGPL-3.0. Full provenance in [NOTICE](NOTICE).
