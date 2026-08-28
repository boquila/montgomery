# boquilens agent guide

This directory is the active Rust/Burn object-detection crate. Work from this directory unless a
task explicitly targets the sibling vendored projects.

To bring up a new model family or scale variant, follow [MODEL_BRINGUP.md](MODEL_BRINGUP.md) end to
end.

## What is here

- `src/models/yolox/`: stable YOLOX (nano/tiny/s/m/l/x) implementation and Apache-2.0 checkpoint
  path. All scales share one graph parameterized by depth/width/depthwise and load their official
  `.pth` checkpoints directly.
- `src/models/yolov3_tiny/`: experimental YOLOv3-Tiny-Ultralytics implementation, including the
  body, split DFL head, native Burnpack loading, and version metadata.
- `src/models/yolov10/`: experimental YOLOv10 (n/s/m/b/l/x) implementation, including the C2f/
  SCDown/SPPF/PSA/C2fCIB bodies (the C2fCIB flavor varies per scale), NMS-free one2one head, native
  Burnpack loading, and version metadata.
- `src/models/yolo26/`: experimental YOLO26 (n/s/m/l/x) implementation, including the
  C3k2/residual-SPPF/C2PSA bodies (m/l/x force the C3k chain onto the early backbone stages) with an
  attention P5 stage, DFL-free NMS-free end-to-end head, native Burnpack loading, and version
  metadata.
- `src/models/yolo11/`: experimental YOLO11 (n/s/m/l/x) implementation, including the
  C3k2/plain-SPPF/C2PSA bodies (m/l/x force the C3k chain onto the early backbone stages; the P5
  stage is a plain C3k2 chain, not attention) with a classic DFL head (reg_max 16), NMS-based
  postprocessing, native Burnpack loading, and version metadata. The n/s scales also ship
  `-seg` instance-segmentation variants: the same bodies plus Ultralytics' Segment head (Proto
  module at stride 4, 32 mask coefficients per anchor) decoded through the same NMS with the
  coefficients carried along (`segment_head.rs`, masks assembled in `src/lib.rs`).
- `src/data/letterbox.rs`: model-specific preprocessing and reversible source-image geometry.
- `src/lib.rs`: `ModelId`, `Predictor`, detection results, NMS integration, and weight packing API.
- `src/main.rs`: the `predict` and `pack-weights` CLI commands.
- `tools/`: development-only Ultralytics checkpoint conversion, golden-fixture generators, and the
  PyTorch CPU benchmark used for the README performance comparison.

## Fast path

Stable YOLOX inference (substitute any `yolox-nano|tiny|s|m|l|x` name; the official checkpoint
downloads to the model cache on first use):

```console
cargo run --release -- predict --model yolox-nano --source assets/dog_bike_man.jpg
```

YOLOv3-Tiny-U, YOLO11, and the YOLOv10/YOLO26 scales require a tensor-only state and then a native
artifact. The complete workflow is documented in `README.md`; the short form (substitute any
`yolov10n/s/m/b/l/x`, `yolo11n/s/m/l/x`, or `yolo26n/s/m/l/x` name):

```console
python tools/export_ultralytics_state.py yolov10n.pt target/yolov10n-state.pt
cargo run --release -- pack-weights --model yolov10n --input target/yolov10n-state.pt --output target/yolov10n-coco-ultralytics-v8.4-boquilens-v1.bpk
cargo run --release -- predict --model yolov10n --weights target/yolov10n-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg

python tools/export_ultralytics_state.py yolo11n.pt target/yolo11n-state.pt
cargo run --release -- pack-weights --model yolo11n --input target/yolo11n-state.pt --output target/yolo11n-coco-ultralytics-v8.4-boquilens-v1.bpk
cargo run --release -- predict --model yolo11n --weights target/yolo11n-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg

python tools/export_ultralytics_state.py yolo26n.pt target/yolo26n-state.pt
cargo run --release -- pack-weights --model yolo26n --input target/yolo26n-state.pt --output target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk
cargo run --release -- predict --model yolo26n --weights target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg

python tools/export_ultralytics_state.py yolo11n-seg.pt target/yolo11n-seg-state.pt
cargo run --release -- pack-weights --model yolo11n-seg --input target/yolo11n-seg-state.pt --output target/yolo11n-seg-coco-ultralytics-v8.4-boquilens-v1.bpk
cargo run --release -- predict --model yolo11n-seg --weights target/yolo11n-seg-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg --masks
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

Single-image batch-1 inference latency per variant (release build, Flex CPU backend, writes the
numbers used by the README's performance table):

```console
cargo test --locked --release measures_single_inference_latency -- --ignored --nocapture --test-threads 1
```

The `--test-threads 1` flag is load-bearing: the parallel default makes the latency tests contend
for CPU and inflates the medians. The README compares those numbers against the official PyTorch
runtime measured with `tools/bench_ultralytics_cpu.py` (same machine, same methodology, fused
conv+bn) using the conversion venv.

PERF_NOTES.md records the methodology audit plus alternative-CPU-backend measurements. The
experimental `cpu-simd` (Flex `x86-v4`) and `cpu-cubecl` (`burn-cpu`) features exist only to
reproduce those measurements; `cpu-cubecl` is numerically unsound on burn 0.21.0-pre.4 (see
`tests/cpu_backend.rs`), and its latency/parity commands are documented there.

GPU numbers (Wgpu backend, requires the gpu feature) come from the `_gpu` variants of the same
harness; the CLI selects the backend with `--device cpu|gpu` (gpu builds print the chosen adapter
and graphics API on stderr):

```console
cargo test --locked --release --features gpu measures_single_inference_latency_gpu -- --ignored --nocapture --test-threads 1
cargo run --locked --release --features gpu -- predict --model yolo26n --device gpu --weights target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets/dog_bike_man.jpg
```

When touching the runtime or a backend boundary, also spot-check GPU/CPU parity by diffing
`--json` detections between `--device cpu` and `--device gpu` on the reference image.

Regenerate those fixtures with the per-family
`tools/export_{ultralytics,yolov10,yolo11,yolo26}_fixtures.py` scripts, passing `--model <id>` to
select the detect or seg scale. The seg variants also compare the full runtime end to end against
the official Ultralytics prediction (including per-detection mask IoU); generate their expectation
with `tools/export_yolo11_seg_e2e.py`. Generated checkpoints, fixtures, images, and build output
belong under `target/` and must not be committed.

## Invariants

- Public detections are continuous, unnormalized source-image `XYXY` box edges in pixels. They are
  not `XYWH` and not normalized; `xmax == width` and `ymax == height` are valid.
- YOLOX uses its existing top-left/raw-pixel transform. The Ultralytics-family models use
  Ultralytics-style stride-aligned rectangular letterboxing and RGB values normalized to `[0, 1]`.
- YOLOX batch norm uses PyTorch defaults (eps 1e-5, momentum 0.1), not the Ultralytics convention
  (eps 1e-3, momentum 0.03). Small running-variance channels make the epsilon difference visible at
  inference time — the golden tests against the official YOLOX sources catch it.
- YOLOv10 (all scales) is NMS-free: its one2one head output is top-300 selected and
  confidence-filtered like Ultralytics' end-to-end postprocess, not passed through non-maximum
  suppression. YOLO26 (all scales) shares that postprocess and is additionally DFL-free: the box
  tower emits the four XYXY side distances directly, decoded anchor-relative and scaled by the
  feature strides.
- YOLO11 (all scales) is NMS-based, unlike v10/26: it keeps the classic DFL head (`reg_max = 16`,
  light DWConv `cv3` tower) and the runtime applies the generic class-aware `nms()` helper to the
  head's center-size boxes and sigmoid scores with the `PredictOptions` thresholds (Ultralytics
  defaults conf 0.25, IoU 0.45). Ultralytics additionally caps post-NMS results at
  `max_det = 300`; the helper has no such cap, which is only observable on extremely dense
  predictions.
- Instance segmentation (YOLO11-seg, n/s) rides the same classic decode: Ultralytics' Segment head
  appends 32 **raw** mask coefficients per anchor to the `[boxes, scores]` rows (no sigmoid, and
  unlike some export paths there is no coefficient normalization in the PyTorch predict path —
  verified in the vendored 8.4.117 source), and the seg NMS is the same class-aware greedy
  suppression with the surviving anchors' coefficients carried along. Masks are assembled exactly
  like `ops.process_mask(..., upsample=True)`: `coefficients @ prototypes` (raw logits), bilinear
  upsample to the letterboxed canvas (`align_corners = False`), threshold `> 0`, crop to the box,
  and post-NMS detections whose cropped mask is fully empty are dropped. The Proto module runs on
  P3 and upsamples one stride level, so prototype maps live at stride 4; `parse_model` width-scales
  the 256 prototype channels (`npr`: 64 at n, 128 at s) and builds the `cv4` mask tower from full
  3x3 Convs (not the light DWConv flavor) with width `max(ch[0] / 4, 32)`.
- Instance masks are boolean coverage over the **source image** (`InstanceMask`: `Vec<bool>` +
  width/height, one byte per element). Each source pixel `(x, y)` samples the canvas mask at the
  nearest canvas pixel to `(x * scale + pad_x, y * scale + pad_y)` — the exact inverse of the
  letterbox geometry that maps box edges back. Boxes of segmentation detections follow the same
  source-image `XYXY` contract as `Detection`; `predict()` on a seg model returns the box branch
  only, and `predict_segmentation()` returns the masks. The CLI exposes the task via `--model
  yolo11n-seg`/`yolo11s-seg` and `--masks` (mask outlines on the annotated image, per-detection
  covered-pixel counts, JSON mask summary).
- f16 artifacts can flip a sharp (multi-peak) DFL side distribution and move one box edge of one
  detection by a couple of pixels versus the official fp32 runtime; on the reference image the
  observed worst case is ~2.8 px on one edge while the detection still matches at box IoU >= 0.98
  and mask IoU >= 0.99. Golden fixtures tolerate this via statistics; the end-to-end seg tests gate
  on box IoU and mask IoU rather than per-edge deltas.
- YOLO11's SPPF input projection keeps its SiLU activation even though current Ultralytics source
  constructs it `act=False`: the official checkpoints predate that refactor and the pickled modules
  still carry the activation. The golden tensor tests enforce the checkpoint behavior. YOLO26's
  SPPF (trained after the refactor) genuinely has no activation there, and its SPPF adds a residual
  (`SPPF, [1024, 5, 3, True]`) that YOLO11's does not.
- The YOLOv10 and YOLO26 scale variants are not pure width/depth rescalings. The official
  per-scale YAMLs swap module flavors: YOLOv10s uses large-kernel C2fCIB towers, YOLOv10
  m/b/l/x use the plain depth-wise C2fCIB flavor (and x converts backbone layer 6), and YOLO26
  m/l/x force `c3k=True` on the early backbone stages at 0.25 expansion. Each variant's body
  declares this explicitly; keep them aligned with the vendored YAMLs and `parse_model`.
  YOLO11 was verified to be a pure width/depth rescaling of one graph (same module flavors at
  every scale; only depth-scaled repeats and the shared m/l/x `c3k` rule change).
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

The next product work is a native weight distribution channel, `max_detections`, batching, YOLOX
training/loss parity, and YOLOX latency rows in the README performance table.
