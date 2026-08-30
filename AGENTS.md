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
  metadata. The family also ships `-seg` ImageNet COCO instance-segmentation variants (n/s/m/l/x;
  the same bodies plus Ultralytics' `Segment26` head: `one2one_cv4` mask tower and the `Proto26`
  module that fuses P4/P5 into the stride-4 prototypes, `segmentation.rs`) and `-cls` ImageNet-1k
  classification variants (n/s/m/l/x): the backbone truncated at the C2PSA stage plus Ultralytics'
  `Classify` head (1x1 conv to 1280 channels, global average pool, linear to 1000), at 224 px input
  (`classification.rs`). Classification checkpoints use plain PyTorch batch-norm defaults
  (eps 1e-5, momentum 0.1), not the Ultralytics-initialized values — see the BnFlavor invariant
  below.
- `src/models/yolo11/`: experimental YOLO11 (n/s/m/l/x) implementation, including the
  C3k2/plain-SPPF/C2PSA bodies (m/l/x force the C3k chain onto the early backbone stages; the P5
  stage is a plain C3k2 chain, not attention) with a classic DFL head (reg_max 16), NMS-based
  postprocessing, native Burnpack loading, and version metadata. All scales ship
  `-seg` instance-segmentation variants: the same bodies plus Ultralytics' Segment head (Proto
  module at stride 4, 32 mask coefficients per anchor) decoded through the same NMS with the
  coefficients carried along (`segment_head.rs`, masks assembled in `src/lib.rs`). The family also
  ships `-cls` ImageNet-1k classification variants (n/s/m/l/x, `classification.rs`): the classify
  graph is identical to YOLO26-cls (same YAML backbone, checkpoint key layout, plain-PyTorch BN
  flavor), so it reuses the shared classification modules from `src/models/yolo26/classification.rs`.
- `src/models/yolov8/`: experimental YOLOv8 (n/s/m/l/x) implementation, including the
  C2f/plain-SPPF bodies (a pure width/depth rescaling of one graph: backbone C2f stages carry
  shortcuts, neck stages do not) with a classic DFL head (reg_max 16) whose classification towers
  are the **legacy full-3x3-conv cv3 flavor** — the v8-era checkpoints predate YOLO11's light
  DWConv towers (verified from the pickled modules) — NMS-based postprocessing, native Burnpack
  loading, and version metadata. All scales ship `-seg` instance-segmentation variants (the same
  bodies plus Ultralytics' Segment head: stride-4 Proto with width-scaled `npr`
  64/128/192/256/320 and full-3x3-conv `cv4` mask towers; the runtime output type is shared with
  YOLO11-seg) and `-cls` ImageNet-1k classification variants: the cls backbone is a C2f chain
  **without** the C2PSA stage (head at model.9), every scale keeps max_channels 1024 with the n/s
  depth gain 0.33, and the batch norms carry plain PyTorch defaults (`BnFlavor::Pytorch`).
- `src/models/yolo12/`: experimental YOLO12 (n/s/m/l/x) implementation, including the
  area-attention bodies — backbone stages 6/8 are `A2C2f` blocks pairing the C2f-style split shell
  with two `ABlock`s each (area 4/1; l/x add the learnable per-channel gamma residual and
  mlp_ratio 1.2), neck stages 11/14/17 are the C3k-chain `A2C2f` flavor, and m/l/x force the C3k
  chain onto the early backbone C3k2 stages at 0.25 expansion — with a classic DFL head that is
  byte-identical to YOLO11's (light DWConv cv3 towers, verified from the checkpoints; the head
  module is shared from `src/models/yolo11/head.rs`), NMS-based postprocessing, native Burnpack
  loading, and version metadata.
- `src/data/letterbox.rs`: model-specific preprocessing and reversible source-image geometry.
- `src/data/augmentation/`: feature-gated native training augmentation pinned to Ultralytics
  `v8.4.117-2-g461196cf0`, including deterministic traceable detect/segment/classify pipelines,
  mixed-image transforms, mask formatting, and classification policies. Compatibility details and
  known tolerance classes live in `AUGMENTATION_COMPATIBILITY.md`.
- `src/lib.rs`: `ModelId`, `Predictor`, detection results, NMS integration, and weight packing API.
- `src/main.rs`: the `predict` and `pack-weights` CLI commands.
- `tools/`: development-only Ultralytics checkpoint conversion, golden-fixture generators, and the
  PyTorch CPU benchmark used for the README performance comparison.

## Fast path

Stable YOLOX inference uses the same native Burnpack workflow as every other family (substitute any
`yolox-nano|tiny|s|m|l|x` name):

```console
cargo run --release -- pack-weights --model yolox-nano --input target/yolox_nano.pth --output target/yolox-nano-coco-official-v0.1.1rc0-boquilens-v1.bpk
cargo run --release -- predict --model yolox-nano --weights target/yolox-nano-coco-official-v0.1.1rc0-boquilens-v1.bpk --source assets/dog_bike_man.jpg
```

Ultralytics-family models require a tensor-only state and then a native artifact. The complete
workflow is documented in `README.md`; the short form
(substitute any `yolov10n/s/m/b/l/x`, `yolo11n/s/m/l/x`, `yolov8n/s/m/l/x`, `yolo12n/s/m/l/x`, or
`yolo26n/s/m/l/x` name; task variants follow the same loop with their suffixes — `yolo11n/s/m/l/x-seg`,
`yolov8n/s/m/l/x-seg`, `yolo26n/s/m/l/x-seg`, and the `-cls` classification variants of the v8/11/26
families):

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

Letterbox preprocessing cost (JPEG decode, both letterbox transforms, scaler candidates, and
canvas parity diffs; writes comparison PNGs under `target/`) is measured by the ignored
`measures_letterbox_resize_cost` test; see PERF_NOTES.md §5 for the recorded
`fast_image_resize` adoption/evaluation numbers:

```console
cargo test --locked --release measures_letterbox_resize_cost -- --ignored --nocapture
```

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
- YOLOX uses top-left letterboxing followed by RGB `/255` and ImageNet mean/std normalization. The
  Ultralytics-family models use stride-aligned rectangular letterboxing and RGB values in `[0, 1]`.
- YOLOX batch norm uses eps 1e-3 and momentum 0.03. The release checkpoints are state dicts, so
  these non-serialized settings must match the official experiment constructor.
- YOLOX uses the same native `.bpk` runtime contract as every other family. The official `.pth` is
  a conversion/parity input only, never the normal runtime format.
- YOLOv10 (all scales) is NMS-free: its one2one head output is top-300 selected and
  confidence-filtered like Ultralytics' end-to-end postprocess, not passed through non-maximum
  suppression. YOLO26 (all scales) shares that postprocess and is additionally DFL-free: the box
  tower emits the four XYXY side distances directly, decoded anchor-relative and scaled by the
  feature strides.
- YOLO11, YOLOv8, and YOLO12 (all scales) are NMS-based, unlike v10/26: they keep the classic DFL
  head (`reg_max = 16`) and the runtime applies the generic class-aware `nms()` helper to the head's
  center-size boxes and sigmoid scores with the `PredictOptions` thresholds (Ultralytics defaults
  conf 0.25, IoU 0.45). Ultralytics additionally caps post-NMS results at `max_det = 300`; the
  helper has no such cap, which is only observable on extremely dense predictions. YOLO11 and
  YOLO12 build the light DWConv `cv3` classification tower; YOLOv8's checkpoints predate that
  refactor and build the legacy full-3x3-conv `cv3` tower (verified from the pickled modules — the
  current `Detect` source would construct the light tower for a fresh v8 YAML, but the released
  checkpoints win).
- Instance segmentation (YOLO11-seg and YOLOv8-seg, n/s/m/l/x) rides the same classic decode:
  Ultralytics' Segment head appends 32 **raw** mask coefficients per anchor to the
  `[boxes, scores]` rows (no sigmoid, and unlike some export paths there is no coefficient
  normalization in the PyTorch predict path — verified in the vendored 8.4.117 source), and the seg
  NMS is the same class-aware greedy suppression with the surviving anchors' coefficients carried
  along. Masks are assembled exactly like `ops.process_mask(..., upsample=True)`:
  `coefficients @ prototypes` (raw logits), bilinear upsample to the letterboxed canvas
  (`align_corners = False`), threshold `> 0`, crop to the box, and post-NMS detections whose
  cropped mask is fully empty are dropped. The Proto module runs on P3 and upsamples one stride
  level, so prototype maps live at stride 4; `parse_model` width-scales the 256 prototype channels
  (`npr`: 64/128 at v8 n/s, 192/256/320 at v8 m/l/x; 256/256/384 at yolo11 m/l/x) and builds the
  `cv4` mask tower from full 3x3 Convs (not the light DWConv flavor) with width
  `max(ch[0] / 4, 32)`. YOLOv8-seg returns YOLO11-seg's runtime output type so the decode and mask
  assembly stay shared.
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
- Instance segmentation (YOLO26-seg, n/s/m/l/x) rides the end2end decode instead of NMS: the
  Segment26 head appends 32 **raw** mask coefficients to the one2one rows, Ultralytics' end2end
  postprocess (top-300 anchors by best score, top-300 pairs, confidence filter, **no NMS**) is
  applied with the coefficients gathered along, and masks assemble exactly like
  `ops.process_mask(upsample=True)` as in YOLO11-seg. The YOLO26 `Proto26` module differs from
  YOLO11's P3-only Proto: P4/P5 are 1x1-refined, nearest-upsampled 2x/4x, summed onto P3, fused by
  a 3x3 conv, and then run through the classic conv/upsample/conv/proto projection; prototype
  channels are `make_divisible(min(256, max_channels) * width, 8)` (64/128/256/256/384) and the
  mask tower is full 3x3 Convs with hidden width `max(ch[0] / 4, 32)` (32/32/64/64/96).
- End2end (one2one) heads keep near-duplicate detections classic NMS would suppress, and the weak
  duplicates' scores sit in the top-k near-tie region where f16 rounding reorders membership: the
  seg end-to-end tests exempt duplicates (same class, IoU >= 0.9 with a stronger expected
  detection) from the confidence gate, require every non-duplicate official detection at conf
  >= 0.55 to be matched, and relax the mask IoU gate to 0.85 for masks under 2000 covered pixels
  (boundary-dominated; 0.92 observed on a 313-px mask).
- Classification inference runs at 224 px with Ultralytics' classify transform: anti-aliased
  shortest-edge resize, centered 224x224 crop, RGB scaled to `[0, 1]` (identity normalization
  constants). The 1000-way softmax is preprocessing-rounding sensitive: near-tied classes can swap
  adjacent ranks between PIL and the Rust resize even though each probability moves by <1%, so the
  end-to-end classification tests compare the top-5 class set plus per-class probabilities (3e-2)
  instead of rank order. Ultralytics fed boquilens' canvas reproduces boquilens' probabilities
  exactly — the golden fixtures pin the graph at 2e-4 on the shared canvas.
- YOLO11's SPPF input projection keeps its SiLU activation even though current Ultralytics source
  constructs it `act=False`: the official checkpoints predate that refactor and the pickled modules
  still carry the activation. YOLOv8's SPPF is the same era and also keeps SiLU on `cv1` (verified
  from the pickled module; the v8 checkpoints carry neither the `n` repeat count nor the `add`
  shortcut attribute). The golden tensor tests enforce the checkpoint behavior. YOLO26's SPPF
  (trained after the refactor) genuinely has no activation there, and its SPPF adds a residual
  (`SPPF, [1024, 5, 3, True]`) that YOLO11's and YOLOv8's do not. YOLO12 has no SPPF at all.
- YOLO12's area-attention blocks (`A2C2f`/`ABlock`/`AAttn`) follow the vendored source with one
  checkpoint-era quirk: the 7x7 depth-wise positional-encoding convolution (`AAttn.pe`) ships a
  conv bias in the official checkpoints even though the current `Conv` wrapper is bias-free — the
  Rust module carries the bias. The l/x scales extend the backbone `A2C2f` YAML args with
  `(residual=True, mlp_ratio=1.2)`, adding the learnable per-channel gamma residual around the
  whole block; n/s/m keep `residual=False, mlp_ratio=2.0` and have no gamma. The neck `A2C2f`
  stages (11/14/17) run the C3k chain flavor (`a2=False`) whose shell concatenates `1 + n` tensors
  (not C3k2's `2 + n` — the split shell starts from a single `c_`-wide tensor, not two halves).
- Classification checkpoints (YOLO26-cls, YOLO11-cls, and YOLOv8-cls) carry plain PyTorch
  `nn.BatchNorm2d` defaults — eps 1e-5, momentum 0.1 — not the Ultralytics-initialized values
  (eps 1e-3, momentum 0.03) the detect families use (verified from the pickled `bn.eps`). The same
  eps-vs-running-variance visibility lesson as YOLOX applies. The yolov8/yolo26 blocks expose a
  `BnFlavor` (`Ultralytics` default, `Pytorch` for classification); every Conv in a classify graph
  must opt into `BnFlavor::Pytorch` explicitly. Golden fixtures are the gate that catches a wrong
  flavor. The YOLOv8-cls graph differs structurally from 26/11-cls (C2f backbone without the
  C2PSA stage, head at model.9, max_channels 1024 at every scale), so only the `Classify` head
  module is shared with YOLO26.
- The YOLOv10 and YOLO26 scale variants are not pure width/depth rescalings. The official
  per-scale YAMLs swap module flavors: YOLOv10s uses large-kernel C2fCIB towers, YOLOv10
  m/b/l/x use the plain depth-wise C2fCIB flavor (and x converts backbone layer 6), and YOLO26
  m/l/x force `c3k=True` on the early backbone stages at 0.25 expansion. Each variant's body
  declares this explicitly; keep them aligned with the vendored YAMLs and `parse_model`.
  YOLO11 was verified to be a pure width/depth rescaling of one graph (same module flavors at
  every scale; only depth-scaled repeats and the shared m/l/x `c3k` rule change), and YOLOv8 the
  same (Conv/C2f/SPPF everywhere; backbone C2f stages carry shortcuts, neck stages do not).
  YOLO12's per-scale rules: the m/l/x scales force `c3k=True` on the early backbone C3k2 stages
  (layers 2/4) at 0.25 expansion, and the l/x scales extend the backbone `A2C2f` stages with
  `(residual=True, mlp_ratio=1.2)`.
- Keep model graph code independent of CLI, filesystem, rendering, and image decoding.
- Training detection/segmentation augmentation stays HWC BGR `u8` until Format; default Format
  emits CHW RGB `u8`, while classification converts to RGB before torchvision-compatible policy
  transforms and emits normalized CHW `f32` after RandomErasing. Native seed output is a stable
  boquilens contract; cross-language parity uses injected parameters/traces, not equal seed values.
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

## Native training invariants

- Training is non-default and WGPU-only (`--features training`); default inference graphs and
  prediction outputs must remain unchanged.
- Run real training and hardware smoke workflows with `--release`; unoptimized Burn model graphs
  are too slow for meaningful trainer verification. Unit tests and static checks keep their
  documented profiles unless the test itself executes a model-training loop.
- Losses consume raw logits. YOLOX alone uses objectness; modern heads use TAL. YOLOv10/YOLO26
  training builds carry one-to-many plus detached-feature one-to-one branches, and YOLO26 remains
  DFL-free. YOLO26-seg also detaches one-to-one prototypes and semantic logits.
- Assignment may synchronize detached values to the host, but totals remain connected to the model
  graph and empty batches remain finite.
- Resumable checkpoints are full precision and include model, optimizer, EMA, scheduler/loss
  schedule, progress, model specification, ordered class names, and payload hashes. Inference
  Burnpacks are lossy exports and are never resume inputs.
- After training changes also run:

```console
cargo test --locked --features training training
cargo clippy --locked --features training --all-targets -- -D warnings
```

Reference fixtures and generated datasets/checkpoints/reports belong under `target/`.
