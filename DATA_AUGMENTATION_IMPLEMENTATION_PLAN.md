# Native Ultralytics Data-Augmentation Port — Detailed Implementation Plan

Status: design only; no augmentation implementation exists yet.

This document is a standalone plan for bringing the Ultralytics training data-augmentation behavior into boquilens. It is intentionally separate from `TRAINING_IMPLEMENTATION_PLAN.md`: the training plan owns model trainability, losses, optimizers, checkpoints, and the training loop; this plan owns decoded-image and annotation transformations up to the batch handed to that loop.

The short answer to “can we copy it?” is **yes, as an attributed AGPL adaptation**. boquilens is AGPL-3.0, and the vendored Ultralytics augmentation source is AGPL-3.0. The implementation must preserve source provenance, identify modifications, ship the applicable license notices, and pin the behavior being copied. Classification augmentation is partly implemented by torchvision rather than Ultralytics itself, so any copied torchvision policy tables or algorithms need their BSD-3-Clause notice as well. This is an engineering plan, not legal advice.

## 1. Decision and desired outcome

Implement a native Rust, CPU-side augmentation subsystem that reproduces the training behavior of the vendored Ultralytics tree closely enough that:

- the same source sample, partner samples, and recorded transform parameters produce equivalent images and annotations;
- detect, instance-segmentation, and classification datasets receive the task-correct pipeline;
- all randomness is deterministic and replayable under a boquilens seed;
- augmentation does not require Python, PyTorch, OpenCV, Albumentations, or torchvision at normal Rust training time;
- the vendored Python stack remains a development-only oracle for fixtures and differential tests;
- the final tensors can be transferred to a single Burn GPU without putting image augmentation on the GPU;
- validation preprocessing remains deterministic and does not accidentally inherit training augmentation;
- the implementation can later support additional tasks without weakening current detect/segment/classify contracts.

“Equivalent” does not mean merely similar-looking. Geometry, target filtering, polygon ordering, mask rasterization, color-space arithmetic, rounding, and transform ordering all affect the loss. The first implementation should favor observable parity over a more elegant but behaviorally different augmentation design.

## 2. Exact behavior baseline

### 2.1 Source pin

Treat the following source as the normative detection and segmentation oracle:

- repository: sibling vendored `../ultralytics` tree;
- version description: `v8.4.117-2-g461196cf0`;
- commit: `461196cf09175b64c9b9bd8babebf081c0540520`;
- primary implementation: `ultralytics/data/augment.py`;
- annotation container: `ultralytics/utils/instance.py`;
- mask rasterization: `ultralytics/data/utils.py`;
- dataset wiring: `ultralytics/data/base.py` and `ultralytics/data/dataset.py`;
- batch-side multi-scale resize: `ultralytics/models/yolo/detect/train.py`;
- late-epoch mosaic closure: `ultralytics/engine/trainer.py`.

Do not silently track the sibling checkout after implementation. Record the commit in Rust module documentation, fixture manifests, and the augmentation compatibility report. Future upstream synchronization must be an explicit reviewed change.

### 2.2 Classification oracle pin

Ultralytics delegates classification transforms to torchvision. Before implementing classification policy transforms:

1. Record the Python, PyTorch, torchvision, Pillow, and OpenCV versions in the fixture environment.
2. Record the source commit or release tag for torchvision.
3. Generate policy fixtures with that exact version.
4. Copy or adapt only behavior actually used by the selected Ultralytics configuration.
5. Retain the torchvision BSD-3-Clause notice if code or policy tables are adapted.

The initial classification compatibility target is the default boquilens-supported YOLO11-cls and YOLO26-cls training configuration, not every historical torchvision transform version.

### 2.3 License and provenance checklist

Before merging implementation code:

- add an entry to `NOTICE` naming Ultralytics, its pinned source/version, AGPL-3.0, the adapted files, and the fact that the Rust code is modified;
- put an attribution header or module-level documentation on each substantially adapted Rust module;
- keep the repository’s AGPL-3.0 license declaration accurate;
- include the torchvision BSD-3-Clause text if torchvision implementation details or policy tables are copied;
- include the legacy Albumentations MIT notice if implementation details are directly adapted from that project rather than independently implemented from observable behavior;
- do not imply that Ultralytics, PyTorch, or Albumentations endorses boquilens;
- document source-code availability obligations for distributed binaries/services as part of the project’s existing AGPL release process;
- do not copy generated checkpoint artifacts into the repository; augmentation compatibility does not change checkpoint licensing.

Suggested adapted-module notice, with final wording reviewed against project conventions:

```text
This module is adapted from the Ultralytics data augmentation pipeline at
commit 461196cf09175b64c9b9bd8babebf081c0540520 and modified for native Rust.
Ultralytics source: https://github.com/ultralytics/ultralytics
License: AGPL-3.0. See LICENSE and NOTICE.
```

Do not invent copyright ownership or years that are not present in the source being adapted.

Primary licensing references for the implementation review:

- [Ultralytics licensing overview](https://www.ultralytics.com/license)
- [Ultralytics AGPL-3.0 explanation](https://www.ultralytics.com/legal/agpl-3-0-software-license)
- [torchvision BSD-3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE)
- [legacy Albumentations repository and MIT-license status](https://github.com/albumentations-team/albumentations)

## 3. Scope

### 3.1 In scope for the first complete release

- detection training augmentation for every detection model family;
- polygon-based instance-segmentation augmentation for every supported segmentation family;
- ImageNet-style classification augmentation for YOLO11-cls and YOLO26-cls;
- deterministic validation preprocessing for all three tasks;
- Mosaic-4, with Mosaic-9 implemented before claiming full class parity;
- random affine/perspective geometry;
- CopyPaste in `flip` and `mixup` modes;
- MixUp and CutMix, even though their default probabilities are zero;
- the default Ultralytics Albumentations photometric set;
- HSV augmentation and horizontal/vertical flips;
- letterbox formatting and instance-mask target formatting;
- classification RandomResizedCrop, flips, RandAugment, ColorJitter fallback, normalization, and RandomErasing;
- optional classification AutoAugment and AugMix, required before claiming all supported `auto_augment` values;
- rectangular-training interactions;
- `close_mosaic` scheduling;
- trainer-side multi-scale resizing;
- seeded replay traces, parity fixtures, benchmarks, and failure diagnostics.

### 3.2 Explicitly out of scope for the first merge

- arbitrary Python Albumentations objects passed through config;
- dynamically loading Python augmentation plugins from Rust;
- GPU augmentation kernels;
- CUDA-specific data loaders;
- pose/keypoint, OBB, semantic-segmentation, depth, and text-prompt training as public supported tasks;
- RLE-mask augmentation without a polygon conversion stage;
- changing inference preprocessing;
- replacing the existing inference letterbox before parity proves a shared implementation is safe;
- automatic upstream synchronization.

The internal annotation types should reserve clean extension points for keypoints, OBBs, semantic masks, and depth maps because the pinned source carries those fields through many transforms. They should not be exposed as supported training tasks until their losses and parity suites exist.

## 4. Pipeline topology

### 4.1 Detection and instance segmentation, training

The exact top-level order is load/resize, then:

```text
primary raw sample
  -> Mosaic
  -> CopyPaste(flip mode only)
  -> RandomPerspective
  -> CopyPaste(mixup mode only; obtains its own pre-transformed partner)
  -> MixUp(partner receives the same pre-transform)
  -> CutMix(partner receives the same pre-transform)
  -> default/custom photometric transforms
  -> RandomHSV
  -> RandomFlip(vertical)
  -> RandomFlip(horizontal)
  -> Format(image + boxes + optional mask targets + batch index)
```

The repeated `pre_transform` behavior is load-bearing. A partner selected by MixUp, CutMix, or mixed-image CopyPaste is not simply decoded and pasted; it goes through Mosaic and RandomPerspective first. The Rust design must model nested transform invocation and nested random draws without borrowing mutable dataset state unsafely.

For `copy_paste_mode = flip`, CopyPaste is inserted between Mosaic and RandomPerspective. It copies eligible polygons from a horizontally flipped version of the same mosaic result.

For `copy_paste_mode = mixup`, CopyPaste is appended after RandomPerspective and obtains a separately pre-transformed secondary sample. Its configured `p` participates both in the outer mix-transform application decision and in selecting a fraction of eligible secondary instances. Preserve that double role unless a future upstream baseline changes it.

### 4.2 Detection and instance segmentation, validation

```text
raw sample
  -> LetterBox(target = imgsz, scaleup = false)
  -> Format(bbox = normalized XYWH, optional mask targets, bgr = 0)
```

No Mosaic, random perspective, color transform, flip, random BGR output, MixUp, CutMix, CopyPaste, or RandomErasing belongs in validation.

### 4.3 Classification, training

Default configuration resolves to:

```text
decoded BGR bytes
  -> BGR-to-RGB conversion
  -> RandomResizedCrop(224, area scale = 0.5..1.0, ratio = 3/4..4/3, bilinear)
  -> RandomHorizontalFlip(p = 0.5)
  -> no vertical flip by default
  -> RandAugment(default torchvision policy)
  -> no ColorJitter while auto augmentation is enabled
  -> uint8 RGB to CHW float [0, 1]
  -> Normalize(mean = 0, std = 1)
  -> RandomErasing(p = 0.4, in place)
```

The crop range is derived by the dataset as `(1.0 - args.scale, 1.0)`. With the global default `scale = 0.5`, classification uses `0.5..1.0`; it does not use the helper function’s generic fallback `0.08..1.0`.

### 4.4 Classification, validation

```text
decoded BGR bytes
  -> BGR-to-RGB conversion
  -> shortest-edge Resize(224, bilinear, aspect preserving)
  -> centered 224 x 224 crop
  -> RGB CHW float [0, 1]
  -> Normalize(mean = 0, std = 1)
```

This must remain aligned with the existing boquilens classification inference transform. Reuse only after exact crop, anti-aliasing, and rounding behavior is proven compatible.

## 5. Default configuration to expose

Define a serializable `AugmentationConfig` with validation and task-specific resolution. The Ultralytics-compatible defaults are:

| Field | Default | Meaning |
|---|---:|---|
| `imgsz` | `640` detect/segment, `224` classify | target input size |
| `rect` | `false` | rectangular batches; disables Mosaic/MixUp/CutMix |
| `close_mosaic` | `10` | final epochs with Mosaic/CopyPaste/MixUp/CutMix disabled |
| `multi_scale` | `0.0` | per-batch scale variation around `imgsz` |
| `overlap_mask` | `true` | one indexed overlapping instance-mask target |
| `mask_ratio` | `4` | target mask downsample ratio |
| `hsv_h` | `0.015` | hue gain fraction |
| `hsv_s` | `0.7` | detection/segment saturation gain |
| `hsv_v` | `0.4` | value gain |
| `degrees` | `0.0` | rotation range in degrees |
| `translate` | `0.1` | translation fraction |
| `scale` | `0.5` | geometric scale range, interpreted as `0.5..1.5` |
| `shear` | `0.0` | shear range in degrees |
| `perspective` | `0.0` | perspective coefficient range |
| `flipud` | `0.0` | vertical flip probability |
| `fliplr` | `0.5` | horizontal flip probability |
| `bgr` | `0.0` | chance final detection tensor remains BGR instead of RGB |
| `mosaic` | `1.0` | Mosaic event probability |
| `mixup` | `0.0` | MixUp event probability |
| `cutmix` | `0.0` | CutMix event probability |
| `copy_paste` | `0.0` | CopyPaste probability/fraction |
| `copy_paste_mode` | `flip` | `flip` or `mixup` |
| `auto_augment` | `randaugment` | classification policy |
| `erasing` | `0.4` | classification RandomErasing probability |

Additional explicit fields should include:

- `mosaic_grid: Four | Nine`, default `Four`;
- `mask_overlap: bool` as the Rust spelling of `overlap_mask` while accepting the compatibility alias;
- `classification_crop_ratio`, default `0.75..1.3333334`;
- `classification_force_color_jitter`, default `false`;
- `classification_mean` and `classification_std`, defaults `[0,0,0]` and `[1,1,1]`;
- `interpolation`, default compatibility bilinear;
- `seed: u64` owned by the trainer rather than hidden in this config;
- `compatibility: Ultralytics84117 | Native`, default `Ultralytics84117` for copied behavior;
- `trace: Off | Failures | All`, default `Failures` in tests and `Off` in release training;
- an explicit list of supported photometric transforms rather than an untyped plugin value.

Reject invalid ranges during config resolution. Examples: probabilities outside `[0,1]`, non-positive image size, `mask_ratio == 0`, `mask_ratio > min(height,width)`, invalid CopyPaste mode, inverted scale tuples, negative perspective, unsupported classification policy, and `scale >= 1.0` where `(1-scale)` would make the classification crop lower bound non-positive.

## 6. Crate/module layout

Do not put the implementation in model graph modules or `src/lib.rs`. Proposed layout:

```text
src/data/
  augmentation/
    mod.rs                 public-in-crate contracts and pipeline builder
    config.rs              validated user config and resolved task config
    rng.rs                 seed derivation, distributions, replay trace
    sample.rs              image/annotation/metadata container
    instances.rs           boxes, polygons, future keypoint hooks
    compose.rs             transform trait and composition
    mosaic.rs              Mosaic-4 and Mosaic-9
    perspective.rs         matrix construction, warps, candidate filtering
    copy_paste.rs          flip and mixup modes
    mixup.rs               Beta(32,32) image mix
    cutmix.rs              empty-region CutMix behavior
    photometric.rs         blur, median blur, grayscale, CLAHE
    hsv.rs                 OpenCV-compatible BGR/HSV LUT transform
    flip.rs                image and annotation flips
    letterbox.rs           training/validation compatibility letterbox
    mask.rs                polygon fill and target formatting
    format.rs              HWC bytes -> CHW tensor-ready sample
    classify.rs            classification crop/policies/erase
    resize.rs              compatibility resize/warp kernels
    trace.rs               serializable applied-transform trace
tools/
  export_augmentation_fixtures.py
tests/
  augmentation_parity.rs
```

If the existing crate wishes to keep training behind a feature, place dependency-heavy modules behind `training` without hiding the pure geometry unit tests. Inference builds must not gain scheduler, data-loader, or training-only public APIs accidentally.

## 7. Core data contracts

### 7.1 Byte image

Use an owned, contiguous `ByteImage` with:

```rust
struct ByteImage {
    width: usize,
    height: usize,
    channels: usize,
    layout: Hwc,
    color: ColorOrder, // Bgr, Rgb, Gray, MultiChannel
    data: Vec<u8>,
}
```

Detection and segmentation augmentation should stay in BGR `u8` through photometric transforms, matching OpenCV-oriented Ultralytics behavior. `Format` performs BGR-to-RGB channel reversal by default. Classification converts to RGB before PIL/torchvision-equivalent transforms.

Do not use `DynamicImage` as the mutation-heavy augmentation container. It is useful at decode/API boundaries but hides channel-order assumptions and makes exact multi-channel behavior awkward. Convert once at the boundary.

### 7.2 Canonical instances

Implement an `Instances` equivalent carrying:

```rust
struct Instances {
    boxes: Vec<BBox>,
    box_format: BoxFormat,       // Xyxy, Xywh, Ltwh
    normalized: bool,
    segments: Option<Vec<Polygon>>,
    keypoints: Option<Keypoints>, // internal extension hook only
}
```

Class IDs remain a parallel vector because that mirrors Ultralytics filtering/reordering. Every operation that filters or reorders instances must return an index map used on classes and any future per-instance metadata. Add debug assertions after every transform:

- `classes.len() == instances.len()`;
- segments are absent, or `segments.len() == instances.len()`;
- all coordinates are finite;
- normalized state agrees across boxes and segments;
- image dimensions are non-zero;
- class IDs are in the dataset range;
- no method adds absolute padding while coordinates are normalized.

Required operations:

- convert box formats without changing normalization;
- normalize/denormalize;
- scale boxes only or boxes plus segments;
- add absolute padding;
- horizontal and vertical flip;
- clip to continuous image edges `[0,width] x [0,height]`;
- remove zero-area boxes and return the keep mask;
- slice/reorder by indexes;
- concatenate, resampling polygon point counts when compatibility requires it;
- calculate areas;
- derive boxes from transformed segments.

Use continuous box edges. Do not clamp `x2` to `width - 1`; `x2 == width` is valid and consistent with project invariants.

### 7.3 Augmentation sample

```rust
struct AugSample {
    image: ByteImage,
    classes: Vec<u32>,
    instances: Instances,
    source: SourceMetadata,
    geometry: GeometryMetadata,
    semantic_mask: Option<LabelMap>, // reserved
    depth: Option<DepthMap>,         // reserved
}
```

`SourceMetadata` should retain the primary source index/path and the list of mixed source indices for diagnostics. `GeometryMetadata` should retain original shape, current resized shape, and validation ratio/padding where relevant. Mixed samples must not pretend their geometry can be reversed to one source image.

### 7.4 Final formatted sample

Do not construct device tensors inside worker threads. Emit a backend-independent sample:

```rust
struct FormattedDetectionSample {
    image_chw_u8: Vec<u8>,
    image_shape: [usize; 3],
    classes: Vec<u32>,
    boxes_xywh_normalized: Vec<[f32; 4]>,
    masks: Option<MaskTargets>,
}
```

The collator stacks images, creates batch indexes, flattens targets, converts image bytes to float `[0,1]`, and transfers the batch to the selected Burn device. Keeping bytes until collation reduces worker memory and preserves exact format output.

Classification emits a CHW `f32` image because RandomErasing happens after float conversion/normalization in the pinned torchvision pipeline.

## 8. Transform interface and composition

Use an object-safe or enum-dispatched transform interface with three concerns separated:

```rust
trait Transform {
    fn sample_params(&self, sample: &AugSample, ctx: &mut AugContext) -> Result<TransformParams>;
    fn apply(&self, sample: AugSample, params: &TransformParams, ctx: &mut AugContext) -> Result<AugSample>;
}
```

Sampling and application must be separable so a fixture can inject Python-recorded parameters. Every transform trace includes:

- transform kind and compatibility version;
- whether the probability gate passed;
- all sampled scalar/vector parameters;
- all partner dataset indexes;
- selected object indexes;
- matrix values and output dimensions for geometric transforms;
- before/after instance counts;
- optional image checksum before/after;
- nested child traces for partner pre-transforms.

`Compose` executes in declaration order and does not reorder “no-op” transforms. Even a no-op can consume random draws in the reference pipeline; compatibility tests must decide whether to preserve that draw. The safest implementation samples exactly where the pinned source samples and records the result.

## 9. Deterministic randomness and replay

Ultralytics currently draws from Python `random`, NumPy random, torch seeds, and third-party libraries. Reproducing those raw streams bit-for-bit in Rust is not a useful production contract. Instead implement two modes:

1. **Parameter-injected oracle mode:** Python exports every chosen parameter and partner index. Rust applies those exact choices. This proves transform mathematics and ordering.
2. **Native seeded mode:** Rust owns a stable RNG algorithm and seed derivation. Repeated boquilens runs are identical even though the random sequence need not equal Python’s sequence.

Use a documented stable generator such as ChaCha12, pinned by dependency version and tested with fixed output vectors. Derive a sample seed from immutable values:

```text
run_seed
  + epoch
  + logical sample position
  + dataset sample index
  + distributed rank (future)
  + worker-independent transform stream tag
```

Hash those fields with an explicitly versioned derivation function. Do not derive behavior from worker scheduling, thread ID, OS entropy, hash-map iteration order, or cache-hit order.

Split RNG streams by transform path. A disabled optional photometric transform should not perturb Mosaic partner selection unless the compatibility version intentionally changes. Nested MixUp/CutMix pre-transforms receive child streams derived from the parent trace path.

Implement and test distributions required by the pipeline:

- uniform `f32/f64` closed/open boundary semantics as selected;
- uniform integer ranges;
- Bernoulli probability gates;
- Beta(32,32) for MixUp;
- Beta(1,1) by default for CutMix;
- random choice from an index list;
- Fisher-Yates or equivalent selection where policy tables require it;
- log-uniform aspect-ratio sampling for RandomResizedCrop;
- normal/uniform fills for RandomErasing according to the pinned torchvision behavior.

The trace schema needs a version number. A replay file from an older schema must fail clearly rather than applying changed defaults.

## 10. Dataset loading, resize, and partner selection

### 10.1 Detection/segmentation load behavior

Match the source loader before augmentation:

- decode to BGR bytes;
- preserve original `(height,width)`;
- for ordinary rectangular-mode loading, resize the long side to `imgsz` while maintaining aspect ratio;
- compute resized dimensions using `ceil(original * ratio)`, capped at `imgsz`;
- use OpenCV-compatible bilinear interpolation;
- preserve one-channel images as HWC with one channel;
- stretch to square only when the dataset explicitly disables rectangular-mode loading;
- cache the resized result, not a mutated augmented result.

All augmentation paths clone or copy-on-write the loaded sample. Mosaic, HSV, and CopyPaste must never mutate the cached source image.

### 10.2 Mix buffer

The source buffer maximum is:

```text
min(dataset_size, batch_size * 8, 1000)
```

When cache mode is not RAM, Mosaic draws partners from recently loaded buffered indexes; with full RAM cache it may draw from the whole dataset. Model this policy explicitly as `PartnerPool::RecentBuffer` or `PartnerPool::WholeDataset`.

For deterministic native training, buffer membership must be independent of worker race timing. Recommended implementation:

- build partner candidates from deterministic sampler history in the main data-loader scheduler; or
- use a deterministic index window keyed by logical sample position instead of a physically race-dependent cache buffer.

If strict parity with Python’s recent-buffer state is required, record partner indexes in oracle traces. Do not make production determinism depend on reproducing Python worker timing.

### 10.3 Rectangular training

When `rect = true`:

- set Mosaic, MixUp, and CutMix probabilities to zero before building the pipeline;
- preserve CopyPaste behavior exactly as the pinned source resolves it, while noting that segment CopyPaste generally becomes ineffective if configured zero by the user/default;
- use batch-specific `rect_shape` as the RandomPerspective output size when present;
- keep images in an aspect-ratio-grouped batch;
- prove the collator can stack every image in a rectangular batch.

Add a dedicated integration test because a square-only implementation can appear correct until rectangular batches are enabled.

## 11. Mosaic implementation

### 11.1 Probability and source selection

Mosaic is a mix transform. On a passed probability gate:

- Mosaic-4 requires the primary plus three partner samples;
- Mosaic-9 requires the primary plus eight partners;
- partner indexes come from the configured partner pool;
- repeated indexes are permitted if the source permits them;
- every loaded partner carries cloned classes/instances;
- text/category remapping is out of current scope but must not be silently corrupted.

### 11.2 Mosaic-4 geometry

For target side `s`:

1. Allocate a `2s x 2s` HWC canvas filled with byte value `114` in every channel.
2. Use border `(-s/2,-s/2)`.
3. Sample mosaic center `(xc,yc)` uniformly from `[-border, 2s + border]`, which resolves to `[s/2, 3s/2]` for each axis.
4. Place the primary sample in the top-left placement case.
5. Place three partner samples in top-right, bottom-left, and bottom-right cases.
6. For each placement, compute destination rectangle clipped to the canvas and the matching source crop rectangle.
7. Record `padw = destination_x1 - source_x1` and `padh` similarly.
8. Denormalize each source’s annotations against its own resized patch size, convert to XYXY, add placement padding, and concatenate.
9. Clip boxes and segment points to `[0,2s]`.
10. Remove zero-area boxes and apply the same keep indexes to classes/segments.
11. Pass the `2s x 2s` result to RandomPerspective, whose target is `s x s`.

Do not resize each image again inside Mosaic unless the source loader behavior requires it. Preserve arbitrary patch dimensions after long-side resize.

### 11.3 Mosaic-9 geometry

Implement the pinned center-plus-eight-neighbors layout, not a generic 3x3 resize grid:

- allocate `3s x 3s`, fill `114`;
- place the primary at the center;
- place the remaining images in the pinned sequence around it using the preceding image dimensions to compute offsets;
- clip each placement;
- crop the final canvas using the negative Mosaic border so the result is `2s x 2s`;
- add the border offset to annotation padding before concatenation;
- then feed the result through RandomPerspective to `s x s`.

Fixture every placement case with differently colored synthetic images and boxes touching all edges. This catches swapped x/y dimensions and off-by-one source crops.

### 11.4 Segments and reserved semantic masks

- Transform polygon vertices using the same scale and padding as boxes.
- Fill uncovered semantic-label pixels with ignore value `255` if semantic support is later enabled.
- Preserve one segment per segmentation instance.
- If polygons have different resampled point counts, follow the pinned `Instances.concatenate` behavior rather than concatenating ragged storage incorrectly.

### 11.5 Mosaic acceptance tests

- exact canvas dimensions and fill values;
- exact source/destination crop rectangles for fixed centers;
- exact transformed box coordinates;
- polygon coordinates within tolerance appropriate to `f32` operations;
- classes remain aligned after clipping/removal;
- no cached source mutation;
- deterministic partner and center selection;
- Mosaic disabled path returns an observationally unchanged sample;
- empty-label images work in every quadrant;
- grayscale/multi-channel behavior is either correct or rejected with a clear supported-format error.

## 12. RandomPerspective implementation

### 12.1 Matrix construction

Build `f32` 3x3 matrices in this exact conceptual order:

```text
M = T * S * R * P * C
```

Applied right-to-left:

1. `C`: move input image center to the origin using `-width/2`, `-height/2`.
2. `P`: sample x and y perspective coefficients uniformly in `[-perspective,+perspective]`.
3. `R`: sample angle in `[-degrees,+degrees]`; sample scale from explicit tuple or `[1-scale,1+scale]`; construct the OpenCV-compatible rotation matrix around origin.
4. `S`: sample x/y shear angles independently and store their tangents.
5. `T`: sample x/y translations around `0.5` of output dimensions with configured fractional range.

Record the fully materialized matrix in the trace. Matrix multiplication order, row/column convention, and `f32` rounding need golden unit tests.

### 12.2 Image warp

- use perspective warp when configured perspective is non-zero;
- otherwise use affine warp from the first two matrix rows;
- output exactly the requested `(width,height)`;
- fill outside pixels with `114` in every channel;
- match OpenCV’s inverse mapping, bilinear interpolation, border handling, coordinate-center convention, and byte rounding;
- preserve one-channel HWC representation.

Do not assume `fast_image_resize` implements affine/perspective warps. Introduce a small compatibility raster kernel or select a Rust crate only after differential tests prove its semantics. OpenCV may be used by the Python fixture generator, not as a required production runtime dependency.

### 12.3 Box transformation

For each XYXY box:

- form four corners in the pinned order;
- multiply homogeneous points by `M^T` under the chosen representation;
- divide x/y by homogeneous z only for perspective box handling;
- take min/max x/y over transformed corners;
- preserve `f32` values until final formatting.

### 12.4 Segment transformation

- transform every polygon point homogeneously;
- divide by z;
- derive the replacement box using only visible segment coordinates within output size;
- clip segment coordinates to the derived box for ordinary axis-aligned tasks;
- clip the resulting instances to image boundaries.

The segment-derived box path is intentional; do not transform the old box and ignore the polygon.

### 12.5 Candidate filtering

Before comparing area, scale each original box by the sampled geometric scale. Keep a transformed candidate only if all are true:

```text
new_width  > 2 pixels
new_height > 2 pixels
new_area / (scaled_old_area + epsilon) > threshold
max(new_width/new_height, new_height/new_width) < 100
```

Use area threshold `0.10` for boxes and `0.01` when segments are present. Pin epsilon and strict-vs-inclusive comparisons to the source.

### 12.6 Tests

- identity matrix and same-size image is unchanged;
- pure translation, scale, rotation, shear, and perspective fixtures;
- border fill and bilinear samples at edges;
- a box transformed by each matrix matches Python;
- polygons partially and fully outside output;
- candidate exactly at each threshold boundary;
- NaN/infinite homogeneous results are rejected deterministically;
- output with zero surviving instances remains valid.

## 13. CopyPaste implementation

### 13.1 Common eligibility

CopyPaste runs only when segment polygons exist and `p > 0`.

1. Convert primary instances to absolute XYXY.
2. Obtain candidate instances:
   - `flip`: deep-copy primary and horizontally flip boxes/segments;
   - `mixup`: use a separately selected and pre-transformed sample.
3. Compute candidate-box intersection-over-area against every current primary box.
4. Keep candidates whose IOA is below `0.30` against every primary box.
5. Sort eligible candidates by ascending maximum IOA.
6. Select the first `round(p * eligible_count)` using Python-compatible rounding semantics.
7. Rasterize selected polygons into one binary paste mask.
8. Copy source pixels wherever the paste mask is true.
9. Append selected classes and instances in selection order.

Python uses banker’s rounding for `round`; Rust’s ordinary integer conversions are not equivalent. Provide and unit-test a compatibility rounding helper.

### 13.2 Flip mode

- source pixels are a horizontal flip of the current image before any paste;
- class IDs come from the current sample;
- pasted geometry is the horizontally flipped candidate geometry;
- no outer Bernoulli event gate is used beyond `p == 0`; `p` primarily determines selected object count.

### 13.3 Mixup mode

- apply the outer mix-transform probability gate using `p`;
- select one partner and run its nested Mosaic/RandomPerspective pre-transform;
- use partner pixels/classes/segments;
- use `p` again to choose `round(p * eligible_count)` objects.

This unintuitive double use must be documented in user-facing compatibility notes.

### 13.4 Raster semantics

Polygon fill must match OpenCV `drawContours(..., FILLED)`/`fillPoly` behavior for integer-cast points, including edge inclusivity and self-intersections. Build fixture polygons for:

- axis-aligned rectangles;
- concave shapes;
- points exactly on image edges;
- negative points clipped by earlier transforms;
- overlapping selected polygons;
- thin one-pixel structures.

## 14. MixUp implementation

On a passed event gate:

1. Select one partner.
2. Run the partner through the shared pre-transform.
3. Sample `r ~ Beta(32,32)`.
4. Require equal image shape and channels.
5. Compute `primary * r + secondary * (1-r)` per channel.
6. Cast to `u8` with NumPy-compatible truncation/saturation behavior.
7. Concatenate primary then secondary classes and instances.

Do not create soft per-instance class targets or weight losses by `r`; the pinned detector implementation simply concatenates labels. Reserved semantic masks select the primary mask when `r >= 0.5` and the secondary mask when `r < 0.5`.

Tests need ratios at `0`, `0.5`, `1`, and fractional values that expose truncation. Native random tests should statistically sanity-check the Beta distribution without using fragile exact histograms.

## 15. CutMix implementation

The pinned detector CutMix differs from textbook classification CutMix.

1. Pass the outer event probability.
2. Select and pre-transform one partner.
3. Generate `num_areas = 3` candidates by default.
4. For each area, sample `lambda ~ Beta(beta,beta)` with `beta = 1`.
5. Compute cut ratio `sqrt(1-lambda)` and integer cut width/height.
6. Sample an integer center and clip rectangle to image bounds.
7. Compute IOA between every candidate rectangle and primary boxes.
8. Retain only rectangles with total primary IOA `<= 0`; if none exist, skip CutMix.
9. Randomly choose one retained rectangle.
10. Find secondary objects with rectangle IOA at least `0.10`, or `0.01` when segments exist.
11. If none qualify, skip.
12. Copy the exact secondary pixel rectangle into the primary image.
13. Slice qualifying secondary instances, shift them to rectangle-local coordinates, clip to rectangle size, then shift back.
14. Append qualifying classes and instances.

Primary boxes are not clipped because the selected cut region has no overlap with them. Do not implement standard label-area interpolation; it would be behaviorally different.

Test no-free-area, no-secondary-object, edge-clipped area, segmentation threshold, exact non-overlap, and empty primary labels.

## 16. Default photometric transforms

The default Albumentations wrapper is invoked with top-level `p = 1.0`; child probabilities decide application:

| Transform | Child probability |
|---|---:|
| Blur | `0.01` |
| MedianBlur | `0.01` |
| ToGray | `0.01` |
| CLAHE | `0.01` |
| RandomBrightnessContrast | `0.0` |
| RandomGamma | `0.0` |
| ImageCompression, quality 75..100 | `0.0` |

Implement the enabled four transforms for default parity. Also define native configurable variants for the disabled three so enabling their documented config does not require Python.

For each operation, pin legacy Albumentations version and parameter defaults in fixtures. A name and probability are not enough: blur kernel selection, CLAHE tile grid/clip limit, grayscale channel replication, gamma rounding, and JPEG codec settings can differ by version and platform.

Recommended release tiers:

- Tier A: byte-exact or tightly bounded parity for default Blur, MedianBlur, ToGray, and CLAHE;
- Tier B: documented native equivalents for brightness/contrast and gamma;
- Tier C: image compression, whose JPEG output may require tolerance-based parity across codec builds.

Arbitrary custom Albumentations transforms are not accepted as opaque YAML/Python values. Define a typed Rust registry. Unknown transforms produce a config error listing supported names. Spatial custom transforms should wait until their annotation behavior is individually implemented and tested.

## 17. RandomHSV implementation

This transform operates on three-channel BGR `u8` images.

1. Sample three independent values uniformly from `[-1,1]`.
2. Multiply by `[hsv_h,hsv_s,hsv_v]`.
3. Build byte-domain lookup inputs `x = 0..255`.
4. Hue LUT: `((x + hue_gain * 180) mod 180)` cast to `u8`.
5. Saturation LUT: clip `x * (1 + saturation_gain)` to `[0,255]`, cast to `u8`.
6. Value LUT: clip `x * (1 + value_gain)` to `[0,255]`, cast to `u8`.
7. Force `sat_lut[0] = 0` so pure white does not gain color.
8. Convert BGR to OpenCV-compatible HSV, apply LUTs, and convert back into the same image storage.

Do not use a floating-point HSV conversion from a generic image crate without parity evidence. OpenCV’s integer conversion ranges hue to `[0,179]` and has observable rounding behavior. Implement a compatibility kernel with exhaustive tests over all single-channel values and a large sampled RGB cube, using Python/OpenCV output as fixtures.

The hue behavior is the post-8.3.78 additive formulation. Do not copy the older multiplicative formula.

## 18. RandomFlip implementation

The order is vertical first, horizontal second. Each transform samples its own gate.

Before annotation flipping, convert boxes to XYWH as the pinned implementation does. For normalized instances use dimensions `1 x 1`; otherwise use actual pixel width/height.

Horizontal flip:

- reverse image columns;
- set XYWH center `x = width - x`;
- set every polygon `x = width - x`;
- future keypoints use the configured reflection index after coordinate flip.

Vertical flip mirrors y equivalently.

Ensure the reversed image becomes contiguous before formatting; a negative-stride view has no Rust equivalent and should not leak into tensor conversion.

Test boxes on all edges, normalized/absolute states, polygons, double flips, probability 0/1, and vertical-then-horizontal ordering.

## 19. LetterBox implementation

Implement all source parameters even if validation initially uses only defaults:

- `new_shape`;
- `auto` minimum rectangle;
- `scale_fill` stretching;
- `scaleup`;
- `center`;
- `stride`;
- padding value `114`;
- interpolation.

Algorithm:

1. `r = min(target_height/source_height, target_width/source_width)`.
2. If `scaleup = false`, cap `r` at `1.0`.
3. Set ratio `(r,r)`.
4. Compute unpadded dimensions with Python-compatible `round(width*r)` and `round(height*r)`.
5. Compute remaining width/height padding.
6. If `auto`, reduce each padding by modulo stride.
7. If `scale_fill`, set padding to zero, stretch to target, and use independent x/y ratios.
8. If centered, divide padding by two.
9. Compute sides using `round(dh - 0.1)`, `round(dh + 0.1)`, and x equivalents.
10. Resize bilinearly and fill borders with `114`.
11. Convert boxes to absolute XYXY, scale, then add left/top padding.
12. Resize reserved semantic masks nearest-neighbor and fill with `255`.

Preserve validation `ratio_pad` metadata for metric conversion. Compare this implementation with `src/data/letterbox.rs`; merge shared primitives only if inference fixtures remain unchanged. A training feature must not silently alter published inference behavior.

## 20. Polygon-to-mask formatting

### 20.1 Rasterization order

For each polygon:

1. allocate a full-resolution zero `u8` mask;
2. cast points to signed 32-bit integers with NumPy-compatible truncation;
3. fill the polygon at full image resolution;
4. resize the completed mask to `(height / mask_ratio, width / mask_ratio)` using the pinned OpenCV default interpolation;

The order is fill then resize. Rasterizing directly at reduced resolution is not equivalent.

### 20.2 Overlap mode, default

- build one integer mask with background `0`;
- compute each binary mask’s reduced-resolution area;
- sort instances by descending area;
- reorder instances and classes by the same indexes;
- write sorted instance IDs `1..N` using running pixelwise maximum;
- use `u8` for at most 255 instances and a wider integer type above 255;
- derive a semantic class map by looking up class ID from nonzero instance IDs if the loss needs it.

The running maximum avoids overflow that older summation logic could trigger. Include a regression fixture with more than 128 and more than 255 overlapping polygons.

### 20.3 Non-overlap mode

- emit one binary reduced mask per instance;
- keep original instance order;
- if deriving a semantic map at overlaps, choose the smallest-area covering instance as the pinned implementation does;
- empty samples emit shape-consistent zero masks.

### 20.4 Output validation

- `mask_ratio <= min(height,width)`;
- mask dimensions use integer floor division;
- overlap IDs never exceed the selected storage type;
- every reordered class and box corresponds to the same mask;
- polygon fill parity is tested independently from resize parity.

## 21. Final detect/segment Format

At formatting time:

1. read final image height/width;
2. pop classes and instances from the mutable augmentation sample;
3. convert boxes to XYWH;
4. denormalize to current image dimensions if needed;
5. format segment masks and reorder instances/classes when overlap mode sorts them;
6. normalize box x/w by width and y/h by height;
7. convert HWC BGR bytes to contiguous CHW;
8. reverse channels to RGB when a random draw is greater than `bgr`; with default `bgr = 0`, always output RGB;
9. retain image as `u8` until the collator converts to float and divides by 255;
10. create batch indexes during collation rather than embedding a meaningless all-zero tensor per worker sample.

The exact BGR probability condition is important: `bgr` is the probability of retaining BGR, not the probability of converting to RGB.

Do not filter tiny boxes during Format; RandomPerspective owns its candidate filter. Do reject malformed non-finite targets before they enter the loss.

## 22. Classification augmentation

### 22.1 RandomResizedCrop

Port the pinned torchvision algorithm, including its bounded retry and center-crop fallback:

- sample target area uniformly from the configured area fraction;
- sample log aspect ratio uniformly from `log(3/4)..log(4/3)`;
- derive integer crop width/height using compatibility rounding;
- accept if crop fits the source;
- sample top/left inclusively over valid origins;
- after the pinned number of failed attempts, compute deterministic centered fallback based on source aspect ratio;
- crop then resize to square output with pinned bilinear and anti-alias behavior.

The fixture exporter must record crop rectangle and resized output, allowing crop math and interpolation to be tested separately.

### 22.2 RandAugment

Default classification training requires the pinned torchvision RandAugment policy. Port:

- default number of operations;
- magnitude-bin count;
- sampled operation selection;
- operation-specific magnitude mapping;
- random sign where applicable;
- interpolation and fill behavior;
- all operations in the exact pinned policy space;
- operation order and dtype conversions.

Build table-driven tests for every operation at minimum, midpoint, and maximum magnitude. Record policy metadata in the trace.

### 22.3 AutoAugment and AugMix

Implement these before advertising compatibility for their config strings. Keep each policy in a dedicated module/table with provenance. Tests should inject policy choices and magnitudes rather than relying only on random end-to-end examples.

If they are deferred in the first PR, config resolution must return a clear unsupported-feature error. It must not silently disable a user-requested policy, even though old torchvision-version guards in Python may log and continue.

### 22.4 ColorJitter fallback

When `auto_augment` is absent, or `force_color_jitter = true`, apply brightness, contrast, saturation, and hue using the pinned torchvision ordering/randomization. Note classification defaults use `hsv_s = 0.7` from global config when called by the dataset, even though the helper signature’s standalone default is `0.4`.

### 22.5 Tensor conversion and normalization

- RGB byte to CHW float divides by `255`;
- default mean `[0,0,0]`, standard deviation `[1,1,1]` is identity after scaling;
- custom standard deviations must be positive;
- preserve `f32` operation order for fixture parity.

### 22.6 RandomErasing

Port pinned torchvision defaults for scale, ratio, value, attempt count, and fallback. Apply after normalization. With default identity normalization this appears like ordinary `[0,1]` erasing, but ordering matters for custom normalization.

Trace whether erasing occurred, rectangle, channels/fill values, and failed-attempt fallback.

## 23. Multi-scale training

Multi-scale is batch-side, after collation and conversion to float/device in the pinned trainer:

1. If `multi_scale > 0`, sample a target side from the stride-bounded range around `imgsz`.
2. Integer-divide by model stride and multiply by stride.
3. Set scale factor relative to the batch’s maximum spatial dimension.
4. If scale differs, compute each new dimension with `ceil(dim * scale / stride) * stride`.
5. Bilinearly resize the full image batch with `align_corners = false`.
6. Leave normalized targets unchanged.

For single-GPU Burn, this can occur immediately before forward. Confirm Burn’s interpolation output against PyTorch. If parity is inadequate, use a CPU compatibility resize before transfer or implement a dedicated tested kernel.

The sampled size must use the trainer’s deterministic batch RNG stream, not a worker RNG. Record it in checkpoints if exact resume is required.

## 24. Late-epoch `close_mosaic`

At epoch `epochs - close_mosaic`:

- set Mosaic, CopyPaste, MixUp, and CutMix probabilities to zero;
- rebuild the dataset transform graph;
- reset/restart the data loader so workers observe the new graph;
- retain RandomPerspective, default photometric transforms, HSV, and flips;
- log the transition once;
- persist the effective augmentation phase in checkpoint/resume state.

On resume after the transition epoch, construct the closed graph immediately. Do not run one accidental epoch of Mosaic because the loader was built from initial defaults.

Test `close_mosaic = 0`, greater than epoch count, exact boundary, and resume on both sides of the boundary.

## 25. Dependency strategy

Likely Rust additions, subject to license/MSRV audit:

- `rand` plus a pinned stable generator crate;
- a Beta/distribution implementation or a small audited implementation;
- a polygon rasterizer if it can match OpenCV fill semantics;
- image-processing primitives only where compatibility tests pass.

Avoid adding OpenCV as a normal dependency. It complicates builds and would make native Rust training depend on a system library. It is acceptable in the Python oracle environment.

Before choosing a crate:

- inspect its license and transitive dependency licenses;
- benchmark allocation behavior;
- test Windows/Linux output consistency;
- verify exact coordinate conventions;
- ensure it supports one-channel and three-channel HWC data;
- pin versions in `Cargo.lock`;
- record rejected candidates and parity results in implementation notes.

## 26. Python oracle and fixture format

Create `tools/export_augmentation_fixtures.py` in the implementation phase. It should import only the pinned sibling Ultralytics tree and emit:

```text
target/augmentation-fixtures/<fixture-id>/
  manifest.json
  input-primary.png
  input-partner-0.png
  input-annotations.json
  params.json
  output.png
  output-annotations.json
  masks.npy or masks.png
```

`manifest.json` includes:

- Ultralytics commit/version;
- Python/package versions;
- operating system and architecture;
- fixture generator commit/hash;
- transform config;
- source image hashes;
- output image hash;
- dtype, layout, color order, and dimensions;
- random parameters and partner indexes;
- tolerance class: exact, bounded pixels, geometry epsilon, or statistical only.

Fixtures belong under `target/` and must not be committed, consistent with current project policy. Commit only small synthetic constants directly in Rust tests when necessary.

## 27. Differential parity methodology

Use three layers of parity.

### Layer 1: parameter-injected transform units

Feed the same synthetic image/annotations and exact parameters to Python and Rust. Compare:

- image dimensions and channel order exactly;
- pixels exactly where feasible;
- maximum/mean absolute pixel error where codec/interpolation variance is unavoidable;
- instance count/order/classes exactly;
- boxes and polygon points with explicit epsilon;
- mask IDs and area ordering exactly.

### Layer 2: recorded full-pipeline traces

Python selects all random values and partner indexes, exports the trace, and Rust replays it. This catches ordering, nested partner transforms, filtering, and formatting.

Cover at least:

- detect with default Mosaic/HSV/hflip;
- detect with all geometric knobs non-zero;
- segment with CopyPaste flip;
- segment with CopyPaste mixup;
- MixUp and CutMix enabled;
- rectangular training;
- empty labels;
- dense overlapping segments;
- classification default RandAugment/erase;
- classification validation.

### Layer 3: native seeded invariants/statistics

Run thousands of native samples and verify:

- deterministic rerun hashes;
- event frequencies within broad confidence bounds;
- sampled parameter ranges;
- no NaNs/panics;
- box/mask/class alignment;
- class distribution does not change unexpectedly except through intended mixing;
- memory remains bounded.

Do not demand Rust seed `N` produce the same random choices as Python seed `N`; injected traces own cross-language mathematical parity.

## 28. Test matrix

### 28.1 Unit tests

- box conversions and edge contracts;
- normalize/denormalize round trip;
- polygon scale/pad/flip/clip;
- banker’s rounding helper;
- matrix construction and multiplication;
- bilinear/nearest resize coordinate conventions;
- warp border and rounding;
- BGR/HSV conversions and LUTs;
- polygon fill;
- overlap mask ordering/storage width;
- every random distribution fixed vector;
- trace serialization round trip;
- config validation and default resolution.

### 28.2 Transform tests

- every transform disabled, forced, and boundary-probability behavior;
- primary/partner shapes and channels mismatch errors;
- no-label samples;
- one-pixel and zero-area geometry;
- very small images;
- large polygons and more than 255 instances;
- nested pre-transform replay;
- source-cache immutability.

### 28.3 Pipeline tests

- detect train/val output schemas;
- segment train/val masks and class alignment;
- classify train/val output schemas;
- default pipeline order snapshot;
- close-mosaic graph snapshot;
- rect graph snapshot;
- batch collation and batch indexes;
- single-GPU transfer with no worker-owned device tensors;
- checkpoint/resume deterministic next batch.

### 28.4 Existing project regression tests

Run the project’s normal handoff commands after implementation:

```console
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --no-default-features --lib
```

Also run ignored inference parity tests when fixtures are present. The augmentation work must not alter inference detections, masks, classification probabilities, CPU/GPU behavior, or weight packing.

## 29. Performance and memory plan

The goal is to keep a single GPU fed, not to maximize augmentation benchmark scores in isolation.

Measure separately:

- image decode;
- initial long-side resize;
- Mosaic copy;
- affine/perspective warp;
- HSV conversion/LUT;
- polygon rasterization;
- final HWC-to-CHW formatting;
- collation and host-to-device transfer;
- end-to-end data wait time visible to the trainer.

Implementation guidelines:

- reuse scratch canvases per worker where sizes match;
- use one allocation for Mosaic canvas;
- use copy-on-write or ownership transfer so no transform clones unnecessarily;
- preserve cache entries as immutable resized images;
- store final detect/segment images as `u8` until batch preprocessing;
- avoid one full-resolution mask per instance when overlap mode can stream into the indexed target, except where area sorting requires temporary masks;
- cap prefetch queues by batches and account their worst-case Mosaic memory;
- pin host memory only if Burn/backend support and benchmarks show benefit;
- keep worker count configurable and default conservatively on Windows;
- log data-loader wait percentage so users can distinguish GPU/model bottlenecks from augmentation bottlenecks.

Acceptance target: after warmup, median batch data wait should be comfortably below model forward/backward time for the smallest supported training model on the reference single-GPU machine. Record actual hardware and batch settings rather than asserting a universal number.

## 30. Error handling and diagnostics

Errors must identify:

- transform name and nested trace path;
- primary and partner sample IDs;
- source/current image dimensions and channel count;
- instance/class/segment counts;
- relevant sampled parameters;
- trace replay path if written;
- whether failure occurred in decode, geometry, rasterization, formatting, collation, or transfer.

In debug/test builds, validate after every transform. In release builds, retain cheap shape/count/finite checks at trust boundaries.

On a worker failure, stop training cleanly and surface the original error. Do not silently drop the sample, substitute a blank image, or loop forever selecting another sample; those policies alter the data distribution and can hide corrupt labels.

## 31. Metrics and run metadata

Record the resolved augmentation config in every run and checkpoint, including:

- compatibility version;
- source oracle commit;
- seed and seed-derivation version;
- effective task pipeline;
- image size, rect, and multi-scale;
- every transform probability/range;
- mask ratio/overlap mode;
- classification policy;
- close-mosaic transition epoch;
- resize/raster kernel versions;
- dataset cache mode and worker count.

Log per-epoch counters for transform application/skip outcomes and average surviving instances. These are debugging metrics, not promises of exact probabilities on small datasets.

## 32. Phased implementation sequence

### Phase A — provenance, contracts, and deterministic skeleton

- add notices/provenance;
- add validated config;
- add byte image, instances, sample, transform, compose, RNG, and trace types;
- add synthetic geometry tests;
- add a no-op detect pipeline and collator contract.

Exit gate: deterministic sample replay and annotation invariants pass without image transforms.

### Phase B — validation and formatting first

- implement compatibility resize;
- implement LetterBox;
- implement HWC BGR to CHW RGB Format;
- implement normalized XYWH targets;
- implement polygon fill and overlap/non-overlap masks;
- compare validation outputs to Python.

Exit gate: detect/segment validation fixtures pass and existing inference is unchanged.

### Phase C — basic single-image training transforms

- RandomPerspective;
- RandomHSV;
- vertical/horizontal RandomFlip;
- default four photometric transforms;
- full default detect pipeline without Mosaic.

Exit gate: parameter-injected fixtures pass, including non-zero geometry knobs.

### Phase D — mixed-image transforms

- deterministic partner provider/buffer;
- Mosaic-4;
- Mosaic-9;
- MixUp;
- CutMix;
- CopyPaste flip and mixup;
- nested trace/replay.

Exit gate: complete detection and segmentation full-pipeline replay passes.

### Phase E — classification

- pin torchvision oracle/version and notice;
- RandomResizedCrop and validation resize/crop;
- flips;
- RandAugment;
- tensor conversion/identity normalization;
- RandomErasing;
- optional ColorJitter, AutoAugment, and AugMix.

Exit gate: YOLO11/26 classification train and validation fixture sets pass.

### Phase F — trainer integration

- worker pool and cache integration;
- collator/device transfer;
- multi-scale batch resize;
- close-mosaic transition and resume;
- run metadata and counters;
- single-GPU throughput benchmarks.

Exit gate: an interrupted/resumed run produces the same next augmented batches and training proceeds without GPU starvation.

### Phase G — hardening

- fuzz geometry/annotation operations;
- cross-platform fixture run;
- large-instance/memory stress;
- no-default-features checks;
- docs and compatibility report;
- audit unsupported custom-transform errors.

Exit gate: all acceptance criteria below are met.

## 33. Pull-request slicing

Keep reviews small enough to verify behavior:

1. contracts/config/RNG/trace;
2. resize/letterbox/format/masks;
3. perspective/HSV/flips;
4. photometric defaults;
5. Mosaic;
6. MixUp/CutMix/CopyPaste;
7. classification transforms;
8. loader/trainer scheduling/performance;
9. compatibility documentation and final audit.

Each PR should include its Python oracle fixture generator changes and parity tests. Do not land a transform based only on visual examples.

## 34. Definition of done

The augmentation port is complete only when all of the following hold:

- [ ] Source commit and third-party versions are pinned.
- [ ] AGPL/BSD/MIT provenance obligations applicable to copied code are represented in source and `NOTICE`.
- [ ] Detect, segment, and classification pipelines use the correct task-specific ordering.
- [ ] Default config resolves to the pinned Ultralytics values.
- [ ] Validation has no random training augmentation.
- [ ] Images remain BGR bytes through detect/segment photometric augmentation and become RGB at Format by default.
- [ ] Boxes use continuous edge coordinates and end as normalized XYWH training targets.
- [ ] Polygon transformations remain aligned with classes and boxes.
- [ ] Overlap and non-overlap mask targets match the oracle.
- [ ] Mosaic-4 and Mosaic-9 match parameter-injected fixtures.
- [ ] RandomPerspective matrix, warp, and candidate filtering match fixtures.
- [ ] CopyPaste flip/mixup, MixUp, and CutMix match fixtures.
- [ ] HSV’s additive hue LUT and white-saturation guard match fixtures.
- [ ] Default photometric operations have documented parity tolerances.
- [ ] Classification default RandAugment and RandomErasing match the pinned torchvision behavior.
- [ ] Arbitrary unsupported custom transforms fail clearly.
- [ ] Native seeded runs are worker-scheduling independent.
- [ ] Every augmentation can emit a replay trace on failure.
- [ ] Rectangular training, multi-scale, and close-mosaic interactions are tested.
- [ ] Resume reconstructs RNG and augmentation phase exactly.
- [ ] Single-GPU training consumes formatted batches without Python runtime dependencies.
- [ ] Data wait and memory benchmarks are recorded.
- [ ] Existing inference and checkpoint-loading tests remain unchanged and passing.
- [ ] `cargo fmt`, tests, clippy, and no-default-features checks pass.

## 35. Known traps to keep visible during implementation

- “Copying the pipeline” includes partner pre-transforms and Format, not only the named augmentations.
- Detection augmentation is BGR until Format; applying RGB HSV math changes colors.
- The current hue transform is additive, not the historical multiplicative formula.
- `lut_sat[0] = 0` is observable on white pixels.
- RandomPerspective matrix order is `T * S * R * P * C`.
- Segment-derived boxes replace transformed old boxes.
- Box candidate area thresholds differ for boxes (`0.10`) and segments (`0.01`).
- Mosaic creates a `2s` intermediate that RandomPerspective reduces to `s`.
- CopyPaste `flip` uses `p` as selected-object fraction without a conventional outer event gate.
- CopyPaste `mixup` uses `p` both as event probability and selected-object fraction.
- Detector MixUp concatenates hard labels; it does not weight targets by the blend ratio.
- Detector CutMix deliberately chooses a primary-object-free rectangle.
- Letterbox uses Python rounding plus `-0.1/+0.1` border rounding.
- Mask polygons are filled at full resolution before downsampling.
- Overlap masks sort instances by descending mask area and reorder classes/boxes.
- More than 255 overlapping instances require a wider index type.
- The classification crop lower scale resolves to `0.5`, not helper fallback `0.08`.
- Auto augmentation disables classification ColorJitter unless forced.
- Classification RandomErasing occurs after float conversion and normalization.
- Multi-scale happens to the whole batch and leaves normalized targets unchanged.
- Rect mode disables Mosaic/MixUp/CutMix at graph construction.
- Late close-mosaic rebuilds and resets the loader.
- Python’s multiple RNG sources should be captured as parameters, not emulated as one fake stream.
- Worker timing must not determine native partner selection or random output.
- Reusing inference resize code before parity can regress existing model output.
- JPEG compression parity may depend on codec version; use explicit tolerances and metadata.

## 36. Relationship to model training

The training implementation should consume this subsystem through a narrow interface:

```text
dataset sample + epoch/sample seed
  -> task augmentation pipeline
  -> formatted CPU sample
  -> collator
  -> optional batch multi-scale
  -> Burn tensor batch on one GPU
  -> model forward and task loss
```

Model-family loss code must not know whether a sample came through Mosaic, CopyPaste, or no augmentation. Conversely, augmentation code must not branch on YOLO11 versus YOLO26 graph internals; it branches only on task, image size/stride requirements, and resolved configuration. YOLOX-specific training recipes may choose different augmentation defaults later, but should reuse the same tested primitives where their source behavior agrees.

## 37. Final recommendation

Proceed with a behaviorally faithful native adaptation, but make **record/replay parity** the organizing principle. The risky parts are not the high-level ideas; they are pixel-center conventions, rounding, BGR/HSV arithmetic, nested partner pipelines, polygon rasterization, and target reordering. Implement validation/formatting first, then single-image transforms, then mixed-image transforms, and only then connect the worker pool to training. That sequence gives every later loss implementation a stable, testable input contract instead of debugging augmentation and model gradients simultaneously.
