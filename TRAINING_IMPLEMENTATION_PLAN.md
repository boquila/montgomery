# Native training implementation plan

Status: design document only  
Scope baseline: repository state on 2026-08-28  
Primary implementation target: one local GPU through Burn/WGPU  
Reference implementations: vendored Ultralytics 8.4.117 at `../ultralytics` (commit `461196cf0`) and
the official YOLOX 0.1.1rc0 source unpacked under `target/yolox-ref/`

This document is the implementation plan for adding native Rust/Burn training to every model and
task currently exposed by `ModelId`. It deliberately does not change the inference implementation.
The implementation should be delivered in small, parity-gated phases; a model is not considered
trainable merely because a loss decreases.

## 1. Goal and definition of done

The finished product must support training, fine-tuning, validation, interruption/resume, and
inference export for all 40 currently registered variants:

- YOLOX detect: nano, tiny, s, m, l, x.
- YOLOv3-Tiny-U detect: tiny-u.
- YOLOv10 detect: n, s, m, b, l, x.
- YOLO11 detect: n, s, m, l, x.
- YOLO11 instance segmentation: n, s.
- YOLO11 classification: n, s, m, l, x.
- YOLO26 detect: n, s, m, l, x.
- YOLO26 instance segmentation: n, s, m, l, x.
- YOLO26 classification: n, s, m, l, x.

"Support" has the following concrete meaning for every row above:

1. A model can be initialized from scratch or fine-tuned from a compatible pretrained checkpoint.
2. A custom class count and ordered class-name table can replace COCO-80 or ImageNet-1k heads.
3. The forward pass exposes training logits and auxiliary branches without changing the existing
   inference output contract.
4. The native loss and target assignment match the appropriate official reference on fixed
   fixtures, including empty-target batches and crowded images.
5. One optimizer step matches the reference after accounting for documented backend floating-point
   tolerance.
6. A tiny dataset can be intentionally overfit to near-zero training error.
7. Validation produces task-appropriate metrics: box mAP, box and mask mAP, or top-1/top-5.
8. `last`, `best`, and periodic checkpoints can be resumed exactly enough to reproduce the next
   sample order, learning rate, loss schedule, optimizer state, and EMA update.
9. The EMA model can be exported into the existing inference path, with training-only branches
   removed and class metadata retained.
10. Existing CPU/GPU inference, weight packing, public box geometry, and golden tests remain green.

The first complete vertical slice should be YOLOX-nano detection. The shared infrastructure must
then be proven on YOLOv3-Tiny-U before extending it to the modern dual-head, segmentation, and
classification variants. Scale variants should require fixture generation and dispatch wiring, not
new loss implementations.

## 2. Non-goals for the first complete release

The architecture must leave room for these features, but they do not block single-GPU training:

- Multi-GPU DDP, FSDP, parameter servers, or multi-node execution.
- TPU, WebGPU-in-browser, CPU-only production training, or the currently unsound `cpu-cubecl`
  backend.
- Pose, oriented boxes, semantic-only segmentation, depth, or model families not currently in
  `ModelId`.
- A Python dependency in the production training path. Python remains a development-only oracle for
  parity fixtures.
- Perfect reproduction of nondeterministic GPU reduction order. Deterministic sample order and
  restart behavior are required; bitwise identity across different adapters/drivers is not.
- Every Ultralytics augmentation in the first milestone. Plain resize/letterbox plus horizontal
  flip must work first; mosaic, mixup, copy-paste, HSV, affine, and multi-scale are later parity
  gates in this same plan.
- Fused optimizer kernels or aggressive mixed precision before an fp32 correctness baseline exists.
- Reusing f16 inference Burnpacks as resumable training checkpoints. They are intentionally lossy.

## 3. Current-state constraints that drive the design

### 3.1 The current crate is inference-only

`Cargo.toml` enables Burn tensor and backend functionality but not Burn's `train`, `dataset`, or
`autodiff` feature set. The CLI has only `predict` and `pack-weights`. There is no dataset manifest,
batch type, train loop, loss, optimizer state, scheduler, EMA, metric evaluator, or resumable
checkpoint format.

The GPU feature currently selects the inference `Wgpu` backend. Training should wrap that backend as
`Autodiff<Wgpu>` and should remain behind a new non-default `training` feature so default inference
builds do not gain trainer dependencies or compile time.

### 3.2 Inference outputs are not training outputs

Training must consume logits and pre-postprocess tensors. It must never compute loss from sigmoid
probabilities, top-k detections, NMS results, source-image boxes, or thresholded masks.

- YOLOX currently applies sigmoid to objectness and class outputs inside `Head::forward` and returns
  only decoded predictions. Training needs raw objectness/class logits, raw box deltas, decoded box
  tensors, grids, and strides.
- YOLOv3-Tiny-U and YOLO11 already expose raw DFL box distributions and class logits at head level,
  but their model-level `forward` methods decode and sigmoid them.
- YOLOv10 and YOLO26 currently implement only the one-to-one inference branch. Official training
  requires both one-to-many and one-to-one branches.
- YOLO11 segmentation currently returns decoded detection values, raw mask coefficients, and
  prototypes. Training needs raw detection logits plus the same coefficients/prototypes.
- YOLO26 segmentation additionally omits the training-only semantic tower in `Proto26`.
- Classification returns logits and probabilities. Cross-entropy must consume `logits` only.

### 3.3 Burn batch norm mode is backend-sensitive

Burn 0.21.0-pre.4 `BatchNorm::forward` uses batch statistics and updates running state when autodiff
is enabled, and uses running statistics when it is not. Therefore:

- Training forwards use `Autodiff<Wgpu>`.
- Validation and export use `model.valid()` (the inner non-autodiff model), never the live autodiff
  model directly.
- Validation must not mutate running means/variances.
- Running-state synchronization must complete before cloning/converting a validation or EMA model.
- A test must prove that one validation epoch leaves all batch-norm running tensors unchanged.

### 3.4 Batch-norm flavor is part of each graph

Preserve the existing checkpoint invariants exactly:

- YOLOX uses epsilon `1e-5`, momentum `0.1` for the currently supported official checkpoint graph.
- Ultralytics detection and segmentation graphs use epsilon `1e-3`, momentum `0.03`.
- YOLO11/YOLO26 classification graphs use the plain-PyTorch flavor, epsilon `1e-5`, momentum `0.1`.

Do not introduce a trainer-wide batch-norm override. Scratch initialization, fine-tuning, EMA,
resume, and export must all retain the model's declared `BnFlavor`. If a historical training recipe
disagrees with a released checkpoint's live attributes, the selected recipe must be explicit in the
run metadata and parity-tested; silently mutating a loaded graph is forbidden.

### 3.5 Training checkpoints and inference artifacts have different responsibilities

The existing `.bpk` artifacts use f16 and intentionally omit training-only branches. A native
training checkpoint must instead contain fp32 model parameters and buffers, optimizer moments,
scheduler/loss schedule state, EMA, progress counters, random state, class metadata, and a resolved
configuration. Export is a separate, one-way operation from a training checkpoint to an inference
artifact.

## 4. Model and loss support matrix

| Family/task | Levels | Training prediction | Assignment | Losses | Missing graph work |
|---|---:|---|---|---|---|
| YOLOX detect | P3/P4/P5 | box deltas, objectness logits, class logits | SimOTA dynamic-k | IoU-squared, objectness BCE, class BCE, optional late L1 | raw train forward |
| YOLOv3-Tiny-U detect | P4/P5 | 16-bin DFL sides, class logits | TAL top-k 10 | CIoU, BCE, DFL | model-level train forward |
| YOLOv10 detect | P3/P4/P5 | one-to-many + detached-feature one-to-one, reg-max 16 | TAL top-k 10 + top-k 1 | CIoU, BCE, DFL for both branches | complete one-to-many head and checkpoint loading |
| YOLO11 detect | P3/P4/P5 | 16-bin DFL sides, class logits | TAL top-k 10 | CIoU, BCE, DFL | model-level train forward |
| YOLO11 segment | P3/P4/P5 + stride-4 proto | YOLO11 raw detect + coefficients + proto | TAL top-k 10 | detect terms + cropped prototype-mask BCE | raw segment forward |
| YOLO11/26 classify | global pooled | class logits | direct labels | mean softmax cross-entropy | configurable class head; train transform |
| YOLO26 detect | P3/P4/P5 | one-to-many + detached-feature one-to-one, direct sides | TAL top-k 10 + top-k 7/secondary 1 | CIoU, BCE, normalized side L1 | complete one-to-many head |
| YOLO26 segment | P3/P4/P5 + stride-4 proto | YOLO26 dual detect + dual coefficients + proto + semantic logits | YOLO26 dual TAL | detect + instance mask + semantic BCE/Dice | one-to-many mask towers and semantic tower |

All modern detector losses are objectness-free. Do not add a YOLOX-style objectness channel to any
Ultralytics head.

## 5. Proposed module layout

Add training code below a feature-gated `src/training/` tree and keep model graph code in its current
family directories:

```text
src/
  training/
    mod.rs                    public Trainer/TrainingConfig entry points
    config.rs                 resolved, serializable run configuration and validation
    dispatch.rs               ModelId -> concrete generic training run
    engine.rs                 explicit single-GPU epoch/step loop
    state.rs                  counters, RNG state, early-stop state, loss schedule state
    checkpoint.rs             atomic full-state save/load and inference export handoff
    ema.rs                    model exponential moving average
    scheduler.rs              warmup + cosine/linear policies
    optimizer.rs              parameter classification and optimizer construction
    data/
      mod.rs
      manifest.rs             Ultralytics-style dataset YAML and canonical resolved paths
      sample.rs               host-side canonical samples and annotations
      yolo.rs                 YOLO txt and polygon parser
      coco.rs                 COCO JSON, polygons, compressed/uncompressed RLE
      classification.rs       class-folder datasets
      transforms.rs           joint deterministic image/target transforms
      mosaic.rs
      masks.rs                polygon/RLE rasterization and mask downsampling
      loader.rs               seeded sampler, workers, prefetch, collation
      batch.rs                task-specific device batch types
    geometry/
      boxes.rs                differentiable and host box conversion helpers
      anchors.rs              shared level metadata, anchor points, strides
      iou.rs                  IoU/GIoU/CIoU and pairwise overlap
    assign/
      mod.rs
      simota.rs               YOLOX assignment
      tal.rs                  TaskAlignedAssigner and candidate conflict resolution
    loss/
      mod.rs
      common.rs               stable BCE-with-logits, CE, reductions, finite checks
      yolox.rs
      ultralytics_detect.rs
      segmentation.rs
      classification.rs
    metrics/
      mod.rs
      confusion.rs
      detection.rs            COCO-style box AP
      segmentation.rs         box + mask AP
      classification.rs       top-1/top-5
    report.rs                 JSONL/CSV console summaries and run metadata
```

Development-only reference exporters belong under `tools/`:

```text
tools/
  export_yolox_training_fixtures.py
  export_ultralytics_training_fixtures.py
  compare_training_checkpoint.py
```

Integration tests belong under `tests/training/` or a single `tests/training.rs` with submodules if
Cargo test binary count becomes expensive. Generated fixture tensors, datasets, and checkpoints
remain under `target/`.

## 6. Core public contracts

### 6.1 Separate catalog identity from trainable model specification

`ModelId` identifies architecture, scale, and task. It must not imply that every trained artifact
has COCO-80 or ImageNet-1k outputs. Introduce a serializable specification conceptually equivalent
to:

```rust
pub struct ModelSpec {
    pub architecture: ModelId,
    pub task: TaskKind,
    pub num_classes: usize,
    pub class_names: Vec<String>,
    pub input_size: [usize; 2],
}

pub enum TaskKind {
    Detect,
    Segment,
    Classify,
}
```

Validation rules:

- `num_classes > 0`.
- `class_names.len() == num_classes`.
- Names are non-empty after trimming and unique byte-for-byte.
- Detect/segment input dimensions are positive multiples of the maximum model stride (32 today).
- Classification uses square 224 by default but may permit a configured square size after parity at
  224 is complete.
- A task suffix and `TaskKind` must agree; `yolo26n-seg` cannot be trained as classification.

Existing pretrained constructors continue to produce the fixed official specifications. A trained
checkpoint carries its own `ModelSpec`; loading it must not fall back to a hard-coded COCO name
table.

### 6.2 Canonical host-side annotations

Use one unambiguous representation before batching:

```rust
pub struct DetectionTarget {
    pub class_id: usize,
    pub box_xyxy: [f32; 4],       // continuous edges in current image pixels
    pub segmentation: Option<SegmentationSource>,
    pub crowd: bool,
    pub source_annotation_id: Option<u64>,
}

pub struct VisionSample {
    pub image: DynamicImage,
    pub targets: Vec<DetectionTarget>,
    pub image_id: String,
    pub source_size: [u32; 2],
}
```

The canonical host contract uses continuous, unnormalized `XYXY` box edges, matching the public
inference invariant. Dataset readers convert exactly once into this form. Joint transforms mutate
the image, boxes, and segmentation together. Loss adapters may derive normalized `XYWH`, anchor
distances, or YOLOX center-size values, but those are not persisted as the canonical annotation.

For each transform:

- Clip boxes to `[0, width] x [0, height]` after geometry.
- Drop a target only through a documented visibility/area rule.
- Preserve the mapping between a target, its class, and its mask.
- Reject NaN, infinity, negative dimensions, invalid class IDs, and malformed polygons with an
  error containing dataset path, image ID, and annotation ID.
- Allow images with zero objects.

### 6.3 Device batches

Do not use a single ragged tensor with sentinel class IDs inside loss code. Collation should emit
padded tensors and an explicit valid mask:

```rust
pub struct DetectionBatch<B: Backend> {
    pub images: Tensor<B, 4>,          // [B, 3, H, W], f32
    pub classes: Tensor<B, 2, Int>,    // [B, M]
    pub boxes_xyxy: Tensor<B, 3>,      // [B, M, 4], canvas pixels
    pub valid: Tensor<B, 2, Bool>,     // [B, M]
    pub metadata: Vec<ImageMeta>,      // host-only validation/debug mapping
}

pub struct SegmentationBatch<B: Backend> {
    pub detection: DetectionBatch<B>,
    pub masks: Tensor<B, 4>,           // [B, M, Hm, Wm], 0/1
    pub semantic_class_map: Tensor<B, 3, Int>, // [B, Hm, Wm], YOLO26 only
}

pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,          // [B, 3, H, W]
    pub classes: Tensor<B, 1, Int>,    // [B]
    pub metadata: Vec<ImageMeta>,
}
```

`M` is the maximum object count in that batch, not a global dataset maximum. Empty batches use
`M = 0` only if Burn operations support the required zero dimension on WGPU; otherwise collate to
`M = 1` with `valid=false` and test that no fake target enters assignment.

Instance masks should initially be explicit per-instance binary masks at `mask_ratio=4`. This costs
more memory than overlap-index encoding but removes ordering ambiguity and makes target gathering
straightforward. An overlap-mask optimization may be added after numerical parity. The YOLO26
semantic class map must reproduce Ultralytics overlap resolution: in overlap pixels choose the
smallest-area covering instance's class; background is represented by all-zero one-hot targets and
must be gated by a foreground-coverage mask so class zero remains a valid object class.

### 6.4 Trainable model interface

Avoid a boxed trait object in the inner loop because Burn optimizers and modules are generic over the
concrete model type. Dispatch once at startup, then run a monomorphized loop. A conceptual interface
is:

```rust
pub trait TrainableTask<B: AutodiffBackend>: AutodiffModule<B> {
    type Batch;
    type TrainOutput;
    type LossState;
    type MetricsInput;

    fn forward_train(&self, images: Tensor<B, 4>) -> Self::TrainOutput;
    fn loss(
        &self,
        output: Self::TrainOutput,
        batch: &Self::Batch,
        state: &Self::LossState,
    ) -> LossOutput<B>;
}
```

In practice, keep model graph methods in the model modules and put assignment/loss adapters in
`training`. Use macros only for scale variants with identical behavior. Do not hide family-specific
loss behavior behind runtime `if family == ...` branches inside tensor kernels.

`LossOutput` should contain:

- A scalar differentiable `total` tensor.
- Named detached scalar components for logging.
- Counts such as targets, foreground anchors, and positive/GT ratio.
- A finite/non-finite status.
- Optional assignment summaries used only when debug tracing is enabled.

## 7. Model graph changes required for training

All graph changes must preserve existing inference method names and outputs. Add explicit
`forward_train`/`forward_raw` paths; do not make inference behavior depend on mutable train/eval
flags outside Burn's backend semantics.

### 7.1 Make class count configurable

Replace private family constants that determine output channel count with config fields. Keep public
official defaults at 80 or 1000. Every head branch must store `num_classes` if reshape logic needs it.

Required tests per family:

- Default configs have exactly the current parameter keys and tensor shapes.
- A three-class head emits three class channels.
- Loading an 80-class pretrained model into an 80-class graph is strict.
- Fine-tuning an 80-class checkpoint into a different class count loads the body and compatible head
  tensors, explicitly skips final class projections, and reports every skipped key.
- No shape mismatch is silently ignored.

Head replacement policy:

- `--weights` plus equal class count: strict full load.
- `--weights` plus different class count: strict partial load with only documented class-output
  projections reinitialized. Segmentation semantic output is also reinitialized. Box towers, mask
  coefficients, prototypes, body, and neck load normally when shapes match.
- `--resume`: class count and names must exactly match checkpoint metadata; partial loading is
  forbidden.
- `--scratch`: initialize all modules with the family recipe and initialize detector biases only
  after strides and class count are known.

### 7.2 YOLOX

Add a raw branch output per level containing:

- `regression`: `[B, 4, H, W]`, raw center offsets and log-width/log-height.
- `objectness_logits`: `[B, 1, H, W]`.
- `class_logits`: `[B, C, H, W]`.
- Static level stride 8, 16, or 32.

`forward_train` must concatenate flattened raw logits while also returning differentiably decoded
`XYWH` boxes for the IoU loss. It must not sigmoid objectness or class logits before BCE. The current
inference path should be rebuilt as a thin decode/sigmoid projection over the same branch computation
to prevent train/inference tower drift.

Implement optional original-space L1 targets for the final no-augmentation phase. The loss-state
flag, not the module structure, determines whether L1 contributes.

### 7.3 YOLOv3-Tiny-U and YOLO11 detect

Promote their existing `RawPredictions` path to the model-level API. Also return level shapes or a
shared `FeatureLevelLayout` so assignment can build P4/P5 or P3/P4/P5 anchors without inspecting
private feature tensors.

Keep raw box layout `[B, 4 * reg_max, A]` and raw class layout `[B, C, A]` at the graph boundary.
Decode inside the loss module for assignment and CIoU. Keep the existing inference decode as a
shared helper and assert parity between train-loss decode and inference decode before sigmoid.

YOLOv3-Tiny-U has only strides 16 and 32; no shared helper may assume three levels or stride 8.

### 7.4 YOLOv10 dual head

Reintroduce the checkpoint-compatible one-to-many `cv2`/`cv3` towers alongside the existing
`one2one_cv2`/`one2one_cv3` towers. Field/key names must preserve both official branches in full
training checkpoints. Inference Burnpacks continue to keep only one-to-one weights.

Training forward:

1. Run the body once to produce P3/P4/P5.
2. Feed normal features to one-to-many towers.
3. Detach each feature tensor and feed detached features to one-to-one towers. This is load-bearing:
   the one-to-one loss updates its head but must not backpropagate into the body.
4. Return raw boxes, scores, feature layouts, and branch identity for both branches.

Use the historical YOLOv10 criterion associated with the official checkpoint family:

- One-to-many: TAL top-k 10.
- One-to-one: TAL top-k 1.
- Sum both total losses with equal coefficient.

Before implementation, generate a fixture from the checkpoint's actual training-compatible class
and verify whether any released scale carries head attribute drift. The checkpoint, not only current
8.4.117 source aliases, determines the compatibility recipe.

### 7.5 YOLO26 dual head

Add one-to-many box/class towers matching the current `Detect` head's `cv2`/`cv3` graph. Preserve the
existing direct-distance one-to-one towers. The forward detach rule is the same as YOLOv10.

YOLO26 has `reg_max=1`; do not instantiate a fake DFL projection. Both branches emit four direct
LTRB side distances. The third detection loss component is normalized L1 side-distance loss and
should be named `l1_loss` in reporting even if the compatible hyperparameter field remains `dfl`.

Implement the pinned 8.4.117 `E2ELoss` schedule:

- One-to-many assigner: top-k 10.
- One-to-one assigner: top-k 7, secondary top-k 1.
- Initial one-to-many weight `0.8`, one-to-one weight `0.2`.
- One-to-many weight decays linearly by epoch toward `0.1`; one-to-one is `1.0 - o2m`.
- Update the schedule once after each completed train epoch.
- Persist the update count and current weights; resume must restore them exactly.
- Report the official one-to-one detached component dictionary plus explicit weighted/unweighted
  branch totals for debugging.

### 7.6 YOLO11 segmentation

Add `forward_train` returning:

- Raw classic detection boxes and class logits.
- Raw mask coefficients `[B, A, 32]` or one documented transpose of it.
- Prototypes `[B, 32, Hp, Wp]`.
- Feature-level layout.

The detection branch remains one-to-many/classic. Positive assignments from detection are reused to
select mask coefficients and the matched GT mask. Do not independently assign masks.

### 7.7 YOLO26 segmentation

Complete all training-only graph pieces:

- One-to-many detect box/class towers.
- One-to-many `cv4` coefficient towers.
- Existing one-to-one coefficient towers.
- `Proto26.semseg`: two full 3x3 Conv blocks and a biased 1x1 projection to `num_classes`.

Training forward behavior must match the reference:

- Compute fused P3/P4/P5 features and prototypes once for the one-to-many path.
- Return `(prototypes, semantic_logits)` to one-to-many segmentation loss.
- Detach prototypes and semantic logits for the one-to-one loss.
- Feed detached P3/P4/P5 features to all one-to-one box/class/mask coefficient towers.
- Keep inference export capable of removing `cv2`, `cv3`, `cv4`, and `semseg` one-to-many-only
  components without changing one-to-one parameter keys.

### 7.8 Classification

Expose a logits-only training method, or have loss read `ClassificationOutput.logits`. Never compute
cross-entropy from `probs`.

Generalize the final linear layer from 1000 to `num_classes`. Preserve the current 1280-channel head
projection and global average pool. The official dropout probability is effectively inert in the
current graph; if configurable dropout is added, default it to the pinned official value and add an
explicit stochastic train/deterministic validation test.

YOLO11 classification reuses the YOLO26 classification graph. Keep one shared implementation and
different model identity/artifact metadata.

## 8. Assignment and loss specifications

Every assignment routine runs without gradient tracking. Predicted boxes and class scores used to
choose matches are detached. Loss computation then gathers from the original differentiable
predictions using the selected indices.

### 8.1 Numerically stable primitives

Implement and unit-test:

- BCE with logits as `max(x, 0) - x*y + log1p(exp(-abs(x)))`; never sigmoid then log.
- Log-softmax cross-entropy with stable max subtraction.
- Pairwise `XYXY` IoU with non-negative intersection widths/heights and epsilon denominators.
- CIoU exactly matching the pinned Ultralytics helper, including aspect-ratio and alpha terms.
- YOLOX center-size IoU and `1 - IoU^2` loss.
- `bbox2dist`, `dist2bbox`, `xyxy <-> xywh`, clipping, and anchor generation.
- DFL target clamping to `reg_max - 1 - 0.01`, left/right bins, and linearly weighted negative
  log-softmax.
- Empty reductions that return differentiable zero instead of NaN.

Run loss accumulation in fp32 even if later mixed-precision model execution is enabled.

### 8.2 YOLOX SimOTA

Port the selected official source literally before refactoring. Per image:

1. Decode all anchors to canvas-pixel `XYWH` boxes.
2. Candidate anchors are inside at least one GT box or inside a center-radius region of `2.5 *
   stride`.
3. Retain a second mask marking candidates inside both the GT box and center region.
4. Compute pairwise IoU between each GT and candidate prediction.
5. Pairwise IoU cost is `-ln(iou + 1e-8)` multiplied by `3.0`.
6. Pairwise class cost is BCE between one-hot GT classes and
   `sqrt(sigmoid(class_logits) * sigmoid(objectness_logits))`, summed over classes.
7. Add penalty `100000.0` where the candidate is not in both box and center.
8. For each GT, dynamic `k` is the integer-clamped-at-least-one sum of its top `min(10,
   candidates)` IoUs.
9. Select the lowest-cost `k` anchors for each GT.
10. If an anchor matches multiple GTs, retain only the lowest-cost GT.
11. Class targets are matched one-hot labels multiplied by matched IoU.
12. Objectness target is one for foreground anchors and zero for every other anchor.

Losses, each normalized by `max(total_foreground, 1)`:

- `iou = sum(1 - iou(pred_xywh, target_xywh)^2)` on positives.
- `obj = BCEWithLogits(objectness_logits, foreground_mask)` on all anchors.
- `cls = BCEWithLogits(class_logits, iou_weighted_one_hot)` on positives.
- `l1 = L1(raw_box_deltas, encoded_gt)` on positives only when enabled.
- `total = 5.0 * iou + obj + cls + l1`.

Return `foreground / max(gt_count, 1)` as a diagnostic. Test the no-GT path separately: box/class/L1
are zero, objectness still trains all anchors toward zero, and the total is finite.

### 8.3 TaskAlignedAssigner (TAL)

Port the pinned Ultralytics assigner with these defaults:

- Alignment metric: `class_score^0.5 * overlap^6.0`.
- Candidate anchor center must be inside the GT box.
- Select per-GT top-k alignment candidates; k depends on branch as listed in the support matrix.
- Apply primary and secondary top-k behavior exactly for YOLO26 one-to-one.
- Resolve multi-GT anchors by highest overlap using the pinned conflict path.
- Build one-hot class scores for selected GTs.
- Normalize target scores by each GT's maximum alignment metric and maximum overlap.

The implementation must support both two-level and three-level heads and arbitrary class count. Do
not copy tensors to CPU on assignment OOM in the first version. Instead fail with a diagnostic that
reports batch size, image size, max objects, and candidate tensor shape; automatic batch reduction
can be implemented at the epoch boundary later.

### 8.4 Ultralytics detection loss

For each raw branch:

1. Transpose predictions to `[B, A, 4*reg_max]` and `[B, A, C]`.
2. Build 0.5-offset anchor points and stride tensors from the actual level shapes.
3. Decode DFL distributions or direct distances to anchor-space `XYXY`.
4. Assign using detached sigmoid scores and detached decoded pixel boxes.
5. Let `target_scores_sum = max(sum(target_scores), 1)`.
6. Classification is unreduced BCE with logits over all anchors/classes, summed and divided by
   `target_scores_sum`.
7. Positive box loss is `(1 - CIoU)`, weighted by each positive's summed target score, summed and
   divided by `target_scores_sum`.
8. For `reg_max=16`, DFL is weighted left/right-bin CE on four sides, then target-score weighted and
   normalized.
9. For `reg_max=1`, compute target and predicted LTRB in pixels, normalize x sides by image width and
   y sides by image height, take mean absolute error over four sides, target-score weight, and
   normalize.
10. Apply configured gains. The official baseline fields are `box`, `cls`, and `dfl`; preserve the
    resolved values in checkpoint metadata rather than relying on future defaults.
11. Match the reference's batch-size scaling before backward. Add a fixture that detects an
    accidental mean-vs-sum difference when batch size changes.

### 8.5 Instance segmentation loss

Reuse detection assignments. For each positive anchor:

1. Gather its matched GT mask and its 32 raw coefficients.
2. Form mask logits by matrix multiplication: `coefficients @ prototypes`.
3. Normalize the matched target `XYXY` box to `[0,1]` and scale it to prototype coordinates.
4. Compute per-pixel BCE with logits against the binary GT mask.
5. Crop the loss map to the matched target box.
6. Mean over prototype height/width, divide by normalized GT box area, sum positives.
7. Divide the batch result by total positive anchors.
8. Multiply by the configured box/seg gain exactly as the reference.

If a batch has no positives, connect a zero-valued sum of coefficients and prototypes to the loss so
all trainable branches participate in autodiff without NaN or missing-gradient behavior.

For YOLO26 semantic logits:

- Build a per-class binary target `[B,C,Hm,Wm]` from the semantic class map.
- Zero all classes where no instance is present.
- Compute `0.5 * BCE + 0.5 * Dice` according to pinned `BCEDiceLoss` reductions.
- Scale by the same configured gain as the reference.
- Include a connected zero for no-positive batches.
- Apply this term to the one-to-many path; the detached one-to-one proto tuple must not update the
  prototype/semantic tower.

### 8.6 Classification loss

Use mean cross-entropy over logits and integer class IDs. Report loss, top-1 correct/count, and top-5
correct/count. Test class counts below five by using `min(5, num_classes)` while labeling the metric
clearly.

## 9. Data system

### 9.1 Dataset manifest

Support Ultralytics-compatible dataset YAML as the user-facing format:

```yaml
path: datasets/example
train: images/train
val: images/val
test: images/test
names:
  0: cat
  1: dog
```

Also accept `names: [cat, dog]`. Resolve relative split paths against `path`, then against the YAML
file's directory. Canonicalize for diagnostics but do not require all images to share one root.

Add an explicit `format` override (`yolo`, `coco`, `classification-folders`) and deterministic
auto-detection with a printed result. Ambiguous layouts are errors. Persist the resolved split file
list and a dataset fingerprint in the run directory.

For classification, support conventional `train/<class>/...` and `val/<class>/...` folders. Class
order must come from the manifest when provided; otherwise sort directory names lexicographically
and persist the resulting mapping.

### 9.2 YOLO labels

Detection lines: `class cx cy width height`, normalized to source image dimensions. Segmentation
lines: class followed by normalized polygon points. Requirements:

- Ignore blank lines only; malformed nonblank lines are errors.
- Require integral class IDs in range.
- Require finite normalized coordinates. Permit a small configurable clipping tolerance for common
  serialization noise, but report clipped counts.
- Reject boxes with non-positive width/height after conversion.
- A missing label file means a valid background image; an unreadable existing file is an error.
- Multiple polygons belonging to one COCO instance are representable internally even if the simple
  YOLO polygon format normally supplies one ring.

### 9.3 COCO JSON

Support image records, categories, annotations, `bbox`, `iscrowd`, polygon segmentation, compressed
RLE, and uncompressed RLE. Build a contiguous training class index independent of sparse COCO
category IDs and persist both mappings.

Initial crowd policy should match the selected reference recipe: crowd/ignore regions do not become
positive targets, and metrics honor their ignore semantics. Record dropped invalid annotations by
reason. Never silently treat an RLE decode failure as an empty mask.

### 9.4 Transform algebra

Each geometric transform returns an affine/projective mapping and enough metadata to replay or
debug it. The training canvas contract differs by recipe:

- Ultralytics detect/segment: RGB, stride-aligned rectangular or square letterbox, values `[0,1]`.
- YOLOX: preserve its top-left canvas and raw-pixel convention as required by the selected training
  reference; do not reuse Ultralytics normalization accidentally.
- Classification: anti-aliased shortest-edge resize and crop for validation; random resized crop and
  flip for training once fixture parity is established.

Implement augmentations in this order, with an image/boxes/masks golden fixture for every step:

1. Decode and orientation handling.
2. Deterministic resize/letterbox only.
3. Horizontal flip.
4. HSV/color jitter.
5. Random affine: scale, translate, shear, perspective, border behavior.
6. Four-image mosaic.
7. Mixup.
8. Segmentation copy-paste.
9. Multi-scale batch resizing.
10. Classification random resized crop, auto-augment, and erasing if needed by the pinned recipe.

Transforms that combine samples receive child sample IDs from the sampler, not hidden global RNG
calls. That is required for exact resume and reproducible fixture replay.

### 9.5 Mask rasterization

- Apply geometry to polygon vertices before rasterization when possible.
- Define pixel-center inclusion and boundary fill rules and parity-test them against the reference.
- Decode RLE in source dimensions, then warp/downsample with nearest-neighbor for target masks.
- Use `mask_ratio=4` by default, matching stride-4 prototypes.
- After geometric clipping, recompute boxes from transformed annotations according to the reference
  and assert mask/box consistency within a documented tolerance.
- Drop empty masks and their detection targets together.

### 9.6 Seeded loader

Use a stable RNG such as ChaCha with state that can be serialized. Derive independent streams for:

- Epoch permutation.
- Worker/sample transforms.
- Mosaic partner selection.
- Mixup/copy-paste selection.
- Model stochastic layers.

The seed input should include global seed, epoch, sample identity, augmentation stage, and draw index.
Do not derive randomness from worker scheduling. This permits changing worker count without changing
sample content, where practical.

The loader should:

- Decode/augment on CPU workers.
- Keep device transfer in the training thread.
- Use bounded prefetch to avoid unbounded RAM.
- Expose `drop_last` explicitly.
- Return structured worker errors to the main thread.
- Report decode, augmentation, collation, host wait, upload, forward, backward, and optimizer timing.

## 10. Training engine

Use a project-owned explicit loop rather than depending on `LearnerBuilder` for the first release.
Burn's module/optimizer primitives should still be used. The custom loop is needed for dual-branch
loss schedules, exact full-state resume, family-specific no-augmentation phases, EMA, parameter
groups, gradient accumulation, and task-specific validation.

### 10.1 Startup sequence

1. Parse CLI and optional config file.
2. Resolve dataset paths, task, class names, model spec, and defaults.
3. Validate feature/backend availability and select one WGPU adapter explicitly.
4. Write `config.requested.yaml` and immutable `config.resolved.json` to a newly created run
   directory.
5. Build datasets and write fingerprints/statistics before allocating the model.
6. Construct the concrete model on a large-stack worker if current Windows stack constraints still
   apply under autodiff.
7. Initialize scratch weights, load fine-tune weights, or restore full resume state.
8. Build optimizer groups and scheduler.
9. Initialize or restore EMA.
10. Run a one-batch dry validation when `--dry-run` is set: data -> forward -> loss -> backward ->
    finite gradients, without optimizer mutation or checkpoint save.
11. Enter the epoch loop.

### 10.2 Step sequence

For each microbatch:

1. Fetch and upload the task batch.
2. Run `forward_train` on `Autodiff<Wgpu>`.
3. Compute assignment and all named losses in fp32.
4. Assert scalar total and detached components are finite. Include batch image IDs in failure output.
5. Divide total by `accumulate_steps` before backward.
6. Call backward and convert gradients with `GradientsParams::from_grads` for the concrete model.
7. Merge gradient sets into an accumulation buffer without moving parameters.
8. On accumulation boundary or final partial group:
   - unscale if AMP is active;
   - compute global gradient norm;
   - clip to configured maximum, official baseline 10 where applicable;
   - reject/handle non-finite gradients according to policy;
   - apply one optimizer step at the scheduler's current learning rate(s);
   - clear accumulated gradients;
   - update EMA once;
   - increment `optimizer_step` and scheduler state.
9. Increment `micro_step`, update streaming metrics, and append a JSONL step event at the configured
   interval.

Define the final partial-accumulation normalization explicitly. Recommended behavior is to divide by
the actual number of microbatches in that final group, not always the configured accumulation count.
Parity tests must cover it.

### 10.3 Optimizer parameter groups

Reference recipes apply weight decay selectively. Classify parameters by module/parameter identity,
not by fragile substring alone:

- Decay: convolution and linear weights.
- No decay: batch-norm gamma/beta and other normalization scale parameters.
- No decay: biases.

Persist a sorted manifest of parameter key -> group, shape, element count, trainable/frozen status.
Assert each trainable parameter appears in exactly one group. Compare counts to a Python oracle.

The first implementations should support:

- SGD with momentum and Nesterov for YOLOX parity.
- AdamW and SGD for Ultralytics-style runs.
- Explicit optimizer selection. `auto` may be added only after its class-count/iteration heuristic is
  pinned and tested.

Burn's optimizer adaptor applies one configuration broadly; if it cannot express per-parameter
decay groups in 0.21.0-pre.4, implement a small project optimizer wrapper/mapper or maintain separate
gradient/optimizer adaptors by group. Do not approximate by decaying BN and bias parameters.

### 10.4 Learning-rate schedules

Implement a serializable scheduler driven by optimizer steps, with epoch context:

- Linear warmup over resolved warmup steps.
- Warmup momentum interpolation where the reference uses it.
- Separate warmup bias LR where required.
- Cosine decay and linear decay.
- YOLOX warm-cos schedule, including five warmup epochs, minimum LR ratio, and the final
  no-augmentation epoch range in the compatibility preset.
- Ultralytics nominal-batch-size scaling: resolved weight decay and accumulation must be recorded.

Log the LR before every optimizer step. Scheduler state must resume without an off-by-one step.
Create tests at step 0, first warmup step, warmup boundary, mid-run, final regular step, resumed next
step, and final step.

### 10.5 EMA

Maintain a non-autodiff fp32 EMA copy of all parameters and running buffers used for validation and
export. Match the selected reference decay schedule, including update-count warmup. Requirements:

- Update once per optimizer step, not per microbatch.
- Do not backpropagate through EMA.
- Decide and parity-test how integer/non-floating buffers are copied.
- Persist EMA model and update count.
- Validate EMA by default and optionally raw model for diagnostics.
- Detect non-finite EMA tensors before save; fail with parameter names rather than silently
  sanitizing the training state.

### 10.6 Freeze and fine-tune

Support `freeze=none`, body stage prefixes, backbone-only, and explicit parameter-key patterns. Print
and persist the exact resolved key list. Frozen batch norm behavior is a separate option:

- Frozen parameters with live running statistics.
- Fully frozen BN using inference statistics.

Do not conflate the two. Fine-tuning presets must state which behavior they choose.

### 10.7 No-augmentation transitions

YOLOX compatibility requires a final no-mosaic/no-mixup phase and enables L1 loss there. Modern
Ultralytics presets close mosaic for their configured final epochs. At transition:

- Change only epoch-scoped transform policy; do not rebuild the dataset with a new sample order.
- Persist the phase in checkpoint state.
- For YOLOX, enable L1 exactly at the transition.
- Optionally increase validation frequency according to the reference preset.
- Add a resume test on the epoch immediately before and after the transition.

### 10.8 Failure handling

- Non-finite loss: stop before optimizer/EMA mutation, save a diagnostic bundle containing resolved
  config, batch metadata, target tensors, named outputs or compact statistics, and RNG state.
- GPU OOM: report adapter, allocated shapes, batch size, image size, max objects, and accumulation.
  The first release may abort cleanly; later `auto_batch` may restart the epoch from its checkpoint.
- Ctrl-C: finish no further optimizer step, atomically save an interrupt checkpoint if state is
  coherent, and return a distinct exit code.
- Worker error: stop and surface the original path/image/annotation context.
- Disk-full/checkpoint error: retain the previous valid checkpoint and fail loudly.

## 11. Checkpoint and artifact design

### 11.1 Run directory

Use a collision-safe structure such as:

```text
runs/<task>/<name>-<timestamp>-<short-id>/
  config.requested.yaml
  config.resolved.json
  dataset.json
  parameter-groups.json
  events.jsonl
  metrics.csv
  checkpoints/
    last/
    best/
    epoch-0009/
  exports/
```

Do not overwrite an existing run unless `--resume` points to it. A user-supplied name is a label, not
authorization to delete or replace a directory.

### 11.2 Full checkpoint contents

Store a versioned manifest and separate records if one monolithic recorder is impractical:

- Format name and version.
- Crate version, git commit, dirty flag, Burn version, backend, adapter name, OS, and architecture.
- Complete resolved config and `ModelSpec`.
- Dataset fingerprints and class/category mapping.
- Raw fp32 train model record, including all training-only branches.
- Raw fp32 EMA model record and EMA update count.
- Optimizer type, hyperparameters, group manifest, and all moments/velocity state.
- Scheduler and warmup state.
- Epoch, next batch index, micro-step, optimizer-step, accumulation position.
- Dual-head loss schedule state and YOLOX L1/no-augmentation phase.
- Best metric, best epoch, patience counter.
- RNG states and epoch permutation/partner-selection state.
- Streaming metric state only if mid-epoch resume is promised.

Prefer epoch-boundary exact resume first. Mid-epoch resume can be supported once loader state and
partially accumulated gradients are serializable. Until then, an interrupt checkpoint should mark
that the current epoch restarts and should preserve the last completed optimizer state; never claim
exact mid-epoch continuation if partial gradients were dropped.

### 11.3 Atomic save

Write a new checkpoint to a sibling temporary directory, flush/close all files, compute hashes,
write the manifest last, then atomically rename into place. Update `last` only after the new
checkpoint validates. Keep at least one previous checkpoint until replacement succeeds.

On load:

- Validate manifest version and hashes.
- Check every model/optimizer key and shape.
- Reject unknown required fields; tolerate only explicitly optional forward-compatible fields.
- Verify architecture, task, class table, optimizer type, and parameter-group manifest.
- Print the exact next epoch/step and LR before proceeding.

### 11.4 Inference export

Add an explicit export operation from `best` or `last`:

1. Select EMA by default.
2. Convert to the non-autodiff valid model.
3. Strip YOLOv10/YOLO26 one-to-many towers and YOLO26 semantic tower.
4. Preserve only the task's inference graph.
5. Convert to f16 through the existing `HalfPrecisionAdapter` if requested.
6. Embed architecture, task, input size, class names, training checkpoint hash, artifact format,
   precision, license/provenance, and metrics.
7. Reload the produced artifact through the public predictor.
8. Run a fixed export smoke image and compare fp32 EMA vs exported predictions within task-specific
   f16 tolerances.

Custom-class inference requires removing the current assumption that class IDs always index the
static COCO/ImageNet tables. The predictor created from a trained artifact must use embedded class
names. Existing official `Predictor::new(ModelId, ...)` remains unchanged.

## 12. CLI and configuration

Add feature-gated commands conceptually like:

```console
cargo run --release --features training -- train \
  --model yolo11n \
  --data datasets/example.yaml \
  --epochs 100 \
  --batch 8 \
  --device gpu

cargo run --release --features training -- val \
  --checkpoint runs/detect/example/checkpoints/best

cargo run --release --features training -- export \
  --checkpoint runs/detect/example/checkpoints/best \
  --output target/example-yolo11n.bpk
```

The trainer should accept a YAML configuration plus CLI overrides. Precedence is:

1. Built-in family/task compatibility preset.
2. Config file.
3. Explicit CLI flags.
4. Resume checkpoint for all immutable fields. Conflicting CLI values on resume are errors except
   for a documented small set such as additional epochs, device, workers, and logging interval.

Minimum `train` options:

- Model, task-derived automatically, data manifest, scratch/weights/resume (mutually constrained).
- Epochs, batch size, accumulation, image size, workers, prefetch, seed, deterministic mode.
- Optimizer, initial/final LR, momentum/betas, weight decay, warmup, scheduler, gradient clip.
- Augmentation probabilities and close-mosaic/no-augmentation epochs.
- Device/adapter selector, fp32/AMP mode, cache policy.
- Validation interval, confidence/IoU/max detections used only for validation, patience.
- Run root/name, save interval, retention, dry-run, trace-batch.
- Freeze policy and class-head replacement policy.

Print a compact resolved summary before training: model/task/classes, parameter counts, trainable
counts, dataset sizes, image/batch/accumulation/effective batch, optimizer groups, LR schedule,
augmentation preset, adapter, checkpoint source, and output directory.

## 13. Validation and metrics

Validation must use the EMA `valid()` model, deterministic transforms, and no training augmentation.
It may batch model forwards internally even though the public predictor remains batch-1.

### 13.1 Detection

- Decode using the same model-family path as inference.
- YOLOX, YOLOv3-Tiny-U, and YOLO11 use class-aware NMS.
- YOLOv10 and YOLO26 use their end-to-end top-k path without NMS.
- Apply configurable `max_detections`, with 300 as the Ultralytics-compatible default.
- Map predictions through validation geometry back to source-image continuous `XYXY`.
- Accumulate per-class precision/recall, AP50, and AP averaged over IoU 0.50:0.05:0.95.
- Honor crowd/ignore annotations in COCO evaluation.

Do not use the training confidence threshold to drop low-score predictions before AP calculation;
use a low evaluator threshold and cap candidates consistently.

### 13.2 Instance segmentation

Compute box metrics and mask metrics independently. Assemble masks using the same coefficients,
prototypes, crop, and source mapping as public inference. COCO mask IoU must operate on source-image
coverage/RLE, not prototype-grid masks.

### 13.3 Classification

Use the deterministic classify validation transform and report mean loss, top-1, top-5, per-class
support, and optionally a confusion matrix. Selection metric defaults to top-1 accuracy.

### 13.4 Best checkpoint and early stopping

Default fitness:

- Detect: box mAP50:95.
- Segment: weighted or explicit tuple of box and mask mAP50:95; choose and record one formula before
  implementation.
- Classify: top-1 accuracy.

Save `best` only on a strictly improved finite fitness. Persist patience state. A resumed run must not
reset best fitness or patience unless the user starts a new fine-tune run rather than resume.

## 14. Parity fixture strategy

Python is an oracle only in ignored/development tests. Fixture generation must pin package/source
version, model, seed, device, dtype, input tensor, targets, and hyperparameters.

### 14.1 Fixture schema

Use a versioned manifest with binary tensor payloads rather than huge JSON arrays. Include:

- Tensor name, dtype, shape, SHA-256, and binary file.
- Model/task/scale/class count and reference commit/version.
- Input images after preprocessing.
- Canonical and reference-form targets.
- Raw per-level outputs for all branches.
- Anchor points and strides.
- Candidate masks, pairwise overlaps/costs, selected positives, matched GT indices, target scores.
- Every unweighted and weighted loss component plus total.
- Gradients for a curated set of body, BN, box, class, mask/proto, semantic, and classifier params.
- Parameter tensors after one optimizer step.
- BN running state after the step.
- Optimizer moment/velocity samples, LR, EMA samples, and counters.

Keep small deterministic tensors for exact assigner unit tests and one real-model fixture for
integration. Statistics-only fixtures are insufficient for assignment because one changed anchor can
be hidden by means/RMS values.

### 14.2 Required reference cases

For each distinct loss family, generate:

- One image with one centered object.
- Multiple classes and differently sized objects.
- Two GTs competing for one anchor.
- Object touching each image boundary.
- Tiny object.
- Crowded/high-object-count image.
- Empty image and an all-empty batch.
- Mixed empty/non-empty batch.
- Non-square stride-aligned image.
- Class count 3 to expose hard-coded 80 assumptions.
- Batch size 2 to expose normalization mistakes.

Additional segmentation cases:

- Overlapping masks of different areas/classes.
- Polygon hole/RLE where supported.
- Mask clipped by affine/image boundary.
- Positive assignment whose downsampled mask is tiny.
- No-positive batch with finite connected zero.

### 14.3 Tolerance policy

- Assignment indices and boolean masks: exact.
- Shapes, counts, class IDs, parameter-group membership: exact.
- fp32 scalar/tensor parity on CPU test backend where operations exist: start at `2e-5` absolute and
  relative, relax only with measured evidence.
- WGPU forward/loss/gradient parity: define per-operation tolerance from observed adapter behavior;
  never use one broad tolerance to conceal discrete assignment changes.
- One-step parameter parity: compare named tensors and update norms.
- End-to-end quality gates are required even when tensor fixtures pass.

If near-tied floating-point values change assignment on GPU, keep a CPU deterministic assigner mode
for fixture diagnosis, document the tie, and gate quality. Do not silently sort by unstable backend
order; define a deterministic secondary key (anchor index) where the reference semantics permit it.

## 15. Implementation phases and gates

Each phase should be a reviewable change set. Do not start broad model wiring until the previous
phase's gates are green.

### Phase 0 - Burn capability spike

Tasks:

- Add the non-default training feature and compile a tiny `Autodiff<Wgpu>` convolution/BN model.
- Prove forward, scalar backward, `GradientsParams`, SGD/AdamW step, model record save/load, optimizer
  record save/load, and `model.valid()` inference.
- Verify required WGPU ops: boolean masking/gather/scatter, top-k/sort, argmax, dynamic slicing,
  interpolation, matrix multiplication, log-softmax, and gradient accumulation.
- Measure memory behavior and identify ops that force host synchronization.
- Decide whether assignment remains on GPU or uses a deliberately explicit host implementation for
  the first YOLOX slice. The final modern path should be GPU-native for performance.
- Prove parameter grouping can express selective decay; implement a wrapper if not.

Gate: a restartable one-layer training test reproduces its uninterrupted next step and validation
does not update BN state.

### Phase 1 - Data and geometry minimum slice

Tasks:

- Implement manifest, YOLO detection parser, canonical sample, deterministic letterbox/top-left
  resize, horizontal flip, padded batch, and seeded loader.
- Add visualization/debug export under `target/` for transformed image and boxes.
- Implement box conversions, anchor layouts, IoU, CIoU, stable BCE/CE.
- Add malformed/empty/boundary dataset tests.

Gate: Rust transformed tensors and targets match both YOLOX and Ultralytics reference fixtures.

### Phase 2 - YOLOX-nano vertical slice

Tasks:

- Add configurable class count and YOLOX raw training forward.
- Port SimOTA and YOLOX loss.
- Add SGD/Nesterov groups, YOLOX warm-cos schedule, EMA, fp32 checkpoint, resume, basic validation,
  and export.
- Add CLI `train`, `val`, and `export` in training builds.
- Match assignment, losses, gradients, BN state, one optimizer step, scheduler, and EMA fixtures.
- Overfit 1 image, then 8 images.

Gate: exact discrete parity, numeric one-step parity, tiny overfit, interrupted/resumed equivalence,
and exported predictor parity.

### Phase 3 - All YOLOX scales

Tasks:

- Wire tiny/s/m/l/x through the same trainer.
- Add scale-aware default image size behavior where current variants differ (nano/tiny official
  checkpoints use 416 for inference in this project).
- Generate one forward/loss fixture per scale and full one-step fixtures for nano plus one large
  scale.
- Run memory smoke tests and publish recommended single-GPU batch/accumulation values.

Gate: every YOLOX `ModelId` completes a one-batch dry-run and export reload; nano/tiny/s complete tiny
overfit, with large scales permitted to use batch 1 plus accumulation.

### Phase 4 - Shared TAL/DFL detector path

Tasks:

- Port TAL, DFL, CIoU, modern detection loss, and Ultralytics parameter initialization/bias rules.
- Expose raw model forward for YOLOv3-Tiny-U and YOLO11.
- Prove two-level YOLOv3-Tiny-U first, then three-level YOLO11n.
- Add AdamW/SGD modern presets and Ultralytics transform/augmentation minimum.

Gate: YOLOv3-Tiny-U and YOLO11n pass assigner/loss/gradient/step parity and tiny overfit.

### Phase 5 - YOLO11 detect scales

Tasks:

- Parameterize all five scales and head class count.
- Add fixtures for each scale and large-scale memory guidance.
- Validate COCO8 fine-tuning against the reference using identical seed/config/data order where
  possible.

Gate: all five scales dry-run/resume/export; n and s meet agreed COCO8 metric delta.

### Phase 6 - YOLOv10 dual-head detect

Tasks:

- Restore one-to-many graph and full branch loading.
- Implement detached-feature one-to-one forward and historical dual loss.
- Extend full checkpoint/export stripping.
- Add branch-specific gradient tests: one-to-one-only loss changes one-to-one head parameters but
  produces zero/no backbone gradient; one-to-many loss updates backbone.

Gate: both branch assignments/losses and one-step updates match reference; all six scales dry-run and
export only one-to-one inference weights.

### Phase 7 - YOLO26 dual-head detect

Tasks:

- Restore one-to-many towers.
- Implement direct-side normalized L1 path and epoch-varying `E2ELoss` weights.
- Add resume tests around loss-weight schedule updates.

Gate: branch detachment, assignments, loss schedule, one-step parity, all-scale dry-run/export, and
YOLO26n tiny overfit.

### Phase 8 - Classification

Tasks:

- Add classification-folder dataset, training transforms, configurable final linear layer,
  cross-entropy, top-k metrics, and classification best-fitness policy.
- Wire shared graph to all YOLO11/YOLO26 classification IDs.
- Test pretrained full load and changed-class head replacement.

Gate: logits/loss/gradient/step parity, tiny class-folder overfit, all ten variants dry-run/export,
and top-1/top-5 validation parity on a fixed set.

### Phase 9 - YOLO11 segmentation

Tasks:

- Add YOLO polygon and COCO mask ingestion/rasterization.
- Add raw segment forward and instance mask loss.
- Add box/mask evaluator and debug mask rendering.
- Wire n and s.

Gate: mask target/proto/coefficient/loss/gradient parity, overlap and empty-target tests, tiny-mask
overfit, and box/mask mAP smoke validation.

### Phase 10 - YOLO26 segmentation

Tasks:

- Restore dual coefficient towers and semantic tower.
- Build semantic target map and BCE/Dice term.
- Apply detached one-to-one proto semantics and dual loss schedule.
- Wire all five scales and export stripping.

Gate: instance plus semantic parity, branch-gradient isolation, all-scale dry-run/export, YOLO26n
tiny overfit, and agreed COCO8-seg metric delta.

### Phase 11 - Full augmentation and quality hardening

Tasks:

- Add mosaic, mixup, copy-paste, HSV, affine, multi-scale, classification auto-augment/erasing as
  selected by compatibility presets.
- Add no-augmentation transitions and resume gates.
- Profile loader/GPU utilization and remove avoidable host synchronizations.
- Run longer reference comparisons and document expected metric ranges.

Gate: every current model/task passes the release matrix below and documentation no longer labels
training experimental if the project chooses to graduate it.

## 16. Test and release matrix

### 16.1 Required on every ordinary change

Keep existing checks and add training-feature compilation/tests:

```console
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --no-default-features --lib
cargo test --locked --features training training
cargo clippy --locked --features training --all-targets -- -D warnings
```

Feature combinations to compile explicitly:

- Default inference.
- `--no-default-features --lib`.
- `gpu` inference.
- `training` (which should imply the required GPU/autodiff features).
- `training,pretrained` if those are not naturally the same feature closure.

### 16.2 Unit tests

- Dataset parsing, path resolution, class mapping, empty labels, malformed records.
- Every box conversion and transform boundary case.
- Polygon/RLE decode and rasterization.
- Stable BCE/CE under logits such as +/-100.
- IoU/CIoU/DFL/direct-L1.
- SimOTA and TAL exact fixtures including conflicts and empty GT.
- Scheduler boundaries and resume.
- Parameter grouping completeness/exclusivity.
- EMA update and save/load.
- Checkpoint corruption/version/shape mismatch.
- Class-head replacement allowlist.

### 16.3 Integration tests

- One real batch forward/loss/backward per distinct task/family.
- Finite gradients and nonzero gradients in expected branches.
- Deliberately isolated branch gradient tests for dual heads.
- BN running state changes in train and does not change in validation.
- Uninterrupted two epochs vs one epoch + resume + one epoch.
- Scratch and fine-tune tiny overfit.
- EMA export -> public predictor reload -> output comparison.
- Existing official checkpoint inference parity after graph additions.

### 16.4 Manual/ignored quality gates

- Reference Python fixture generation.
- COCO8 detect for YOLOX-nano, YOLOv3-Tiny-U, YOLO11n, YOLOv10n, YOLO26n.
- A tiny classification dataset for YOLO11n-cls and YOLO26n-cls.
- COCO8-seg or an equivalent small segmentation set for YOLO11n-seg and YOLO26n-seg.
- At least one larger scale per graph flavor to catch width-dependent shape/key issues.
- Full COCO/ImageNet training is a later resource gate, not required for each PR, but published
  training claims require it.

## 17. Single-GPU memory and performance policy

Correctness comes first, but the implementation must be usable on one GPU:

- Default fp32 until AMP parity is established.
- Batch 1 must work for every scale that fits one forward/backward; accumulation provides effective
  batch size.
- Log peak device memory if the backend exposes it; otherwise report tensor-shape estimates and
  adapter limits.
- Keep assignment tensors bounded by filtering candidate anchors before constructing full
  GT-by-anchor class tensors where this preserves exact semantics.
- Avoid downloading predictions to host during training except logging/debug intervals.
- Validation may stream detections/masks to host per batch to bound GPU memory.
- Segmentation should gather positive coefficients before expanding mask logits; never materialize
  `[B, all_anchors, Hproto, Wproto]`.
- Cache static anchor grids by level shape/device/dtype, but invalidate on multi-scale shape changes.
- Use bounded CPU prefetch and expose cache modes (`none`, metadata, decoded`) explicitly.

AMP is a separate gated feature:

1. Establish fp32 fixture and quality baselines.
2. Determine WGPU f16 operation support per adapter.
3. Keep assignment, reductions, loss, optimizer master weights, BN running stats, and EMA in fp32.
4. Add dynamic loss scaling with persisted scaler state.
5. Compare finite behavior, gradients, one-step updates, memory, throughput, and quality.
6. Fall back to fp32 with an explicit message when unsupported; never silently change precision.

## 18. Important edge cases and invariants

- Empty images are valid training examples.
- A whole empty batch must produce finite background classification/objectness behavior and connected
  zeros for unused segmentation branches.
- Public and canonical boxes are continuous source/current-image `XYXY`; model loss representations
  are adapters, not new public contracts.
- Boundary edges equal to width/height are valid.
- Class count is not hard-coded to 80/1000 in trainable heads, losses, metrics, or artifacts.
- Class names and order are checkpoint data, not inferred at resume.
- YOLOv3-Tiny-U uses two feature levels; shared code must not assume three.
- YOLOX has objectness; modern Ultralytics heads do not.
- YOLO11 is classic NMS; YOLOv10/YOLO26 remain NMS-free in validation/export.
- YOLOv10 uses the historical dual-loss compatibility recipe; YOLO26 uses the pinned decaying
  one-to-many/one-to-one recipe.
- YOLO26 is DFL-free. Naming a config field `dfl` for compatibility must not instantiate DFL.
- One-to-one dual-head inputs are detached from the body.
- YOLO26 one-to-one prototypes/semantic outputs are detached.
- Mask coefficients and prototype logits stay raw; no sigmoid precedes mask BCE-with-logits.
- Validation uses `valid()` and cannot update BN running statistics.
- f16 inference artifacts are not resumable training checkpoints.
- Generated checkpoints, datasets, fixtures, and debug images stay outside git, under `target/` or a
  user-selected run root.
- Training code remains feature-gated so no-default-feature inference stays lean.
- The Apache-2.0/AGPL provenance boundary remains explicit. YOLOX-derived code/weights retain their
  Apache option; Ultralytics architectures, official weights, and derivatives remain AGPL-3.0.

## 19. File-by-file implementation checklist

This is the expected impact map; exact grouping into commits may differ.

### Manifest/build

- `Cargo.toml`: non-default training feature; Burn train/autodiff/dataset features; optional config,
  RNG, and reporting dependencies.
- `Cargo.lock`: dependency lock changes.

### Existing model files

- `src/models/yolox/head.rs`, `model.rs`: configurable classes and raw train forward.
- `src/models/yolov3_tiny/head.rs`, `model.rs`: configurable classes and model raw forward.
- `src/models/yolov10/head.rs`, `model.rs`: both dual branches and full training records.
- `src/models/yolo11/head.rs`, `segment_head.rs`, `model.rs`, `classification.rs`: raw task outputs
  and configurable classes.
- `src/models/yolo26/head.rs`, `segmentation.rs`, `model.rs`, `classification.rs`: dual branches,
  semantic tower, raw outputs, configurable classes.
- Family weight loaders: strict full vs allowlisted head-replacement loading and explicit reports.

### Runtime/API

- `src/lib.rs`: trained artifact metadata/class catalog loading and any shared batch decode needed by
  validation; preserve existing APIs.
- `src/main.rs`: feature-gated `train`, `val`, and `export` subcommands.
- `src/data/`: expose or carefully share preprocessing primitives without coupling graph code to
  filesystems.

### New training tree

- All modules listed in section 5.

### Tools/tests/docs

- Python fixture exporters and comparator.
- Unit/integration/ignored tests.
- `README.md`: user workflows and support table only after the relevant phase works.
- `AGENTS.md`: training invariants, verification commands, generated-file policy.
- `ROADMAP.md`: milestone status.
- `NOTICE`: provenance for redistributed trained artifacts.

## 20. Pull-request slicing recommendation

Keep PRs narrow enough that inference reviewers can verify graph preservation:

1. Burn autodiff/checkpoint capability spike and feature wiring.
2. Canonical data/geometry primitives and dataset minimum.
3. YOLOX raw forward only, with inference parity unchanged.
4. SimOTA/loss fixtures.
5. YOLOX trainer/checkpoint/CLI vertical slice.
6. YOLOX scales and quality evidence.
7. TAL/DFL primitives.
8. YOLOv3-Tiny-U trainability.
9. YOLO11 detect and scales.
10. YOLOv10 dual graph/loss.
11. YOLO26 dual graph/loss.
12. Classification task.
13. YOLO11 segmentation.
14. YOLO26 segmentation/semantic branch.
15. Full augmentations, quality hardening, and docs graduation.

Every graph-changing PR must include before/after official inference golden results. Every
loss-changing PR must include a Python parity fixture. Every checkpoint-format change must include
round-trip, corruption, and compatibility tests.

## 21. Final release acceptance checklist

Training is ready to advertise for all current models only when all items below are checked:

- [ ] Default inference and no-default-feature builds remain green.
- [ ] All 40 `ModelId` values complete a training dry-run on their correct task.
- [ ] Every distinct assignment/loss flavor has exact discrete and numeric parity fixtures.
- [ ] Every family has one optimizer-step and BN-running-state parity evidence.
- [ ] Every family/task has a tiny-overfit proof.
- [ ] Empty/mixed-empty batches are finite for every task.
- [ ] Dual-head detach behavior is gradient-tested.
- [ ] Changed-class fine-tuning replaces only allowlisted class projections.
- [ ] Epoch-boundary resume reproduces sample order, LR, loss schedule, optimizer, EMA, and next
  update.
- [ ] Validation leaves BN state unchanged.
- [ ] Detection, segmentation, and classification metrics are tested against reference outputs.
- [ ] EMA export reloads in the public predictor with embedded custom class names.
- [ ] Training-only dual/semantic branches are absent from inference artifacts.
- [ ] Single-GPU batch-1 plus accumulation works for large variants within documented hardware
  limits.
- [ ] COCO8/classification-small/COCO8-seg reference deltas meet agreed thresholds.
- [ ] Checkpoints are atomic, hash-validated, versioned, and corruption-tested.
- [ ] Run metadata records source versions, backend/adapter, config, dataset fingerprint, licenses,
  and metrics.
- [ ] README, AGENTS, ROADMAP, NOTICE, and CLI help describe the shipped behavior accurately.

The central rule throughout implementation is: inference parity protects the graph, tensor parity
protects assignment and loss semantics, one-step parity protects optimization, and tiny/full-dataset
quality gates protect the complete system. None of those gates substitutes for another.
