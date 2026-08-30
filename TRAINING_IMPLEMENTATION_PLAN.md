# Native training completion tracker

Status: implementation is complete except for the external-reference quality/parity campaign.
Keep this file until that last release gate is closed, then delete it; the durable workflow and
verification commands now live in `README.md` and `AGENTS.md`.

The non-default `training` feature currently provides native WGPU training from scratch,
epoch-boundary resume, EMA/full-state checkpoints, validation, and inference-artifact export for the
supported rows below. Training losses consume raw logits and the default inference API remains
unchanged.

## Current support

| Family/task | Scratch + resume | Validation | Training-checkpoint export |
|---|---|---|---|
| YOLOX detect (nano/tiny/s/m/l/x) | yes | box AP | yes |
| YOLOv3-Tiny-U detect | yes | box AP | yes |
| YOLOv10 detect (n/s/m/b/l/x) | yes | box AP | yes |
| YOLO11 detect (n/s/m/l/x) | yes | box AP | yes |
| YOLO11 segment (n/s/m/l/x) | yes | box + mask AP | yes |
| YOLO11 classify (n/s/m/l/x) | yes | top-1/top-5 | yes, including custom classes |
| YOLOv8 detect/segment (n/s/m/l/x) | yes | box + mask AP | yes |
| YOLOv8 classify (n/s/m/l/x) | yes | top-1/top-5 | yes, including custom classes |
| YOLO12 detect (n/s/m/l/x) | yes | box AP | yes |
| YOLO26 detect (n/s/m/l/x) | yes | box AP | yes |
| YOLO26 segment (n/s/m/l/x) | yes | box + mask AP | yes |
| YOLO26 classify (n/s/m/l/x) | yes | top-1/top-5 | yes, including custom classes |

Validation uses the EMA record on the non-autodiff WGPU backend, applies the family-appropriate NMS
or end-to-end top-k path, and uses checkpointed confidence, IoU, and maximum-detection settings.
Boxes and masks report independent AP50/AP50--95 in source-image coordinates; classification
reports top-1/top-5. YOLOX retains its top-left/raw-pixel preprocessing, while the Ultralytics
families use centered validation letterboxing. Trained artifacts embed model/task metadata,
ordered class names, and input geometry, and the public predictor reloads those custom contracts.

## Remaining release gates

- [x] Wire YOLOv8 detect/segment and YOLO12 detect through raw training forwards, losses, CLI
  dispatch, validation, and export.
- [x] Add pretrained `--weights` fine-tuning. Equal-class loads must be strict; changed-class loads
  may replace only documented class projections (and the YOLO26 semantic projection). Resume must
  remain an exact full-state load.
- [x] Add segmentation validation with independent source-space box and mask AP, using the same mask
  assembly and geometry as public inference.
- [x] Export detector and segment EMA graphs, omit training-only one-to-many/semantic branches, and
  smoke-reload each artifact through the public predictor.
- [x] Embed ordered custom class names and model metadata in trained artifacts, and add a predictor
  constructor that uses those names instead of the static COCO/ImageNet tables.
- [x] Finish validation configuration (`max_detections`, confidence/IoU policy) and COCO dataset
  orchestration, including crowd/ignore evaluation.
- [x] Run native release-mode COCO8, COCO8-seg, and ImageNet-small optimizer/EMA/validation gates
  on the maintained GPU.
- [ ] Exercise the official one-step fixtures for every loss family, prove tiny overfit per task,
  and record the external-reference quality deltas.
- [x] Confirm exported inference CPU/GPU parity and leave all default/no-default inference tests
  unchanged.

## Hardware evidence (2026-08-30)

- Release-mode custom-class optimizer/EMA steps completed on the maintained RTX 5080 for YOLO26n
  detect, YOLO26n-seg, and YOLO11n-cls. Each checkpoint validated, exported, and reloaded through
  the public predictor.
- One-epoch native release runs completed for YOLO26n on COCO8 (loss 3.1506), YOLO26n-seg on
  COCO8-seg (loss 7.3249), and YOLO26n-cls on ImageNet-10 (loss 2.2592). Validation completed for
  all three tasks; generated reports and full-state checkpoints remain under
  `target/training-quality/native-runs/`.
- DFL-free YOLO26 raw side distances can briefly decode to inverted boxes early in fine-tuning.
  TAL now treats those finite predictions as zero-overlap candidates instead of aborting the batch;
  non-finite predictions remain errors.
- YOLO26n-seg independently reported box/mask AP. Its semantic branch now uses an exact
  differentiable 2x half-pixel bilinear expression because Burn's WGPU JIT does not provide the
  generic bilinear-interpolation backward; one-class semantic targets bypass Burn's two-class
  minimum for one-hot tensors.
- Exported detect, segment (including mask pixel counts), and classification artifacts matched
  between Flex CPU and WGPU within ordinary floating-point drift. The maintained release-mode
  `wgpu_autodiff_capability` gate passed.
- The YOLOX-nano release smoke remains quarantined: on this adapter it exited during the first
  forward/loss step after successful strict weight transfer. No checkpoint or metric event was
  written. Resolve that backend path and complete the official fixture/tiny-overfit/quality reports
  before deleting this tracker.

## Verification

Run after every training change:

```console
cargo fmt --check
cargo test --locked
cargo test --locked --features training training
cargo clippy --locked --features training --all-targets -- -D warnings
cargo check --locked --no-default-features --lib
```

When external checkpoints and generated fixtures are available, also run the ignored parity and
quality tests. Generated datasets, checkpoints, fixtures, exports, and reports belong under
`target/` and must not be committed.
