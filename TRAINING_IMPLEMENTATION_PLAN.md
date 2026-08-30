# Native training completion tracker

Status: substantially implemented, but not complete. Keep this file until every release gate below
is closed; once that happens, move the durable workflow and verification commands into `README.md`
and `AGENTS.md`, then delete this tracker.

The non-default `training` feature currently provides native WGPU training from scratch,
epoch-boundary resume, EMA/full-state checkpoints, validation, and inference-artifact export for the
supported rows below. Training losses consume raw logits and the default inference API remains
unchanged.

## Current support

| Family/task | Scratch + resume | Validation | Training-checkpoint export |
|---|---|---|---|
| YOLOX detect (nano/tiny/s/m/l/x) | yes | box AP | no |
| YOLOv3-Tiny-U detect | yes | box AP | no |
| YOLOv10 detect (n/s/m/b/l/x) | yes | box AP | no |
| YOLO11 detect (n/s/m/l/x) | yes | box AP | no |
| YOLO11 segment (n/s/m/l/x) | yes | no | no |
| YOLO11 classify (n/s/m/l/x) | yes | top-1/top-5 | official 1000-class artifacts only |
| YOLOv8 detect/segment (n/s/m/l/x) | no | no | no |
| YOLOv8 classify (n/s/m/l/x) | yes | top-1/top-5 | official 1000-class artifacts only |
| YOLO12 detect (n/s/m/l/x) | no | no | no |
| YOLO26 detect (n/s/m/l/x) | yes | box AP | no |
| YOLO26 segment (n/s/m/l/x) | yes | no | no |
| YOLO26 classify (n/s/m/l/x) | yes | top-1/top-5 | official 1000-class artifacts only |

Detector validation uses the EMA record on the non-autodiff WGPU backend, applies the
family-appropriate NMS or end-to-end top-k path, caps results at 300, and reports AP50 and
AP50--95 in source-image coordinates. YOLOX retains its top-left/raw-pixel preprocessing; the
Ultralytics families use centered validation letterboxing. Classification validation and export
are implemented, but custom-class artifacts are deliberately rejected because the public
`Predictor` cannot reload embedded class metadata yet.

## Remaining release gates

- [ ] Wire YOLOv8 detect/segment and YOLO12 detect through raw training forwards, losses, CLI
  dispatch, validation, and export.
- [ ] Add pretrained `--weights` fine-tuning. Equal-class loads must be strict; changed-class loads
  may replace only documented class projections (and the YOLO26 semantic projection). Resume must
  remain an exact full-state load.
- [ ] Add segmentation validation with independent source-space box and mask AP, using the same mask
  assembly and geometry as public inference.
- [ ] Export detector and segment EMA graphs, omit training-only one-to-many/semantic branches, and
  smoke-reload each artifact through the public predictor.
- [ ] Embed ordered custom class names and model metadata in trained artifacts, and add a predictor
  constructor that uses those names instead of the static COCO/ImageNet tables.
- [ ] Finish validation configuration (`max_detections`, confidence/IoU policy) and COCO dataset
  orchestration, including crowd/ignore evaluation.
- [ ] Exercise the official one-step fixtures for every loss family, prove tiny overfit per task,
  record COCO8/COCO8-seg/ImageNet-small quality deltas, and run the maintained-GPU hardware gates.
- [ ] Confirm exported inference CPU/GPU parity and leave all default/no-default inference tests
  unchanged.

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
