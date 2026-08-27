# boquilens roadmap

The product goal is an Ultralytics-like computer-vision toolkit whose runtime and training stack are
native Rust/Burn. The useful comparison is the workflow—predict, train, validate, export—not the
number of model names in the catalog.

The sequencing rule is vertical completeness. One model should work from weights to useful output,
then from dataset to trained checkpoint, before broad model and task coverage. Preprocessing and
augmentation become their own layer only after the training interfaces force the right design.

## Architecture boundaries

The public surface is split conceptually into five layers. They can remain one crate during the MVP
and become workspace crates when compile times or independent consumers justify it.

1. **Catalog** — stable model IDs, metadata, configuration, official weight manifests, and class
   schemas. `ModelId` is the first piece of this API.
2. **Models** — native Burn modules. Architectures must not depend on CLI, image decoding, download,
   or rendering code. `models::yolox` is stable; `models::yolov3_tiny` is the first experimental
   second-family probe.
3. **Engine** — backend/device selection, checkpoint loading, batched forward calls, task-specific
   decode, NMS, and typed results. `Predictor` is the first inference engine.
4. **Data** — source discovery, decoding, transforms, batching, datasets, labels, caching, and later
   augmentation. This stays intentionally small in the MVP.
5. **Workflows** — `predict`, `train`, `val`, `export`, benchmark, configuration, callbacks, and CLI.

Model forward methods should accept tensors and return typed raw predictions. They must never know
about filesystem paths. The engine owns postprocessing. Dataset transforms must be testable without
a model. These boundaries keep a future YOLO26 or YOLO10 implementation from duplicating the whole
application.

## M0 — executable detector (current MVP)

- [x] Native YOLOX backbone, PAFPN, decoupled head, box decode, and class-aware NMS in boquilens
- [x] Official YOLOX-Nano COCO checkpoint import and cache
- [x] Typed Rust `Predictor` and `Detection` API
- [x] CLI with model selection, thresholds, JSON, and annotated-image output
- [x] Real-image smoke run detecting person, bicycle, and dog
- [x] Unit tests, Clippy, rustfmt, architecture-only build, and CI definition
- [x] Letterbox transform with exact inverse coordinate mapping

The executable MVP is complete. Golden tensor parity against the original PyTorch checkpoint and a
release-mode CPU/GPU timing and memory baseline are the first hardening gates for M1.

The experimental YOLOv3-Tiny-U path now has golden intermediate/decode parity, stride-aligned
rectangular preprocessing, and a versioned half-precision native Burnpack artifact. It remains
outside the stable MVP promise until the artifact has a license-compatible distribution channel,
max-detection behavior is implemented, and validation is expanded beyond one parity image.

## M1 — inference product

- Introduce `Source` and `Batch` types for files, directories, byte buffers, and decoded images.
- Make class schemas owned data so custom checkpoints are not tied to COCO static strings.
- Extend the checksum-verified weight manifest beyond the MVP checkpoint and version its schema.
- Add backend/device selection and report the backend actually chosen by Burn Flex.
- Add batching, structured timings, max-detection limits, and deterministic postprocessing tests.
- Render labels/confidence and offer callbacks or iterators rather than baking output policy into the
  engine.
- Publish the library and CLI from a reproducible lockfile; add Windows/Linux/macOS CI.

## M2 — one trainable detector

Finish the YOLOX vertical first because its license is permissive and the inference implementation
already exists.

- Define task-neutral `Sample`, detection target, dataset, batcher, and transform traits.
- Implement YOLOX assignment and losses (SimOTA, IoU, objectness, classification) as isolated,
  tensor-level modules.
- Build `Trainer`, optimizer/scheduler configuration, checkpoint resume, metrics, and validation.
- Add mandatory PyTorch parity fixtures for assignment, every loss term, decoded predictions, and one
  optimizer step.
- Prove overfit on a tiny fixture, then train/fine-tune COCO8 and compare mAP with the reference.

This is the real “Ultralytics competitor” threshold: `boquilens train`, `val`, and `predict` all use
the same native checkpoint and model definition.

## M3 — data and preprocessing system

Only after the M2 interfaces stabilize:

- letterbox/resize policies, geometric transform algebra, and reversible box mapping;
- mosaic, mixup, copy-paste, HSV/color transforms, flips, crops, and multi-scale training;
- deterministic seeded pipelines, parallel decoding, cache policy, and visualization/debug hooks;
- COCO and YOLO dataset formats first, then plugin-style format adapters.

Transforms should update images and targets together and carry enough geometry metadata to map
predictions back to source coordinates. That invariant matters more than matching a long feature
list.

## M4 — modern architecture and broader tasks

After training is proven, add a modern detector through the same interfaces. The YOLOv3-Tiny-U
experiment de-risks Ultralytics-style split heads, DFL decode, preprocessing, and state mapping;
the YOLOv10n port extends that to the modern C2f/SCDown/PSA stack and the NMS-free one2one head
before attempting YOLO26. The license decision is recorded: boquilens is AGPL-3.0, which makes
Ultralytics architectures and official checkpoints license-compatible and lets derived artifacts
ship under the same terms.

Then expand to segmentation, pose, oriented boxes, and classification by adding task heads/results,
not parallel applications. Export (Burn native, ONNX where practical), quantization, deployment
profiles, model cards, benchmarks, and a registry follow once two architectures prove that the
abstractions are real.

## Near-term order

1. Publish the versioned YOLOv3-Tiny-U and YOLOv10n Burnpack artifacts (AGPL-3.0 release channel)
   and wire their checksum-verified downloaders into the catalog.
2. Add `max_detections`, multi-image parity fixtures, and validation metrics.
3. Resume batched predictor/backend selection and YOLOX loss/assignment parity.

The AGPL-3.0 licensing question is settled; the distribution channel is the remaining blocker for
promoting the Ultralytics-family models out of experimental.
