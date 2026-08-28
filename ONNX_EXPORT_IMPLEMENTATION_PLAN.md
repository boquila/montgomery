# ONNX Export for Every boquilens Model — Detailed Implementation Plan

Status: design only. This document does not implement export and does not modify model code.

This plan defines how boquilens will export the exact model and weights loaded by Rust into portable ONNX inference artifacts. It covers every model registered in `ModelId`, every currently supported task, pretrained Burnpack/PyTorch sources, and future Burn-trained checkpoints.

The central constraint is that Burn 0.21 has an ONNX **importer**, not an ONNX exporter. The first production-quality implementation must therefore not depend on a nonexistent Burn tracing API. The recommended implementation is a controlled two-stage bridge:

1. Rust loads the requested boquilens model and exact requested checkpoint, validates it, and writes a temporary SafeTensors parameter snapshot plus a strict export manifest.
2. A pinned, repository-owned Python adapter reconstructs the corresponding reference graph, strictly loads that snapshot, calls the standard PyTorch ONNX exporter, validates the ONNX model, runs ONNX Runtime parity, and atomically publishes the artifact.

The model parameters always come from the loaded boquilens model. The Python graph is an export adapter, not an alternate checkpoint source. This makes the path work for future weights trained natively in Burn, not only for official upstream weights.

Python is required only while producing and validating the artifact. The resulting ONNX file has no Python, PyTorch, Burn, or boquilens runtime dependency; it can be loaded by any runtime that supports the declared standard ONNX operators and opset.

## 1. Required outcome

The finished feature must provide one user-facing command such as:

```console
cargo run --locked --release -- export-onnx \
  --model yolo26n \
  --weights target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk \
  --output target/yolo26n.onnx
```

PowerShell equivalent:

```powershell
cargo run --locked --release -- export-onnx `
  --model yolo26n `
  --weights target/yolo26n-coco-ultralytics-v8.4-boquilens-v1.bpk `
  --output target/yolo26n.onnx
```

Successful export means all of the following, not merely that a file exists:

- the requested checkpoint was loaded strictly into the requested architecture;
- every required inference parameter was exported exactly once;
- the ONNX graph passes the ONNX checker and strict shape inference;
- ONNX Runtime can create a CPU session and execute the graph;
- outputs match the same boquilens model on deterministic test inputs;
- task, class names, image shape, color order, coordinate convention, license, and output schema are embedded as metadata and emitted in a sidecar manifest;
- the final artifact is written atomically and has a recorded SHA-256;
- no network download or package installation occurs implicitly during export;
- temporary files are removed on success and retained with a diagnostic path on failure.

## 2. Scope

### 2.1 Initial model coverage

Coverage is defined by `ModelId`, not by directories that happen to exist in `src/models`. At the current catalog snapshot this is 40 variants:

| Family | Task | Variants | Count |
|---|---|---|---:|
| YOLOX | detect | nano, tiny, s, m, l, x | 6 |
| YOLOv3-Tiny-U | detect | tiny | 1 |
| YOLOv10 | detect | n, s, m, b, l, x | 6 |
| YOLO11 | detect | n, s, m, l, x | 5 |
| YOLO11 | segment | n, s | 2 |
| YOLO11 | classify | n, s, m, l, x | 5 |
| YOLO26 | detect | n, s, m, l, x | 5 |
| YOLO26 | segment | n, s, m, l, x | 5 |
| YOLO26 | classify | n, s, m, l, x | 5 |

Concurrent or future model work becomes export scope only when its variants are registered in `ModelId`. Add a compile-time or unit-test coverage table so adding a `ModelId` without an `ExportSpec` fails immediately.

### 2.2 Initial feature scope

- inference-only ONNX graphs;
- fixed spatial input size by default;
- fixed batch size `1` by default;
- optional dynamic batch after its parity gate passes;
- float32 as the required portable artifact;
- optional float16 as a separately validated artifact;
- decoded/model-native output and Ultralytics-compatible output profiles;
- detection, instance segmentation, and classification;
- embedded weights by default;
- optional ONNX external-data layout;
- ONNX Runtime CPU validation as a required gate;
- optional CUDA execution-provider validation;
- official/pretrained and future Burn-trained weights;
- class-count/name overrides carried by a training checkpoint manifest.

### 2.3 Explicitly out of initial scope

- exporting an ONNX training graph, gradients, optimizer, or scheduler;
- serializing image/JPEG decoding into ONNX;
- including source-image letterboxing inside the default graph;
- including mask rendering to source-image resolution inside the default graph;
- universal in-graph NMS across every runtime;
- INT8 quantization in the first export milestone;
- TensorRT engine, OpenVINO IR, Core ML, TFLite, NCNN, or vendor accelerator generation;
- arbitrary custom ONNX operators;
- silently invoking the installed `ultralytics` package without checking its version/source;
- treating Burn’s ONNX importer as if it could export Burn modules.

## 3. Source and toolchain pins

The implementation must record exact versions in both logs and output metadata.

### 3.1 Rust source pins

- boquilens git commit and dirty state;
- `ModelId` string and export-spec version;
- Burn `0.21.0-pre.4` as currently locked;
- burn-store `0.21.0-pre.4` as currently locked;
- checkpoint format and checkpoint hash;
- model architecture/version metadata already associated with the checkpoint.

### 3.2 Ultralytics graph source

For Ultralytics-family models, the normative graph adapter is the sibling vendored tree:

- path: `../ultralytics`;
- commit: `461196cf09175b64c9b9bd8babebf081c0540520`;
- description: `v8.4.117-2-g461196cf0`.

The current conversion environment may have a different `ultralytics` wheel installed. The exporter must prepend the sibling tree to `sys.path`, import it, and assert the imported module’s resolved file lives under that sibling directory. Version-string equality alone is insufficient.

### 3.3 YOLOX graph source

Reuse the reference mechanism already established by `tools/export_yolox_fixtures.py`:

- official Megvii YOLOX source checkout;
- tag `0.1.1rc0` unless the model’s provenance changes;
- default checkout location under `target/yolox-ref/YOLOX-0.1.1rc0`;
- Apache-2.0 provenance already described in `NOTICE`.

The exporter may reuse/refactor the development-time shim, but it must not download the checkout automatically. Missing source should produce an actionable setup command and expected hash/tag.

### 3.4 Python export environment

Create a dedicated locked requirements file during implementation, separate from ordinary Rust inference. Pin at minimum:

- Python supported range;
- PyTorch;
- ONNX;
- ONNX Runtime CPU;
- SafeTensors;
- NumPy;
- vendored Ultralytics import path rather than a floating pip package;
- optional `onnxslim` only if simplification becomes supported.

The existing `target/.venv` currently lacks ONNX. Export preflight must test imports and print the exact environment setup command. Never call pip automatically from the Rust CLI.

## 4. Licensing and provenance

ONNX export does not erase model or weight licensing.

- boquilens and Ultralytics-family adapted graphs/artifacts remain under the project’s AGPL-3.0 boundary.
- YOLOX code and official weights retain their Apache-2.0 provenance and notices.
- A model trained from an AGPL Ultralytics checkpoint remains subject to that checkpoint/source provenance.
- User-trained datasets and resulting weights may carry additional obligations that boquilens cannot infer.
- Every ONNX model should include `model_license`, source/model provenance, and a link/reference to `NOTICE` in metadata.
- The sidecar manifest must state whether the artifact came from an official pretrained checkpoint or a user training checkpoint.
- Do not label every artifact simply “boquilens AGPL” if its YOLOX path has a distinct Apache provenance.

Relevant technical specifications and compatibility references:

- [Burn ONNX repository: ONNX-to-Burn import](https://github.com/tracel-ai/burn-onnx)
- [Burn discussion confirming export was not supported](https://github.com/tracel-ai/burn/discussions/1792)
- [ONNX IR specification and metadata](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX Runtime compatibility table](https://onnxruntime.ai/docs/reference/compatibility.html)
- [ONNX checker and shape-inference API](https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html)

## 5. Architecture decision

### 5.1 Chosen first implementation: snapshot plus graph adapter

Use `burn_store::ModuleSnapshot` to collect the loaded model’s parameters. Write them to SafeTensors using `SafetensorsStore` and `BurnToPyTorchAdapter`, with family-specific reversible key mapping. Pass the snapshot and manifest to a repository-owned Python exporter.

Why this route is preferred:

- it exports the actual loaded Burn parameters;
- SafeTensors is typed, shape-aware, deterministic, and avoids Python pickle for the handoff;
- BurnToPyTorchAdapter already handles linear-weight transposition and normalization names;
- current model import paths already encode most PyTorch-to-Burn key mappings;
- PyTorch’s ONNX exporter and ONNX checker handle graph serialization correctly;
- the vendored/reference source graphs already express all complicated blocks;
- it can be implemented and validated incrementally per family;
- it avoids maintaining a second full native graph description before export has users.

### 5.2 Rejected as the first implementation: Burn ONNX API

Burn’s ONNX support imports ONNX to Burn code. There is no supported `model.export_onnx()` equivalent in the pinned Burn release. Do not add a dependency named `burn-onnx` expecting it to solve export.

### 5.3 Deferred: tracing backend

A custom Burn backend could theoretically delegate computation while recording operations. It is not the first implementation because:

- the backend operation surface is large;
- parameters and input leaves need stable identities;
- Rust control flow has already concretized many shapes;
- a batch-1 trace bakes literal reshape dimensions unless symbolic-shape provenance is added;
- exported graph semantics would depend on backend lowering details rather than model-level operations.

Keep this as an experiment, not the delivery path.

### 5.4 Deferred: fully native ONNX emitter

A future native exporter may write `ModelProto` directly from model-level `OnnxExport` implementations. It removes Python but duplicates every block’s forward graph. It should begin only after the bridge defines stable output contracts and parity fixtures. See Section 39.

## 6. End-to-end export flow

```text
CLI arguments
  -> resolve ModelId / task / scale / input contract
  -> locate and hash checkpoint
  -> load exact model on CPU with strict validation
  -> select EMA or raw training weights
  -> collect Burn parameter snapshots
  -> Burn-to-PyTorch tensor adaptation
  -> reversible family key mapping
  -> write temporary .safetensors + manifest.json
  -> invoke pinned Python adapter
      -> verify source checkout/import
      -> instantiate exact reference graph without downloading weights
      -> load every tensor strictly
      -> configure eval/export profile
      -> dry-run output contract
      -> torch ONNX export
      -> ONNX checker + shape inference
      -> ONNX Runtime CPU execution
      -> compare against Rust reference outputs
      -> write metadata + final manifest
  -> Rust verifies output hashes/manifest
  -> atomic rename into requested location
```

If any step fails, no partial file should appear at the final output path.

## 7. CLI design

Add a separate subcommand; do not overload `pack-weights`:

```text
boquilens export-onnx
  --model <MODEL_ID>
  --weights <PATH>
  --output <PATH>
  [--imgsz <H,W|S>]
  [--batch <N>]
  [--dynamic-batch]
  [--dynamic-spatial]
  [--opset <N>]
  [--profile portable|ultralytics|end2end]
  [--precision fp32|fp16]
  [--external-data never|auto|always]
  [--python <PATH>]
  [--yolox-repo <PATH>]
  [--verify-source <IMAGE> ...]
  [--simplify]
  [--force]
  [--keep-intermediate]
```

### 7.1 Defaults

- `imgsz=640` for detect/segment;
- `imgsz=224` for classify;
- `batch=1`;
- fixed spatial dimensions;
- fixed batch until dynamic-batch parity is implemented, then dynamic batch may become the default;
- opset `17` for broad runtime compatibility;
- profile `portable`;
- precision `fp32`;
- external data `auto`, with embedded tensors for all current sub-2GB models;
- verify with synthetic inputs automatically and one reference image when supplied;
- refuse overwrite without `--force`.

### 7.2 Python resolution

Resolution order:

1. explicit `--python`;
2. task-specific environment variable such as `BOQUILENS_ONNX_PYTHON`;
3. `target/.venv/Scripts/python.exe` on Windows or `target/.venv/bin/python` on Unix if present;
4. fail with setup instructions.

Do not fall through to an arbitrary system Python silently.

### 7.3 Exit behavior

- zero only after structural and numerical validation;
- non-zero for missing package, source mismatch, unsupported model, key mismatch, checker failure, runtime failure, parity failure, or unsafe overwrite;
- print the retained intermediate directory on failure;
- output machine-readable JSON with `--json` in addition to concise stderr progress.

## 8. Export specification registry

Define one `ExportSpec` for every `ModelId`:

```rust
struct ExportSpec {
    model_id: ModelId,
    family: ExportFamily,
    task: ExportTask,
    scale: Scale,
    default_input: InputShape,
    stride: usize,
    num_classes: usize,
    output_contract: OutputContract,
    graph_source: GraphSource,
    key_map_version: &'static str,
    license_profile: LicenseProfile,
}
```

Required tests:

- every `ModelId` has exactly one spec;
- no spec points to another `ModelId` accidentally;
- detect/segment dimensions are stride-compatible;
- class counts agree with model/checkpoint metadata;
- output profile is supported by task/family;
- default filenames are unique;
- adding a new model fails the exhaustiveness test until export support is declared.

Do not infer a family from string prefixes in core logic. Use exhaustive enum matching.

## 9. Input contract

The default ONNX graph input is named `images`:

```text
dtype: float32
layout: NCHW
channels: 3
color: RGB
range: [0, 1]
shape detect/segment: [batch, 3, 640, 640] by default
shape classify:       [batch, 3, 224, 224] by default
```

The graph does not decode files, convert BGR, perform letterbox, or remember source geometry. Consumers must use metadata/sidecar preprocessing instructions.

### 9.1 Detection/segmentation preprocessing metadata

For Ultralytics families:

- stride-aligned centered letterbox;
- fill value `114`;
- RGB;
- divide by `255`;
- boxes emitted in model-input pixel space unless an end-to-end profile states otherwise.

For YOLOX:

- top-left/raw-pixel resize behavior used by the current runtime;
- fill value `114`;
- RGB tensor after source BGR conversion;
- scale/range exactly matching current YOLOX preprocessing.

Do not describe YOLOX with Ultralytics letterbox metadata.

### 9.2 Classification preprocessing metadata

- aspect-preserving shortest-edge resize;
- centered crop to `224 x 224`;
- RGB scaled to `[0,1]`;
- mean `[0,0,0]`, standard deviation `[1,1,1]`;
- anti-alias/interpolation notes from current classification invariant.

## 10. Output profiles

Offer explicit profiles because there is no single ideal ONNX output for every deployment.

### 10.1 `portable` profile — required default

Exports model tensors before NMS, source-coordinate conversion, mask assembly, and confidence filtering. Outputs are named and task-specific. This profile uses only ordinary ONNX tensor operators and maximizes runtime compatibility.

Advantages:

- thresholds remain runtime configurable;
- avoids runtime-specific `NonMaxSuppression` behavior;
- avoids dynamic-length detection outputs;
- allows boquilens and other consumers to implement their preferred postprocess;
- preserves all candidates needed for parity debugging.

### 10.2 `ultralytics` profile — required for Ultralytics families

Matches the pinned Ultralytics ONNX export layout as closely as possible:

- classic detect: packed channel-first predictions;
- classic segment: packed predictions plus prototypes;
- end-to-end detect/segment: top-k rows in the pinned exported layout;
- classification: probability tensor.

This profile is the interoperability target for existing Ultralytics ONNX consumers.

### 10.3 `end2end` profile — later milestone

Includes postprocess with fixed-size padded outputs and an explicit valid-count tensor. It must be separately gated per runtime. It is not synonymous with YOLO10/26’s one-to-one head: those models are NMS-free, but confidence filtering, max-detection policy, source mapping, and mask assembly still exist outside the core graph.

Never silently switch profile based on model family.

## 11. Portable output contracts

### 11.1 YOLOX detection

One output:

```text
predictions: float32 [N, anchors, 5 + classes]
row: [center_x, center_y, width, height, objectness_probability, class_probabilities...]
```

Coordinates are model-input pixels. Consumers compute class confidence as objectness multiplied by class probability before NMS, matching current runtime behavior.

### 11.2 YOLOv3-Tiny-U detection

```text
boxes:  float32 [N, anchors, 4]       # XYXY input pixels
scores: float32 [N, anchors, classes] # sigmoid probabilities
```

### 11.3 YOLO11 detection

```text
boxes:  float32 [N, anchors, 4]       # center-size XYWH input pixels
scores: float32 [N, anchors, classes] # sigmoid probabilities
```

Class-aware NMS remains outside the graph.

### 11.4 YOLOv10 detection

```text
boxes:  float32 [N, anchors, 4]       # XYXY input pixels
scores: float32 [N, anchors, classes] # one-to-one sigmoid probabilities
```

Consumers apply the pinned two-stage top-k selection, confidence filter, and maximum 300 detections. They do not apply classic NMS.

### 11.5 YOLO26 detection

Same tensor contract as YOLOv10. Boxes are DFL-free decoded one-to-one XYXY input pixels. Do not insert a DFL projection.

### 11.6 YOLO11 segmentation

```text
boxes:        float32 [N, anchors, 4]             # center-size XYWH
scores:       float32 [N, anchors, classes]
coefficients: float32 [N, 32, anchors]            # raw, no sigmoid
prototypes:   float32 [N, 32, input_h/4, input_w/4]
```

NMS, coefficient gathering, coefficient/prototype multiplication, upsample, threshold, crop, empty-mask removal, and source-image mapping remain outside the graph.

### 11.7 YOLO26 segmentation

```text
boxes:        float32 [N, anchors, 4]             # XYXY
scores:       float32 [N, anchors, classes]
coefficients: float32 [N, 32, anchors]            # raw
prototypes:   float32 [N, 32, input_h/4, input_w/4]
```

Use end-to-end top-k selection instead of NMS in the consumer.

### 11.8 Classification

```text
logits:        float32 [N, classes]
probabilities: float32 [N, classes]
```

Export both in the portable profile. This supports exact loss/logit diagnostics and ordinary probability consumers. The Ultralytics profile may expose only probabilities if required for compatibility.

## 12. Ultralytics-compatible layouts

The pinned adapter must probe and assert these contracts rather than assuming them from documentation.

### 12.1 Classic detect families

Expected packed form:

```text
output0: [N, 4 + classes, anchors]
channels: boxes followed by sigmoid class scores
```

YOLO11 boxes are XYWH. YOLOv3-Tiny-U’s exact pinned reference layout/box form must be captured in a fixture because its boquilens portable contract is XYXY.

### 12.2 Classic segmentation

Expected:

```text
output0: [N, 4 + classes + 32, anchors]
output1: [N, 32, proto_h, proto_w]
```

Mask coefficients remain raw.

### 12.3 End-to-end YOLOv10/YOLO26 detect

Expected exported top-k rows:

```text
output0: [N, max_det, 6]
row: [x1, y1, x2, y2, confidence, class_id_as_float]
```

The graph selects top candidates but does not apply classic NMS. Verify whether confidence thresholding occurs inside or outside the pinned export and record that in metadata; do not infer it from the word “end-to-end.”

### 12.4 End-to-end YOLO26 segmentation

Expected:

```text
output0: [N, max_det, 6 + 32]
row: [x1, y1, x2, y2, confidence, class_id_as_float, coefficients...]
output1: [N, 32, proto_h, proto_w]
```

### 12.5 Classification

```text
output0: [N, classes] # softmax probabilities
```

## 13. Checkpoint inputs

Support these sources:

- official YOLOX `.pth` accepted by current model loading;
- native `.bpk` inference artifacts;
- future native training checkpoints containing model and optional EMA state;
- an already loaded in-memory model through a Rust API, after CLI support is stable.

### 13.1 Strict architecture checks

Before snapshot:

- requested `ModelId` equals checkpoint architecture metadata;
- task matches;
- scale/depth/width/module flavors match;
- class count matches head tensors;
- class-name count matches class count;
- BN flavor matches model invariant;
- required tensors all load;
- unexpected checkpoint tensors are reported;
- f16/f32 checkpoint precision is recorded;
- segmentation prototype/mask widths match the variant.

Allow legacy artifacts without a manifest only through existing proven loader logic, then synthesize export metadata from `ModelId` and weight metadata. Warn that custom class names cannot be recovered if absent.

### 13.2 EMA selection

For training checkpoints:

- default to EMA when present and valid;
- support `--checkpoint-state ema|model`;
- record the selection;
- never merge EMA and raw tensors;
- fail if `ema` is requested but absent.

## 14. Rust parameter snapshot

### 14.1 Collection

After loading the model on the Flex CPU backend:

1. call `ModuleSnapshot::collect` or save through `SafetensorsStore`;
2. use `BurnToPyTorchAdapter` for linear transposition and normalization parameter naming;
3. materialize tensors on CPU;
4. cast according to requested export precision;
5. attach native path, adapted path, shape, dtype, and byte hash to the manifest;
6. sort tensors lexicographically for deterministic output;
7. detect duplicate final keys before writing.

### 14.2 Required parameter semantics

- Conv2d weights retain `[out_channels, in_channels/groups, kernel_h, kernel_w]` ordering.
- Linear weights are converted from Burn `[in,out]` to PyTorch `[out,in]`.
- BatchNorm `gamma/beta` become `weight/bias`.
- BatchNorm running mean and variance are preserved.
- PyTorch-only `num_batches_tracked` may be synthesized as zero if the reference graph state dict requires it; it is not an ONNX inference initializer.
- dropout has no parameters and is disabled in eval.
- DFL projection constants may come from graph construction rather than checkpoint tensors.
- anchor grids/strides should be graph constants or shape-derived values, not mistaken for trained parameters.

### 14.3 Audit counts

Emit and compare:

- number of Burn snapshots;
- number after adapter/remapping;
- total scalar elements;
- total bytes by dtype;
- consumed PyTorch state keys;
- synthesized non-parameter keys;
- ignored inference-irrelevant keys;
- unmatched and duplicate keys.

Zero unmatched trained parameters is mandatory.

## 15. Bidirectional key mapping

Current loaders contain many one-way `PytorchStore.with_key_remapping` patterns. Export needs a reviewed reverse mapping.

Do not mechanically reverse arbitrary regular expressions. Some patterns may not be injective. Introduce family-specific typed mapping code:

```rust
trait CheckpointKeyMap {
    fn pytorch_to_burn(&self, key: &str) -> Result<String>;
    fn burn_to_pytorch(&self, key: &str) -> Result<String>;
    fn version(&self) -> &'static str;
}
```

Implementation requirements:

- exact branch/layer mappings for body, head, segmentation tower, and classify head;
- collision detection;
- round-trip tests for every real checkpoint key;
- expected-key fixture generated from reference `state_dict()`;
- no permissive “best effort” fallback;
- version bump when graph naming changes;
- shared use by import and export where practical, reducing drift.

YOLOX needs inverse mappings for C3 names, darknet indexed blocks, SPP/C3 split, and head convolution vectors. Ultralytics families need inverse mappings for numeric YAML layers, one2one branches, mask towers, prototypes, and classify heads.

## 16. Export manifest between Rust and Python

Write a JSON manifest beside the temporary SafeTensors file:

```json
{
  "schema": "boquilens-onnx-export-input-v1",
  "model_id": "yolo26n-seg",
  "family": "yolo26",
  "task": "segment",
  "scale": "n",
  "checkpoint_sha256": "...",
  "checkpoint_state": "ema",
  "weights_file": "weights.safetensors",
  "weights_sha256": "...",
  "input": {"dtype": "float32", "shape": [1, 3, 640, 640]},
  "profile": "portable",
  "precision": "fp32",
  "opset": 17,
  "class_names": ["person"],
  "key_map_version": "yolo26-v1",
  "graph_source": {"kind": "ultralytics", "commit": "461196cf..."}
}
```

Use relative paths within the private temporary directory. Reject path traversal and schema versions newer than the Python adapter supports.

## 17. Python graph construction

### 17.1 Common rules

- set deterministic CPU behavior where available;
- instantiate architecture without network access;
- set `eval()` before dry run/export;
- disable dropout/stochastic depth;
- load adapted state dict strictly;
- check every tensor shape before calling `load_state_dict`;
- prevent reference code from auto-downloading checkpoints;
- use explicit export wrapper classes owned by boquilens;
- avoid monkey-patching global library state beyond a tightly scoped context;
- restore changed flags in `finally` blocks.

### 17.2 Ultralytics families

- import from the pinned sibling tree by resolved path;
- construct the task/scale graph from its exact YAML/model class;
- override class count when checkpoint metadata requires it;
- use the correct head branch: classic, one2one, Segment, Segment26, or Classify;
- preserve per-scale YAML module flavor differences;
- preserve YOLO11 versus YOLO26 SPPF behavior;
- preserve classification BN defaults;
- configure export flags only through the wrapper/profile logic;
- never substitute a similarly named pip model.

### 17.3 YOLOX

- refactor the existing reference shim into a reusable helper;
- construct depth, width, and depthwise settings from the scale table;
- use custom class count from checkpoint metadata;
- load the adapted state dict strictly;
- set head decode mode to match requested output profile;
- keep Apache source provenance in generated metadata.

## 18. PyTorch ONNX export settings

Use one explicitly selected exporter path and pin its behavior. If the modern dynamo exporter is adopted, retain a tested fallback only if both produce the same declared contract.

Required arguments/concepts:

- input name `images`;
- stable named outputs from the profile;
- opset 17 by default;
- constant folding enabled only after parity proves it safe;
- no training mode;
- parameters exported as initializers;
- dynamic axes only when requested and supported;
- external data controlled by boquilens, not an exporter surprise;
- diagnostic artifacts retained on failure.

Do not use whatever PyTorch’s default opset happens to be. Do not rely on output names such as `output0` in the portable profile.

## 19. Opset and IR policy

### 19.1 Default

Use ONNX opset 17 for the first compatibility target. It is old enough for broad ONNX Runtime support and new enough for the operators required by these convolutional/attention graphs.

### 19.2 Supported range

Initially accept only tested opsets, for example `17`, `18`, and `19`. A user-provided number outside the tested matrix should fail unless an explicit experimental flag is added later.

### 19.3 IR version

- let the ONNX library choose an IR compatible with the selected opset/exporter;
- require IR no newer than the minimum supported ONNX Runtime baseline declared by the project;
- do not blindly decrement `ir_version` without checking model fields/features;
- run checker and ORT after any compatibility rewrite;
- record opset imports and IR in metadata/sidecar.

## 20. Operator inventory

The expected standard-ONNX operator set includes:

- `Conv` with groups for depthwise convolution;
- `BatchNormalization` or fused Conv parameters;
- `Sigmoid` and `Mul` for SiLU;
- `Relu`/other explicit activations where used;
- `MaxPool`;
- `Resize` for nearest/bilinear upsample;
- `Concat`, `Split`, `Slice`, `Gather`, `GatherElements`;
- `Reshape`, `Transpose`, `Flatten`, `Unsqueeze`, `Squeeze`;
- `Add`, `Sub`, `Mul`, `Div`, `Exp`;
- `ReduceSum`, `ReduceMean`, `GlobalAveragePool`;
- `MatMul`/`Gemm`;
- `Softmax`;
- `Range`, `Expand`, `Shape`, and casts for dynamic anchor grids if dynamic spatial export is enabled;
- `TopK` for Ultralytics-compatible end-to-end outputs.

The portable profile must contain no custom domains. Add an automated node inventory and reject unexpected `com.microsoft`, exporter-private, or boquilens custom operators.

## 21. BatchNorm and graph fusion

Default export should preserve unfused inference BatchNorm unless the pinned reference export fuses it before ONNX generation.

Two valid strategies:

1. Export Conv and BatchNormalization separately, preserving checkpoint values directly.
2. Fold BN into Conv in a controlled adapter and compare the folded graph to unfused Burn output.

Choose one per profile and record it. Do not accidentally apply BN momentum at inference; only epsilon, running mean/variance, gamma, and beta matter. Respect the model-specific epsilon invariants:

- YOLOX and classification use PyTorch defaults;
- Ultralytics detect/segment families use their configured flavor.

If fusing, calculate in float32 even when source artifacts are f16, then cast to requested output precision after folding.

## 22. Fixed and dynamic shapes

### 22.1 Fixed shape — first milestone

Export exact `[batch,3,height,width]`. Require positive sizes divisible by the maximum stride for detect/segment. Bake anchor grids and strides where the reference exporter does so. Fixed shapes maximize runtime support and simplify parity.

### 22.2 Dynamic batch — second milestone

Mark input dimension 0 as `batch` and every output’s dimension 0 as the same symbol. Test batch sizes `1`, `2`, and `4`. Avoid graph constants that bake the trace batch into reshapes.

### 22.3 Dynamic spatial — separate opt-in milestone

For detect/segment:

- mark input height/width symbolic;
- reconstruct feature-grid anchors from runtime shapes;
- express output anchor dimensions symbolically where possible;
- test at least `320x320`, `640x640`, `640x384`, and `960x640`, all stride-compatible;
- test segmentation prototype shape changes;
- reject end-to-end profiles when fixed TopK exceeds available anchors at small sizes;
- preserve non-square coordinate decode.

ONNX shape inference cannot always express the sum of multiscale anchor products. Unknown symbolic output dimensions are acceptable if runtime execution and metadata are correct.

Classification remains fixed `224x224` initially even though its adaptive pooling graph could accept other dimensions; the checkpoint’s documented preprocessing contract is 224.

## 23. Precision policy

### 23.1 FP32 — mandatory

- materialize/export initializers as float32;
- keep input/output float32;
- required to run under ONNX Runtime CPU;
- baseline for all parity tests.

### 23.2 FP16 — optional

- expose only after an FP32 model passes;
- keep I/O float32 by default and insert casts, or document FP16 I/O explicitly;
- validate on a GPU execution provider because ordinary ORT CPU does not support all FP16 operations;
- keep numerically sensitive operations in FP32 if required;
- apply looser but explicit task-level parity thresholds;
- name artifact with `-fp16.onnx` unless output path is explicit.

Do not call an f16 checkpoint an FP16 ONNX graph unless graph tensors/operators are actually exported at FP16.

### 23.3 INT8 — future

INT8 requires representative calibration data, preprocessing parity, Q/DQ graph policy, execution-provider target, and task accuracy validation. It belongs in a separate plan/milestone after FP32/FP16 export is stable.

## 24. External tensor data

Default current models should fit in one `.onnx` file. Still implement explicit policy:

- `never`: fail if protobuf limits would be exceeded;
- `auto`: use external data over a documented size threshold;
- `always`: write `<model>.onnx` and `<model>.onnx.data`.

Requirements:

- external-data location is relative and contains no parent traversal;
- write model and data into a temporary directory, then publish both atomically as far as the platform allows;
- sidecar hashes every file;
- checker/shape inference receives the model path when external data is used;
- moving the pair together remains valid;
- `--force` validates all replacement targets first.

## 25. Metadata embedded in ONNX

Use unique `metadata_props` keys. At minimum:

```text
boquilens.schema = onnx-metadata-v1
boquilens.model_id = yolo26n-seg
boquilens.family = yolo26
boquilens.task = segment
boquilens.profile = portable
boquilens.input.layout = NCHW
boquilens.input.color = RGB
boquilens.input.range = 0,1
boquilens.input.shape = 1,3,640,640
boquilens.box.format = xyxy
boquilens.box.space = model-input-pixels
boquilens.stride = 32
boquilens.class_names = <JSON array>
boquilens.checkpoint.sha256 = ...
boquilens.exporter.version = ...
boquilens.git.commit = ...
boquilens.graph_source = ultralytics@461196cf...
model_license = AGPL-3.0-or-applicable-model-license
```

Segmentation metadata adds coefficient count, prototype stride, raw-coefficient flag, mask threshold/crop semantics, and postprocess profile. NMS-free models add `nms=false`, `max_det=300`, and top-k semantics. Classic models add NMS defaults as recommendations, not baked behavior.

Avoid embedding absolute local paths, usernames, or temporary directories.

## 26. Sidecar export manifest

Write `<model>.onnx.json` with information too structured or large for metadata:

- schema version;
- all artifact filenames and SHA-256 hashes;
- model/task/scale/class names;
- checkpoint source/hash/state selection;
- boquilens/Burn/source graph versions;
- Python, PyTorch, ONNX, and ONNX Runtime versions;
- opset and IR version;
- fixed/dynamic axes;
- input preprocessing contract;
- output names, dtypes, shapes, and semantic fields;
- postprocessing pseudocode/version;
- license/provenance;
- validation cases and numerical statistics;
- operator-domain inventory;
- whether simplification/fusion/fp16 conversion occurred;
- export timestamp in UTC;
- reproducibility caveats.

The ONNX file is usable alone, but boquilens consumers should prefer the sidecar when present.

## 27. Postprocessing contract

Ship language-independent pseudocode in documentation/manifest, not executable source hidden in metadata.

### 27.1 YOLOX

- multiply objectness by each class probability;
- convert center-size boxes to XYXY;
- confidence filter;
- class-aware greedy NMS;
- map from top-left padded canvas to source pixels.

### 27.2 YOLOv3-Tiny-U and YOLO11

- choose class scores;
- confidence filter;
- class-aware NMS with configured IoU;
- convert YOLO11 XYWH to XYXY;
- cap to `max_detections` when matching Ultralytics if that option lands;
- reverse letterbox geometry.

### 27.3 YOLOv10 and YOLO26

- best class per anchor;
- first top-k anchors by best score;
- top-k flattened anchor/class pairs;
- gather boxes/classes/scores;
- confidence filter;
- no NMS;
- maximum 300;
- reverse letterbox.

### 27.4 Segmentation

- select/gather coefficients with surviving anchors;
- raw `coefficients @ prototypes`;
- bilinear upsample with `align_corners=false` semantics;
- threshold logits at `>0`;
- crop to model-input box;
- drop fully empty cropped masks;
- sample canvas mask back to source-image pixels using current letterbox geometry.

## 28. Structural ONNX validation

Python adapter must run:

1. `onnx.load` from the actual written path;
2. `onnx.checker.check_model`;
3. strict shape inference with type checking;
4. a second checker pass on the inferred graph;
5. input/output name/type/rank assertions;
6. initializer uniqueness checks;
7. graph topological validity;
8. operator-domain allowlist;
9. metadata key uniqueness/completeness;
10. external-data file/range checks if applicable.

The checker is necessary but not sufficient; ORT execution is mandatory.

## 29. Numerical parity validation

### 29.1 Three-way oracle

For each export compare:

1. native boquilens/Burn output from the loaded checkpoint;
2. PyTorch adapter output before ONNX export;
3. ONNX Runtime output from the exported file.

This separates weight/key-map errors from ONNX lowering errors.

### 29.2 Inputs

Use at least:

- all-zero float input;
- deterministic pseudo-random `[0,1]` input with recorded seed;
- structured gradient/checkerboard input to expose layout/resize errors;
- optional preprocessed reference image through the real task preprocessing;
- batch >1 when dynamic batch is enabled;
- multiple spatial sizes when dynamic spatial is enabled.

### 29.3 Tensor checks

For every named output:

- exact dtype/rank/shape;
- all finite;
- maximum absolute error;
- maximum relative error with safe denominator;
- mean absolute error;
- RMS error;
- selected indexed samples;
- task-specific semantic comparisons.

Set thresholds from observed cross-runtime data and commit them per precision/profile. Suggested starting gates for FP32 are `1e-4` class/prototype values and `1e-3` to `1e-2` decoded pixel boxes, tightened where observations allow. Never hide a large error behind a loose global tolerance.

### 29.4 End-to-end task checks

- detect: match detections by class/IoU and compare confidence/box coordinates;
- NMS-free detect: account for deterministic near-tie ordering while requiring the same strong detections;
- segment: compare boxes plus assembled source-space mask IoU;
- classify: compare logits, probability vector, and top-5 class set;
- YOLOX: ensure objectness/class multiplication occurs exactly once.

## 30. ONNX Runtime validation matrix

Required CI/development baseline:

- ONNX Runtime CPU on Windows and Linux;
- batch 1 fixed input;
- one smallest variant from every distinct graph family/task on ordinary PRs;
- all variants in scheduled/release validation.

Optional but recommended:

- ONNX Runtime CUDA on the reference GPU;
- DirectML on Windows if users need it;
- TensorRT parser smoke test for portable graphs;
- `tract` parse/run smoke test for pure-Rust consumers;
- Netron visual inspection for one artifact per graph family.

Do not make optional-provider failure invalidate the portable artifact unless that provider is an explicitly requested target.

## 31. Simplification and optimization

`--simplify` is opt-in initially.

Workflow:

1. validate the unsimplified model;
2. simplify into a separate temporary path with a pinned tool version;
3. rerun checker, shape inference, operator inventory, ORT, and numerical parity;
4. publish only if all checks pass;
5. record simplifier/tool version and before/after hashes/node counts.

Never replace a passing unsimplified graph with a failed simplified graph. Avoid relying on simplification to make an invalid export valid.

## 32. Error handling

Errors must identify the failing layer:

- argument validation;
- checkpoint loading;
- snapshot materialization;
- key remapping;
- reference graph construction;
- state-dict loading;
- dry run;
- PyTorch ONNX export;
- ONNX checker/shape inference;
- ORT session creation;
- ORT execution;
- parity comparison;
- publication.

On key mismatch, print grouped summaries and write the complete lists to the intermediate directory. On tensor mismatch, identify tensor name, shapes, first failing index, actual/expected values, and aggregate errors.

Do not catch an exception and still return a generated `.onnx` path.

## 33. Security and reproducibility

- prefer `.bpk`/SafeTensors over untrusted pickle inputs;
- existing `.pth` loading remains a trusted-development operation and should be documented as such;
- never execute code from a checkpoint;
- import Python graph code only from resolved pinned directories;
- disable network use in the export subprocess where practical;
- use a private temporary directory under `target` or OS temp;
- validate output paths before overwrite;
- do not include secrets/environment dumps in diagnostics;
- sort metadata/tensor reporting for reproducibility;
- record dirty source state;
- export the same model/config twice and compare graph/initializer hashes, excluding timestamp metadata;
- offer `--reproducible` to omit timestamp and stabilize protobuf ordering if required.

## 34. Atomic publication

1. Resolve final absolute output and sidecar paths.
2. Refuse existing targets unless `--force`.
3. Export within a private temporary directory on the same filesystem when possible.
4. Flush and close all files.
5. Re-open and hash every artifact.
6. Validate the final temporary paths.
7. Rename data file(s), ONNX file, then sidecar manifest according to a documented recovery order.
8. If publication fails, report which files moved and how to recover; do not delete valid user files.

The output path must have `.onnx` or receive it automatically with an explicit log message.

## 35. Rust API design

After CLI stability, expose a library API behind the relevant feature:

```rust
pub struct OnnxExportOptions {
    pub output: PathBuf,
    pub input_shape: [usize; 4],
    pub profile: OnnxProfile,
    pub opset: u32,
    pub precision: OnnxPrecision,
    pub dynamic_batch: bool,
    pub dynamic_spatial: bool,
    pub external_data: ExternalDataPolicy,
    pub verify: bool,
}

pub fn export_onnx(
    model_id: ModelId,
    weights: &Path,
    options: OnnxExportOptions,
) -> Result<OnnxArtifact>;
```

Because the first implementation launches a Python tool, name/document the dependency clearly. Do not imply the API is pure Rust until the native emitter exists.

## 36. Proposed implementation files

```text
src/export/
  mod.rs                  public orchestration and errors
  spec.rs                 exhaustive ModelId export registry
  snapshot.rs             ModuleSnapshot -> SafeTensors handoff
  keymap.rs               bidirectional key map contracts
  metadata.rs             input/output/license metadata
  manifest.rs             bridge and final manifest schemas
  verify.rs               Rust reference fixture generation
  families/
    yolox.rs
    yolov3_tiny.rs
    yolov10.rs
    yolo11.rs
    yolo26.rs
tools/onnx/
  export.py               Python entry point
  environment.py          source/version/no-network preflight
  common.py               strict loading/export/checker/ORT helpers
  contracts.py            output wrapper schemas
  ultralytics_adapter.py
  yolox_adapter.py
  compare.py
  requirements.lock.txt
tests/
  onnx_export.rs
```

Model files should not gain CLI/process logic. Key-map definitions may move out of model loaders only in a dedicated refactor with import parity tests.

## 37. Test plan

### 37.1 Pure Rust tests

- exhaustive `ModelId` coverage;
- defaults and argument validation;
- input/output contract serialization;
- key-map forward/reverse round trips;
- collision detection;
- tensor count/shape/hash manifest;
- class-name metadata escaping;
- output path/overwrite behavior;
- temporary-directory cleanup;
- subprocess error propagation;
- artifact hash verification.

### 37.2 Python tests

- pinned source import resolution;
- no-network model construction;
- strict state load for each family/task;
- wrapper output names/shapes;
- dynamic-axis declarations;
- checker and strict shape inference;
- operator allowlist;
- ORT comparison helpers;
- external-data relocation;
- simplification revalidation;
- malformed manifest/path rejection.

### 37.3 Integration test tiers

Tier 1, ordinary CI:

- tiny synthetic model/key map;
- one smallest detect model per distinct head type;
- one smallest classic seg and end-to-end seg;
- one classification model;
- fixed FP32 portable profile.

Tier 2, ignored/local with checkpoints:

- every family/task;
- Ultralytics profile;
- dynamic batch;
- reference image end-to-end.

Tier 3, release/nightly:

- all 40 registered variants;
- fixed and supported dynamic shapes;
- FP32 and supported FP16;
- CPU and optional GPU providers;
- repeatability/hash audit;
- export latency and artifact size report.

## 38. Performance expectations

Export is offline; correctness dominates speed. Still measure:

- checkpoint load time;
- tensor snapshot/materialization time;
- SafeTensors size/write time;
- reference graph construction/state load;
- ONNX export time;
- checker/shape-inference time;
- ORT session creation and first inference;
- final artifact size/node/initializer counts.

Avoid holding duplicate model weights in multiple Rust buffers longer than necessary. For large variants, close/drop the Rust model after snapshot/reference fixtures are complete before Python loads its copy if process memory is constrained.

## 39. Native exporter follow-up

Once bridge export is stable, evaluate a pure-Rust ONNX emitter.

### 39.1 Entry criteria

- stable v1 input/output contracts;
- complete parity fixtures for all graph families;
- proven demand for Python-free export;
- selected maintained Rust protobuf/ONNX schema dependency;
- willingness to maintain graph emission alongside every model block.

### 39.2 Recommended native design

- model-level `OnnxExport` trait, not backend-kernel tracing;
- graph builder with typed values/shapes and standard operators;
- parameter source from `ModuleSnapshot`;
- export methods adjacent to blocks so private module structure is accessible;
- explicit symbolic dimensions;
- reusable emitters for Conv-BN-SiLU, depthwise blocks, C2f/C3k/CIB, SPPF, PSA, heads, prototypes, and classification;
- same checker/ORT/parity harness as bridge export;
- differential graph-output tests against bridge artifacts.

Do not remove the bridge until native output passes the full matrix. The bridge remains a valuable independent oracle.

## 40. Implementation phases

### Phase A — contracts and environment preflight

- add export profile/input/output schemas;
- add exhaustive `ExportSpec` registry;
- add Python environment/source resolution;
- add bridge/final manifest types;
- document setup and no-network policy.

Exit gate: every current `ModelId` resolves to a complete spec, and the tool detects the currently missing ONNX dependency clearly.

### Phase B — parameter snapshot and reversible mappings

- collect loaded tensors;
- write SafeTensors with Burn-to-PyTorch adaptation;
- implement typed reverse key maps;
- produce tensor audit report;
- round-trip official state dicts PyTorch -> Burn -> SafeTensors -> PyTorch and compare every tensor.

Exit gate: zero missing/duplicate/unexpected parameter keys for one representative of every family/task.

### Phase C — classification first

- construct YOLO11/26 classification graphs;
- export logits/probabilities portable profile;
- add checker/ORT/parity;
- validate all classification scales.

Exit gate: top-5 and full probability parity for all ten registered classification variants.

### Phase D — classic detection

- YOLOv3-Tiny-U and YOLO11 adapters;
- portable boxes/scores outputs;
- Ultralytics packed profile;
- decoded tensor and final detection parity.

Exit gate: all classic detect variants pass.

### Phase E — YOLOX

- refactor official source shim;
- reverse YOLOX key mapping;
- portable YOLOX output;
- postprocessing parity;
- all six scales.

Exit gate: exact tensor inventory and reference-image detection parity.

### Phase F — end-to-end detection

- YOLOv10 and YOLO26 portable candidate outputs;
- Ultralytics top-k profile;
- no-NMS postprocess parity;
- near-tie deterministic tests.

Exit gate: all eleven variants pass both portable and compatibility profiles.

### Phase G — segmentation

- YOLO11 segment outputs;
- YOLO26 Segment26/Proto26 outputs;
- compatibility packed outputs;
- mask assembly/end-to-end IoU tests.

Exit gate: every registered segmentation variant passes box and mask gates.

### Phase H — shape/precision/options

- dynamic batch;
- dynamic spatial opt-in;
- external data;
- FP16;
- optional simplification;
- sidecar/metadata completeness.

Exit gate: each option passes its declared runtime matrix and cannot bypass baseline FP32 validation.

### Phase I — release hardening

- all-variant export run;
- Windows/Linux checks;
- artifact reproducibility;
- documentation and examples;
- license/NOTICE audit;
- performance/size report.

## 41. Pull-request slicing

Recommended review units:

1. contracts/spec/manifest/preflight;
2. snapshot plus one bidirectional key map;
3. classification export end to end;
4. classic detection;
5. YOLOX;
6. YOLOv10/26 end-to-end detection;
7. segmentation;
8. dynamic shapes/FP16/external data;
9. documentation/release matrix.

Every family PR must include its Rust-to-PyTorch tensor audit and ORT parity. Do not merge graph generation based only on successful `onnx.checker` output.

## 42. Definition of done

- [ ] One documented `export-onnx` command creates a validated artifact.
- [ ] Every registered `ModelId` has exactly one `ExportSpec`.
- [ ] All 40 current variants export in required FP32 portable profile.
- [ ] Ultralytics-family models support the pinned-compatible profile.
- [ ] Exported parameters originate from the loaded boquilens model/checkpoint.
- [ ] Future Burn-trained/EMA checkpoint selection is represented and tested.
- [ ] Rust-to-PyTorch key mapping is strict, reversible, collision-free, and versioned.
- [ ] The exporter proves it imported the pinned sibling/source graph.
- [ ] Export performs no hidden download or package installation.
- [ ] Input layout/color/range/shape is explicit.
- [ ] Output names, shapes, box formats, score semantics, and mask semantics are explicit.
- [ ] ONNX checker passes.
- [ ] Strict shape/type inference passes or a documented dynamic limitation is asserted.
- [ ] ONNX Runtime CPU loads and executes every FP32 artifact.
- [ ] Three-way Burn/PyTorch/ORT parity passes.
- [ ] End-to-end detection and mask parity pass on the reference image.
- [ ] Metadata and sidecar contain model, preprocessing, outputs, versions, hashes, and licensing.
- [ ] No unexpected/custom operator domains occur in portable graphs.
- [ ] Output publication is atomic and overwrite-safe.
- [ ] Dynamic batch/spatial flags fail clearly until their own gates pass.
- [ ] FP16 cannot be published without separate GPU/runtime parity.
- [ ] Existing prediction, packing, training, and inference behavior is unchanged.
- [ ] Project format/test/clippy/no-default-features commands pass.

## 43. High-risk traps

- Burn ONNX tooling imports; it does not export the current Rust model graph.
- The installed Python `ultralytics` version may differ from the sibling vendored tree.
- A successfully loaded ONNX protobuf can still be numerically wrong.
- A strict state dict load can still be wrong if two keys were swapped but have equal shapes.
- Linear weights require Burn-to-PyTorch transposition; Conv weights do not.
- BatchNorm `gamma/beta` names and epsilon flavors differ by path.
- YOLO11 boxes are XYWH while YOLOv10/26 portable boxes are XYXY.
- YOLOX confidence is objectness times class probability.
- YOLO10/26 are NMS-free; adding classic NMS changes behavior.
- YOLO26 is DFL-free; adding a 16-bin projection changes boxes.
- YOLO11/YOLOv3-Tiny-U use DFL.
- Segmentation coefficients are raw and must not receive sigmoid.
- YOLO11 and YOLO26 prototype graphs are different.
- End-to-end top-k output is not the same as confidence-filtered final detections.
- Source-image coordinate mapping requires preprocessing geometry and should remain outside the default graph.
- Classification inference output is softmax probabilities, but portable export should retain logits too.
- Dynamic tracing can bake batch `1` into reshape constants.
- Non-square dynamic input changes anchor and prototype dimensions.
- ONNX IR and opset versions are separate compatibility concerns.
- Changing only the IR version integer does not rewrite unsupported model features.
- FP16 CPU runtime support is incomplete.
- Simplification can change outputs and must be revalidated.
- External-data paths must remain relative and travel with the ONNX file.
- Timestamps make otherwise identical artifacts hash differently.
- New model directories are not export scope until the variants enter `ModelId`, but registry coverage must catch them immediately when they do.

## 44. Final recommendation

Implement the SafeTensors/Python graph bridge first and make three-way numerical parity the release gate. It is the shortest path that exports **the actual boquilens weights**, works for future Burn-trained checkpoints, uses proven upstream graph definitions, and avoids pretending Burn already has ONNX export. Keep the default graph portable: float32 RGB NCHW input, named decoded outputs, and task postprocessing outside ONNX. Add Ultralytics-compatible packed/top-k layouts as an explicit profile, then dynamic shapes, FP16, and a pure-Rust emitter only after the fixed FP32 contract is stable across every registered variant.
