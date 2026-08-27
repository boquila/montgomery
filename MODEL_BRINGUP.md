# Model bring-up guide

How to add a new Ultralytics-family detector (a new family or a new scale variant) to boquilens as
a native Burn implementation. Written for agent sessions: follow the phases in order; every step
exists because skipping it has already caused a bug once.

Orientation first:

- boquilens runs inference natively in Rust/Burn. Python + PyTorch + the `ultralytics` package are
  conversion-time only; the environment for that is the venv at `target/.venv`.
- The architecture reference is the vendored repo at `../ultralytics` (pinned 8.4.117). The venv's
  installed package may be a slightly newer 8.4.x. When the two disagree, the **official checkpoint
  wins** — it was produced by whichever version trained it.
- Copy an existing module as the template: `src/models/yolov10/` (DFL head, one2one top-k), or
  `src/models/yolo26/` (DFL-free end-to-end), or `src/models/yolov3_tiny/` (classic NMS path).
- Inference modes in the wild: one2one end-to-end heads are top-k selected + confidence filtered
  (`end2end_topk_detections` in `src/lib.rs`); plain heads go through class-aware NMS.

## 1. Identify the architecture

1. Read the YAML at `ultralytics/cfg/models/<family>/<model>.yaml`. Note `nc`, `reg_max`,
   `end2end`, and the scale row (depth, width, max_channels) for the variant you are porting.
2. Read `parse_model` in `ultralytics/nn/tasks.py` — it rewrites the YAML args, and getting this
   wrong silently produces a wrong graph:
   - channels scale as `make_divisible(min(c, max_channels) * width, 8)`;
   - repeat modules get the depth-scaled repeat count inserted at args position 2 (`n = 1` after);
     SPPF is **not** a repeat module and passes its own `k/n/shortcut` args;
   - `C3k2` forces `c3k=True` for m/l/x scales;
   - `Detect` receives `(nc, reg_max, end2end, ch)`; `legacy` flips when a modern module
     (C3k2/C2fCIB/...) appears, switching `cv3` between the old 3x3 tower and the light
     DWConv/Conv tower.
3. Read the exact class sources in `ultralytics/nn/modules/{block,conv,head}.py` for every module
   in the graph — signatures drift between releases (SPPF gained `n` and `shortcut`; C3k2 gained
   `attn`; C2PSA gained an `m.0` wrapper in 8.3.0). Diff vendored vs installed when in doubt.
4. Derive the per-layer channel table and the head flavor: box hidden width is
   `max(16, ch[0] // 4, reg_max * 4)`, cls width is `max(ch[0], min(nc, 100))`.

## 2. Get ground truth from the checkpoint

1. Put the official `.pt` under `target/` (external AGPL asset; never commit it) and dump the
   state dict keys and shapes:

   ```python
   import torch, re
   state = torch.load("target/<id>.pt", map_location="cpu", weights_only=False)["model"].state_dict()
   for k, v in state.items(): print(k, tuple(v.shape))
   ```

2. Verify every layer index and channel count against your step-1 table. Shapes expose things the
   YAML cannot: `Bottleneck` defaults to `e=0.5` (half-width) but C3k's inner chain passes
   `e=1.0`; attention wrappers appear as `m.0.attn.*`; Sequential indices become path segments.
3. Remember that released checkpoints can predate source refactors: `yolov10n.pt` has the old
   inlined C2PSA (`model.10.attn.*`) while 8.3+ builds `model.10.m.0.attn.*`. Always structure the
   Rust module around the checkpoint's keys, not the current source.

## 3. Implement `src/models/<id>/`

Create `blocks.rs`, `body.rs`, `head.rs`, `model.rs`, `weights.rs`, `mod.rs` following the
existing templates. Hard rules:

- **Field names are the checkpoint keys.** After remapping, `model.22.m.0.1.attn.qkv.conv.weight`
  must land on `body.model_22.m.0.1.attn.qkv.conv.weight`. Burn serializes tuple fields as `0`/`1`
  and `Vec<(A, B)>` as `m.<i>.<j>` — use that; do **not** use enums (they prepend variant names and
  break key matching). Duplicate a small struct rather than inventing a clever generic.
- `Conv` = conv(bias=False) + BN(eps 1e-3, momentum 0.03) + SiLU unless the source says otherwise
  (YOLO26's SPPF `cv1` has no activation). Head output convs keep bias.
- Body fields are `model_<index>` for graph layers, `forward` returns P3/P4/P5 (or the model's
  scales), and there is a shape test with a 32–64 MB worker thread (deep module construction
  overflows the default stack on Windows in debug builds).
- Head: `make_anchors` uses 0.5-cell offsets and per-level strides; decode depends on `reg_max` —
  softmax/DFL projection when `reg_max > 1` (v10), direct left-top/right-bottom distances when
  `reg_max == 1` (v26); end-to-end heads decode XYXY.
- `model.rs` gates weight loading behind `#[cfg(feature = "pretrained")]`: `load_pytorch_weights`
  with regex remaps (body layers in one rule, head towers one rule per path segment pattern; only
  the inference branch of the head is mapped — the one2many branch is intentionally dropped),
  `load_burnpack_weights`/`save_burnpack_weights` with the `HalfPrecisionAdapter` and
  `boquilens.*` metadata, plus the ignored checkpoint-import and golden-tensor tests (2e-4
  tolerance, fixture JSON per step 5).
- `weights.rs`: `artifact_format(<id>)` returning `<id>-v1`, `coco_artifact_filename(<id>)` returning
  `<id>-coco-ultralytics-v<version>-boquilens-v1.bpk` (e.g. `v8.4`); fill bytes/SHA-256 constants
  after packing a verified artifact.
- Keep the graph independent of CLI, filesystem, rendering, and image decoding.

## 4. Wire into the crate

- `src/models/mod.rs`: add the module.
- `src/lib.rs`:
  - `ModelId` variant + `as_str` + `FromStr` aliases + unknown-model error text;
  - `RuntimeModel` variant;
  - `Predictor::new` (experimental models require `--weights`), `from_checkpoint` arm
    (`.bpk` → burnpack, else PyTorch state, inside a 64 MB-stack worker thread);
  - `predict` arms: `LetterboxedImage::ultralytics(image, 640, 32)` for stride-32 Ultralytics
    models, input `/ 255.0`, then the right decode path;
  - `pack_weights` arm + its error message;
  - the `parses_stable_model_names` and packer-rejection tests.
- `src/main.rs`: `--model` help text lists the new name.

## 5. Conversion tooling

- `tools/export_ultralytics_state.py` is model-agnostic; run it against the official `.pt`.
- Copy `tools/export_yolo26_fixtures.py` to `tools/export_<id>_fixtures.py` and adjust: the hooked
  body layer indices, the head index, the `preds["one2one"]` (or plain-tensor) access, and all
  `<id>` filenames. It writes the source/preprocessed reference PNGs and the golden JSON under
  `target/`.

## 6. Verify

```console
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo check --locked --no-default-features --lib
```

Then the parity loop (checkpoint and fixtures live under `target/`):

```console
& target\.venv\Scripts\python.exe tools\export_ultralytics_state.py target/<id>.pt target/<id>-state.pt
cargo run --release -- pack-weights --model <id> --input target/<id>-state.pt --output target/<id>-coco-ultralytics-<ver>-boquilens-v1.bpk
& target\.venv\Scripts\python.exe tools\export_<id>_fixtures.py target/<id>.pt assets/dog_bike_man.jpg target
cargo test --locked <id> -- --ignored
```

Both ignored tests must pass: `imports_official_checkpoint_and_runs_forward` (keys + remaps
correct) and `matches_ultralytics_golden_tensors` (graph numerics correct at 2e-4). If keys mismatch
the remapper silently skips them and the model runs with default weights — the golden test is what
catches this, so never skip it.

Finally compare end-to-end against Ultralytics on `assets/dog_bike_man.jpg` (`conf=0.25`, same
image, CPU): expect the same detections with confidences within ~0.1% and boxes within ~1 source
pixel (f16 artifact rounding accounts for the residual). Record the artifact bytes/SHA-256 in
`weights.rs`.

## 7. Document

- `README.md`: row in the Models table (family name in the Model column, variant in the Variants
  column), row in the artifacts table, adjust the weight-prep snippet if a new bridge step exists.
- `AGENTS.md`: "What is here" entry, fast-path commands, invariants (decode/postprocess specifics).
- `NOTICE`: provenance and licensing (Ultralytics-family weights and derived artifacts are
  AGPL-3.0; the YOLOX path is Apache-2.0).

## Pitfall checklist

- Channels: `make_divisible(min(c, max_channels) * width, 8)`, depth-scaled repeats, SPPF not
  being a repeat module, C3k2's m/l/x `c3k` override.
- Bottleneck widths: half-width default (`e=0.5`) vs explicit `e=1.0` chains — verify from shapes.
- Checkpoint key drift between the training-time release and current source (C2PSA `m.0`).
- Enum modules break key matching; tuples/Vecs serialize as `0/1`/index segments.
- One2many head keys are silently dropped — that is desired; missing one2one keys are not.
- BN epsilon is 1e-3 (Ultralytics initialization), not PyTorch's 1e-5.
- End-to-end heads: no NMS anywhere; top-k + confidence filter only.
- Generated checkpoints, states, fixtures, images, and `.bpk` artifacts stay under `target/`.
