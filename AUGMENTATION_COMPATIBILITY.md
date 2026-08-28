# Native augmentation compatibility

The `training` feature contains a CPU-side, Python-free augmentation subsystem targeting
Ultralytics `v8.4.117-2-g461196cf0`, commit
`461196cf09175b64c9b9bd8babebf081c0540520`. Detection and segmentation stay as HWC BGR bytes
until formatting; classification converts to RGB before its torchvision-compatible policy.
The classification fixture oracle is pinned to Python 3.11.15, PyTorch 2.13.0+cpu,
torchvision 0.28.0+cpu (release tag v0.28.0), Pillow 12.3.0, and OpenCV 5.0.0.

Implemented contracts include validated task defaults, worker-independent ChaCha12 streams,
versioned traces, continuous-edge boxes and polygons, validation LetterBox, formatting and mask
targets, affine/perspective geometry, HSV, flips, the four default photometric operations,
Mosaic-4/9, MixUp, detector CutMix, both CopyPaste modes, classification RandomResizedCrop,
RandAugment, ColorJitter fallback, normalization and RandomErasing. Rect mode disables the three
upstream-disabled mixed transforms; close-mosaic phase and stride-rounded multi-scale shape
calculation are persisted/exposed by training state and batch contracts.

`autoaugment` and `augmix` are deliberately rejected during config resolution. They are not
advertised as compatible and are never silently ignored. Arbitrary Python/Albumentations objects
are likewise outside the typed native API.

## Determinism and parity

Native seed output is stable for `(run seed, epoch, logical position, dataset index, rank,
transform path)` and does not promise Python seed equivalence. Cross-language parity uses injected
parameters and fixture traces. `tools/export_augmentation_fixtures.py` verifies the sibling source
pin and writes fixtures under `target/augmentation-fixtures/` with environment versions and hashes.

The resize and warp kernels are intentionally separate from inference preprocessing. Their pixel
center and border conventions are unit tested, while codec-dependent JPEG and CLAHE comparisons
must use bounded-pixel tolerances recorded in fixture manifests.

## Dependency and performance notes

No OpenCV, Pillow, torchvision, PyTorch, or Python dependency is linked into Rust training.
`rand`/`rand_chacha` were already part of the feature-gated training dependency set and are
Apache-2.0/MIT. Beta sampling is implemented from the pinned ChaCha stream to avoid another
distribution dependency. OpenCV-backed Rust bindings and general polygon crates were rejected as
normal dependencies because of system-library deployment and unproven raster parity respectively.

Formatted detector images remain `u8` through worker-side processing and are converted to `[0,1]`
at collation. Cache samples are cloned before mutation. The deterministic partner pool caps recent
history at `min(dataset size, batch_size * 8, 1000)`. Hardware-specific data-wait benchmarks remain
a run-level measurement; no universal throughput number is asserted.
