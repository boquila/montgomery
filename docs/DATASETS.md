# Dataset CLI

`montgomery data` provides offline inspection, label conversion, and bulk image resizing. Transformations
require exactly one of `--out-dir DIRECTORY` (write a new dataset) or `--override` (replace the source).
It needs
no model, GPU, Python, or `training` feature. JPEG, PNG, and WebP inputs are supported.

## Inspect

```console
montgomery data --path dataset.yaml
montgomery data inspect --path dataset.yaml --json
montgomery data inspect --path dataset.yaml --verify-images --check
```

`data --path` defaults to inspection. `info` is an alias for `inspect`, and `--data` is an alias for
`--path`. These options, `--json`, and `--workers` work before or after the subcommand.

The report includes:

- Images, unique image paths, object labels, and image storage in bytes.
- Train/validation/test counts, backgrounds, missing and empty YOLO label files.
- Per-class image and label counts, including classes with no examples.
- Bounding box, polygon, RLE mask, or classification label counts.
- Image dimension frequencies and invalid images/annotations with paths and line numbers.
- Images shared between splits, which can leak training data into evaluation.

Inspection reads image headers by default, without decoding every pixel. `--verify-images` decodes
pixels to find damaged images. `--check` returns a failing status for invalid data or shared paths
between splits. Missing YOLO label files are legal backgrounds and produce a warning; they alone
do not fail the check. Counts of labels and image sizes exclude invalid records. This is a path-based
leakage check, not a search for visually identical images stored under different paths.

## Convert labels

### Instance segmentation to detection

```console
montgomery data convert --path segments/data.yaml --to detect --out-dir boxes
```

YOLO polygons become their tight enclosing boxes; existing box rows retain their boxes. COCO
objects use their declared `bbox`, including RLE masks. Output labels use YOLO normalized
`class cx cy width height`. Images are copied byte-for-byte, including background images.
Class ordering and split membership are preserved. Mask information is intentionally discarded.
Crop, resizing, and JPEG encoding options apply to classification conversion; detection conversion
rejects those options when they would have no effect on copied images.

### Detection to classification

```console
montgomery data convert --path boxes/data.yaml --to classify --out-dir crops
montgomery data convert --path boxes/data.yaml --to classify --out-dir crops-small --imgsz 224
montgomery data convert --path boxes/data.yaml --to classify --out-dir crops-context --padding 0.1
```

Each object produces a JPEG crop under `<split>/<class-name>/`. The crop encloses the continuous
box using floor/ceil pixel edges and clips to the image boundary. Multiple objects in one image
produce separate crops. YOLO polygons and COCO masks can go directly to classification using their
boxes, so an intermediate detection dataset is optional.

The first command is the complete conversion: it crops every object, organizes the JPEGs into
`crops/train/<class>/`, `crops/val/<class>/`, and `crops/test/<class>/` for the source splits that
contain objects, and writes `crops/data.yaml`. Use that YAML directly in subsequent data or
classification training commands; no manual cropping, sorting, or YAML editing is needed.

`--padding 0.1` adds 10% of the box width/height on each side, clipped at the image edges.
Background images produce no crops. COCO crowd objects are skipped unless `--include-crowd` is
specified; the result reports both skip counts. An empty output training split is an error.
Validation/test splits containing only backgrounds or excluded crowds are omitted from the output;
if no validation images remain, train with `--no-val` or provide a labeled validation split first.
Empty class directories are created to keep the class table consistent. Class names must be valid,
portable folder names for classification; unsafe names produce an actionable error.
Names that differ only in letter case are rejected to prevent class folders merging on Windows
or other case-insensitive filesystems.

Without `--imgsz`, crops retain their original pixel dimensions. With it, the same resize controls
below apply. Every crop retains the source image's split, avoiding new train/validation leakage.

## Resize images before training

```console
# Reduce 4K photos to an aspect-ratio-preserving bound of 224×224.
montgomery data resize --path classes/data.yaml --imgsz 224 --out-dir classes-small

# Produce exactly 224×224 while preserving the whole image with padding.
montgomery data resize --path classes/data.yaml --imgsz 224 --mode letterbox --out-dir classes-square

# Replace the source dataset, retaining its original YAML filename and a recovery backup.
montgomery data resize --path classes/data.yaml --imgsz 224 --override

# Produce an exact rectangular size by changing aspect ratio.
montgomery data resize --path dataset/data.yaml --imgsz 640x384 --mode stretch --out-dir resized
```

`--imgsz N` means an N×N target; `--imgsz WIDTHxHEIGHT` supports rectangular bounds.

| Mode | Result | Small images |
| --- | --- | --- |
| `fit` (default) | Fit within bounds, preserve aspect ratio, no padding | Keep native size |
| `letterbox` | Exact target canvas, centered padding in RGB 114 | Pad without enlarging |
| `stretch` | Exact target dimensions, change aspect ratio | Resize to target |

`--upscale` allows fit/letterbox to enlarge small images. Dataset resizing is an offline storage
operation; the model's normal preprocessing still applies during training and inference.

Resized images and classification crops use RGB JPEG at quality 90. `--quality 1..100` controls
JPEG quality. Resampling uses the existing SIMD image-resize dependency with an anti-aliased
bilinear filter. Alpha channels and EXIF metadata are not preserved in transformed JPEGs;
geometry uses the decoded pixel grid without EXIF rotation. Detection conversion copies original
files instead of re-encoding them.

YOLO box/polygon coordinates are preserved for fit/stretch and remapped for padding. COCO resize
retains COCO format, updates dimensions, boxes, polygons, and areas, and resizes RLE masks with
nearest-neighbor sampling. Both compressed and uncompressed input RLEs are accepted; resized RLEs
use uncompressed counts. Very small masks can disappear when reduced to a coarse pixel grid.

## Outputs, performance, and scripting

```console
montgomery data resize --path dataset.yaml --imgsz 640 --override --dry-run --json
montgomery data convert --path dataset/data.yaml --to classify --out-dir crops --workers 8 --json
montgomery data --path crops/data.yaml
```

There is no implicit output destination. `--out-dir DIRECTORY` preserves the source dataset and writes
a new one outside the source root; existing directories are refused even if empty. `--override` and
`--out-dir` are mutually exclusive. The old `--output` flag is not accepted for dataset operations.

`--override` replaces the dataset directory only after all images and annotations have been processed
successfully. The original directory is renamed to a sibling `<dataset>.backup-<timestamp>` recovery
backup, whose path is printed and included in the JSON report. This retains original files (including
unrelated files in that directory) in the backup. The new directory contains the transformed dataset;
its YAML keeps the original filename. Remove the backup yourself when you no longer need it.
To recover, move the replacement directory aside and rename the backup to the original directory
name. Backups preserve the original YAML byte-for-byte; manifests containing absolute paths may
only resolve correctly after restoring that original location. Override provenance records point
`source` to the retained backup files and keep their former paths as `original_source`.
Override requires the YAML directly inside a dedicated dataset root and all images/annotations inside
that root. External/nested manifests, external image/label references, filesystem roots, and Git
repository roots require `--out-dir` instead.

Both modes write into a private staging directory and publish only after success. Processing errors
remove staging files and leave the source unchanged; publication failure attempts to restore the
original directory and reports the recovery path if restoration fails. `--dry-run` validates and
reports planned counts and paths without writing files; it does not verify pixel decoding or predict
compressed output sizes.

Each output contains:

- `data.yaml` (or the original YAML filename with `--override`): a self-contained, relative manifest
  ready for the next data or training command.
- Images and labels/annotations in the appropriate YOLO, COCO, or classification layout.
- `sources.jsonl`: output-to-input mapping, with source geometry and crop coordinates.
- `transform.json`: operation counts, image-byte totals, skip counts, and output paths.

The default worker count uses available cores, capped at four to bound decoded-image memory.
`--workers` overrides it (1–256). Each classification source is decoded once for all its crops.
Image processing is concurrent; output names and report ordering are deterministic regardless of
worker completion order. Annotation indexing is held in memory; full decoded images are bounded
by worker count. Use a release build for large datasets.

## Manifests

Relative `path` resolves against the YAML directory. Relative split paths resolve against `path`;
relative entries in image-list files resolve against the list file's directory. Splits can be
directories, individual images, text lists, or lists of those paths. Directory discovery is recursive
and deduplicates canonical image paths within each split. Files named `images/.../a.jpg` map to
`labels/.../a.txt`; otherwise YOLO labels sit beside the image with a `.txt` extension.

```yaml
path: .
train: images/train
val: images/val
test: images/test
names: [person, bicycle]
format: yolo
```

`names` can also be a zero-based contiguous ID map. `nc`, when present, must match its length.
`val` and `test` are optional; `train` must contain supported images. Format can usually be inferred;
set it explicitly when the directory layout is ambiguous.

```yaml
# Class folders: train/person/*.jpg, train/bicycle/*.jpg, etc.
path: .
train: train
val: val
format: classification-folders
names: [person, bicycle]
```

Class folders can infer names from the configured training directories when the table is omitted;
the directories need not be called `train`. Empty class folders remain in the class table. For
individual-image or image-list classification splits, provide `names` explicitly.

```yaml
path: .
train: images/train
val: images/val
format: coco
train_annotations: annotations/train.json
val_annotations: annotations/val.json
names: [person, bicycle]
```

COCO manifests need annotations for each nonempty split. Category names ordered by category ID
must match `names`; IDs need not be contiguous. Nested `file_name` paths are supported, and
ambiguous basename matches are errors. Remote downloads, pose/keypoint, oriented-box, semantic,
and depth labels are outside these commands' scope. Unsupported task declarations are rejected.
