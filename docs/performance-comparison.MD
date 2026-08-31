# Native training performance

Montgomery native training is competitive today, but not uniformly faster. On the tested RTX 5080,
it wins **22 of 30** comparisons and **11 of 12** smallest-variant family/task comparisons,
including every tested classification model and the detection-only YOLOv3-Tiny-U, YOLOv10n, and
YOLO12n models. YOLOv8n segmentation is **2.00× faster** than Ultralytics. YOLO26n detection is
ahead at batch 2; segmentation is effectively tied but 0.7% behind in this run.

![Median wall time for the nine family and task comparisons](assets/family-task-overview.png)

## At a glance

These are medians of complete training commands. A speedup above `1.00×` favors Montgomery.

| Model | Task | Montgomery | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: |
| YOLOv8n | Classify | 3.51 s | 14.34 s | **4.09×** |
| YOLOv8n | Detect | 5.53 s | 12.72 s | **2.30×** |
| YOLOv8n | Segment | 6.76 s | 13.49 s | **2.00×** |
| YOLO11n | Classify | 4.73 s | 12.52 s | **2.65×** |
| YOLO11n | Detect | 8.61 s | 13.48 s | **1.57×** |
| YOLO11n | Segment | 10.14 s | 14.08 s | **1.39×** |
| YOLO26n | Classify | 4.70 s | 12.80 s | **2.72×** |
| YOLO26n | Detect | 14.19 s | 15.32 s | **1.08×** |
| YOLO26n | Segment | 16.84 s | 16.73 s | 0.99× |

The detect-only families use the same dataset and settings:

| Model | Montgomery | Ultralytics | Native speedup |
| --- | ---: | ---: | ---: |
| YOLOv3-Tiny-U | 4.01 s | 12.37 s | **3.09×** |
| YOLOv10n | 13.73 s | 14.78 s | **1.08×** |
| YOLO12n | 14.52 s | 14.70 s | **1.01×** |

![Speedup heatmap for model families and tasks](assets/task-speedup-heatmap.png)

## Segmentation optimization

The original mask loss created a separate slice, matrix multiplication, BCE, and reduction graph
for every positive match. It now groups positives per image and evaluates one padded batched matrix
multiplication. The same change removes repeated prototype tensors. YOLO26's identical one-to-many
and detached one-to-one semantic targets are evaluated once.

Vectorizing the mask loss reduced the same 48-step YOLOv8n-seg command from **98.08 s to 9.64 s**.
The runtime and checkpoint work described below reduced the current full-command median again to
**6.76 s**, which is **2.00× faster** than Ultralytics' 13.49 s median.

![YOLOv8n segmentation before and after vectorizing the mask loss](assets/segmentation-before-after.png)

The correctness pass also aligned mask-crop lower bounds with Ultralytics and applied the shared
segmentation gain to the YOLO26 semantic term. Unit tests cover padded grouping, scalar loss parity,
backpropagation, semantic-value equality, and detached gradients.

## Runtime and checkpoint optimization

Burn/WGPU's grouped-convolution backward falls through a generic path. The light heads now use an
exact channelwise 3×3 stencil only while building an autodiff graph; normal inference and stored
parameters are unchanged. Forward values and weight gradients are tested against grouped
convolution. Together with flattened role-separated AdamW updates, this cut YOLO26n detection from
23.76 s to 14.19 s and YOLO11n segmentation from 17.38 s to 10.14 s.

Full resumable checkpoints still contain the model, EMA, and optimizer in FP32. Their independent
immutable snapshots are downloaded and encoded concurrently instead of making three serial
GPU-to-host passes, and hashing plus atomic file publication overlap the next epoch on a bounded
worker. Assignment targets stay on the host instead of being uploaded and immediately downloaded
each step. AdamW moment tensors are stored as two flat parameter-role groups rather than one record
per tensor. Resume, atomic-publication, and step-for-step optimizer parity are tested.

## Scale comparison

YOLO26 classification stays ahead through medium. Detection wins at nano and small and is 1.6%
slower at medium. Segmentation wins narrowly at small but remains the largest medium-scale gap.

| Model | Task | Montgomery | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: |
| YOLO26n | Classify | 4.70 s | 12.80 s | **2.72×** |
| YOLO26n | Detect | 14.19 s | 15.32 s | **1.08×** |
| YOLO26n | Segment | 16.84 s | 16.73 s | 0.99× |
| YOLO26s | Classify | 5.11 s | 13.21 s | **2.58×** |
| YOLO26s | Detect | 14.48 s | 15.86 s | **1.10×** |
| YOLO26s | Segment | 17.69 s | 17.94 s | **1.01×** |
| YOLO26m | Classify | 7.51 s | 14.19 s | **1.89×** |
| YOLO26m | Detect | 17.82 s | 17.55 s | 0.98× |
| YOLO26m | Segment | 24.47 s | 19.59 s | 0.80× |

![YOLO26 scale comparison](assets/model-scale.png)

## Batch scaling

The batch sweep uses YOLO26n and changes only batch size. Native wins all three tasks at batch 4;
batch 1 exposes fixed WGPU dispatch and synchronization costs.

| Task | Batch | Montgomery | Ultralytics | Native speedup |
| --- | ---: | ---: | ---: | ---: |
| Classify | 1 | 7.84 s | 14.18 s | **1.81×** |
| Classify | 2 | 4.70 s | 12.80 s | **2.72×** |
| Classify | 4 | 3.02 s | 12.24 s | **4.05×** |
| Detect | 1 | 22.97 s | 18.77 s | 0.82× |
| Detect | 2 | 14.19 s | 15.32 s | **1.08×** |
| Detect | 4 | 8.69 s | 13.22 s | **1.52×** |
| Segment | 1 | 28.55 s | 23.78 s | 0.83× |
| Segment | 2 | 16.84 s | 16.73 s | 0.99× |
| Segment | 4 | 11.31 s | 14.33 s | **1.27×** |

![Batch-size scaling](assets/batch-scaling.png)

## Resolution scaling

This sweep uses YOLO26n detection, batch 2, and three epochs.

| Image size | Montgomery | Ultralytics | Native speedup |
| ---: | ---: | ---: | ---: |
| 64 px | 14.18 s | 15.15 s | **1.07×** |
| 128 px | 13.42 s | 14.82 s | **1.10×** |
| 320 px | 14.19 s | 15.32 s | **1.08×** |
| 640 px | 17.55 s | 15.20 s | 0.87× |

![YOLO26n detection resolution scaling](assets/resolution-scaling.png)

The near-flat Ultralytics results show that framework and process overhead dominate these deliberately
small jobs. Native wins through 320 px; 640 px still exposes backend work to do.

## Ten-epoch smoke runs

The longer runs exercise checkpointing, scheduling, loss reduction, and validation. They are still
too small to measure real model quality.

| Task | Montgomery | Ultralytics | Native speedup |
| --- | ---: | ---: | ---: |
| Classify | 16.21 s | 23.54 s | **1.45×** |
| Detect | 43.07 s | 32.10 s | 0.75× |
| Segment | 52.12 s | 35.95 s | 0.69× |

![Normalized loss over ten epochs](assets/convergence-normalized.png)

Loss magnitudes are normalized in the chart because the trainers report and reduce components
differently. Both losses stay finite and decrease, but this is a smoke signal rather than a claim of
equal optimization dynamics.

Post-training validation is outside the timed region:

| Task | Montgomery | Ultralytics | Interpretation |
| --- | --- | --- | --- |
| Classify | top-1 41.7%, top-5 58.3% | top-1 41.7%, top-5 91.7% | 12 validation images; not quality evidence |
| Detect | mAP50 0.0, mAP50-95 0.0 | mAP50 0.0, mAP50-95 0.0 | 4 validation images; the run is too short |
| Segment | box/mask mAP 0.0 | box/mask mAP 0.0 | 4 validation images; the run is too short |

The benchmark proves that both implementations train, checkpoint, and validate the tested tasks. It
does **not** establish COCO or ImageNet convergence parity. That requires full datasets and standard
training schedules.

## All measured scenarios

![All 30 benchmark scenarios](assets/all-scenarios-speedup.png)

![Effective end-to-end throughput](assets/effective-throughput.png)

![Per-trial repeatability](assets/trial-repeatability.png)

![First measured run versus warm runs](assets/first-vs-warm.png)

Per-trial values, medians, environment metadata, and captured commands are preserved in the
[raw benchmark results](assets/training-benchmark-results.json).

## Methodology

- Hardware: NVIDIA GeForce RTX 5080; native Burn/WGPU used Vulkan driver 610.47.
- Software: Montgomery/Burn release build versus Ultralytics 8.4.117 and PyTorch 2.11 CUDA 13.0.
- Timing: external wall clock around the complete command, including startup, model loading,
  training, and checkpoint writes. Validation is excluded and recorded separately.
- Precision: FP32 for both implementations.
- Optimizer: AdamW, `lr0=0.001`, `lrf=0.05`, weight decay `0.0005`, momentum `0.9`, cosine decay,
  and no warmup. Nominal batch size equals actual batch size.
- Data loading: four workers; native prefetch depth two.
- Sampling: seed zero, alternating framework order, three trials normally and two for expensive
  segmentation cells.
- Inputs: the same real source fixtures, deterministically replicated with hard links. Classification
  uses 48 train/12 validation images; detection and segmentation use 32 train/4 validation images.
- Weights: the same upstream `.pt` checkpoint supplies both runs; native uses its tensor-only export.
- YOLOX is not in this matrix because Ultralytics does not provide a YOLOX training implementation;
  comparing it with the separate official YOLOX trainer would not be a 1:1 Ultralytics comparison.
- Warmup: native WGPU compilation is primed outside the measured trials. The first-vs-warm plot still
  exposes residual first-run effects.
- Augmentation policies and optimizer settings are matched where both APIs expose them. Random-number
  generators, scheduler details, and some loss reductions are not bit-identical.

This methodology emphasizes repeatable developer feedback and short-job latency. It is not
steady-state multi-hour throughput or a cross-framework quality leaderboard.

## Reproduce it

The complete comparison is one command after installing the pinned GPU benchmark environment and
preparing the upstream checkpoints:

```console
uv run --locked tools/benchmark_training.py --publish
```

It builds Montgomery in release mode, prepares deterministic datasets, primes WGPU, runs the matrix,
validates the ten-epoch checkpoints, renders every chart, and publishes JSON and images under
`docs/assets/`.

For a quick iteration on one bottleneck:

```console
uv run --locked tools/benchmark_training.py --scenario yolov8n-segment --repeats 1 --segmentation-repeats 1
```

Use `--list` on `tools/bench_training_matrix.py` to see every scenario. Runs can be resumed, and full
checkpoints can be retained with `--keep-runs`; they are deleted by default to keep output small.

## Follow-up status

- WGPU aggregation was swept at 8, 16, 32, 64, and 128 queued tasks across batch-1 detect/segment,
  640 px detect, and YOLO26m segment. The existing 32-task default was best overall; higher values
  consistently regressed, while apparent 16-task wins disappeared on repeat trials. Re-run with
  `uv run --locked tools/profile_wgpu_training.py`.
- Long runs now extract full-precision model/EMA/optimizer records only on a fitness improvement,
  every `--save-period` epochs (10 by default), at early stop, and at completion. Publication still
  overlaps the next epoch, and only `best`/`last` remain.
- The same profile isolates medium segmentation: backward is about 54–57% of measured GPU step
  time. Mask/semantic ablations and alternate depthwise kernels did not clear the 5% acceptance
  gate, so no correctness-risking shortcut was retained.
- A strict full-COCO runner now rejects partial datasets and executes matched detect/segment
  training plus validation: `uv run --locked tools/bench_full_convergence.py --download`. Full
  convergence results are still required before any quality-parity claim.
