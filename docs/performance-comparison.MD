# Native training performance

boquilens native training is competitive today, but not uniformly faster. On the tested RTX 5080,
it wins **6 of 9** nano family/task comparisons and every tested classification comparison.
YOLOv8n segmentation is **1.40× faster** than Ultralytics after vectorizing its mask loss.
YOLO26 detection and segmentation remain substantially slower and are the next optimization target.

![Median wall time for the nine family and task comparisons](assets/family-task-overview.png)

## At a glance

These are medians of complete training commands. A speedup above `1.00×` favors boquilens.

| Model | Task | boquilens | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: |
| YOLOv8n | Classify | 4.17 s | 13.79 s | **3.31×** |
| YOLOv8n | Detect | 8.59 s | 13.11 s | **1.53×** |
| YOLOv8n | Segment | 9.64 s | 13.48 s | **1.40×** |
| YOLO11n | Classify | 7.70 s | 13.98 s | **1.82×** |
| YOLO11n | Detect | 13.20 s | 14.71 s | **1.11×** |
| YOLO11n | Segment | 17.38 s | 14.71 s | 0.85× |
| YOLO26n | Classify | 7.00 s | 13.99 s | **2.00×** |
| YOLO26n | Detect | 23.76 s | 15.99 s | 0.67× |
| YOLO26n | Segment | 27.52 s | 17.67 s | 0.64× |

![Speedup heatmap for model families and tasks](assets/task-speedup-heatmap.png)

## Segmentation optimization

The original mask loss created a separate slice, matrix multiplication, BCE, and reduction graph
for every positive match. It now groups positives per image and evaluates one padded batched matrix
multiplication. The same change removes repeated prototype tensors. YOLO26's identical one-to-many
and detached one-to-one semantic targets are evaluated once.

That reduced the same 48-step YOLOv8n-seg command from **98.08 s to 9.64 s**, a **10.2×** native
improvement. The optimized command is **1.40× faster** than Ultralytics' 13.48 s median.

![YOLOv8n segmentation before and after vectorizing the mask loss](assets/segmentation-before-after.png)

The correctness pass also aligned mask-crop lower bounds with Ultralytics and applied the shared
segmentation gain to the YOLO26 semantic term. Unit tests cover padded grouping, scalar loss parity,
backpropagation, semantic-value equality, and detached gradients.

YOLO11n-seg and YOLO26n-seg are still slower. Their larger graphs expose remaining WGPU dispatch,
assignment, and dual-head overhead that the mask-loss vectorization does not remove.

## Scale comparison

YOLO26 classification stays ahead through medium. Detection and segmentation fall further behind as
the dual one-to-many/one-to-one training graph grows.

| Model | Task | boquilens | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: |
| YOLO26n | Classify | 7.00 s | 13.99 s | **2.00×** |
| YOLO26n | Detect | 23.76 s | 15.99 s | 0.67× |
| YOLO26n | Segment | 27.52 s | 17.67 s | 0.64× |
| YOLO26s | Classify | 8.35 s | 14.02 s | **1.68×** |
| YOLO26s | Detect | 30.75 s | 17.53 s | 0.57× |
| YOLO26s | Segment | 34.78 s | 19.65 s | 0.57× |
| YOLO26m | Classify | 10.97 s | 15.86 s | **1.45×** |
| YOLO26m | Detect | 45.00 s | 19.48 s | 0.43× |
| YOLO26m | Segment | 53.87 s | 23.09 s | 0.43× |

![YOLO26 scale comparison](assets/model-scale.png)

## Batch scaling

The batch sweep uses YOLO26n and changes only batch size. Native classification benefits strongly
from batching; detection and segmentation improve but remain behind.

| Task | Batch | boquilens | Ultralytics | Native speedup |
| --- | ---: | ---: | ---: | ---: |
| Classify | 1 | 11.46 s | 15.17 s | **1.32×** |
| Classify | 2 | 7.00 s | 13.99 s | **2.00×** |
| Classify | 4 | 4.66 s | 13.47 s | **2.89×** |
| Detect | 1 | 38.43 s | 20.11 s | 0.52× |
| Detect | 2 | 23.76 s | 15.99 s | 0.67× |
| Detect | 4 | 16.06 s | 14.78 s | 0.92× |
| Segment | 1 | 48.97 s | 25.58 s | 0.52× |
| Segment | 2 | 27.52 s | 17.67 s | 0.64× |
| Segment | 4 | 19.45 s | 15.72 s | 0.81× |

![Batch-size scaling](assets/batch-scaling.png)

## Resolution scaling

This sweep uses YOLO26n detection, batch 2, and three epochs.

| Image size | boquilens | Ultralytics | Native speedup |
| ---: | ---: | ---: | ---: |
| 64 px | 23.29 s | 16.31 s | 0.70× |
| 128 px | 20.99 s | 16.33 s | 0.78× |
| 320 px | 23.76 s | 15.99 s | 0.67× |
| 640 px | 40.05 s | 15.97 s | 0.40× |

![YOLO26n detection resolution scaling](assets/resolution-scaling.png)

The near-flat Ultralytics results show that framework and process overhead dominate these deliberately
small jobs. The 640 px native gap nevertheless identifies real graph/backend work to do.

## Ten-epoch smoke runs

The longer runs exercise checkpointing, scheduling, loss reduction, and validation. They are still
too small to measure real model quality.

| Task | boquilens | Ultralytics | Native speedup |
| --- | ---: | ---: | ---: |
| Classify | 23.37 s | 25.61 s | **1.10×** |
| Detect | 80.35 s | 34.39 s | 0.43× |
| Segment | 86.06 s | 36.90 s | 0.43× |

![Normalized loss over ten epochs](assets/convergence-normalized.png)

Loss magnitudes are normalized in the chart because the trainers report and reduce components
differently. Both losses stay finite and decrease, but this is a smoke signal rather than a claim of
equal optimization dynamics.

Post-training validation is outside the timed region:

| Task | boquilens | Ultralytics | Interpretation |
| --- | --- | --- | --- |
| Classify | top-1 41.7%, top-5 58.3% | top-1 41.7%, top-5 91.7% | 12 validation images; not quality evidence |
| Detect | mAP50 0.0, mAP50-95 0.0 | mAP50 0.0, mAP50-95 0.0 | 4 validation images; the run is too short |
| Segment | box/mask mAP 0.0 | box/mask mAP 0.0 | 4 validation images; the run is too short |

The benchmark proves that both implementations train, checkpoint, and validate the tested tasks. It
does **not** establish COCO or ImageNet convergence parity. That requires full datasets and standard
training schedules.

## All measured scenarios

![All 27 benchmark scenarios](assets/all-scenarios-speedup.png)

![Effective end-to-end throughput](assets/effective-throughput.png)

![Per-trial repeatability](assets/trial-repeatability.png)

![First measured run versus warm runs](assets/first-vs-warm.png)

The segmentation ten-epoch native run had one 128.11 s outlier; its other runs were 81.16 s and
86.06 s. The table reports the median. Per-trial values and captured commands are preserved in the
[raw benchmark results](assets/training-benchmark-results.json).

## Methodology

- Hardware: NVIDIA GeForce RTX 5080; native Burn/WGPU used Vulkan driver 610.47.
- Software: boquilens/Burn release build versus Ultralytics 8.4.117 and PyTorch 2.11 CUDA 13.0.
- Timing: external wall clock around the complete command, including startup, model loading,
  training, and checkpoint writes. Validation is excluded and recorded separately.
- Precision: FP32 for both implementations.
- Optimizer: AdamW, `lr0=0.001`, `lrf=0.05`, weight decay `0.0005`, momentum `0.9`, cosine decay,
  and no warmup. Nominal batch size equals actual batch size.
- Data loading: four workers; native prefetch depth two.
- Sampling: seed zero, alternating framework order, three trials normally and two for expensive
  segmentation cells. The ten-epoch segmentation cell has three trials after an additional pair.
- Inputs: the same real source fixtures, deterministically replicated with hard links. Classification
  uses 48 train/12 validation images; detection and segmentation use 32 train/4 validation images.
- Weights: the same upstream `.pt` checkpoint supplies both runs; native uses its tensor-only export.
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

It builds boquilens in release mode, prepares deterministic datasets, primes WGPU, runs the matrix,
validates the ten-epoch checkpoints, renders every chart, and publishes JSON and images under
`docs/assets/`.

For a quick iteration on one bottleneck:

```console
uv run --locked tools/benchmark_training.py --scenario yolov8n-segment --repeats 1 --segmentation-repeats 1
```

Use `--list` on `tools/bench_training_matrix.py` to see every scenario. Runs can be resumed, and full
checkpoints can be retained with `--keep-runs`; they are deleted by default to keep output small.

## What to optimize next

1. Profile YOLO26's one-to-many plus detached one-to-one training graph and eliminate duplicate
   compute before the heads.
2. Reduce WGPU dispatch and synchronization in TAL assignment and DFL/detection reductions.
3. Batch more segmentation crop and target work on-device, especially for YOLO11/YOLO26.
4. Add full-dataset convergence runs before making quality-parity claims.
