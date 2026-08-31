# Native training performance

boquilens native training is competitive today, but not uniformly faster. On the tested RTX 5080,
it wins **7 of 9** nano family/task comparisons and every tested classification comparison.
YOLOv8n segmentation is **1.65× faster** than Ultralytics. YOLO26n detection is within 8%; its
segmentation path is within 15% at batch 2 and faster at batch 4.

![Median wall time for the nine family and task comparisons](assets/family-task-overview.png)

## At a glance

These are medians of complete training commands. A speedup above `1.00×` favors boquilens.

| Model | Task | boquilens | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: |
| YOLOv8n | Classify | 3.36 s | 12.27 s | **3.65×** |
| YOLOv8n | Detect | 6.18 s | 12.07 s | **1.95×** |
| YOLOv8n | Segment | 7.75 s | 12.79 s | **1.65×** |
| YOLO11n | Classify | 4.98 s | 12.71 s | **2.55×** |
| YOLO11n | Detect | 10.33 s | 13.87 s | **1.34×** |
| YOLO11n | Segment | 11.58 s | 13.41 s | **1.16×** |
| YOLO26n | Classify | 4.92 s | 12.38 s | **2.52×** |
| YOLO26n | Detect | 15.80 s | 14.68 s | 0.93× |
| YOLO26n | Segment | 20.11 s | 17.51 s | 0.87× |

![Speedup heatmap for model families and tasks](assets/task-speedup-heatmap.png)

## Segmentation optimization

The original mask loss created a separate slice, matrix multiplication, BCE, and reduction graph
for every positive match. It now groups positives per image and evaluates one padded batched matrix
multiplication. The same change removes repeated prototype tensors. YOLO26's identical one-to-many
and detached one-to-one semantic targets are evaluated once.

Vectorizing the mask loss reduced the same 48-step YOLOv8n-seg command from **98.08 s to 9.64 s**.
The checkpoint work described below reduced the current full-command median again to **7.75 s**,
which is **1.65× faster** than Ultralytics' 12.79 s median.

![YOLOv8n segmentation before and after vectorizing the mask loss](assets/segmentation-before-after.png)

The correctness pass also aligned mask-crop lower bounds with Ultralytics and applied the shared
segmentation gain to the YOLO26 semantic term. Unit tests cover padded grouping, scalar loss parity,
backpropagation, semantic-value equality, and detached gradients.

## Runtime and checkpoint optimization

Burn/WGPU's grouped-convolution backward falls through a generic path. The light heads now use an
exact channelwise 3×3 stencil only while building an autodiff graph; normal inference and stored
parameters are unchanged. Forward values and weight gradients are tested against grouped
convolution. This cut YOLO26n detection from 23.76 s to 15.80 s and YOLO11n segmentation from
17.38 s to 11.58 s.

Full resumable checkpoints still contain the model, EMA, and optimizer in FP32. Their independent
immutable snapshots are now downloaded and encoded concurrently instead of making three serial
GPU-to-host passes. This benefits every task without weakening resume semantics.

## Scale comparison

YOLO26 classification stays ahead through medium. Detection is near parity through small and within
16% at medium. Segmentation remains the largest scale-dependent gap.

| Model | Task | boquilens | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: |
| YOLO26n | Classify | 4.92 s | 12.38 s | **2.52×** |
| YOLO26n | Detect | 15.80 s | 14.68 s | 0.93× |
| YOLO26n | Segment | 20.11 s | 17.51 s | 0.87× |
| YOLO26s | Classify | 5.70 s | 13.65 s | **2.39×** |
| YOLO26s | Detect | 17.27 s | 16.76 s | 0.97× |
| YOLO26s | Segment | 21.57 s | 18.62 s | 0.86× |
| YOLO26m | Classify | 8.40 s | 14.46 s | **1.72×** |
| YOLO26m | Detect | 21.13 s | 18.27 s | 0.86× |
| YOLO26m | Segment | 28.63 s | 21.80 s | 0.76× |

![YOLO26 scale comparison](assets/model-scale.png)

## Batch scaling

The batch sweep uses YOLO26n and changes only batch size. Native wins all three tasks at batch 4;
batch 1 exposes fixed WGPU dispatch and synchronization costs.

| Task | Batch | boquilens | Ultralytics | Native speedup |
| --- | ---: | ---: | ---: | ---: |
| Classify | 1 | 9.38 s | 14.55 s | **1.55×** |
| Classify | 2 | 4.92 s | 12.38 s | **2.52×** |
| Classify | 4 | 3.35 s | 12.45 s | **3.71×** |
| Detect | 1 | 26.26 s | 19.32 s | 0.74× |
| Detect | 2 | 15.80 s | 14.68 s | 0.93× |
| Detect | 4 | 10.41 s | 13.53 s | **1.30×** |
| Segment | 1 | 34.01 s | 23.98 s | 0.71× |
| Segment | 2 | 20.11 s | 17.51 s | 0.87× |
| Segment | 4 | 12.74 s | 14.65 s | **1.15×** |

![Batch-size scaling](assets/batch-scaling.png)

## Resolution scaling

This sweep uses YOLO26n detection, batch 2, and three epochs.

| Image size | boquilens | Ultralytics | Native speedup |
| ---: | ---: | ---: | ---: |
| 64 px | 16.33 s | 15.33 s | 0.94× |
| 128 px | 15.42 s | 14.86 s | 0.96× |
| 320 px | 15.80 s | 14.68 s | 0.93× |
| 640 px | 19.31 s | 15.32 s | 0.79× |

![YOLO26n detection resolution scaling](assets/resolution-scaling.png)

The near-flat Ultralytics results show that framework and process overhead dominate these deliberately
small jobs. Native is near parity through 320 px; 640 px still exposes backend work to do.

## Ten-epoch smoke runs

The longer runs exercise checkpointing, scheduling, loss reduction, and validation. They are still
too small to measure real model quality.

| Task | boquilens | Ultralytics | Native speedup |
| --- | ---: | ---: | ---: |
| Classify | 17.04 s | 23.42 s | **1.37×** |
| Detect | 50.55 s | 31.54 s | 0.62× |
| Segment | 62.33 s | 36.11 s | 0.58× |

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

Per-trial values, medians, environment metadata, and captured commands are preserved in the
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
  segmentation cells.
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

1. Reduce WGPU dispatch overhead for batch-1 and high-resolution YOLO26 workloads.
2. Reduce per-epoch full-precision record extraction cost in long runs.
3. Profile the remaining YOLO26 mask-head backward path at medium and larger scales.
4. Add full-dataset convergence runs before making quality-parity claims.
