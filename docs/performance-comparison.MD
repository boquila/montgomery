# Native training performance at production resolutions

The real-world-resolution results are materially different from the earlier 224/320 px comparison.
On the tested RTX 5080, Montgomery records a lower median in **4 of 9** nano detection and
segmentation comparisons at 640 × 640, although only YOLOv8n detection leads by more than 5%. At
1280 × 1280, Ultralytics is faster in all nine comparisons, by **2.49× to 3.94×**.

Classification is reported separately at its architecture-defined 224 × 224 input. Montgomery
continues to lead every classification comparison. This page therefore does not claim that native
training is uniformly faster: it is strong on classification and larger 640 px batches, but
Burn/WGPU has substantial high-resolution vision work left to do.

![Matched nano-family results at 640 and 1280](assets/resolution-scaling.png)

## 640 × 640 detection and segmentation

These are medians of complete three-epoch training commands at batch 2. A native speedup above
`1.00×` favors Montgomery; a value below `1.00×` favors Ultralytics.

| Model | Task | Montgomery | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: |
| YOLOv8n | Detect | 9.74 s | 12.02 s | **1.24×** |
| YOLOv8n | Segment | 12.52 s | 12.77 s | **1.02×** |
| YOLO11n | Detect | 12.32 s | 12.70 s | **1.03×** |
| YOLO11n | Segment | 14.93 s | 13.32 s | 0.89× |
| YOLO26n | Detect | 16.97 s | 14.34 s | 0.85× |
| YOLO26n | Segment | 21.12 s | 15.65 s | 0.74× |
| YOLOv3-Tiny-U | Detect | 11.42 s | 11.83 s | **1.04×** |
| YOLOv10n | Detect | 19.03 s | 13.66 s | 0.72× |
| YOLO12n | Detect | 19.65 s | 13.74 s | 0.70× |

The 2–4% leads should be read as near-parity, not decisive wins. YOLOv8n detection is the only cell
with a clear native advantage at this resolution. The end-to-end YOLOv10n and YOLO26n heads and
YOLO12n show the largest 640 px detection gaps.

## 1280 × 1280 detection and segmentation

The same nano models, dataset, epoch count, batch size, optimizer, precision, and trial policy are
repeated at 1280. This isolates the fourfold increase in canvas area.

| Model | Task | Montgomery | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: |
| YOLOv8n | Detect | 31.55 s | 12.64 s | 0.40× |
| YOLOv8n | Segment | 40.74 s | 13.80 s | 0.34× |
| YOLO11n | Detect | 34.74 s | 13.42 s | 0.39× |
| YOLO11n | Segment | 43.49 s | 14.85 s | 0.34× |
| YOLO26n | Detect | 43.29 s | 15.05 s | 0.35× |
| YOLO26n | Segment | 58.62 s | 16.60 s | 0.28× |
| YOLOv3-Tiny-U | Detect | 44.56 s | 13.12 s | 0.29× |
| YOLOv10n | Detect | 56.11 s | 14.25 s | 0.25× |
| YOLO12n | Detect | 56.05 s | 15.04 s | 0.27× |

Montgomery wall time grows by 2.55–3.90× from 640 to 1280 across these cells. Ultralytics grows by
only 1.04–1.11× in this deliberately short workload, so its process, data, and checkpoint overhead
still hides much of the extra CUDA work. The native increase makes its resolution-scaling cost
visible and identifies the main performance priority.

The 1280 matrix is intentionally limited to nano variants. During preflight, YOLO26m segmentation
at 1280, batch 2 exceeded the RTX 5080's 16 GB VRAM in Burn/WGPU. That failed stress case is not
included as a timing result and the benchmark does not silently lower one framework's batch size.

## Family and task overview

The primary matrix combines 224 px classification with 640 px detection and segmentation. The
input size is stated in the chart title so unrelated task contracts are not mistaken for a single
resolution comparison.

![Median wall time across primary family and task comparisons](assets/family-task-overview.png)

![Primary-matrix speedup heatmap](assets/task-speedup-heatmap.png)

| Model | Task | Input | Montgomery | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n | Classify | 224 px | 2.90 s | 11.98 s | **4.13×** |
| YOLO11n | Classify | 224 px | 4.52 s | 12.23 s | **2.71×** |
| YOLO26n | Classify | 224 px | 4.53 s | 12.55 s | **2.77×** |

Classification uses its real 224 px preprocessing contract and is not repeated at 640 or 1280.
Doing so would measure a non-standard workload rather than production classification behavior.

## YOLO26 scale comparison

Classification stays ahead through medium at 224 px. At 640 px, the native detection and
segmentation gap grows with model scale.

| Model | Task | Input | Montgomery | Ultralytics | Native speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLO26n | Classify | 224 px | 4.53 s | 12.55 s | **2.77×** |
| YOLO26n | Detect | 640 px | 16.97 s | 14.34 s | 0.85× |
| YOLO26n | Segment | 640 px | 21.12 s | 15.65 s | 0.74× |
| YOLO26s | Classify | 224 px | 4.96 s | 12.72 s | **2.56×** |
| YOLO26s | Detect | 640 px | 21.11 s | 14.79 s | 0.70× |
| YOLO26s | Segment | 640 px | 28.56 s | 16.41 s | 0.57× |
| YOLO26m | Classify | 224 px | 7.45 s | 13.44 s | **1.80×** |
| YOLO26m | Detect | 640 px | 38.31 s | 16.69 s | 0.44× |
| YOLO26m | Segment | 640 px | 61.66 s | 18.83 s | 0.31× |

![YOLO26 scale comparison](assets/model-scale.png)

## Batch scaling

The batch sweep changes only batch size. Classification uses 224 px; detection and segmentation
use 640 px. Batch 4 amortizes native dispatch and synchronization overhead enough for YOLO26n
detection to edge ahead and segmentation to come within 6%.

| Task | Input | Batch | Montgomery | Ultralytics | Native speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classify | 224 px | 1 | 8.26 s | 13.54 s | **1.64×** |
| Classify | 224 px | 2 | 4.53 s | 12.55 s | **2.77×** |
| Classify | 224 px | 4 | 2.88 s | 12.14 s | **4.21×** |
| Detect | 640 px | 1 | 25.54 s | 17.40 s | 0.68× |
| Detect | 640 px | 2 | 16.97 s | 14.34 s | 0.85× |
| Detect | 640 px | 4 | 12.17 s | 12.75 s | **1.05×** |
| Segment | 640 px | 1 | 31.22 s | 22.04 s | 0.71× |
| Segment | 640 px | 2 | 21.12 s | 15.65 s | 0.74× |
| Segment | 640 px | 4 | 16.43 s | 15.54 s | 0.95× |

![Batch-size scaling](assets/batch-scaling.png)

## Ten-epoch smoke runs

The longer runs exercise checkpointing, scheduling, loss reduction, and validation. They are still
too small to measure real model quality.

| Task | Input | Montgomery | Ultralytics | Native speedup |
| --- | ---: | ---: | ---: | ---: |
| Classify | 224 px | 15.79 s | 22.16 s | **1.40×** |
| Detect | 640 px | 54.37 s | 29.12 s | 0.54× |
| Segment | 640 px | 69.78 s | 33.89 s | 0.49× |

![Normalized loss over ten epochs](assets/convergence-normalized.png)

Loss magnitudes are normalized in the chart because the trainers report and reduce components
differently. Both losses stay finite, but this is a smoke signal rather than a claim of equal
optimization dynamics.

Post-training validation is outside the timed region:

| Task | Montgomery | Ultralytics | Interpretation |
| --- | --- | --- | --- |
| Classify | top-1 16.7%, top-5 83.3% | top-1 41.7%, top-5 91.7% | 12 validation images; not quality evidence |
| Detect | mAP50 0.0, mAP50-95 0.0 | mAP50 0.0, mAP50-95 0.0 | 4 validation images; the run is too short |
| Segment | box/mask mAP 0.0 | box/mask mAP 0.0 | 4 validation images; the run is too short |

The benchmark proves that both implementations train, checkpoint, and validate the tested tasks.
It does **not** establish COCO or ImageNet convergence parity. That requires full datasets and
standard training schedules.

## All measured scenarios

![All 36 benchmark scenarios](assets/all-scenarios-speedup.png)

![Effective end-to-end throughput](assets/effective-throughput.png)

![Per-trial repeatability](assets/trial-repeatability.png)

![First measured run versus warm runs](assets/first-vs-warm.png)

Per-trial values, medians, environment metadata, and captured commands are preserved in the
[raw benchmark results](assets/training-benchmark-results.json).

## Methodology

- Hardware: NVIDIA GeForce RTX 5080 with 16 GB VRAM; native Burn/WGPU used Vulkan driver 610.47.
- Software: Montgomery/Burn release build with fat LTO versus Ultralytics 8.4.117 and PyTorch 2.11
  CUDA 13.0.
- Inputs: classification is 224 × 224; the primary detection/segmentation matrix is 640 × 640; all
  nano detection/segmentation configurations are also tested at 1280 × 1280.
- Timing: external wall clock around the complete command, including startup, model loading,
  training, and checkpoint writes. Validation is excluded and recorded separately.
- Workload: three epochs, batch 2, and one checkpoint per epoch unless a scenario explicitly varies
  epochs or batch size.
- Precision: FP32 for both implementations.
- Optimizer: AdamW, `lr0=0.001`, `lrf=0.05`, weight decay `0.0005`, momentum `0.9`, cosine decay,
  and no warmup. Nominal batch size equals actual batch size.
- Data loading: four workers; native prefetch depth two.
- Sampling: seed zero, alternating framework order, three trials normally and two for segmentation.
- Data: the same real source fixtures, deterministically replicated with hard links. Classification
  uses 48 train/12 validation images; detection and segmentation use 32 train/4 validation images.
- Weights: the same upstream `.pt` checkpoint supplies both runs; native uses its tensor-only export.
- Warmup: native WGPU compilation is primed outside the measured trials. The first-vs-warm plot
  still exposes residual first-run effects.
- Augmentation policies and optimizer settings are matched where both APIs expose them. Random-number
  generators, scheduler details, and some loss reductions are not bit-identical.
- YOLOX is not included because Ultralytics does not provide a YOLOX training implementation; the
  separate official YOLOX trainer would not be a 1:1 comparison.

These short jobs emphasize complete-command developer feedback, not steady-state multi-hour
throughput. The nearly flat 640-to-1280 Ultralytics wall times in particular show that fixed process
and checkpoint overhead remains significant. Use the full-convergence runner before making quality
or long-run throughput claims.

## Reproduce it

After installing the pinned GPU benchmark environment and preparing the official checkpoints, run:

```console
uv run --project tools tools/benchmark_training.py --publish
```

The command builds Montgomery in release mode, prepares deterministic datasets, primes WGPU, runs
the 36-scenario matrix, validates the ten-epoch checkpoints, renders every chart, and publishes JSON
and images under `docs/assets/`.

For a quick 640/1280 comparison of one model:

```console
uv run --project tools tools/benchmark_training.py \
  --scenario yolov8n-detect \
  --scenario yolov8n-detect-1280px \
  --repeats 1
```

Use `--list` on `tools/bench_training_matrix.py` to see every scenario. Runs can be resumed, and full
checkpoints can be retained with `--keep-runs`; they are deleted by default to keep output small.

For strict full-COCO convergence, use:

```console
uv run --project tools tools/bench_full_convergence.py --download
```

That runner rejects partial datasets and performs matched training plus validation. Full convergence
results are still required before any quality-parity claim.
