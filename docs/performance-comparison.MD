# Montgomery vs Ultralytics training performance

## What was tested

| Setting | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16 GB |
| Montgomery | Burn/WGPU, Vulkan driver 610.47 |
| Ultralytics | 8.4.117, PyTorch 2.11, CUDA 13.0 |
| Precision | FP32 for both |
| Main vision resolution | **640 × 640** model input |
| High vision resolution | **1280 × 1280** model input |
| Classification resolution | **224 × 224** model input |
| Default workload | 3 epochs, batch 2, one checkpoint per epoch |
| Timed region | Complete command: startup, loading, training, and checkpoint writes |
| Trials | 3 per detect/classify result; 2 per segment result |
| Memory pass | One extra trial per 640/1280 cell; 100 ms target sampling interval |

Every time below is the median command wall time. Lower is better. `Advantage` is always written
for the winner, so `Ultralytics 2.50×` means Montgomery took 2.50 times as long.

| Comparison detail | Montgomery | Ultralytics | Treatment |
| --- | --- | --- | --- |
| Input workload | Same model family, task, dataset, epochs, batch, and resolution | Same | Held constant |
| Training recipe | FP32 AdamW, matched learning-rate settings, seed 0 | Same | Held constant |
| CPU workers | 4 requested | 4 requested | Held constant |
| Loader design | Threads, global prefetch queue of 2 batches | Processes, pinned memory, prefetch factor 4 per worker | Measured as part of each program |
| GPU stack | Burn/WGPU over Vulkan | PyTorch over CUDA | Measured as part of each program |
| Validation | Disabled inside timed training | Disabled inside timed training | Held constant |
| Saving | Resumable checkpoint each epoch | Checkpoint each epoch | Held constant at the workflow level |

## Main result

| Workload | Resolution | Comparisons | Montgomery wins | Ultralytics wins | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Nano detection + segmentation | **640 × 640** | 9 | 4 | 5 | Near parity in 3 cells; YOLOv8n detect is the clear Montgomery win |
| Nano detection + segmentation | **1280 × 1280** | 9 | 0 | 9 | Ultralytics is **2.50–3.94× faster** |
| Nano classification | **224 × 224** | 3 | 3 | 0 | Montgomery is **2.71–4.13× faster** |

![Nano model timing at 640 and 1280](assets/resolution-scaling.png)

## Detection and segmentation at 640 × 640

All rows use batch 2 and 3 epochs.

| Model | Task | Resolution | Montgomery | Ultralytics | Winner | Advantage |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| YOLOv8n | Detect | 640 × 640 | **9.74 s** | 12.02 s | Montgomery | **1.23×** |
| YOLOv8n | Segment | 640 × 640 | **12.52 s** | 12.77 s | Montgomery | 1.02× |
| YOLO11n | Detect | 640 × 640 | **12.32 s** | 12.70 s | Montgomery | 1.03× |
| YOLO11n | Segment | 640 × 640 | 14.93 s | **13.32 s** | Ultralytics | 1.12× |
| YOLO26n | Detect | 640 × 640 | 16.97 s | **14.34 s** | Ultralytics | 1.18× |
| YOLO26n | Segment | 640 × 640 | 21.12 s | **15.65 s** | Ultralytics | 1.35× |
| YOLOv3-Tiny-U | Detect | 640 × 640 | **11.42 s** | 11.83 s | Montgomery | 1.04× |
| YOLOv10n | Detect | 640 × 640 | 19.03 s | **13.66 s** | Ultralytics | 1.39× |
| YOLO12n | Detect | 640 × 640 | 19.65 s | **13.74 s** | Ultralytics | 1.43× |

Only YOLOv8n detection has a Montgomery lead greater than 5%. The other three Montgomery wins are
within 4% and should be treated as near-parity.

## Detection and segmentation at 1280 × 1280

These are the same nano models, batch 2, and 3 epochs. Only the model input changes from 640 to
1280, which increases canvas area by 4×.

| Model | Task | Resolution | Montgomery | Ultralytics | Winner | Advantage |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| YOLOv8n | Detect | 1280 × 1280 | 31.55 s | **12.64 s** | Ultralytics | **2.50×** |
| YOLOv8n | Segment | 1280 × 1280 | 40.74 s | **13.80 s** | Ultralytics | **2.95×** |
| YOLO11n | Detect | 1280 × 1280 | 34.74 s | **13.42 s** | Ultralytics | **2.59×** |
| YOLO11n | Segment | 1280 × 1280 | 43.49 s | **14.85 s** | Ultralytics | **2.93×** |
| YOLO26n | Detect | 1280 × 1280 | 43.29 s | **15.05 s** | Ultralytics | **2.88×** |
| YOLO26n | Segment | 1280 × 1280 | 58.62 s | **16.60 s** | Ultralytics | **3.53×** |
| YOLOv3-Tiny-U | Detect | 1280 × 1280 | 44.56 s | **13.12 s** | Ultralytics | **3.40×** |
| YOLOv10n | Detect | 1280 × 1280 | 56.11 s | **14.25 s** | Ultralytics | **3.94×** |
| YOLO12n | Detect | 1280 × 1280 | 56.05 s | **15.04 s** | Ultralytics | **3.73×** |

The 1280 sweep is limited to nano models. In a preflight run, YOLO26m segmentation at 1280,
batch 2 exceeded the RTX 5080's 16 GB VRAM in Burn/WGPU. It is not reported as a timing result, and
the benchmark does not lower the batch size for only one framework.

## RAM and VRAM

These are end-to-end operational footprints, not model tensors in isolation. Both commands request
four CPU data-loading workers. Montgomery implements them as threads with a global two-batch
prefetch queue. Ultralytics uses four Python worker processes, pinned memory, and four prefetched
tasks per worker. Those implementation choices are intentionally included because they determine
what users must provision to run each program.
On this Windows run, every native trial peaked at one OS process and every Ultralytics trial at six
process-tree members: the environment launcher, trainer, and four loader workers.

The tables show one dedicated memory-observation pass per cell. Values are sampled peaks in GiB;
they are separate from the repeated timing medians above. The standard benchmark now records these
measurements for every trial, so the next complete published run will report median memory peaks.
Actual cadence is query-time dependent and is recorded with the sample count and duration.

### Memory at 640 × 640

All rows use batch 2 and 3 epochs.

| Model | Task | Montgomery RAM | Ultralytics RAM | Montgomery VRAM | Ultralytics VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n | Detect | **0.54 GiB** | 4.76 GiB | 1.14 GiB | **0.96 GiB** |
| YOLOv8n | Segment | **0.59 GiB** | 4.76 GiB | 2.51 GiB | **1.42 GiB** |
| YOLO11n | Detect | **0.53 GiB** | 4.74 GiB | 1.34 GiB | **1.05 GiB** |
| YOLO11n | Segment | **0.57 GiB** | 4.82 GiB | 2.63 GiB | **1.47 GiB** |
| YOLO26n | Detect | **0.55 GiB** | 4.77 GiB | 1.65 GiB | **1.13 GiB** |
| YOLO26n | Segment | **0.60 GiB** | 4.84 GiB | 3.25 GiB | **1.62 GiB** |
| YOLOv3-Tiny-U | Detect | **1.10 GiB** | 4.78 GiB | 2.31 GiB | **1.38 GiB** |
| YOLOv10n | Detect | **0.56 GiB** | 4.80 GiB | 1.71 GiB | **1.19 GiB** |
| YOLO12n | Detect | **0.54 GiB** | 4.79 GiB | 1.53 GiB | **1.15 GiB** |

![Peak process-tree RAM at 640 and 1280](assets/resolution-ram.png)

### Memory at 1280 × 1280

All rows use batch 2 and 3 epochs.

| Model | Task | Montgomery RAM | Ultralytics RAM | Montgomery VRAM | Ultralytics VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n | Detect | **0.61 GiB** | 5.55 GiB | 4.56 GiB | **2.85 GiB** |
| YOLOv8n | Segment | **0.66 GiB** | 5.41 GiB | 5.84 GiB | **3.80 GiB** |
| YOLO11n | Detect | **0.59 GiB** | 5.55 GiB | 5.37 GiB | **2.97 GiB** |
| YOLO11n | Segment | **0.63 GiB** | 5.43 GiB | 6.65 GiB | **3.55 GiB** |
| YOLO26n | Detect | **0.62 GiB** | 5.56 GiB | 6.42 GiB | **3.62 GiB** |
| YOLO26n | Segment | **0.67 GiB** | 5.46 GiB | 8.49 GiB | **4.38 GiB** |
| YOLOv3-Tiny-U | Detect | **1.10 GiB** | 5.57 GiB | 4.60 GiB | **3.64 GiB** |
| YOLOv10n | Detect | **0.61 GiB** | 5.62 GiB | 6.97 GiB | **3.55 GiB** |
| YOLO12n | Detect | **0.59 GiB** | 5.59 GiB | 8.79 GiB | **4.42 GiB** |

![Peak dedicated GPU memory at 640 and 1280](assets/resolution-vram.png)

At 640, Montgomery uses 0.53-1.10 GiB of resident process-tree RAM versus 4.74-4.84 GiB for
Ultralytics. At 1280, the ranges are 0.59-1.10 GiB and 5.41-5.62 GiB. Dedicated GPU memory moves in
the other direction: Montgomery uses more VRAM in every measured cell, reaching 8.79 GiB for
YOLO12n detection and 8.49 GiB for YOLO26n segmentation at 1280.

RAM is the simultaneous sum of resident working sets for the root process and all descendants. On
Windows, shared pages may appear in more than one process working set, so this is an operational
footprint rather than an exact count of unique physical pages. The raw data also retains private
committed bytes, per-PID peaks, process names/counts, dedicated VRAM, shared GPU memory, sampler
provenance, sample counts, and any measurement errors. VRAM comes from Windows' per-process GPU
counters rather than whole-device `nvidia-smi` usage.
The observer itself runs in the benchmark parent and is excluded from these memory totals. Its
final shutdown query is outside command wall time; concurrent sampling work is not corrected for.

## What batch size changes at 640 × 640

These YOLO26n runs use 3 epochs. The batch-2 rows are the main-table results above.

| Task | Resolution | Batch | Montgomery | Ultralytics | Winner | Advantage |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Detect | 640 × 640 | 1 | 25.54 s | **17.40 s** | Ultralytics | 1.47× |
| Detect | 640 × 640 | 2 | 16.97 s | **14.34 s** | Ultralytics | 1.18× |
| Detect | 640 × 640 | 4 | **12.17 s** | 12.75 s | Montgomery | 1.05× |
| Segment | 640 × 640 | 1 | 31.22 s | **22.04 s** | Ultralytics | 1.42× |
| Segment | 640 × 640 | 2 | 21.12 s | **15.65 s** | Ultralytics | 1.35× |
| Segment | 640 × 640 | 4 | 16.43 s | **15.54 s** | Ultralytics | 1.06× |

Larger batches amortize Montgomery's fixed dispatch and synchronization costs. At batch 4,
detection moves slightly ahead and segmentation comes within 6%.

![YOLO26n batch-size timing](assets/batch-scaling.png)

## What model size changes at 640 × 640

These YOLO26 runs use batch 2 and 3 epochs.

| Model | Task | Resolution | Montgomery | Ultralytics | Winner | Advantage |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| YOLO26n | Detect | 640 × 640 | 16.97 s | **14.34 s** | Ultralytics | 1.18× |
| YOLO26n | Segment | 640 × 640 | 21.12 s | **15.65 s** | Ultralytics | 1.35× |
| YOLO26s | Detect | 640 × 640 | 21.11 s | **14.79 s** | Ultralytics | 1.43× |
| YOLO26s | Segment | 640 × 640 | 28.56 s | **16.41 s** | Ultralytics | 1.74× |
| YOLO26m | Detect | 640 × 640 | 38.31 s | **16.69 s** | Ultralytics | 2.30× |
| YOLO26m | Segment | 640 × 640 | 61.66 s | **18.83 s** | Ultralytics | 3.27× |

The Burn/WGPU gap grows with YOLO26 model size, especially for segmentation.

![YOLO26 model-size timing](assets/model-scale.png)

## Classification at 224 × 224

Classification uses its standard 224 × 224 input. It is not run at 640 or 1280.

| Model | Resolution | Montgomery | Ultralytics | Winner | Advantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n-cls | 224 × 224 | **2.90 s** | 11.98 s | Montgomery | **4.13×** |
| YOLO11n-cls | 224 × 224 | **4.52 s** | 12.23 s | Montgomery | **2.71×** |
| YOLO26n-cls | 224 × 224 | **4.53 s** | 12.55 s | Montgomery | **2.77×** |
| YOLO26s-cls | 224 × 224 | **4.96 s** | 12.72 s | Montgomery | **2.56×** |
| YOLO26m-cls | 224 × 224 | **7.45 s** | 13.44 s | Montgomery | **1.80×** |

## Ten-epoch smoke test

This checks longer execution, checkpointing, scheduling, and validation. The datasets are too small
to make model-quality claims.

| Task | Resolution | Batch | Montgomery | Ultralytics | Winner | Advantage |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Classify | 224 × 224 | 2 | **15.79 s** | 22.16 s | Montgomery | 1.40× |
| Detect | 640 × 640 | 2 | 54.37 s | **29.12 s** | Ultralytics | 1.87× |
| Segment | 640 × 640 | 2 | 69.78 s | **33.89 s** | Ultralytics | 2.06× |

## Trial repeatability

The chart below uses separate, aligned panels with the same time scale. Each circle is one measured
trial, the diamond is the median, and the horizontal line covers the minimum-to-maximum range.

![Aligned per-trial timing ranges](assets/trial-repeatability.png)

## Bottom line

- For nano models at **640 × 640, batch 2**, the result is mixed. YOLOv8n detection is Montgomery's
  only clear win.
- At **1280 × 1280, batch 2**, Ultralytics wins every tested model by 2.50–3.94×.
- At **640 × 640, batch 4**, Montgomery catches up: YOLO26n detection is 1.05× faster and
  segmentation is 1.06× slower.
- At the standard **224 × 224 classification resolution**, Montgomery wins every tested scale.
- Montgomery uses much less process-tree RAM in the measured vision cells, while Ultralytics uses
  less dedicated VRAM in every one of them.
- The largest native optimization targets are high resolution, larger YOLO26 scales, and
  segmentation.

## Data and reproduction

The benchmark uses the same upstream checkpoint for both frameworks, FP32, AdamW, four data-loading
workers, seed zero, alternating framework order, and matched augmentation settings where both APIs
expose them. Detection and segmentation use 32 training images; classification uses 48. These are
short developer-feedback workloads, not steady-state or model-quality benchmarks.

All 36 scenarios, individual trials, commands, medians, and environment metadata are in the
[raw benchmark results](assets/training-benchmark-results.json).

The one-pass 640/1280 resource sweep, including per-process RAM/VRAM records, is in the
[raw memory results](assets/training-memory-results.json).

Run the complete comparison after preparing the official checkpoints:

```console
uv run --project tools tools/benchmark_training.py --publish
```

Run one model at both vision resolutions:

```console
uv run --project tools tools/benchmark_training.py \
  --scenario yolov8n-detect \
  --scenario yolov8n-detect-1280px \
  --repeats 1
```
