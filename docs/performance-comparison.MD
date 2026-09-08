# Montgomery vs Ultralytics training performance

This is an end-to-end comparison of the two programs people actually run. It measures one short
training job at the common **640 × 640** and **1280 × 1280** vision resolutions, including process
startup, data loading, training, and checkpoint writes.

## Exact test settings

| Setting | Value for both programs |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16 GB |
| Vision data | 32 images |
| Vision resolutions | **640 × 640** and **1280 × 1280** |
| Classification data and resolution | 48 images at **224 × 224** |
| Training work | 3 epochs, batch 2 |
| Precision and optimizer | Strict FP32, AdamW |
| Data-loader workers | `--workers 4` |
| Validation and export | Disabled |
| Saving | One full, resumable checkpoint per epoch |
| Timing result | Median of 3 trials; segmentation uses 2 |

Montgomery uses Burn/WGPU over Vulkan. Ultralytics 8.4.117 uses PyTorch over CUDA. Lower numbers
are better. Both programs receive one untimed prime run before measurement, and their measured
trial order alternates.

## Bottom line

| Question | Answer |
| --- | --- |
| Which is faster at **640 × 640**? | Mixed. Montgomery clearly wins 3 cells, 2 are within 1%, and Ultralytics clearly wins 4. |
| Which is faster at **1280 × 1280**? | Ultralytics wins all 9 cells by **2.28–3.60×**. |
| Which uses less process-tree RAM? | Montgomery: **0.53–1.08 GiB**; Ultralytics: **4.68–5.50 GiB**. |
| Which uses less dedicated VRAM? | Ultralytics in every measured vision cell. |

![Timing at 640 and 1280](assets/resolution-scaling.png)

## Timing at 640 × 640

Every row is 32 images, 3 epochs, batch 2, strict FP32, and four requested workers.

| Model | Task | Montgomery | Ultralytics | Faster result |
| --- | --- | ---: | ---: | --- |
| YOLOv8n | Detect | **9.50 s** | 13.17 s | Montgomery **1.39×** |
| YOLOv8n | Segment | **12.13 s** | 13.87 s | Montgomery **1.14×** |
| YOLO11n | Detect | **11.96 s** | 14.04 s | Montgomery **1.17×** |
| YOLO11n | Segment | **14.28 s** | 14.39 s | Within 1% |
| YOLO26n | Detect | 15.61 s | **15.48 s** | Within 1% |
| YOLO26n | Segment | 19.94 s | **17.60 s** | Ultralytics **1.13×** |
| YOLOv3-Tiny-U | Detect | 13.34 s | **12.40 s** | Ultralytics **1.08×** |
| YOLOv10n | Detect | 18.63 s | **15.48 s** | Ultralytics **1.20×** |
| YOLO12n | Detect | 18.97 s | **15.40 s** | Ultralytics **1.23×** |

## Timing at 1280 × 1280

Only the input changes. Doubling each edge makes the image canvas four times larger.

| Model | Task | Montgomery | Ultralytics | Faster result |
| --- | --- | ---: | ---: | --- |
| YOLOv8n | Detect | 30.96 s | **13.59 s** | Ultralytics **2.28×** |
| YOLOv8n | Segment | 40.22 s | **15.80 s** | Ultralytics **2.54×** |
| YOLO11n | Detect | 33.75 s | **14.51 s** | Ultralytics **2.33×** |
| YOLO11n | Segment | 43.49 s | **15.83 s** | Ultralytics **2.75×** |
| YOLO26n | Detect | 40.98 s | **16.37 s** | Ultralytics **2.50×** |
| YOLO26n | Segment | 56.27 s | **17.81 s** | Ultralytics **3.16×** |
| YOLOv3-Tiny-U | Detect | 47.66 s | **15.81 s** | Ultralytics **3.01×** |
| YOLOv10n | Detect | 53.62 s | **14.89 s** | Ultralytics **3.60×** |
| YOLO12n | Detect | 54.31 s | **15.51 s** | Ultralytics **3.50×** |

The batch remains 2 for both programs. YOLO26m segmentation at 1280 exceeded the card's 16 GiB
during preflight, so it is omitted instead of lowering the batch for only one program.

## RAM and VRAM

Each memory number comes from one separate observed training command, sampled at a 100 ms target
cadence. RAM is the simultaneous resident memory of the root process and all descendants. VRAM is
the sum of their dedicated GPU memory, not whole-device `nvidia-smi` usage.

### Memory at 640 × 640

| Model | Task | Montgomery RAM | Ultralytics RAM | Montgomery VRAM | Ultralytics VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n | Detect | **0.55 GiB** | 4.70 GiB | 1.15 GiB | **0.84 GiB** |
| YOLOv8n | Segment | **0.59 GiB** | 4.71 GiB | 2.51 GiB | **1.41 GiB** |
| YOLO11n | Detect | **0.55 GiB** | 4.68 GiB | 1.34 GiB | **0.93 GiB** |
| YOLO11n | Segment | **0.55 GiB** | 4.69 GiB | 2.64 GiB | **1.47 GiB** |
| YOLO26n | Detect | **0.56 GiB** | 4.69 GiB | 1.65 GiB | **1.01 GiB** |
| YOLO26n | Segment | **0.63 GiB** | 4.71 GiB | 3.27 GiB | **1.62 GiB** |
| YOLOv3-Tiny-U | Detect | **1.08 GiB** | 4.82 GiB | 2.31 GiB | **1.16 GiB** |
| YOLOv10n | Detect | **0.57 GiB** | 4.72 GiB | 1.78 GiB | **1.08 GiB** |
| YOLO12n | Detect | **0.53 GiB** | 4.73 GiB | 1.53 GiB | **1.03 GiB** |

### Memory at 1280 × 1280

| Model | Task | Montgomery RAM | Ultralytics RAM | Montgomery VRAM | Ultralytics VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOv8n | Detect | **0.59 GiB** | 5.45 GiB | 4.56 GiB | **2.70 GiB** |
| YOLOv8n | Segment | **0.61 GiB** | 5.34 GiB | 5.60 GiB | **3.80 GiB** |
| YOLO11n | Detect | **0.57 GiB** | 5.46 GiB | 5.37 GiB | **2.81 GiB** |
| YOLO11n | Segment | **0.62 GiB** | 5.31 GiB | 6.65 GiB | **3.55 GiB** |
| YOLO26n | Detect | **0.65 GiB** | 5.44 GiB | 6.48 GiB | **3.17 GiB** |
| YOLO26n | Segment | **0.67 GiB** | 5.35 GiB | 8.53 GiB | **4.38 GiB** |
| YOLOv3-Tiny-U | Detect | **1.08 GiB** | 5.50 GiB | 4.60 GiB | **3.23 GiB** |
| YOLOv10n | Detect | **0.61 GiB** | 5.45 GiB | 6.97 GiB | **3.39 GiB** |
| YOLO12n | Detect | **0.56 GiB** | 5.50 GiB | 8.79 GiB | **4.08 GiB** |

![Peak process-tree RAM](assets/resolution-ram.png)

![Peak dedicated GPU memory](assets/resolution-vram.png)

The raw data also retains private committed memory, shared GPU memory, per-process peaks, process
names and counts, provider details, actual sample counts, cadence, and measurement errors. Windows
may count shared pages in more than one process working set, so process-tree RAM is an operational
footprint rather than an exact count of unique physical pages.

### Does observability slow Montgomery down?

Not in the reported timings. The observer is part of the external benchmark parent, not Montgomery,
and every timing trial runs with monitoring disabled. Sampling occurs only in the separate resource
pass. That pass can experience a small amount of system-level polling disturbance, but no telemetry
code, thread, or sampling call was added to either trainer.

## Are these settings apples to apples?

At the user-visible workload level, yes. Both receive the same upstream model parameters, model
family, task, data, image size, epochs, batch, seed, strict FP32 precision, AdamW settings, requested
worker count, validation policy, and checkpoint cadence. Montgomery receives converted native
Burnpack weights; Ultralytics receives the original PyTorch checkpoint.

The same `--workers 4` request does not make the implementations identical:

| Internal detail | Montgomery | Ultralytics |
| --- | --- | --- |
| Four workers | Four threads | Four PyTorch worker processes |
| Prefetch | One global queue, 2 prepared batches | Per-worker queues, up to 4 tasks each |
| Host-to-device path | Burn/WGPU | Pinned-memory PyTorch/CUDA path |
| GPU runtime | Vulkan | CUDA |

Those differences are exactly what an end user pays for. Artificially matching queue depth or
process structure would compare hand-tuned internals instead of the same public four-worker request.

Ultralytics normally forces validation on its final epoch even when `val=False`, then strips
optimizer state during its final evaluation step. The benchmark wrapper overrides only those two
hooks: it performs no validation, just like Montgomery's `--no-val`, and leaves `last.pt` fully
resumable. Its training, loss, augmentation, optimizer, and checkpoint creation paths are unchanged.

## Trial repeatability

Each workload has one aligned row on one shared time axis. Circles are trials, diamonds are medians,
thick lines show min–max, and the thin connector shows the gap between framework medians.

![Aligned per-trial timing ranges](assets/trial-repeatability.png)

## Classification uses 224 × 224

Classification is separate because its standard input here is 224 × 224, not 640 or 1280.

| Model | Montgomery | Ultralytics | Faster result |
| --- | ---: | ---: | --- |
| YOLOv8n-cls | **3.13 s** | 12.27 s | Montgomery **3.91×** |
| YOLO11n-cls | **4.77 s** | 13.17 s | Montgomery **2.76×** |
| YOLO26n-cls | **4.72 s** | 12.59 s | Montgomery **2.67×** |

These short runs measure developer-feedback cost, not steady-state throughput or model quality.

## Raw data and reproduction

All timing trials, resource passes, commands, framework metadata, and failures are in the
[schema-v5 raw results](assets/training-benchmark-results.json).

Run and publish the same 21-cell report matrix:

```console
uv run --project tools tools/benchmark_training.py --docs-matrix --publish
```

Run one model at both real-world vision resolutions:

```console
uv run --project tools tools/benchmark_training.py \
  --scenario yolov8n-detect \
  --scenario yolov8n-detect-1280px \
  --repeats 1
```
