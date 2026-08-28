# PERF_NOTES.md — CPU performance investigation

Date: 2026-08-28. Machine: AMD Ryzen 9 9950X3D (16C/32T), 32 GB RAM, Windows 11, rustc stable
x86_64-pc-windows-msvc, Burn 0.21.0-pre.4, PyTorch 2.13.0+cpu / Ultralytics 8.4.131 (conversion
venv). All latency numbers are median / minimum of 10 timed runs after 3 warmups, measured
sequentially (`--test-threads 1`), single image, batch 1, on zeros `[1, 3, 640, 640]` unless noted.
Motivating mystery: boquilens Flex CPU inference measured ~5-7.5x slower than the official
Ultralytics/PyTorch CPU runtime (README table).

## 1. Methodology audit — the 5-7.5x comparison is apples-to-apples

Compared `src/models/yolo11/model.rs:798-852` / `src/models/yolov10/model.rs:690-744` /
`src/models/yolo26/model.rs:630-678` (`latency_test!`) against `tools/bench_ultralytics_cpu.py`.

| Aspect | boquilens `latency_test!` | `bench_ultralytics_cpu.py` | Comparable? |
| --- | --- | --- | --- |
| Input | `Tensor::zeros([1,3,640,640])` (model.rs:821) | `torch.zeros(1,3,640,640)` (py:28) | Yes — identical |
| Preprocessing / letterbox | Excluded (zeros input) | Excluded (raw `model(input)`) | Yes — neither counts it |
| Timed region | `model.forward` = body + head decode; NMS excluded for yolo11 (it lives in `Predictor::predict`, not `forward`) | `model(input)` = `DetectionModel.forward` | Yes — same scope (see below) |
| Head decode | v10/26: DFL-free decode + top-300 selection inside `head.forward`; yolo11: DFL softmax + anchor decode + sigmoid (head.rs:121-168) | v10/26 (`end2end=True`): `Detect.forward` runs `_inference` **and `self.postprocess`** (top-300) inside `model(input)` (vendored `nn/modules/head.py:170-183`); yolo11: `_inference` decode only, NMS is outside in the predictor | Yes — per family the same work is timed |
| Result sync | `output.boxes.sum().into_data()` + scores (model.rs:836-838) — a no-op on CPU, load-bearing on GPU | eager PyTorch: implicit (every op synchronizes) | Yes — both measure completed compute |
| Warmup / timing | 3 warmups, 10 timed, median + min (model.rs:822-841) | identical (py:30-39, CLI defaults) | Yes |
| Threads | Flex/rayon pool = `available_parallelism` = 32; `--test-threads 1` avoids cross-test contention (AGENTS.md) | `torch.get_num_threads()` = 16 (default physical-core count), single process | Different policy, both "the library's default" — not a methodological bias |
| Precision | f16-stored artifacts upcast to f32 at load; f32 compute | fp32 model (`.float()`), fused conv+bn via `model.fuse()` (py:22-24) | Yes — both fp32 compute |
| Model load | outside timed region | outside timed region (py:22) | Yes |

Verdict: **the comparison is fair**. The 5-7.5x gap is genuine compute-speed difference between
Burn's Flex CPU backend (im2col+gemm via the `gemm` crate, rayon-parallel, simd/macerator element
kernels) and PyTorch's oneDNN-backed fused convolutions — not a measurement artifact. Preprocessing
is excluded on both sides; the head decode and (for v10/26) the top-300 postprocess are included on
both sides; yolo11's NMS is excluded on both sides (Ultralytics' `model(input)` returns raw decoded
predictions and NMS happens in its predictor).

Run-to-run drift (full 17-variant harness, default features): README run vs this re-run:

- yolov10n 129.1 → 130.9 ms; yolo26n 116.2 → 116.0; yolo11n 130.1 → 129.8; yolo11x 991.6 → 1038.6
  (min 988.5 vs 970.6 across runs — the x-scale is the noisiest).
- Typical drift ≤ 3% on medians; worst single outlier ~5%. Conclusions below survive this noise.

## 2. Alternative CPU backends / features

### 2.1 `cpu-simd` (Flex + `burn-flex/x86-v4`, i.e. AVX-512 GEMM microkernels)

`burn-flex` default features **already include `simd` (macerator runtime-dispatched
AVX2/AVX-512/SSE element kernels) and `rayon`** (burn-flex Cargo.toml features `default =
["std", "simd", "rayon"]`). The only unenabled knob is `x86-v4`, which compiles the `gemm-f32`
AVX-512 microkernels (`gemm-f32/src/microkernel.rs:182-288`). Added as boquilens feature
`cpu-simd = ["burn-flex/simd", "burn-flex/x86-v4"]`.

| Model | Flex default (ms) | Flex x86-v4 (ms) | Δ |
| --- | ---: | ---: | ---: |
| yolo11n | 129.8 | 133.4 | +2.8% |
| yolo11s | 244.3 | 256.3 | +4.9% |
| yolo11x | 1038.6 | 1021.3 | -1.7% |
| yolov10n | 130.9 | 134.1 | +2.4% |
| yolov10s | 246.5 | 252.1 | +2.3% |
| yolov10x | 946.4 | 954.9 | +0.9% |
| yolo26n | 116.0 | 120.4 | +3.8% |
| yolo26s | 236.0 | 252.8 | +7.1% |
| yolo26x | 980.9 | 1000.3 | +2.0% |

Parity: all 19 in-tree Flex golden tests pass under `cpu-simd` — numerically sound. Performance:
consistently ~2-5% **slower** (only yolo11x inside noise). On Zen 5, the gemm AVX-512 path does not
beat the tuned AVX-2 path at these shapes. **Reject** (feature kept for reproduction only).

### 2.2 `cpu-cubecl` (the `burn-cpu` CubeCL CPU backend, fusion + autotune defaults)

Added `burn-cpu = "=0.21.0-pre.4"` + feature `cpu-cubecl = ["burn/cpu", "dep:burn-cpu"]` and an
ignored measurement harness `tests/cpu_backend.rs` (same methodology as the in-tree Flex tests).

| Model | Flex default (ms) | burn-cpu steady state (ms) | burn-cpu first run incl. JIT/autotune (ms) |
| --- | ---: | ---: | ---: |
| yolo11n | 129.8 | 93.0 / 90.5 | 96,194 |
| yolo11s | 244.3 | 236.7 / 213.6 | 36,361 |
| yolo11x | 1038.6 | 1465.3 / 1417.9 | 253,067 |
| yolov10n | 130.9 | 107.7 / 99.0 | 2,067 |
| yolov10s | 246.5 | 293.1 / 269.3 | 5,648 |
| yolov10x | 946.4 | 1262.1 / 1255.1 | 75,798 |
| yolo26n | 116.0 | 100.9 / 96.7 | 24,642 |
| yolo26s | 236.0 | 243.7 / 238.2 | 12,829 |
| yolo26x | 980.9 | 1483.0 / 1451.4 | 2,313 |

Steady state is attractive for n-scale (~1.2-1.4x faster than Flex) but slower for x-scale, and the
first run of a fresh process JIT-compiles every kernel shape: 2-250 **seconds** per variant on a
cold on-disk compilation cache (96 s for yolo11n, 253 s for yolo11x on the first-ever run; a warm
cache trims this to ~0.2-2 s) — still a poor fit for a CLI that loads per invocation on new machines.

**Verdict: reject — numerically unsound on burn 0.21.0-pre.4.** The golden parity tests
(`cubecl_cpu_{yolo11n,yolov10n,yolo26n}_matches_golden` in `tests/cpu_backend.rs`, same fixtures and
2e-4 tolerance as the in-tree Flex tests) fail far beyond tolerance:

- yolo11n decoded boxes mean 224.43 vs fixture 177.95; raw scores mean +1.70 vs -15.16.
- yolov10n decoded boxes mean 289.29 vs 287.97; yolo26n similar.

Root cause isolated by element-wise probes against Flex (`tests/cpu_backend.rs`):

- All primitive ops match: conv2d (incl. depthwise, biased, non-square 80x64, model shapes 640→320
  etc.), matmul (2D and 4D batched), sigmoid, silu, softmax, maxpool, nearest upsample, broadcast
  affine, reshape/cat of contiguous data — every probe ≤ 1e-5.
- Loaded weights are byte-identical across backends (standalone module load of
  `head.p3.box_0.conv.weight`: max|diff| = 0); `load_from` applies 102/102 head and 315/315 body
  params; zero-copy on/off and fusion on/off do not change the mismatch; results are deterministic
  run-to-run.
- Every one of the nine detection towers (3 branches x box/cls), run standalone with the real
  weights on materialized inputs, matches Flex (max|diff| ≤ 2e-5).
- **But `reshape` applied to a conv output scrambles the values**: a biased 1x1 conv on
  `[1,64,80,64]` followed by `.reshape([1,64,5120])` gives max|diff| = 1.1557 vs Flex, while the
  same conv without the reshape gives 0.000000
  (`cubecl_cpu_reshape_after_conv_is_unsound`). Each detection tower's final 1x1 conv output is
  reshaped to `[batch, channels, anchors]` inside the heads, so every family's raw predictions are
  scrambled, and everything downstream (decode, sigmoid, top-k) is garbage.

This is a CubeCL-CPU/cubecl-cpu 0.10.0-pre.4 contract bug (conv output layout not canonicalized
across `reshape`), not something boquilens can or should work around. The failing parity tests are
kept as the upgrade gate: if they pass after a Burn upgrade, re-measure §2.2.

### 2.3 Threading (Flex; `RAYON_NUM_THREADS`)

Flex default = one rayon thread per logical core (32). Measured on yolo11n / yolo11x / yolo26n /
yolov10n (default features, release):

| RAYON_NUM_THREADS | yolo11n | yolo11x | yolo26n | yolov10n |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 128.5 | 1134.4 | 109.0 | 122.4 |
| 8 | 125.2 | 971.7 | 106.5 | 117.8 |
| 16 | 129.6 | 955.5 | 108.5 | 119.8 |
| 32 (default) | 140.4 | 994.0 | 116.3 | 128.7 |

- n-scale models barely parallelize (4 ≈ 8 ≈ 16 threads) and are ~8-11% **slower** at 32 threads —
  classic oversubscription/thread-launch overhead on many small kernels.
- x-scale benefits from 8-16 threads (~4% over 32), collapses at 4.
- Sweet spot: 8-16 threads. Setting `RAYON_NUM_THREADS=16` is a free ~5% (large models) to ~10%
  (nano models) improvement over the default — worth documenting, not worth a code change.

## 3. Product-path comparison (yolo11n, `assets/dog_bike_man.jpg`)

| Pipeline | Measured | Notes |
| --- | ---: | --- |
| **boquilens CLI** `boquilens predict --model yolo11n --weights target/yolo11n-...bpk --source assets/dog_bike_man.jpg` | **150 ms** median (n=5: 148.6-162.5) | process start + weights load + letterbox + forward + NMS + annotate + PNG write |
| — Flex forward on the actual 512x640 canvas (throwaway harness) | 117.4 ms median | the model compute inside the CLI |
| — CLI overhead beyond forward | ~33 ms | weights load, letterbox, NMS (3 dets), annotation render, PNG write, process start |
| **Ultralytics** `YOLO('yolo11n.pt').predict(src)` (model pre-loaded) | **13.5-14.1 ms** median (n=10) | letterbox + fused PyTorch forward + NMS + results; model load excluded |
| — fused PyTorch forward on the same 512x640 canvas (zeros) | 15.7 ms median | within drift of the end-to-end predict number |
| — Ultralytics overhead beyond forward | ~0-2 ms | pre/post are trivially cheap on CPU |

Notes:

- The PyTorch forward-only harness drifts between sessions (17.7 / 21.3 ms median for identical
  work) — ±20%; conclusions below survive this.
- boquilens' forward at the real canvas is **~7.5x** PyTorch's (117.4 vs 15.7 ms); the full CLI is
  **~11x** the predict call (150 vs 13.5 ms).
- Both pipelines are compute-dominated: boquilens' own overhead beyond the model is ~33 ms out of
  150 ms (≈22%), Ultralytics' is ~0-2 ms out of 13.5 ms. **Preprocessing/postprocessing is not the
  problem — the Flex forward is.** Closing the product gap means closing the forward gap (or using
  the GPU path, already 12-15x faster than Flex CPU per the README table).

## 4. Recommendation

1. **Keep the default as-is** (Flex with its default `simd`+`rayon` features). Nothing measured
   beats it soundly on CPU.
2. **Reject `cpu-simd`/x86-v4 for default use** — sound but ~2-5% slower; feature kept for
   reproduction (`--features cpu-simd`).
3. **Reject `burn-cpu`** — fast at n-scale (93 ms) but numerically broken on
   0.21.0-pre.4 (reshape-after-conv scramble, proven in `tests/cpu_backend.rs`), slower at x-scale,
   and its JIT warm-up (seconds to minutes per shape) is incompatible with one-shot CLI use.
   Re-run the kept parity + latency tests after Burn upgrades; revisit only if parity passes.
4. **Document `RAYON_NUM_THREADS=8-16`** as a free ~5-10% CPU win over the 32-thread default.
5. If CPU latency becomes a product priority, the leverage is upstream (Burn Flex kernels /
   oneDNN-style conv fusion), not in this crate. The GPU path already exists for latency-critical
   use.

## 5. Reproduction commands

```console
# baseline + product-path binary
cargo test --locked --release measures_single_inference_latency -- --ignored --nocapture --test-threads 1
cargo build --locked --release

# Flex x86-v4
cargo test --locked --release --features cpu-simd measures_single_inference_latency -- --ignored --nocapture --test-threads 1
cargo test --locked --release --features cpu-simd matches_ultralytics_golden -- --ignored --nocapture --test-threads 1

# CubeCL CPU backend (parity tests intentionally fail on 0.21.0-pre.4 — see §2.2)
cargo test --locked --release --features cpu-cubecl cubecl_cpu -- --ignored --nocapture --test-threads 1

# rayon thread scaling
$env:RAYON_NUM_THREADS='16'
cargo test --locked --release -- --ignored --nocapture --test-threads 1 yolo11n_measures yolo11x_measures yolo26n_measures yolov10n_measures

# product path
& target\boquilens-default.exe predict --model yolo11n --weights target\yolo11n-coco-ultralytics-v8.4-boquilens-v1.bpk --source assets\dog_bike_man.jpg
& target\.venv\Scripts\python.exe tools\bench_ultralytics_predict.py target\yolo11n.pt assets\dog_bike_man.jpg
& target\.venv\Scripts\python.exe tools\bench_ultralytics_cpu.py target\yolo11n.pt
```
