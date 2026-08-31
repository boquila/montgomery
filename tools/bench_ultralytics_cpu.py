"""Benchmark official Ultralytics (PyTorch, CPU) single-image inference for comparison.

Mirrors Montgomery's latency harness methodology so the README comparison is apples-to-apples:
batch 1, 640 px input, fp32 compute on CPU, 3 warmup runs, 10 timed runs, median and minimum,
models measured sequentially. The model forward includes the head decode and, for the end-to-end
families, the top-k postprocess — the same scope as the Rust `measures_single_inference_latency`
tests. Run from the repository root with uv:

    uv run --locked tools\\bench_ultralytics_cpu.py <checkpoint.pt> ...
"""

from __future__ import annotations

import argparse
import time

import torch
from ultralytics import YOLO


def benchmark(checkpoint: str, warmup_runs: int, timed_runs: int) -> None:
    model = YOLO(checkpoint).model.eval().float()
    try:
        model.fuse()
        fused = "fused"
    except Exception:  # noqa: BLE001 - fuse is an optimization, not a requirement
        fused = "unfused"
    # Classification models run at Ultralytics' 224 px classify default; detection at 640 px.
    # Newer checkpoints carry a `task` attribute; the v8-era pickles do not, so fall back to the
    # head module type (Classify ends the cls graphs).
    head_type = type(model.model[-1]).__name__ if hasattr(model, "model") else ""
    is_classify = getattr(model, "task", None) == "classify" or head_type == "Classify"
    input_size = 224 if is_classify else 640
    input_tensor = torch.zeros(1, 3, input_size, input_size)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(input_tensor)
        samples = []
        for _ in range(timed_runs):
            started = time.perf_counter()
            model(input_tensor)
            samples.append((time.perf_counter() - started) * 1e3)

    samples.sort()
    model_id = checkpoint.removesuffix(".pt").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    print(
        f"{model_id:>9}: {samples[len(samples) // 2]:>7.1f} ms median, {samples[0]:>7.1f} ms min  "
        f"(single image, batch 1, {input_size} px, {timed_runs} runs, "
        f"PyTorch {torch.__version__} CPU, {fused})",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="official .pt checkpoints under target/")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=10)
    args = parser.parse_args()

    print(f"torch threads: {torch.get_num_threads()}", flush=True)
    for checkpoint in args.checkpoints:
        benchmark(checkpoint, args.warmup_runs, args.timed_runs)


if __name__ == "__main__":
    main()
