"""Benchmark official Ultralytics predict() end to end (PyTorch, CPU) for comparison.

Complements bench_ultralytics_cpu.py (forward-only) with the full product path: the timed call
includes Ultralytics' preprocessing (letterbox), model forward, NMS, and result assembly — the
same scope as boquilens' CLI `predict` (which additionally decodes, annotates, and writes a PNG).
The model is loaded and warmed up outside the timed region. Run from the repository root:

    uv run --locked tools\\bench_ultralytics_predict.py target\\yolo11n.pt assets\\dog_bike_man.jpg
"""

from __future__ import annotations

import argparse
import time

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="official .pt checkpoint under target/")
    parser.add_argument("source", help="input image")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=10)
    args = parser.parse_args()

    model = YOLO(args.checkpoint)
    for _ in range(args.warmup_runs):
        model.predict(args.source, verbose=False, device="cpu")

    samples = []
    for _ in range(args.timed_runs):
        started = time.perf_counter()
        results = model.predict(args.source, verbose=False, device="cpu")
        samples.append((time.perf_counter() - started) * 1e3)
        if len(samples) == 1:
            print(f"first timed run detections: {len(results[0].boxes)}", flush=True)

    samples.sort()
    print(
        f"{args.checkpoint}: {samples[len(samples) // 2]:>7.1f} ms median, {samples[0]:>7.1f} ms min "
        f"({args.timed_runs} runs, predict() end to end: preprocess + forward + NMS + results)",
        flush=True,
    )


if __name__ == "__main__":
    main()
