"""One-command native-vs-Ultralytics training benchmark and chart generator.

Full reproducible run:

    uv run --locked tools/benchmark_training.py --publish

Fast iteration on one bottleneck:

    uv run --locked tools/benchmark_training.py \
        --scenario yolov8n-segment --repeats 1 --segmentation-repeats 1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "tools" / "bench_training_matrix.py"
PLOT = ROOT / "tools" / "plot_training_comparison.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "target" / "performance-comparison")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--segmentation-repeats", type=int, default=2)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument(
        "--native-binary",
        type=Path,
        help="benchmark an existing native executable; skips the default release build",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-prime", action="store_true")
    parser.add_argument("--keep-runs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    results = output / "results.json"
    charts = output / "charts"
    if not args.skip_build and args.native_binary is None:
        subprocess.run(
            ["cargo", "build", "--locked", "--release", "--features", "training"],
            cwd=ROOT,
            check=True,
        )

    command = [
        sys.executable,
        str(MATRIX),
        "--output",
        str(results),
        "--repeats",
        str(args.repeats),
        "--segmentation-repeats",
        str(args.segmentation_repeats),
    ]
    for scenario in args.scenario:
        command.extend(("--scenario", scenario))
    if args.native_binary is not None:
        command.extend(("--native-binary", str(args.native_binary.resolve())))
    for enabled, flag in (
        (args.skip_prime, "--skip-prime"),
        (args.keep_runs, "--keep-runs"),
        (args.resume, "--resume"),
    ):
        if enabled:
            command.append(flag)
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(PLOT), str(results), str(charts)], cwd=ROOT, check=True)

    if args.publish:
        destination = ROOT / "docs" / "assets"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results, destination / "training-benchmark-results.json")
        for chart in charts.glob("*.png"):
            shutil.copy2(chart, destination / chart.name)
        print(f"published results and charts to {destination}")


if __name__ == "__main__":
    main()
