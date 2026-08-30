"""Plot one training run or compare native and Ultralytics metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_losses(path: Path) -> tuple[list[int], list[float]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"no metric rows in {path}")
    loss_key = "loss" if "loss" in rows[0] else "train/loss"
    epochs = [int(row["epoch"]) for row in rows]
    if epochs[0] == 0:
        epochs = [epoch + 1 for epoch in epochs]
    return epochs, [float(row[loss_key]) for row in rows]


def plot_loss(axis, epochs: list[int], losses: list[float], label: str, color: str) -> None:
    axis.plot(
        epochs,
        losses,
        label=label,
        color=color,
        linewidth=3,
        marker="o",
        markersize=6,
        markerfacecolor="#ffffff",
        markeredgewidth=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--native-seconds", type=float)
    parser.add_argument("--compare-seconds", type=float)
    parser.add_argument("--title", default="Native training smoke run")
    parser.add_argument("--subtitle", default="Training loss by epoch")
    args = parser.parse_args()

    epochs, losses = read_losses(args.metrics)

    plt.style.use("seaborn-v0_8-whitegrid")
    if args.compare:
        compare_epochs, compare_losses = read_losses(args.compare)
        figure, (axis, timing_axis) = plt.subplots(
            1,
            2,
            figsize=(12, 6.75),
            dpi=160,
            gridspec_kw={"width_ratios": [2.1, 1]},
        )
    else:
        figure, axis = plt.subplots(figsize=(12, 6.75), dpi=160)
        timing_axis = None
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#f8fafc")
    plot_loss(axis, epochs, losses, "boquilens / Burn", "#7c3aed")
    if args.compare:
        plot_loss(axis, compare_epochs, compare_losses, "Ultralytics / PyTorch", "#0284c7")
        axis.legend(frameon=False, fontsize=11)
    axis.set_xticks(epochs)
    axis.set_xlabel("Epoch", fontsize=12)
    axis.set_ylabel("Cross-entropy loss", fontsize=12)
    axis.set_title(args.title, loc="left", fontsize=22, fontweight="bold", pad=28)
    axis.text(
        0,
        1.025,
        args.subtitle,
        transform=axis.transAxes,
        color="#475569",
        fontsize=12,
    )
    axis.spines[["top", "right"]].set_visible(False)

    if timing_axis is not None:
        if args.native_seconds is None or args.compare_seconds is None:
            raise ValueError("comparison plots require both timing arguments")
        labels = ["boquilens\nBurn/WGPU", "Ultralytics\nPyTorch/CUDA"]
        timings = [args.native_seconds, args.compare_seconds]
        bars = timing_axis.bar(labels, timings, color=["#7c3aed", "#0284c7"], width=0.62)
        timing_axis.set_facecolor("#f8fafc")
        timing_axis.set_title("Measured command time", fontsize=15, fontweight="bold", pad=14)
        timing_axis.set_ylabel("Seconds · lower is better", fontsize=11)
        timing_axis.spines[["top", "right"]].set_visible(False)
        timing_axis.bar_label(bars, labels=[f"{value:.2f}s" for value in timings], padding=5, fontsize=11)
        timing_axis.set_ylim(0, max(timings) * 1.18)

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
