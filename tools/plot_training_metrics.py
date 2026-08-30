"""Plot epoch loss from a native boquilens training run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="Native training smoke run")
    parser.add_argument("--subtitle", default="Training loss by epoch")
    args = parser.parse_args()

    with args.metrics.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    epochs = [int(row["epoch"]) + 1 for row in rows]
    losses = [float(row["loss"]) for row in rows]
    if not losses:
        raise ValueError(f"no loss rows in {args.metrics}")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(12, 6.75), dpi=160)
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#f8fafc")
    axis.plot(
        epochs,
        losses,
        color="#7c3aed",
        linewidth=3,
        marker="o",
        markersize=7,
        markerfacecolor="#ffffff",
        markeredgewidth=2,
    )
    axis.fill_between(epochs, losses, min(losses) - 0.05, color="#7c3aed", alpha=0.08)
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
    axis.annotate(
        f"{losses[-1]:.3f}",
        (epochs[-1], losses[-1]),
        xytext=(-8, 14),
        textcoords="offset points",
        ha="right",
        color="#5b21b6",
        fontweight="bold",
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
