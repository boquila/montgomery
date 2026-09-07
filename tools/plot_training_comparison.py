"""Generate the charts for docs/performance-comparison.MD from benchmark JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


NATIVE = "#33da72"
ULTRA = "#0284c7"
GRID = "#dbe3ee"
TEXT = "#172033"
TASKS = ("classify", "detect", "segment")
FAMILIES = ("yolov8n", "yolo11n", "yolo26n")
TRAIN_IMAGES = {"classify": 48, "detect": 32, "segment": 32}


def configure() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": GRID,
            "font.size": 10,
        }
    )


def load(path: Path) -> tuple[dict, dict[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {entry["scenario"]["id"]: entry for entry in data["scenarios"]}


def median(entry: dict, framework: str) -> float:
    return float(entry["summary"][framework]["median_seconds"])


def ratio(entry: dict) -> float:
    return median(entry, "ultralytics") / median(entry, "native")


def finish(figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def family_task_overview(entries: dict[str, dict], output: Path) -> None:
    ids = [f"{family}-{task}" for family in FAMILIES for task in TASKS]
    ids = [scenario_id for scenario_id in ids if scenario_id in entries]
    labels = [scenario_id.replace("yolov", "YOLOv").replace("yolo", "YOLO") for scenario_id in ids]
    native = [median(entries[scenario_id], "native") for scenario_id in ids]
    ultra = [median(entries[scenario_id], "ultralytics") for scenario_id in ids]
    y = list(range(len(ids)))
    figure, axis = plt.subplots(figsize=(12, 7.5))
    axis.barh([value + 0.19 for value in y], native, height=0.36, color=NATIVE, label="Montgomery / Burn-WGPU")
    axis.barh([value - 0.19 for value in y], ultra, height=0.36, color=ULTRA, label="Ultralytics / PyTorch-CUDA")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("External command wall time (seconds, lower is better)")
    axis.set_title("Training command time: 224 px classify / 640 px vision", loc="left", fontsize=20, fontweight="bold")
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "family-task-overview.png")


def speedup_heatmap(entries: dict[str, dict], output: Path) -> None:
    values = []
    for family in FAMILIES:
        row = []
        for task in TASKS:
            entry = entries.get(f"{family}-{task}")
            row.append(math.log2(ratio(entry)) if entry else math.nan)
        values.append(row)
    cmap = LinearSegmentedColormap.from_list("speed", ["#dc2626", "#f8fafc", "#16a34a"])
    figure, axis = plt.subplots(figsize=(9, 5.5))
    image = axis.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3), aspect="auto")
    axis.set_xticks(range(3), [task.title() for task in TASKS])
    axis.set_yticks(range(3), [family.replace("yolov", "YOLOv").replace("yolo", "YOLO") for family in FAMILIES])
    axis.set_title("Montgomery speedup over Ultralytics", loc="left", fontsize=20, fontweight="bold", pad=18)
    for row in range(3):
        for column in range(3):
            if math.isfinite(values[row][column]):
                speedup = 2 ** values[row][column]
                axis.text(column, row, f"{speedup:.2f}×", ha="center", va="center", fontsize=14, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("log2(Ultralytics time / Montgomery time) · green favors Montgomery")
    finish(figure, output / "task-speedup-heatmap.png")


def axis_scaling(entries: dict[str, dict], output: Path, axis_kind: str) -> None:
    if axis_kind == "scale":
        x_values = ("n", "s", "m")
        path = output / "model-scale.png"
        title = "YOLO26 model-scale behavior: 640 x 640, batch 2"
        xlabel = "Model scale"
        scenario_id = lambda task, value: f"yolo26{value}-{task}"
    else:
        x_values = (1, 2, 4)
        path = output / "batch-scaling.png"
        title = "YOLO26n batch-size behavior: 640 x 640, 3 epochs"
        xlabel = "Batch size"
        scenario_id = lambda task, value: (
            f"yolo26n-{task}" if value == 2 else f"yolo26n-{task}-batch{value}"
        )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for task, axis in zip(("detect", "segment"), axes):
        present = [(value, entries.get(scenario_id(task, value))) for value in x_values]
        present = [(value, entry) for value, entry in present if entry]
        xs = [value for value, _ in present]
        axis.plot(xs, [median(entry, "native") for _, entry in present], marker="o", linewidth=2.5, color=NATIVE, label="Montgomery")
        axis.plot(xs, [median(entry, "ultralytics") for _, entry in present], marker="o", linewidth=2.5, color=ULTRA, label="Ultralytics")
        axis.set_title(task.title(), fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Seconds")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.suptitle(title, x=0.04, ha="left", fontsize=20, fontweight="bold")
    finish(figure, path)


def resolution_scaling(entries: dict[str, dict], output: Path) -> None:
    baseline_ids = [
        f"{family}-{task}"
        for family in FAMILIES
        for task in ("detect", "segment")
    ] + [f"{family}-detect" for family in ("yolov3-tinyu", "yolov10n", "yolo12n")]
    baseline_ids = [
        scenario_id
        for scenario_id in baseline_ids
        if scenario_id in entries and f"{scenario_id}-1280px" in entries
    ]
    if not baseline_ids:
        return
    labels = [
        scenario_id.replace("yolov", "YOLOv").replace("yolo", "YOLO")
        for scenario_id in baseline_ids
    ]
    y = list(range(len(baseline_ids)))
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for axis, imgsz in zip(axes, (640, 1280)):
        resolution_entries = [
            entries[scenario_id] if imgsz == 640 else entries[f"{scenario_id}-1280px"]
            for scenario_id in baseline_ids
        ]
        axis.barh(
            [value + 0.19 for value in y],
            [median(entry, "native") for entry in resolution_entries],
            height=0.36,
            color=NATIVE,
            label="Montgomery / Burn-WGPU",
        )
        axis.barh(
            [value - 0.19 for value in y],
            [median(entry, "ultralytics") for entry in resolution_entries],
            height=0.36,
            color=ULTRA,
            label="Ultralytics / PyTorch-CUDA",
        )
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(f"{imgsz} x {imgsz}", fontweight="bold", fontsize=15)
        axis.set_xlabel("External command wall time (seconds)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    figure.suptitle(
        "Real-world training resolutions", x=0.04, ha="left", fontsize=20, fontweight="bold"
    )
    finish(figure, output / "resolution-scaling.png")


def all_scenarios(entries: dict[str, dict], output: Path) -> None:
    ordered = sorted(entries.values(), key=ratio)
    labels = [entry["scenario"]["id"] for entry in ordered]
    values = [ratio(entry) for entry in ordered]
    colors = ["#16a34a" if value >= 1 else "#dc2626" for value in values]
    figure, axis = plt.subplots(figsize=(12, max(6, len(values) * 0.38)))
    axis.barh(range(len(values)), values, color=colors)
    axis.axvline(1, color="#334155", linewidth=1.5)
    axis.set_yticks(range(len(values)), labels)
    axis.set_xlabel("Speedup = Ultralytics wall time / native wall time (>1 favors native)")
    axis.set_title("Every measured scenario", loc="left", fontsize=20, fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "all-scenarios-speedup.png")


def trial_ranges(entries: dict[str, dict], output: Path) -> None:
    selected = [
        entries[key]
        for key in (f"{family}-{task}" for family in FAMILIES for task in TASKS)
        if key in entries
    ]
    labels = []
    for entry in selected:
        scenario = entry["scenario"]
        model = scenario["id"].split("-")[0].replace("yolov", "YOLOv").replace("yolo", "YOLO")
        labels.append(f"{model} — {scenario['task'].title()} ({scenario['imgsz']} px)")

    figure, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)
    for axis, framework, color, title in (
        (axes[0], "native", NATIVE, "Montgomery / Burn-WGPU"),
        (axes[1], "ultralytics", ULTRA, "Ultralytics / PyTorch-CUDA"),
    ):
        for index, entry in enumerate(selected):
            if index % 2 == 0:
                axis.axhspan(index - 0.5, index + 0.5, color="#eef2f7", zorder=0)
            samples = [float(trial["wall_seconds"]) for trial in entry["trials"][framework]]
            sample_median = median(entry, framework)
            axis.plot(
                [min(samples), max(samples)],
                [index, index],
                color=color,
                linewidth=4,
                alpha=0.55,
                solid_capstyle="round",
                zorder=1,
            )
            axis.scatter(
                samples,
                [index] * len(samples),
                color=color,
                edgecolor="white",
                linewidth=0.8,
                s=48,
                alpha=0.8,
                zorder=2,
            )
            axis.scatter(
                [sample_median],
                [index],
                color=color,
                edgecolor=TEXT,
                linewidth=1,
                marker="D",
                s=76,
                zorder=3,
            )
            axis.annotate(
                f"{sample_median:.2f}",
                (sample_median, index),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=TEXT,
                fontsize=9,
                fontweight="bold",
            )
        axis.set_title(title, color=color, fontsize=14, fontweight="bold")
        axis.set_xlabel("External command wall time (seconds)")
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    figure.suptitle(
        "Trial repeatability",
        x=0.06,
        ha="left",
        fontsize=20,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.94,
        "Circles are individual alternating trials; diamonds are medians; lines span min–max.",
        color="#475569",
        fontsize=10,
    )
    finish(figure, output / "trial-repeatability.png")


def convergence(entries: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    found = False
    for task, axis in zip(TASKS, axes):
        entry = entries.get(f"yolo26n-{task}-10epochs")
        if not entry:
            axis.set_visible(False)
            continue
        found = True
        for framework, color, label in (("native", NATIVE, "Montgomery"), ("ultralytics", ULTRA, "Ultralytics")):
            curve = entry["trials"][framework][0]["loss_curve"]
            normalized = [value / curve[0] for value in curve]
            axis.plot(range(1, len(curve) + 1), normalized, marker="o", linewidth=2.5, color=color, label=label)
        axis.axhline(1, color="#94a3b8", linewidth=1)
        axis.set_title(task.title(), fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss / epoch-1 loss")
        axis.spines[["top", "right"]].set_visible(False)
    if found:
        next(axis for axis in axes if axis.get_visible()).legend(frameon=False)
        figure.suptitle("Ten-epoch convergence sanity check", x=0.04, ha="left", fontsize=20, fontweight="bold")
        finish(figure, output / "convergence-normalized.png")
    else:
        plt.close(figure)


def first_vs_warm(entries: dict[str, dict], output: Path) -> None:
    selected = [entry for entry in entries.values() if "native_prime" in entry]
    selected = selected[:15]
    if not selected:
        return
    labels = [entry["scenario"]["id"] for entry in selected]
    first = [entry["native_prime"]["wall_seconds"] for entry in selected]
    warm = [median(entry, "native") for entry in selected]
    x = list(range(len(selected)))
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.bar([value - 0.2 for value in x], first, width=0.4, color="#a7f3c2", label="first suite run")
    axis.bar([value + 0.2 for value in x], warm, width=0.4, color=NATIVE, label="warm median")
    axis.set_xticks(x, labels, rotation=40, ha="right")
    axis.set_ylabel("Seconds")
    axis.set_title("First-run versus warmed native command", loc="left", fontsize=20, fontweight="bold")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "first-vs-warm.png")


def effective_throughput(entries: dict[str, dict], output: Path) -> None:
    ids = [f"{family}-{task}" for family in FAMILIES for task in TASKS]
    ids = [scenario_id for scenario_id in ids if scenario_id in entries]
    labels = [scenario_id for scenario_id in ids]
    native = []
    ultra = []
    for scenario_id in ids:
        entry = entries[scenario_id]
        scenario = entry["scenario"]
        image_epochs = TRAIN_IMAGES[scenario["task"]] * int(scenario["epochs"])
        native.append(image_epochs / median(entry, "native"))
        ultra.append(image_epochs / median(entry, "ultralytics"))
    x = list(range(len(ids)))
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.bar([value - 0.2 for value in x], native, width=0.4, color=NATIVE, label="Montgomery")
    axis.bar([value + 0.2 for value in x], ultra, width=0.4, color=ULTRA, label="Ultralytics")
    axis.set_xticks(x, labels, rotation=40, ha="right")
    axis.set_ylabel("Training image-epochs / wall second")
    axis.set_title("Effective end-to-end throughput", loc="left", fontsize=20, fontweight="bold")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "effective-throughput.png")


def optimization_before_after(
    baseline_entries: dict[str, dict], entries: dict[str, dict], output: Path
) -> None:
    scenario_id = "yolov8n-segment"
    if scenario_id not in baseline_entries or scenario_id not in entries:
        return
    baseline = median(baseline_entries[scenario_id], "native")
    optimized = median(entries[scenario_id], "native")
    ultralytics = median(entries[scenario_id], "ultralytics")
    labels = ["Montgomery before\nvectorization", "Montgomery after\nvectorization", "Ultralytics"]
    values = [baseline, optimized, ultralytics]
    figure, axis = plt.subplots(figsize=(9, 5.8))
    bars = axis.bar(labels, values, color=["#bdf6d1", NATIVE, ULTRA], width=0.62)
    axis.bar_label(bars, labels=[f"{value:.2f}s" for value in values], padding=5, fontsize=12)
    axis.set_ylabel("External command wall time (seconds)")
    figure.suptitle(
        "YOLOv8n segmentation optimization",
        x=0.10,
        y=0.98,
        ha="left",
        fontsize=20,
        fontweight="bold",
    )
    axis.set_title(
        f"Same 48-step workload · native improved {baseline / optimized:.1f}×",
        loc="left",
        color="#475569",
        fontsize=11,
        pad=12,
    )
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "segmentation-before-after.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    configure()
    _, entries = load(args.results)
    family_task_overview(entries, args.output)
    speedup_heatmap(entries, args.output)
    axis_scaling(entries, args.output, "scale")
    axis_scaling(entries, args.output, "batch")
    resolution_scaling(entries, args.output)
    all_scenarios(entries, args.output)
    trial_ranges(entries, args.output)
    convergence(entries, args.output)
    first_vs_warm(entries, args.output)
    effective_throughput(entries, args.output)
    if args.baseline:
        _, baseline_entries = load(args.baseline)
        optimization_before_after(baseline_entries, entries, args.output)


if __name__ == "__main__":
    main()
