"""Plot the same-revision no-LTO, fat-LTO, and Ultralytics training comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


NO_LTO = "#94a3b8"
FAT_LTO = "#16a34a"
ULTRALYTICS = "#0284c7"
GRID = "#dbe3ee"
TEXT = "#172033"
TASKS = ("classify", "detect", "segment")
FAMILIES = ("yolov8n", "yolo11n", "yolo26n")


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


def native_median(entry: dict) -> float:
    return float(entry["summary"]["native"]["median_seconds"])


def ultralytics_samples(no_lto: dict, fat_lto: dict) -> list[float]:
    return [
        float(trial["wall_seconds"])
        for entry in (no_lto, fat_lto)
        for trial in entry["trials"]["ultralytics"]
    ]


def ultralytics_median(no_lto: dict, fat_lto: dict) -> float:
    return statistics.median(ultralytics_samples(no_lto, fat_lto))


def label(scenario_id: str) -> str:
    return scenario_id.replace("yolov", "YOLOv").replace("yolo", "YOLO")


def finish(figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def three_values(no_lto: dict, fat_lto: dict) -> tuple[float, float, float]:
    return (
        native_median(no_lto),
        native_median(fat_lto),
        ultralytics_median(no_lto, fat_lto),
    )


def family_task_overview(
    no_lto: dict[str, dict], fat_lto: dict[str, dict], output: Path
) -> None:
    ids = [f"{family}-{task}" for family in FAMILIES for task in TASKS]
    ids = [scenario_id for scenario_id in ids if scenario_id in no_lto and scenario_id in fat_lto]
    values = [three_values(no_lto[scenario_id], fat_lto[scenario_id]) for scenario_id in ids]
    y = list(range(len(ids)))
    figure, axis = plt.subplots(figsize=(12.5, 8))
    axis.barh(
        [value + 0.25 for value in y],
        [value[0] for value in values],
        height=0.23,
        color=NO_LTO,
        label="Montgomery / no LTO",
    )
    axis.barh(
        y,
        [value[1] for value in values],
        height=0.23,
        color=FAT_LTO,
        label="Montgomery / fat LTO",
    )
    axis.barh(
        [value - 0.25 for value in y],
        [value[2] for value in values],
        height=0.23,
        color=ULTRALYTICS,
        label="Ultralytics / pooled",
    )
    axis.set_yticks(y, [label(scenario_id) for scenario_id in ids])
    axis.invert_yaxis()
    axis.set_xlabel("External command wall time (seconds, lower is better)")
    axis.set_title(
        "Training time across families and tasks",
        loc="left",
        fontsize=20,
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "family-task-three-way.png")


def axis_scaling(
    no_lto: dict[str, dict], fat_lto: dict[str, dict], output: Path, axis_kind: str
) -> None:
    if axis_kind == "scale":
        x_values: tuple[str | int, ...] = ("n", "s", "m")
        path = output / "model-scale-three-way.png"
        title = "YOLO26 model-scale behavior"
        xlabel = "Model scale"

        def scenario_id(task: str, value: str | int) -> str:
            return f"yolo26{value}-{task}"

    else:
        x_values = (1, 2, 4)
        path = output / "batch-scaling-three-way.png"
        title = "YOLO26n batch-size behavior"
        xlabel = "Batch size"

        def scenario_id(task: str, value: str | int) -> str:
            return f"yolo26n-{task}" if value == 2 else f"yolo26n-{task}-batch{value}"

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)
    for task, axis in zip(TASKS, axes):
        ids = [scenario_id(task, value) for value in x_values]
        present = [
            (value, scenario)
            for value, scenario in zip(x_values, ids)
            if scenario in no_lto and scenario in fat_lto
        ]
        xs = [value for value, _ in present]
        triples = [three_values(no_lto[scenario], fat_lto[scenario]) for _, scenario in present]
        for index, color, series_label in (
            (0, NO_LTO, "Montgomery / no LTO"),
            (1, FAT_LTO, "Montgomery / fat LTO"),
            (2, ULTRALYTICS, "Ultralytics / pooled"),
        ):
            axis.plot(
                xs,
                [triple[index] for triple in triples],
                marker="o",
                linewidth=2.5,
                color=color,
                label=series_label,
            )
        axis.set_title(task.title(), fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Seconds")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(title, x=0.04, ha="left", fontsize=20, fontweight="bold")
    finish(figure, path)


def resolution_scaling(
    no_lto: dict[str, dict], fat_lto: dict[str, dict], output: Path
) -> None:
    resolutions = (64, 128, 320, 640)
    ids = [
        f"yolo26n-detect-{resolution}px" if resolution != 320 else "yolo26n-detect"
        for resolution in resolutions
    ]
    triples = [three_values(no_lto[scenario], fat_lto[scenario]) for scenario in ids]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for index, color, series_label in (
        (0, NO_LTO, "Montgomery / no LTO"),
        (1, FAT_LTO, "Montgomery / fat LTO"),
        (2, ULTRALYTICS, "Ultralytics / pooled"),
    ):
        axis.plot(
            resolutions,
            [triple[index] for triple in triples],
            marker="o",
            linewidth=3,
            color=color,
            label=series_label,
        )
    axis.set_title(
        "YOLO26n detection resolution scaling",
        loc="left",
        fontsize=20,
        fontweight="bold",
    )
    axis.set_xlabel("Square training canvas (pixels)")
    axis.set_ylabel("External command wall time (seconds)")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "resolution-scaling-three-way.png")


def convergence_times(
    no_lto: dict[str, dict], fat_lto: dict[str, dict], output: Path
) -> None:
    ids = [f"yolo26n-{task}-10epochs" for task in TASKS]
    triples = [three_values(no_lto[scenario], fat_lto[scenario]) for scenario in ids]
    x = list(range(len(ids)))
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    width = 0.24
    for offset, index, color, series_label in (
        (-width, 0, NO_LTO, "Montgomery / no LTO"),
        (0, 1, FAT_LTO, "Montgomery / fat LTO"),
        (width, 2, ULTRALYTICS, "Ultralytics / pooled"),
    ):
        bars = axis.bar(
            [value + offset for value in x],
            [triple[index] for triple in triples],
            width=width,
            color=color,
            label=series_label,
        )
        axis.bar_label(bars, fmt="%.1fs", padding=3, fontsize=9)
    axis.set_xticks(x, [task.title() for task in TASKS])
    axis.set_ylabel("External command wall time (seconds)")
    axis.set_title(
        "Ten-epoch training command time",
        loc="left",
        fontsize=20,
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3)
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "convergence-time-three-way.png")


def all_scenarios_speedup(
    no_lto: dict[str, dict], fat_lto: dict[str, dict], output: Path
) -> None:
    values = []
    for scenario_id, fat_entry in fat_lto.items():
        no_entry = no_lto[scenario_id]
        ultra = ultralytics_median(no_entry, fat_entry)
        values.append(
            (
                scenario_id,
                ultra / native_median(no_entry),
                ultra / native_median(fat_entry),
            )
        )
    values.sort(key=lambda value: value[2])
    figure, axis = plt.subplots(figsize=(12.5, max(7, len(values) * 0.38)))
    for row, (_, no_speedup, fat_speedup) in enumerate(values):
        axis.plot(
            [no_speedup, fat_speedup],
            [row, row],
            color="#cbd5e1",
            linewidth=2,
            zorder=1,
        )
    axis.scatter(
        [value[1] for value in values],
        range(len(values)),
        color=NO_LTO,
        s=48,
        label="Montgomery / no LTO",
        zorder=2,
    )
    axis.scatter(
        [value[2] for value in values],
        range(len(values)),
        color=FAT_LTO,
        s=48,
        label="Montgomery / fat LTO",
        zorder=3,
    )
    axis.axvline(1, color=ULTRALYTICS, linewidth=2, label="Ultralytics baseline")
    axis.set_xscale("log", base=2)
    ticks = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    axis.set_xticks(ticks, [f"{tick:g}x" for tick in ticks])
    axis.set_yticks(range(len(values)), [value[0] for value in values])
    axis.set_xlabel("Speedup over pooled Ultralytics wall time (>1 favors Montgomery)")
    axis.set_title(
        "All 30 scenarios: speedup over Ultralytics",
        loc="left",
        fontsize=20,
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "all-scenarios-three-way.png")


def group_summary(
    no_lto: dict[str, dict], fat_lto: dict[str, dict], output: Path
) -> None:
    groups: dict[str, list[tuple[float, float]]] = {}
    for scenario_id, fat_entry in fat_lto.items():
        no_entry = no_lto[scenario_id]
        ultra = ultralytics_median(no_entry, fat_entry)
        groups.setdefault(fat_entry["scenario"]["group"], []).append(
            (
                ultra / native_median(no_entry),
                ultra / native_median(fat_entry),
            )
        )
    labels = ["family-task", "scale", "batch", "resolution", "convergence"]
    no_values = [
        math.exp(statistics.mean(math.log(value[0]) for value in groups[group]))
        for group in labels
    ]
    fat_values = [
        math.exp(statistics.mean(math.log(value[1]) for value in groups[group]))
        for group in labels
    ]
    display = ["Family / task", "Scale", "Batch", "Resolution", "10 epochs"]
    x = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    width = 0.34
    axis.bar(
        [value - width / 2 for value in x],
        no_values,
        width=width,
        color=NO_LTO,
        label="Montgomery / no LTO",
    )
    axis.bar(
        [value + width / 2 for value in x],
        fat_values,
        width=width,
        color=FAT_LTO,
        label="Montgomery / fat LTO",
    )
    axis.axhline(1, color=ULTRALYTICS, linewidth=2, label="Ultralytics baseline")
    axis.set_xticks(x, display)
    axis.set_ylabel("Geometric-mean speedup over pooled Ultralytics")
    axis.set_title(
        "Aggregate speedup by benchmark group",
        loc="left",
        fontsize=20,
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3)
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "group-speedup-three-way.png")


def trial_ranges(
    no_lto: dict[str, dict], fat_lto: dict[str, dict], output: Path
) -> None:
    ids = [f"{family}-{task}" for family in FAMILIES for task in TASKS]
    figure, axis = plt.subplots(figsize=(12.5, 7.5))
    for row, scenario_id in enumerate(ids):
        no_entry = no_lto[scenario_id]
        fat_entry = fat_lto[scenario_id]
        series = (
            (
                0.22,
                [float(trial["wall_seconds"]) for trial in no_entry["trials"]["native"]],
                NO_LTO,
            ),
            (
                0,
                [float(trial["wall_seconds"]) for trial in fat_entry["trials"]["native"]],
                FAT_LTO,
            ),
            (-0.22, ultralytics_samples(no_entry, fat_entry), ULTRALYTICS),
        )
        for offset, samples, color in series:
            axis.scatter(samples, [row + offset] * len(samples), color=color, s=35, alpha=0.8)
            axis.plot([min(samples), max(samples)], [row + offset] * 2, color=color, linewidth=2)
    axis.scatter([], [], color=NO_LTO, label="Montgomery / no LTO")
    axis.scatter([], [], color=FAT_LTO, label="Montgomery / fat LTO")
    axis.scatter([], [], color=ULTRALYTICS, label="Ultralytics / pooled")
    axis.set_yticks(range(len(ids)), ids)
    axis.invert_yaxis()
    axis.set_xlabel("Seconds; dots are individual trials")
    axis.set_title(
        "Family/task trial repeatability",
        loc="left",
        fontsize=20,
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "trial-repeatability-three-way.png")


def winner_counts(
    no_lto: dict[str, dict], fat_lto: dict[str, dict], output: Path
) -> None:
    counts = {"No LTO": 0, "Fat LTO": 0, "Ultralytics": 0}
    for scenario_id, fat_entry in fat_lto.items():
        no_entry = no_lto[scenario_id]
        values = {
            "No LTO": native_median(no_entry),
            "Fat LTO": native_median(fat_entry),
            "Ultralytics": ultralytics_median(no_entry, fat_entry),
        }
        counts[min(values, key=values.get)] += 1
    labels = list(counts)
    values = [counts[value] for value in labels]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(labels, values, color=[NO_LTO, FAT_LTO, ULTRALYTICS], width=0.6)
    axis.bar_label(bars, padding=5, fontsize=14, fontweight="bold")
    axis.set_ylim(0, max(values) + 4)
    axis.set_ylabel("Scenarios with the lowest median wall time")
    axis.set_title(
        "Fastest implementation across 30 scenarios",
        loc="left",
        fontsize=20,
        fontweight="bold",
    )
    axis.spines[["top", "right"]].set_visible(False)
    finish(figure, output / "winner-counts-three-way.png")


def validate(no_lto_data: dict, fat_lto_data: dict) -> None:
    no_revision = no_lto_data.get("source", {}).get("revision")
    fat_revision = fat_lto_data.get("source", {}).get("revision")
    if no_revision != fat_revision:
        raise ValueError(f"source revisions differ: {no_revision} != {fat_revision}")
    no_ids = {entry["scenario"]["id"] for entry in no_lto_data["scenarios"]}
    fat_ids = {entry["scenario"]["id"] for entry in fat_lto_data["scenarios"]}
    if no_ids != fat_ids:
        raise ValueError("benchmark scenario sets differ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("no_lto", type=Path)
    parser.add_argument("fat_lto", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    configure()
    no_data, no_lto = load(args.no_lto)
    fat_data, fat_lto = load(args.fat_lto)
    validate(no_data, fat_data)
    family_task_overview(no_lto, fat_lto, args.output)
    axis_scaling(no_lto, fat_lto, args.output, "scale")
    axis_scaling(no_lto, fat_lto, args.output, "batch")
    resolution_scaling(no_lto, fat_lto, args.output)
    convergence_times(no_lto, fat_lto, args.output)
    all_scenarios_speedup(no_lto, fat_lto, args.output)
    group_summary(no_lto, fat_lto, args.output)
    trial_ranges(no_lto, fat_lto, args.output)
    winner_counts(no_lto, fat_lto, args.output)


if __name__ == "__main__":
    main()
