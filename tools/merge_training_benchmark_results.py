"""Merge additional repeated trials into a canonical training benchmark result."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summary(trials: list[dict]) -> dict[str, float]:
    values = [float(trial["wall_seconds"]) for trial in trials]
    return {
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "mean_seconds": statistics.mean(values),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("extra", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    extra = json.loads(args.extra.read_text(encoding="utf-8"))
    if base.get("schema") != extra.get("schema"):
        raise ValueError("result schemas differ")
    entries = {entry["scenario"]["id"]: entry for entry in base["scenarios"]}
    for extra_entry in extra["scenarios"]:
        scenario_id = extra_entry["scenario"]["id"]
        if scenario_id not in entries:
            base["scenarios"].append(extra_entry)
            entries[scenario_id] = extra_entry
            continue
        entry = entries[scenario_id]
        for framework in ("native", "ultralytics"):
            destination = entry["trials"][framework]
            for trial in extra_entry["trials"][framework]:
                trial["label"] = f"extra-{len(destination) + 1}"
                destination.append(trial)
            entry["summary"][framework] = summary(destination)
        native = entry["summary"]["native"]["median_seconds"]
        ultra = entry["summary"]["ultralytics"]["median_seconds"]
        entry["summary"]["native_over_ultralytics"] = native / ultra
        entry["repeat_count"] = len(entry["trials"]["native"])
    output = args.output or args.base
    output.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
