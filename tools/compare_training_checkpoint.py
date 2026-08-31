"""Compare native/reference training fixture summaries with explicit tolerances.

Generated fixtures and reports belong under ``target/``. The command exits non-zero on a schema,
shape, discrete-assignment, or numeric tolerance mismatch so it can gate ignored parity tests.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def compare(path: str, expected: Any, actual: Any, atol: float, rtol: float, errors: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path}: expected object")
            return
        for key in expected.keys() | actual.keys():
            if key not in expected or key not in actual:
                errors.append(f"{path}.{key}: missing from {'expected' if key not in expected else 'actual'}")
            else:
                compare(f"{path}.{key}", expected[key], actual[key], atol, rtol, errors)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            errors.append(f"{path}: list shape differs")
            return
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(f"{path}[{index}]", left, right, atol, rtol, errors)
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            if expected != actual:
                errors.append(f"{path}: discrete value {actual} != {expected}")
        elif not (math.isfinite(float(actual)) and math.isclose(float(expected), float(actual), abs_tol=atol, rel_tol=rtol)):
            errors.append(f"{path}: {actual} != {expected} (atol={atol}, rtol={rtol})")
    elif expected != actual:
        errors.append(f"{path}: {actual!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    errors: list[str] = []
    compare("$", expected, actual, args.atol, args.rtol, errors)
    result = {"format": "montgomery-training-comparison-v1", "passed": not errors, "errors": errors}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("\n".join(errors[:100]))
    print("training fixtures match")


if __name__ == "__main__":
    main()
