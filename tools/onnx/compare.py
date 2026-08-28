"""Numerical comparison helpers shared by PyTorch and ONNX Runtime validation."""

from __future__ import annotations

import numpy as np


def compare_output(name: str, expected: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    if expected.dtype != actual.dtype:
        raise RuntimeError(f"{name}: dtype mismatch: PyTorch {expected.dtype}, ORT {actual.dtype}")
    if expected.shape != actual.shape:
        raise RuntimeError(f"{name}: shape mismatch: PyTorch {expected.shape}, ORT {actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError(f"{name}: non-finite output")
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-8)
    maximum = float(delta.max(initial=0.0))
    mean = float(delta.mean()) if delta.size else 0.0
    rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    relative = float((delta / denominator).max(initial=0.0))
    tolerance = 1e-2 if name in {"boxes", "output0"} else 2e-4
    if maximum > tolerance:
        index = np.unravel_index(int(delta.argmax()), delta.shape)
        raise RuntimeError(
            f"{name}: parity failed at {index}: PyTorch={expected[index]!r}, ORT={actual[index]!r}, "
            f"max_abs={maximum:.8g}, tolerance={tolerance:.8g}, mean_abs={mean:.8g}, rms={rms:.8g}"
        )
    return {
        "name": name,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "max_abs": maximum,
        "max_rel": relative,
        "mean_abs": mean,
        "rms": rms,
    }
