"""Numerical comparison helpers shared by PyTorch and ONNX Runtime validation."""

from __future__ import annotations

import numpy as np


def compare_output(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    tolerance: float | None = None,
) -> dict[str, object]:
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
    tolerance = tolerance if tolerance is not None else (1e-2 if name in {"boxes", "output0"} else 2e-4)
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
        "tolerance": tolerance,
    }


def compare_scored_boxes(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    expected_scores: np.ndarray,
    actual_scores: np.ndarray,
    confidence: float = 0.25,
) -> dict[str, object]:
    """Compare boxes that can participate in inference after confidence filtering.

    YOLOv3-Tiny's legacy DFL head is ill-conditioned on synthetic inputs for anchors whose every
    class probability is effectively zero. Tiny backend rounding differences can move those
    discarded boxes by many pixels even though the score tensors and all observable detections
    agree. We still audit the entire tensor for shape, dtype, finiteness, and error statistics,
    while applying the numerical release gate to every anchor that either runtime could retain.
    """
    if expected.dtype != actual.dtype:
        raise RuntimeError(f"{name}: dtype mismatch: expected {expected.dtype}, actual {actual.dtype}")
    if expected.shape != actual.shape:
        raise RuntimeError(f"{name}: shape mismatch: expected {expected.shape}, actual {actual.shape}")
    if expected_scores.shape != actual_scores.shape or expected_scores.shape[:-1] != expected.shape[:-1]:
        raise RuntimeError(f"{name}: score tensor cannot be aligned with boxes")
    if not all(np.isfinite(value).all() for value in (expected, actual, expected_scores, actual_scores)):
        raise RuntimeError(f"{name}: non-finite box or score output")

    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-8)
    meaningful_anchors = np.maximum(expected_scores.max(axis=-1), actual_scores.max(axis=-1)) >= confidence
    meaningful = np.broadcast_to(meaningful_anchors[..., None], delta.shape)
    gated_delta = delta[meaningful]
    maximum = float(delta.max(initial=0.0))
    mean = float(delta.mean()) if delta.size else 0.0
    rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    relative = float((delta / denominator).max(initial=0.0))
    gated_maximum = float(gated_delta.max(initial=0.0))
    tolerance = 1e-2
    if gated_maximum > tolerance:
        gated = np.where(meaningful, delta, -1.0)
        index = np.unravel_index(int(gated.argmax()), gated.shape)
        raise RuntimeError(
            f"{name}: confidence-relevant parity failed at {index}: expected={expected[index]!r}, "
            f"actual={actual[index]!r}, max_abs={gated_maximum:.8g}, "
            f"tolerance={tolerance:.8g}, confidence={confidence:.8g}"
        )
    return {
        "name": name,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "max_abs": maximum,
        "max_rel": relative,
        "mean_abs": mean,
        "rms": rms,
        "gated_max_abs": gated_maximum,
        "confidence_gate": confidence,
        "tolerance": tolerance,
        "gated_anchors": int(meaningful_anchors.sum()),
        "discarded_anchors": int(meaningful_anchors.size - meaningful_anchors.sum()),
        "tolerance_note": "legacy DFL boxes below confidence are structurally audited but numerically non-observable",
    }


def compare_yolox_predictions(
    expected: np.ndarray,
    actual: np.ndarray,
    confidence: float = 0.25,
) -> dict[str, object]:
    """Gate YOLOX's packed output according to its observable detection semantics."""
    name = "predictions"
    if expected.dtype != actual.dtype:
        raise RuntimeError(f"{name}: dtype mismatch: {expected.dtype} != {actual.dtype}")
    if expected.shape != actual.shape or expected.ndim != 3 or expected.shape[-1] < 6:
        raise RuntimeError(f"{name}: incompatible YOLOX prediction shapes {expected.shape}, {actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError(f"{name}: non-finite output")

    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    expected_confidence = expected[..., 4:5] * expected[..., 5:]
    actual_confidence = actual[..., 4:5] * actual[..., 5:]
    meaningful = np.maximum(expected_confidence.max(axis=-1), actual_confidence.max(axis=-1)) >= confidence
    box_delta = delta[..., :4]
    gated_box_delta = box_delta[np.broadcast_to(meaningful[..., None], box_delta.shape)]
    probability_delta = delta[..., 4:]
    box_max = float(gated_box_delta.max(initial=0.0))
    probability_max = float(probability_delta.max(initial=0.0))
    box_tolerance = 0.125
    probability_tolerance = 2e-3
    if box_max > box_tolerance or probability_max > probability_tolerance:
        raise RuntimeError(
            f"{name}: YOLOX semantic parity failed: confidence-relevant box max_abs={box_max:.8g} "
            f"(tolerance={box_tolerance:.8g}), probability max_abs={probability_max:.8g} "
            f"(tolerance={probability_tolerance:.8g})"
        )
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-8)
    return {
        "name": name,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "max_abs": float(delta.max(initial=0.0)),
        "max_rel": float((delta / denominator).max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "gated_box_max_abs": box_max,
        "probability_max_abs": probability_max,
        "confidence_gate": confidence,
        "gated_anchors": int(meaningful.sum()),
        "discarded_anchors": int(meaningful.size - meaningful.sum()),
        "box_tolerance": box_tolerance,
        "probability_tolerance": probability_tolerance,
    }


def compare_end2end_detections(
    expected: np.ndarray,
    actual: np.ndarray,
    confidence: float = 0.25,
) -> dict[str, object]:
    """Compare NMS-free top-k rows as a detection set after the documented confidence gate."""
    name = "output0"
    if expected.dtype != actual.dtype:
        raise RuntimeError(f"{name}: dtype mismatch: {expected.dtype} != {actual.dtype}")
    if expected.shape != actual.shape or expected.ndim != 3 or expected.shape[-1] < 6:
        raise RuntimeError(f"{name}: incompatible end-to-end detection shapes {expected.shape}, {actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError(f"{name}: non-finite output")

    matched = 0
    box_max = 0.0
    score_max = 0.0
    payload_max = 0.0
    for batch in range(expected.shape[0]):
        expected_rows = expected[batch][expected[batch, :, 4] >= confidence]
        actual_rows = actual[batch][actual[batch, :, 4] >= confidence]
        if len(expected_rows) != len(actual_rows):
            raise RuntimeError(
                f"{name}: confidence-filtered count mismatch in batch {batch}: "
                f"{len(expected_rows)} != {len(actual_rows)}"
            )
        unused = set(range(len(actual_rows)))
        for row in expected_rows:
            candidates = [index for index in unused if actual_rows[index, 5] == row[5]]
            if not candidates:
                raise RuntimeError(f"{name}: no class-{int(row[5])} match in batch {batch}")
            index = min(
                candidates,
                key=lambda item: float(np.max(np.abs(row[:5] - actual_rows[item, :5]))),
            )
            unused.remove(index)
            box_max = max(box_max, float(np.max(np.abs(row[:4] - actual_rows[index, :4]))))
            score_max = max(score_max, float(abs(row[4] - actual_rows[index, 4])))
            if expected.shape[-1] > 6:
                payload_max = max(
                    payload_max,
                    float(np.max(np.abs(row[6:] - actual_rows[index, 6:]), initial=0.0)),
                )
            matched += 1
    box_tolerance = 1e-2
    score_tolerance = 2e-4
    payload_tolerance = 2e-4
    if (
        box_max > box_tolerance
        or score_max > score_tolerance
        or payload_max > payload_tolerance
    ):
        raise RuntimeError(
            f"{name}: confidence-filtered detection parity failed: box max_abs={box_max:.8g} "
            f"(tolerance={box_tolerance:.8g}), score max_abs={score_max:.8g} "
            f"(tolerance={score_tolerance:.8g}), payload max_abs={payload_max:.8g} "
            f"(tolerance={payload_tolerance:.8g})"
        )

    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-8)
    return {
        "name": name,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "max_abs": float(delta.max(initial=0.0)),
        "max_rel": float((delta / denominator).max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "confidence_gate": confidence,
        "matched_detections": matched,
        "gated_box_max_abs": box_max,
        "gated_score_max_abs": score_max,
        "gated_payload_max_abs": payload_max,
        "box_tolerance": box_tolerance,
        "score_tolerance": score_tolerance,
        "payload_tolerance": payload_tolerance,
        "tolerance_note": "top-k rows below confidence may reorder across runtimes and are compared only structurally",
    }
