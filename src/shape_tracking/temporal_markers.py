"""Bidirectional repair of short gaps and snaps in stereo marker tracks."""

from __future__ import annotations

import numpy as np


def _chord_residuals(
        values: np.ndarray, trusted: np.ndarray,
        timestamps_ns: np.ndarray, maximum_gap_ms: float) -> np.ndarray:
    residual = np.full(len(values), np.nan, dtype=np.float64)
    indices = np.flatnonzero(trusted)
    for index in indices:
        before = indices[indices < index]
        after = indices[indices > index]
        if not len(before) or not len(after):
            continue
        previous, following = int(before[-1]), int(after[0])
        duration_ms = (timestamps_ns[following] - timestamps_ns[previous]) * 1e-6
        if duration_ms <= 0.0 or duration_ms > float(maximum_gap_ms):
            continue
        alpha = float(
            (timestamps_ns[index] - timestamps_ns[previous])
            / (timestamps_ns[following] - timestamps_ns[previous]))
        predicted = (1.0 - alpha) * values[previous] + alpha * values[following]
        residual[index] = float(np.linalg.norm(values[index] - predicted))
    return residual


def _repair_one_track(
        values: np.ndarray, widths: np.ndarray, confidence: np.ndarray,
        observed: np.ndarray, timestamps_ns: np.ndarray,
        maximum_gap_ms: float, maximum_chord_residual_px: float,
        interpolated_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).copy()
    widths = np.asarray(widths, dtype=np.float64).copy()
    confidence = np.asarray(confidence, dtype=np.float64).copy()
    direct = (
        np.asarray(observed, dtype=bool)
        & np.all(np.isfinite(values), axis=1))
    trusted = direct.copy()
    # Iteration is needed when a short run contains more than one mutually
    # consistent wrong observation. Removing its largest chord residual first
    # exposes the remaining samples to the same bidirectional test.
    for _ in range(4):
        residual = _chord_residuals(
            values, trusted, timestamps_ns, maximum_gap_ms)
        bad = trusted & np.isfinite(residual) & (
            residual > float(maximum_chord_residual_px))
        if not np.any(bad):
            break
        trusted[bad] = False

    repaired = values.copy()
    repaired_width = widths.copy()
    repaired_confidence = confidence.copy()
    interpolated = np.zeros(len(values), dtype=bool)
    anchors = np.flatnonzero(trusted)
    for index in np.flatnonzero(~trusted):
        before = anchors[anchors < index]
        after = anchors[anchors > index]
        if not len(before) or not len(after):
            continue
        previous, following = int(before[-1]), int(after[0])
        duration_ns = int(timestamps_ns[following] - timestamps_ns[previous])
        if duration_ns <= 0 or duration_ns * 1e-6 > float(maximum_gap_ms):
            continue
        alpha = float(
            (timestamps_ns[index] - timestamps_ns[previous]) / duration_ns)
        repaired[index] = (
            (1.0 - alpha) * values[previous] + alpha * values[following])
        if np.isfinite(widths[[previous, following]]).all():
            repaired_width[index] = (
                (1.0 - alpha) * widths[previous] + alpha * widths[following])
        repaired_confidence[index] = min(
            float(interpolated_confidence),
            float(max(confidence[previous], confidence[following])))
        interpolated[index] = True
    available = trusted | interpolated
    repaired[~available] = np.nan
    repaired_width[~available] = np.nan
    repaired_confidence[~available] = 0.0
    return (repaired, repaired_width, repaired_confidence,
            available.astype(np.uint8), interpolated.astype(np.uint8))


def repair_stereo_marker_tracks(
        centers_by_view: dict[str, np.ndarray],
        widths_by_view: dict[str, np.ndarray],
        confidence_by_view: dict[str, np.ndarray],
        observed_by_view: dict[str, np.ndarray],
        timestamps_ns: np.ndarray,
        maximum_gap_ms: float = 500.0,
        maximum_chord_residual_px: float = 12.0,
        interpolated_confidence: float = 0.55,
) -> dict[str, dict[str, np.ndarray]]:
    """Repair marker tracks in any number of views with future/past evidence.

    The output remains image-space evidence. No ring centroid is made a hard
    curve constraint, and approximate inter-ring distances are not imposed.
    """
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    output: dict[str, dict[str, np.ndarray]] = {}
    views = tuple(centers_by_view)
    if not views:
        return output
    required = (widths_by_view, confidence_by_view, observed_by_view)
    if any(set(mapping) != set(views) for mapping in required):
        raise ValueError("marker track view mappings do not have matching keys")
    for view in views:
        centers = np.asarray(centers_by_view[view], dtype=np.float64).copy()
        widths = np.asarray(widths_by_view[view], dtype=np.float64).copy()
        confidence = np.asarray(
            confidence_by_view[view], dtype=np.float64).copy()
        observed = np.asarray(observed_by_view[view], dtype=np.uint8).copy()
        interpolated = np.zeros_like(observed, dtype=np.uint8)
        for marker in range(centers.shape[1]):
            (centers[:, marker], widths[:, marker], confidence[:, marker],
             observed[:, marker], interpolated[:, marker]) = _repair_one_track(
                centers[:, marker], widths[:, marker], confidence[:, marker],
                observed[:, marker], timestamps, maximum_gap_ms,
                maximum_chord_residual_px, interpolated_confidence)
        output[view] = {
            "centers": centers, "widths": widths,
            "confidence": confidence, "observed": observed,
            "interpolated": interpolated}
    return output
