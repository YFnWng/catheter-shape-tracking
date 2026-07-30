"""Curve assembly, resampling, differential geometry, and quality metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splprep, splev


OBSERVED_PROXIMAL = np.uint8(1)
OBSERVED_DISTAL = np.uint8(2)
BASE_BRIDGE = np.uint8(3)
TIP_BRIDGE = np.uint8(4)


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        return vector / norm
    if fallback is not None:
        return _normalize(fallback)
    raise ValueError("cannot normalize a near-zero vector")


def cumulative_arclength(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate([
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def resample_polyline(
        points: np.ndarray,
        count: int,
        labels: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resample a polyline at uniform physical arc length."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("polyline requires at least two points")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    points = points[keep]
    if labels is not None:
        labels = np.asarray(labels)[keep]
    if len(points) < 2:
        raise ValueError("polyline has zero arc length")
    source_s = cumulative_arclength(points)
    target_s = np.linspace(0.0, source_s[-1], int(count))
    sampled = np.column_stack([
        np.interp(target_s, source_s, points[:, axis]) for axis in range(3)])
    sampled_labels = None
    if labels is not None:
        right = np.searchsorted(source_s, target_s, side="left")
        right = np.clip(right, 0, len(source_s) - 1)
        left = np.clip(right - 1, 0, len(source_s) - 1)
        choose_right = (
            np.abs(source_s[right] - target_s)
            < np.abs(source_s[left] - target_s))
        nearest = np.where(choose_right, right, left)
        sampled_labels = labels[nearest].astype(np.uint8)
    return sampled, target_s, sampled_labels


def _hermite_bridge(
        start: np.ndarray,
        end: np.ndarray,
        start_tangent: np.ndarray,
        end_tangent: np.ndarray,
        spacing_mm: float = 1.0) -> np.ndarray:
    """Cubic Hermite bridge including both endpoints."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(distance / max(spacing_mm, 0.1))) + 1)
    chord = _normalize(end - start, fallback=start_tangent)
    t0 = _normalize(start_tangent, fallback=chord)
    t1 = _normalize(end_tangent, fallback=chord)
    if np.dot(t0, chord) < 0:
        t0 = -t0
    if np.dot(t1, chord) < 0:
        t1 = -t1
    derivative_scale = max(distance, spacing_mm)
    u = np.linspace(0.0, 1.0, count)[:, None]
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return (
        h00 * start + h10 * derivative_scale * t0
        + h01 * end + h11 * derivative_scale * t1)


@dataclass(frozen=True)
class AssembledShape:
    full_points_mm: np.ndarray
    full_s_mm: np.ndarray
    full_observation_class: np.ndarray
    distal_points_mm: np.ndarray
    distal_s_mm: np.ndarray
    distal_observation_class: np.ndarray
    distal_boundary_s_mm: float
    tip_bridge_length_mm: float
    base_bridge_length_mm: float


def assemble_anchored_shape(
        visible_points_base_mm: np.ndarray,
        distal_boundary_fraction: float,
        tip_position_base_mm: np.ndarray,
        tip_z_direction_base: np.ndarray,
        full_count: int = 128,
        distal_count: int = 64,
        base_position_base_mm: np.ndarray | None = None,
        bridge_spacing_mm: float = 1.0,
        snap_distance_mm: float = 1.0) -> AssembledShape:
    """Connect an observed stereo centerline to the registered base and EM tip.

    ``visible_points_base_mm`` must be ordered base-to-tip.  The image-observed
    material boundary is supplied as normalized visible arc length.
    """
    visible = np.asarray(visible_points_base_mm, dtype=np.float64)
    if visible.ndim != 2 or visible.shape[1] != 3 or len(visible) < 4:
        raise ValueError("visible curve must be an (N,3) array with N >= 4")
    if not np.all(np.isfinite(visible)):
        raise ValueError("visible curve contains non-finite coordinates")
    base = (np.zeros(3, dtype=np.float64) if base_position_base_mm is None
            else np.asarray(base_position_base_mm, dtype=np.float64))
    tip = np.asarray(tip_position_base_mm, dtype=np.float64)
    tip_z = _normalize(np.asarray(tip_z_direction_base, dtype=np.float64))

    visible_s = cumulative_arclength(visible)
    boundary_fraction = float(np.clip(distal_boundary_fraction, 0.0, 1.0))
    boundary_visible_s = boundary_fraction * visible_s[-1]
    boundary_index = int(np.searchsorted(visible_s, boundary_visible_s))
    boundary_index = int(np.clip(boundary_index, 1, len(visible) - 2))
    visible_labels = np.full(len(visible), OBSERVED_PROXIMAL, dtype=np.uint8)
    visible_labels[boundary_index:] = OBSERVED_DISTAL

    # Registered base to the first blue-shaft observation.
    base_distance = float(np.linalg.norm(visible[0] - base))
    if base_distance <= snap_distance_mm:
        visible = visible.copy()
        visible[0] = base
        full_points = visible
        full_labels = visible_labels
        base_bridge_length = base_distance
    else:
        first_tangent = visible[1] - visible[0]
        bridge = _hermite_bridge(
            base, visible[0], first_tangent, first_tangent, bridge_spacing_mm)
        full_points = np.vstack([bridge[:-1], visible])
        full_labels = np.concatenate([
            np.full(len(bridge) - 1, BASE_BRIDGE, dtype=np.uint8),
            visible_labels])
        base_bridge_length = float(cumulative_arclength(bridge)[-1])

    # Last blue observation through tape/housing to the coil-defined tip.
    last_tangent = visible[-1] - visible[-2]
    chord = tip - visible[-1]
    if np.dot(tip_z, chord) < 0:
        tip_z = -tip_z
    tip_distance = float(np.linalg.norm(chord))
    if tip_distance <= snap_distance_mm:
        full_points[-1] = tip
        tip_bridge_length = tip_distance
    else:
        bridge = _hermite_bridge(
            visible[-1], tip, last_tangent, tip_z, bridge_spacing_mm)
        full_points = np.vstack([full_points, bridge[1:]])
        full_labels = np.concatenate([
            full_labels,
            np.full(len(bridge) - 1, TIP_BRIDGE, dtype=np.uint8)])
        tip_bridge_length = float(cumulative_arclength(bridge)[-1])

    source_s = cumulative_arclength(full_points)
    first_distal = np.flatnonzero(
        (full_labels == OBSERVED_DISTAL) | (full_labels == TIP_BRIDGE))
    if len(first_distal) == 0:
        raise ValueError("assembled curve has no distal segment")
    distal_start = int(first_distal[0])
    distal_boundary_s = float(source_s[distal_start])

    full_sampled, full_s, full_sampled_labels = resample_polyline(
        full_points, full_count, full_labels)
    distal_source = full_points[distal_start:]
    distal_source_labels = full_labels[distal_start:]
    distal_sampled, distal_s, distal_sampled_labels = resample_polyline(
        distal_source, distal_count, distal_source_labels)
    return AssembledShape(
        full_points_mm=full_sampled,
        full_s_mm=full_s,
        full_observation_class=full_sampled_labels,
        distal_points_mm=distal_sampled,
        distal_s_mm=distal_s,
        distal_observation_class=distal_sampled_labels,
        distal_boundary_s_mm=distal_boundary_s,
        tip_bridge_length_mm=tip_bridge_length,
        base_bridge_length_mm=base_bridge_length,
    )


@dataclass(frozen=True)
class CurveGeometry:
    points_mm: np.ndarray
    tangent: np.ndarray
    curvature_per_mm: np.ndarray


def curve_geometry(
        points_mm: np.ndarray,
        smoothing_mm: float = 0.25) -> CurveGeometry:
    """Return a smoothed curve, unit tangent, and curvature magnitude.

    The spline parameter is normalized arc length. Curvature uses the
    parameterization-invariant cross-product expression and is returned in
    ``1/mm``.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    if len(points) < 4:
        raise ValueError("curvature requires at least four points")
    source_s = cumulative_arclength(points)
    if source_s[-1] <= 1e-9:
        raise ValueError("curve has zero arc length")
    u = source_s / source_s[-1]
    smooth = float(max(smoothing_mm, 0.0) ** 2 * len(points))
    try:
        tck, _ = splprep(
            points.T, u=u, s=smooth, k=min(3, len(points) - 1))
        fitted = np.asarray(splev(u, tck, der=0)).T
        first = np.asarray(splev(u, tck, der=1)).T
        second = np.asarray(splev(u, tck, der=2)).T
    except (TypeError, ValueError):
        fitted = points.copy()
        first = np.gradient(fitted, u, axis=0, edge_order=2)
        second = np.gradient(first, u, axis=0, edge_order=2)
    speed = np.linalg.norm(first, axis=1)
    safe_speed = np.clip(speed, 1e-9, None)
    tangent = first / safe_speed[:, None]
    curvature = (
        np.linalg.norm(np.cross(first, second), axis=1)
        / safe_speed**3)
    curvature[~np.isfinite(curvature)] = np.nan
    # Smoothing must never move registered hard anchors. Differential geometry
    # at an open-spline edge is poorly conditioned, so expose it as missing
    # rather than reporting a large, physically meaningless endpoint spike.
    fitted[0] = points[0]
    fitted[-1] = points[-1]
    edge = max(2, int(np.ceil(0.03 * len(points))))
    curvature[:edge] = np.nan
    curvature[-edge:] = np.nan
    return CurveGeometry(
        points_mm=fitted,
        tangent=tangent,
        curvature_per_mm=curvature,
    )


def stereo_condition_score(centerline_pixels: np.ndarray) -> float:
    """Heuristic 0..1 observability score for rectified stereo depth.

    A horizontal curve is parallel to the ZED baseline and scores near zero.
    A curve with substantial vertical travel relative to its 2D arc length
    scores closer to one.
    """
    points = np.asarray(centerline_pixels, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    length = float(cumulative_arclength(
        np.column_stack([points, np.zeros(len(points))]))[-1])
    if length <= 1e-9:
        return 0.0
    vertical_travel = float(np.sum(np.abs(np.diff(points[:, 1]))))
    return float(np.clip(vertical_travel / length, 0.0, 1.0))
