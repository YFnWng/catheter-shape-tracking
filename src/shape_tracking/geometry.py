"""Curve assembly, resampling, differential geometry, and quality metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq, minimize


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


def assemble_image_only_shape(
        visible_points_base_mm: np.ndarray,
        distal_boundary_fraction: float,
        full_count: int = 128,
        distal_count: int = 64,
        base_position_base_mm: np.ndarray | None = None,
        bridge_spacing_mm: float = 1.0,
        snap_distance_mm: float = 1.0,
        bridge_base: bool = True,
        distal_length_mm: float | None = None) -> AssembledShape:
    """Assemble an image-observed catheter, optionally bridging its base."""
    visible = np.asarray(visible_points_base_mm, dtype=np.float64)
    if visible.ndim != 2 or visible.shape[1] != 3 or len(visible) < 4:
        raise ValueError("visible curve must be an (N,3) array with N >= 4")
    if not np.all(np.isfinite(visible)):
        raise ValueError("visible curve contains non-finite coordinates")
    base = (np.zeros(3, dtype=np.float64) if base_position_base_mm is None
            else np.asarray(base_position_base_mm, dtype=np.float64))
    visible_s = cumulative_arclength(visible)
    if visible_s[-1] <= 1e-9:
        raise ValueError("visible curve has zero arc length")
    boundary_visible_s = float(np.clip(
        distal_boundary_fraction, 0.0, 1.0)) * visible_s[-1]
    boundary_index = int(np.clip(
        np.searchsorted(visible_s, boundary_visible_s), 1, len(visible) - 2))
    labels = np.full(len(visible), OBSERVED_PROXIMAL, dtype=np.uint8)
    labels[boundary_index:] = OBSERVED_DISTAL

    base_distance = float(np.linalg.norm(visible[0] - base))
    if not bridge_base:
        source_points = visible.copy()
        if base_distance <= snap_distance_mm:
            source_points[0] = base
        source_labels = labels
        base_bridge_length = 0.0
    elif base_distance <= snap_distance_mm:
        source_points = visible.copy()
        source_points[0] = base
        source_labels = labels
        base_bridge_length = base_distance
    else:
        first_tangent = visible[1] - visible[0]
        bridge = _hermite_bridge(
            base, visible[0], first_tangent, first_tangent, bridge_spacing_mm)
        source_points = np.vstack([bridge[:-1], visible])
        source_labels = np.concatenate([
            np.full(len(bridge) - 1, BASE_BRIDGE, dtype=np.uint8), labels])
        base_bridge_length = float(cumulative_arclength(bridge)[-1])

    source_s = cumulative_arclength(source_points)
    if distal_length_mm is not None:
        target_length = float(distal_length_mm)
        if target_length <= 0.0:
            raise ValueError("distal length must be positive")
        if source_s[-1] + 1e-6 < target_length:
            raise ValueError(
                f"visible_curve_shorter_than_distal:{source_s[-1]:.2f}mm")
        distal_boundary_s = float(source_s[-1] - target_length)
        right = int(np.searchsorted(source_s, distal_boundary_s, side="right"))
        right = int(np.clip(right, 1, len(source_points) - 1))
        left = right - 1
        denominator = max(source_s[right] - source_s[left], 1e-12)
        alpha = (distal_boundary_s - source_s[left]) / denominator
        boundary_point = (
            (1.0 - alpha) * source_points[left]
            + alpha * source_points[right])
        distal_source = np.vstack([boundary_point, source_points[right:]])
        full_sampled, full_s, _ = resample_polyline(
            source_points, full_count)
        full_labels = np.where(
            full_s >= distal_boundary_s,
            OBSERVED_DISTAL, OBSERVED_PROXIMAL).astype(np.uint8)
        if bridge_base and base_distance > snap_distance_mm:
            full_labels[full_s < base_bridge_length] = BASE_BRIDGE
        distal_sampled, distal_s, _ = resample_polyline(
            distal_source, distal_count)
        distal_labels = np.full(
            distal_count, OBSERVED_DISTAL, dtype=np.uint8)
        return AssembledShape(
            full_points_mm=full_sampled,
            full_s_mm=full_s,
            full_observation_class=full_labels,
            distal_points_mm=distal_sampled,
            distal_s_mm=distal_s,
            distal_observation_class=distal_labels,
            distal_boundary_s_mm=distal_boundary_s,
            tip_bridge_length_mm=0.0,
            base_bridge_length_mm=base_bridge_length,
        )

    distal_indices = np.flatnonzero(source_labels == OBSERVED_DISTAL)
    if len(distal_indices) == 0:
        raise ValueError("assembled curve has no distal segment")
    distal_start = int(distal_indices[0])
    distal_boundary_s = float(source_s[distal_start])
    full_sampled, full_s, full_labels = resample_polyline(
        source_points, full_count, source_labels)
    distal_sampled, distal_s, distal_labels = resample_polyline(
        source_points[distal_start:], distal_count,
        source_labels[distal_start:])
    return AssembledShape(
        full_points_mm=full_sampled,
        full_s_mm=full_s,
        full_observation_class=full_labels,
        distal_points_mm=distal_sampled,
        distal_s_mm=distal_s,
        distal_observation_class=distal_labels,
        distal_boundary_s_mm=distal_boundary_s,
        tip_bridge_length_mm=0.0,
        base_bridge_length_mm=base_bridge_length,
    )


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
    spline_degree: int
    spline_basis_count: int
    spline_internal_knot_count: int
    spline_smoothing_lambda: float
    spline_rms_residual_mm: float
    spline_arc_length_mm: float


def curve_geometry(
        points_mm: np.ndarray,
        smoothing_mm: float = 0.25,
        basis_count: int = 12,
        target_arc_length_mm: float | None = None) -> CurveGeometry:
    """Return a penalized multi-basis 3D spline and its geometry.

    A clamped uniform cubic B-spline is fit jointly in x/y/z. Its endpoints are
    exact, and a second-difference penalty on its control points suppresses
    noise without allowing FITPACK to collapse the fit to one global cubic.
    Curvature uses the parameterization-invariant cross-product expression and
    is returned in ``1/mm``.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    if len(points) < 8:
        raise ValueError("multi-basis curvature spline requires at least eight points")
    source_s = cumulative_arclength(points)
    if source_s[-1] <= 1e-9:
        raise ValueError("curve has zero arc length")
    u = source_s / source_s[-1]
    degree = 3
    bases = int(np.clip(int(basis_count), degree + 2, len(points)))
    if bases < 8:
        raise ValueError("curvature spline requires at least eight basis functions")
    internal_count = bases - degree - 1
    internal_knots = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    knots = np.concatenate([
        np.zeros(degree + 1), internal_knots, np.ones(degree + 1)])
    design = BSpline.design_matrix(u, knots, degree).toarray()

    # Clamped knots make the first and last coefficients the endpoints. Solve
    # only for interior coefficients so the observed endpoint positions stay
    # exact while their derivatives remain consistent with the fitted spline.
    fixed_indices = np.array([0, bases - 1])
    free_indices = np.arange(1, bases - 1)
    coefficients = np.zeros((bases, 3), dtype=np.float64)
    coefficients[0] = points[0]
    coefficients[-1] = points[-1]
    adjusted = points - design[:, fixed_indices] @ coefficients[fixed_indices]
    free_design = design[:, free_indices]
    second_difference = np.diff(np.eye(bases), n=2, axis=0)
    free_penalty = second_difference[:, free_indices]
    fixed_penalty = second_difference[:, fixed_indices]
    smoothing_lambda = float(max(smoothing_mm, 0.0) ** 2)
    normal = (
        free_design.T @ free_design
        + smoothing_lambda * (free_penalty.T @ free_penalty)
        + 1e-10 * np.eye(len(free_indices)))
    rhs = (
        free_design.T @ adjusted
        - smoothing_lambda
        * free_penalty.T @ fixed_penalty @ coefficients[fixed_indices])
    coefficients[free_indices] = np.linalg.solve(normal, rhs)

    if target_arc_length_mm is not None:
        target_length = float(target_arc_length_mm)
        endpoint_distance = float(np.linalg.norm(points[-1] - points[0]))
        if target_length + 1e-6 < endpoint_distance:
            raise ValueError(
                "target spline arc length is shorter than endpoint distance")
        # Preserve the fitted bending mode while correcting only its amplitude
        # about the endpoint chord.  A high-dimensional equality optimizer can
        # satisfy length by inventing an off-image loop; this scalar correction
        # cannot change the bend's direction or introduce a new mode.
        segment_design = np.diff(design, axis=0)
        chord = np.linspace(points[0], points[-1], bases)
        deviation = coefficients - chord

        def scaled_length(scale: float) -> float:
            trial = chord + float(scale) * deviation
            return float(np.sum(np.linalg.norm(
                segment_design @ trial, axis=1)))

        current_length = scaled_length(1.0)
        if abs(current_length - target_length) > 1e-8:
            if current_length > target_length:
                lower, upper = 0.0, 1.0
            else:
                lower, upper = 1.0, 2.0
                while scaled_length(upper) < target_length and upper < 1024.0:
                    upper *= 2.0
            if (scaled_length(lower) > target_length + 1e-8
                    or scaled_length(upper) < target_length - 1e-8):
                raise ValueError("target_spline_arc_length_constraint_failed")
            scale = brentq(
                lambda value: scaled_length(value) - target_length,
                lower, upper, xtol=1e-12, rtol=1e-12)
            coefficients = chord + scale * deviation

        # The scalar correction above supplies a feasible initial point.  Now
        # recover the closest penalized multi-basis fit while remaining on the
        # exact-length manifold.  Starting feasible avoids the convergence
        # failures seen when SLSQP was initialized by a shortened smooth fit.
        flat_normal = np.kron(normal, np.eye(3))
        flat_rhs = rhs.reshape(-1)

        def objective(free_flat):
            return float(
                0.5 * free_flat @ flat_normal @ free_flat
                - flat_rhs @ free_flat)

        def objective_jac(free_flat):
            return flat_normal @ free_flat - flat_rhs

        def length_constraint(free_flat):
            trial = coefficients.copy()
            trial[free_indices] = free_flat.reshape(-1, 3)
            return float(np.sum(np.linalg.norm(
                segment_design @ trial, axis=1)) - target_length)

        def length_constraint_jac(free_flat):
            trial = coefficients.copy()
            trial[free_indices] = free_flat.reshape(-1, 3)
            segments = segment_design @ trial
            unit = segments / np.clip(
                np.linalg.norm(segments, axis=1)[:, None], 1e-12, None)
            return (segment_design.T @ unit)[free_indices].reshape(-1)

        optimized = minimize(
            objective, coefficients[free_indices].reshape(-1),
            jac=objective_jac, method="SLSQP",
            constraints={
                "type": "eq", "fun": length_constraint,
                "jac": length_constraint_jac},
            options={"ftol": 1e-10, "maxiter": 500, "disp": False})
        if not optimized.success:
            raise ValueError("target_spline_arc_length_constraint_failed")
        coefficients[free_indices] = optimized.x.reshape(-1, 3)

    spline = BSpline(knots, coefficients, degree, axis=0)
    fitted = np.asarray(spline(u), dtype=np.float64)
    first = np.asarray(spline.derivative(1)(u), dtype=np.float64)
    second = np.asarray(spline.derivative(2)(u), dtype=np.float64)
    speed = np.linalg.norm(first, axis=1)
    safe_speed = np.clip(speed, 1e-9, None)
    tangent = first / safe_speed[:, None]
    curvature = (
        np.linalg.norm(np.cross(first, second), axis=1)
        / safe_speed**3)
    curvature[~np.isfinite(curvature)] = np.nan
    # Differential geometry at an open-spline edge is poorly conditioned, so
    # expose it as missing rather than reporting a meaningless endpoint spike.
    edge = max(2, int(np.ceil(0.03 * len(points))))
    curvature[:edge] = np.nan
    curvature[-edge:] = np.nan
    fitted_arc_length = float(cumulative_arclength(fitted)[-1])
    return CurveGeometry(
        points_mm=fitted,
        tangent=tangent,
        curvature_per_mm=curvature,
        spline_degree=degree,
        spline_basis_count=bases,
        spline_internal_knot_count=internal_count,
        spline_smoothing_lambda=smoothing_lambda,
        spline_rms_residual_mm=float(np.sqrt(np.mean(np.sum(
            (fitted - points) ** 2, axis=1)))),
        spline_arc_length_mm=fitted_arc_length,
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
