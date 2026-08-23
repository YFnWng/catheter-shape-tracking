"""Correspondence-free two-view fitting of a catheter B-spline.

The disparity pipeline remains useful for initialization, but it must choose a
reference eye and a point correspondence.  This module instead adjusts one 3D
curve so that its projections explain both images at once.  The graph
centerlines are only an initializer/coverage observation.  Optional medial
ridge samples let the final spline choose the one topology which jointly
satisfies the two masks, the marker ordering, stereo geometry, temporal
continuity, and a single localized high-curvature allowance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .geometry import cumulative_arclength, resample_polyline
from .session import project_points


@dataclass(frozen=True)
class JointSplineResult:
    points_base_mm: np.ndarray
    coefficients_base_mm: np.ndarray
    left_model_mean_px: float
    right_model_mean_px: float
    left_model_p95_px: float
    right_model_p95_px: float
    left_coverage_mean_px: float
    right_coverage_mean_px: float
    initial_symmetric_mean_px: float
    final_symmetric_mean_px: float
    arc_length_mm: float
    length_residual_mm: float
    optimizer_cost: float
    optimizer_evaluations: int
    optimizer_success: bool
    left_sharp_turn_clusters: int = 0
    right_sharp_turn_clusters: int = 0


@lru_cache(maxsize=32)
def _basis(sample_count: int, basis_count: int, degree: int = 3) -> np.ndarray:
    basis_count = int(max(degree + 1, basis_count))
    internal_count = basis_count - degree - 1
    internal = (
        np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
        if internal_count else np.empty(0))
    knots = np.concatenate([
        np.zeros(degree + 1), internal, np.ones(degree + 1)])
    u = np.linspace(0.0, 1.0, int(sample_count))
    design = np.empty((len(u), basis_count), dtype=np.float64)
    for index in range(basis_count):
        coefficient = np.zeros(basis_count, dtype=np.float64)
        coefficient[index] = 1.0
        design[:, index] = BSpline(
            knots, coefficient, degree, extrapolate=False)(u)
    design.setflags(write=False)
    return design


def _resample_2d(points: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < 2:
        raise ValueError("joint spline requires at least two image points")
    keep = np.r_[True, np.linalg.norm(np.diff(values, axis=0), axis=1) > 1e-6]
    values = values[keep]
    source = cumulative_arclength(values)
    if source[-1] <= 1e-6:
        raise ValueError("image centerline has zero length")
    target = np.linspace(0.0, source[-1], int(count))
    return np.column_stack([
        np.interp(target, source, values[:, coordinate])
        for coordinate in range(2)])


def _resample_material_2d(points: np.ndarray, count: int) -> np.ndarray:
    """Resample an already material-indexed image corridor without reordering."""
    values = np.asarray(points, dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < 2:
        raise ValueError("ordered image corridor requires at least two points")
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, int(count))
    return np.column_stack([
        np.interp(target, source, values[:, coordinate])
        for coordinate in range(2)])


def projected_sharp_turn_clusters(
        points_xy: np.ndarray,
        threshold_deg: float = 40.0,
        span: int = 4,
        merge_fraction: float = 0.16) -> int:
    """Count physically separated sharp turns in a smooth projection.

    A broad single bend can cross an angle threshold more than once due to
    sampling noise. It is one turn unless the high-angle regions are separated
    by a substantial material interval.
    """
    points = _resample_material_2d(points_xy, 96)
    points = gaussian_filter1d(points, sigma=1.2, axis=0, mode="nearest")
    if len(points) < 2 * span + 3:
        return 0
    before = points[span:-span] - points[:-2 * span]
    after = points[2 * span:] - points[span:-span]
    denominator = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    cosine = np.divide(
        np.sum(before * after, axis=1), denominator,
        out=np.ones_like(denominator), where=denominator > 1e-6)
    sharp = np.flatnonzero(
        np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        >= float(threshold_deg))
    if not len(sharp):
        return 0
    separation = max(2 * span, int(round(float(merge_fraction) * len(points))))
    return 1 + int(np.count_nonzero(np.diff(sharp) > separation))


def project_point_to_polyline(
        points_xy: np.ndarray, point_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the closest continuous polyline point and normalized arc length."""
    points = np.asarray(points_xy, dtype=np.float64)
    point = np.asarray(point_xy, dtype=np.float64)
    if len(points) < 2 or not np.all(np.isfinite(point)):
        raise ValueError("invalid marker/polyline observation")
    starts, vectors = points[:-1], np.diff(points, axis=0)
    denominator = np.sum(vectors * vectors, axis=1)
    alpha = np.divide(
        np.sum((point - starts) * vectors, axis=1), denominator,
        out=np.zeros(len(vectors)), where=denominator > 1e-12)
    alpha = np.clip(alpha, 0.0, 1.0)
    candidates = starts + alpha[:, None] * vectors
    segment = int(np.argmin(np.sum((candidates - point) ** 2, axis=1)))
    segment_length = np.sqrt(np.maximum(denominator, 0.0))
    total = float(np.sum(segment_length))
    position = float(
        np.sum(segment_length[:segment])
        + alpha[segment] * segment_length[segment])
    return candidates[segment], position / max(total, 1e-12)


def distal_image_observation(
        centerline_xy: np.ndarray,
        interface_marker_xy: np.ndarray | None,
        tip_marker_xy: np.ndarray | None,
        count: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Crop interface-to-tip image evidence and return axial marker points.

    Returned marker points lie on the cyan centerline, not at the potentially
    off-axis annulus centroids.
    """
    curve = np.asarray(centerline_xy, dtype=np.float64)
    axial = np.full((2, 2), np.nan, dtype=np.float64)
    fractions = [0.0, 1.0]
    for index, marker in enumerate((interface_marker_xy, tip_marker_xy)):
        if marker is not None and np.all(np.isfinite(marker)):
            axial[index], fractions[index] = project_point_to_polyline(
                curve, marker)
    start_fraction, stop_fraction = fractions
    if stop_fraction <= start_fraction + 0.05:
        start_fraction, stop_fraction = 0.0, 1.0
        axial[0], axial[1] = curve[0], curve[-1]
    source_s = cumulative_arclength(curve)
    start_s, stop_s = start_fraction * source_s[-1], stop_fraction * source_s[-1]
    target = np.linspace(start_s, stop_s, int(count))
    observed = np.column_stack([
        np.interp(target, source_s, curve[:, coordinate])
        for coordinate in range(2)])
    if not np.all(np.isfinite(axial[0])):
        axial[0] = observed[0]
    if not np.all(np.isfinite(axial[1])):
        axial[1] = observed[-1]
    return observed, axial


def _view_metrics(projected: np.ndarray, observed: np.ndarray) -> tuple[float, ...]:
    model_distance = cKDTree(observed).query(projected)[0]
    coverage_distance = cKDTree(projected).query(observed)[0]
    return (
        float(np.mean(model_distance)),
        float(np.percentile(model_distance, 95)),
        float(np.mean(coverage_distance)))


def _finite_evidence(points_xy: np.ndarray | None) -> np.ndarray | None:
    if points_xy is None:
        return None
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    return points if len(points) >= 2 else None


def _project_spline_with_jacobian(
        camera_matrix: np.ndarray,
        camera_T_base: np.ndarray,
        points_base_mm: np.ndarray,
        design: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Project spline samples and differentiate pixels w.r.t. coefficients.

    ``points = design @ coefficients`` is linear in the coefficients.  Only
    the perspective divide is nonlinear, and its 2-by-3 derivative is formed
    analytically here.  Coefficients are flattened in ``(basis, xyz)`` order.
    """
    points = np.asarray(points_base_mm, dtype=np.float64)
    basis = np.asarray(design, dtype=np.float64)
    transform = np.asarray(camera_T_base, dtype=np.float64)
    rotation_mm = transform[:3, :3] * 1e-3
    camera = points * 1e-3 @ transform[:3, :3].T + transform[:3, 3]
    z = camera[:, 2]
    if np.any(z <= 1e-6):
        raise ValueError("joint spline projection crossed the focal plane")
    K = np.asarray(camera_matrix, dtype=np.float64)
    numerator = camera @ K[:2, :3].T
    pixels = numerator / z[:, None]
    perspective = np.empty((len(points), 2, 3), dtype=np.float64)
    perspective[:] = K[:2, :3][None, :, :] / z[:, None, None]
    perspective[:, :, 2] -= numerator / (z * z)[:, None]
    point_jacobian = perspective @ rotation_mm
    coefficient_jacobian = np.einsum(
        "ib,ikc->ikbc", basis, point_jacobian, optimize=True
    ).reshape(len(points), 2, 3 * basis.shape[1])
    return pixels, coefficient_jacobian


def fit_joint_two_view_spline(
        initial_points_base_mm: np.ndarray,
        left_centerline_xy: np.ndarray,
        right_centerline_xy: np.ndarray,
        K: np.ndarray,
        left_camera_T_base: np.ndarray,
        right_camera_T_base: np.ndarray,
        nominal_length_mm: float,
        left_axial_markers_xy: np.ndarray | None = None,
        right_axial_markers_xy: np.ndarray | None = None,
        left_ridge_evidence_xy: np.ndarray | None = None,
        right_ridge_evidence_xy: np.ndarray | None = None,
        left_ordered_corridor_xy: np.ndarray | None = None,
        right_ordered_corridor_xy: np.ndarray | None = None,
        output_samples: int = 64,
        basis_count: int = 20,
        fit_samples: int = 64,
        coverage_samples: int = 48,
        length_sigma_mm: float = 3.0,
        marker_sigma_px: float = 2.5,
        body_marker_sigma_px: float = 5.0,
        coefficient_prior_sigma_mm: float = 8.0,
        temporal_prior_points_base_mm: np.ndarray | None = None,
        temporal_prior_sigma_mm: float = 2.0,
        local_stretch_sigma_mm: float = 0.75,
        variation_sigma_mm: float = 1.5,
        bending_sigma_mm: float = 0.22,
        turn_fraction: float | None = None,
        turn_half_width_fraction: float = 0.10,
        coverage_weight: float = 0.30,
        corridor_sigma_px: float = 1.5,
        projected_variation_sigma_px: float = 0.15,
        left_observation_weight: float = 1.0,
        right_observation_weight: float = 1.0,
        max_nfev: int = 35,
        analytic_jacobian: bool = True) -> JointSplineResult:
    """Fit one material-coordinate cubic spline to both image views.

    In the unified topology path, paired ordered corridors assign every image
    observation to one fixed material coordinate.  This prevents nearest-
    ridge reassignment between the two arms of a projected V.  Unordered ridge
    evidence remains available only for legacy callers without corridors.
    """
    initial = np.asarray(initial_points_base_mm, dtype=np.float64)
    initial, _, _ = resample_polyline(initial, fit_samples)
    left_observed = _resample_2d(left_centerline_xy, coverage_samples)
    right_observed = _resample_2d(right_centerline_xy, coverage_samples)
    left_corridor = (
        None if left_ordered_corridor_xy is None else
        _resample_material_2d(left_ordered_corridor_xy, fit_samples))
    right_corridor = (
        None if right_ordered_corridor_xy is None else
        _resample_material_2d(right_ordered_corridor_xy, fit_samples))
    if (left_corridor is None) != (right_corridor is None):
        raise ValueError("ordered corridors must be supplied for both eyes")
    design = _basis(fit_samples, basis_count)
    output_design = _basis(output_samples, basis_count)
    coefficients, *_ = np.linalg.lstsq(design, initial, rcond=None)
    initial_coefficients = coefficients.copy()
    temporal_prior_coefficients = None
    if temporal_prior_points_base_mm is not None:
        temporal_points = np.asarray(
            temporal_prior_points_base_mm, dtype=np.float64)
        temporal_points, _, _ = resample_polyline(
            temporal_points, fit_samples)
        temporal_prior_coefficients, *_ = np.linalg.lstsq(
            design, temporal_points, rcond=None)
        # Start in the preceding topology basin. Current ordered evidence can
        # move it, but a bad per-frame disparity initializer cannot choose the
        # opposite V arm before optimization begins.
        coefficients = (
            0.35 * coefficients + 0.65 * temporal_prior_coefficients)
    marker_u = np.array([0.0, 10.0 / 57.0, 30.0 / 57.0, 1.0])
    marker_design_full = _basis(257, basis_count)
    marker_indices = np.rint(marker_u * 256).astype(int)
    marker_design = marker_design_full[marker_indices]

    left_ridge = _finite_evidence(left_ridge_evidence_xy)
    right_ridge = _finite_evidence(right_ridge_evidence_xy)
    left_model_evidence = left_observed if left_ridge is None else left_ridge
    right_model_evidence = right_observed if right_ridge is None else right_ridge
    left_observed_tree = cKDTree(left_model_evidence)
    right_observed_tree = cKDTree(right_model_evidence)
    left_scale = np.sqrt(max(float(left_observation_weight), 1e-6))
    right_scale = np.sqrt(max(float(right_observation_weight), 1e-6))
    segment_design = np.diff(design, axis=0)
    bending_design = np.diff(design, n=2, axis=0)
    variation_design = np.diff(design, n=3, axis=0)
    bending_u = np.linspace(0.0, 1.0, len(bending_design))
    if turn_fraction is None or not np.isfinite(turn_fraction):
        bending_weight = np.ones(len(bending_design), dtype=np.float64)
    else:
        width = max(float(turn_half_width_fraction), 1e-3)
        distance = (bending_u - np.clip(float(turn_fraction), 0.0, 1.0)) / width
        # Exactly one material interval receives a relaxed curvature cost.
        # The non-zero floor prevents a free hinge at the turn.
        bending_weight = 0.12 + 0.88 * (
            1.0 - np.exp(-0.5 * distance * distance))
    bending_jacobian = (
        np.kron(bending_weight[:, None] * bending_design,
                np.eye(3, dtype=np.float64))
        / max(bending_sigma_mm, 1e-6))
    projected_variation_u = np.linspace(0.0, 1.0, fit_samples - 3)
    if turn_fraction is None or not np.isfinite(turn_fraction):
        projected_variation_weight = np.ones(
            fit_samples - 3, dtype=np.float64)
    else:
        width = max(float(turn_half_width_fraction), 1e-3)
        distance = (
            projected_variation_u - np.clip(
                float(turn_fraction), 0.0, 1.0)) / width
        projected_variation_weight = np.sqrt(
            0.12 + 0.88 * (1.0 - np.exp(-0.5 * distance * distance)))
    variation_jacobian = (
        np.kron(variation_design, np.eye(3, dtype=np.float64))
        / max(variation_sigma_mm, 1e-6))
    prior_jacobian = (
        np.eye(3 * basis_count, dtype=np.float64)
        / max(coefficient_prior_sigma_mm, 1e-6))
    temporal_prior_jacobian = (
        np.eye(3 * basis_count, dtype=np.float64)
        / max(temporal_prior_sigma_mm, 1e-6))
    marker_observations = []
    for axial, transform in (
            (left_axial_markers_xy, left_camera_T_base),
            (right_axial_markers_xy, right_camera_T_base)):
        if axial is None:
            continue
        axial = np.asarray(axial, dtype=np.float64)
        for index in range(min(4, len(axial))):
            if np.all(np.isfinite(axial[index])):
                sigma = (
                    marker_sigma_px if index in (0, 3)
                    else body_marker_sigma_px)
                marker_observations.append(
                    (transform, index, axial[index], max(sigma, 1e-6)))

    def projections(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = design @ values
        left = project_points(K, left_camera_T_base, points)[0]
        right = project_points(K, right_camera_T_base, points)[0]
        return points, left, right

    def evaluate(
            flattened: np.ndarray,
            calculate_jacobian: bool) -> tuple[np.ndarray, np.ndarray | None]:
        values = flattened.reshape(basis_count, 3)
        points = design @ values
        left, left_projection_jacobian = _project_spline_with_jacobian(
            K, left_camera_T_base, points, design)
        right, right_projection_jacobian = _project_spline_with_jacobian(
            K, right_camera_T_base, points, design)
        left_model_indices = left_observed_tree.query(left)[1]
        right_model_indices = right_observed_tree.query(right)[1]
        left_coverage_indices = cKDTree(left).query(left_observed)[1]
        right_coverage_indices = cKDTree(right).query(right_observed)[1]
        if left_corridor is not None:
            corridor_scale = 1.0 / max(float(corridor_sigma_px), 1e-6)
            terms = [
                corridor_scale * left_scale * (left - left_corridor).ravel(),
                corridor_scale * right_scale * (right - right_corridor).ravel(),
            ]
            jacobian_terms = None
            if calculate_jacobian:
                jacobian_terms = [
                    corridor_scale * left_scale
                    * left_projection_jacobian.reshape(len(left) * 2, -1),
                    corridor_scale * right_scale
                    * right_projection_jacobian.reshape(len(right) * 2, -1),
                ]
        else:
            terms = [
                left_scale
                * (left - left_model_evidence[left_model_indices]).ravel(),
                right_scale
                * (right - right_model_evidence[right_model_indices]).ravel(),
                coverage_weight * left_scale
                * (left_observed - left[left_coverage_indices]).ravel(),
                coverage_weight * right_scale
                * (right_observed - right[right_coverage_indices]).ravel(),
            ]
            jacobian_terms = None
            if calculate_jacobian:
                jacobian_terms = [
                    left_scale * left_projection_jacobian.reshape(
                        len(left) * 2, -1),
                    right_scale * right_projection_jacobian.reshape(
                        len(right) * 2, -1),
                    -coverage_weight * left_scale * left_projection_jacobian[
                        left_coverage_indices].reshape(
                            len(left_observed) * 2, -1),
                    -coverage_weight * right_scale * right_projection_jacobian[
                        right_coverage_indices].reshape(
                            len(right_observed) * 2, -1),
                ]
        if left_corridor is not None:
            # Suppress *changes* in projected curvature only in the
            # non-bijective topology mode. Normal stereo uses the more
            # calibration-tolerant unordered medial evidence objective.
            projected_scale = 1.0 / max(
                float(projected_variation_sigma_px), 1e-6)
            left_projected_variation = np.diff(left, n=3, axis=0)
            right_projected_variation = np.diff(right, n=3, axis=0)
            terms.extend([
                projected_scale * (
                    projected_variation_weight[:, None]
                    * left_projected_variation).ravel(),
                projected_scale * (
                    projected_variation_weight[:, None]
                    * right_projected_variation).ravel(),
            ])
            if jacobian_terms is not None:
                jacobian_terms.extend([
                    projected_scale * (
                        projected_variation_weight[:, None, None]
                        * np.diff(
                            left_projection_jacobian, n=3, axis=0)
                    ).reshape(len(left_projected_variation) * 2, -1),
                    projected_scale * (
                        projected_variation_weight[:, None, None]
                        * np.diff(
                            right_projection_jacobian, n=3, axis=0)
                    ).reshape(len(right_projected_variation) * 2, -1),
                ])
        marker_points = marker_design @ values
        marker_projection_cache = {}
        for transform, index, axial, sigma in marker_observations:
            key = id(transform)
            if key not in marker_projection_cache:
                marker_projection_cache[key] = _project_spline_with_jacobian(
                    K, transform, marker_points, marker_design)
            projected, projected_jacobian = marker_projection_cache[key]
            terms.append((projected[index] - axial) / sigma)
            if jacobian_terms is not None:
                jacobian_terms.append(projected_jacobian[index] / sigma)
        segments = np.diff(points, axis=0)
        segment_length = np.linalg.norm(segments, axis=1)
        length = float(np.sum(segment_length))
        terms.append(np.asarray([
            (length - float(nominal_length_mm)) / max(length_sigma_mm, 1e-6)]))
        target_segment = length / max(len(segment_length), 1)
        terms.append(
            (segment_length - target_segment) / max(local_stretch_sigma_mm, 1e-6))
        terms.append(
            (bending_weight[:, None] * (bending_design @ values)).ravel()
            / max(bending_sigma_mm, 1e-6))
        # Minimum variation, not minimum curvature: tight bends remain legal.
        terms.append(
            np.diff(points, n=3, axis=0).ravel()
            / max(variation_sigma_mm, 1e-6))
        terms.append(
            (values - initial_coefficients).ravel()
            / max(coefficient_prior_sigma_mm, 1e-6))
        if temporal_prior_coefficients is not None:
            terms.append(
                (values - temporal_prior_coefficients).ravel()
                / max(temporal_prior_sigma_mm, 1e-6))
        if jacobian_terms is not None:
            unit_segments = np.divide(
                segments, segment_length[:, None],
                out=np.zeros_like(segments),
                where=segment_length[:, None] > 1e-9)
            segment_jacobian = (
                segment_design[:, :, None] * unit_segments[:, None, :]
            ).reshape(len(segment_design), 3 * basis_count)
            length_jacobian = np.sum(segment_jacobian, axis=0, keepdims=True)
            jacobian_terms.append(
                length_jacobian / max(length_sigma_mm, 1e-6))
            jacobian_terms.append(
                (segment_jacobian
                 - np.mean(segment_jacobian, axis=0, keepdims=True))
                / max(local_stretch_sigma_mm, 1e-6))
            jacobian_terms.extend([
                bending_jacobian, variation_jacobian, prior_jacobian])
            if temporal_prior_coefficients is not None:
                jacobian_terms.append(temporal_prior_jacobian)
        return (
            np.concatenate(terms),
            None if jacobian_terms is None else np.vstack(jacobian_terms))

    evaluation_cache: dict[str, np.ndarray | None] = {
        "x": None, "residual": None, "jacobian": None}

    def cached_evaluation(
            flattened: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cached_x = evaluation_cache["x"]
        if (cached_x is None
                or not np.array_equal(cached_x, flattened)):
            residual_value, jacobian_value = evaluate(flattened, True)
            evaluation_cache["x"] = np.asarray(flattened).copy()
            evaluation_cache["residual"] = residual_value
            evaluation_cache["jacobian"] = jacobian_value
        return (
            evaluation_cache["residual"],
            evaluation_cache["jacobian"])

    def residual(flattened: np.ndarray) -> np.ndarray:
        if analytic_jacobian:
            return cached_evaluation(flattened)[0]
        return evaluate(flattened, False)[0]

    def jacobian(flattened: np.ndarray) -> np.ndarray:
        return cached_evaluation(flattened)[1]

    _, initial_left, initial_right = projections(initial_coefficients)
    initial_metrics = (
        _view_metrics(initial_left, left_observed),
        _view_metrics(initial_right, right_observed))
    solution = least_squares(
        residual, coefficients.ravel(), method="trf",
        jac=jacobian if analytic_jacobian else "2-point",
        loss="soft_l1", f_scale=1.0, max_nfev=max(1, int(max_nfev)),
        xtol=2e-4, ftol=2e-4, gtol=2e-4)
    final_coefficients = solution.x.reshape(basis_count, 3)
    output_points = output_design @ final_coefficients
    final_left = project_points(K, left_camera_T_base, output_points)[0]
    final_right = project_points(K, right_camera_T_base, output_points)[0]
    left_metrics = _view_metrics(final_left, left_observed)
    right_metrics = _view_metrics(final_right, right_observed)
    length = float(cumulative_arclength(output_points)[-1])
    return JointSplineResult(
        points_base_mm=output_points,
        coefficients_base_mm=final_coefficients,
        left_model_mean_px=left_metrics[0],
        right_model_mean_px=right_metrics[0],
        left_model_p95_px=left_metrics[1],
        right_model_p95_px=right_metrics[1],
        left_coverage_mean_px=left_metrics[2],
        right_coverage_mean_px=right_metrics[2],
        initial_symmetric_mean_px=float(np.mean([
            initial_metrics[0][0], initial_metrics[0][2],
            initial_metrics[1][0], initial_metrics[1][2]])),
        final_symmetric_mean_px=float(np.mean([
            left_metrics[0], left_metrics[2],
            right_metrics[0], right_metrics[2]])),
        arc_length_mm=length,
        length_residual_mm=length - float(nominal_length_mm),
        optimizer_cost=float(solution.cost),
        optimizer_evaluations=int(solution.nfev),
        optimizer_success=bool(solution.success),
        left_sharp_turn_clusters=projected_sharp_turn_clusters(final_left),
        right_sharp_turn_clusters=projected_sharp_turn_clusters(final_right))
