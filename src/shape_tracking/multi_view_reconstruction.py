"""Common-spline reconstruction from an arbitrary set of calibrated views.

This module is deliberately separate from ``joint_spline_reconstruction``.
The existing two-eye pipeline remains unchanged; dual-ZED processing uses the
generalized objective below with four independently calibrated observations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .geometry import cumulative_arclength, resample_polyline
from .joint_spline_reconstruction import (
    _basis,
    _project_spline_with_jacobian,
    _resample_2d,
    _view_metrics,
    projected_sharp_turn_clusters,
)
from .session import project_points


@dataclass(frozen=True)
class ViewObservation:
    view_id: str
    K: np.ndarray
    camera_T_base: np.ndarray
    centerline_xy: np.ndarray
    axial_markers_xy: np.ndarray | None = None
    weight: float = 1.0
    timestamp_offset_s: float = 0.0
    topology_weight: float = 1.0


@dataclass(frozen=True)
class MultiViewSplineResult:
    points_base_mm: np.ndarray
    coefficients_base_mm: np.ndarray
    view_model_mean_px: dict[str, float]
    view_model_p95_px: dict[str, float]
    view_coverage_mean_px: dict[str, float]
    view_coverage_p95_px: dict[str, float]
    view_terminal_start_px: dict[str, float]
    view_terminal_end_px: dict[str, float]
    view_sharp_turn_clusters: dict[str, int]
    initial_symmetric_mean_px: float
    final_symmetric_mean_px: float
    arc_length_mm: float
    length_residual_mm: float
    optimizer_cost: float
    optimizer_evaluations: int
    optimizer_success: bool


def _soft_l1_residual(
        residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform residuals so linear least squares has a soft-L1 data loss.

    Returning the derivative of the transformed residual keeps the analytic
    spline Jacobian exact. Physical, temporal, and endpoint priors are not
    passed through this transform and therefore retain quadratic influence.
    """
    value = np.asarray(residual, dtype=np.float64)
    root = np.sqrt(1.0 + value * value)
    magnitude = np.sqrt(np.maximum(2.0 * (root - 1.0), 0.0))
    transformed = np.copysign(magnitude, value)
    derivative = np.ones_like(value)
    nonzero = np.abs(value) > 1e-5
    derivative[nonzero] = (
        np.abs(value[nonzero])
        / np.maximum(magnitude[nonzero] * root[nonzero], 1e-12))
    return transformed, derivative


def projected_route_topology_weight(
        centerline_xy: np.ndarray,
        axial_markers_xy: np.ndarray | None = None) -> float:
    """Return a soft reliability for material ordering in one projection.

    A genuinely ill-posed eye has non-adjacent material samples occupying the
    same small pixel neighborhood.  This is distinct from ordinary curvature:
    a smooth arc can turn substantially without developing nonlocal overlap.
    The score is deliberately soft because the other three views, rather than
    this heuristic alone, must make the final reconstruction decision.
    """
    try:
        points = _resample_2d(centerline_xy, 96)
    except ValueError:
        return 0.08
    delta = np.diff(points, axis=0)
    spacing = float(np.median(np.linalg.norm(delta, axis=1)))
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :], axis=2)
    indices = np.arange(len(points))
    local = np.abs(indices[:, None] - indices[None, :]) <= 10
    distances[local] = np.inf
    overlap_threshold = max(2.5, 2.25 * max(spacing, 0.5))
    overlap_fraction = float(np.mean(
        np.min(distances, axis=1) < overlap_threshold))
    unit = delta / np.maximum(
        np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
    cosine = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)
    maximum_turn_deg = float(np.degrees(np.max(np.arccos(cosine))))
    overlap_weight = float(np.exp(
        -max(0.0, overlap_fraction - 0.025) / 0.10))
    turn_weight = float(np.exp(
        -max(0.0, maximum_turn_deg - 105.0) / 45.0))
    marker_weight = 1.0
    if axial_markers_xy is not None:
        markers = np.asarray(axial_markers_xy, dtype=np.float64)
        finite = np.all(np.isfinite(markers), axis=1)
        if np.count_nonzero(finite) >= 3:
            tree = cKDTree(points)
            distance, material_index = tree.query(markers[finite])
            # Ring centroids need not lie exactly on the catheter axis, so this
            # remains a broad soft gate. Reversed/coincident material order is
            # stronger evidence that the image route took a shortcut.
            distance_weight = float(np.exp(
                -max(0.0, float(np.percentile(distance, 75)) - 10.0) / 12.0))
            order_margin = np.diff(material_index)
            order_violations = int(np.count_nonzero(order_margin <= 1))
            order_weight = float(np.exp(-0.8 * order_violations))
            marker_weight = distance_weight * order_weight
    return float(np.clip(
        overlap_weight * turn_weight * marker_weight, 0.08, 1.0))


def triangulate_material_curve(
        observations: list[ViewObservation],
        samples: int = 64) -> np.ndarray:
    """DLT initializer using normalized material order in available views.

    This is only an initializer. The final optimizer is correspondence-free,
    so approximate normalized-order correspondences cannot pin the solution.
    Coordinates are returned in the robot-base frame in millimetres.
    """
    usable = []
    for observation in observations:
        try:
            curve = _resample_2d(observation.centerline_xy, samples)
        except ValueError:
            continue
        projection = np.asarray(observation.K, float) @ np.asarray(
            observation.camera_T_base, float)[:3]
        usable.append((projection, curve, max(float(observation.weight), 1e-3)))
    if len(usable) < 2:
        raise ValueError("multi-view triangulation requires at least two views")
    output = np.full((samples, 3), np.nan, dtype=np.float64)
    for sample in range(samples):
        rows = []
        for projection, curve, weight in usable:
            x, y = curve[sample]
            scale = np.sqrt(weight)
            rows.extend([
                scale * (x * projection[2] - projection[0]),
                scale * (y * projection[2] - projection[1]),
            ])
        _, _, vt = np.linalg.svd(np.asarray(rows), full_matrices=False)
        homogeneous = vt[-1]
        if abs(homogeneous[3]) > 1e-10:
            output[sample] = homogeneous[:3] / homogeneous[3] * 1000.0
    finite = np.all(np.isfinite(output), axis=1)
    if np.count_nonzero(finite) < 4:
        raise ValueError("multi-view triangulation produced too few points")
    output = output[finite]
    # Orient every candidate from interface/base toward marker 3. Image routes
    # are already stored in that direction, but DLT can retain an accidental
    # reversal if most source routes were malformed.
    return output


def candidate_symmetric_error(
        points_base_mm: np.ndarray,
        observations: list[ViewObservation]) -> float:
    """Trimmed cross-view score used to select a robust initializer."""
    scores = []
    weights = []
    for observation in observations:
        projected, in_front = project_points(
            observation.K, observation.camera_T_base, points_base_mm)
        if not np.all(in_front):
            scores.append(1000.0)
            weights.append(max(float(observation.weight), 0.01))
            continue
        observed = _resample_2d(observation.centerline_xy, 48)
        metric = _view_metrics(projected, observed)
        scores.append(float(0.5 * (metric[0] + metric[2])))
        weights.append(max(float(observation.weight), 0.01))
    scores = np.asarray(scores, float)
    weights = np.asarray(weights, float)
    # Retain at least one view from each physical rig when names are available.
    # This prevents two mutually consistent but ill-posed eyes on one ZED from
    # outvoting the oblique rig merely because the old trimmed score discarded
    # only a single view.
    rig_names = [item.view_id.rsplit("_", 1)[0] for item in observations]
    if len(set(rig_names)) >= 2:
        rig_scores = []
        rig_weights = []
        for rig in sorted(set(rig_names)):
            selected = np.asarray([name == rig for name in rig_names])
            rig_scores.append(float(np.average(
                scores[selected], weights=weights[selected])))
            rig_weights.append(float(np.max(weights[selected])))
        return float(np.average(rig_scores, weights=rig_weights))
    ordered = np.argsort(scores)
    retained = ordered[:max(2, len(ordered) - 1)]
    return float(np.average(scores[retained], weights=weights[retained]))


def fit_multi_view_spline(
        initial_points_base_mm: np.ndarray,
        observations: list[ViewObservation],
        nominal_length_mm: float = 57.0,
        output_samples: int = 64,
        basis_count: int = 20,
        fit_samples: int = 64,
        coverage_samples: int = 48,
        length_sigma_mm: float = 3.0,
        marker_sigma_px: float = 2.0,
        body_marker_sigma_px: float = 6.0,
        coefficient_prior_sigma_mm: float = 8.0,
        temporal_prior_points_base_mm: np.ndarray | None = None,
        temporal_prior_sigma_mm: float = 3.0,
        coefficient_velocity_base_mm_s: np.ndarray | None = None,
        stereo_curve_points_base_mm: np.ndarray | None = None,
        stereo_curve_timestamp_offset_s: float = 0.0,
        stereo_curve_sigma_mm: float = 1.5,
        stereo_curve_weight: float = 0.0,
        stereo_curve_quadratic: bool = False,
        local_stretch_sigma_mm: float = 0.75,
        variation_sigma_mm: float = 1.5,
        bending_sigma_mm: float = 0.22,
        coverage_weight: float = 0.30,
        max_nfev: int = 40) -> MultiViewSplineResult:
    """Fit one cubic material-coordinate spline to two to four views."""
    if len(observations) < 2:
        raise ValueError("multi-view spline fit requires at least two views")
    initial = np.asarray(initial_points_base_mm, dtype=np.float64)
    initial, _, _ = resample_polyline(initial, fit_samples)
    design = _basis(fit_samples, basis_count)
    output_design = _basis(output_samples, basis_count)
    coefficients, *_ = np.linalg.lstsq(design, initial, rcond=None)
    initial_coefficients = coefficients.copy()
    temporal_coefficients = None
    if temporal_prior_points_base_mm is not None:
        temporal, _, _ = resample_polyline(
            np.asarray(temporal_prior_points_base_mm, float), fit_samples)
        temporal_coefficients, *_ = np.linalg.lstsq(design, temporal, rcond=None)
        coefficients = 0.45 * coefficients + 0.55 * temporal_coefficients
    velocity = (
        np.zeros_like(coefficients)
        if coefficient_velocity_base_mm_s is None else
        np.asarray(coefficient_velocity_base_mm_s, dtype=np.float64))
    if velocity.shape != coefficients.shape:
        raise ValueError("coefficient velocity shape does not match spline basis")

    prepared = []
    for observation in observations:
        observed = _resample_2d(
            observation.centerline_xy, coverage_samples)
        prepared.append((
            observation, observed, cKDTree(observed),
            np.sqrt(max(float(observation.weight), 1e-6))))
    stereo_curve = None
    stereo_tree = None
    if stereo_curve_points_base_mm is not None and stereo_curve_weight > 0.0:
        candidate = np.asarray(
            stereo_curve_points_base_mm, dtype=np.float64)
        candidate = candidate[np.all(np.isfinite(candidate), axis=1)]
        if len(candidate) >= 4:
            stereo_curve = candidate
            stereo_tree = cKDTree(candidate)

    marker_u = np.array([0.0, 10.0 / 57.0, 30.0 / 57.0, 1.0])
    marker_design_full = _basis(257, basis_count)
    marker_design = marker_design_full[np.rint(marker_u * 256).astype(int)]
    segment_design = np.diff(design, axis=0)
    bending_design = np.diff(design, n=2, axis=0)
    variation_design = np.diff(design, n=3, axis=0)
    bending_jacobian = np.kron(
        bending_design, np.eye(3)) / max(bending_sigma_mm, 1e-6)
    variation_jacobian = np.kron(
        variation_design, np.eye(3)) / max(variation_sigma_mm, 1e-6)
    identity = np.eye(3 * basis_count)

    def evaluate(flattened: np.ndarray, need_jacobian: bool):
        values = flattened.reshape(basis_count, 3)
        midpoint_points = design @ values
        terms = []
        jacobians = [] if need_jacobian else None
        for observation, observed, tree, scale in prepared:
            view_values = values + float(observation.timestamp_offset_s) * velocity
            view_points = design @ view_values
            projected, projection_jacobian = _project_spline_with_jacobian(
                observation.K, observation.camera_T_base, view_points, design)
            model_index = tree.query(projected)[1]
            coverage_index = cKDTree(projected).query(observed)[1]
            model_raw = scale * (
                projected - observed[model_index]).ravel()
            coverage_raw = coverage_weight * scale * (
                observed - projected[coverage_index]).ravel()
            # Image centerlines can still contain a wrong branch in an
            # ill-conditioned projection.  They must remain robust even when
            # a separately gated stereo-depth observation is trusted.
            model_term, model_derivative = _soft_l1_residual(model_raw)
            coverage_term, coverage_derivative = _soft_l1_residual(
                coverage_raw)
            terms.extend([model_term, coverage_term])
            if need_jacobian:
                model_jacobian = scale * projection_jacobian.reshape(
                    len(projected) * 2, -1)
                coverage_jacobian = (
                    -coverage_weight * scale * projection_jacobian[
                        coverage_index].reshape(len(observed) * 2, -1))
                jacobians.extend([
                    model_derivative[:, None] * model_jacobian,
                    coverage_derivative[:, None] * coverage_jacobian,
                ])
            markers = observation.axial_markers_xy
            if markers is not None:
                markers = np.asarray(markers, float)
                marker_points = marker_design @ view_values
                marker_projection, marker_jacobian = (
                    _project_spline_with_jacobian(
                        observation.K, observation.camera_T_base,
                        marker_points, marker_design))
                for marker in range(min(4, len(markers))):
                    if not np.all(np.isfinite(markers[marker])):
                        continue
                    sigma = (marker_sigma_px if marker in (0, 3)
                             else body_marker_sigma_px)
                    marker_raw = (
                        scale * (marker_projection[marker] - markers[marker])
                        / max(sigma, 1e-6))
                    marker_jacobian_scaled = (
                        scale * marker_jacobian[marker] / max(sigma, 1e-6))
                    if marker in (0, 3):
                        # Interface/tip observations have already been
                        # projected onto the extracted centerline. Retain a
                        # strong but finite quadratic endpoint regularizer.
                        terms.append(marker_raw)
                        if need_jacobian:
                            jacobians.append(marker_jacobian_scaled)
                    else:
                        marker_term, marker_derivative = (
                            _soft_l1_residual(marker_raw))
                        terms.append(marker_term)
                        if need_jacobian:
                            jacobians.append(
                                marker_derivative[:, None]
                                * marker_jacobian_scaled)

        if stereo_curve is not None:
            stereo_values = (
                values
                + float(stereo_curve_timestamp_offset_s) * velocity)
            stereo_model = design @ stereo_values
            model_index = stereo_tree.query(stereo_model)[1]
            coverage_index = cKDTree(stereo_model).query(stereo_curve)[1]
            scale = np.sqrt(max(float(stereo_curve_weight), 1e-9))
            sigma = max(float(stereo_curve_sigma_mm), 1e-6)
            model_raw = scale * (
                stereo_model - stereo_curve[model_index]).ravel() / sigma
            coverage_raw = 0.35 * scale * (
                stereo_curve - stereo_model[coverage_index]
            ).ravel() / sigma
            if stereo_curve_quadratic:
                # The caller only enables this after the ordered stereo curve
                # passes support, length, and temporal-innovation gates.  A
                # finite quadratic residual then lets the selected well-posed
                # rig determine depth instead of saturating while a bad 2-D
                # route drags the solution.
                model_term = model_raw
                coverage_term = coverage_raw
                model_derivative = np.ones_like(model_raw)
                coverage_derivative = np.ones_like(coverage_raw)
            else:
                model_term, model_derivative = _soft_l1_residual(model_raw)
                coverage_term, coverage_derivative = _soft_l1_residual(
                    coverage_raw)
            terms.extend([model_term, coverage_term])
            if need_jacobian:
                point_jacobian = np.kron(design, np.eye(3))
                model_jacobian = scale * point_jacobian / sigma
                coverage_jacobian = (
                    -0.35 * scale
                    * np.kron(design[coverage_index], np.eye(3)) / sigma)
                jacobians.extend([
                    model_derivative[:, None] * model_jacobian,
                    coverage_derivative[:, None] * coverage_jacobian,
                ])

        segments = np.diff(midpoint_points, axis=0)
        segment_length = np.linalg.norm(segments, axis=1)
        length = float(np.sum(segment_length))
        terms.extend([
            np.asarray([(length - nominal_length_mm)
                        / max(length_sigma_mm, 1e-6)]),
            (segment_length - np.mean(segment_length))
            / max(local_stretch_sigma_mm, 1e-6),
            (bending_design @ values).ravel() / max(bending_sigma_mm, 1e-6),
            (variation_design @ values).ravel() / max(variation_sigma_mm, 1e-6),
            (values - initial_coefficients).ravel()
            / max(coefficient_prior_sigma_mm, 1e-6),
        ])
        if temporal_coefficients is not None:
            terms.append((values - temporal_coefficients).ravel()
                         / max(temporal_prior_sigma_mm, 1e-6))
        if need_jacobian:
            unit = np.divide(
                segments, segment_length[:, None], out=np.zeros_like(segments),
                where=segment_length[:, None] > 1e-9)
            segment_jacobian = (
                segment_design[:, :, None] * unit[:, None, :]
            ).reshape(len(segment_design), 3 * basis_count)
            jacobians.extend([
                np.sum(segment_jacobian, axis=0, keepdims=True)
                / max(length_sigma_mm, 1e-6),
                (segment_jacobian - np.mean(segment_jacobian, axis=0))
                / max(local_stretch_sigma_mm, 1e-6),
                bending_jacobian,
                variation_jacobian,
                identity / max(coefficient_prior_sigma_mm, 1e-6),
            ])
            if temporal_coefficients is not None:
                jacobians.append(
                    identity / max(temporal_prior_sigma_mm, 1e-6))
        return np.concatenate(terms), (None if jacobians is None
                                       else np.vstack(jacobians))

    cache = {"x": None, "r": None, "j": None}

    def cached(x):
        if cache["x"] is None or not np.array_equal(cache["x"], x):
            cache["x"] = np.asarray(x).copy()
            cache["r"], cache["j"] = evaluate(x, True)
        return cache["r"], cache["j"]

    initial_metrics = []
    initial_weights = []
    for observation, observed, _, _ in prepared:
        points = design @ (
            initial_coefficients
            + float(observation.timestamp_offset_s) * velocity)
        projected = project_points(
            observation.K, observation.camera_T_base, points)[0]
        initial_metrics.append(_view_metrics(projected, observed))
        initial_weights.extend([observation.weight, observation.weight])
    solution = least_squares(
        lambda x: cached(x)[0], coefficients.ravel(),
        jac=lambda x: cached(x)[1], method="trf", loss="linear",
        f_scale=1.0, max_nfev=max(1, int(max_nfev)),
        xtol=2e-4, ftol=2e-4, gtol=2e-4)
    final_coefficients = solution.x.reshape(basis_count, 3)
    output_points = output_design @ final_coefficients
    means = {}
    p95 = {}
    coverage = {}
    coverage_p95 = {}
    terminal_start = {}
    terminal_end = {}
    turns = {}
    final_symmetric = []
    final_symmetric_weights = []
    for observation, observed, _, _ in prepared:
        view_points = output_design @ (
            final_coefficients
            + float(observation.timestamp_offset_s) * velocity)
        projected = project_points(
            observation.K, observation.camera_T_base, view_points)[0]
        metric = _view_metrics(projected, observed)
        coverage_distance = cKDTree(projected).query(observed)[0]
        means[observation.view_id] = metric[0]
        p95[observation.view_id] = metric[1]
        coverage[observation.view_id] = metric[2]
        coverage_p95[observation.view_id] = float(np.percentile(
            coverage_distance, 95))
        terminal_start[observation.view_id] = float(np.linalg.norm(
            projected[0] - observed[0]))
        terminal_end[observation.view_id] = float(np.linalg.norm(
            projected[-1] - observed[-1]))
        turns[observation.view_id] = projected_sharp_turn_clusters(projected)
        final_symmetric.extend([metric[0], metric[2]])
        final_symmetric_weights.extend([
            observation.weight, observation.weight])
    length = float(cumulative_arclength(output_points)[-1])
    return MultiViewSplineResult(
        points_base_mm=output_points,
        coefficients_base_mm=final_coefficients,
        view_model_mean_px=means,
        view_model_p95_px=p95,
        view_coverage_mean_px=coverage,
        view_coverage_p95_px=coverage_p95,
        view_terminal_start_px=terminal_start,
        view_terminal_end_px=terminal_end,
        view_sharp_turn_clusters=turns,
        initial_symmetric_mean_px=float(np.average(
            [value for metric in initial_metrics
             for value in (metric[0], metric[2])],
            weights=initial_weights)),
        final_symmetric_mean_px=float(np.average(
            final_symmetric, weights=final_symmetric_weights)),
        arc_length_mm=length,
        length_residual_mm=length - float(nominal_length_mm),
        optimizer_cost=float(solution.cost),
        optimizer_evaluations=int(solution.nfev),
        optimizer_success=bool(solution.success))
