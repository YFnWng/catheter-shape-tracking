"""Stereo reconstruction helpers for the multimodal sequence processor."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .reconstruction import (
    arc_length,
    epipolar_correspond,
    rectified_projection_matrices,
    smooth_polyline_2d,
)


def regularize_disparity_local(
        raw_disparity: np.ndarray,
        observed: np.ndarray,
        anchors: list[tuple[int, float, float]] | None = None,
        first_difference_weight: float = 0.25,
        second_difference_weight: float = 10.0,
        huber_delta_px: float = 1.5,
        iterations: int = 8,
        observation_weights: np.ndarray | None = None,
        first_difference_weights: np.ndarray | None = None,
        second_difference_weights: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
    """Robustly regularize one disparity value per centerline sample.

    Unlike the former global polynomial, this keeps all samples as degrees of
    freedom. First- and second-difference penalties provide local continuity,
    while Huber IRLS reduces the influence of incorrect epipolar crossings.
    `anchors` contains (sample index, disparity px, residual weight).
    """
    raw = np.asarray(raw_disparity, dtype=np.float64)
    valid = np.asarray(observed, dtype=bool) & np.isfinite(raw) & (raw > 0.0)
    if np.count_nonzero(valid) < 4:
        raise ValueError("insufficient_positive_disparity_matches")
    n = len(raw)
    indices = np.arange(n)
    estimate = np.interp(indices, indices[valid], raw[valid])
    first = np.diff(np.eye(n), n=1, axis=0)
    second = np.diff(np.eye(n), n=2, axis=0)
    first_weights = (
        np.full(n - 1, max(float(first_difference_weight), 0.0))
        if first_difference_weights is None
        else np.asarray(first_difference_weights, dtype=np.float64))
    second_weights = (
        np.full(n - 2, max(float(second_difference_weight), 0.0))
        if second_difference_weights is None
        else np.asarray(second_difference_weights, dtype=np.float64))
    if first_weights.shape != (n - 1,) or second_weights.shape != (n - 2,):
        raise ValueError("difference weights do not match disparity samples")
    regularizer = (
        first.T @ (np.clip(first_weights, 0.0, None)[:, None] * first)
        + second.T @ (np.clip(second_weights, 0.0, None)[:, None] * second))
    data_weights = (
        np.ones(n, dtype=np.float64) if observation_weights is None
        else np.asarray(observation_weights, dtype=np.float64))
    if data_weights.shape != (n,):
        raise ValueError("observation weights do not match disparity samples")
    data_weights = np.clip(data_weights, 0.0, None)
    anchor_list = anchors or []
    robust_weights = np.ones(np.count_nonzero(valid), dtype=np.float64)
    valid_indices = indices[valid]
    delta = max(float(huber_delta_px), 1e-6)
    for _ in range(max(1, int(iterations))):
        normal = regularizer.copy()
        rhs = np.zeros(n, dtype=np.float64)
        effective = robust_weights * data_weights[valid]
        normal[valid_indices, valid_indices] += effective
        rhs[valid_indices] += effective * raw[valid]
        for index, value, weight in anchor_list:
            weight_squared = max(float(weight), 0.0) ** 2
            normal[int(index), int(index)] += weight_squared
            rhs[int(index)] += weight_squared * float(value)
        estimate = np.linalg.solve(normal + 1e-9 * np.eye(n), rhs)
        residual = estimate[valid] - raw[valid]
        magnitude = np.abs(residual)
        robust_weights = np.where(
            magnitude <= delta, 1.0, delta / np.maximum(magnitude, 1e-12))

    positive = raw[valid]
    floor = max(0.5, 0.25 * float(np.percentile(positive, 5)))
    return np.clip(estimate, floor, None), robust_weights


def _mask_epipolar_candidates(
        reference: np.ndarray,
        other_centerline,
        reference_view: str,
        row_search_px: int = 2
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Return unordered positive-disparity candidates from the other-eye mask.

    A projected catheter can overlap itself in one eye, so its ordered skeleton
    is not a one-to-one observation of material arc length.  Each reference
    sample therefore sees every connected mask run close to its rectified
    epipolar row.  Candidate order is deliberately discarded; two different
    reference samples are allowed to select the same other-eye pixel.
    """
    if reference_view not in ("left", "right"):
        raise ValueError("reference_view must be 'left' or 'right'")
    mask = np.asarray(other_centerline.mask) > 0
    other_points = np.asarray(other_centerline.points, dtype=np.float64)
    roi_x, roi_y, _, _ = (int(value) for value in other_centerline.roi)
    disparities: list[np.ndarray] = []
    data_costs: list[np.ndarray] = []
    ambiguous_rows: list[bool] = []
    search = max(0, int(row_search_px))
    for x_reference, y_reference in np.asarray(reference, dtype=np.float64):
        values: list[tuple[float, float]] = []
        exact_crossings: list[float] = []
        # Exact intersections with the unordered centerline preserve separate
        # branches even when their finite-width mask runs touch.
        for start, end in zip(other_points[:-1], other_points[1:]):
            y0, y1 = float(start[1]), float(end[1])
            if y0 == y1 or (y0 - y_reference) * (y1 - y_reference) > 0.0:
                continue
            alpha = (float(y_reference) - y0) / (y1 - y0)
            x_other = float(start[0] + alpha * (end[0] - start[0]))
            disparity = (
                float(x_reference) - x_other
                if reference_view == "left"
                else x_other - float(x_reference))
            if disparity > 0.25:
                values.append((disparity, 0.0))
                exact_crossings.append(disparity)
        center_row = int(round(float(y_reference) - roi_y))
        for offset in range(-search, search + 1):
            row = center_row + offset
            if row < 0 or row >= mask.shape[0]:
                continue
            columns = np.flatnonzero(mask[row])
            if len(columns) == 0:
                continue
            splits = np.flatnonzero(np.diff(columns) > 1) + 1
            for run in np.split(columns, splits):
                x_other = roi_x + 0.5 * float(run[0] + run[-1])
                disparity = (
                    float(x_reference) - x_other
                    if reference_view == "left"
                    else x_other - float(x_reference))
                if disparity > 0.25:
                    # Prefer the exact epipolar row, but retain neighboring
                    # rows for sub-pixel rectification and skeleton gaps.
                    values.append((disparity, float(offset * offset)))
        if values:
            values.sort(key=lambda item: item[0])
            merged: list[tuple[float, float]] = []
            for disparity, cost in values:
                if merged and abs(disparity - merged[-1][0]) < 1.0:
                    if cost < merged[-1][1]:
                        merged[-1] = (disparity, cost)
                else:
                    merged.append((disparity, cost))
            disparities.append(np.asarray([item[0] for item in merged]))
            data_costs.append(np.asarray([item[1] for item in merged]))
        else:
            disparities.append(np.empty(0, dtype=np.float64))
            data_costs.append(np.empty(0, dtype=np.float64))
        exact_crossings.sort()
        distinct = 0
        previous = None
        for value in exact_crossings:
            if previous is None or abs(value - previous) >= 2.0:
                distinct += 1
                previous = value
        ambiguous_rows.append(distinct > 1)
    return disparities, data_costs, np.asarray(ambiguous_rows, dtype=bool)


def select_overlap_aware_disparity_path(
        candidates: list[np.ndarray],
        data_costs: list[np.ndarray],
        disparity_prior_px: np.ndarray | None = None,
        disparity_prior_weight: float = 0.0,
        anchors: list[tuple[int, float, float]] | None = None,
        transition_weight: float = 0.25
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select a smooth path through unordered epipolar mask candidates.

    This is a first-order dynamic program over disparity, not over other-eye
    arc position.  Consequently it supports many-to-one image projection.  A
    later robust second-difference fit supplies the final local smoothness.
    """
    count = len(candidates)
    if count == 0 or len(data_costs) != count:
        raise ValueError("invalid overlap-aware candidate sequence")
    prior = None
    if disparity_prior_px is not None:
        prior = np.asarray(disparity_prior_px, dtype=np.float64)
        if len(prior) != count:
            prior = np.interp(
                np.linspace(0.0, 1.0, count),
                np.linspace(0.0, 1.0, len(prior)), prior)
    anchor_map: dict[int, list[tuple[float, float]]] = {}
    for index, value, weight in anchors or []:
        anchor_map.setdefault(int(index), []).append(
            (float(value), float(weight)))

    valid_indices = [index for index, values in enumerate(candidates) if len(values)]
    if len(valid_indices) < 4:
        raise ValueError("insufficient_overlap_aware_mask_rows")
    ambiguity_fraction = float(np.mean([
        len(values) > 1 for values in candidates if len(values)]))
    backpointers: list[np.ndarray | None] = [None] * count
    accumulated: np.ndarray | None = None
    previous_values: np.ndarray | None = None
    previous_index: int | None = None
    scale = 2.0
    for index in valid_indices:
        values = np.asarray(candidates[index], dtype=np.float64)
        unary = np.asarray(data_costs[index], dtype=np.float64).copy()
        if prior is not None and np.isfinite(prior[index]) and prior[index] > 0.0:
            unary += max(float(disparity_prior_weight), 0.0) * (
                (values - prior[index]) / scale) ** 2
        for anchor_value, anchor_weight in anchor_map.get(index, []):
            unary += max(anchor_weight, 0.0) ** 2 * (
                (values - anchor_value) / scale) ** 2
        if accumulated is None:
            accumulated = unary
            backpointers[index] = np.full(len(values), -1, dtype=np.int32)
        else:
            gap = max(index - int(previous_index), 1)
            transitions = (
                accumulated[:, None]
                + max(float(transition_weight), 0.0)
                * (values[None, :] - previous_values[:, None]) ** 2 / gap)
            parents = np.argmin(transitions, axis=0)
            accumulated = unary + transitions[parents, np.arange(len(values))]
            backpointers[index] = parents.astype(np.int32)
        previous_values = values
        previous_index = index

    selected = np.full(count, np.nan, dtype=np.float64)
    cursor = int(np.argmin(accumulated))
    for index in reversed(valid_indices):
        selected[index] = candidates[index][cursor]
        parent = int(backpointers[index][cursor])
        if parent >= 0:
            cursor = parent
    observed = np.isfinite(selected)
    return selected, observed, ambiguity_fraction


def project_camera_points(
        points_camera_m: np.ndarray,
        K: np.ndarray,
        baseline_m: float,
        right: bool = False) -> np.ndarray:
    """Project left-camera-frame points into a rectified ZED view."""
    left_projection, right_projection = rectified_projection_matrices(
        K, baseline_m)
    projection = right_projection if right else left_projection
    points = np.asarray(points_camera_m, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    pixels_h = (projection @ homogeneous.T).T
    return pixels_h[:, :2] / pixels_h[:, 2, None]


def reprojection_distances(
        points_camera_m: np.ndarray,
        left_centerline,
        right_centerline,
        K: np.ndarray,
        baseline_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centerline distances in the two rectified images."""
    left_tree = cKDTree(left_centerline.points)
    right_tree = cKDTree(right_centerline.points)
    left_distance, _ = left_tree.query(
        project_camera_points(points_camera_m, K, baseline_m, right=False))
    right_distance, _ = right_tree.query(
        project_camera_points(points_camera_m, K, baseline_m, right=True))
    return left_distance, right_distance


def projected_self_overlap_fraction(
        points: np.ndarray,
        distance_px: float = 8.0,
        minimum_arclength_separation: float = 0.12) -> float:
    """Fraction of a projected curve touching a nonlocal arc-length branch."""
    curve = np.asarray(points, dtype=np.float64)
    if len(curve) < 8:
        return 0.0
    dense = smooth_polyline_2d(curve, max(128, len(curve)))
    pairs = cKDTree(dense).query_pairs(max(float(distance_px), 0.0))
    minimum_index_gap = max(
        2, int(np.ceil(float(minimum_arclength_separation) * (len(dense) - 1))))
    involved: set[int] = set()
    for first, second in pairs:
        if abs(int(second) - int(first)) >= minimum_index_gap:
            involved.add(int(first))
            involved.add(int(second))
    return float(len(involved) / len(dense))


def reconstruct_disparity_anchored(
        left_centerline,
        right_centerline,
        K: np.ndarray,
        baseline_m: float,
        base_camera_m: np.ndarray | None,
        tip_camera_m: np.ndarray | None,
        n_samples: int = 96,
        disparity_order: int = 3,
        smooth_2d: float | None = None,
        base_depth_weight: float = 5.0,
        tip_depth_weight: float = 2.0,
        disparity_first_difference_weight: float = 0.25,
        disparity_second_difference_weight: float = 10.0,
        disparity_huber_delta_px: float = 1.5,
        reference_view: str = "left",
        disparity_prior_px: np.ndarray | None = None,
        disparity_prior_weight: float = 0.0,
        overlap_aware: bool = True,
        overlap_row_search_px: int = 2,
        overlap_self_fraction_threshold: float = 0.05,
        overlap_self_distance_px: float = 8.0,
        overlap_min_arclength_separation: float = 0.12,
        centerline_tip_depth_weight: float = 5.0,
        max_centerline_tip_epipolar_error_px: float = 8.0,
        terminal_refinement: bool = True,
        terminal_refinement_fraction: float = 0.20,
        terminal_refinement_smoothness_scale: float = 0.10,
        terminal_refinement_observation_weight: float = 4.0,
        terminal_refinement_tip_weight: float = 20.0,
        terminal_refinement_max_p95_degradation_px: float = 0.25) -> dict:
    """Lift a 2D shaft centerline with robust locally regularized disparity.

    The registered base depth strongly constrains the proximal end. When an EM
    tip is available, its depth is a weaker distal prior because the visible blue
    segment stops before the tape and printed tip housing. Either anchor may be
    ``None`` for a fully image-observed endpoint.
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if reference_view not in ("left", "right"):
        raise ValueError("reference_view must be 'left' or 'right'")
    reference_centerline = (
        left_centerline if reference_view == "left" else right_centerline)
    other_centerline = (
        right_centerline if reference_view == "left" else left_centerline)
    reference = smooth_polyline_2d(
        reference_centerline.points, n_samples, smooth_2d)
    other_dense = smooth_polyline_2d(
        other_centerline.points, max(4 * n_samples, 300), smooth_2d)
    anchors: list[tuple[int, float, float]] = []
    base_camera_m = (
        None if base_camera_m is None
        else np.asarray(base_camera_m, dtype=np.float64))
    tip_camera_m = (
        None if tip_camera_m is None
        else np.asarray(tip_camera_m, dtype=np.float64))
    if base_camera_m is not None and base_camera_m[2] > 1e-6:
        anchors.append((
            0, float(fx * baseline_m / base_camera_m[2]),
            float(base_depth_weight)))
    if tip_camera_m is not None and tip_camera_m[2] > 1e-6:
        anchors.append((
            n_samples - 1, float(fx * baseline_m / tip_camera_m[2]),
            float(tip_depth_weight)))
    left_tip = np.asarray(left_centerline.points[-1], dtype=np.float64)
    right_tip = np.asarray(right_centerline.points[-1], dtype=np.float64)
    centerline_tip_epipolar_error_px = float(abs(left_tip[1] - right_tip[1]))
    centerline_tip_disparity_px = float(left_tip[0] - right_tip[0])
    centerline_tip_anchor_used = bool(
        np.isfinite(centerline_tip_disparity_px)
        and centerline_tip_disparity_px > 0.25
        and centerline_tip_epipolar_error_px
        <= float(max_centerline_tip_epipolar_error_px))
    if centerline_tip_anchor_used:
        anchors.append((
            n_samples - 1, centerline_tip_disparity_px,
            float(centerline_tip_depth_weight)))
    mask_candidates, mask_costs, ambiguous_rows = _mask_epipolar_candidates(
        reference, other_centerline, reference_view,
        row_search_px=overlap_row_search_px)
    nonempty = [values for values in mask_candidates if len(values)]
    observed_rows = np.asarray([len(values) > 0 for values in mask_candidates])
    mask_ambiguity_fraction = float(np.mean(
        ambiguous_rows[observed_rows])) if np.any(observed_rows) else 0.0
    other_self_overlap_fraction = projected_self_overlap_fraction(
        other_centerline.points, overlap_self_distance_px,
        overlap_min_arclength_separation)
    reference_self_overlap_fraction = projected_self_overlap_fraction(
        reference_centerline.points, overlap_self_distance_px,
        overlap_min_arclength_separation)
    use_overlap_aware = bool(
        overlap_aware
        and len(nonempty) >= max(8, n_samples // 4)
        and other_self_overlap_fraction >= float(overlap_self_fraction_threshold)
        and other_self_overlap_fraction
        > reference_self_overlap_fraction + 0.02)
    if use_overlap_aware:
        raw_disparity, matched, _ = (
            select_overlap_aware_disparity_path(
                mask_candidates, mask_costs,
                disparity_prior_px=disparity_prior_px,
                disparity_prior_weight=disparity_prior_weight,
                anchors=anchors,
                transition_weight=disparity_first_difference_weight))
        correspondence_model = "unordered_mask_overlap_aware"
    else:
        other, matched = epipolar_correspond(reference, other_dense)
        if reference_view == "left":
            left, right = reference, other
        else:
            left, right = other, reference
        raw_disparity = left[:, 0] - right[:, 0]
        if np.count_nonzero(matched) < max(8, n_samples // 4):
            other_arc = smooth_polyline_2d(
                other_centerline.points, n_samples, smooth_2d)
            if reference_view == "left":
                left, right = reference, other_arc
            else:
                left, right = other_arc, reference
            raw_disparity = left[:, 0] - right[:, 0]
            matched = np.ones(n_samples, dtype=bool)
        correspondence_model = "ordered_epipolar_crossing"
    if disparity_prior_px is not None and disparity_prior_weight > 0.0:
        prior = np.asarray(disparity_prior_px, dtype=np.float64)
        if len(prior) != n_samples:
            prior = np.interp(
                np.linspace(0.0, 1.0, n_samples),
                np.linspace(0.0, 1.0, len(prior)), prior)
        anchors.extend([
            (index, float(value), float(disparity_prior_weight))
            for index, value in enumerate(prior)
            if np.isfinite(value) and value > 0.0])
    disparity, robust_weights = regularize_disparity_local(
        raw_disparity, matched, anchors,
        first_difference_weight=disparity_first_difference_weight,
        second_difference_weight=disparity_second_difference_weight,
        huber_delta_px=disparity_huber_delta_px)

    terminal_refinement_used = False
    terminal_refinement_improvement_px = 0.0
    terminal_start = int(np.clip(
        np.floor((1.0 - float(terminal_refinement_fraction)) * n_samples),
        2, n_samples - 3))
    terminal_observed = observed_rows[terminal_start:]
    terminal_ambiguity = ambiguous_rows[terminal_start:]
    terminal_confident = bool(
        terminal_refinement
        and not use_overlap_aware
        and centerline_tip_anchor_used
        and np.mean(terminal_observed) >= 0.75
        and (not np.any(terminal_observed)
             or np.mean(terminal_ambiguity[terminal_observed]) <= 0.10)
        and reference_self_overlap_fraction
        < float(overlap_self_fraction_threshold)
        and other_self_overlap_fraction
        < float(overlap_self_fraction_threshold))
    if terminal_confident:
        observation_weights = np.ones(n_samples, dtype=np.float64)
        terminal_u = np.linspace(0.0, 1.0, n_samples - terminal_start)
        observation_weights[terminal_start:] = (
            1.0 + terminal_u
            * (max(float(terminal_refinement_observation_weight), 1.0) - 1.0))
        second_weights = np.full(
            n_samples - 2, max(float(disparity_second_difference_weight), 0.0))
        second_start = max(0, terminal_start - 1)
        taper = np.linspace(0.0, 1.0, len(second_weights) - second_start)
        scale = float(np.clip(
            terminal_refinement_smoothness_scale, 0.0, 1.0))
        second_weights[second_start:] *= 1.0 - taper * (1.0 - scale)
        refined_anchors = list(anchors)
        refined_anchors.append((
            n_samples - 1, centerline_tip_disparity_px,
            float(terminal_refinement_tip_weight)))
        refined, refined_robust_weights = regularize_disparity_local(
            raw_disparity, matched, refined_anchors,
            first_difference_weight=disparity_first_difference_weight,
            second_difference_weight=disparity_second_difference_weight,
            huber_delta_px=disparity_huber_delta_px,
            observation_weights=observation_weights,
            second_difference_weights=second_weights)

        def points_from_disparity(values: np.ndarray) -> np.ndarray:
            if reference_view == "left":
                left_pixels = reference
            else:
                left_pixels = np.column_stack([
                    reference[:, 0] + values, reference[:, 1]])
            depth_values = fx * baseline_m / values
            return np.column_stack([
                (left_pixels[:, 0] - cx) * depth_values / fx,
                (left_pixels[:, 1] - cy) * depth_values / fy,
                depth_values])

        original_points = points_from_disparity(disparity)
        refined_points = points_from_disparity(refined)
        original_distances = reprojection_distances(
            original_points, left_centerline, right_centerline, K, baseline_m)
        refined_distances = reprojection_distances(
            refined_points, left_centerline, right_centerline, K, baseline_m)
        other_index = 1 if reference_view == "left" else 0
        original_terminal = float(np.mean(
            original_distances[other_index][terminal_start:]))
        refined_terminal = float(np.mean(
            refined_distances[other_index][terminal_start:]))
        original_p95 = float(np.percentile(
            np.concatenate(original_distances), 95))
        refined_p95 = float(np.percentile(
            np.concatenate(refined_distances), 95))
        terminal_refinement_improvement_px = (
            original_terminal - refined_terminal)
        if (refined_terminal < original_terminal
                and refined_p95 <= original_p95
                + float(terminal_refinement_max_p95_degradation_px)):
            disparity = refined
            robust_weights = refined_robust_weights
            terminal_refinement_used = True
    # Use the regularized disparity to construct a finite, rectified pixel pair.
    # In particular, right-reference epipolar matching can leave unmatched left
    # coordinates as NaN even though neighboring observations determine a valid
    # regularized disparity there.
    if reference_view == "left":
        left = reference
        right = np.column_stack([left[:, 0] - disparity, left[:, 1]])
    else:
        right = reference
        left = np.column_stack([right[:, 0] + disparity, right[:, 1]])
    depth = fx * baseline_m / disparity
    x = (left[:, 0] - cx) * depth / fx
    y = (left[:, 1] - cy) * depth / fy
    points_camera_m = np.column_stack([x, y, depth])
    left_distance, right_distance = reprojection_distances(
        points_camera_m, left_centerline, right_centerline, K, baseline_m)
    combined_distance = np.concatenate([left_distance, right_distance])
    terminal_count = min(8, len(left_distance))
    return {
        "points_camera_m": points_camera_m,
        "visible_arc_length_mm": arc_length(points_camera_m) * 1000.0,
        "reprojection_left_px": float(np.mean(left_distance)),
        "reprojection_right_px": float(np.mean(right_distance)),
        "reprojection_max_px": float(max(
            np.max(left_distance), np.max(right_distance))),
        'reprojection_p95_px': float(np.percentile(combined_distance, 95)),
        "matched_epipolar": int(np.count_nonzero(matched)),
        "raw_disparity_px": raw_disparity,
        "fitted_disparity_px": disparity,
        "disparity_robust_inlier_count": int(np.count_nonzero(
            robust_weights >= 0.5)),
        "disparity_model": "local_huber_first_second_difference",
        "correspondence_model": correspondence_model,
        "overlap_aware_used": int(use_overlap_aware),
        "epipolar_ambiguity_fraction": mask_ambiguity_fraction,
        "other_eye_self_overlap_fraction": other_self_overlap_fraction,
        "reference_eye_self_overlap_fraction": reference_self_overlap_fraction,
        "centerline_tip_anchor_used": int(centerline_tip_anchor_used),
        "centerline_tip_epipolar_error_px": centerline_tip_epipolar_error_px,
        "terminal_refinement_used": int(terminal_refinement_used),
        "terminal_refinement_improvement_px": float(
            terminal_refinement_improvement_px),
        "terminal_refinement_start_fraction": float(
            terminal_start / max(n_samples - 1, 1)),
        "terminal_reprojection_left_px": float(np.mean(
            left_distance[-terminal_count:])),
        "terminal_reprojection_right_px": float(np.mean(
            right_distance[-terminal_count:])),
        "reference_view": reference_view,
        "ordered_left_px": left,
        "ordered_right_px": right,
    }
