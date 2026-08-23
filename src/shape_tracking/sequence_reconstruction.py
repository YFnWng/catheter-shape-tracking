"""Stereo reconstruction helpers for the multimodal sequence processor."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import find_peaks
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
        row_search_px: int = 2,
        reference_interval_labels: np.ndarray | None = None,
        other_segment_interval_labels: np.ndarray | None = None,
        include_centerline_crossings: bool = True,
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
    medial = cv2.distanceTransform(
        np.asarray(mask, dtype=np.uint8), cv2.DIST_L2, 5)
    other_points = np.asarray(other_centerline.points, dtype=np.float64)
    roi_x, roi_y, _, _ = (int(value) for value in other_centerline.roi)
    disparities: list[np.ndarray] = []
    data_costs: list[np.ndarray] = []
    ambiguous_rows: list[bool] = []
    search = max(0, int(row_search_px))
    for reference_index, (x_reference, y_reference) in enumerate(
            np.asarray(reference, dtype=np.float64)):
        values: list[tuple[float, float]] = []
        exact_crossings: list[float] = []
        # Exact intersections with the unordered centerline preserve separate
        # branches even when their finite-width mask runs touch.
        for segment_index, (start, end) in enumerate(zip(
                other_points[:-1], other_points[1:])):
            if not include_centerline_crossings:
                break
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
                interval_cost = 0.0
                if (reference_interval_labels is not None
                        and other_segment_interval_labels is not None
                        and reference_index < len(reference_interval_labels)
                        and segment_index
                        < len(other_segment_interval_labels)
                        and reference_interval_labels[reference_index]
                        != other_segment_interval_labels[segment_index]):
                    interval_cost = 16.0
                values.append((disparity, interval_cost))
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
                profile = medial[row, run]
                # A connected run can contain both arms of a tight V. Its
                # midpoint is then between the physical centerlines. Preserve
                # each distance-ridge peak as a separate branch candidate.
                peaks, _ = find_peaks(
                    profile, distance=3,
                    prominence=max(0.20, 0.08 * float(np.max(profile))))
                strongest = int(np.argmax(profile))
                if not len(peaks):
                    peaks = np.array([strongest])
                elif strongest not in peaks:
                    peaks = np.append(peaks, strongest)
                for peak in np.unique(peaks):
                    column = int(run[int(peak)])
                    x_other = float(roi_x + column)
                    disparity = (
                        float(x_reference) - x_other
                        if reference_view == "left"
                        else x_other - float(x_reference))
                    if disparity > 0.25:
                        mask_base_cost = (
                            2.0 if reference_interval_labels is not None
                            else 0.0)
                        ridge_cost = 2.0 / max(
                            float(medial[row, column]), 0.5)
                        values.append((
                            disparity,
                            mask_base_cost + float(offset * offset)
                            + ridge_cost))
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


def _add_temporal_mask_candidates(
        reference: np.ndarray,
        other_centerline,
        reference_view: str,
        disparity_prior_px: np.ndarray,
        candidates: list[np.ndarray],
        data_costs: list[np.ndarray],
        row_search_px: int) -> None:
    """Add the nearest mask-supported realization of each temporal sample.

    When two projected arms are closer than the mask thickness, their binary
    union is one wide run. Its midpoint is not a catheter centerline. The prior
    3D curve still predicts which side of that run each material coordinate
    occupied, so retain the nearest currently supported pixel as an explicit
    dynamic-programming candidate.
    """
    prior = np.asarray(disparity_prior_px, dtype=np.float64)
    if len(prior) != len(reference):
        prior = np.interp(
            np.linspace(0.0, 1.0, len(reference)),
            np.linspace(0.0, 1.0, len(prior)), prior)
    mask = np.asarray(other_centerline.mask) > 0
    roi_x, roi_y, _, _ = (int(value) for value in other_centerline.roi)
    search = max(0, int(row_search_px))
    for index, ((x_reference, y_reference), prior_disparity) in enumerate(
            zip(reference, prior)):
        if not np.isfinite(prior_disparity) or prior_disparity <= 0.25:
            continue
        predicted_x = (
            float(x_reference) - float(prior_disparity)
            if reference_view == "left" else
            float(x_reference) + float(prior_disparity))
        center_row = int(round(float(y_reference) - roi_y))
        best = None
        for offset in range(-search, search + 1):
            row = center_row + offset
            if row < 0 or row >= mask.shape[0]:
                continue
            columns = np.flatnonzero(mask[row])
            if not len(columns):
                continue
            local = int(np.argmin(np.abs(columns + roi_x - predicted_x)))
            column = int(columns[local])
            distance = float(column + roi_x - predicted_x)
            score = distance * distance + 4.0 * float(offset * offset)
            if best is None or score < best[0]:
                x_other = float(column + roi_x)
                disparity = (
                    float(x_reference) - x_other
                    if reference_view == "left" else
                    x_other - float(x_reference))
                if disparity > 0.25:
                    best = (score, disparity, float(offset * offset))
        if best is None:
            continue
        _, disparity, cost = best
        values = np.asarray(candidates[index], dtype=np.float64)
        costs = np.asarray(data_costs[index], dtype=np.float64)
        if len(values) and np.min(np.abs(values - disparity)) < 0.75:
            nearest = int(np.argmin(np.abs(values - disparity)))
            costs[nearest] = min(float(costs[nearest]), cost)
        else:
            values = np.append(values, disparity)
            costs = np.append(costs, cost)
            order = np.argsort(values)
            values, costs = values[order], costs[order]
        candidates[index] = values
        data_costs[index] = costs


def _marker_interval_labels(
        reference: np.ndarray,
        other_centerline,
        reference_view: str,
        marker_left_px: np.ndarray | None,
        marker_right_px: np.ndarray | None,
        marker_confidence_left: np.ndarray | None,
        marker_confidence_right: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Label corresponding cyan subpaths using common ordered ring IDs.

    No disparity or physical spacing is interpolated. Labels merely prevent an
    exact epipolar crossing on interval 2--3 from masquerading as interval
    0--1 when both projected arms occupy the same rows.
    """
    if marker_left_px is None or marker_right_px is None:
        return None, None, 0
    left = np.asarray(marker_left_px, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(marker_right_px, dtype=np.float64).reshape(-1, 2)
    count = min(len(left), len(right))
    if count == 0:
        return None, None, 0
    confidence_left = (
        np.ones(count, dtype=np.float64)
        if marker_confidence_left is None else
        np.asarray(marker_confidence_left, dtype=np.float64)[:count])
    confidence_right = (
        np.ones(count, dtype=np.float64)
        if marker_confidence_right is None else
        np.asarray(marker_confidence_right, dtype=np.float64)[:count])
    reference_markers = left if reference_view == "left" else right
    other_markers = right if reference_view == "left" else left
    other_points = np.asarray(other_centerline.points, dtype=np.float64)
    if len(other_points) < 2:
        return None, None, 0
    reference_tree = cKDTree(reference)
    other_tree = cKDTree(other_points)
    boundaries: list[tuple[int, int]] = []
    common_count = 0
    for marker_id in range(count):
        confidence = float(np.sqrt(max(
            confidence_left[marker_id] * confidence_right[marker_id], 0.0)))
        if (confidence < 0.25
                or not np.all(np.isfinite(reference_markers[marker_id]))
                or not np.all(np.isfinite(other_markers[marker_id]))):
            continue
        reference_index = int(reference_tree.query(
            reference_markers[marker_id])[1])
        other_index = int(other_tree.query(other_markers[marker_id])[1])
        if (boundaries and (
                reference_index <= boundaries[-1][0]
                or other_index <= boundaries[-1][1])):
            continue
        boundaries.append((reference_index, other_index))
        common_count += 1
    if common_count < 2:
        return None, None, common_count
    reference_labels = np.zeros(len(reference), dtype=np.int16)
    other_point_labels = np.zeros(len(other_points), dtype=np.int16)
    for reference_index, other_index in boundaries:
        reference_labels[reference_index:] += 1
        other_point_labels[other_index:] += 1
    return reference_labels, other_point_labels[:-1], common_count


def _one_turn_candidate_path(
        values_sequence: list[np.ndarray],
        unary_costs: list[np.ndarray],
        valid_indices: list[int],
        reference_xy: np.ndarray,
        reference_view: str,
        transition_weight: float,
        curvature_weight: float,
        sharp_turn_threshold_deg: float,
        expected_turn_index: int | None,
        turn_window_samples: int) -> np.ndarray:
    """Pair-state DP whose topology state permits at most one sharp turn."""
    if len(values_sequence) < 2:
        raise ValueError("one-turn path requires at least two observed rows")

    def other_pixels(position: int) -> np.ndarray:
        reference = np.asarray(reference_xy[valid_indices[position]], float)
        disparity = values_sequence[position]
        x = (
            reference[0] - disparity if reference_view == "left"
            else reference[0] + disparity)
        return np.column_stack([x, np.full(len(disparity), reference[1])])

    first_gap = max(valid_indices[1] - valid_indices[0], 1)
    base_pair = (
        unary_costs[0][:, None] + unary_costs[1][None, :]
        + max(float(transition_weight), 0.0)
        * (values_sequence[1][None, :] - values_sequence[0][:, None]) ** 2
        / first_gap)
    # Topology states are: 0 before the turn, 1 inside the single contiguous
    # sharp-turn cluster, and 2 after it. A physical tight V normally exceeds
    # the angle threshold at several adjacent collocation points; counting
    # every such point as a new turn made valid paths infeasible.
    pair_cost = np.full((*base_pair.shape, 3), np.inf, dtype=np.float64)
    pair_cost[:, :, 0] = base_pair
    parents: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]
    threshold = float(sharp_turn_threshold_deg)
    for position in range(2, len(valid_indices)):
        previous_gap = max(
            valid_indices[position - 1] - valid_indices[position - 2], 1)
        current_gap = max(
            valid_indices[position] - valid_indices[position - 1], 1)
        previous_previous = values_sequence[position - 2]
        previous = values_sequence[position - 1]
        current = values_sequence[position]
        previous_slope = (
            previous[None, :, None]
            - previous_previous[:, None, None]) / previous_gap
        current_slope = (
            current[None, None, :]
            - previous[None, :, None]) / current_gap
        smooth_cost = (
            unary_costs[position][None, None, :]
            + max(float(transition_weight), 0.0)
            * current_slope ** 2 * current_gap
            + max(float(curvature_weight), 0.0)
            * (current_slope - previous_slope) ** 2)

        first_vector = (
            other_pixels(position - 1)[None, :, None, :]
            - other_pixels(position - 2)[:, None, None, :])
        second_vector = (
            other_pixels(position)[None, None, :, :]
            - other_pixels(position - 1)[None, :, None, :])
        denominator = (
            np.linalg.norm(first_vector, axis=3)
            * np.linalg.norm(second_vector, axis=3))
        cosine = np.divide(
            np.sum(first_vector * second_vector, axis=3), denominator,
            out=np.ones_like(denominator), where=denominator > 1e-6)
        sharp = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) >= threshold
        location_cost = 0.0
        if expected_turn_index is not None:
            location_error = (
                valid_indices[position - 1] - int(expected_turn_index)) / max(
                    1, int(turn_window_samples))
            # The good-eye epipolar extremum is a location prior, not a hard
            # geometric constraint. Calibration residual and finite mask
            # thickness can shift the bad-eye turn by several samples.
            location_cost = 2.0 * float(location_error * location_error)

        next_cost = np.full(
            (len(previous), len(current), 3), np.inf, dtype=np.float64)
        parent_candidate = np.full(next_cost.shape, -1, dtype=np.int32)
        parent_state = np.full(next_cost.shape, -1, dtype=np.int8)
        for old_state in (0, 1, 2):
            total = pair_cost[:, :, old_state, None] + smooth_cost
            total = total + sharp * location_cost
            for new_state in (0, 1, 2):
                if old_state == 0:
                    allowed = sharp if new_state == 1 else (
                        ~sharp if new_state == 0 else np.zeros_like(sharp))
                elif old_state == 1:
                    allowed = sharp if new_state == 1 else (
                        ~sharp if new_state == 2 else np.zeros_like(sharp))
                else:
                    allowed = ~sharp if new_state == 2 else np.zeros_like(sharp)
                masked = np.where(allowed, total, np.inf)
                candidate = np.min(masked, axis=0)
                cursor = np.argmin(masked, axis=0).astype(np.int32)
                improve = candidate < next_cost[:, :, new_state]
                next_cost[:, :, new_state][improve] = candidate[improve]
                parent_candidate[:, :, new_state][improve] = cursor[improve]
                parent_state[:, :, new_state][improve] = old_state
        if not np.any(np.isfinite(next_cost)):
            raise ValueError("no_mask_path_satisfies_one_turn_topology")
        pair_cost = next_cost
        parents.append((parent_candidate, parent_state))

    terminal = np.unravel_index(
        int(np.argmin(pair_cost)), pair_cost.shape)
    chosen = np.empty(len(valid_indices), dtype=np.int32)
    chosen[-2], chosen[-1] = int(terminal[0]), int(terminal[1])
    state = int(terminal[2])
    for position in range(len(valid_indices) - 1, 1, -1):
        candidate_parent, state_parent = parents[position]
        old_candidate = int(candidate_parent[
            chosen[position - 1], chosen[position], state])
        old_state = int(state_parent[
            chosen[position - 1], chosen[position], state])
        if old_candidate < 0 or old_state < 0:
            raise ValueError("one_turn_topology_backtrack_failed")
        chosen[position - 2] = old_candidate
        state = old_state
    return chosen


def select_overlap_aware_disparity_path(
        candidates: list[np.ndarray],
        data_costs: list[np.ndarray],
        disparity_prior_px: np.ndarray | None = None,
        disparity_prior_weight: float = 0.0,
        anchors: list[tuple[int, float, float]] | None = None,
        transition_weight: float = 0.25,
        curvature_weight: float = 1.0,
        reference_xy: np.ndarray | None = None,
        reference_view: str = "left",
        maximum_sharp_turns: int | None = None,
        sharp_turn_threshold_deg: float = 40.0,
        expected_turn_index: int | None = None,
        turn_window_samples: int = 12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select a smooth path through unordered epipolar mask candidates.

    This is a pair-state dynamic program over disparity, not over other-eye
    arc position. Consequently it supports many-to-one image projection while
    penalizing one-sample ridge switches before nearest-candidate snapping.
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
    scale = 2.0
    unary_costs: list[np.ndarray] = []
    for index in valid_indices:
        values = np.asarray(candidates[index], dtype=np.float64)
        unary = np.asarray(data_costs[index], dtype=np.float64).copy()
        if prior is not None and np.isfinite(prior[index]) and prior[index] > 0.0:
            unary += max(float(disparity_prior_weight), 0.0) * (
                (values - prior[index]) / scale) ** 2
        for anchor_value, anchor_weight in anchor_map.get(index, []):
            unary += max(anchor_weight, 0.0) ** 2 * (
                (values - anchor_value) / scale) ** 2
        unary_costs.append(unary)

    values_sequence = [
        np.asarray(candidates[index], dtype=np.float64)
        for index in valid_indices]
    if maximum_sharp_turns is not None:
        if int(maximum_sharp_turns) != 1:
            raise ValueError("only the one-sharp-turn topology is supported")
        if reference_xy is None:
            raise ValueError("one-turn topology requires reference pixels")
        chosen = _one_turn_candidate_path(
            values_sequence, unary_costs, valid_indices,
            np.asarray(reference_xy, dtype=np.float64), reference_view,
            transition_weight, curvature_weight, sharp_turn_threshold_deg,
            expected_turn_index, turn_window_samples)
        selected = np.full(count, np.nan, dtype=np.float64)
        for index, cursor in zip(valid_indices, chosen):
            selected[index] = candidates[index][cursor]
        observed = np.isfinite(selected)
        return selected, observed, ambiguity_fraction

    first_gap = max(valid_indices[1] - valid_indices[0], 1)
    pair_cost = (
        unary_costs[0][:, None] + unary_costs[1][None, :]
        + max(float(transition_weight), 0.0)
        * (values_sequence[1][None, :] - values_sequence[0][:, None]) ** 2
        / first_gap)
    parent_tables: list[np.ndarray | None] = [None, None]
    for position in range(2, len(valid_indices)):
        previous_gap = max(
            valid_indices[position - 1] - valid_indices[position - 2], 1)
        current_gap = max(
            valid_indices[position] - valid_indices[position - 1], 1)
        previous_previous = values_sequence[position - 2]
        previous = values_sequence[position - 1]
        current = values_sequence[position]
        previous_slope = (
            previous[None, :, None]
            - previous_previous[:, None, None]) / previous_gap
        current_slope = (
            current[None, None, :]
            - previous[None, :, None]) / current_gap
        costs = (
            pair_cost[:, :, None]
            + unary_costs[position][None, None, :]
            + max(float(transition_weight), 0.0)
            * current_slope ** 2 * current_gap
            + max(float(curvature_weight), 0.0)
            * (current_slope - previous_slope) ** 2)
        parent_tables.append(np.argmin(costs, axis=0).astype(np.int32))
        pair_cost = np.min(costs, axis=0)

    selected = np.full(count, np.nan, dtype=np.float64)
    last_pair = np.unravel_index(int(np.argmin(pair_cost)), pair_cost.shape)
    chosen = np.empty(len(valid_indices), dtype=np.int32)
    chosen[-2], chosen[-1] = int(last_pair[0]), int(last_pair[1])
    for position in range(len(valid_indices) - 1, 1, -1):
        chosen[position - 2] = int(parent_tables[position][
            chosen[position - 1], chosen[position]])
    for index, cursor in zip(valid_indices, chosen):
        selected[index] = candidates[index][cursor]
    observed = np.isfinite(selected)
    return selected, observed, ambiguity_fraction


def epipolar_mask_ambiguity_fraction(
        reference_centerline,
        other_centerline,
        reference_view: str,
        n_samples: int = 96,
        smooth_2d: float | None = None,
        row_search_px: int = 2) -> float:
    """Measure how often a reference row crosses multiple other-eye branches."""
    reference = smooth_polyline_2d(
        reference_centerline.points, int(n_samples), smooth_2d)
    candidates, _, ambiguous = _mask_epipolar_candidates(
        reference, other_centerline, reference_view,
        row_search_px=row_search_px)
    observed = np.asarray([len(values) > 0 for values in candidates])
    return float(np.mean(ambiguous[observed])) if np.any(observed) else 0.0


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
        force_overlap_aware: bool = False,
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
        terminal_refinement_max_p95_degradation_px: float = 0.25,
        marker_left_px: np.ndarray | None = None,
        marker_right_px: np.ndarray | None = None,
        marker_confidence_left: np.ndarray | None = None,
        marker_confidence_right: np.ndarray | None = None,
        marker_disparity_weight: float = 8.0,
        max_marker_epipolar_error_px: float = 6.0) -> dict:
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
    marker_anchor_count = 0
    marker_epipolar_errors: list[float] = []
    if marker_left_px is not None and marker_right_px is not None:
        marker_left = np.asarray(marker_left_px, dtype=np.float64).reshape(-1, 2)
        marker_right = np.asarray(marker_right_px, dtype=np.float64).reshape(-1, 2)
        count = min(len(marker_left), len(marker_right))
        confidence_left = (
            np.ones(count, dtype=np.float64)
            if marker_confidence_left is None else
            np.asarray(marker_confidence_left, dtype=np.float64)[:count])
        confidence_right = (
            np.ones(count, dtype=np.float64)
            if marker_confidence_right is None else
            np.asarray(marker_confidence_right, dtype=np.float64)[:count])
        for index in range(count):
            left_marker, right_marker = marker_left[index], marker_right[index]
            if not (np.all(np.isfinite(left_marker))
                    and np.all(np.isfinite(right_marker))):
                continue
            epipolar_error = float(abs(left_marker[1] - right_marker[1]))
            marker_epipolar_errors.append(epipolar_error)
            disparity_value = float(left_marker[0] - right_marker[0])
            confidence = float(np.sqrt(max(
                confidence_left[index] * confidence_right[index], 0.0)))
            if (epipolar_error > float(max_marker_epipolar_error_px)
                    or disparity_value <= 0.25 or confidence <= 0.0):
                continue
            reference_marker = (
                left_marker if reference_view == "left" else right_marker)
            sample = int(np.argmin(np.linalg.norm(
                reference - reference_marker, axis=1)))
            # These are direct image correspondences.  Their finite weight is a
            # soft disparity observation; no inter-marker length is imposed.
            anchors.append((
                sample, disparity_value,
                float(marker_disparity_weight) * confidence))
            marker_anchor_count += 1
    (reference_interval_labels,
     other_segment_interval_labels,
     marker_interval_count) = (
        _marker_interval_labels(
            reference, other_centerline, reference_view,
            marker_left_px, marker_right_px,
            marker_confidence_left, marker_confidence_right))
    # A neighboring-row candidate cannot be represented by a rectified stereo
    # pair without also carrying its y coordinate.  Forced ill-eye recovery
    # therefore uses exact epipolar rows: every selected sample is guaranteed
    # to reproject into the current bad-eye mask, rather than one or two pixels
    # beyond its boundary after y is reset to the reference row.
    candidate_row_search_px = (
        0 if force_overlap_aware else max(0, int(overlap_row_search_px)))
    mask_candidates, mask_costs, ambiguous_rows = _mask_epipolar_candidates(
        reference, other_centerline, reference_view,
        row_search_px=candidate_row_search_px,
        reference_interval_labels=reference_interval_labels,
        other_segment_interval_labels=other_segment_interval_labels,
        # In forced recovery the existing bad-eye cyan curve may itself be the
        # shortcut. Do not let its exact crossings receive zero cost and
        # self-confirm; current mask ridges and temporal disparity remain.
        include_centerline_crossings=not force_overlap_aware)
    if disparity_prior_px is not None:
        _add_temporal_mask_candidates(
            reference, other_centerline, reference_view,
            disparity_prior_px, mask_candidates, mask_costs,
            row_search_px=candidate_row_search_px)
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
    overlap_evidence_available = bool(
        len(nonempty) >= max(8, n_samples // 4))
    if force_overlap_aware and not overlap_evidence_available:
        # Falling back to ordered intersections in a detected projected loop
        # recreates the shortcut topology that forced recovery is intended to
        # eliminate. Let frame-level temporal recovery handle a genuinely
        # unsupported frame instead of producing a plausible-looking wrong
        # curve.
        raise ValueError("insufficient_forced_overlap_mask_rows")
    use_overlap_aware = bool(
        overlap_aware
        and overlap_evidence_available
        and (
            force_overlap_aware
            or (
                other_self_overlap_fraction
                >= float(overlap_self_fraction_threshold)
                and other_self_overlap_fraction
                > reference_self_overlap_fraction + 0.02)))
    if use_overlap_aware:
        # Interval labels affect only the categorical branch cost attached to
        # exact cyan crossings. They never interpolate disparity or impose
        # within-interval arc length under foreshortening.
        # The farthest epipolar sweep from the registered base in the good eye
        # localizes the only allowed projected hairpin.  Endpoint extrema mean
        # an ordinary monotone view and do not define an interior turn window.
        sweep = np.abs(reference[:, 1] - reference[0, 1])
        expected_turn_index = int(np.argmax(sweep))
        if expected_turn_index < int(0.08 * n_samples) or expected_turn_index > int(0.92 * n_samples):
            expected_turn_index = None
        enforce_one_turn = bool(force_overlap_aware)
        raw_disparity, matched, _ = (
            select_overlap_aware_disparity_path(
                mask_candidates, mask_costs,
                disparity_prior_px=disparity_prior_px,
                disparity_prior_weight=disparity_prior_weight,
                anchors=anchors,
                transition_weight=disparity_first_difference_weight,
                curvature_weight=max(
                    0.25, 0.10 * float(disparity_second_difference_weight)),
                reference_xy=reference,
                reference_view=reference_view,
                maximum_sharp_turns=1 if enforce_one_turn else None,
                expected_turn_index=(
                    expected_turn_index if enforce_one_turn else None),
                turn_window_samples=max(6, int(round(0.12 * n_samples)))))
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
    if use_overlap_aware and force_overlap_aware:
        # The pair-state DP has already selected one globally consistent mask
        # branch at every observed epipolar row. A second unconstrained robust
        # solve followed by nearest-candidate snapping can silently switch those
        # choices and recreate two or three V turns. Preserve selected samples
        # exactly and interpolate only genuinely unsupported rows.
        selected = matched & np.isfinite(raw_disparity)
        selected_indices = np.flatnonzero(selected)
        if len(selected_indices) < max(8, n_samples // 4):
            raise ValueError("insufficient_forced_overlap_selected_rows")
        disparity = np.interp(
            np.arange(n_samples), selected_indices,
            raw_disparity[selected_indices])
        robust_weights = selected.astype(np.float64)
    else:
        disparity, robust_weights = regularize_disparity_local(
            raw_disparity, matched, anchors,
            first_difference_weight=disparity_first_difference_weight,
            second_difference_weight=disparity_second_difference_weight,
            huber_delta_px=disparity_huber_delta_px)
    if use_overlap_aware and not force_overlap_aware:
        # Smoothness and temporal anchors must choose among current-frame mask
        # branches, not pull the reconstructed path into unsupported image
        # space. Snap each observed material sample to its closest epipolar
        # mask candidate. Missing rows retain the regularized interpolation.
        for index, values in enumerate(mask_candidates):
            if len(values):
                disparity[index] = values[int(np.argmin(
                    np.abs(values - disparity[index])))]

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
        "overlap_aware_forced": int(
            use_overlap_aware and force_overlap_aware),
        "overlap_one_turn_enforced": int(
            use_overlap_aware and force_overlap_aware),
        "overlap_expected_turn_index": (
            int(expected_turn_index)
            if use_overlap_aware and force_overlap_aware
            and expected_turn_index is not None else -1),
        "epipolar_ambiguity_fraction": mask_ambiguity_fraction,
        "other_eye_self_overlap_fraction": other_self_overlap_fraction,
        "reference_eye_self_overlap_fraction": reference_self_overlap_fraction,
        "centerline_tip_anchor_used": int(centerline_tip_anchor_used),
        "centerline_tip_epipolar_error_px": centerline_tip_epipolar_error_px,
        "marker_anchor_count": int(marker_anchor_count),
        "marker_interval_boundary_count": int(marker_interval_count),
        "marker_epipolar_error_max_px": (
            float(max(marker_epipolar_errors))
            if marker_epipolar_errors else float("nan")),
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
