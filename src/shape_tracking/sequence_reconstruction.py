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
        iterations: int = 8) -> tuple[np.ndarray, np.ndarray]:
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
    regularizer = (
        max(float(first_difference_weight), 0.0) * (first.T @ first)
        + max(float(second_difference_weight), 0.0) * (second.T @ second))
    anchor_list = anchors or []
    robust_weights = np.ones(np.count_nonzero(valid), dtype=np.float64)
    valid_indices = indices[valid]
    delta = max(float(huber_delta_px), 1e-6)
    for _ in range(max(1, int(iterations))):
        normal = regularizer.copy()
        rhs = np.zeros(n, dtype=np.float64)
        normal[valid_indices, valid_indices] += robust_weights
        rhs[valid_indices] += robust_weights * raw[valid]
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
        disparity_prior_weight: float = 0.0) -> dict:
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
    # Use the regularized disparity to construct a finite, rectified pixel pair.
    # In particular, right-reference epipolar matching can leave unmatched left
    # coordinates as NaN even though neighboring observations determine a valid
    # regularized disparity there.
    if reference_view == "left":
        right = np.column_stack([left[:, 0] - disparity, left[:, 1]])
    else:
        left = np.column_stack([right[:, 0] + disparity, right[:, 1]])
    depth = fx * baseline_m / disparity
    x = (left[:, 0] - cx) * depth / fx
    y = (left[:, 1] - cy) * depth / fy
    points_camera_m = np.column_stack([x, y, depth])
    left_distance, right_distance = reprojection_distances(
        points_camera_m, left_centerline, right_centerline, K, baseline_m)
    combined_distance = np.concatenate([left_distance, right_distance])
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
        "reference_view": reference_view,
        "ordered_left_px": left,
        "ordered_right_px": right,
    }
