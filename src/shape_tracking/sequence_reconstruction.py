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
        base_camera_m: np.ndarray,
        tip_camera_m: np.ndarray,
        n_samples: int = 96,
        disparity_order: int = 3,
        smooth_2d: float | None = None,
        base_depth_weight: float = 5.0,
        tip_depth_weight: float = 2.0) -> dict:
    """Lift a 2D shaft centerline with endpoint-regularized disparity.

    The registered base depth strongly constrains the proximal end. The EM tip
    depth is a weaker distal prior because the visible blue segment stops before
    the tape and printed tip housing. Exact base/tip connection is handled in
    :func:`shape_tracking.geometry.assemble_anchored_shape`.
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    left = smooth_polyline_2d(
        left_centerline.points, n_samples, smooth_2d)
    right_dense = smooth_polyline_2d(
        right_centerline.points, max(4 * n_samples, 300), smooth_2d)
    right, matched = epipolar_correspond(left, right_dense)
    normalized_arc = np.linspace(0.0, 1.0, n_samples)
    raw_disparity = left[:, 0] - right[:, 0]
    if np.count_nonzero(matched) < disparity_order + 2:
        right_arc = smooth_polyline_2d(
            right_centerline.points, n_samples, smooth_2d)
        raw_disparity = left[:, 0] - right_arc[:, 0]
        matched = np.ones(n_samples, dtype=bool)

    fit_arc = list(normalized_arc[matched])
    fit_disparity = list(raw_disparity[matched])
    fit_weight = [1.0] * len(fit_arc)
    base_camera_m = np.asarray(base_camera_m, dtype=np.float64)
    tip_camera_m = np.asarray(tip_camera_m, dtype=np.float64)
    if base_camera_m[2] > 1e-6:
        fit_arc.append(0.0)
        fit_disparity.append(float(fx * baseline_m / base_camera_m[2]))
        fit_weight.append(float(base_depth_weight))
    if tip_camera_m[2] > 1e-6:
        fit_arc.append(1.0)
        fit_disparity.append(float(fx * baseline_m / tip_camera_m[2]))
        fit_weight.append(float(tip_depth_weight))
    order = int(min(disparity_order, len(fit_arc) - 1))
    coefficients = np.polyfit(
        np.asarray(fit_arc), np.asarray(fit_disparity), order,
        w=np.asarray(fit_weight))
    disparity = np.polyval(coefficients, normalized_arc)
    if np.any(disparity <= 0):
        positive = disparity[disparity > 0]
        floor = max(1e-3, float(np.percentile(positive, 5))
                    if len(positive) else 1e-3)
        disparity = np.clip(disparity, floor, None)
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
        "disparity_coefficients": coefficients,
    }
