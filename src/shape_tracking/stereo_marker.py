"""Stereo pose estimation for a single square ArUco marker.

The ZED images used by this project are rectified.  The right camera therefore
has the same orientation as the left camera and is translated by ``-baseline``
along x.  The implementation also accepts a general right-camera transform so
future captures can store the complete SDK calibration.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def square_object_points(marker_size_m):
    half = float(marker_size_m) / 2.0
    return np.asarray([
        [-half, +half, 0.0],
        [+half, +half, 0.0],
        [+half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def rectified_right_T_left(baseline_m):
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = -float(baseline_m)
    return transform


def _project(object_points, rvec, tvec, K, dist):
    pixels, _ = cv2.projectPoints(
        object_points, np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1), K, dist)
    return pixels.reshape(-1, 2)


def estimate_stereo_object_pose(
        object_points_left, image_points_left,
        object_points_right, image_points_right,
        K_left, dist_left, K_right, dist_right, right_T_left,
        initial_rvec, initial_tvec):
    """Refine a common rigid-object pose against two camera observations.

    Left and right observations may contain different object-point subsets,
    which is required for partially visible ChArUco boards.
    """
    object_points_left = np.asarray(
        object_points_left, dtype=np.float64).reshape(-1, 3)
    object_points_right = np.asarray(
        object_points_right, dtype=np.float64).reshape(-1, 3)
    image_points_left = np.asarray(
        image_points_left, dtype=np.float64).reshape(-1, 2)
    image_points_right = np.asarray(
        image_points_right, dtype=np.float64).reshape(-1, 2)
    if len(object_points_left) != len(image_points_left):
        raise ValueError("left object/image point counts do not match")
    if len(object_points_right) != len(image_points_right):
        raise ValueError("right object/image point counts do not match")
    if len(object_points_left) < 4 or len(object_points_right) < 4:
        raise ValueError("stereo pose requires at least 4 points per camera")
    K_left = np.asarray(K_left, dtype=np.float64)
    K_right = np.asarray(K_right, dtype=np.float64)
    dist_left = np.asarray(dist_left, dtype=np.float64)
    dist_right = np.asarray(dist_right, dtype=np.float64)
    right_T_left = np.asarray(right_T_left, dtype=np.float64)
    right_R_left = right_T_left[:3, :3]
    right_t_left = right_T_left[:3, 3]

    def residual(parameters):
        left_R_object = Rotation.from_rotvec(parameters[:3]).as_matrix()
        left_t_object = parameters[3:6]
        right_R_object = right_R_left @ left_R_object
        right_t_object = right_R_left @ left_t_object + right_t_left
        predicted_left = _project(
            object_points_left, parameters[:3], left_t_object,
            K_left, dist_left)
        predicted_right = _project(
            object_points_right,
            Rotation.from_matrix(right_R_object).as_rotvec(),
            right_t_object, K_right, dist_right)
        return np.concatenate([
            (predicted_left - image_points_left).reshape(-1),
            (predicted_right - image_points_right).reshape(-1),
        ])

    initial = np.concatenate([
        np.asarray(initial_rvec, dtype=np.float64).reshape(3),
        np.asarray(initial_tvec, dtype=np.float64).reshape(3)])
    result = least_squares(
        residual, initial, method="trf", loss="soft_l1", f_scale=1.0,
        max_nfev=300)
    rvec, tvec = result.x[:3], result.x[3:6]
    right_t_object = right_R_left @ tvec + right_t_left
    if tvec[2] <= 0 or right_t_object[2] <= 0:
        raise RuntimeError("stereo object pose is behind a camera")
    flat_residual = residual(result.x)
    left_count = len(image_points_left)
    left_residual = flat_residual[:2 * left_count].reshape(-1, 2)
    right_residual = flat_residual[2 * left_count:].reshape(-1, 2)
    left_errors = np.linalg.norm(left_residual, axis=1)
    right_errors = np.linalg.norm(right_residual, axis=1)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    transform[:3, 3] = tvec
    return {
        "left_camera_T_object_m": transform,
        "rvec": rvec.reshape(3, 1),
        "tvec": tvec.reshape(3, 1),
        "left_reprojection_rmse_px": float(np.sqrt(np.mean(
            left_errors ** 2))),
        "right_reprojection_rmse_px": float(np.sqrt(np.mean(
            right_errors ** 2))),
        "stereo_reprojection_rmse_px": float(np.sqrt(np.mean(
            np.concatenate([left_errors, right_errors]) ** 2))),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def estimate_stereo_marker_pose(
        object_points, left_corners, right_corners,
        K_left, dist_left, K_right, dist_right, right_T_left):
    """Estimate ``left_camera_T_marker`` from both synchronized images.

    A left-image IPPE solution initializes a nonlinear optimization whose
    residual contains all four corners in both cameras.  Scoring both IPPE
    branches with the stereo observations removes the planar pose ambiguity
    that can survive a monocular reprojection-error comparison.
    """
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    left_corners = np.asarray(left_corners, dtype=np.float64).reshape(-1, 2)
    right_corners = np.asarray(right_corners, dtype=np.float64).reshape(-1, 2)
    K_left = np.asarray(K_left, dtype=np.float64)
    K_right = np.asarray(K_right, dtype=np.float64)
    dist_left = np.asarray(dist_left, dtype=np.float64)
    dist_right = np.asarray(dist_right, dtype=np.float64)
    right_T_left = np.asarray(right_T_left, dtype=np.float64)
    if object_points.shape[0] != left_corners.shape[0]:
        raise ValueError("left corner count does not match object points")
    if left_corners.shape != right_corners.shape:
        raise ValueError("left/right corner counts do not match")

    solved = cv2.solvePnPGeneric(
        object_points, left_corners, K_left, dist_left,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not solved[0]:
        raise RuntimeError("left-camera IPPE failed")
    candidates = []
    for rvec, tvec in zip(solved[1], solved[2]):
        if np.asarray(tvec).reshape(3)[2] <= 0:
            continue
        try:
            result = estimate_stereo_object_pose(
                object_points, left_corners, object_points, right_corners,
                K_left, dist_left, K_right, dist_right, right_T_left,
                rvec, tvec)
        except RuntimeError:
            continue
        candidates.append((result["stereo_reprojection_rmse_px"], result))
    if not candidates:
        raise RuntimeError("no physically valid stereo marker-pose solution")

    _, result = min(candidates, key=lambda item: item[0])
    result = dict(result)
    result["left_camera_T_marker_m"] = result.pop(
        "left_camera_T_object_m")
    result.update({
        "left_corners": left_corners,
        "right_corners": right_corners,
        "rectified_row_mismatch_mean_px": float(np.mean(np.abs(
            left_corners[:, 1] - right_corners[:, 1]))),
        "rectified_row_mismatch_max_px": float(np.max(np.abs(
            left_corners[:, 1] - right_corners[:, 1]))),
    })
    return result
