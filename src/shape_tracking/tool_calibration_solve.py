"""Solve stereo ArUco-to-dual-5DOF-EM calibration from a capture session.

Example:
    python -m shape_tracking.tool_calibration_solve SESSION_DIRECTORY

Only the observable part of each 5-DOF coil orientation is used: its local
z-axis.  Roll about z is intentionally absent from the calibration residual.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import boards, registration
from .stereo_marker import (
    estimate_stereo_object_pose,
    estimate_stereo_marker_pose,
    rectified_right_T_left,
    square_object_points,
)


# CAD uses a corner origin and +z into the housing. OpenCV uses the center of
# the 25 mm marker and +z out of its printed front, hence x/y recentering and
# z negation. Values below are coil physical tip positions, in millimetres.
CAD_COIL_TIPS_MARKER_MM = {
    "003": np.asarray([1.9, 0.0, -3.0]),
    "07222026_01": np.asarray([-1.9, 0.0, -3.0]),
}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _detect_marker_corners(image, marker_id):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(boards.get_dictionary(), parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        raise RuntimeError(f"marker {marker_id} was not detected")
    matches = np.flatnonzero(ids.reshape(-1) == int(marker_id))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one marker {marker_id}, detected {len(matches)}")
    return np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)


def _load_stereo_calibration(session_dir):
    with np.load(session_dir / "left_intrinsics.npz") as values:
        K_left = np.asarray(values["K"], dtype=np.float64)
        dist_left = np.asarray(values["dist"], dtype=np.float64)
        K_right = np.asarray(
            values["K_right"] if "K_right" in values.files else K_left,
            dtype=np.float64)
        dist_right = np.asarray(
            values["dist_right"] if "dist_right" in values.files
            else np.zeros(5), dtype=np.float64)
        if "right_camera_T_left_camera_m" in values.files:
            right_T_left = np.asarray(
                values["right_camera_T_left_camera_m"], dtype=np.float64)
        else:
            right_T_left = rectified_right_T_left(float(values["baseline_m"]))
    return K_left, dist_left, K_right, dist_right, right_T_left


def estimate_saved_field_pose_stereo(session_dir):
    """Re-estimate the saved field-lock ChArUco pose from both images."""
    session_dir = Path(session_dir)
    metadata = json.loads(
        (session_dir / "capture_metadata.json").read_text(encoding="utf-8"))
    field = metadata["field_generator"]
    _, entries = boards.build_boards(additional=[(
        int(field["board_index"]), int(field["marker_ids"][0]))])
    board_entry = next(
        entry for entry in entries
        if entry.index == int(field["board_index"]))
    left_image = cv2.imread(str(session_dir / "field_lock_left.png"))
    right_image = cv2.imread(str(session_dir / "field_lock_right.png"))
    if left_image is None or right_image is None:
        raise RuntimeError("saved field-lock stereo images are missing")
    K_left, dist_left, K_right, dist_right, right_T_left = (
        _load_stereo_calibration(session_dir))
    left_results, _ = registration.detect_boards(
        cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY),
        [board_entry], K_left, dist_left)
    right_results, _ = registration.detect_boards(
        cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY),
        [board_entry], K_right, dist_right)
    if not left_results or not right_results or not left_results[0].has_pose:
        raise RuntimeError("field ChArUco board lacks a stereo pose")
    left_pose, right_pose = left_results[0], right_results[0]
    if left_pose.charuco_ids is None or right_pose.charuco_ids is None:
        raise RuntimeError("field ChArUco corners are missing")
    left_ids = np.asarray(left_pose.charuco_ids, dtype=np.int32).reshape(-1)
    right_ids = np.asarray(right_pose.charuco_ids, dtype=np.int32).reshape(-1)
    chessboard_points = np.asarray(
        board_entry.board.getChessboardCorners(), dtype=np.float64)
    result = estimate_stereo_object_pose(
        chessboard_points[left_ids], left_pose.charuco_corners,
        chessboard_points[right_ids], right_pose.charuco_corners,
        K_left, dist_left, K_right, dist_right, right_T_left,
        left_pose.rvec, left_pose.tvec)
    result["left_camera_T_field_marker_m"] = result.pop(
        "left_camera_T_object_m")
    result["left_corner_count"] = int(len(left_ids))
    result["right_corner_count"] = int(len(right_ids))
    result["source"] = "saved synchronized stereo field-lock images"
    return result


def estimate_session_marker_poses(session_dir):
    """Re-estimate every saved fixture-marker pose using both ZED images."""
    session_dir = Path(session_dir)
    metadata = json.loads(
        (session_dir / "capture_metadata.json").read_text(encoding="utf-8"))
    snapshots = json.loads(
        (session_dir / "snapshots.json").read_text(encoding="utf-8"))
    marker_id = int(metadata["tip_marker"]["id"])
    marker_size_m = (
        float(metadata["tip_marker"]["black_edge_size_mm"]) * 0.001)
    object_points = square_object_points(marker_size_m)
    calibration = _load_stereo_calibration(session_dir)
    results = []
    for snapshot in snapshots:
        image_names = snapshot["images"]
        left = cv2.imread(str(
            session_dir / "snapshots" / image_names["left"]))
        right = cv2.imread(str(
            session_dir / "snapshots" / image_names["right"]))
        if left is None or right is None:
            raise RuntimeError(
                f"could not load snapshot {snapshot['snapshot_index']} images")
        left_corners = _detect_marker_corners(left, marker_id)
        right_corners = _detect_marker_corners(right, marker_id)
        pose = estimate_stereo_marker_pose(
            object_points, left_corners, right_corners, *calibration)
        pose["snapshot_index"] = int(snapshot["snapshot_index"])
        pose["snapshot"] = snapshot
        results.append(pose)
    return results


def _initial_coil_centers(marker_poses, nominal_camera_T_aurora):
    centers = {}
    aurora_T_camera = np.linalg.inv(nominal_camera_T_aurora)
    for pose in marker_poses:
        marker_T_camera = np.linalg.inv(pose["left_camera_T_marker_m"])
        for coil in pose["snapshot"]["coils"]:
            point_aurora = np.r_[
                np.asarray(coil["position_aurora_mm"], dtype=np.float64)
                * 0.001, 1.0]
            point_marker = marker_T_camera @ nominal_camera_T_aurora @ point_aurora
            centers.setdefault(coil["part_number"], []).append(
                point_marker[:3] * 1000.0)
    return {
        part: np.median(np.asarray(points), axis=0)
        for part, points in centers.items()}


def solve_roll_invariant_calibration(
        marker_poses, nominal_camera_T_aurora,
        expected_axis_marker=(0.0, -1.0, 0.0),
        position_scale_mm=0.5, direction_scale_deg=0.5):
    """Jointly fit camera/Aurora pose and marker-frame coil centers.

    Position observations constrain the two coil origins.  Orientation
    observations compare only each Aurora coil z-axis with the known housing
    direction in the marker frame.  The reported 5-DOF coil roll is never used.
    """
    nominal = np.asarray(nominal_camera_T_aurora, dtype=np.float64)
    expected_axis = np.asarray(expected_axis_marker, dtype=np.float64)
    expected_axis /= np.linalg.norm(expected_axis)
    parts = sorted({
        coil["part_number"]
        for pose in marker_poses for coil in pose["snapshot"]["coils"]})
    if len(parts) != 2:
        raise ValueError(f"expected two coils, found {parts}")
    part_index = {part: index for index, part in enumerate(parts)}
    initial_centers = _initial_coil_centers(marker_poses, nominal)
    initial = np.concatenate([
        Rotation.from_matrix(nominal[:3, :3]).as_rotvec(),
        nominal[:3, 3] * 1000.0,
        initial_centers[parts[0]],
        initial_centers[parts[1]],
    ])
    direction_scale_rad = np.deg2rad(float(direction_scale_deg))

    def physical_residuals(parameters):
        camera_R_aurora = Rotation.from_rotvec(parameters[:3]).as_matrix()
        camera_t_aurora_mm = parameters[3:6]
        centers_marker_mm = parameters[6:12].reshape(2, 3)
        position_residuals = []
        direction_residuals = []
        labels = []
        for pose in marker_poses:
            camera_T_marker = pose["left_camera_T_marker_m"]
            camera_R_marker = camera_T_marker[:3, :3]
            camera_t_marker_mm = camera_T_marker[:3, 3] * 1000.0
            target_axis_camera = camera_R_marker @ expected_axis
            for coil in pose["snapshot"]["coils"]:
                index = part_index[coil["part_number"]]
                aurora_T_coil = np.asarray(
                    coil["aurora_T_coil_m"], dtype=np.float64)
                point_aurora_mm = aurora_T_coil[:3, 3] * 1000.0
                axis_aurora = aurora_T_coil[:3, 2]
                predicted_camera = (
                    camera_R_aurora @ point_aurora_mm
                    + camera_t_aurora_mm)
                marker_camera = (
                    camera_R_marker @ centers_marker_mm[index]
                    + camera_t_marker_mm)
                position_residuals.append(predicted_camera - marker_camera)
                measured_axis_camera = camera_R_aurora @ axis_aurora
                direction_residuals.append(
                    measured_axis_camera - target_axis_camera)
                labels.append((pose["snapshot_index"], coil["part_number"]))
        return (
            np.asarray(position_residuals),
            np.asarray(direction_residuals), labels)

    def normalized_residuals(parameters):
        position, direction, _ = physical_residuals(parameters)
        return np.concatenate([
            (position / float(position_scale_mm)).reshape(-1),
            (direction / direction_scale_rad).reshape(-1),
        ])

    optimized = least_squares(
        normalized_residuals, initial, method="trf", loss="soft_l1",
        f_scale=1.0, max_nfev=3000)
    position, direction_cross, labels = physical_residuals(optimized.x)
    position_norm = np.linalg.norm(position, axis=1)
    direction_deg = np.rad2deg(2.0 * np.arcsin(np.clip(
        np.linalg.norm(direction_cross, axis=1) * 0.5, 0.0, 1.0)))
    camera_T_aurora = np.eye(4)
    camera_T_aurora[:3, :3] = Rotation.from_rotvec(
        optimized.x[:3]).as_matrix()
    camera_T_aurora[:3, 3] = optimized.x[3:6] * 0.001
    centers = optimized.x[6:12].reshape(2, 3)
    correction = camera_T_aurora @ np.linalg.inv(nominal)

    per_snapshot = []
    for snapshot_index in sorted({label[0] for label in labels}):
        selected = [
            index for index, label in enumerate(labels)
            if label[0] == snapshot_index]
        per_snapshot.append({
            "snapshot_index": int(snapshot_index),
            "position_rms_mm": float(np.sqrt(np.mean(
                position_norm[selected] ** 2))),
            "position_max_mm": float(np.max(position_norm[selected])),
            "axis_mean_deg": float(np.mean(direction_deg[selected])),
            "axis_max_deg": float(np.max(direction_deg[selected])),
        })

    coil_parallel_angles = []
    for pose in marker_poses:
        axes = [
            np.asarray(coil["aurora_T_coil_m"])[:3, 2]
            for coil in pose["snapshot"]["coils"]]
        angle = np.rad2deg(np.arccos(np.clip(
            np.dot(axes[0], axes[1]), -1.0, 1.0)))
        coil_parallel_angles.append(float(angle))
    baseline = centers[part_index["003"]] - centers[
        part_index["07222026_01"]]

    return {
        "method": (
            "joint robust stereo-position and 5DOF-z-axis fit; "
            "coil roll ignored"),
        "optimizer": {
            "success": bool(optimized.success),
            "message": str(optimized.message),
            "cost": float(optimized.cost),
            "position_scale_mm": float(position_scale_mm),
            "direction_scale_deg": float(direction_scale_deg),
        },
        "calibrated_left_camera_T_aurora_m": camera_T_aurora,
        "nominal_left_camera_T_aurora_m": nominal,
        "left_camera_correction_m": correction,
        "correction_translation_mm": correction[:3, 3] * 1000.0,
        "correction_rotation_deg": float(np.linalg.norm(
            Rotation.from_matrix(correction[:3, :3]).as_rotvec())
            * 180.0 / np.pi),
        "coil_centers_marker_mm": {
            part: centers[index] for part, index in part_index.items()},
        "coil_baseline_072_to_003_marker_mm": baseline,
        "coil_separation_mm": float(np.linalg.norm(baseline)),
        "expected_coil_axis_marker": expected_axis,
        "position_residual_mm": {
            "rms": float(np.sqrt(np.mean(position_norm ** 2))),
            "mean": float(np.mean(position_norm)),
            "median": float(np.median(position_norm)),
            "p95": float(np.percentile(position_norm, 95)),
            "max": float(np.max(position_norm)),
        },
        "coil_axis_residual_deg": {
            "rms": float(np.sqrt(np.mean(direction_deg ** 2))),
            "mean": float(np.mean(direction_deg)),
            "median": float(np.median(direction_deg)),
            "p95": float(np.percentile(direction_deg, 95)),
            "max": float(np.max(direction_deg)),
        },
        "measured_coil_parallelism_deg": {
            "mean": float(np.mean(coil_parallel_angles)),
            "p95": float(np.percentile(coil_parallel_angles, 95)),
            "max": float(np.max(coil_parallel_angles)),
        },
        "per_snapshot": per_snapshot,
    }


def solve_cad_constrained_calibration(
        marker_poses, nominal_camera_T_aurora,
        coil_tips_marker_mm=CAD_COIL_TIPS_MARKER_MM,
        expected_axis_marker=(0.0, -1.0, 0.0),
        position_scale_mm=0.5, direction_scale_deg=0.5):
    """Fit camera/Aurora pose and one axial sensing offset per 5-DOF coil."""
    nominal = np.asarray(nominal_camera_T_aurora, dtype=np.float64)
    expected_axis = np.asarray(expected_axis_marker, dtype=np.float64)
    expected_axis /= np.linalg.norm(expected_axis)
    tips = {
        part: np.asarray(point, dtype=np.float64)
        for part, point in coil_tips_marker_mm.items()}
    parts = sorted(tips)
    observed_parts = sorted({
        coil["part_number"]
        for pose in marker_poses for coil in pose["snapshot"]["coils"]})
    if observed_parts != parts:
        raise ValueError(
            f"CAD parts {parts} do not match observed parts {observed_parts}")
    part_index = {part: index for index, part in enumerate(parts)}

    unconstrained = solve_roll_invariant_calibration(
        marker_poses, nominal, expected_axis,
        position_scale_mm, direction_scale_deg)
    initial_transform = np.asarray(
        unconstrained["calibrated_left_camera_T_aurora_m"])
    initial_centers = unconstrained["coil_centers_marker_mm"]
    initial_offsets = [
        np.dot(np.asarray(initial_centers[part]) - tips[part], expected_axis)
        for part in parts]
    initial = np.concatenate([
        Rotation.from_matrix(initial_transform[:3, :3]).as_rotvec(),
        initial_transform[:3, 3] * 1000.0,
        initial_offsets,
    ])
    direction_scale_rad = np.deg2rad(float(direction_scale_deg))

    def physical_residuals(parameters):
        camera_R_aurora = Rotation.from_rotvec(parameters[:3]).as_matrix()
        camera_t_aurora_mm = parameters[3:6]
        offsets = parameters[6:8]
        position_residuals = []
        direction_residuals = []
        labels = []
        for pose in marker_poses:
            camera_T_marker = pose["left_camera_T_marker_m"]
            camera_R_marker = camera_T_marker[:3, :3]
            camera_t_marker_mm = camera_T_marker[:3, 3] * 1000.0
            target_axis_camera = camera_R_marker @ expected_axis
            for coil in pose["snapshot"]["coils"]:
                part = coil["part_number"]
                index = part_index[part]
                aurora_T_coil = np.asarray(
                    coil["aurora_T_coil_m"], dtype=np.float64)
                center_marker_mm = (
                    tips[part] + offsets[index] * expected_axis)
                predicted_camera = (
                    camera_R_aurora @ (aurora_T_coil[:3, 3] * 1000.0)
                    + camera_t_aurora_mm)
                expected_camera = (
                    camera_R_marker @ center_marker_mm
                    + camera_t_marker_mm)
                position_residuals.append(
                    predicted_camera - expected_camera)
                direction_residuals.append(
                    camera_R_aurora @ aurora_T_coil[:3, 2]
                    - target_axis_camera)
                labels.append((pose["snapshot_index"], part))
        return (np.asarray(position_residuals),
                np.asarray(direction_residuals), labels)

    def normalized_residuals(parameters):
        position, direction, _ = physical_residuals(parameters)
        return np.concatenate([
            (position / float(position_scale_mm)).reshape(-1),
            (direction / direction_scale_rad).reshape(-1),
        ])

    optimized = least_squares(
        normalized_residuals, initial, method="trf", loss="soft_l1",
        f_scale=1.0, max_nfev=3000)
    position, direction_cross, labels = physical_residuals(optimized.x)
    position_norm = np.linalg.norm(position, axis=1)
    direction_deg = np.rad2deg(2.0 * np.arcsin(np.clip(
        np.linalg.norm(direction_cross, axis=1) * 0.5, 0.0, 1.0)))
    camera_T_aurora = np.eye(4)
    camera_T_aurora[:3, :3] = Rotation.from_rotvec(
        optimized.x[:3]).as_matrix()
    camera_T_aurora[:3, 3] = optimized.x[3:6] * 0.001
    correction = camera_T_aurora @ np.linalg.inv(nominal)
    offsets = optimized.x[6:8]
    centers = {
        part: tips[part] + offsets[index] * expected_axis
        for part, index in part_index.items()}
    per_snapshot = []
    for snapshot_index in sorted({label[0] for label in labels}):
        selected = [
            index for index, label in enumerate(labels)
            if label[0] == snapshot_index]
        per_snapshot.append({
            "snapshot_index": int(snapshot_index),
            "position_rms_mm": float(np.sqrt(np.mean(
                position_norm[selected] ** 2))),
            "position_max_mm": float(np.max(position_norm[selected])),
            "axis_mean_deg": float(np.mean(direction_deg[selected])),
            "axis_max_deg": float(np.max(direction_deg[selected])),
        })
    baseline = centers["003"] - centers["07222026_01"]
    return {
        "method": (
            "CAD-constrained robust stereo-position and 5DOF-z-axis fit; "
            "one axial sensing offset per coil; coil roll ignored"),
        "optimizer": {
            "success": bool(optimized.success),
            "message": str(optimized.message),
            "cost": float(optimized.cost),
            "position_scale_mm": float(position_scale_mm),
            "direction_scale_deg": float(direction_scale_deg),
        },
        "calibrated_left_camera_T_aurora_m": camera_T_aurora,
        "nominal_left_camera_T_aurora_m": nominal,
        "left_camera_correction_m": correction,
        "correction_translation_mm": correction[:3, 3] * 1000.0,
        "correction_rotation_deg": float(np.linalg.norm(
            Rotation.from_matrix(correction[:3, :3]).as_rotvec())
            * 180.0 / np.pi),
        "coil_tip_positions_marker_mm": tips,
        "coil_axial_sensing_offsets_mm": {
            part: float(offsets[index])
            for part, index in part_index.items()},
        "coil_centers_marker_mm": centers,
        "coil_baseline_072_to_003_marker_mm": baseline,
        "coil_separation_mm": float(np.linalg.norm(
            tips["003"] - tips["07222026_01"])),
        "expected_coil_axis_marker": expected_axis,
        "position_residual_mm": {
            "rms": float(np.sqrt(np.mean(position_norm ** 2))),
            "mean": float(np.mean(position_norm)),
            "median": float(np.median(position_norm)),
            "p95": float(np.percentile(position_norm, 95)),
            "max": float(np.max(position_norm)),
        },
        "coil_axis_residual_deg": {
            "rms": float(np.sqrt(np.mean(direction_deg ** 2))),
            "mean": float(np.mean(direction_deg)),
            "median": float(np.median(direction_deg)),
            "p95": float(np.percentile(direction_deg, 95)),
            "max": float(np.max(direction_deg)),
        },
        "per_snapshot": per_snapshot,
    }


def leave_one_out_stability(
        marker_poses, nominal_camera_T_aurora,
        solver=solve_roll_invariant_calibration):
    """Report calibration sensitivity to removal of any accepted pose."""
    if len(marker_poses) < 7:
        return {"performed": False, "reason": "requires at least 7 poses"}
    folds = []
    transforms = []
    separations = []
    for omitted_index in range(len(marker_poses)):
        training = (
            marker_poses[:omitted_index] + marker_poses[omitted_index + 1:])
        fit = solver(training, nominal_camera_T_aurora)
        transform = np.asarray(
            fit["calibrated_left_camera_T_aurora_m"], dtype=np.float64)
        transforms.append(transform)
        separations.append(fit["coil_separation_mm"])
        folds.append({
            "omitted_snapshot_index": int(
                marker_poses[omitted_index]["snapshot_index"]),
            "correction_translation_mm": fit["correction_translation_mm"],
            "correction_rotation_deg": fit["correction_rotation_deg"],
            "coil_separation_mm": fit["coil_separation_mm"],
            **({"coil_axial_sensing_offsets_mm":
                fit["coil_axial_sensing_offsets_mm"]}
               if "coil_axial_sensing_offsets_mm" in fit else {}),
        })
    translations = np.asarray(
        [transform[:3, 3] for transform in transforms]) * 1000.0
    translation_center = np.mean(translations, axis=0)
    translation_deviation = np.linalg.norm(
        translations - translation_center, axis=1)
    mean_rotation = Rotation.from_matrix(np.asarray(
        [transform[:3, :3] for transform in transforms])).mean()
    rotation_deviation = np.asarray([
        np.linalg.norm((
            mean_rotation.inv()
            * Rotation.from_matrix(transform[:3, :3])).as_rotvec())
        * 180.0 / np.pi
        for transform in transforms])
    return {
        "performed": True,
        "fold_count": len(folds),
        "translation_axis_std_mm": np.std(translations, axis=0),
        "translation_max_deviation_from_mean_mm": float(np.max(
            translation_deviation)),
        "rotation_max_deviation_from_mean_deg": float(np.max(
            rotation_deviation)),
        "coil_separation_std_mm": float(np.std(separations)),
        "folds": folds,
    }


def build_report(
        session_dir, expected_separation_mm=3.8,
        max_stereo_reprojection_rmse_px=3.0):
    session_dir = Path(session_dir)
    marker_poses = estimate_session_marker_poses(session_dir)
    field_lock = json.loads(
        (session_dir / "field_frame_lock.json").read_text(encoding="utf-8"))
    if str(field_lock.get("pose_method", "")).startswith("multi-frame"):
        field_stereo = {
            "source": "stored multi-frame synchronized stereo ChArUco lock",
            "left_camera_T_field_marker_m": np.asarray(
                field_lock["left_camera_T_field_marker_m"],
                dtype=np.float64),
            "mean_reprojection_rmse_px": field_lock[
                "field_marker_optical_summary"].get(
                    "mean_reprojection_rmse_px"),
            "max_reprojection_rmse_px": field_lock[
                "field_marker_optical_summary"].get(
                    "max_reprojection_rmse_px"),
        }
    else:
        field_stereo = estimate_saved_field_pose_stereo(session_dir)
    accepted_poses = [
        pose for pose in marker_poses
        if pose["stereo_reprojection_rmse_px"]
        <= max_stereo_reprojection_rmse_px]
    if len(accepted_poses) < 6:
        raise RuntimeError(
            f"only {len(accepted_poses)} stereo poses pass the "
            f"{max_stereo_reprojection_rmse_px:.3f}px reprojection limit; "
            "at least 6 are required")
    unconstrained_result = solve_roll_invariant_calibration(
        accepted_poses, field_lock["left_camera_T_aurora_m"])
    result = solve_cad_constrained_calibration(
        accepted_poses, field_lock["left_camera_T_aurora_m"])
    cross_validation = leave_one_out_stability(
        accepted_poses, field_lock["left_camera_T_aurora_m"],
        solver=solve_cad_constrained_calibration)
    stereo = [{
        "snapshot_index": pose["snapshot_index"],
        "left_reprojection_rmse_px": pose["left_reprojection_rmse_px"],
        "right_reprojection_rmse_px": pose["right_reprojection_rmse_px"],
        "stereo_reprojection_rmse_px": pose[
            "stereo_reprojection_rmse_px"],
        "rectified_row_mismatch_mean_px": pose[
            "rectified_row_mismatch_mean_px"],
        "rectified_row_mismatch_max_px": pose[
            "rectified_row_mismatch_max_px"],
        "left_camera_T_marker_m": pose["left_camera_T_marker_m"],
    } for pose in marker_poses]
    separation_error = (
        unconstrained_result["coil_separation_mm"] - expected_separation_mm)
    camera_T_aurora = np.asarray(
        result["calibrated_left_camera_T_aurora_m"], dtype=np.float64)
    camera_T_field = np.asarray(
        field_stereo["left_camera_T_field_marker_m"], dtype=np.float64)
    calibrated_aurora_T_field = np.linalg.inv(
        camera_T_aurora) @ camera_T_field
    nominal_aurora_T_field = np.asarray(
        field_lock["aurora_T_field_marker_m"], dtype=np.float64)
    field_correction = (
        calibrated_aurora_T_field @ np.linalg.inv(nominal_aurora_T_field))
    accepted_indices = {
        pose["snapshot_index"] for pose in accepted_poses}
    report = {
        "schema_version": 3,
        "session": str(session_dir.resolve()),
        "snapshot_count": len(marker_poses),
        "accepted_snapshot_indices": [
            pose["snapshot_index"] for pose in accepted_poses],
        "rejected_snapshots": [{
            "snapshot_index": pose["snapshot_index"],
            "reason": "stereo_reprojection_rmse_exceeds_limit",
            "stereo_reprojection_rmse_px": pose[
                "stereo_reprojection_rmse_px"],
        } for pose in marker_poses
            if pose["snapshot_index"] not in accepted_indices],
        "quality_gates": {
            "max_stereo_reprojection_rmse_px": float(
                max_stereo_reprojection_rmse_px),
            "minimum_accepted_poses": 6,
        },
        "transform_convention": "parent_T_child; p_parent = T @ p_child",
        "five_dof_handling": (
            "Aurora quaternion z-axis used; unobservable roll about z ignored"),
        "known_housing_geometry": {
            "expected_coil_axis_marker": [0.0, -1.0, 0.0],
            "expected_coil_separation_mm": float(expected_separation_mm),
            "cad_marker_frame": (
                "corner origin; +z into housing; converted to OpenCV center "
                "origin with +z out of printed front"),
            "coil_tip_positions_opencv_marker_mm": CAD_COIL_TIPS_MARKER_MM,
        },
        "stereo_marker_poses": stereo,
        "field_generator_registration": {
            **field_stereo,
            "nominal_aurora_T_field_marker_m": nominal_aurora_T_field,
            "calibrated_aurora_T_field_marker_m":
                calibrated_aurora_T_field,
            "nominal_to_calibrated_correction_m": field_correction,
            "correction_translation_mm": field_correction[:3, 3] * 1000.0,
            "correction_rotation_deg": float(np.linalg.norm(
                Rotation.from_matrix(
                    field_correction[:3, :3]).as_rotvec())
                * 180.0 / np.pi),
        },
        "calibration": result,
        "unconstrained_geometry_check": unconstrained_result,
        "cross_validation": cross_validation,
        "validation": {
            "unconstrained_separation_error_mm": float(separation_error),
            "unconstrained_absolute_separation_error_mm": float(
                abs(separation_error)),
        },
    }
    return _jsonable(report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--expected-separation-mm", type=float, default=3.8)
    parser.add_argument(
        "--max-stereo-reprojection-rmse-px", type=float, default=3.0)
    parser.add_argument(
        "--output", type=Path,
        help="default: SESSION/tool_calibration_report.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(
        args.session, args.expected_separation_mm,
        args.max_stereo_reprojection_rmse_px)
    output = args.output or args.session / "tool_calibration_report.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    calibration = report["calibration"]
    print(f"Wrote {output}")
    print(
        "Stereo+5DOF fit: position RMS={:.3f} mm, p95={:.3f} mm; "
        "axis RMS={:.3f} deg, p95={:.3f} deg".format(
            calibration["position_residual_mm"]["rms"],
            calibration["position_residual_mm"]["p95"],
            calibration["coil_axis_residual_deg"]["rms"],
            calibration["coil_axis_residual_deg"]["p95"]))
    offsets = calibration["coil_axial_sensing_offsets_mm"]
    unconstrained = report["unconstrained_geometry_check"]
    print(
        "Axial sensing offsets: 003={:.3f} mm, 07222026_01={:.3f} mm; "
        "unconstrained separation={:.3f} mm".format(
            offsets["003"], offsets["07222026_01"],
            unconstrained["coil_separation_mm"]))


if __name__ == "__main__":
    main()
