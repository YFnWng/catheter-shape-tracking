import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from shape_tracking.stereo_marker import (
    estimate_stereo_object_pose,
    estimate_stereo_marker_pose,
    rectified_right_T_left,
    square_object_points,
)
from shape_tracking.tool_calibration_solve import (
    solve_cad_constrained_calibration,
    solve_roll_invariant_calibration,
)


def project(points, camera_T_marker, K):
    rvec = Rotation.from_matrix(camera_T_marker[:3, :3]).as_rotvec()
    pixels, _ = cv2.projectPoints(
        points, rvec, camera_T_marker[:3, 3], K, np.zeros(5))
    return pixels.reshape(-1, 2)


def rotation_with_z(z_axis, roll):
    z_axis = np.asarray(z_axis, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(np.dot(reference, z_axis)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    x_axis = reference - np.dot(reference, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    base = np.column_stack([x_axis, y_axis, z_axis])
    return base @ Rotation.from_rotvec([0.0, 0.0, roll]).as_matrix()


class StereoMarkerTest(unittest.TestCase):
    def test_recovers_pose_from_both_rectified_cameras(self):
        K = np.asarray([
            [1050.0, 0.0, 960.0],
            [0.0, 1050.0, 540.0],
            [0.0, 0.0, 1.0]])
        object_points = square_object_points(0.025)
        left_T_marker = np.eye(4)
        left_T_marker[:3, :3] = Rotation.from_euler(
            "xyz", [12.0, -20.0, 35.0], degrees=True).as_matrix()
        left_T_marker[:3, 3] = [0.035, -0.015, 0.28]
        right_T_left = rectified_right_T_left(0.1198)
        right_T_marker = right_T_left @ left_T_marker
        left = project(object_points, left_T_marker, K)
        right = project(object_points, right_T_marker, K)
        result = estimate_stereo_marker_pose(
            object_points, left, right, K, np.zeros(5), K, np.zeros(5),
            right_T_left)
        np.testing.assert_allclose(
            result["left_camera_T_marker_m"], left_T_marker, atol=1e-8)
        self.assertLess(result["stereo_reprojection_rmse_px"], 1e-7)

    def test_stereo_object_pose_accepts_different_corner_subsets(self):
        K = np.asarray([
            [900.0, 0.0, 640.0],
            [0.0, 900.0, 360.0],
            [0.0, 0.0, 1.0]])
        points = np.asarray([
            [x, y, 0.0]
            for y in (0.025, 0.05, 0.075)
            for x in (0.025, 0.05, 0.075)])
        left_T_object = np.eye(4)
        left_T_object[:3, :3] = Rotation.from_euler(
            "xyz", [8.0, -14.0, 22.0], degrees=True).as_matrix()
        left_T_object[:3, 3] = [0.02, -0.01, 0.35]
        right_T_left = rectified_right_T_left(0.1198)
        right_T_object = right_T_left @ left_T_object
        left_indices = np.asarray([0, 1, 2, 3, 4, 5, 6])
        right_indices = np.asarray([2, 3, 4, 5, 6, 7, 8])
        result = estimate_stereo_object_pose(
            points[left_indices],
            project(points[left_indices], left_T_object, K),
            points[right_indices],
            project(points[right_indices], right_T_object, K),
            K, np.zeros(5), K, np.zeros(5), right_T_left,
            Rotation.from_matrix(left_T_object[:3, :3]).as_rotvec(),
            left_T_object[:3, 3])
        np.testing.assert_allclose(
            result["left_camera_T_object_m"], left_T_object, atol=1e-8)
        self.assertLess(result["stereo_reprojection_rmse_px"], 1e-7)


class FiveDofCalibrationTest(unittest.TestCase):
    def test_ignores_roll_and_recovers_camera_aurora_transform(self):
        camera_T_aurora = np.eye(4)
        camera_T_aurora[:3, :3] = Rotation.from_euler(
            "xyz", [4.0, -7.0, 13.0], degrees=True).as_matrix()
        camera_T_aurora[:3, 3] = [0.06, -0.02, 0.31]
        centers = {
            "003": np.asarray([1.9, 4.2, -3.0]),
            "07222026_01": np.asarray([-1.9, 4.6, -3.0]),
        }
        poses = []
        for index in range(8):
            camera_T_marker = np.eye(4)
            camera_T_marker[:3, :3] = Rotation.from_euler(
                "xyz", [5 * index - 12, 9 * index - 25, 17 * index],
                degrees=True).as_matrix()
            camera_T_marker[:3, 3] = [
                0.03 + 0.006 * index,
                -0.025 + 0.004 * (index % 3),
                0.22 + 0.01 * (index % 2)]
            coils = []
            for coil_index, (part, center) in enumerate(centers.items()):
                point_camera_mm = (
                    camera_T_marker[:3, :3] @ center
                    + camera_T_marker[:3, 3] * 1000.0)
                point_aurora_m = (
                    camera_T_aurora[:3, :3].T @ (
                        point_camera_mm * 0.001
                        - camera_T_aurora[:3, 3]))
                axis_camera = camera_T_marker[:3, :3] @ [0.0, -1.0, 0.0]
                axis_aurora = (
                    camera_T_aurora[:3, :3].T @ axis_camera)
                aurora_T_coil = np.eye(4)
                aurora_T_coil[:3, :3] = rotation_with_z(
                    axis_aurora, roll=0.7 * index + 1.1 * coil_index)
                aurora_T_coil[:3, 3] = point_aurora_m
                coils.append({
                    "part_number": part,
                    "position_aurora_mm": (point_aurora_m * 1000.0).tolist(),
                    "aurora_T_coil_m": aurora_T_coil.tolist(),
                })
            poses.append({
                "snapshot_index": index + 1,
                "left_camera_T_marker_m": camera_T_marker,
                "snapshot": {"snapshot_index": index + 1, "coils": coils},
            })
        nominal = camera_T_aurora.copy()
        nominal[:3, :3] = (
            Rotation.from_euler("z", 2.0, degrees=True).as_matrix()
            @ nominal[:3, :3])
        nominal[:3, 3] += [0.002, -0.001, 0.003]
        result = solve_roll_invariant_calibration(poses, nominal)
        np.testing.assert_allclose(
            result["calibrated_left_camera_T_aurora_m"],
            camera_T_aurora, atol=1e-7)
        self.assertAlmostEqual(
            result["coil_separation_mm"], np.hypot(3.8, 0.4), places=6)
        self.assertLess(result["position_residual_mm"]["rms"], 1e-6)
        self.assertLess(result["coil_axis_residual_deg"]["rms"], 1e-6)

        cad_result = solve_cad_constrained_calibration(poses, nominal)
        np.testing.assert_allclose(
            cad_result["calibrated_left_camera_T_aurora_m"],
            camera_T_aurora, atol=1e-7)
        self.assertAlmostEqual(
            cad_result["coil_axial_sensing_offsets_mm"]["003"],
            -4.2, places=6)
        self.assertAlmostEqual(
            cad_result["coil_axial_sensing_offsets_mm"]["07222026_01"],
            -4.6, places=6)
        self.assertLess(cad_result["position_residual_mm"]["rms"], 1e-6)


if __name__ == "__main__":
    unittest.main()
