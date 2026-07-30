import unittest

import cv2
import numpy as np

from shape_tracking.boards import get_dictionary
from shape_tracking.tool_calibration_capture import (
    TipMarkerDetector,
    build_field_frame_lock,
    build_snapshot_geometry,
    pose_sample,
    summarize_stationary_samples,
)


class ToolCalibrationCaptureTest(unittest.TestCase):
    def test_detects_expected_marker_and_recovers_front_pose(self):
        marker = np.empty((400, 400), dtype=np.uint8)
        cv2.aruco.generateImageMarker(
            get_dictionary(), 33, 400, marker, borderBits=1)
        gray = np.full((600, 600), 255, dtype=np.uint8)
        gray[100:500, 100:500] = marker
        K = np.asarray([
            [800.0, 0.0, 300.0],
            [0.0, 800.0, 300.0],
            [0.0, 0.0, 1.0]])
        pose, _, _, _ = TipMarkerDetector(33, 0.025).detect(
            gray, K, np.zeros(5))
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(float(pose['tvec'][0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(pose['tvec'][1, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(pose['tvec'][2, 0]), 0.05, places=3)
        self.assertLess(pose['reprojection_rmse_px'], 0.1)

    def test_pose_sample_uses_scalar_first_quaternion_and_mm(self):
        sample = pose_sample(
            123, np.zeros((3, 1)), np.asarray([[0.1], [0.2], [0.3]]), 0.2)
        self.assertEqual(sample['timestamp_ns'], 123)
        self.assertEqual(sample['position_mm'], [100.0, 200.0, 300.0])
        self.assertEqual(sample['quaternion_wxyz'], [1.0, 0.0, 0.0, 0.0])

    def test_stationarity_summary_requires_a_complete_time_span(self):
        samples = [{
            'timestamp_ns': index * 50_000_000,
            'position_mm': [1.0, 2.0, 3.0],
            'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
            'reprojection_rmse_px': 0.2,
        } for index in range(11)]
        summary = summarize_stationary_samples(
            samples, 500_000_000, 0.5, 10, 0.5, 1.0)
        self.assertEqual(summary['sample_count'], 11)
        self.assertEqual(summary['position_mm'], [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(RuntimeError, 'history spans'):
            summarize_stationary_samples(
                samples[-3:], 500_000_000, 0.5, 2, 0.5, 1.0)

    def test_low_rate_diagnostic_reports_requested_window_and_modality(self):
        samples = [{
            'timestamp_ns': index * 100_000_000,
            'position_mm': [1.0, 2.0, 3.0],
            'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
        } for index in range(3)]
        with self.assertRaisesRegex(
                RuntimeError,
                r'3 valid stereo marker samples.*1\.000s.*require 5'):
            summarize_stationary_samples(
                samples, 1_000_000_000, 1.0, 5, 0.5, 1.0,
                sample_label='stereo marker')

    def test_field_lock_accepts_stable_two_hz_history(self):
        samples = [{
            'timestamp_ns': index * 500_000_000,
            'position_mm': [100.0 + 0.05 * (index % 2), 20.0, 300.0],
            'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
            'reprojection_rmse_px': None,
        } for index in range(4)]
        summary = summarize_stationary_samples(
            samples, 1_500_000_000, 1.5, 3, 0.5, 1.0)
        self.assertEqual(summary['sample_count'], 4)
        self.assertLess(
            summary['stationarity']['position_p95_deviation_mm'], 0.5)

    def test_composes_coil_to_tip_marker_transform(self):
        optical = {
            'position_mm': [0.0, 0.0, 1000.0],
            'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0]}
        coils = [{
            'part_number': '003',
            'position_aurora_mm': [10.0, 0.0, 0.0],
            'quaternion_aurora_wxyz': [1.0, 0.0, 0.0, 0.0]}, {
            'part_number': '07222026_01',
            'position_aurora_mm': [-10.0, 0.0, 0.0],
            'quaternion_aurora_wxyz': [1.0, 0.0, 0.0, 0.0]}]
        camera_T_aurora = np.eye(4)
        camera_T_aurora[2, 3] = 1.0
        geometry = build_snapshot_geometry(
            optical, coils, camera_T_aurora)
        np.testing.assert_allclose(
            geometry['aurora_T_tip_marker_m'], np.eye(4), atol=1e-12)
        self.assertEqual(
            geometry['coils'][0]['coil_to_marker_translation_mm'],
            [-10.0, 0.0, 0.0])
        self.assertEqual(
            geometry['coils'][1]['coil_to_marker_translation_mm'],
            [10.0, 0.0, 0.0])

    def test_field_lock_composes_camera_to_aurora_once(self):
        field = {
            'position_mm': [100.0, 0.0, 1000.0],
            'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0]}
        config = {'aurora_T_marker': np.eye(4)}
        lock = build_field_frame_lock(field, config, 123)
        self.assertEqual(lock['locked_at_ns'], 123)
        np.testing.assert_allclose(
            np.asarray(lock['left_camera_T_aurora_m'])[:3, 3],
            [0.1, 0.0, 1.0])


if __name__ == '__main__':
    unittest.main()
