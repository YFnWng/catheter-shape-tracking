import unittest

import numpy as np

from shape_tracking.segmentation import Centerline
from shape_tracking.reconstruction import smooth_polyline_2d
from shape_tracking.sequence_reconstruction import (
    project_camera_points,
    regularize_disparity_local,
    reconstruct_disparity_anchored,
)


class SequenceReconstructionTests(unittest.TestCase):
    def test_default_2d_resampling_preserves_piecewise_linear_corner(self):
        points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        sampled = smooth_polyline_2d(points, 21)
        self.assertTrue(np.all(
            np.isclose(sampled[:, 1], 0.0)
            | np.isclose(sampled[:, 0], 10.0)))
        np.testing.assert_allclose(sampled[10], [10.0, 0.0])

    def test_local_disparity_regularizer_rejects_isolated_outlier(self):
        expected = np.linspace(25.0, 35.0, 80)
        raw = expected.copy()
        raw[37] = 80.0
        fitted, weights = regularize_disparity_local(
            raw, np.ones(80, dtype=bool),
            first_difference_weight=0.25,
            second_difference_weight=10.0,
            huber_delta_px=1.5)
        self.assertLess(abs(fitted[37] - expected[37]), 2.0)
        self.assertLess(weights[37], 0.5)
        self.assertLess(np.max(np.abs(fitted[:30] - expected[:30])), 0.2)

    def test_constant_depth_curve(self):
        K = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        baseline = 0.12
        parameter = np.linspace(0.0, 1.0, 100)
        points = np.column_stack([
            -0.04 + 0.08 * parameter,
            -0.04 + 0.08 * parameter**1.3,
            np.full_like(parameter, 0.5),
        ])
        left = project_camera_points(points, K, baseline, right=False)
        right = project_camera_points(points, K, baseline, right=True)
        dummy_mask = np.ones((10, 10), dtype=np.uint8)
        left_centerline = Centerline(left, (0, 0, 10, 10), dummy_mask, 1.0)
        right_centerline = Centerline(right, (0, 0, 10, 10), dummy_mask, 1.0)
        result = reconstruct_disparity_anchored(
            left_centerline, right_centerline, K, baseline,
            points[0], points[-1], n_samples=80, disparity_order=2)
        np.testing.assert_allclose(
            result["points_camera_m"][:, 2], 0.5, atol=2e-3)
        self.assertLess(result["reprojection_left_px"], 1.0)
        self.assertLess(result["reprojection_right_px"], 1.0)

    def test_right_reference_reconstructs_in_left_camera_coordinates(self):
        K = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        baseline = 0.12
        parameter = np.linspace(0.0, 1.0, 100)
        points = np.column_stack([
            0.02 * np.sin(parameter * np.pi),
            -0.05 + 0.10 * parameter,
            0.45 + 0.10 * parameter,
        ])
        left = project_camera_points(points, K, baseline, right=False)
        right = project_camera_points(points, K, baseline, right=True)
        mask = np.ones((10, 10), dtype=np.uint8)
        result = reconstruct_disparity_anchored(
            Centerline(left, (0, 0, 10, 10), mask, 1.0),
            Centerline(right, (0, 0, 10, 10), mask, 1.0),
            K, baseline, points[0], None, n_samples=80,
            reference_view="right")
        self.assertEqual(result["reference_view"], "right")
        self.assertLess(result["reprojection_left_px"], 1.0)
        self.assertLess(result["reprojection_right_px"], 1.0)
        self.assertGreater(np.mean(result["points_camera_m"][:, 2]), 0.45)


if __name__ == "__main__":
    unittest.main()
