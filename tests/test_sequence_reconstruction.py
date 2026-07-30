import unittest

import numpy as np

from shape_tracking.segmentation import Centerline
from shape_tracking.sequence_reconstruction import (
    project_camera_points,
    reconstruct_disparity_anchored,
)


class SequenceReconstructionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
