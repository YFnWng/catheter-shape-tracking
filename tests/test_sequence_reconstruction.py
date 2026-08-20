import unittest

import numpy as np

from shape_tracking.segmentation import Centerline
from shape_tracking.reconstruction import smooth_polyline_2d
from shape_tracking.sequence_reconstruction import (
    project_camera_points,
    regularize_disparity_local,
    reconstruct_disparity_anchored,
    select_overlap_aware_disparity_path,
)


class SequenceReconstructionTests(unittest.TestCase):
    def test_overlap_path_uses_disparity_not_other_eye_arc_order(self):
        candidates = [np.array([20.0, 40.0]) for _ in range(12)]
        costs = [np.zeros(2) for _ in candidates]
        selected, observed, ambiguity = select_overlap_aware_disparity_path(
            candidates, costs,
            disparity_prior_px=np.full(12, 40.0),
            disparity_prior_weight=2.0)
        np.testing.assert_allclose(selected, 40.0)
        self.assertTrue(np.all(observed))
        self.assertEqual(ambiguity, 1.0)

    def test_overlap_aware_reconstruction_uses_unordered_mask_branches(self):
        K = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        baseline = 0.12
        y = np.linspace(140.0, 260.0, 100)
        left_points = np.column_stack([np.full(100, 300.0), y])
        # The bad-eye projection contains two branches on every epipolar row.
        # Its ordered path traverses one branch and returns along the other.
        right_points = np.vstack([
            np.column_stack([np.full(100, 200.0), y]),
            np.column_stack([np.full(100, 205.0), y[::-1]])])
        left_mask = np.zeros((480, 640), np.uint8)
        right_mask = np.zeros_like(left_mask)
        left_mask[138:263, 298:303] = 255
        right_mask[138:263, 198:203] = 255
        right_mask[138:263, 203:208] = 255
        left = Centerline(left_points, (0, 0, 640, 480), left_mask, 2.0)
        right = Centerline(right_points, (0, 0, 640, 480), right_mask, 2.0)
        base = np.array([-0.024, -0.12, 0.6])
        tip = np.array([-0.024, 0.024, 0.6])
        result = reconstruct_disparity_anchored(
            left, right, K, baseline, base, tip, n_samples=80,
            reference_view="left", overlap_aware=True,
            overlap_self_fraction_threshold=0.05)
        self.assertEqual(result["overlap_aware_used"], 1)
        self.assertEqual(result["correspondence_model"],
                         "unordered_mask_overlap_aware")
        np.testing.assert_allclose(
            result["points_camera_m"][:, 2], 0.6, atol=0.025)

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

    def test_terminal_refinement_tracks_confident_distal_disparity_bend(self):
        K = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        baseline = 0.12
        parameter = np.linspace(0.0, 1.0, 120)
        terminal = np.clip((parameter - 0.78) / 0.22, 0.0, 1.0)
        disparity = 95.0 + 25.0 * terminal**2
        left_pixels = np.column_stack([
            350.0 + 12.0 * parameter,
            140.0 + 180.0 * parameter])
        right_pixels = left_pixels.copy()
        right_pixels[:, 0] -= disparity
        depth = K[0, 0] * baseline / disparity
        points = np.column_stack([
            (left_pixels[:, 0] - K[0, 2]) * depth / K[0, 0],
            (left_pixels[:, 1] - K[1, 2]) * depth / K[1, 1],
            depth])
        mask = np.zeros((480, 640), dtype=np.uint8)
        left = Centerline(left_pixels, (0, 0, 640, 480), mask, 1.0)
        right = Centerline(right_pixels, (0, 0, 640, 480), mask, 1.0)
        common = dict(
            n_samples=96, reference_view="right",
            disparity_prior_px=np.full(96, 95.0),
            disparity_prior_weight=2.0)
        original = reconstruct_disparity_anchored(
            left, right, K, baseline, points[0], points[-1],
            terminal_refinement=False, **common)
        refined = reconstruct_disparity_anchored(
            left, right, K, baseline, points[0], points[-1],
            terminal_refinement=True, **common)
        self.assertEqual(refined["terminal_refinement_used"], 1)
        self.assertGreater(
            refined["terminal_refinement_improvement_px"], 0.25)
        self.assertLess(
            refined["terminal_reprojection_left_px"],
            original["terminal_reprojection_left_px"])
        self.assertLessEqual(
            refined["reprojection_p95_px"], original["reprojection_p95_px"])

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
        self.assertEqual(result["centerline_tip_anchor_used"], 1)

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
