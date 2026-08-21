import unittest

import numpy as np

from shape_tracking.geometry import cumulative_arclength
from shape_tracking.joint_spline_reconstruction import (
    _basis,
    _project_spline_with_jacobian,
    distal_image_observation,
    fit_joint_two_view_spline,
    project_point_to_polyline,
)
from shape_tracking.session import project_points


class JointSplineReconstructionTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([
            [700.0, 0.0, 640.0],
            [0.0, 700.0, 360.0],
            [0.0, 0.0, 1.0],
        ])
        self.left_T_base = np.eye(4)
        self.left_T_base[2, 3] = 0.40
        self.right_T_base = self.left_T_base.copy()
        self.right_T_base[0, 3] = -0.05

    def test_marker_centroid_is_projected_to_centerline(self):
        curve = np.column_stack([np.linspace(10, 110, 101), np.full(101, 20.0)])
        projected, fraction = project_point_to_polyline(curve, [43.0, 28.0])
        np.testing.assert_allclose(projected, [43.0, 20.0], atol=1e-9)
        self.assertAlmostEqual(fraction, 0.33, places=2)
        observed, axial = distal_image_observation(
            curve, np.array([30.0, 27.0]), np.array([90.0, 14.0]), count=25)
        np.testing.assert_allclose(observed[0], [30.0, 20.0], atol=1e-9)
        np.testing.assert_allclose(observed[-1], [90.0, 20.0], atol=1e-9)
        np.testing.assert_allclose(axial[:, 1], 20.0, atol=1e-9)

    def test_joint_fit_reduces_two_view_error(self):
        u = np.linspace(0.0, 1.0, 96)
        truth = np.column_stack([
            10.0 * np.sin(np.pi * u),
            48.0 * u,
            7.0 * np.sin(0.5 * np.pi * u),
        ])
        truth *= 57.0 / cumulative_arclength(truth)[-1]
        left = project_points(self.K, self.left_T_base, truth)[0]
        right = project_points(self.K, self.right_T_base, truth)[0]
        initial = truth.copy()
        bump = np.sin(np.pi * u) ** 2
        initial[:, 0] += 3.0 * bump
        initial[:, 2] -= 4.0 * bump
        result = fit_joint_two_view_spline(
            initial, left, right, self.K,
            self.left_T_base, self.right_T_base,
            nominal_length_mm=57.0,
            output_samples=64, basis_count=12, fit_samples=48,
            coverage_samples=48, max_nfev=30)
        self.assertTrue(np.all(np.isfinite(result.points_base_mm)))
        self.assertLess(
            result.final_symmetric_mean_px,
            0.6 * result.initial_symmetric_mean_px)
        self.assertLess(result.left_model_p95_px, 1.5)
        self.assertLess(result.right_model_p95_px, 1.5)
        self.assertLess(abs(result.length_residual_mm), 2.0)

    def test_analytic_projection_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(4)
        design = _basis(17, 9)
        coefficients = rng.normal(size=(9, 3))
        coefficients[:, 2] *= 0.2
        points = design @ coefficients
        pixels, analytic = _project_spline_with_jacobian(
            self.K, self.left_T_base, points, design)
        numeric = np.empty_like(analytic.reshape(-1, analytic.shape[-1]))
        epsilon = 1e-5
        flat = coefficients.ravel()
        for column in range(len(flat)):
            perturbed = flat.copy()
            perturbed[column] += epsilon
            changed = design @ perturbed.reshape(coefficients.shape)
            changed_pixels = project_points(
                self.K, self.left_T_base, changed)[0]
            numeric[:, column] = (
                (changed_pixels - pixels) / epsilon).ravel()
        np.testing.assert_allclose(
            analytic.reshape(numeric.shape), numeric,
            rtol=2e-5, atol=2e-6)

    def test_analytic_and_numeric_joint_fits_agree(self):
        u = np.linspace(0.0, 1.0, 64)
        truth = np.column_stack([
            8.0 * np.sin(np.pi * u),
            50.0 * u,
            5.0 * np.sin(0.7 * np.pi * u),
        ])
        truth *= 57.0 / cumulative_arclength(truth)[-1]
        left = project_points(self.K, self.left_T_base, truth)[0]
        right = project_points(self.K, self.right_T_base, truth)[0]
        initial = truth.copy()
        initial[:, 0] += 1.5 * np.sin(np.pi * u) ** 2
        arguments = (
            initial, left, right, self.K,
            self.left_T_base, self.right_T_base, 57.0)
        analytic = fit_joint_two_view_spline(
            *arguments, basis_count=10, fit_samples=40,
            coverage_samples=32, max_nfev=20,
            analytic_jacobian=True)
        numeric = fit_joint_two_view_spline(
            *arguments, basis_count=10, fit_samples=40,
            coverage_samples=32, max_nfev=20,
            analytic_jacobian=False)
        np.testing.assert_allclose(
            analytic.points_base_mm, numeric.points_base_mm,
            rtol=2e-4, atol=2e-3)
        self.assertAlmostEqual(
            analytic.final_symmetric_mean_px,
            numeric.final_symmetric_mean_px, places=4)


if __name__ == "__main__":
    unittest.main()
