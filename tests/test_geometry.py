import unittest

import numpy as np

from shape_tracking.geometry import (
    OBSERVED_DISTAL,
    TIP_BRIDGE,
    assemble_image_only_shape,
    assemble_anchored_shape,
    curve_geometry,
)


class GeometryTests(unittest.TestCase):
    def test_image_only_shape_has_observed_tip_and_distal_base(self):
        visible = np.column_stack([
            np.zeros(20), np.zeros(20), np.linspace(3.0, 100.0, 20)])
        shape = assemble_image_only_shape(
            visible, distal_boundary_fraction=0.6,
            full_count=32, distal_count=16)
        np.testing.assert_allclose(shape.full_points_mm[0], np.zeros(3))
        np.testing.assert_allclose(shape.full_points_mm[-1], visible[-1])
        self.assertEqual(shape.tip_bridge_length_mm, 0.0)
        self.assertGreater(shape.distal_boundary_s_mm, 50.0)
        self.assertTrue(np.all(
            shape.distal_observation_class == OBSERVED_DISTAL))

    def test_image_only_shape_can_omit_base_bridge(self):
        visible = np.column_stack([
            np.full(20, 4.0), np.zeros(20), np.linspace(3.0, 100.0, 20)])
        shape = assemble_image_only_shape(
            visible, distal_boundary_fraction=0.6,
            full_count=32, distal_count=16, bridge_base=False)
        np.testing.assert_allclose(shape.full_points_mm[0], visible[0])
        self.assertEqual(shape.base_bridge_length_mm, 0.0)

    def test_image_only_shape_enforces_exact_distal_length(self):
        visible = np.column_stack([
            np.zeros(101), np.zeros(101), np.linspace(0.0, 100.0, 101)])
        shape = assemble_image_only_shape(
            visible, distal_boundary_fraction=0.9,
            full_count=32, distal_count=64, bridge_base=False,
            distal_length_mm=60.0)
        self.assertAlmostEqual(shape.distal_boundary_s_mm, 40.0, places=8)
        self.assertAlmostEqual(shape.distal_s_mm[-1], 60.0, places=8)
        np.testing.assert_allclose(shape.distal_points_mm[0], [0.0, 0.0, 40.0])

    def test_image_only_shape_rejects_curve_shorter_than_distal(self):
        visible = np.column_stack([
            np.zeros(41), np.zeros(41), np.linspace(0.0, 40.0, 41)])
        with self.assertRaisesRegex(ValueError, "visible_curve_shorter_than_distal"):
            assemble_image_only_shape(
                visible, distal_boundary_fraction=0.5,
                bridge_base=False, distal_length_mm=60.0)

    def test_circle_curvature(self):
        radius = 50.0
        angle = np.linspace(0.0, np.pi / 2.0, 100)
        points = np.column_stack([
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.zeros_like(angle),
        ])
        geometry = curve_geometry(points, smoothing_mm=0.0)
        middle = geometry.curvature_per_mm[10:-10]
        self.assertAlmostEqual(float(np.median(middle)), 1.0 / radius,
                               delta=5e-4)
        np.testing.assert_allclose(geometry.points_mm[0], points[0])
        np.testing.assert_allclose(geometry.points_mm[-1], points[-1])
        self.assertTrue(np.isnan(geometry.curvature_per_mm[0]))
        self.assertTrue(np.isnan(geometry.curvature_per_mm[-1]))
        self.assertEqual(geometry.spline_degree, 3)
        self.assertEqual(geometry.spline_basis_count, 12)
        self.assertEqual(geometry.spline_internal_knot_count, 8)

    def test_curvature_spline_never_collapses_to_global_cubic(self):
        first = np.column_stack([
            np.linspace(0.0, 40.0, 40), np.zeros(40), np.zeros(40)])
        second = np.column_stack([
            np.full(40, 40.0), np.linspace(1.0, 40.0, 40), np.zeros(40)])
        points = np.vstack([first, second])
        geometry = curve_geometry(points, smoothing_mm=1.0, basis_count=16)
        self.assertEqual(geometry.spline_basis_count, 16)
        self.assertEqual(geometry.spline_internal_knot_count, 12)
        self.assertLess(geometry.spline_rms_residual_mm, 1.0)

    def test_curvature_spline_can_enforce_saved_arc_length(self):
        first = np.column_stack([
            np.linspace(0.0, 30.0, 40), np.zeros(40), np.zeros(40)])
        second = np.column_stack([
            np.full(40, 30.0), np.linspace(1.0, 31.0, 40), np.zeros(40)])
        points = np.vstack([first, second])
        target = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        geometry = curve_geometry(
            points, smoothing_mm=1.0, basis_count=12,
            target_arc_length_mm=target)
        self.assertAlmostEqual(geometry.spline_arc_length_mm, target, places=3)

    def test_arc_constraint_handles_observed_segment_with_different_length(self):
        first = np.column_stack([
            np.linspace(0.0, 35.0, 40), np.zeros(40), np.zeros(40)])
        second = np.column_stack([
            np.full(40, 35.0), np.linspace(1.0, 36.0, 40), np.zeros(40)])
        points = np.vstack([first, second])
        geometry = curve_geometry(
            points, smoothing_mm=1.0, basis_count=12,
            target_arc_length_mm=60.0)
        self.assertAlmostEqual(geometry.spline_arc_length_mm, 60.0, places=3)
        np.testing.assert_allclose(geometry.points_mm[0], points[0])
        np.testing.assert_allclose(geometry.points_mm[-1], points[-1])

    def test_assembled_shape_has_exact_base_and_tip(self):
        z = np.linspace(2.0, 80.0, 50)
        visible = np.column_stack([
            2.0 * np.sin(z / 30.0), np.zeros_like(z), z])
        tip = np.array([1.0, 0.0, 100.0])
        shape = assemble_anchored_shape(
            visible, distal_boundary_fraction=0.6,
            tip_position_base_mm=tip,
            tip_z_direction_base=np.array([0.0, 0.0, 1.0]),
            full_count=128, distal_count=64)
        np.testing.assert_allclose(shape.full_points_mm[0], np.zeros(3),
                                   atol=1e-9)
        np.testing.assert_allclose(shape.full_points_mm[-1], tip, atol=1e-9)
        np.testing.assert_allclose(shape.distal_points_mm[-1], tip, atol=1e-9)
        self.assertIn(OBSERVED_DISTAL, shape.full_observation_class)
        self.assertEqual(shape.full_observation_class[-1], TIP_BRIDGE)


if __name__ == "__main__":
    unittest.main()
