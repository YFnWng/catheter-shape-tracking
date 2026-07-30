import unittest

import numpy as np

from shape_tracking.geometry import (
    OBSERVED_DISTAL,
    TIP_BRIDGE,
    assemble_anchored_shape,
    curve_geometry,
)


class GeometryTests(unittest.TestCase):
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
