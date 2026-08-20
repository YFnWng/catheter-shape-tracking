import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from shape_tracking.interface_smoothing import (
    robust_zero_phase_smooth,
    smooth_interface_hdf5,
)
from shape_tracking.stereo_smoothing import smooth_stereo_disparity_hdf5


class InterfaceSmoothingTests(unittest.TestCase):
    def test_offline_disparity_smoothing_reduces_frame_alternation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.h5"
            count, stereo_count, full_count = 30, 32, 64
            timestamps = np.arange(count, dtype=np.int64) * 33_333_333
            y = np.linspace(10.0, 90.0, stereo_count)
            disparity = 10.0 + 0.8 * ((-1.0) ** np.arange(count))[:, None]
            disparity = np.repeat(disparity, stereo_count, axis=1)
            left = np.repeat(
                np.column_stack([np.full(stereo_count, 50.0), y])[None],
                count, axis=0)
            right = left.copy()
            right[:, :, 0] -= disparity
            depth = 10.0 / disparity
            camera = np.stack([
                (left[:, :, 0] - 50.0) * depth / 100.0,
                (left[:, :, 1] - 50.0) * depth / 100.0,
                depth], axis=2)
            with h5py.File(path, "w") as output:
                output["frames/valid"] = np.ones(count, np.uint8)
                output["frames/timestamp_ns"] = timestamps
                output["stereo/fitted_disparity_px"] = disparity
                output["stereo/smoothed_disparity_px"] = np.full_like(
                    disparity, np.nan)
                output["stereo/ordered_left_px"] = left
                output["stereo/ordered_right_px"] = right
                output["stereo/visible_points_base_mm"] = camera * 1000.0
                output["stereo/causal_visible_points_base_mm"] = camera * 1000.0
                output["quality/stereo_reference_view"] = np.ones(count)
                output["quality/reprojection_p95_px"] = np.ones(count)
                output["quality/stereo_condition"] = np.ones(count)
                for name in (
                        "visible_arc_length_mm", "full_spline_basis_count",
                        "full_spline_internal_knot_count",
                        "full_spline_rms_residual_mm", "reprojection_left_px",
                        "reprojection_right_px", "reprojection_max_px"):
                    output[f"quality/{name}"] = np.full(count, np.nan)
                for view, pixels in (("left", left), ("right", right)):
                    output[f"images/{view}/centerline_px"] = pixels
                output["full/points_base_mm"] = np.zeros(
                    (count, full_count, 3), np.float32)
                output["full/s_mm"] = np.zeros((count, full_count), np.float32)
                output["full/tangent_base"] = np.zeros(
                    (count, full_count, 3), np.float32)
                output["full/curvature_per_mm"] = np.zeros(
                    (count, full_count), np.float32)
            registration = SimpleNamespace(
                K=np.array([[100.0, 0.0, 50.0],
                            [0.0, 100.0, 50.0],
                            [0.0, 0.0, 1.0]]),
                baseline_m=0.1,
                left_camera_T_base=np.eye(4))
            summary = smooth_stereo_disparity_hdf5(
                path, registration, cutoff_hz=2.0)
            self.assertEqual(summary["updated_frames"], count)
            with h5py.File(path, "r") as output:
                smoothed = output["stereo/smoothed_disparity_px"][:, 10]
                self.assertLess(np.std(smoothed), 0.25)
                np.testing.assert_allclose(
                    output["stereo/smoothed_disparity_px"][:, 0],
                    disparity[:, 0])
                self.assertTrue(np.all(np.isfinite(
                    output["full/points_base_mm"][:])))

    def test_robust_zero_phase_filter_reduces_outlier_without_phase_shift(self):
        count = 301
        timestamps = np.arange(count, dtype=np.int64) * 33_333_333
        center = count // 2
        values = 60.0 + 2.0 * np.exp(-((np.arange(count) - center) / 20.0) ** 2)
        noisy = values.copy()
        noisy[center + 30] += 12.0
        smoothed = robust_zero_phase_smooth(
            noisy, timestamps, cutoff_hz=2.0, huber_delta=1.5)
        self.assertLess(abs(smoothed[center + 30] - values[center + 30]), 2.0)
        self.assertLess(abs(int(np.argmax(smoothed)) - center), 3)

    def test_hdf_smoother_preserves_causal_interface_and_rebuilds_distal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shapes.h5"
            count = 20
            full_count = 128
            distal_count = 64
            x = np.linspace(0.0, 100.0, full_count)
            full = np.column_stack([x, 0.002 * x**2, np.zeros_like(x)])
            with h5py.File(path, "w") as output:
                output["frames/valid"] = np.ones(count, np.uint8)
                output["frames/timestamp_ns"] = (
                    np.arange(count, dtype=np.int64) * 33_333_333)
                output["full/points_base_mm"] = np.repeat(
                    full[None], count, axis=0)
                output["full/observation_class"] = np.zeros(
                    (count, full_count), np.uint8)
                output["distal/points_base_mm"] = np.zeros(
                    (count, distal_count, 3), np.float32)
                output["distal/s_mm"] = np.zeros((count, distal_count), np.float32)
                output["distal/tangent_base"] = np.zeros(
                    (count, distal_count, 3), np.float32)
                output["distal/curvature_per_mm"] = np.zeros(
                    (count, distal_count), np.float32)
                output["distal/observation_class"] = np.zeros(
                    (count, distal_count), np.uint8)
                causal_base = np.column_stack([
                    np.linspace(35.0, 45.0, count), np.zeros(count), np.zeros(count)])
                output["distal/base_position_base_mm"] = causal_base
                causal_length = np.full(count, 60.0)
                causal_length[6:14] = 45.0
                output["quality/distal_length_filtered_mm"] = causal_length
                output["quality/interface_uncertainty_mm"] = np.ones(count)
                output["quality/material_boundary_confidence"] = np.ones(count)
                output["quality/material_boundary_stereo_consistent"] = (
                    np.ones(count, np.uint8))
                output["quality/reprojection_p95_px"] = np.ones(count)
                for name in (
                        "distal_boundary_s_mm", "material_boundary_fraction",
                        "distal_spline_basis_count",
                        "distal_spline_internal_knot_count",
                        "distal_spline_rms_residual_mm",
                        "distal_spline_arc_length_mm"):
                    output[f"quality/{name}"] = np.full(count, np.nan)
            summary = smooth_interface_hdf5(path, cutoff_hz=2.0)
            self.assertEqual(summary["updated_frames"], count)
            self.assertEqual(summary["length_gate_rejected_frames"], 8)
            with h5py.File(path, "r") as output:
                np.testing.assert_allclose(
                    output["distal/causal_base_position_base_mm"][:], causal_base)
                smoothed = output["quality/distal_length_smoothed_mm"][:]
                self.assertGreater(smoothed[10], 59.0)
                np.testing.assert_allclose(
                    output["distal/base_position_base_mm"][:, 1],
                    0.002 * output["distal/base_position_base_mm"][:, 0] ** 2,
                    atol=0.1)


if __name__ == "__main__":
    unittest.main()
