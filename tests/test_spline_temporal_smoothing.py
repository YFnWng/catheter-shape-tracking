import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from shape_tracking.spline_temporal_smoothing import (
    LEARNING_REJECT_MASK_WIDTH,
    LEARNING_REJECT_SHAPE_OUTLIER,
    LEARNING_REJECT_TERMINAL_OUTLIER,
    _bridge_short_point_outlier_runs,
    _confirmed_support_runs,
    _local_motion_prediction,
    _long_rejected_runs,
    interpolate_short_spline_gaps_hdf5,
    smooth_distal_spline_coefficients_hdf5,
)


class SplineTemporalSmoothingTests(unittest.TestCase):
    def test_short_gap_interpolates_complete_common_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gap.h5"
            count, samples, bases = 9, 16, 8
            coefficients = np.zeros((count, bases, 3), dtype=np.float32)
            coefficients[:, :, 0] = np.arange(bases)[None]
            coefficients[:, :, 1] = np.arange(count)[:, None]
            coefficients[3:6] = np.nan
            with h5py.File(path, "w") as output:
                output["frames/timestamp_ns"] = (
                    np.arange(count, dtype=np.int64) * 33_333_333)
                output["frames/valid"] = np.array(
                    [1, 1, 1, 0, 0, 0, 1, 1, 1], np.uint8)
                output["frames/learning_valid"] = np.array(
                    [1, 1, 1, 0, 0, 0, 1, 1, 1], np.uint8)
                output["frames/learning_rejection_flags"] = np.array(
                    [0, 0, 0, 1, 1, 1, 0, 0, 0], np.uint16)
                output["distal/filtered_spline_coefficients_base_mm"] = (
                    coefficients)
                output["distal/points_base_mm"] = np.full(
                    (count, samples, 3), np.nan, np.float32)
                output["distal/s_mm"] = np.full(
                    (count, samples), np.nan, np.float32)
                output["distal/tangent_base"] = np.full(
                    (count, samples, 3), np.nan, np.float32)
                output["distal/curvature_per_mm"] = np.full(
                    (count, samples), np.nan, np.float32)
                output["distal/base_position_base_mm"] = np.full(
                    (count, 3), np.nan, np.float32)
                for name in (
                        "distal_spline_arc_length_mm",
                        "distal_spline_basis_count",
                        "distal_spline_internal_knot_count"):
                    output[f"quality/{name}"] = np.full(count, np.nan)
                output.attrs["distal_temporal_spline_degree"] = 3
            summary = interpolate_short_spline_gaps_hdf5(
                path, maximum_gap_ms=500.0)
            self.assertEqual(summary["interpolated_frame_count"], 3)
            self.assertEqual(summary["interpolated_runs"], [[3, 5]])
            with h5py.File(path, "r") as output:
                repaired = output[
                    "distal/filtered_spline_coefficients_base_mm"][:]
                self.assertTrue(np.all(np.isfinite(repaired[3:6])))
                np.testing.assert_allclose(
                    repaired[3:6, :, 1],
                    np.repeat(
                        np.arange(3, 6, dtype=float)[:, None], bases, axis=1),
                    atol=1e-5)
                np.testing.assert_array_equal(
                    output["frames/curve_temporally_interpolated"][:],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0])
                np.testing.assert_array_equal(
                    output["frames/pre_interpolation_valid"][:],
                    [1, 1, 1, 0, 0, 0, 1, 1, 1])
                self.assertTrue(np.all(output["frames/valid"][3:6]))
                self.assertTrue(np.all(np.isfinite(
                    output["distal/points_base_mm"][3:6])))

    def test_outlier_edge_bridging_does_not_chain_alternating_modes(self):
        timestamps = np.arange(20, dtype=np.int64) * 33_333_333
        outlier = np.zeros((20, 1), bool)
        # Three close enter/return edge pairs. The old overlapping-pair loop
        # filled the whole interval from frame 2 through frame 14.
        outlier[[2, 5, 8, 11, 14], 0] = True
        bridged = _bridge_short_point_outlier_runs(
            outlier, np.ones(20, bool), timestamps, 500.0)
        self.assertTrue(np.all(bridged[2:6, 0]))
        self.assertTrue(np.all(bridged[8:12, 0]))
        self.assertTrue(bridged[14, 0])
        self.assertFalse(bridged[6, 0])
        self.assertFalse(bridged[12, 0])

    def test_local_motion_prediction_allows_fast_sustained_motion(self):
        count = 40
        timestamps = np.arange(count, dtype=np.int64) * 33_333_333
        observed = np.zeros((count, 3, 1), dtype=float)
        observed[:, :, 0] = np.arange(count)[:, None] * 2.5
        predicted = _local_motion_prediction(
            observed, timestamps, np.ones(count, bool), 500.0)
        np.testing.assert_allclose(predicted, observed, atol=1e-9)
        observed[20:23] += 8.0
        predicted = _local_motion_prediction(
            observed, timestamps, np.ones(count, bool), 500.0)
        self.assertGreater(np.max(np.abs(
            observed[19:24] - predicted[19:24])), 3.0)

    def test_reacquisition_requires_persistent_support(self):
        timestamps = np.arange(12, dtype=np.int64) * 33_333_333
        supported = np.zeros(12, bool)
        supported[2] = True
        supported[6:9] = True
        confirmed = _confirmed_support_runs(
            supported, np.ones(12, bool), timestamps, 500.0,
            minimum_frames=3)
        self.assertFalse(confirmed[2])
        self.assertTrue(np.all(confirmed[6:9]))

    def test_common_basis_filter_rejects_frame_and_local_tip_outliers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shapes.h5"
            count, samples = 90, 64
            u = np.linspace(0.0, 1.0, samples)
            truth = np.empty((count, samples, 3), dtype=np.float64)
            for frame in range(count):
                phase = 0.012 * frame
                truth[frame, :, 0] = 60.0 * u
                truth[frame, :, 1] = 9.0 * np.sin(np.pi * u + phase)
                truth[frame, :, 2] = 3.0 * np.sin(2.0 * np.pi * u) + 0.01 * frame
            observed = truth.copy()
            observed += 0.15 * ((-1.0) ** np.arange(count))[:, None, None]
            observed[45] += np.array([0.0, 7.0, 0.0])
            observed[65:70, u >= 0.8, 1] += 6.0

            with h5py.File(path, "w") as output:
                output["frames/valid"] = np.ones(count, np.uint8)
                output["frames/timestamp_ns"] = (
                    np.arange(count, dtype=np.int64) * 33_333_333)
                output["distal/points_base_mm"] = observed.astype(np.float32)
                output["distal/s_mm"] = np.zeros((count, samples), np.float32)
                output["distal/tangent_base"] = np.zeros(
                    (count, samples, 3), np.float32)
                output["distal/curvature_per_mm"] = np.zeros(
                    (count, samples), np.float32)
                output["distal/base_position_base_mm"] = np.zeros(
                    (count, 3), np.float32)
                output["quality/reprojection_p95_px"] = np.ones(count)
                output["quality/fitted_reprojection_p95_px"] = np.ones(count)
                output["quality/joint_final_symmetric_mean_px"] = np.ones(count)
                # Joint reconstruction remains usable when independent-ray
                # triangulation is ill-conditioned.
                output["quality/stereo_condition"] = np.full(count, 0.05)
                output["quality/mask_effective_width_left_px"] = np.full(
                    count, 14.0)
                output["quality/mask_effective_width_right_px"] = np.full(
                    count, 14.0)
                output["quality/mask_effective_width_left_px"][80] = 24.0
                for name in (
                        "distal_spline_arc_length_mm",
                        "distal_spline_basis_count",
                        "distal_spline_internal_knot_count",
                        "distal_spline_rms_residual_mm"):
                    output[f"quality/{name}"] = np.full(count, np.nan)

            summary = smooth_distal_spline_coefficients_hdf5(
                path, cutoff_hz=3.0, outlier_floor_mm=0.5,
                outlier_sigma=3.5)
            self.assertEqual(summary["updated_frames"], count)
            self.assertGreaterEqual(summary["frame_outlier_count"], 1)
            with h5py.File(path, "r") as output:
                filtered = output["distal/points_base_mm"][:]
                before = output["distal/pre_temporal_points_base_mm"][:]
                self.assertLess(
                    np.sqrt(np.mean((filtered[45] - truth[45]) ** 2)),
                    0.35 * np.sqrt(np.mean((before[45] - truth[45]) ** 2)))
                self.assertLess(
                    np.linalg.norm(filtered[67, -1] - truth[67, -1]),
                    0.4 * np.linalg.norm(before[67, -1] - truth[67, -1]))
                self.assertTrue(np.all(
                    output["distal/temporal_outlier_mask"][67, u >= 0.8]))
                self.assertEqual(
                    output.attrs["distal_temporal_spline_basis_count"], 20)
                self.assertTrue(np.all(np.isfinite(
                    output["distal/filtered_spline_coefficients_base_mm"][:])))
                self.assertTrue(np.all(np.isfinite(
                    output["distal/curvature_per_mm"][:, 3:-3])))
                flags = output["frames/learning_rejection_flags"][:]
                self.assertTrue(
                    flags[45] & LEARNING_REJECT_SHAPE_OUTLIER)
                self.assertTrue(
                    flags[67] & LEARNING_REJECT_TERMINAL_OUTLIER)
                self.assertTrue(flags[80] & LEARNING_REJECT_MASK_WIDTH)
                self.assertEqual(output["frames/learning_valid"][67], 0)

    def test_long_rejected_run_exceeds_interpolation_limit(self):
        count = 60
        timestamps = np.arange(count, dtype=np.int64) * 33_333_333
        rejected = np.zeros((count, 3), dtype=bool)
        rejected[20:40, 2] = True
        unsupported = _long_rejected_runs(
            rejected, np.ones(count, dtype=bool), timestamps, 500.0)
        self.assertTrue(np.all(unsupported[20:40]))
        self.assertFalse(np.any(unsupported[:20]))
        self.assertFalse(np.any(unsupported[40:]))

        rejected[:] = False
        rejected[20:30, 2] = True
        supported = _long_rejected_runs(
            rejected, np.ones(count, dtype=bool), timestamps, 500.0)
        self.assertFalse(np.any(supported))

    def test_local_long_rejected_run_does_not_invalidate_whole_curve(self):
        count, samples = 60, 64
        timestamps = np.arange(count, dtype=np.int64) * 33_333_333
        rejected = np.zeros((count, samples), dtype=bool)
        rejected[20:40, 30:37] = True
        supported = _long_rejected_runs(
            rejected, np.ones(count, dtype=bool), timestamps, 500.0,
            minimum_sample_fraction=0.5)
        self.assertFalse(np.any(supported))

        rejected[20:40, :40] = True
        unsupported = _long_rejected_runs(
            rejected, np.ones(count, dtype=bool), timestamps, 500.0,
            minimum_sample_fraction=0.5)
        self.assertTrue(np.all(unsupported[20:40]))


if __name__ == "__main__":
    unittest.main()
