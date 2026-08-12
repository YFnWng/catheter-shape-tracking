import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from shape_tracking.image_sequence import (
    ImageProcessingConfig,
    PropagatedMaskCache,
    _fitted_curve_reprojection_metrics,
    _stereo_tip_camera_point,
)
from shape_tracking.session import FrameRecord


class ImageSequenceGeometryTests(unittest.TestCase):
    def test_final_curve_reprojection_is_measured_after_fitting(self):
        registration = SimpleNamespace(
            K=np.array([[100.0, 0.0, 0.0],
                        [0.0, 100.0, 0.0],
                        [0.0, 0.0, 1.0]]),
            baseline_m=0.0,
            left_camera_T_base=np.eye(4),
            right_camera_T_base=np.eye(4),
        )
        points_mm = np.array([
            [0.0, 0.0, 1000.0],
            [10.0, 0.0, 1000.0],
            [20.0, 0.0, 1000.0],
        ])
        centerline_px = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metrics = _fitted_curve_reprojection_metrics(
            registration, points_mm, centerline_px, centerline_px)
        self.assertAlmostEqual(metrics["fitted_reprojection_left_px"], 0.0)
        self.assertAlmostEqual(metrics["fitted_reprojection_right_px"], 0.0)
        self.assertAlmostEqual(metrics["fitted_reprojection_p95_px"], 0.0)

    def test_temporal_disparity_default_is_stabilizing(self):
        self.assertEqual(ImageProcessingConfig().temporal_disparity_weight, 2.0)

    def test_rectified_yellow_tip_is_triangulated(self):
        point, error = _stereo_tip_camera_point(
            np.array([20.0, 10.0]), np.array([10.0, 10.0]),
            np.array([[100.0, 0.0, 0.0],
                      [0.0, 100.0, 0.0],
                      [0.0, 0.0, 1.0]]), 0.1)
        np.testing.assert_allclose(point, [0.2, 0.1, 1.0])
        self.assertEqual(error, 0.0)

    def test_inconsistent_yellow_tip_is_not_used(self):
        point, error = _stereo_tip_camera_point(
            np.array([20.0, 10.0]), np.array([10.0, 20.0]),
            np.eye(3), 0.1, max_epipolar_error_px=5.0)
        self.assertIsNone(point)
        self.assertEqual(error, 10.0)

    def test_completed_image_h5_can_be_used_as_mask_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed_shapes.h5"
            with h5py.File(path, "w") as output:
                output["frames/svo_frame"] = np.array([6, 7, 8], np.int32)
                for view in ("left", "right"):
                    dataset = output.create_dataset(
                        f"images/{view}/mask_packbits", (3, 2), dtype=np.uint8)
                    dataset.attrs["roi_xywh"] = (10, 20, 4, 4)
            cache = PropagatedMaskCache(path, [FrameRecord(7, 100)])
            try:
                self.assertEqual(cache.layout, "processed_image")
                self.assertEqual(cache.source_indices.tolist(), [1])
            finally:
                cache.close()


if __name__ == "__main__":
    unittest.main()
