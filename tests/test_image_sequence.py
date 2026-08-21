import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import cv2
import numpy as np

from shape_tracking.image_sequence import (
    _build_parser,
    _bridge_short_good_gaps,
    ImageProcessingConfig,
    ImageSequenceWriter,
    OverlayWriter,
    PropagatedMaskCache,
    _fuse_distal_length,
    _fitted_curve_reprojection_metrics,
    _project_point_to_polyline_s,
    _projected_workspace_roi_mask,
    _select_stereo_reference,
    _stereo_candidate_score,
    _stereo_tip_camera_point,
    _temporal_centerline_metrics,
)
from shape_tracking.robot_data import AlignedRobotData
from shape_tracking.session import FrameRecord


class ImageSequenceGeometryTests(unittest.TestCase):
    def test_resume_cli_exposes_all_persisted_stage_boundaries(self):
        parser = _build_parser()
        for stage in ("observations", "stereo", "joint"):
            args = parser.parse_args([
                "--session", "/tmp/session", "--resume-from", stage,
                "--resume-h5", "/tmp/source.h5"])
            self.assertEqual(args.resume_from, stage)
            self.assertEqual(args.resume_h5, "/tmp/source.h5")

    def test_workspace_mask_is_projected_box_not_bounding_rectangle(self):
        registration = SimpleNamespace(
            K=np.array([[100.0, 0.0, 50.0],
                        [0.0, 100.0, 50.0],
                        [0.0, 0.0, 1.0]]),
            left_camera_T_base=np.eye(4),
            right_camera_T_base=np.eye(4),
            roi_left_xywh=(0, 0, 100, 100),
            roi_right_xywh=(0, 0, 100, 100),
            workspace_base_m={
                "x": [-0.1, 0.1], "y": [-0.1, 0.1],
                "z": [0.5, 1.0], "margin_px": 25},
        )
        mask = _projected_workspace_roi_mask(registration, right=False)
        self.assertEqual(mask[50, 50], 255)
        self.assertEqual(mask[5, 5], 0)

    def test_workspace_mask_excludes_projected_proximal_z_sheath(self):
        transform = np.eye(4)
        transform[2, 3] = 1.0
        registration = SimpleNamespace(
            K=np.array([[100.0, 0.0, 50.0],
                        [0.0, 100.0, 50.0],
                        [0.0, 0.0, 1.0]]),
            left_camera_T_base=transform,
            right_camera_T_base=transform,
            roi_left_xywh=(0, 0, 100, 100),
            roi_right_xywh=(0, 0, 100, 100),
            workspace_base_m={
                "x": [-0.1, 0.1], "y": [-0.1, 0.1],
                "z": [0.0, 0.5], "margin_px": 25},
        )
        mask = _projected_workspace_roi_mask(
            registration, right=False,
            sheath_exclusion_radius_mm=20.0,
            sheath_exclusion_length_mm=60.0)
        self.assertEqual(mask[50, 50], 0)
        self.assertEqual(mask[50, 58], 255)

    def test_snapshot_writer_emits_selected_full_resolution_png(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            writer = OverlayWriter(
                output, fps=10.0, scale=0.5, video_enabled=False,
                snapshots_enabled=True, snapshot_indices={2})
            image = np.zeros((40, 60, 3), dtype=np.uint8)
            writer.write(
                "left", image, None, None, None, "test",
                frame_index=1, svo_frame=100)
            writer.write(
                "left", image, None, None, None, "test",
                frame_index=2, svo_frame=101)
            writer.close()
            path = output / "overlay_snapshots/left/frame_000002_svo_000101.png"
            self.assertTrue(path.exists())
            saved = cv2.imread(str(path))
            self.assertEqual(saved.shape, image.shape)

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

    def test_stereo_candidate_scoring_can_choose_the_better_eye(self):
        points = np.column_stack([
            np.linspace(0.0, 0.05, 8), np.zeros(8), np.ones(8)])
        left = {
            "reprojection_left_px": 4.0,
            "reprojection_right_px": 3.0,
            "reprojection_p95_px": 8.0,
            "points_camera_m": points + [0.0, 0.01, 0.0],
        }
        right = {
            "reprojection_left_px": 1.0,
            "reprojection_right_px": 1.2,
            "reprojection_p95_px": 2.0,
            "points_camera_m": points,
        }
        left_score, _ = _stereo_candidate_score(
            left, "left", 80.0, 130.0, points, "left")
        right_score, _ = _stereo_candidate_score(
            right, "right", 80.0, 130.0, points, "left")
        self.assertLess(right_score, left_score)

    def test_self_overlapped_eye_cannot_be_selected_as_reference(self):
        candidates = {
            "left": (10.0, {
                "reference_eye_self_overlap_fraction": 0.0,
                "other_eye_self_overlap_fraction": 0.8}),
            "right": (1.0, {
                "reference_eye_self_overlap_fraction": 0.8,
                "other_eye_self_overlap_fraction": 0.0}),
        }
        self.assertEqual(_select_stereo_reference(
            candidates, "right", hysteresis_score=100.0), "left")

    def test_reference_hysteresis_retains_previous_close_candidate(self):
        candidates = {
            "left": (5.0, {
                "reference_eye_self_overlap_fraction": 0.0,
                "other_eye_self_overlap_fraction": 0.0}),
            "right": (4.0, {
                "reference_eye_self_overlap_fraction": 0.0,
                "other_eye_self_overlap_fraction": 0.0}),
        }
        self.assertEqual(_select_stereo_reference(
            candidates, "left", hysteresis_score=2.0), "left")

    def test_interface_length_is_softly_fused(self):
        raw, filtered, uncertainty = _fuse_distal_length(
            60.0, [56.0, 58.0], [0.8, 1.0], 59.0,
            prior_sigma_mm=4.0, color_sigma_mm=3.0,
            temporal_sigma_mm=1.5)
        self.assertGreater(raw, 56.0)
        self.assertLess(raw, 60.0)
        self.assertGreater(filtered, raw)
        self.assertLess(filtered, 60.0)
        self.assertGreater(uncertainty, 0.0)

    def test_previous_interface_is_projected_locally_on_current_curve(self):
        curve = np.column_stack([
            np.linspace(0.0, 100.0, 101), np.zeros(101), np.zeros(101)])
        projected = _project_point_to_polyline_s(
            curve, np.array([52.25, 3.0, 0.0]), 50.0, 10.0)
        self.assertAlmostEqual(projected, 52.25)

    def test_temporal_centerline_metric_detects_collapsed_path(self):
        previous = np.column_stack([np.arange(101.0), np.zeros(101)])
        current = previous[:25]
        coverage, p95 = _temporal_centerline_metrics(
            previous, current, tolerance_px=5.0)
        self.assertLess(coverage, 0.4)
        self.assertGreater(p95, 50.0)

    def test_final_quality_bridges_only_short_bounded_good_islands(self):
        rejected = np.array(
            [False, True, False, False, True, False, True, False])
        valid = np.ones(len(rejected), dtype=bool)
        timestamps = np.arange(len(rejected), dtype=np.int64) * 30_000_000
        stabilized = _bridge_short_good_gaps(
            rejected, valid, timestamps, maximum_gap_ms=100.0)
        np.testing.assert_array_equal(
            stabilized,
            [False, True, True, True, True, True, True, False])

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

    def test_hdf_writer_batches_records_and_preserves_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "buffered.h5"
            config = ImageProcessingConfig(
                full_samples=8, distal_samples=8,
                image_centerline_samples=8, store_masks=False,
                hdf_buffer_frames=2, hdf_queue_chunks=1)
            registration = SimpleNamespace(
                roi_left_xywh=(0, 0, 20, 10),
                roi_right_xywh=(0, 0, 20, 10),
                K=np.eye(3), baseline_m=0.1,
                left_camera_T_base=np.eye(4),
                right_camera_T_base=np.eye(4))
            values = np.arange(18, dtype=float).reshape(3, 6)
            scalar = np.arange(3, dtype=float)
            flags = np.array([True, False, True])
            robot = AlignedRobotData(
                values, values + 1, values + 2,
                scalar, scalar + 1, scalar + 2,
                flags, flags, flags)
            writer = ImageSequenceWriter(
                path, 3, config, registration, {"test": True})
            for index in range(3):
                writer.write_identity(
                    index, FrameRecord(index + 10, 100 + index),
                    200 + index, robot)
                metrics = {"reprojection_left_px": 1.25} if index == 1 else {}
                writer.write_failure(index, f"failed_{index}", metrics)
            writer.close()

            with h5py.File(path, "r") as output:
                np.testing.assert_array_equal(
                    output["frames/svo_frame"][:], [10, 11, 12])
                self.assertEqual(
                    [value.decode() if isinstance(value, bytes) else value
                     for value in output["frames/status"][:]],
                    ["failed_0", "failed_1", "failed_2"])
                np.testing.assert_allclose(
                    output["robot/joint_velocity_command"][:], values)
                quality = output["quality/reprojection_left_px"][:]
                self.assertTrue(np.isnan(quality[0]))
                self.assertEqual(quality[1], np.float32(1.25))
                self.assertTrue(np.isnan(quality[2]))


if __name__ == "__main__":
    unittest.main()
