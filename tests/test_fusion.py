import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from shape_tracking.fusion import (
    FusionConfig,
    nearest_sample_indices,
    select_actuation_timestamps,
    select_image_timestamps,
    write_fused_dataset,
)
from shape_tracking.robot_data import RobotStreams, TimedJointSeries, align_robot_streams
from shape_tracking.session import TipPose


def _series(timestamps, value=1.0):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    return TimedJointSeries(
        timestamps, np.full((len(timestamps), 6), value, dtype=np.float64))


class _FakeEm:
    def tip_pose(self, timestamp_ns):
        if timestamp_ns == 20_000_000:
            return TipPose.invalid(timestamp_ns, "synthetic_gap")
        return TipPose(
            timestamp_ns=timestamp_ns,
            valid=True,
            position_base_mm=np.array([timestamp_ns / 1e6, 1.0, 2.0]),
            z_direction_base=np.array([0.0, 0.0, 1.0]),
            quaternion_base_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            coil_positions_base_mm=np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            coil_separation_mm=2.0,
            max_sync_offset_ms=1.0,
            max_bracket_span_ms=10.0,
            status="valid",
        )


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.query = np.array([10_000_000, 20_000_000, 30_000_000], np.int64)
        streams = RobotStreams(
            _series(self.query, 1.0),
            _series([0, 20_000_000, 40_000_000], 2.0),
            _series([0, 20_000_000, 40_000_000], 3.0),
        )
        self.robot = align_robot_streams(streams, self.query)

    def _write_image(self, path):
        with h5py.File(path, "w") as output:
            output.attrs["schema_version"] = 6
            frames = output.create_group("frames")
            frames["timestamp_ns"] = np.array([9_000_000, 29_000_000], np.int64)
            frames["valid"] = np.array([1, 0], np.uint8)
            frames["learning_valid"] = np.array([1, 0], np.uint8)
            frames["learning_rejection_flags"] = np.array([0, 1], np.uint16)
            frames["observation_valid"] = np.array([1, 1], np.uint8)
            frames["pre_interpolation_valid"] = np.array([1, 0], np.uint8)
            frames["curve_temporally_interpolated"] = np.array([0, 1], np.uint8)
            frames["svo_frame"] = np.array([3, 4], np.int32)
            frames.create_dataset(
                "status", data=np.array(["valid", "bad"], object),
                dtype=h5py.string_dtype("utf-8"))
            for name, samples in (("full", 4), ("distal", 3)):
                group = output.create_group(name)
                group["points_base_mm"] = np.arange(
                    2 * samples * 3, dtype=np.float32).reshape(2, samples, 3)
                group["curvature_per_mm"] = np.ones((2, samples), np.float32)
            output["distal/base_position_base_mm"] = np.ones((2, 3), np.float32)
            quality = output.create_group("quality")
            quality["reprojection_max_px"] = np.array([1.0, 2.0], np.float32)

    def test_nearest_matching_reports_signed_offset_and_gap(self):
        index, offset, valid = nearest_sample_indices(
            np.array([0, 20_000_000]),
            np.array([4_000_000, 16_000_000, 40_000_000]), 5.0)
        self.assertEqual(index.tolist(), [0, 1, 1])
        np.testing.assert_allclose(offset, [-4.0, 4.0, -20.0])
        self.assertEqual(valid.tolist(), [1, 1, 0])

    def test_run_window_uses_actuation_clock(self):
        streams = RobotStreams(
            _series([0, 10, 20, 30]), _series([0, 30]), _series([0, 30]))
        selected = select_actuation_timestamps(
            streams,
            {"run_start": {"stamp_ns": 10}, "run_end": {"stamp_ns": 30}},
            stride=2)
        self.assertEqual(selected.tolist(), [10])

    def test_run_window_can_use_unique_image_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.h5"
            self._write_image(image_path)
            selected = select_image_timestamps(
                image_path,
                {"run_start": {"stamp_ns": 9_000_000},
                 "run_end": {"stamp_ns": 30_000_000}})
            self.assertEqual(selected.tolist(), [9_000_000, 29_000_000])

    def test_image_only_em_only_and_both(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image_path = directory / "image.h5"
            self._write_image(image_path)
            for use_image, use_em in ((True, False), (False, True), (True, True)):
                output_path = directory / f"fused_{use_image}_{use_em}.h5"
                summary = write_fused_dataset(
                    output_path, self.query, self.robot,
                    image_h5=image_path, use_image=use_image,
                    em_synchronizer=_FakeEm(), use_em=use_em,
                    config=FusionConfig(max_image_offset_ms=15.0))
                self.assertEqual(summary["sample_count"], 3)
                with h5py.File(output_path, "r") as output:
                    self.assertEqual(output.attrs["master_clock"],
                                     "rosbag_joint_velocity_command")
                    self.assertEqual(bool(output["image"].attrs["enabled"]), use_image)
                    self.assertEqual(bool(output["em"].attrs["enabled"]), use_em)
                    if use_image:
                        self.assertIn("distal/points_base_mm", output["image"])
                        self.assertEqual(
                            output["image/source_index"][:].tolist(), [0, 1, 1])
                        self.assertEqual(
                            output["image/is_new_sample"][:].tolist(), [1, 1, 0])
                        self.assertEqual(
                            output["image/reconstruction_valid"][:].tolist(),
                            [1, 0, 0])
                        self.assertEqual(
                            output["image/learning_valid"][:].tolist(),
                            [1, 0, 0])
                        self.assertEqual(
                            output["image/learning_rejection_flags"][:].tolist(),
                            [0, 1, 1])
                        self.assertEqual(
                            output["image/curve_temporally_interpolated"][:]
                            .tolist(), [0, 1, 1])
                    expected = [1, 1, 1]
                    if use_image:
                        expected[1:] = [0, 0]
                    if use_em:
                        expected[1] = 0
                    self.assertEqual(output["frames/fusion_valid"][:].tolist(), expected)

    def test_image_learning_label_rejects_reconstructed_outlier(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image_path = directory / "image.h5"
            self._write_image(image_path)
            with h5py.File(image_path, "r+") as output:
                output["frames/learning_valid"][0] = 0
            output_path = directory / "fused.h5"
            write_fused_dataset(
                output_path, self.query, self.robot,
                image_h5=image_path, use_image=True, use_em=False,
                config=FusionConfig(max_image_offset_ms=15.0))
            with h5py.File(output_path, "r") as output:
                self.assertEqual(
                    output["image/reconstruction_valid"][:].tolist(), [1, 0, 0])
                self.assertEqual(
                    output["image/valid"][:].tolist(), [0, 0, 0])
                self.assertEqual(
                    output["frames/fusion_valid"][:].tolist(), [0, 0, 0])

    def test_image_clock_metadata_and_exact_image_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image_path = directory / "image.h5"
            self._write_image(image_path)
            query = np.array([9_000_000, 29_000_000], np.int64)
            streams = RobotStreams(
                _series(query), _series(query), _series(query))
            robot = align_robot_streams(streams, query)
            output_path = directory / "fused.h5"
            write_fused_dataset(
                output_path, query, robot, image_h5=image_path,
                use_image=True, config=FusionConfig(timeline="image"))
            with h5py.File(output_path, "r") as output:
                self.assertEqual(output.attrs["master_clock"],
                                 "processed_image_frame")
                self.assertEqual(output["image/source_index"][:].tolist(),
                                 [0, 1])
                self.assertEqual(output["image/is_new_sample"][:].tolist(),
                                 [1, 1])


if __name__ == "__main__":
    unittest.main()
