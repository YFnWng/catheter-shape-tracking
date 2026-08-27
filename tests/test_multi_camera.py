import csv
import tempfile
import unittest
from pathlib import Path

from shape_tracking.multi_camera import (
    CameraCaptureWorker,
    CameraRigSpec,
    frame_pair_tolerance_ns,
    RecordedFrame,
    pair_recorded_frames,
    validate_rig_specs,
    write_frame_pairs,
)
from shape_tracking.dual_cli import _right_pose_to_left


class RigConfigTests(unittest.TestCase):
    def test_accepts_two_unique_serial_selected_rigs(self):
        rigs = validate_rig_specs([
            CameraRigSpec("primary", 20757336),
            CameraRigSpec("oblique", 26080456),
        ])
        self.assertEqual([item.rig_id for item in rigs], ["primary", "oblique"])

    def test_rejects_duplicate_serials(self):
        with self.assertRaisesRegex(ValueError, "serial numbers must be unique"):
            validate_rig_specs([
                CameraRigSpec("primary", 20757336),
                CameraRigSpec("oblique", 20757336),
            ])

    def test_preview_wait_returns_none_when_no_new_frame_arrives(self):
        class Camera:
            is_recording = False

        worker = CameraCaptureWorker("test", Camera())
        self.assertIsNone(worker.wait_for_preview(timeout=0.001))

    def test_right_eye_pose_is_expressed_in_left_frame(self):
        import numpy as np

        left_camera_T_right_camera = np.eye(4)
        left_camera_T_right_camera[0, 3] = 0.12
        rvec, tvec = _right_pose_to_left(
            [0, 0, 0], [0, 0, 1], left_camera_T_right_camera)
        np.testing.assert_allclose(rvec, 0, atol=1e-12)
        np.testing.assert_allclose(tvec, [0.12, 0, 1], atol=1e-12)


class PairingTests(unittest.TestCase):
    def test_pairing_tolerance_scales_with_frame_period(self):
        records = [RecordedFrame(0, 0), RecordedFrame(1, 67_000_000)]
        self.assertEqual(frame_pair_tolerance_ns(records, records), 40_200_000)

    def test_pairs_nearest_frames_monotonically_with_drops(self):
        reference = [RecordedFrame(i, ts) for i, ts in enumerate(
            [0, 33_000_000, 66_000_000, 99_000_000])]
        secondary = [RecordedFrame(i, ts) for i, ts in enumerate(
            [4_000_000, 70_000_000, 103_000_000])]
        pairs = pair_recorded_frames(reference, secondary)
        self.assertEqual(
            [pair.secondary_frame for pair in pairs], [0, None, 1, 2])
        self.assertEqual(
            [pair.timestamp_delta_ns for pair in pairs],
            [4_000_000, None, 4_000_000, 4_000_000])

    def test_writes_pair_sidecar_and_statistics(self):
        reference = [RecordedFrame(0, 10), RecordedFrame(1, 30)]
        secondary = [RecordedFrame(0, 12), RecordedFrame(1, 27)]
        pairs = pair_recorded_frames(reference, secondary)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_frame_pairs.csv"
            stats = write_frame_pairs(path, pairs)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["paired_frames"], 2)
        self.assertEqual(stats["unpaired_reference_frames"], 0)


if __name__ == "__main__":
    unittest.main()
