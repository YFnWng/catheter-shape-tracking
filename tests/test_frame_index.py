import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from shape_tracking.session import (
    FrameRecord,
    load_frame_index,
    normalize_frame_records,
)
from shape_tracking.cli import _limit_frame_index, parse_args


class FrameIndexTest(unittest.TestCase):
    def test_image_only_cli_rejects_aurora_port(self):
        self.assertTrue(parse_args(["--image-only"]).image_only)
        with self.assertRaises(SystemExit):
            parse_args(["--image-only", "--aurora-port", "COM4"])

    def test_limits_sidecar_to_playable_svo_count(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame_index.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["svo_frame", "timestamp_ns"])
                writer.writerows([[0, 100], [1, 133], [2, 166]])
            self.assertEqual(_limit_frame_index(str(path), 2), 2)
            self.assertEqual(
                load_frame_index(path),
                [FrameRecord(0, 100), FrameRecord(1, 133)])

    def test_drops_repeated_timestamps_and_renumbers_svo_frames(self):
        records = [
            FrameRecord(0, 100),
            FrameRecord(1, 133),
            FrameRecord(2, 133),
            FrameRecord(3, 166),
        ]
        self.assertEqual(
            normalize_frame_records(records),
            [FrameRecord(0, 100), FrameRecord(1, 133), FrameRecord(2, 166)])

    def test_load_frame_index_normalizes_legacy_sidecar(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame_index.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["svo_frame", "timestamp_ns"])
                writer.writerows([[7, 100], [8, 133], [9, 133], [10, 166]])
            self.assertEqual(
                load_frame_index(path),
                [FrameRecord(0, 100), FrameRecord(1, 133), FrameRecord(2, 166)])

    def test_rejects_backward_timestamp(self):
        with self.assertRaisesRegex(ValueError, "not monotonic"):
            normalize_frame_records([
                FrameRecord(0, 100), FrameRecord(1, 99)])


if __name__ == "__main__":
    unittest.main()
