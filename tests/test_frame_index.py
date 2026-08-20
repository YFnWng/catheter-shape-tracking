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

    def test_manual_exposure_and_gain_must_be_paired(self):
        with self.assertRaises(SystemExit):
            parse_args(["--camera-config", "", "--exposure", "20"])
        with self.assertRaises(SystemExit):
            parse_args(["--camera-config", "", "--gain", "20"])
        args = parse_args([
            "--camera-config", "", "--exposure", "20", "--gain", "30",
            "--white-balance-temperature", "4500"])
        self.assertEqual(args.exposure, 20)
        self.assertEqual(args.gain, 30)
        self.assertEqual(args.white_balance_temperature, 4500)

    def test_camera_yaml_defaults_and_cli_override(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "camera.yaml"
            config.write_text(
                "camera:\n"
                "  exposure: 8\n"
                "  gain: 1\n"
                "  white_balance_temperature: 2800\n"
                "  hue: 0\n"
                "recording:\n"
                "  svo_compression: H264_LOSSLESS\n",
                encoding="utf-8")
            args = parse_args([
                "--camera-config", str(config), "--exposure", "10",
                "--gain", "2"])
            self.assertEqual(args.exposure, 10)
            self.assertEqual(args.gain, 2)
            self.assertEqual(args.white_balance_temperature, 2800)
            self.assertEqual(args.hue, 0)
            self.assertEqual(args.svo_compression, "H264_LOSSLESS")

    def test_white_balance_requires_hardware_supported_100k_step(self):
        with self.assertRaises(SystemExit):
            parse_args([
                "--camera-config", "",
                "--white-balance-temperature", "4999"])
        args = parse_args([
            "--camera-config", "",
            "--white-balance-temperature", "5000"])
        self.assertEqual(args.white_balance_temperature, 5000)

    def test_auto_freeze_excludes_manual_white_balance(self):
        with self.assertRaises(SystemExit):
            parse_args([
                "--camera-config", "",
                "--white-balance-temperature", "5000",
                "--white-balance-auto-freeze-s", "5"])
        args = parse_args([
            "--camera-config", "",
            "--white-balance-temperature", "-1",
            "--white-balance-auto-freeze-s", "5"])
        self.assertEqual(args.white_balance_auto_freeze_s, 5.0)

    def test_auto_freeze_retry_count_must_be_nonnegative(self):
        with self.assertRaises(SystemExit):
            parse_args([
                "--camera-config", "",
                "--white-balance-auto-freeze-retries", "-1"])
        args = parse_args([
            "--camera-config", "",
            "--white-balance-auto-freeze-retries", "3"])
        self.assertEqual(args.white_balance_auto_freeze_retries, 3)

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
