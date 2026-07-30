import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from shape_tracking.session import EmSynchronizer


FIELDS = [
    "timestamp_ns", "tool_role", "part_number", "sensor_status", "valid",
    "tx_mm", "ty_mm", "tz_mm", "qw", "qx", "qy", "qz",
]


class EmFusionTests(unittest.TestCase):
    def _write_csv(self, path, timestamps):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for timestamp in timestamps:
                alpha = timestamp / 10_000_000
                for role, part, y in (
                        ("tip_coil_1", "003", 0.0),
                        ("tip_coil_0", "07222026_01", 2.0)):
                    writer.writerow({
                        "timestamp_ns": timestamp,
                        "tool_role": role,
                        "part_number": part,
                        "sensor_status": "01",
                        "valid": "1",
                        "tx_mm": alpha,
                        "ty_mm": y,
                        "tz_mm": 10.0,
                        "qw": 1.0,
                        "qx": 0.0,
                        "qy": 0.0,
                        "qz": 0.0,
                    })

    def test_interpolates_two_coils_and_builds_tip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "em.csv"
            self._write_csv(path, [0, 10_000_000])
            synchronizer = EmSynchronizer.from_csv(
                path, np.eye(4), max_interp_gap_ms=20,
                max_nearest_ms=5)
            tip = synchronizer.tip_pose(5_000_000)
        self.assertTrue(tip.valid)
        np.testing.assert_allclose(
            tip.position_base_mm, [0.5, 1.0, 10.0], atol=1e-9)
        np.testing.assert_allclose(
            tip.z_direction_base, [0.0, 0.0, 1.0], atol=1e-9)
        quaternion_xyzw = np.roll(tip.quaternion_base_wxyz, -1)
        tip_rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
        np.testing.assert_allclose(
            tip_rotation[:, 0], [0.0, -1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            tip.coil_positions_base_mm,
            [[0.5, 0.0, 10.0], [0.5, 2.0, 10.0]], atol=1e-9)
        self.assertAlmostEqual(tip.coil_separation_mm, 2.0)
        self.assertAlmostEqual(tip.max_sync_offset_ms, 5.0)

    def test_rejects_long_missing_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "em.csv"
            self._write_csv(path, [0, 200_000_000])
            synchronizer = EmSynchronizer.from_csv(
                path, np.eye(4), max_interp_gap_ms=75,
                max_nearest_ms=25)
            tip = synchronizer.tip_pose(100_000_000)
        self.assertFalse(tip.valid)
        self.assertIn("missing_or_long_gap", tip.status)


if __name__ == "__main__":
    unittest.main()
