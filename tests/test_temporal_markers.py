import unittest

import numpy as np

from shape_tracking.temporal_markers import repair_stereo_marker_tracks


class TemporalMarkerTests(unittest.TestCase):
    def test_bidirectional_track_replaces_snap_and_missing_samples(self):
        count = 12
        timestamps = np.arange(count, dtype=np.int64) * 33_333_333
        centers = np.zeros((count, 4, 2), dtype=float)
        centers[..., 0] = np.arange(count)[:, None] * 2.0
        centers[..., 1] = 10.0 + np.arange(4)[None]
        widths = np.full((count, 4), 6.0)
        confidence = np.ones((count, 4))
        observed = np.ones((count, 4), np.uint8)
        centers[5, 1] += [45.0, -30.0]
        centers[7, 2] = np.nan
        observed[7, 2] = 0
        result = repair_stereo_marker_tracks(
            {"left": centers, "right": centers.copy()},
            {"left": widths, "right": widths.copy()},
            {"left": confidence, "right": confidence.copy()},
            {"left": observed, "right": observed.copy()}, timestamps)
        np.testing.assert_allclose(result["left"]["centers"][5, 1], [10, 11])
        np.testing.assert_allclose(result["left"]["centers"][7, 2], [14, 12])
        self.assertEqual(result["left"]["interpolated"][5, 1], 1)
        self.assertEqual(result["left"]["interpolated"][7, 2], 1)


if __name__ == "__main__":
    unittest.main()
