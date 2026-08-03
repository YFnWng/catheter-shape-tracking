import unittest

import numpy as np

from shape_tracking.robot_data import (
    RobotStreams,
    TimedJointSeries,
    align_robot_streams,
)


class RobotAlignmentTests(unittest.TestCase):
    def test_zero_order_holds_commands_and_rejects_stale_values(self):
        empty = TimedJointSeries(np.empty(0, np.int64), np.empty((0, 6)))
        commands = TimedJointSeries(
            np.array([100, 200], dtype=np.int64),
            np.array([[1] * 6, [2] * 6], dtype=float))
        aligned = align_robot_streams(
            RobotStreams(commands, empty, empty),
            np.array([150, 205, 300], dtype=np.int64),
            max_command_age_ms=0.00006)
        np.testing.assert_allclose(aligned.command_velocity[:2, 0], [1, 2])
        self.assertEqual(aligned.command_valid.tolist(), [1, 1, 0])
        self.assertTrue(np.isnan(aligned.command_velocity[2]).all())

    def test_interpolates_position_through_rotation_wrap(self):
        empty = TimedJointSeries(np.empty(0, np.int64), np.empty((0, 6)))
        values = np.zeros((2, 6), dtype=float)
        values[:, 0] = [0.0, 10.0]
        values[:, 1] = [179.0, -179.0]
        positions = TimedJointSeries(
            np.array([0, 10_000_000], dtype=np.int64), values)
        aligned = align_robot_streams(
            RobotStreams(empty, positions, empty),
            np.array([5_000_000], dtype=np.int64),
            max_feedback_gap_ms=20.0)
        self.assertEqual(aligned.position_valid[0], 1)
        self.assertAlmostEqual(aligned.measured_position[0, 0], 5.0)
        self.assertAlmostEqual(abs(aligned.measured_position[0, 1]), 180.0)


if __name__ == "__main__":
    unittest.main()
