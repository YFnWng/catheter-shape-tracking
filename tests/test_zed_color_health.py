import unittest

import numpy as np

from shape_tracking.zed_capture import (
    assess_multi_camera_color, assess_stereo_color)


class StereoColorHealthTest(unittest.TestCase):

    def test_neutral_stereo_pair_is_healthy(self):
        image = np.full((80, 120, 3), 180, dtype=np.uint8)
        report = assess_stereo_color(image, image.copy())
        self.assertTrue(report["healthy"])
        self.assertTrue(report["eyes"]["left"]["healthy"])
        self.assertTrue(report["eyes"]["right"]["healthy"])

    def test_green_clipped_right_eye_is_rejected(self):
        neutral = np.full((80, 120, 3), 180, dtype=np.uint8)
        green = np.empty_like(neutral)
        green[..., 0] = 18
        green[..., 1] = 255
        green[..., 2] = 18
        report = assess_stereo_color(neutral, green)
        self.assertFalse(report["healthy"])
        self.assertTrue(report["eyes"]["left"]["healthy"])
        self.assertFalse(report["eyes"]["right"]["healthy"])
        self.assertGreater(
            report["eyes"]["right"]["green_clip_fraction"], 0.99)
        self.assertFalse(report["runtime_healthy"])

    def test_blue_white_background_is_rejected(self):
        blue = np.empty((80, 120, 3), dtype=np.uint8)
        blue[..., 0] = 210
        blue[..., 1] = 115
        blue[..., 2] = 95
        report = assess_stereo_color(blue, blue.copy())
        self.assertFalse(report["healthy"])
        self.assertGreater(
            report["eyes"]["left"]["neutral_channel_ratio"], 2.0)

    def test_small_colored_object_does_not_bias_white_background(self):
        image = np.full((80, 120, 3), 180, dtype=np.uint8)
        image[30:40, 50:70] = [20, 20, 240]
        self.assertTrue(assess_stereo_color(image, image.copy())["healthy"])

    def test_visible_magenta_cast_is_rejected_per_camera(self):
        left = np.empty((80, 120, 3), dtype=np.uint8)
        left[..., 0], left[..., 1], left[..., 2] = 230, 164, 208
        right = np.empty_like(left)
        right[..., 0], right[..., 1], right[..., 2] = 234, 165, 200
        self.assertFalse(assess_stereo_color(left, right)["healthy"])

    def test_observed_primary_fixed_wb_cast_is_rejected(self):
        left = np.empty((80, 120, 3), dtype=np.uint8)
        left[..., 0], left[..., 1], left[..., 2] = 230, 158, 202
        right = np.empty_like(left)
        right[..., 0], right[..., 1], right[..., 2] = 224, 159, 202
        self.assertFalse(assess_stereo_color(left, right)["healthy"])

    def test_latest_single_eye_red_cast_is_rejected(self):
        left = np.empty((80, 120, 3), dtype=np.uint8)
        left[..., 0], left[..., 1], left[..., 2] = 216, 196, 213
        right = np.empty_like(left)
        right[..., 0], right[..., 1], right[..., 2] = 211, 211, 212
        report = assess_stereo_color(left, right)
        self.assertFalse(report["healthy"])
        self.assertFalse(report["eyes"]["left"]["healthy"])
        self.assertTrue(report["runtime_healthy"])

    def test_consistent_slight_green_balance_at_exposure_six_is_accepted(self):
        left = np.empty((80, 120, 3), dtype=np.uint8)
        left[..., 0], left[..., 1], left[..., 2] = 174, 181, 171
        right = np.empty_like(left)
        right[..., 0], right[..., 1], right[..., 2] = 173, 178, 172
        report = assess_stereo_color(left, right)
        self.assertTrue(report["healthy"])
        self.assertTrue(report["runtime_healthy"])

    def test_cross_camera_color_mismatch_is_rejected(self):
        neutral = [1 / 3, 1 / 3, 1 / 3]
        warm = [0.30, 0.33, 0.37]
        reports = {
            "primary": {
                "healthy": True,
                "eyes": {
                    "left": {"neutral_chromaticity_bgr": neutral},
                    "right": {"neutral_chromaticity_bgr": neutral}}},
            "oblique": {
                "healthy": True,
                "eyes": {
                    "left": {"neutral_chromaticity_bgr": warm},
                    "right": {"neutral_chromaticity_bgr": warm}}}}
        combined = assess_multi_camera_color(reports)
        self.assertFalse(combined["healthy"])
        self.assertGreater(
            combined["max_rig_chromaticity_delta"],
            combined["max_rig_chromaticity_delta_limit"])

    def test_matching_camera_rigs_are_healthy(self):
        image = np.empty((80, 120, 3), dtype=np.uint8)
        image[..., 0] = 180
        image[..., 1] = 183
        image[..., 2] = 188
        reports = {
            "primary": assess_stereo_color(image, image.copy()),
            "oblique": assess_stereo_color(image, image.copy())}
        self.assertTrue(assess_multi_camera_color(reports)["healthy"])

    def test_invalid_image_is_rejected(self):
        valid = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            assess_stereo_color(np.zeros((10, 10), dtype=np.uint8), valid)


if __name__ == "__main__":
    unittest.main()
