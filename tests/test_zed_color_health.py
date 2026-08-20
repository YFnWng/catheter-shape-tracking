import unittest

import numpy as np

from shape_tracking.zed_capture import assess_stereo_color


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

    def test_invalid_image_is_rejected(self):
        valid = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            assess_stereo_color(np.zeros((10, 10), dtype=np.uint8), valid)


if __name__ == "__main__":
    unittest.main()
