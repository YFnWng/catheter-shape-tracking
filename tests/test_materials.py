import unittest

import cv2
import numpy as np

from shape_tracking.materials import extract_material_centerline


class MaterialSegmentationTests(unittest.TestCase):
    def test_detects_dark_to_bright_blue_transition(self):
        image = np.full((200, 320, 3), 210, dtype=np.uint8)
        cv2.line(image, (20, 100), (145, 100), (70, 0, 0), 9,
                 cv2.LINE_AA)
        cv2.line(image, (145, 100), (275, 100), (255, 30, 20), 9,
                 cv2.LINE_AA)
        material = extract_material_centerline(
            image, (0, 0, 320, 200),
            base_point=np.array([15.0, 100.0]),
            tip_point=np.array([295.0, 100.0]))
        self.assertIsNotNone(material)
        self.assertTrue(material.material_valid)
        self.assertGreater(material.boundary_contrast, 10.0)
        self.assertGreater(material.distal_boundary_fraction, 0.35)
        self.assertLess(material.distal_boundary_fraction, 0.65)
        self.assertLess(
            np.linalg.norm(material.points[0] - [20.0, 100.0]), 10.0)


if __name__ == "__main__":
    unittest.main()
