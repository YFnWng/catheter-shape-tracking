import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "generate_tip_aruco_marker.py")
SPEC = importlib.util.spec_from_file_location(
    "generate_tip_aruco_marker", SCRIPT)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class TipArucoGeneratorTest(unittest.TestCase):
    def test_generates_verified_dimension_manifest(self):
        with TemporaryDirectory() as tmp:
            manifest_path, pdf_path, png_path = GENERATOR.generate(
                Path(tmp), marker_id=33, marker_size_mm=15.0,
                quiet_zone_mm=3.0, pixels=400, copies=2,
                page_name="letter", prefix="tip_test")
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["dictionary"], "DICT_4X4_50")
            self.assertEqual(manifest["marker_id"], 33)
            self.assertEqual(manifest["marker_size_mm"], 15.0)
            self.assertEqual(manifest["cut_target_size_mm"], [21.0, 21.0])
            self.assertTrue(pdf_path.is_file())
            image = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(
                GENERATOR.detected_ids(image, GENERATOR.get_dictionary()),
                [33])

    def test_rejects_reserved_marker_id(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "reserved"):
                GENERATOR.generate(
                    Path(tmp), marker_id=24, pixels=400, copies=1)

    def test_rejects_insufficient_quiet_zone(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "one marker bit"):
                GENERATOR.generate(
                    Path(tmp), marker_id=33, marker_size_mm=16.0,
                    quiet_zone_mm=1.0, pixels=400, copies=1)


if __name__ == "__main__":
    unittest.main()
