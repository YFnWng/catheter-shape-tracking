import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / 'scripts' / 'generate_charuco_patterns.py')
SPEC = importlib.util.spec_from_file_location('generate_charuco_patterns', SCRIPT)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class CharucoGeneratorTest(unittest.TestCase):
    def test_generates_distinct_verified_100mm_boards(self):
        with TemporaryDirectory() as tmp:
            manifest_path, pdf_path = GENERATOR.generate(
                Path(tmp), count=2, first_id=17, pixels=800,
                page_name='letter', prefix='field_test')
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest['board_size_mm'], [100.0, 100.0])
            self.assertEqual(
                manifest['patterns'][0]['ids'], list(range(17, 25)))
            self.assertEqual(
                manifest['patterns'][1]['ids'], list(range(25, 33)))
            self.assertTrue(pdf_path.is_file())
            dictionary = GENERATOR.get_dictionary()
            for pattern in manifest['patterns']:
                image = cv2.imread(
                    str(Path(tmp) / pattern['png']), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(image.shape, (800, 800))
                self.assertEqual(
                    GENERATOR.detected_ids(image, dictionary), pattern['ids'])

    def test_rejects_ids_used_by_existing_boards(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'overlap existing boards'):
                GENERATOR.generate(
                    Path(tmp), count=1, first_id=9, pixels=800,
                    page_name='letter', prefix='overlap')


if __name__ == '__main__':
    unittest.main()
