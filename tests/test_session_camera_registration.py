import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from shape_tracking import boards as boards_mod
from shape_tracking.register import save_session_camera_registration


class SessionCameraRegistrationTest(unittest.TestCase):
    def test_merges_camera_registration_and_writes_overlays(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'registration_config.yaml'
            config.write_text(
                'units: m\n'
                'markers:\n'
                '  0:\n'
                '    matrix:\n'
                '      - [1, 0, 0, 0]\n'
                '      - [0, 1, 0, 0]\n'
                '      - [0, 0, 1, 0]\n'
                '      - [0, 0, 0, 1]\n'
                'workspace:\n'
                '  x: [-0.05, 0.05]\n'
                '  y: [-0.05, 0.05]\n'
                '  z: [0.0, 0.2]\n'
                '  margin_px: 5\n')
            registration_path = root / 'registration.json'
            registration_path.write_text(json.dumps({
                'schema_version': 1,
                'session_id': 'test',
                'registration_config': str(config),
                'em': {'transform_status': 'solved'},
                'camera': None,
            }))
            observations = {
                0: {
                    'r': [[0.0, 0.0, 0.0]] * 10,
                    't': [[0.0, 0.0, 1.0]] * 10,
                    'c': [12] * 10,
                }
            }
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            K = np.array([
                [200.0, 0.0, 160.0],
                [0.0, 200.0, 120.0],
                [0.0, 0.0, 1.0],
            ])
            _, boards = boards_mod.build_boards()

            camera = save_session_camera_registration(
                registration_path=str(registration_path),
                output_dir=str(root),
                config_path=str(config),
                collected=observations,
                left_bgr=image,
                right_bgr=image,
                boards=boards,
                K=K,
                dist=np.zeros(5),
                resolution='TEST',
                zed_serial='123',
                baseline_m=0.12,
                image_timestamp_ns=456,
                min_frames=10,
            )

            document = json.loads(registration_path.read_text())
            self.assertTrue(document['complete'])
            self.assertEqual(document['camera']['status'], 'solved')
            self.assertEqual(camera['primary_board'], 0)
            self.assertEqual(camera['image_timestamp_ns'], 456)
            self.assertEqual(
                camera['left_camera_T_robot_base'][2][3], 1.0)
            self.assertTrue((root / 'registration_left.png').is_file())
            self.assertTrue((root / 'registration_right.png').is_file())
            self.assertIsNotNone(
                cv2.imread(str(root / 'registration_left.png')))


if __name__ == '__main__':
    unittest.main()
