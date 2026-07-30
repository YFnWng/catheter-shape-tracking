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
                'em': {
                    'transform_status': 'solved',
                    'robot_base_T_aurora': [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                },
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
            em_tool_poses = [
                {
                    'tool_role': 'tip_coil_0',
                    'part_number': '003',
                    'serial_number': 'A',
                    'timestamp_ns': 450,
                    'timestamp_delta_ms': -0.006,
                    'position_aurora_mm': [-20.0, 0.0, 50.0],
                    'quaternion_aurora_wxyz': [
                        0.70710678, 0.0, 0.70710678, 0.0],
                },
                {
                    'tool_role': 'tip_coil_1',
                    'part_number': '07222026_01',
                    'serial_number': 'B',
                    'timestamp_ns': 452,
                    'timestamp_delta_ms': -0.004,
                    'position_aurora_mm': [20.0, 0.0, 50.0],
                    'quaternion_aurora_wxyz': [
                        0.70710678, 0.0, 0.70710678, 0.0],
                },
            ]

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
                em_tool_poses=em_tool_poses,
            )

            document = json.loads(registration_path.read_text())
            self.assertTrue(document['complete'])
            self.assertEqual(document['camera']['status'], 'solved')
            self.assertEqual(camera['primary_board'], 0)
            self.assertEqual(camera['image_timestamp_ns'], 456)
            self.assertEqual(
                camera['left_camera_T_robot_base'][2][3], 1.0)
            self.assertEqual(
                len(camera['em_overlay']['coil_poses']), 2)
            self.assertEqual(
                camera['em_overlay']['coil_poses'][0][
                    'position_robot_base_mm'], [-20.0, 0.0, 50.0])
            self.assertTrue((root / 'registration_left.png').is_file())
            self.assertTrue((root / 'registration_right.png').is_file())
            overlay = cv2.imread(str(root / 'registration_left.png'))
            self.assertIsNotNone(overlay)
            self.assertTrue(np.any(np.all(
                overlay == np.array([255, 0, 255], dtype=np.uint8), axis=2)))
            self.assertTrue(np.any(np.all(
                overlay == np.array([0, 255, 255], dtype=np.uint8), axis=2)))


if __name__ == '__main__':
    unittest.main()
