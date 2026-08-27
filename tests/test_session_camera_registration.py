import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from shape_tracking import boards as boards_mod
from shape_tracking.register import save_session_camera_registration
from shape_tracking.session import load_session_registration


class SessionCameraRegistrationTest(unittest.TestCase):
    def test_image_only_registration_uses_base_board_without_em(self):
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
                'field_generator_registration:\n'
                '  board_index: 2\n'
                '  marker_id_offset: 17\n'
                '  aurora_T_marker:\n'
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
            observations = {
                0: {
                    'r': [[0.0, 0.0, 0.0]] * 10,
                    't': [[0.0, 0.0, 1.0]] * 10,
                    'c': [9] * 10,
                }
            }
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            K = np.array([
                [200.0, 0.0, 160.0],
                [0.0, 200.0, 120.0],
                [0.0, 0.0, 1.0],
            ])
            _, boards = boards_mod.build_boards()
            registration_path = root / 'registration.json'
            right_camera_T_left_camera = np.eye(4)
            right_camera_T_left_camera[0, 3] = -0.2

            camera = save_session_camera_registration(
                registration_path=str(registration_path),
                output_dir=str(root), config_path=str(config),
                collected=observations, left_bgr=image, right_bgr=image,
                boards=boards, K=K, dist=np.zeros(5), resolution='TEST',
                zed_serial='123', baseline_m=0.12,
                image_timestamp_ns=100, min_frames=10,
                image_only=True, K_right=K.copy(), dist_right=np.zeros(5),
                right_camera_T_left_camera=right_camera_T_left_camera,
                overlay_prefix='primary')

            document = json.loads(registration_path.read_text())
            self.assertEqual(document['mode'], 'image_only')
            self.assertEqual(document['modalities'], {
                'camera': True, 'em': False})
            self.assertIsNone(document['em'])
            self.assertEqual(camera['boards_used'], [0])
            self.assertIsNone(camera['field_generator_board'])
            self.assertIsNone(camera['em_overlay'])
            self.assertEqual(
                camera['right_camera_T_left_camera'][0][3], -0.2)
            self.assertTrue((root / 'registration_primary_left.png').is_file())
            self.assertTrue((root / 'registration_primary_right.png').is_file())
            with self.assertRaisesRegex(ValueError, 'EM registration'):
                load_session_registration(root)
            loaded = load_session_registration(root, require_em=False)
            self.assertIsNone(loaded.base_T_aurora)
            self.assertIsNone(loaded.aurora_T_base)

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

    def test_solves_aurora_transform_from_field_charuco_board(self):
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
                'field_generator_registration:\n'
                '  board_index: 2\n'
                '  marker_id_offset: 17\n'
                '  aurora_T_marker:\n'
                '    matrix:\n'
                '      - [1, 0, 0, 0]\n'
                '      - [0, 1, 0, 0]\n'
                '      - [0, 0, 1, 0]\n'
                '      - [0, 0, 0, 1]\n'
                'workspace:\n'
                '  x: [-0.05, 0.05]\n'
                '  y: [-0.05, 0.05]\n'
                '  z: [0.0, 0.5]\n'
                '  margin_px: 5\n')
            observations = {
                0: {
                    'r': [[0.0, 0.0, 0.0]] * 10,
                    't': [[0.0, 0.0, 1.0]] * 10,
                    'c': [9] * 10,
                },
                2: {
                    'r': [[0.0, 0.0, 0.0]] * 10,
                    't': [[0.1, 0.2, 1.3]] * 10,
                    'c': [9] * 10,
                },
            }
            K = np.array([
                [200.0, 0.0, 160.0],
                [0.0, 200.0, 120.0],
                [0.0, 0.0, 1.0],
            ])
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            _, boards = boards_mod.build_boards(additional=[(2, 17)])
            em_tool_poses = [
                {
                    'tool_role': f'tip_coil_{index}',
                    'part_number': part,
                    'serial_number': str(index),
                    'timestamp_ns': 100,
                    'timestamp_delta_ms': 0.0,
                    'position_aurora_mm': [index * 5.0, 0.0, 0.0],
                    'quaternion_aurora_wxyz': [1.0, 0.0, 0.0, 0.0],
                }
                for index, part in enumerate(('003', '07222026_01'))
            ]

            save_session_camera_registration(
                registration_path=str(root / 'registration.json'),
                output_dir=str(root), config_path=str(config),
                collected=observations, left_bgr=image, right_bgr=image,
                boards=boards, K=K, dist=np.zeros(5), resolution='TEST',
                zed_serial='123', baseline_m=0.12,
                image_timestamp_ns=100, min_frames=10,
                em_tool_poses=em_tool_poses)

            document = json.loads(
                (root / 'registration.json').read_text())
            self.assertEqual(
                document['em']['method'],
                'optical_charuco_field_generator')
            transform = np.asarray(
                document['em']['robot_base_T_aurora'])
            np.testing.assert_allclose(
                transform[:3, 3], [100.0, 200.0, 300.0], atol=1e-9)
            self.assertEqual(
                document['em']['field_generator_marker']['marker_ids'],
                list(range(17, 25)))
            self.assertEqual(
                document['camera']['boards_used'], [0, 2])


if __name__ == '__main__':
    unittest.main()
