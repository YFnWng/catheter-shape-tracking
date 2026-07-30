import csv
import json
import math
import struct
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from shape_tracking.aurora_capture import (
    AuroraRecorder,
    load_em_slot_centers,
    qualify_stationary_probe_samples,
    solve_rigid_registration,
)


class FakeSerial:
    is_open = True
    timeout = 10.0

    def close(self):
        self.is_open = False

    def reset_input_buffer(self):
        pass


class FakeHandle:
    SENSOR_STATUS_VALID = '01'

    def __init__(self, index):
        self.id = f'{10 + index:02X}'
        self.sensor_status = '01'
        self.trans = [1.0 + index, 2.0, 3.0]
        self.rot = [1.0, 0.0, 0.0, 0.0]
        self.error = 0.2
        self.frame_number = 100
        self.occupied = self.initialized = self.enabled = True


class FakeDriver:
    tool_count = 3
    identities = [
        ('NDI-MF2', 'NDI', '000', '3EA21800', '610175   T6E0-S00923'),
        ('BM2', 'NDI', '001', '3F332803', '07222026_01'),
        ('BM2Lab', 'NDI', '001', '3E3BA801', '003'),
    ]

    def __init__(self, port):
        self.port = port
        self.n_port_handles = self.tool_count
        self.port_handles = [
            FakeHandle(i) for i in range(self.n_port_handles)]
        self.serial_port = FakeSerial()
        self.stopped = False

    def init(self):
        return 1

    def detect_and_assign_port_handles(self):
        pass

    def init_port_handle_all(self):
        pass

    def enable_port_handle_dynamic_all(self):
        pass

    def phinf(self, handle, option):
        index = int(handle, 16) - 10
        tool_type, manufacturer, revision, serial, part = self.identities[index]
        if option == '0001':
            return (
                f'{tool_type:<8}{manufacturer:<12}{revision:<3}{serial:<8}ABCD')
        if option == '0004':
            return f'{part:<20}ABCD'
        raise AssertionError(option)

    def start_tracking(self, fast):
        self.fast = fast

    def update_sensor_data_all(self):
        time.sleep(0.001)

    def stop_tracking(self):
        self.stopped = True


class OneToolDriver(FakeDriver):
    tool_count = 1


class TwoCoilDriver(FakeDriver):
    tool_count = 2
    identities = FakeDriver.identities[1:]


class OneShortFrameDriver(FakeDriver):
    short_frames_remaining = 1

    def __init__(self, port):
        super().__init__(port)
        self.short_frames_remaining = 1

    def update_sensor_data_all(self):
        if self.short_frames_remaining:
            self.short_frames_remaining -= 1
            raise struct.error('unpack requires a buffer of 4 bytes')
        super().update_sensor_data_all()


class AuroraRecorderTest(unittest.TestCase):
    def test_records_batched_pose_rows_and_metadata(self):
        with TemporaryDirectory() as tmp:
            recorder = AuroraRecorder(
                'COM_TEST', driver_root='unused', expected_tools=3,
                em_registration_dwell_s=0.0,
                em_registration_min_samples=5,
                driver_factory=FakeDriver).open()
            recorder.start(tmp)
            deadline = time.monotonic() + 1.0
            while recorder.samples < 10 and time.monotonic() < deadline:
                time.sleep(0.005)

            coil_poses = recorder.tip_coil_poses_at(time.time_ns())
            self.assertEqual(len(coil_poses), 2)
            self.assertEqual(
                {pose['part_number'] for pose in coil_poses},
                {'003', '07222026_01'})
            histories = recorder.recent_tip_coil_samples()
            self.assertEqual(len(histories), 2)
            self.assertTrue(all(item['samples'] for item in histories))

            for slot in (1, 2, 3, 4):
                recorder.capture_registration_slot(slot)
            recorder.stop()

            with open(Path(tmp) / 'em_poses.csv', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreaterEqual(len(rows), 6)
            self.assertEqual(
                {row['tool_role'] for row in rows},
                {'base_probe', 'tip_coil_0', 'tip_coil_1'})
            self.assertEqual(rows[0]['qw'], '1.0')
            self.assertEqual(rows[0]['valid'], '1')

            document = json.loads(
                (Path(tmp) / 'registration.json').read_text())
            registration = document['em']
            self.assertIsNone(document['camera'])
            self.assertTrue(registration['complete'])
            self.assertEqual(
                registration['transform_status'], 'awaiting_valid_config')
            self.assertEqual(
                registration['probe_identity']['part_number'],
                '610175   T6E0-S00923')
            self.assertTrue(
                registration['slots']['1']['stationarity']['passed'])

            metadata = json.loads(
                (Path(tmp) / 'em_metadata.json').read_text())
            self.assertEqual(metadata['tool_count'], 3)
            self.assertEqual(len(metadata['tools']), 3)
            self.assertGreaterEqual(metadata['hardware_frames'], 3)
            self.assertIsNone(metadata['error'])
            recorder.close()

    def test_stationarity_gate_accepts_static_probe(self):
        samples = [
            {
                'timestamp_ns': index * 50_000_000,
                'position_mm': [1.0 + 0.01 * (index % 2), 2.0, 3.0],
                'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
            }
            for index in range(21)
        ]
        selected, diagnostics = qualify_stationary_probe_samples(
            samples, window_s=0.75, min_samples=10,
            max_position_deviation_mm=0.15,
            max_orientation_deviation_deg=1.0)
        self.assertGreaterEqual(len(selected), 10)
        self.assertTrue(diagnostics['passed'])
        self.assertLess(diagnostics['position_p95_deviation_mm'], 0.15)

    def test_stationarity_gate_rejects_probe_motion(self):
        samples = [
            {
                'timestamp_ns': index * 50_000_000,
                'position_mm': [0.05 * index, 0.0, 0.0],
                'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
            }
            for index in range(21)
        ]
        with self.assertRaisesRegex(RuntimeError, 'probe is not stationary'):
            qualify_stationary_probe_samples(
                samples, window_s=0.75, min_samples=10,
                max_position_deviation_mm=0.15,
                max_orientation_deviation_deg=1.0)

    def test_stationarity_gate_rejects_orientation_motion(self):
        samples = []
        for index in range(21):
            angle = math.radians(index * 0.5)
            samples.append({
                'timestamp_ns': index * 50_000_000,
                'position_mm': [0.0, 0.0, 0.0],
                'quaternion_wxyz': [
                    math.cos(angle / 2.0), 0.0, 0.0,
                    math.sin(angle / 2.0),
                ],
            })
        with self.assertRaisesRegex(RuntimeError, 'orientation p95'):
            qualify_stationary_probe_samples(
                samples, window_s=0.75, min_samples=10,
                max_position_deviation_mm=0.15,
                max_orientation_deviation_deg=1.0)

    def test_rejects_wrong_tool_count(self):
        with self.assertRaisesRegex(
                RuntimeError, 'expected 3 Aurora tools, found 1'):
            AuroraRecorder(
                'COM_TEST', driver_root='unused', expected_tools=3,
                driver_factory=OneToolDriver).open()

    def test_optical_registration_allows_probe_to_be_unplugged(self):
        recorder = AuroraRecorder(
            'COM_TEST', driver_root='unused', expected_tools=2,
            require_probe=False, driver_factory=TwoCoilDriver).open()
        self.assertEqual(
            {item['role'] for item in recorder.tool_info.values()},
            {'tip_coil_0', 'tip_coil_1'})
        recorder.close()

    def test_recovers_from_one_short_binary_frame(self):
        with TemporaryDirectory() as tmp:
            recorder = AuroraRecorder(
                'COM_TEST', driver_root='unused', expected_tools=3,
                driver_factory=OneShortFrameDriver).open()
            recorder.start(tmp)
            deadline = time.monotonic() + 1.0
            while recorder.samples < 3 and time.monotonic() < deadline:
                time.sleep(0.005)
            recorder.stop()
            self.assertGreaterEqual(recorder.samples, 3)
            self.assertEqual(recorder.dropped_frame_reads, 1)
            self.assertIsNone(recorder.error)
            recorder.close()

    def test_loads_slot_centers_and_solves_rigid_transform(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'registration_config.yaml'
            config_path.write_text(
                'units: m\n'
                'em_registration:\n'
                '  slot_centers_base:\n'
                '    1: [0.100, 0.200, 0.300]\n'
                '    2: [0.110, 0.200, 0.300]\n'
                '    3: [0.100, 0.210, 0.300]\n'
                '    4: [0.110, 0.210, 0.300]\n')
            centers = load_em_slot_centers(str(config_path))
            self.assertEqual(centers['1'], [100.0, 200.0, 300.0])

            aurora = [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [10.0, 10.0, 0.0],
            ]
            base = [centers[str(slot)] for slot in (1, 2, 3, 4)]
            fit = solve_rigid_registration(aurora, base)
            self.assertLess(fit['rms_residual_mm'], 1e-10)
            self.assertAlmostEqual(
                fit['robot_base_T_aurora'][0][3], 100.0)
            self.assertAlmostEqual(
                fit['robot_base_T_aurora'][1][3], 200.0)
            self.assertAlmostEqual(
                fit['robot_base_T_aurora'][2][3], 300.0)

    def test_four_slots_write_solved_transform_to_session(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'registration_config.yaml'
            config_path.write_text(
                'units: mm\n'
                'em_registration:\n'
                '  slot_centers_base:\n'
                '    1: [100, 200, 300]\n'
                '    2: [110, 200, 300]\n'
                '    3: [100, 210, 300]\n'
                '    4: [110, 210, 300]\n')
            recorder = AuroraRecorder(
                'COM_TEST', driver_root='unused', expected_tools=3,
                registration_config=str(config_path),
                driver_factory=FakeDriver).open()
            recorder._output_dir = tmp
            measured = (
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [10.0, 10.0, 0.0],
            )
            for slot, point in enumerate(measured, start=1):
                recorder._update_registration_file(
                    slot, {'slot': slot, 'mean_aurora_mm': point})

            document = json.loads(
                (Path(tmp) / 'registration.json').read_text())
            registration = document['em']
            self.assertEqual(registration['transform_status'], 'solved')
            self.assertLess(
                registration['registration_fit']['rms_residual_mm'], 1e-10)
            self.assertEqual(
                registration['bracket_slot_positions_base_mm']['1'],
                [100.0, 200.0, 300.0])
            recorder.close()


if __name__ == '__main__':
    unittest.main()
