'''Independent, maximum-rate Aurora pose recording worker.

The NDI serial driver is reused from the SlicerRobot source tree, but this
module runs in ordinary Python and does not import or launch 3D Slicer.
'''
from __future__ import annotations

import csv
from collections import deque
import importlib
import json
import os
import statistics
import struct
import sys
import threading
import time

import numpy as np
import yaml


CSV_FIELDS = [
    'timestamp_ns', 'request_timestamp_ns', 'read_duration_ns', 'sample_index',
    'tool_index', 'tool_role', 'port_handle', 'part_number', 'serial_number',
    'manufacturer', 'revision', 'tool_type',
    'aurora_frame_number', 'sensor_status', 'valid',
    'tx_mm', 'ty_mm', 'tz_mm', 'qw', 'qx', 'qy', 'qz', 'error_mm',
    'handle_status',
]


def load_em_slot_centers(config_path: str) -> dict:
    '''Load four bracket slot centers and return them in millimetres.'''
    path = os.path.abspath(os.path.expanduser(config_path))
    if not os.path.isfile(path):
        raise ValueError(f'registration config not found: {path}')
    with open(path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    units = str(config.get('units', 'mm')).lower()
    if units not in ('mm', 'm'):
        raise ValueError("registration config units must be 'mm' or 'm'")
    raw = (config.get('em_registration') or {}).get('slot_centers_base')
    if not isinstance(raw, dict):
        raise ValueError(
            'registration config must define '
            'em_registration.slot_centers_base for slots 1..4')
    scale = 1000.0 if units == 'm' else 1.0
    centers = {}
    for slot in (1, 2, 3, 4):
        value = raw.get(slot, raw.get(str(slot)))
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(
                f'em_registration.slot_centers_base.{slot} must be [x, y, z]')
        point = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(point)):
            raise ValueError(
                f'em_registration.slot_centers_base.{slot} must contain '
                'three finite coordinates')
        centers[str(slot)] = (point * scale).tolist()
    return centers


def solve_rigid_registration(aurora_points_mm, base_points_mm) -> dict:
    '''Fit p_base = R @ p_aurora + t using the Kabsch algorithm.'''
    source = np.asarray(aurora_points_mm, dtype=float)
    target = np.asarray(base_points_mm, dtype=float)
    if source.shape != (4, 3) or target.shape != (4, 3):
        raise ValueError('rigid registration requires four 3D point pairs')
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    if np.linalg.matrix_rank(source_centered) < 2:
        raise ValueError('measured registration slots are collinear or coincident')
    u, singular_values, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    predicted = (rotation @ source.T).T + translation
    residuals = np.linalg.norm(predicted - target, axis=1)

    base_from_aurora = np.eye(4)
    base_from_aurora[:3, :3] = rotation
    base_from_aurora[:3, 3] = translation
    aurora_from_base = np.linalg.inv(base_from_aurora)
    return {
        'robot_base_T_aurora': base_from_aurora.tolist(),
        'aurora_T_robot_base': aurora_from_base.tolist(),
        'rms_residual_mm': float(np.sqrt(np.mean(residuals ** 2))),
        'max_residual_mm': float(np.max(residuals)),
        'per_slot_residual_mm': {
            str(slot): float(residuals[slot - 1]) for slot in (1, 2, 3, 4)},
        'singular_values': singular_values.tolist(),
        'rotation_determinant': float(np.linalg.det(rotation)),
    }


def qualify_stationary_probe_samples(
        samples, window_s: float, min_samples: int,
        max_position_deviation_mm: float,
        max_orientation_deviation_deg: float) -> tuple[list, dict]:
    '''Select a recent probe window and reject it unless it is stationary.'''
    window_s = float(window_s)
    min_samples = int(min_samples)
    max_position_deviation_mm = float(max_position_deviation_mm)
    max_orientation_deviation_deg = float(max_orientation_deviation_deg)
    if window_s < 0:
        raise ValueError('registration stationarity window must be non-negative')
    if min_samples < 2:
        raise ValueError('registration stationarity requires at least 2 samples')
    if max_position_deviation_mm <= 0:
        raise ValueError(
            'registration position-deviation threshold must be positive')
    if max_orientation_deviation_deg <= 0:
        raise ValueError(
            'registration orientation-deviation threshold must be positive')
    if not samples:
        raise RuntimeError('no valid base_probe readings are available')

    samples = sorted(samples, key=lambda sample: sample['timestamp_ns'])
    end_ns = samples[-1]['timestamp_ns']
    cutoff_ns = end_ns - int(window_s * 1e9)
    if window_s > 0 and samples[0]['timestamp_ns'] > cutoff_ns:
        available_s = (end_ns - samples[0]['timestamp_ns']) * 1e-9
        raise RuntimeError(
            'probe has not been observed for the complete stationary dwell: '
            f'available={available_s:.3f}s required={window_s:.3f}s')
    selected = (
        samples if window_s == 0 else [
            sample for sample in samples
            if sample['timestamp_ns'] >= cutoff_ns])
    if len(selected) < min_samples:
        raise RuntimeError(
            f'only {len(selected)} valid probe samples in the '
            f'{window_s:.3f}s stationary window; require {min_samples}')

    positions = np.asarray(
        [sample['position_mm'] for sample in selected], dtype=float)
    quaternions = np.asarray(
        [sample['quaternion_wxyz'] for sample in selected], dtype=float)
    if (positions.shape != (len(selected), 3)
            or quaternions.shape != (len(selected), 4)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(quaternions))):
        raise RuntimeError('stationary probe window contains invalid pose data')

    position_center = np.median(positions, axis=0)
    position_deviation = np.linalg.norm(
        positions - position_center[None, :], axis=1)
    position_p95 = float(np.percentile(position_deviation, 95))
    position_max = float(np.max(position_deviation))

    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    if np.any(quaternion_norms < 1e-12):
        raise RuntimeError(
            'stationary probe window contains a zero-norm quaternion')
    quaternions = quaternions / quaternion_norms[:, None]
    reference = quaternions[0]
    signs = np.where(quaternions @ reference < 0.0, -1.0, 1.0)
    aligned = quaternions * signs[:, None]
    quaternion_center = np.sum(aligned, axis=0)
    center_norm = np.linalg.norm(quaternion_center)
    if center_norm < 1e-12:
        raise RuntimeError(
            'stationary probe window has no stable mean orientation')
    quaternion_center /= center_norm
    orientation_deviation = np.degrees(
        2.0 * np.arccos(np.clip(
            np.abs(aligned @ quaternion_center), 0.0, 1.0)))
    orientation_p95 = float(np.percentile(orientation_deviation, 95))
    orientation_max = float(np.max(orientation_deviation))

    diagnostics = {
        'passed': (
            position_p95 <= max_position_deviation_mm
            and orientation_p95 <= max_orientation_deviation_deg),
        'requested_window_s': window_s,
        'observed_window_s': (
            selected[-1]['timestamp_ns']
            - selected[0]['timestamp_ns']) * 1e-9,
        'min_samples': min_samples,
        'sample_count': len(selected),
        'position_center_median_mm': position_center.tolist(),
        'position_p95_deviation_mm': position_p95,
        'position_max_deviation_mm': position_max,
        'max_position_deviation_mm': max_position_deviation_mm,
        'orientation_center_wxyz': quaternion_center.tolist(),
        'orientation_p95_deviation_deg': orientation_p95,
        'orientation_max_deviation_deg': orientation_max,
        'max_orientation_deviation_deg': max_orientation_deviation_deg,
    }
    failures = []
    if position_p95 > max_position_deviation_mm:
        failures.append(
            f'position p95={position_p95:.3f} mm '
            f'(limit {max_position_deviation_mm:.3f} mm)')
    if orientation_p95 > max_orientation_deviation_deg:
        failures.append(
            f'orientation p95={orientation_p95:.3f} deg '
            f'(limit {max_orientation_deviation_deg:.3f} deg)')
    if failures:
        raise RuntimeError(
            'probe is not stationary: ' + '; '.join(failures))
    return selected, diagnostics


def load_aurora_driver(slicer_robot_root: str):
    '''Load AuroraDriver without loading any Slicer application modules.'''
    root = os.path.abspath(os.path.expanduser(slicer_robot_root))
    driver_file = os.path.join(
        root, 'AuroraTracker', 'Scripts', 'aurora_driver.py')
    if not os.path.isfile(driver_file):
        raise FileNotFoundError(f'Aurora driver not found: {driver_file}')
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module(
        'AuroraTracker.Scripts.aurora_driver')
    return module.AuroraDriver


class AuroraRecorder:
    '''Own the Aurora serial device and write poses on a worker thread.'''

    def __init__(self, port: str, driver_root: str, expected_tools: int = 3,
                 probe_part_number: str = '610175 T6E0-S00923',
                 registration_config: str | None = None,
                 em_registration_dwell_s: float = 0.75,
                 em_registration_min_samples: int = 20,
                 em_registration_max_position_deviation_mm: float = 0.15,
                 em_registration_max_orientation_deviation_deg: float = 1.0,
                 driver_factory=None):
        self.port = port
        self.driver_root = driver_root
        self.expected_tools = int(expected_tools)
        self.probe_part_number = probe_part_number
        self.registration_config = (
            os.path.abspath(os.path.expanduser(registration_config))
            if registration_config else None)
        self.em_registration_dwell_s = float(em_registration_dwell_s)
        self.em_registration_min_samples = int(em_registration_min_samples)
        self.em_registration_max_position_deviation_mm = float(
            em_registration_max_position_deviation_mm)
        self.em_registration_max_orientation_deviation_deg = float(
            em_registration_max_orientation_deviation_deg)
        if self.em_registration_dwell_s < 0:
            raise ValueError('EM registration dwell must be non-negative')
        if self.em_registration_min_samples < 2:
            raise ValueError('EM registration requires at least 2 samples')
        if self.em_registration_max_position_deviation_mm <= 0:
            raise ValueError(
                'EM registration position-deviation limit must be positive')
        if self.em_registration_max_orientation_deviation_deg <= 0:
            raise ValueError(
                'EM registration orientation-deviation limit must be positive')
        self._factory = driver_factory
        self._device = None
        self._tracking = False
        self._thread = None
        self._stop = threading.Event()
        self._data_lock = threading.Lock()
        self._recent_probe = deque(maxlen=200)
        self._recent_tools = {}
        self._output_dir = None
        self.tool_info = {}
        self.error = None
        self.samples = 0
        self.rows = 0
        self.valid_rows = 0
        self.dropped_frame_reads = 0
        self.started_ns = None
        self.stopped_ns = None

    @property
    def tool_count(self) -> int:
        return self._device.n_port_handles if self._device is not None else 0

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def open(self):
        factory = self._factory or load_aurora_driver(self.driver_root)
        self._device = factory(self.port)
        if self._device.init() != 1:
            raise RuntimeError('Aurora initialization failed')
        self._device.detect_and_assign_port_handles()
        self._device.init_port_handle_all()
        found = self.tool_count
        if self.expected_tools and found != self.expected_tools:
            self.close()
            raise RuntimeError(
                f'expected {self.expected_tools} Aurora tools, found {found}')
        self.tool_info = self._query_tool_info()
        self._device.enable_port_handle_dynamic_all()
        # TSTART can take several seconds. A short timeout here leaves its
        # delayed ASCII OKAY reply queued ahead of the first binary BX frame.
        self._device.serial_port.timeout = 10.0
        self._device.start_tracking(True)
        self._tracking = True
        # BX frames are fast at the configured baud rate. Clear only stale
        # startup bytes before polling; no BX request has been sent yet.
        self._device.serial_port.reset_input_buffer()
        self._device.serial_port.timeout = 2.0
        return self

    @staticmethod
    def _phinf_payload(reply: str) -> str:
        '''Remove the four-character CRC suffix from an ASCII PHINF reply.'''
        return reply[:-4] if len(reply) >= 4 else reply

    @staticmethod
    def _identity_key(value: str) -> str:
        return ''.join(c for c in value.upper() if c.isalnum())

    def _query_tool_info(self) -> dict:
        info = {}
        for ph in self._device.port_handles:
            basic_raw = self._device.phinf(ph.id, '0001')
            part_raw = self._device.phinf(ph.id, '0004')
            basic = self._phinf_payload(basic_raw)
            part = self._phinf_payload(part_raw).strip()
            info[ph.id] = {
                'port_handle': ph.id,
                'tool_type': basic[0:8].strip(),
                'manufacturer': basic[8:20].strip(),
                'revision': basic[20:23].strip(),
                'serial_number': basic[23:31].strip(),
                'part_number': part,
                'phinf_basic_raw': basic_raw,
                'phinf_part_raw': part_raw,
            }

        probe_key = self._identity_key(self.probe_part_number)
        probe_handles = [
            handle for handle, item in info.items()
            if probe_key and probe_key in self._identity_key(item['part_number'])
        ]
        if len(probe_handles) != 1:
            found = {h: item['part_number'] for h, item in info.items()}
            raise RuntimeError(
                f'expected exactly one probe matching {self.probe_part_number!r}; '
                f'found {probe_handles}, tool parts={found}')
        info[probe_handles[0]]['role'] = 'base_probe'

        coils = sorted(
            (item for handle, item in info.items() if handle not in probe_handles),
            key=lambda item: (item['serial_number'], item['part_number']))
        if len(coils) != 2:
            raise RuntimeError(f'expected two tip coils, found {len(coils)}')
        for index, item in enumerate(coils):
            item['role'] = f'tip_coil_{index}'
        return info

    def start(self, output_dir: str) -> None:
        if self._device is None:
            raise RuntimeError('Aurora recorder is not open')
        if self.is_recording:
            return
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.error = None
        self.samples = self.rows = self.valid_rows = 0
        self.dropped_frame_reads = 0
        self.started_ns = time.time_ns()
        self.stopped_ns = None
        self._stop.clear()
        with self._data_lock:
            self._recent_probe.clear()
            self._recent_tools = {
                handle: deque(maxlen=200) for handle in self.tool_info}
        self._thread = threading.Thread(
            target=self._record_loop, name='aurora-recorder', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            raise RuntimeError('Aurora recording worker did not stop')
        self._thread = None
        self.stopped_ns = time.time_ns()
        self._write_metadata()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            if self._device is not None:
                try:
                    if self._tracking:
                        self._device.stop_tracking()
                finally:
                    self._tracking = False
                    serial_port = getattr(self._device, 'serial_port', None)
                    if serial_port is not None and serial_port.is_open:
                        serial_port.close()
                    self._device = None

    def _record_loop(self) -> None:
        path = os.path.join(self._output_dir, 'em_poses.csv')
        try:
            with open(path, 'w', newline='', buffering=1) as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                while not self._stop.is_set():
                    request_ns = time.time_ns()
                    self._read_frame_with_recovery()
                    timestamp_ns = time.time_ns()
                    sample_index = self.samples
                    for tool_index, ph in enumerate(self._device.port_handles):
                        identity = self.tool_info[ph.id]
                        valid = ph.sensor_status == ph.SENSOR_STATUS_VALID
                        nan = float('nan')
                        trans = ph.trans if valid else [nan] * 3
                        rot = ph.rot if valid else [nan] * 4
                        writer.writerow({
                            'timestamp_ns': timestamp_ns,
                            'request_timestamp_ns': request_ns,
                            'read_duration_ns': timestamp_ns - request_ns,
                            'sample_index': sample_index,
                            'tool_index': tool_index,
                            'tool_role': identity['role'],
                            'port_handle': ph.id,
                            'part_number': identity['part_number'],
                            'serial_number': identity['serial_number'],
                            'manufacturer': identity['manufacturer'],
                            'revision': identity['revision'],
                            'tool_type': identity['tool_type'],
                            'aurora_frame_number': ph.frame_number,
                            'sensor_status': ph.sensor_status,
                            'valid': int(valid),
                            'tx_mm': trans[0], 'ty_mm': trans[1],
                            'tz_mm': trans[2],
                            'qw': rot[0], 'qx': rot[1],
                            'qy': rot[2], 'qz': rot[3],
                            'error_mm': ph.error if valid else nan,
                            'handle_status': self._handle_status(ph),
                        })
                        self.rows += 1
                        self.valid_rows += int(valid)
                        if valid:
                            recent_pose = {
                                'timestamp_ns': timestamp_ns,
                                'aurora_frame_number': ph.frame_number,
                                'position_mm': list(ph.trans),
                                'quaternion_wxyz': list(ph.rot),
                                'error_mm': ph.error,
                            }
                            with self._data_lock:
                                self._recent_tools[ph.id].append(recent_pose)
                                if identity['role'] == 'base_probe':
                                    self._recent_probe.append(recent_pose)
                    self.samples += 1
        except BaseException as exc:
            self.error = exc
            self._stop.set()

    def _read_frame_with_recovery(self, max_attempts: int = 3) -> None:
        '''Retry short/misaligned serial frames without stopping acquisition.'''
        for attempt in range(1, max_attempts + 1):
            try:
                self._device.update_sensor_data_all()
                return
            except struct.error as exc:
                self.dropped_frame_reads += 1
                serial_port = self._device.serial_port
                serial_port.reset_input_buffer()
                if attempt == max_attempts:
                    raise RuntimeError(
                        f'Aurora BX frame remained incomplete after '
                        f'{max_attempts} attempts') from exc
                time.sleep(0.01)

    def capture_registration_slot(
            self, slot: int, window_s: float | None = None) -> dict:
        '''Save a robust static probe measurement for bracket slot 1 through 4.'''
        if slot not in (1, 2, 3, 4):
            raise ValueError('registration slot must be 1, 2, 3, or 4')
        if not self.is_recording or self._output_dir is None:
            raise RuntimeError('start unified recording before capturing registration')
        with self._data_lock:
            samples = list(self._recent_probe)
        window_s = (
            self.em_registration_dwell_s if window_s is None
            else float(window_s))
        samples, stationarity = qualify_stationary_probe_samples(
            samples,
            window_s=window_s,
            min_samples=self.em_registration_min_samples,
            max_position_deviation_mm=(
                self.em_registration_max_position_deviation_mm),
            max_orientation_deviation_deg=(
                self.em_registration_max_orientation_deviation_deg),
        )

        axes = list(zip(*(sample['position_mm'] for sample in samples)))
        mean_mm = [statistics.fmean(axis) for axis in axes]
        std_mm = [statistics.pstdev(axis) for axis in axes]
        result = {
            'slot': slot,
            'captured_at_ns': time.time_ns(),
            'window_start_ns': samples[0]['timestamp_ns'],
            'window_end_ns': samples[-1]['timestamp_ns'],
            'sample_count': len(samples),
            'mean_aurora_mm': mean_mm,
            'std_aurora_mm': std_mm,
            'max_std_mm': max(std_mm),
            'mean_error_mm': statistics.fmean(s['error_mm'] for s in samples),
            'first_aurora_frame': samples[0]['aurora_frame_number'],
            'last_aurora_frame': samples[-1]['aurora_frame_number'],
            'stationarity': stationarity,
        }
        registration = self._update_registration_file(slot, result)
        result['registration_complete'] = registration['complete']
        result['transform_status'] = registration['transform_status']
        result['transform_error'] = registration.get('transform_error')
        result['registration_fit'] = registration.get('registration_fit')
        return result

    def tip_coil_poses_at(
            self, timestamp_ns: int, max_age_s: float = 0.25) -> list[dict]:
        '''Return the nearest valid pose of each tip coil to an image time.'''
        timestamp_ns = int(timestamp_ns)
        max_age_ns = int(float(max_age_s) * 1e9)
        if max_age_ns < 0:
            raise ValueError('maximum EM/image pose age must be non-negative')
        with self._data_lock:
            recent = {
                handle: list(samples)
                for handle, samples in self._recent_tools.items()}
        result = []
        coils = sorted(
            (item for item in self.tool_info.values()
             if item['role'].startswith('tip_coil_')),
            key=lambda item: self._identity_key(item['part_number']))
        for identity in coils:
            candidates = recent.get(identity['port_handle'], [])
            if not candidates:
                raise RuntimeError(
                    f"no recent pose for {identity['role']} "
                    f"({identity['part_number']})")
            pose = min(
                candidates,
                key=lambda item: abs(item['timestamp_ns'] - timestamp_ns))
            delta_ns = pose['timestamp_ns'] - timestamp_ns
            if abs(delta_ns) > max_age_ns:
                raise RuntimeError(
                    f"nearest pose for {identity['role']} "
                    f"({identity['part_number']}) is "
                    f'{abs(delta_ns) * 1e-6:.1f} ms from the image; '
                    f'limit={max_age_s * 1000.0:.1f} ms')
            result.append({
                'tool_role': identity['role'],
                'part_number': identity['part_number'],
                'serial_number': identity['serial_number'],
                'port_handle': identity['port_handle'],
                'timestamp_ns': pose['timestamp_ns'],
                'timestamp_delta_ms': delta_ns * 1e-6,
                'aurora_frame_number': pose['aurora_frame_number'],
                'position_aurora_mm': pose['position_mm'],
                'quaternion_aurora_wxyz': pose['quaternion_wxyz'],
                'error_mm': pose['error_mm'],
            })
        if len(result) != 2:
            raise RuntimeError(f'expected two tip coil poses, found {len(result)}')
        return result

    def _update_registration_file(self, slot: int, result: dict) -> dict:
        path = os.path.join(self._output_dir, 'registration.json')
        if os.path.isfile(path):
            with open(path) as stream:
                document = json.load(stream)
            registration = document['em']
        else:
            probe = next(
                item for item in self.tool_info.values()
                if item['role'] == 'base_probe')
            document = {
                'schema_version': 1,
                'session_id': os.path.basename(
                    os.path.normpath(self._output_dir)),
                'registration_config': self.registration_config,
                'em': None,
                'camera': None,
            }
            registration = {
                'frame': 'aurora',
                'probe_identity': probe,
                'slot_order': [1, 2, 3, 4],
                'slots': {},
                'bracket_slot_positions_base_mm': None,
                'robot_base_T_aurora': None,
                'aurora_T_robot_base': None,
                'registration_fit': None,
            }
            document['em'] = registration
        registration['slots'][str(slot)] = result
        registration['complete'] = all(
            str(index) in registration['slots'] for index in (1, 2, 3, 4))
        registration['completed_at_ns'] = (
            time.time_ns() if registration['complete'] else None)
        registration['transform_status'] = 'waiting_for_all_slots'
        registration['transform_error'] = None
        if registration['complete']:
            try:
                if not self.registration_config:
                    raise ValueError('no registration config path was provided')
                centers = load_em_slot_centers(self.registration_config)
                measured = [
                    registration['slots'][str(index)]['mean_aurora_mm']
                    for index in (1, 2, 3, 4)]
                known = [centers[str(index)] for index in (1, 2, 3, 4)]
                fit = solve_rigid_registration(measured, known)
                registration['bracket_slot_positions_base_mm'] = centers
                registration['robot_base_T_aurora'] = fit['robot_base_T_aurora']
                registration['aurora_T_robot_base'] = fit['aurora_T_robot_base']
                registration['registration_fit'] = {
                    key: value for key, value in fit.items()
                    if key not in ('robot_base_T_aurora', 'aurora_T_robot_base')}
                registration['transform_status'] = 'solved'
            except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
                registration['transform_status'] = 'awaiting_valid_config'
                registration['transform_error'] = str(exc)
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w') as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
        return registration

    @staticmethod
    def _handle_status(ph) -> str:
        bits = 0
        for name, bit in (
            ('occupied', 0x01), ('initialized', 0x10), ('enabled', 0x20),
            ('out_of_volume', 0x40), ('partial_out_of_volume', 0x80),
            ('sensor_broken', 0x100),
        ):
            if getattr(ph, name, False):
                bits |= bit
        return f'0x{bits:08X}'

    def _write_metadata(self) -> None:
        if self._output_dir is None:
            return
        duration_s = None
        if self.started_ns is not None and self.stopped_ns is not None:
            duration_s = (self.stopped_ns - self.started_ns) * 1e-9
        metadata = {
            'port': self.port,
            'driver_root': os.path.abspath(self.driver_root),
            'tool_count': self.tool_count,
            'expected_tools': self.expected_tools,
            'probe_part_number_match': self.probe_part_number,
            'registration_config': self.registration_config,
            'registration_stationarity': {
                'dwell_s': self.em_registration_dwell_s,
                'min_samples': self.em_registration_min_samples,
                'max_position_deviation_mm': (
                    self.em_registration_max_position_deviation_mm),
                'max_orientation_deviation_deg': (
                    self.em_registration_max_orientation_deviation_deg),
                'deviation_percentile': 95,
            },
            'tools': sorted(
                self.tool_info.values(), key=lambda item: item['role']),
            'started_ns': self.started_ns,
            'stopped_ns': self.stopped_ns,
            'duration_s': duration_s,
            'hardware_frames': self.samples,
            'pose_rows': self.rows,
            'valid_pose_rows': self.valid_rows,
            'dropped_frame_reads': self.dropped_frame_reads,
            'rate_hz': self.samples / duration_s if duration_s else None,
            'position_units': 'mm',
            'quaternion_order': 'qw,qx,qy,qz (NDI scalar-first)',
            'timestamp': (
                'Windows time.time_ns at receipt of complete batched BX frame'),
            'error': repr(self.error) if self.error is not None else None,
        }
        path = os.path.join(self._output_dir, 'em_metadata.json')
        with open(path, 'w') as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
