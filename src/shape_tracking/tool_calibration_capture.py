'''Interactive synchronized capture for a rigid ArUco plus dual-EM fixture.'''
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from . import boards as boards_mod
from . import register as camera_register
from . import registration
from .aurora_capture import AuroraRecorder, qualify_stationary_probe_samples
from .stereo_marker import (
    estimate_stereo_object_pose,
    estimate_stereo_marker_pose,
    rectified_right_T_left,
    square_object_points,
)


DEFAULT_OUTPUT_ROOT = Path(r'D:\robot-dev\catheter_sessions')
DEFAULT_DRIVER_ROOT = Path(
    r'D:\GaTech Dropbox\Yifan Wang\SlicerRobot')
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'registration_config.yaml'
POSE_CSV_FIELDS = [
    'timestamp_ns', 'target', 'detected', 'corner_count',
    'tx_m', 'ty_m', 'tz_m', 'qw', 'qx', 'qy', 'qz',
    'rx', 'ry', 'rz', 'reprojection_rmse_px']


def matrix_from_pose(position_m, quaternion_wxyz):
    transform = np.eye(4, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat([
        quaternion[1], quaternion[2], quaternion[3], quaternion[0]
    ]).as_matrix()
    transform[:3, 3] = np.asarray(position_m, dtype=np.float64)
    return transform


def pose_sample(timestamp_ns, rvec, tvec, reprojection_rmse_px=None):
    rotation = Rotation.from_rotvec(np.asarray(rvec).reshape(3))
    q = rotation.as_quat()
    return {
        'timestamp_ns': int(timestamp_ns),
        'position_mm': (
            np.asarray(tvec, dtype=np.float64).reshape(3) * 1000.0).tolist(),
        'quaternion_wxyz': [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
        'rvec': np.asarray(rvec, dtype=np.float64).reshape(3).tolist(),
        'reprojection_rmse_px': (
            None if reprojection_rmse_px is None
            else float(reprojection_rmse_px)),
    }


def summarize_stationary_samples(
        samples, timestamp_ns, window_s, min_samples,
        max_position_deviation_mm, max_orientation_deviation_deg,
        sample_label='pose'):
    cutoff_ns = int(timestamp_ns - window_s * 1e9)
    candidates = [
        sample for sample in samples
        if cutoff_ns <= sample['timestamp_ns'] <= timestamp_ns]
    if len(candidates) < min_samples:
        raise RuntimeError(
            f'only {len(candidates)} valid {sample_label} samples in the '
            f'recent {window_s:.3f}s window; require {min_samples}')
    try:
        selected, diagnostics = qualify_stationary_probe_samples(
            candidates, window_s=0.0, min_samples=min_samples,
            max_position_deviation_mm=max_position_deviation_mm,
            max_orientation_deviation_deg=max_orientation_deviation_deg)
    except RuntimeError as exc:
        raise RuntimeError(str(exc).replace('probe', sample_label)) from exc
    observed_s = (
        selected[-1]['timestamp_ns'] - selected[0]['timestamp_ns']) * 1e-9
    minimum_span_s = max(0.0, window_s * 0.8)
    if observed_s < minimum_span_s:
        raise RuntimeError(
            f'stationary history spans only {observed_s:.3f}s; '
            f'require {minimum_span_s:.3f}s')
    diagnostics['requested_window_s'] = float(window_s)
    diagnostics['observed_window_s'] = float(observed_s)
    positions = np.asarray(
        [sample['position_mm'] for sample in selected], dtype=np.float64)
    errors = [
        sample['reprojection_rmse_px'] for sample in selected
        if sample.get('reprojection_rmse_px') is not None]
    return {
        'window_start_ns': int(selected[0]['timestamp_ns']),
        'window_end_ns': int(selected[-1]['timestamp_ns']),
        'sample_count': len(selected),
        'position_mm': np.mean(positions, axis=0).tolist(),
        'quaternion_wxyz': diagnostics['orientation_center_wxyz'],
        'mean_reprojection_rmse_px': (
            float(np.mean(errors)) if errors else None),
        'max_reprojection_rmse_px': (
            float(np.max(errors)) if errors else None),
        'stationarity': diagnostics,
    }


class TipMarkerDetector:
    def __init__(self, marker_id, marker_size_m):
        self.marker_id = int(marker_id)
        self.marker_size_m = float(marker_size_m)
        if self.marker_size_m <= 0:
            raise ValueError('marker size must be positive')
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            boards_mod.get_dictionary(), params)

    def detect(self, gray, K, dist):
        corners, ids, rejected = self.detector.detectMarkers(gray)
        if ids is None:
            return None, corners, ids, rejected
        matches = [
            index for index, value in enumerate(ids.reshape(-1))
            if int(value) == self.marker_id]
        if len(matches) > 1:
            raise RuntimeError(
                f'multiple visible copies of marker ID {self.marker_id}')
        if not matches:
            return None, corners, ids, rejected
        image_points = np.asarray(
            corners[matches[0]], dtype=np.float32).reshape(4, 2)
        half = self.marker_size_m / 2.0
        object_points = np.asarray([
            [-half, +half, 0], [+half, +half, 0],
            [+half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        solved = cv2.solvePnPGeneric(
            object_points, image_points, K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = []
        if solved[0]:
            for rvec, tvec in zip(solved[1], solved[2]):
                tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
                if tvec[2, 0] <= 0:
                    continue
                projected, _ = cv2.projectPoints(
                    object_points, rvec, tvec, K, dist)
                residual = projected.reshape(4, 2) - image_points
                rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
                candidates.append((rmse, rvec, tvec))
        if not candidates:
            return None, corners, ids, rejected
        rmse, rvec, tvec = min(candidates, key=lambda item: item[0])
        return {
            'rvec': np.asarray(rvec).reshape(3, 1),
            'tvec': tvec,
            'corners': image_points,
            'reprojection_rmse_px': rmse,
        }, corners, ids, rejected

    def detect_stereo(
            self, left_gray, right_gray, K_left, dist_left,
            K_right, dist_right, right_T_left):
        left_pose, left_all, left_ids, _ = self.detect(
            left_gray, K_left, dist_left)
        right_pose, right_all, right_ids, _ = self.detect(
            right_gray, K_right, dist_right)
        if left_pose is None or right_pose is None:
            return None
        result = estimate_stereo_marker_pose(
            square_object_points(self.marker_size_m),
            left_pose['corners'], right_pose['corners'],
            K_left, dist_left, K_right, dist_right, right_T_left)
        return {
            'rvec': result['rvec'],
            'tvec': result['tvec'],
            'corners': result['left_corners'],
            'right_corners': result['right_corners'],
            'reprojection_rmse_px': result[
                'stereo_reprojection_rmse_px'],
            'left_reprojection_rmse_px': result[
                'left_reprojection_rmse_px'],
            'right_reprojection_rmse_px': result[
                'right_reprojection_rmse_px'],
            'rectified_row_mismatch_max_px': result[
                'rectified_row_mismatch_max_px'],
            'left_detected_corners': left_all,
            'left_detected_ids': left_ids,
            'right_detected_corners': right_all,
            'right_detected_ids': right_ids,
            'left_camera_T_marker_m': result[
                'left_camera_T_marker_m'],
        }


def estimate_stereo_charuco_pose(
        left_pose, right_pose, board_entry,
        K_left, dist_left, K_right, dist_right, right_T_left):
    """Estimate a board pose from all ChArUco corners in both cameras."""
    if left_pose is None or right_pose is None or not left_pose.has_pose:
        return None
    if left_pose.charuco_ids is None or right_pose.charuco_ids is None:
        return None
    left_ids = np.asarray(left_pose.charuco_ids, dtype=np.int32).reshape(-1)
    right_ids = np.asarray(right_pose.charuco_ids, dtype=np.int32).reshape(-1)
    if len(left_ids) < 4 or len(right_ids) < 4:
        return None
    chessboard_points = np.asarray(
        board_entry.board.getChessboardCorners(), dtype=np.float64)
    result = estimate_stereo_object_pose(
        chessboard_points[left_ids], left_pose.charuco_corners,
        chessboard_points[right_ids], right_pose.charuco_corners,
        K_left, dist_left, K_right, dist_right, right_T_left,
        left_pose.rvec, left_pose.tvec)
    result['left_corner_count'] = int(len(left_ids))
    result['right_corner_count'] = int(len(right_ids))
    return result


def summarize_coils(recorder, timestamp_ns, args):
    max_age_ns = int(args.max_em_image_age_ms * 1e6)
    summaries = []
    for tool in recorder.recent_tip_coil_samples():
        samples = [sample for sample in tool['samples']
                   if sample['timestamp_ns'] <= timestamp_ns + max_age_ns]
        if not samples:
            raise RuntimeError(
                'no valid samples for ' + tool['part_number'])
        nearest = min(samples, key=lambda sample:
                      abs(sample['timestamp_ns'] - timestamp_ns))
        delta_ns = nearest['timestamp_ns'] - timestamp_ns
        if abs(delta_ns) > max_age_ns:
            raise RuntimeError(
                'nearest {} sample is {:.1f}ms from image'.format(
                    tool['part_number'], abs(delta_ns) * 1e-6))
        summary = summarize_stationary_samples(
            samples, nearest['timestamp_ns'], args.stationary_window_s,
            args.em_min_samples, args.em_max_position_deviation_mm,
            args.em_max_orientation_deviation_deg,
            sample_label='EM ' + tool['part_number'])
        summary.update({key: tool[key] for key in (
            'tool_role', 'part_number', 'serial_number', 'port_handle')})
        summary['position_aurora_mm'] = summary.pop('position_mm')
        summary['quaternion_aurora_wxyz'] = summary.pop('quaternion_wxyz')
        summary['nearest_timestamp_ns'] = nearest['timestamp_ns']
        summary['nearest_timestamp_delta_ms'] = delta_ns * 1e-6
        selected_errors = [sample['error_mm'] for sample in samples
                           if summary['window_start_ns']
                           <= sample['timestamp_ns']
                           <= summary['window_end_ns']]
        summary['mean_error_mm'] = float(np.mean(selected_errors))
        summaries.append(summary)
    return summaries


def build_snapshot_geometry(tip_optical, coil_poses, camera_T_aurora_m):
    camera_T_tip = matrix_from_pose(
        np.asarray(tip_optical['position_mm']) * 0.001,
        tip_optical['quaternion_wxyz'])
    camera_T_aurora = np.asarray(camera_T_aurora_m, dtype=np.float64)
    aurora_T_tip = np.linalg.inv(camera_T_aurora) @ camera_T_tip
    coils = []
    for pose in coil_poses:
        aurora_T_coil = matrix_from_pose(
            np.asarray(pose['position_aurora_mm']) * 0.001,
            pose['quaternion_aurora_wxyz'])
        coil_T_tip = np.linalg.inv(aurora_T_coil) @ aurora_T_tip
        item = dict(pose)
        item['aurora_T_coil_m'] = aurora_T_coil.tolist()
        item['coil_T_tip_marker_m'] = coil_T_tip.tolist()
        item['coil_to_marker_translation_mm'] = (
            coil_T_tip[:3, 3] * 1000.0).tolist()
        coils.append(item)
    return {
        'transform_convention': 'parent_T_child; p_parent = T @ p_child',
        'matrix_translation_units': 'm',
        'left_camera_T_tip_marker_m': camera_T_tip.tolist(),
        'left_camera_T_aurora_m': camera_T_aurora.tolist(),
        'aurora_T_tip_marker_m': aurora_T_tip.tolist(),
        'aurora_tip_marker_position_mm': (
            aurora_T_tip[:3, 3] * 1000.0).tolist(),
        'coils': coils,
    }


def write_pose_row(writer, timestamp_ns, target, sample, corner_count=0):
    row = {
        'timestamp_ns': timestamp_ns, 'target': target,
        'detected': int(sample is not None), 'corner_count': corner_count}
    if sample is not None:
        p = np.asarray(sample['position_mm']) * 0.001
        q, r = sample['quaternion_wxyz'], sample['rvec']
        row.update({
            'tx_m': p[0], 'ty_m': p[1], 'tz_m': p[2],
            'qw': q[0], 'qx': q[1], 'qy': q[2], 'qz': q[3],
            'rx': r[0], 'ry': r[1], 'rz': r[2],
            'reprojection_rmse_px': sample.get('reprojection_rmse_px')})
    writer.writerow(row)


def build_field_frame_lock(field_summary, field_config, locked_at_ns):
    camera_T_field = matrix_from_pose(
        np.asarray(field_summary['position_mm']) * 0.001,
        field_summary['quaternion_wxyz'])
    camera_T_aurora = (
        camera_T_field @ np.linalg.inv(field_config['aurora_T_marker']))
    return {
        'locked_at_ns': int(locked_at_ns),
        'transform_convention': 'parent_T_child; p_parent = T @ p_child',
        'matrix_translation_units': 'm',
        'pose_method': 'multi-frame synchronized stereo ChArUco',
        'field_marker_optical_summary': field_summary,
        'left_camera_T_field_marker_m': camera_T_field.tolist(),
        'aurora_T_field_marker_m':
            field_config['aurora_T_marker'].tolist(),
        'left_camera_T_aurora_m': camera_T_aurora.tolist(),
        'valid_while': 'ZED camera and Aurora field generator remain fixed',
    }


def draw_live_coils(
        image, camera, recorder, camera_T_aurora_m, timestamp_ns,
        right_camera=False):
    if camera_T_aurora_m is None:
        return 0
    try:
        poses = recorder.tip_coil_poses_at(timestamp_ns, max_age_s=0.1)
    except RuntimeError:
        return 0
    camera_T_aurora = np.asarray(camera_T_aurora_m, dtype=np.float64)
    K, dist = camera.K, camera.dist
    if right_camera:
        camera_T_aurora = (
            rectified_right_T_left(camera.baseline_m) @ camera_T_aurora)
        K, dist = camera.K_right, camera.dist_right
    colors = [(255, 0, 255), (0, 255, 255)]
    for index, pose in enumerate(poses):
        aurora_T_coil = matrix_from_pose(
            np.asarray(pose['position_aurora_mm']) * 0.001,
            pose['quaternion_aurora_wxyz'])
        camera_T_coil = camera_T_aurora @ aurora_T_coil
        rvec = Rotation.from_matrix(camera_T_coil[:3, :3]).as_rotvec()
        points, _ = cv2.projectPoints(
            np.asarray([[0, 0, 0], [0, 0, 0.01]], dtype=np.float64),
            rvec, camera_T_coil[:3, 3], K, dist)
        uv = np.rint(points.reshape(2, 2)).astype(int)
        color = colors[index % len(colors)]
        cv2.circle(image, tuple(uv[0]), 7, color, 2, cv2.LINE_AA)
        cv2.arrowedLine(
            image, tuple(uv[0]), tuple(uv[1]), color, 2, cv2.LINE_AA,
            tipLength=0.25)
        cv2.putText(
            image, 'EM ' + pose['part_number'], tuple(uv[0] + [8, -8]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return len(poses)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', default='HD1080',
                        choices=['HD2K', 'HD1080', 'HD720', 'VGA'])
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--sharpness', type=int, default=4)
    parser.add_argument('--exposure', type=int, default=-1)
    parser.add_argument('--gain', type=int, default=-1)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--marker-id', type=int, default=33)
    parser.add_argument('--marker-size-mm', type=float, default=25.0)
    parser.add_argument('--aurora-port', required=True)
    parser.add_argument('--aurora-expected-tools', type=int, default=3)
    parser.add_argument(
        '--aurora-probe-part-number', default='610175 T6E0-S00923')
    parser.add_argument(
        '--aurora-driver-root', type=Path, default=DEFAULT_DRIVER_ROOT)
    parser.add_argument(
        '--registration-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--field-lock-window-s', type=float, default=1.5)
    parser.add_argument('--field-lock-min-samples', type=int, default=3)
    parser.add_argument('--stationary-window-s', type=float, default=1.0)
    parser.add_argument('--optical-min-samples', type=int, default=5)
    parser.add_argument('--em-min-samples', type=int, default=20)
    parser.add_argument(
        '--optical-max-position-deviation-mm', type=float, default=0.5)
    parser.add_argument(
        '--optical-max-orientation-deviation-deg', type=float, default=1.0)
    parser.add_argument(
        '--em-max-position-deviation-mm', type=float, default=0.15)
    parser.add_argument(
        '--em-max-orientation-deviation-deg', type=float, default=1.0)
    parser.add_argument('--max-em-image-age-ms', type=float, default=100.0)
    parser.add_argument(
        '--max-stereo-reprojection-rmse-px', type=float, default=3.0)
    parser.add_argument('--expected-coil-separation-mm', type=float, default=3.8)
    return parser.parse_args(argv)


def run_capture(
        args, camera, recorder, tip_detector, field_board, field_config,
        session_dir, snapshots_dir, snapshots):
    info = camera.info()
    np.savez(
        session_dir / 'left_intrinsics.npz', K=camera.K, dist=camera.dist,
        K_right=camera.K_right, dist_right=camera.dist_right,
        right_camera_T_left_camera_m=rectified_right_T_left(
            info['baseline_m']),
        fx=info['fx'], fy=info['fy'], cx=info['cx'], cy=info['cy'],
        resolution=info['resolution'], baseline_m=info['baseline_m'])
    metadata = {
        'created_at_ns': time.time_ns(),
        'purpose': 'rigid ArUco-to-dual-EM tool calibration',
        'tip_marker': {
            'dictionary': 'DICT_4X4_50', 'id': args.marker_id,
            'black_edge_size_mm': args.marker_size_mm,
            'coordinate_frame': (
                'center origin; +x printed right; +y printed up; '
                '+z out of printed front')},
        'dual_coil_fixture': {
            'coil_type': '5DOF',
            'observable_axis': 'Aurora quaternion local +z',
            'expected_axis_in_tip_marker': [0.0, -1.0, 0.0],
            'expected_separation_mm': args.expected_coil_separation_mm,
            'cad_marker_frame': 'corner origin; +z into housing',
            'cad_coil_tip_positions_mm': {
                '003': [14.4, 12.5, 3.0],
                '07222026_01': [10.6, 12.5, 3.0]},
            'opencv_marker_coil_tip_positions_mm': {
                '003': [1.9, 0.0, -3.0],
                '07222026_01': [-1.9, 0.0, -3.0]},
            'roll_handling': 'ignored; roll about coil z is unobservable'},
        'field_generator': {
            'board_index': field_config['board_index'],
            'marker_ids': [field_config['marker_id_offset'],
                           field_config['marker_id_offset'] + 7],
            'aurora_T_field_marker_m':
                field_config['aurora_T_marker'].tolist()},
        'camera': info,
        'stationarity': {
            'field_lock_window_s': args.field_lock_window_s,
            'field_lock_min_samples': args.field_lock_min_samples,
            'window_s': args.stationary_window_s,
            'optical_min_samples': args.optical_min_samples,
            'em_min_samples': args.em_min_samples,
            'optical_max_position_deviation_mm':
                args.optical_max_position_deviation_mm,
            'optical_max_orientation_deviation_deg':
                args.optical_max_orientation_deviation_deg,
            'em_max_position_deviation_mm':
                args.em_max_position_deviation_mm,
            'em_max_orientation_deviation_deg':
                args.em_max_orientation_deviation_deg,
            'max_em_image_age_ms': args.max_em_image_age_ms,
            'max_stereo_reprojection_rmse_px':
                args.max_stereo_reprojection_rmse_px,
        },
        'tools': sorted(
            recorder.tool_info.values(), key=lambda item: item['role']),
    }
    (session_dir / 'capture_metadata.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    pose_path = session_dir / 'optical_poses.csv'
    with pose_path.open('w', newline='', buffering=1) as pose_file:
        pose_writer = csv.DictWriter(pose_file, fieldnames=POSE_CSV_FIELDS)
        pose_writer.writeheader()
        run_capture_loop(
            args, camera, recorder, tip_detector, field_board,
            field_config, snapshots_dir, snapshots, pose_writer)


def run_capture_loop(
        args, camera, recorder, tip_detector, field_board, field_config,
        snapshots_dir, snapshots, pose_writer):
    history_length = max(
        30, int(np.ceil(
            args.fps * max(
                args.stationary_window_s, args.field_lock_window_s) * 2)))
    tip_history = deque(maxlen=history_length)
    field_history = deque(maxlen=history_length)
    field_lock = None
    last_field_lock_error = None
    window = 'ArUco + Aurora calibration capture'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 450)
    print('Phase 1: keep the calibration object out of view and expose '
          'field-board IDs 17-24 until FIELD LOCKED appears.')
    print('Controls after lock: Space=capture  Backspace=delete last  '
          'q/ESC=finish')
    while True:
        grabbed = camera.grab()
        if recorder.error is not None:
            raise RuntimeError(
                'Aurora recording failed: {}'.format(recorder.error))
        if grabbed is None:
            continue
        timestamp_ns, left_bgr, right_bgr = grabbed
        gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        overlay = left_bgr.copy()
        overlay_right = right_bgr.copy()

        tip_pose = None
        if field_lock is not None:
            tip_pose = tip_detector.detect_stereo(
                gray, gray_right, camera.K, camera.dist,
                camera.K_right, camera.dist_right,
                rectified_right_T_left(camera.baseline_m))
        tip_sample = None
        if tip_pose is not None:
            cv2.aruco.drawDetectedMarkers(
                overlay,
                [tip_pose['corners'].reshape(1, 4, 2)],
                np.asarray([[args.marker_id]], dtype=np.int32))
            cv2.drawFrameAxes(
                overlay, camera.K, camera.dist,
                tip_pose['rvec'], tip_pose['tvec'],
                args.marker_size_mm * 0.00075)
            cv2.aruco.drawDetectedMarkers(
                overlay_right,
                [tip_pose['right_corners'].reshape(1, 4, 2)],
                np.asarray([[args.marker_id]], dtype=np.int32))
            right_T_marker = (
                rectified_right_T_left(camera.baseline_m)
                @ np.asarray(tip_pose['left_camera_T_marker_m']))
            cv2.drawFrameAxes(
                overlay_right, camera.K_right, camera.dist_right,
                Rotation.from_matrix(
                    right_T_marker[:3, :3]).as_rotvec(),
                right_T_marker[:3, 3],
                args.marker_size_mm * 0.00075)
            tip_sample = pose_sample(
                timestamp_ns, tip_pose['rvec'], tip_pose['tvec'],
                tip_pose['reprojection_rmse_px'])
            tip_history.append(tip_sample)

        field_pose, field_pose_right, field_stereo_pose = None, None, None
        field_sample, field_lock_error = None, None
        if field_lock is None:
            field_results, _ = registration.detect_boards(
                gray, [field_board], camera.K, camera.dist)
            field_results_right, _ = registration.detect_boards(
                gray_right, [field_board],
                camera.K_right, camera.dist_right)
            field_board_left = field_results[0] if field_results else None
            field_board_right = (
                field_results_right[0] if field_results_right else None)
            field_pose = (
                field_board_left
                if field_board_left is not None and field_board_left.has_pose
                else None)
            field_pose_right = (
                field_board_right
                if field_board_right is not None and field_board_right.has_pose
                else None)
            for result in field_results:
                registration.draw_pose(
                    overlay, result, camera.K, camera.dist, 0.03,
                    draw_axes=False)
            for result in field_results_right:
                registration.draw_pose(
                    overlay_right, result,
                    camera.K_right, camera.dist_right, 0.03,
                    draw_axes=False)
            if field_pose is not None and field_board_right is not None:
                try:
                    field_stereo_pose = estimate_stereo_charuco_pose(
                        field_board_left, field_board_right, field_board,
                        camera.K, camera.dist,
                        camera.K_right, camera.dist_right,
                        rectified_right_T_left(camera.baseline_m))
                except (RuntimeError, ValueError):
                    field_stereo_pose = None
            if field_stereo_pose is not None:
                cv2.drawFrameAxes(
                    overlay, camera.K, camera.dist,
                    field_stereo_pose['rvec'], field_stereo_pose['tvec'],
                    0.03)
                right_T_field = (
                    rectified_right_T_left(camera.baseline_m)
                    @ field_stereo_pose['left_camera_T_object_m'])
                cv2.drawFrameAxes(
                    overlay_right, camera.K_right, camera.dist_right,
                    Rotation.from_matrix(
                        right_T_field[:3, :3]).as_rotvec(),
                    right_T_field[:3, 3], 0.03)
                field_sample = pose_sample(
                    timestamp_ns, field_stereo_pose['rvec'],
                    field_stereo_pose['tvec'],
                    field_stereo_pose['stereo_reprojection_rmse_px'])
                field_history.append(field_sample)
            field_summary = None
            try:
                field_summary = summarize_stationary_samples(
                    field_history, timestamp_ns, args.field_lock_window_s,
                    args.field_lock_min_samples,
                    args.optical_max_position_deviation_mm,
                    args.optical_max_orientation_deviation_deg,
                    sample_label='field marker')
            except RuntimeError as exc:
                field_lock_error = str(exc)
            else:
                reprojection_rmse = field_summary[
                    'max_reprojection_rmse_px']
                if (reprojection_rmse is None
                        or reprojection_rmse
                        > args.max_stereo_reprojection_rmse_px):
                    field_lock_error = (
                        'field stereo reprojection RMSE {:.3f}px exceeds '
                        '{:.3f}px'.format(
                            float('nan') if reprojection_rmse is None
                            else reprojection_rmse,
                            args.max_stereo_reprojection_rmse_px))
                    field_summary = None
            if field_summary is not None:
                field_lock = build_field_frame_lock(
                    field_summary, field_config, timestamp_ns)
                (snapshots_dir.parent / 'field_frame_lock.json').write_text(
                    json.dumps(field_lock, indent=2, sort_keys=True),
                    encoding='utf-8')
                cv2.imwrite(
                    str(snapshots_dir.parent / 'field_lock_left.png'),
                    left_bgr)
                cv2.imwrite(
                    str(snapshots_dir.parent / 'field_lock_right.png'),
                    right_bgr)
                cv2.imwrite(
                    str(snapshots_dir.parent / 'field_lock_right_overlay.png'),
                    overlay_right)
                cv2.imwrite(
                    str(snapshots_dir.parent / 'field_lock_overlay.png'),
                    overlay)
                cv2.imwrite(
                    str(snapshots_dir.parent / 'field_lock_stereo_overlay.png'),
                    np.hstack([overlay, overlay_right]))
                print('[FIELD LOCKED] Place the ArUco/EM fixture; the field '
                      'board may now be occluded.')

        field_recent = [
            sample for sample in field_history
            if sample['timestamp_ns'] >= (
                timestamp_ns - int(args.field_lock_window_s * 1e9))]
        field_span_s = (
            (field_recent[-1]['timestamp_ns']
             - field_recent[0]['timestamp_ns']) * 1e-9
            if len(field_recent) >= 2 else 0.0)
        if (field_lock_error is not None
                and 'not stationary' in field_lock_error
                and field_lock_error != last_field_lock_error):
            print('[FIELD LOCK WAIT] ' + field_lock_error)
        last_field_lock_error = field_lock_error

        em_count = draw_live_coils(
            overlay, camera, recorder,
            None if field_lock is None
            else field_lock['left_camera_T_aurora_m'],
            timestamp_ns)
        draw_live_coils(
            overlay_right, camera, recorder,
            None if field_lock is None
            else field_lock['left_camera_T_aurora_m'],
            timestamp_ns, right_camera=True)

        write_pose_row(
            pose_writer, timestamp_ns, 'tip_marker', tip_sample,
            4 if tip_sample else 0)
        write_pose_row(
            pose_writer, timestamp_ns, 'field_charuco', field_sample,
            (min(field_stereo_pose['left_corner_count'],
                 field_stereo_pose['right_corner_count'])
             if field_stereo_pose is not None else 0))
        field_status = (
            'LOCKED' if field_lock is not None
            else ('DETECTED' if field_sample else 'WAITING'))
        right_field_status = (
            'LOCKED' if field_lock is not None
            else ('DETECTED' if field_pose_right is not None else 'WAITING'))
        status = 'tip={} field L/R={}/{} EM={}/2 snapshots={}'.format(
            'YES' if tip_sample else 'NO',
            field_status, right_field_status, em_count, len(snapshots))
        cv2.putText(
            overlay, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 220, 255), 2)
        instruction = (
            'Hold fixture still, then press SPACE'
            if field_lock is not None else
            'Field lock: n={}/{} span={:.1f}/{:.1f}s'.format(
                len(field_recent), args.field_lock_min_samples,
                field_span_s, args.field_lock_window_s * 0.8))
        cv2.putText(
            overlay, instruction,
            (15, overlay.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 220, 255), 2)
        cv2.putText(
            overlay, 'LEFT', (15, 75), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 220, 255), 2)
        cv2.putText(
            overlay_right, 'RIGHT', (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 220, 255), 2)
        cv2.imshow(window, np.hstack([overlay, overlay_right]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == 8 and snapshots:
            delete_last_snapshot(snapshots_dir, snapshots)
            continue
        if key != ord(' '):
            continue
        if field_lock is None:
            print('[NOT READY] Waiting for stationary field-frame lock')
            continue
        try:
            save_snapshot(
                args, recorder, field_lock, timestamp_ns,
                left_bgr, right_bgr, overlay, tip_history,
                snapshots_dir, snapshots)
        except (RuntimeError, ValueError) as exc:
            print('[REJECTED] {}'.format(exc))
    cv2.destroyWindow(window)


def write_snapshot_index(snapshots_dir, snapshots):
    (snapshots_dir.parent / 'snapshots.json').write_text(
        json.dumps(snapshots, indent=2, sort_keys=True), encoding='utf-8')


def delete_last_snapshot(snapshots_dir, snapshots):
    removed = snapshots.pop()
    index = removed['snapshot_index']
    for path in snapshots_dir.glob('snapshot_{:03d}_*'.format(index)):
        path.unlink()
    json_path = snapshots_dir / 'snapshot_{:03d}.json'.format(index)
    if json_path.exists():
        json_path.unlink()
    write_snapshot_index(snapshots_dir, snapshots)
    print('[DELETE] snapshot {:03d}'.format(index))


def save_snapshot(
        args, recorder, field_lock, timestamp_ns,
        left_bgr, right_bgr, overlay, tip_history,
        snapshots_dir, snapshots):
    tip_summary = summarize_stationary_samples(
        tip_history, timestamp_ns, args.stationary_window_s,
        args.optical_min_samples,
        args.optical_max_position_deviation_mm,
        args.optical_max_orientation_deviation_deg,
        sample_label='stereo marker')
    stereo_rmse = tip_summary['max_reprojection_rmse_px']
    if (stereo_rmse is None
            or stereo_rmse > args.max_stereo_reprojection_rmse_px):
        raise RuntimeError(
            'stereo marker reprojection RMSE {:.3f}px exceeds {:.3f}px'.format(
                float('nan') if stereo_rmse is None else stereo_rmse,
                args.max_stereo_reprojection_rmse_px))
    coil_poses = summarize_coils(recorder, timestamp_ns, args)
    geometry = build_snapshot_geometry(
        tip_summary, coil_poses, field_lock['left_camera_T_aurora_m'])
    index = len(snapshots) + 1
    stem = 'snapshot_{:03d}'.format(index)
    names = {
        'left': stem + '_left.png',
        'right': stem + '_right.png',
        'overlay': stem + '_overlay.png'}
    for name, image in (
            (names['left'], left_bgr), (names['right'], right_bgr),
            (names['overlay'], overlay)):
        if not cv2.imwrite(str(snapshots_dir / name), image):
            raise OSError('failed to save ' + name)
    snapshot = {
        'snapshot_index': index,
        'accepted_at_ns': time.time_ns(),
        'image_timestamp_ns': int(timestamp_ns),
        'images': names,
        'tip_marker_optical_summary': tip_summary,
        'field_frame_lock': field_lock,
        **geometry,
    }
    snapshots.append(snapshot)
    (snapshots_dir / (stem + '.json')).write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding='utf-8')
    write_snapshot_index(snapshots_dir, snapshots)
    print('[CAPTURED] snapshot {:03d}: tip p95={:.3f}mm'.format(
        index, tip_summary['stationarity']['position_p95_deviation_mm']))


def main(argv=None):
    from .zed_capture import ZedCamera

    args = parse_args(argv)
    field_config = camera_register.load_field_generator_config(
        str(args.registration_config))
    if field_config is None:
        raise ValueError(
            'registration config lacks field_generator_registration')
    _, all_boards = boards_mod.build_boards(additional=[(
        field_config['board_index'], field_config['marker_id_offset'])])
    field_board = next(
        board for board in all_boards
        if board.index == field_config['board_index'])
    tip_detector = TipMarkerDetector(
        args.marker_id, args.marker_size_mm * 0.001)
    session_name = 'aruco_em_calibration_' + time.strftime('%Y%m%d_%H%M%S')
    session_dir = args.outdir / session_name
    snapshots_dir = session_dir / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=False)
    print('Opening Aurora on {} ...'.format(args.aurora_port))
    recorder = AuroraRecorder(
        args.aurora_port, driver_root=str(args.aurora_driver_root),
        expected_tools=args.aurora_expected_tools,
        probe_part_number=args.aurora_probe_part_number,
        require_probe=False)
    snapshots = []
    try:
        recorder.open()
        print('Aurora ready: {} tools'.format(recorder.tool_count))
        for item in sorted(
                recorder.tool_info.values(), key=lambda value: value['role']):
            print('  {}: handle={} serial={} part={}'.format(
                item['role'], item['port_handle'], item['serial_number'],
                item['part_number']))
        recorder.start(str(session_dir))
        with ZedCamera(
                args.resolution, args.fps, args.sharpness,
                args.exposure, args.gain) as camera:
            run_capture(
                args, camera, recorder, tip_detector, field_board,
                field_config, session_dir, snapshots_dir, snapshots)
    finally:
        try:
            recorder.close()
        finally:
            cv2.destroyAllWindows()
    print('Done: {} snapshots in {}'.format(len(snapshots), session_dir))
    if len(snapshots) >= 6:
        from .tool_calibration_solve import build_report
        try:
            report = build_report(
                session_dir, args.expected_coil_separation_mm,
                args.max_stereo_reprojection_rmse_px)
            report_path = session_dir / 'tool_calibration_report.json'
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding='utf-8')
            fit = report['calibration']
            print('[CALIBRATION] position RMS={:.3f}mm; axis RMS={:.3f}deg; '
                  'separation={:.3f}mm -> {}'.format(
                      fit['position_residual_mm']['rms'],
                      fit['coil_axis_residual_deg']['rms'],
                      fit['coil_separation_mm'], report_path))
        except Exception as exc:
            print('[CALIBRATION FAILED] {}'.format(exc))


if __name__ == '__main__':
    main()
