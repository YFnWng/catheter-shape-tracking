"""Camera <-> robot-base registration from the ChArUco fixture.

Reads a YAML config that gives, per ChArUco board, the rigid transform from the
marker frame to the robot BASE frame, plus a workspace volume. Averages the
board pose over many static frames, composes the base pose, projects the
workspace box into the image to form a base-fixed 2D ROI, and saves everything.

BASE FRAME convention: origin at the catheter base; +z along the catheter at
rest, pointing toward the tip. The saved overlay draws the base axes and the
workspace box so the convention can be verified by eye.

Run:
    python -m shape_tracking.register --config registration_config.yaml \
        [--frames 150] [--outdir DIR] [--show]

The board-frame origin location does not affect accuracy: the base pose is a
physical quantity invariant to the origin choice. Board detection works with the
camera in any roll (portrait ok).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

from . import boards as boards_mod
from . import registration
from .boards import AXIS_LEN_M

WARMUP = 15  # frames skipped for auto-exposure to settle


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_transform(entry) -> np.ndarray:
    """Parse one marker->base transform (p_base = T @ p_marker) into a 4x4."""
    if "matrix" in entry:
        T = np.asarray(entry["matrix"], dtype=float)
        if T.shape != (4, 4):
            raise ValueError("transform 'matrix' must be 4x4")
        return T
    t = np.asarray(entry["translation"], dtype=float)
    if "rotation_matrix" in entry:
        M = np.asarray(entry["rotation_matrix"], dtype=float)
        if M.shape != (3, 3):
            raise ValueError("'rotation_matrix' must be 3x3")
        R = Rot.from_matrix(M)
    elif "rotation_euler_deg" in entry:
        R = Rot.from_euler("xyz", entry["rotation_euler_deg"], degrees=True)
    elif "rotation_quat" in entry:
        R = Rot.from_quat(entry["rotation_quat"])
    else:
        raise ValueError("give 'rotation_matrix', 'rotation_euler_deg' or "
                         "'rotation_quat' (or a full 4x4 'matrix')")
    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    T[:3, 3] = t
    return T


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # 'units' scales ALL translations + the workspace box (not margin_px).
    # Rotations are dimensionless and never scaled.
    scale = 0.001 if str(cfg.get("units", "m")).lower() == "mm" else 1.0
    markers = {}
    for k, v in cfg["markers"].items():
        T = load_transform(v)
        T[:3, 3] *= scale
        markers[int(k)] = T
    ws = cfg["workspace"]
    workspace = {
        "x": [float(ws["x"][0]) * scale, float(ws["x"][1]) * scale],
        "y": [float(ws["y"][0]) * scale, float(ws["y"][1]) * scale],
        "z": [float(ws["z"][0]) * scale, float(ws["z"][1]) * scale],
        "margin_px": int(ws.get("margin_px", 20)),
    }
    return markers, workspace


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def homogeneous(rot: Rot, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def robust_average_pose(rvecs, tvecs):
    """Robustly average (rvec, tvec) sets. Returns (Rot, t, stats)."""
    rvecs = np.asarray(rvecs, dtype=float)
    tvecs = np.asarray(tvecs, dtype=float)
    rots = Rot.from_rotvec(rvecs)
    t_med = np.median(tvecs, axis=0)
    dt = np.linalg.norm(tvecs - t_med, axis=1)
    t_in = dt < (3.0 * np.median(dt) + 1e-9)
    ang0 = np.degrees((rots * rots.mean().inv()).magnitude())
    r_in = ang0 < (3.0 * np.median(ang0) + 1.0)
    inl = t_in & r_in
    if inl.sum() < 3:
        inl = np.ones(len(rvecs), dtype=bool)
    mean_rot = Rot.from_rotvec(rvecs[inl]).mean()
    mean_t = tvecs[inl].mean(axis=0)
    ang = np.degrees((Rot.from_rotvec(rvecs[inl]) * mean_rot.inv()).magnitude())
    stats = {
        "n_total": int(len(rvecs)),
        "n_used": int(inl.sum()),
        "t_std_mm": tvecs[inl].std(axis=0) * 1000.0,
        "rot_std_deg": float(ang.std()),
    }
    return mean_rot, mean_t, stats


def workspace_corners(workspace):
    """8 corners (base frame) of the axis-aligned workspace box."""
    xs, ys, zs = workspace["x"], workspace["y"], workspace["z"]
    return np.array([[x, y, z, 1.0] for x in xs for y in ys for z in zs])


def project(K, cam_T_base, pts_base):
    """Project base-frame homogeneous points to pixels. Returns (uv, in_front)."""
    cc = (cam_T_base @ pts_base.T).T          # (N,4) in camera frame
    in_front = cc[:, 2] > 1e-4
    z = np.where(in_front, cc[:, 2], 1.0)
    uv = (K @ (cc[:, :3] / z[:, None]).T).T[:, :2]
    return uv, in_front


def workspace_roi(K, cam_T_base, workspace, img_w, img_h):
    """Axis-aligned 2D ROI (x,y,w,h) covering the projected workspace box."""
    uv, in_front = project(K, cam_T_base, workspace_corners(workspace))
    uv = uv[in_front]
    m = workspace["margin_px"]
    x0 = int(np.clip(uv[:, 0].min() - m, 0, img_w - 1))
    y0 = int(np.clip(uv[:, 1].min() - m, 0, img_h - 1))
    x1 = int(np.clip(uv[:, 0].max() + m, 1, img_w))
    y1 = int(np.clip(uv[:, 1].max() + m, 1, img_h))
    return (x0, y0, x1 - x0, y1 - y0)


def draw_box(vis, K, cam_T_base, workspace, color=(0, 200, 255)):
    corners = workspace_corners(workspace)
    uv, in_front = project(K, cam_T_base, corners)
    pts = uv.astype(int)
    for i in range(8):
        for j in range(i + 1, 8):
            # edge if the two corner indices differ in exactly one axis bit
            if bin(i ^ j).count("1") == 1 and in_front[i] and in_front[j]:
                cv2.line(vis, tuple(pts[i]), tuple(pts[j]), color, 1, cv2.LINE_AA)


def _mm_transform_to_m(transform):
    result = np.asarray(transform, dtype=float).copy()
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError('EM registration transform must be a finite 4x4 matrix')
    result[:3, 3] *= 0.001
    return result


def build_em_overlay_geometry(em_registration, coil_poses):
    '''Transform synchronized Aurora coil poses into the robot-base frame.'''
    if not coil_poses or len(coil_poses) != 2:
        raise ValueError('camera registration overlay requires two EM coil poses')
    base_T_aurora_mm = np.asarray(
        em_registration.get('robot_base_T_aurora'), dtype=float)
    base_T_aurora = _mm_transform_to_m(base_T_aurora_mm)
    coils = []
    seen_parts = set()
    for source in coil_poses:
        part_number = str(source.get('part_number', '')).strip()
        part_key = ''.join(char for char in part_number if char.isalnum())
        if not part_key or part_key in seen_parts:
            raise ValueError(
                f'EM overlay coil part numbers must be unique: {part_number!r}')
        seen_parts.add(part_key)
        position = np.asarray(source.get('position_aurora_mm'), dtype=float)
        quaternion = np.asarray(
            source.get('quaternion_aurora_wxyz'), dtype=float)
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(quaternion))):
            raise ValueError(f'invalid EM overlay pose for coil {part_number!r}')
        norm = np.linalg.norm(quaternion)
        if norm < 1e-12:
            raise ValueError(f'zero-norm EM quaternion for coil {part_number!r}')
        quaternion /= norm
        aurora_T_coil_mm = np.eye(4)
        aurora_T_coil_mm[:3, :3] = Rot.from_quat([
            quaternion[1], quaternion[2], quaternion[3], quaternion[0],
        ]).as_matrix()
        aurora_T_coil_mm[:3, 3] = position
        base_T_coil_mm = base_T_aurora_mm @ aurora_T_coil_mm
        base_T_coil = _mm_transform_to_m(base_T_coil_mm)
        saved = dict(source)
        saved['position_robot_base_mm'] = base_T_coil_mm[:3, 3].tolist()
        saved['z_axis_robot_base'] = base_T_coil_mm[:3, 2].tolist()
        coils.append({
            'part_number': part_number,
            'base_T_coil': base_T_coil,
            'saved_pose': saved,
        })
    return base_T_aurora, coils


def draw_registered_frame(
        vis, K, cam_T_base, base_T_frame, axis_len_m, label):
    '''Draw a registered 3D coordinate frame using rectified-image projection.'''
    points_frame = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [axis_len_m, 0.0, 0.0, 1.0],
        [0.0, axis_len_m, 0.0, 1.0],
        [0.0, 0.0, axis_len_m, 1.0],
    ])
    points_base = (base_T_frame @ points_frame.T).T
    uv, in_front = project(K, cam_T_base, points_base)
    if not in_front[0]:
        return
    pixels = np.rint(uv).astype(int)
    origin = tuple(pixels[0])
    cv2.circle(vis, origin, 5, (255, 255, 255), -1, cv2.LINE_AA)
    for index, (color, axis_name) in enumerate((
            ((0, 0, 255), 'X'),
            ((0, 255, 0), 'Y'),
            ((255, 0, 0), 'Z'),
            ), start=1):
        if in_front[index]:
            cv2.arrowedLine(
                vis, origin, tuple(pixels[index]), color, 2,
                cv2.LINE_AA, tipLength=0.18)
            cv2.putText(
                vis, axis_name, tuple(pixels[index]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.putText(
        vis, label, (origin[0] + 7, origin[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        cv2.LINE_AA)


def draw_coil_z_axis(
        vis, K, cam_T_base, base_T_coil, axis_len_m, label, color):
    points_coil = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, axis_len_m, 1.0],
    ])
    points_base = (base_T_coil @ points_coil.T).T
    uv, in_front = project(K, cam_T_base, points_base)
    if not np.all(in_front):
        return
    pixels = np.rint(uv).astype(int)
    origin, endpoint = tuple(pixels[0]), tuple(pixels[1])
    cv2.circle(vis, origin, 6, color, -1, cv2.LINE_AA)
    cv2.circle(vis, origin, 7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.arrowedLine(
        vis, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.22)
    cv2.putText(
        vis, label, (origin[0] + 8, origin[1] + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def annotate_view(
        bgr, boards, K, dist, board_poses, cam_T_base, workspace, axis_len,
        base_T_aurora=None, coils=None):
    """Overlay markers+ids, ChArUco corners, every board frame, the base frame,
    the workspace box and 2D ROI. Poses must be expressed in THIS view's camera
    frame. board_poses maps board index -> its 4x4 pose."""
    vis = bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    results, _ = registration.detect_boards(gray, boards, K, dist)
    for res in results:
        if res.marker_ids is not None and len(res.marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, res.marker_corners, res.marker_ids)
        if res.charuco_ids is not None and len(res.charuco_ids) >= 4:
            cc = np.asarray(res.charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
            ci = np.asarray(res.charuco_ids, dtype=np.int32).reshape(-1, 1)
            cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci)
    for T in board_poses.values():                                # each board frame (small)
        cv2.drawFrameAxes(vis, K, dist,
                          Rot.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3], axis_len)
    cv2.drawFrameAxes(vis, K, dist,
                      Rot.from_matrix(cam_T_base[:3, :3]).as_rotvec(),
                      cam_T_base[:3, 3], axis_len * 2)            # base frame (large)
    if base_T_aurora is not None:
        draw_registered_frame(
            vis, K, cam_T_base, base_T_aurora, axis_len * 2,
            'AURORA FIELD')
    coil_colors = ((255, 0, 255), (0, 255, 255))
    for index, coil in enumerate(coils or []):
        draw_coil_z_axis(
            vis, K, cam_T_base, coil['base_T_coil'], axis_len,
            f"EM {coil['part_number']} +Z",
            coil_colors[index % len(coil_colors)])
    draw_box(vis, K, cam_T_base, workspace)
    h, w = bgr.shape[:2]
    x, y, ww, hh = workspace_roi(K, cam_T_base, workspace, w, h)
    cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
    return vis


def save_session_camera_registration(
        registration_path, output_dir, config_path, collected,
        left_bgr, right_bgr, boards, K, dist, resolution,
        zed_serial, baseline_m, image_timestamp_ns, min_frames=10,
        em_tool_poses=None):
    """Fit camera-to-base registration and merge it into a session JSON.

    The collected mapping contains r, t and c observation lists per board.
    """
    with open(registration_path, encoding='utf-8') as stream:
        document = json.load(stream)
    if not document.get('em') or (
            document['em'].get('transform_status') != 'solved'):
        raise ValueError('EM registration must be solved before camera registration')
    base_T_aurora, overlay_coils = build_em_overlay_geometry(
        document['em'], em_tool_poses)

    markers, workspace = load_config(config_path)
    per_board = {}
    for board_index in sorted(markers):
        data = collected.get(board_index, {'r': [], 't': [], 'c': []})
        if len(data['r']) < min_frames:
            continue
        rot, translation, stats = robust_average_pose(data['r'], data['t'])
        left_camera_T_board = homogeneous(rot, translation)
        left_camera_T_robot_base = (
            left_camera_T_board @ np.linalg.inv(markers[board_index]))
        per_board[board_index] = {
            'left_camera_T_board': left_camera_T_board,
            'left_camera_T_robot_base': left_camera_T_robot_base,
            'stats': stats,
            'mean_corners': float(np.mean(data['c'])),
        }
    if not per_board:
        counts = {
            str(index): len(collected.get(index, {}).get('r', []))
            for index in sorted(markers)}
        raise ValueError(
            f'no configured ChArUco board has {min_frames} valid frames; '
            f'counts={counts}')

    agreement = None
    if len(per_board) > 1:
        first, second = list(per_board)[:2]
        A = per_board[first]['left_camera_T_robot_base']
        B = per_board[second]['left_camera_T_robot_base']
        agreement = {
            'boards': [first, second],
            'translation_mm': float(
                np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000.0),
            'rotation_deg': float(np.degrees(
                (Rot.from_matrix(A[:3, :3]).inv()
                 * Rot.from_matrix(B[:3, :3])).magnitude())),
        }

    primary = max(
        per_board, key=lambda index: per_board[index]['stats']['n_used'])
    left_camera_T_robot_base = per_board[primary][
        'left_camera_T_robot_base']
    right_camera_T_left_camera = np.eye(4)
    right_camera_T_left_camera[0, 3] = -float(baseline_m)
    right_camera_T_robot_base = (
        right_camera_T_left_camera @ left_camera_T_robot_base)
    left_roi = workspace_roi(
        K, left_camera_T_robot_base, workspace,
        left_bgr.shape[1], left_bgr.shape[0])
    right_roi = workspace_roi(
        K, right_camera_T_robot_base, workspace,
        right_bgr.shape[1], right_bgr.shape[0])

    poses_left = {
        index: item['left_camera_T_board']
        for index, item in per_board.items()}
    poses_right = {
        index: right_camera_T_left_camera @ pose
        for index, pose in poses_left.items()}
    overlay_left = annotate_view(
        left_bgr, boards, K, dist, poses_left,
        left_camera_T_robot_base, workspace, AXIS_LEN_M,
        base_T_aurora=base_T_aurora, coils=overlay_coils)
    overlay_right = annotate_view(
        right_bgr, boards, K, dist, poses_right,
        right_camera_T_robot_base, workspace, AXIS_LEN_M,
        base_T_aurora=base_T_aurora, coils=overlay_coils)
    left_name = 'registration_left.png'
    right_name = 'registration_right.png'
    if not cv2.imwrite(os.path.join(output_dir, left_name), overlay_left):
        raise OSError('failed to write left registration overlay')
    if not cv2.imwrite(os.path.join(output_dir, right_name), overlay_right):
        raise OSError('failed to write right registration overlay')

    camera = {
        'status': 'solved',
        'completed_at_ns': time.time_ns(),
        'image_timestamp_ns': int(image_timestamp_ns),
        'primary_board': int(primary),
        'boards_used': sorted(int(index) for index in per_board),
        'left_camera_T_robot_base': left_camera_T_robot_base.tolist(),
        'robot_base_T_left_camera': np.linalg.inv(
            left_camera_T_robot_base).tolist(),
        'right_camera_T_robot_base': right_camera_T_robot_base.tolist(),
        'robot_base_T_right_camera': np.linalg.inv(
            right_camera_T_robot_base).tolist(),
        'right_camera_T_left_camera': right_camera_T_left_camera.tolist(),
        'intrinsics': {
            'K': np.asarray(K).tolist(),
            'distortion': np.asarray(dist).reshape(-1).tolist(),
            'resolution': resolution,
            'zed_serial': str(zed_serial),
            'baseline_m': float(baseline_m),
        },
        'workspace_base_m': {
            'x': workspace['x'], 'y': workspace['y'], 'z': workspace['z'],
            'margin_px': workspace['margin_px'],
        },
        'roi_left_xywh': list(left_roi),
        'roi_right_xywh': list(right_roi),
        'robot_base_T_marker': {
            str(index): transform.tolist()
            for index, transform in markers.items()},
        'per_board': {
            str(index): {
                'left_camera_T_board': item[
                    'left_camera_T_board'].tolist(),
                'left_camera_T_robot_base': item[
                    'left_camera_T_robot_base'].tolist(),
                'mean_corners': item['mean_corners'],
                'stats': {
                    'n_total': item['stats']['n_total'],
                    'n_used': item['stats']['n_used'],
                    't_std_mm': np.asarray(
                        item['stats']['t_std_mm']).tolist(),
                    'rot_std_deg': item['stats']['rot_std_deg'],
                },
            } for index, item in per_board.items()
        },
        'board_agreement': agreement,
        'em_overlay': {
            'image_timestamp_ns': int(image_timestamp_ns),
            'field_frame': 'aurora',
            'field_axis_length_mm': float(AXIS_LEN_M * 2000.0),
            'coil_z_axis_length_mm': float(AXIS_LEN_M * 1000.0),
            'coil_poses': [
                coil['saved_pose'] for coil in overlay_coils],
        },
        'overlay_left': left_name,
        'overlay_right': right_name,
    }

    document['camera'] = camera
    document['complete'] = True
    document['completed_at_ns'] = camera['completed_at_ns']
    tmp_path = registration_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
    os.replace(tmp_path, registration_path)
    return camera


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    from .zed_capture import ZedCamera

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML registration config.")
    ap.add_argument("--resolution", default="HD1080",
                    choices=["HD2K", "HD1080", "HD720", "VGA"])
    ap.add_argument("--sharpness", type=int, default=8)
    ap.add_argument("--frames", type=int, default=150,
                    help="static frames to average per board. Default 150.")
    ap.add_argument("--min-corners", type=int, default=6)
    ap.add_argument("--outdir", default=os.path.join(os.getcwd(), "registration"))
    ap.add_argument("--show", action="store_true",
                    help="save an overlay (board+base axes, workspace box, ROI).")
    args = ap.parse_args(argv)

    markers, workspace = load_config(args.config)
    os.makedirs(args.outdir, exist_ok=True)
    target = sorted(markers.keys())
    _, boards = boards_mod.build_boards()
    collected = {b: {"r": [], "t": [], "c": []} for b in target}

    with ZedCamera(args.resolution, 30, args.sharpness) as cam:
        info = cam.info()
        baseline = float(info["baseline_m"])
        print(f"ZED {info['serial']} {info['resolution']} — hold the fixture still; "
              f"averaging up to {args.frames} frames for boards {target}")
        max_grabs = WARMUP + args.frames * 5
        last_left = last_right = None
        for i in range(max_grabs):
            g = cam.grab()
            if g is None:
                continue
            _, last_left, last_right = g
            if i < WARMUP:
                continue
            gray = cv2.cvtColor(last_left, cv2.COLOR_BGR2GRAY)
            results, _ = registration.detect_boards(gray, boards, cam.K, cam.dist)
            for res in results:
                if res.index in collected and res.has_pose \
                        and res.n_corners >= args.min_corners:
                    collected[res.index]["r"].append(res.rvec.flatten())
                    collected[res.index]["t"].append(res.tvec.flatten())
                    collected[res.index]["c"].append(res.n_corners)
            if max(len(collected[b]["r"]) for b in target) >= args.frames:
                break
        K, dist = cam.K.copy(), cam.dist.copy()
        img_h, img_w = last_left.shape[:2]

    # per-board base pose
    per_board = {}
    for b in target:
        data = collected[b]
        if len(data["r"]) < 10:
            print(f"board {b}: only {len(data['r'])} frames — skipped")
            continue
        rot, t, stats = robust_average_pose(data["r"], data["t"])
        cam_T_board = homogeneous(rot, t)
        cam_T_base = cam_T_board @ np.linalg.inv(markers[b])
        per_board[b] = {"cam_T_board": cam_T_board, "cam_T_base": cam_T_base,
                        "stats": stats, "mean_corners": float(np.mean(data["c"]))}
        print(f"board {b} (ids {b*8+1}..{b*8+8}): {stats['n_used']}/{stats['n_total']} "
              f"frames, corners {per_board[b]['mean_corners']:.1f}, "
              f"t_std(mm)={np.round(stats['t_std_mm'],2)}, rot_std={stats['rot_std_deg']:.3f} deg")

    if not per_board:
        raise SystemExit("registration failed: no configured board seen enough.")

    # consistency check between boards (both should give the same base)
    if len(per_board) > 1:
        bs = list(per_board)
        A, B = per_board[bs[0]]["cam_T_base"], per_board[bs[1]]["cam_T_base"]
        dt_mm = np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000.0
        dr_deg = np.degrees((Rot.from_matrix(A[:3, :3]).inv()
                             * Rot.from_matrix(B[:3, :3])).magnitude())
        print(f"base agreement between boards {bs}: "
              f"{dt_mm:.1f} mm, {dr_deg:.2f} deg "
              f"({'OK' if dt_mm < 5 and dr_deg < 2 else 'CHECK transforms/config'})")

    primary = max(per_board, key=lambda b: per_board[b]["stats"]["n_used"])
    cam_T_base = per_board[primary]["cam_T_base"]
    cam_T_board = per_board[primary]["cam_T_board"]
    roi = workspace_roi(K, cam_T_base, workspace, img_w, img_h)
    print(f"\nprimary board {primary}; base position in camera (mm): "
          f"{np.round(cam_T_base[:3,3]*1000,1)}")
    print(f"workspace 2D ROI (x,y,w,h): {roi}")

    out_path = os.path.join(args.outdir, "camera_base_registration.npz")
    np.savez(
        out_path,
        primary_board=primary,
        cam_T_base=cam_T_base,
        cam_T_board=cam_T_board,
        K=K, dist=dist,
        resolution=args.resolution,
        workspace_x=workspace["x"], workspace_y=workspace["y"],
        workspace_z=workspace["z"], workspace_margin_px=workspace["margin_px"],
        roi_xywh=np.asarray(roi),
        boards_used=np.asarray(sorted(per_board.keys())),
    )
    print("saved:", out_path)

    if args.show and last_left is not None:
        # Right rectified view shares K; a left-camera-frame pose expressed in the
        # right camera frame is shifted by -baseline along x.
        T_shift = np.eye(4)
        T_shift[0, 3] = -baseline
        poses_l = {b: per_board[b]["cam_T_board"] for b in per_board}
        poses_r = {b: T_shift @ per_board[b]["cam_T_board"] for b in per_board}
        vis_l = annotate_view(last_left, boards, K, dist,
                              poses_l, cam_T_base, workspace, AXIS_LEN_M)
        vis_r = annotate_view(last_right, boards, K, dist,
                              poses_r, T_shift @ cam_T_base, workspace, AXIS_LEN_M)
        pl = os.path.join(args.outdir, "registration_left.png")
        pr = os.path.join(args.outdir, "registration_right.png")
        cv2.imwrite(pl, vis_l)
        cv2.imwrite(pr, vis_r)
        print(f"overlays: {pl}\n          {pr}\n  (markers+ids, ChArUco corners, "
              "board axes small, base axes LARGE, workspace box orange, ROI green)")


if __name__ == "__main__":
    main()
