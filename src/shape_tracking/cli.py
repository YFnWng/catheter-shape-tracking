"""Record time-stamped stereo images/SVO from a ZED2 and track the ChArUco base.

Entry point:  python -m shape_tracking   (or the `shape-tracking-record` script)

Controls (focus the OpenCV preview window)
    r : start/stop saving rectified LEFT+RIGHT PNG pairs
    v : start/stop unified SVO2 + Aurora recording (when EM is configured)
    s : save a single LEFT+RIGHT snapshot right now
    q / ESC : quit
"""

import argparse
import csv
from collections import deque
import os
import time

import numpy as np

from . import boards as boards_mod
from . import registration
from .boards import AXIS_LEN_M, MARKERS_PER_BOARD


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="shape_tracking",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", default="HD1080",
                    choices=["HD2K", "HD1080", "HD720", "VGA"],
                    help="ZED capture resolution (default HD1080).")
    ap.add_argument("--fps", type=int, default=30, help="capture fps (0 = max).")
    ap.add_argument("--outdir",
                    default=os.environ.get(
                        "CATHETER_SESSION_ROOT",
                        r"D:\robot-dev\catheter_sessions"),
                    help="session root (default D:\\robot-dev\\catheter_sessions).")
    ap.add_argument("--sharpness", type=int, default=4,
                    help="ZED digital sharpness 0..8 (fixed-focus; digital unsharp, "
                         "not real focus). Default 4 (SDK default); try 6-8 for "
                         "crisper edges.")
    ap.add_argument("--exposure", type=int, default=-1,
                    help="Manual exposure 0..100 (%% of max time); disables auto. "
                         "Lower = less motion blur (needs more light/gain). "
                         "Default -1 = auto.")
    ap.add_argument("--gain", type=int, default=-1,
                    help="Manual gain 0..100; disables auto. Raise to compensate a "
                         "short exposure (adds noise). Default -1 = auto.")
    ap.add_argument("--no-display", action="store_true",
                    help="headless: no preview window.")
    ap.add_argument("--autorecord", action="store_true",
                    help="start unified ZED+Aurora recording immediately.")
    ap.add_argument('--aurora-port', default='',
                    help='Aurora serial port, e.g. COM4; empty disables EM.')
    ap.add_argument('--aurora-expected-tools', type=int, default=3,
                    help='required Aurora tool count (default 3; 0 accepts any).')
    ap.add_argument(
        '--aurora-probe-part-number', default='610175 T6E0-S00923',
        help='PHINF part number used to identify the base registration probe.')
    ap.add_argument(
        '--aurora-driver-root',
        default=os.environ.get(
            'SLICER_ROBOT_ROOT',
            r'D:\GaTech Dropbox\Yifan Wang\SlicerRobot'),
        help='SlicerRobot source root containing the standalone Aurora driver.')
    ap.add_argument(
        '--registration-config',
        default=os.environ.get(
            'SHAPE_TRACKING_REGISTRATION_CONFIG',
            os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'registration_config.yaml'))),
        help='YAML containing EM bracket slot centers in the robot-base frame.')
    ap.add_argument(
        '--camera-registration-frames', type=int, default=150,
        help='valid ChArUco poses collected after EM registration.')
    ap.add_argument(
        '--camera-registration-min-corners', type=int, default=6,
        help='minimum ChArUco corners for a camera-registration observation.')
    return ap.parse_args(argv)


def main(argv=None):
    import cv2                      # local import: keeps --help fast, errors clear
    from . import register as camera_register
    from .zed_capture import ZedCamera

    args = parse_args(argv)

    em_recorder = None
    if args.aurora_port:
        import atexit
        from .aurora_capture import AuroraRecorder
        print(f'Opening Aurora on {args.aurora_port} ...')
        em_recorder = AuroraRecorder(
            args.aurora_port,
            driver_root=args.aurora_driver_root,
            expected_tools=args.aurora_expected_tools,
            probe_part_number=args.aurora_probe_part_number,
            registration_config=args.registration_config,
        )
        try:
            em_recorder.open()
        except Exception:
            em_recorder.close()
            raise
        atexit.register(em_recorder.close)
        print(f'Aurora ready: {em_recorder.tool_count} tools')
        for item in sorted(
                em_recorder.tool_info.values(), key=lambda value: value['role']):
            print(
                f"  {item['role']}: handle={item['port_handle']} "
                f"serial={item['serial_number']} part={item['part_number']}")

    session = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.outdir, session)
    left_dir = os.path.join(session_dir, "left")
    right_dir = os.path.join(session_dir, "right")
    for d in (left_dir, right_dir):
        os.makedirs(d, exist_ok=True)

    _, boards = boards_mod.build_boards()
    camera_markers, _ = camera_register.load_config(args.registration_config)
    camera_observations = {
        index: {
            'r': deque(maxlen=args.camera_registration_frames),
            't': deque(maxlen=args.camera_registration_frames),
            'c': deque(maxlen=args.camera_registration_frames),
        } for index in camera_markers
    }

    with ZedCamera(args.resolution, args.fps, args.sharpness,
                   args.exposure, args.gain) as cam:
        info = cam.info()
        print(f"ZED serial {info['serial']}, model {info['model']}, SDK {info['sdk']}")
        print(f"Resolution {info['resolution']}  fx={info['fx']:.1f} fy={info['fy']:.1f} "
              f"cx={info['cx']:.1f} cy={info['cy']:.1f}  (fixed focus)")
        print(f"Session dir: {session_dir}")

        np.savez(os.path.join(session_dir, "left_intrinsics.npz"),
                 K=cam.K, dist=cam.dist, fx=info["fx"], fy=info["fy"],
                 cx=info["cx"], cy=info["cy"], resolution=info["resolution"],
                 baseline_m=info["baseline_m"])

        pose_csv = open(
            os.path.join(session_dir, "board_poses.csv"),
            "w", newline="", buffering=1)
        pose_writer = csv.writer(pose_csv)
        pose_writer.writerow(["timestamp_ns", "board", "n_corners",
                              "tx_m", "ty_m", "tz_m", "rx", "ry", "rz"])

        recording_pngs = False
        captured_registration_slots = set()
        camera_registration_collecting = False
        camera_registration_done = False
        camera_registration_notice = None
        last_registration_pair = None
        frame_idx = 0
        svo_frame = 0
        frame_index_file = None
        frame_index_writer = None

        def svo_path():
            return os.path.join(session_dir, f"stereo_{session}.svo2")

        def start_svo():
            nonlocal frame_index_file, frame_index_writer, svo_frame
            if not cam.start_recording(svo_path()):
                return False
            # Sidecar mapping each SVO frame -> absolute capture time so offline
            # alignment needs no SVO decode. timestamp_ns is epoch ns on the
            # Windows host clock (ZED TIME_REFERENCE.IMAGE).
            frame_index_file = open(
                os.path.join(session_dir, "frame_index.csv"), "w", newline="")
            frame_index_writer = csv.writer(frame_index_file)
            frame_index_writer.writerow(["svo_frame", "timestamp_ns"])
            svo_frame = 0
            print(f"[SVO] recording -> {svo_path()}")
            if em_recorder is not None:
                try:
                    em_recorder.start(session_dir)
                    em_path = os.path.join(session_dir, 'em_poses.csv')
                    print(
                        f'[EM] recording {em_recorder.tool_count} tools -> '
                        f'{em_path}')
                except Exception:
                    cam.stop_recording()
                    if frame_index_file is not None:
                        frame_index_file.close()
                        frame_index_file = None
                        frame_index_writer = None
                    raise
            return True

        def stop_svo():
            nonlocal frame_index_file, frame_index_writer
            cam.stop_recording()
            if frame_index_file is not None:
                frame_index_file.close()
                frame_index_file = None
                frame_index_writer = None
            print("[SVO] stopped")

            if em_recorder is not None and em_recorder.is_recording:
                em_recorder.stop()
                print(
                    f'[EM] stopped: {em_recorder.samples} frames, '
                    f'{em_recorder.valid_rows}/{em_recorder.rows} valid poses')

        def finish_camera_registration_if_ready():
            nonlocal camera_registration_collecting
            nonlocal camera_registration_done, camera_registration_notice
            if (not camera_registration_collecting or camera_registration_done
                    or last_registration_pair is None):
                return
            counts = {
                index: len(data['r'])
                for index, data in camera_observations.items()}
            if not counts or max(counts.values()) < args.camera_registration_frames:
                notice = (
                    f'waiting for {args.camera_registration_frames} valid '
                    f'ChArUco frames; counts={counts}')
                if notice != camera_registration_notice:
                    print(f'[CAM-REG] {notice}')
                    camera_registration_notice = notice
                return
            image_ts, image_left, image_right = last_registration_pair
            try:
                camera = camera_register.save_session_camera_registration(
                    registration_path=os.path.join(
                        session_dir, 'registration.json'),
                    output_dir=session_dir,
                    config_path=args.registration_config,
                    collected=camera_observations,
                    left_bgr=image_left,
                    right_bgr=image_right,
                    boards=boards,
                    K=cam.K,
                    dist=cam.dist,
                    resolution=info['resolution'],
                    zed_serial=info['serial'],
                    baseline_m=info['baseline_m'],
                    image_timestamp_ns=image_ts,
                    min_frames=args.camera_registration_frames,
                )
            except (OSError, TypeError, ValueError) as exc:
                notice = str(exc)
                if notice != camera_registration_notice:
                    print(f'[CAM-REG] not completed: {notice}')
                    camera_registration_notice = notice
                return
            camera_registration_done = True
            camera_registration_collecting = False
            camera_registration_notice = None
            print(
                f"[CAM-REG] solved with board {camera['primary_board']}; "
                f"left ROI={camera['roi_left_xywh']} "
                f"right ROI={camera['roi_right_xywh']}")
            print(
                '[CAM-REG] saved registration.json, '
                'registration_left.png, registration_right.png')

        if args.autorecord:
            start_svo()

        show = not args.no_display
        win = "ZED2 ChArUco tracker"
        if show:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1280, 720)

        print("\nControls: r=PNGs v=SVO+EM 1..4=capture probe slot "
              "s=snapshot q/ESC=quit\n")

        try:
            while True:
                grabbed = cam.grab()
                if em_recorder is not None and em_recorder.error is not None:
                    raise RuntimeError(
                        f'Aurora recording failed: {em_recorder.error}'
                    ) from em_recorder.error
                if grabbed is None:
                    continue
                ts, left_bgr, right_bgr = grabbed
                if frame_index_writer is not None:      # one SVO frame per grab
                    frame_index_writer.writerow([svo_frame, ts])
                    svo_frame += 1
                overlay = left_bgr.copy()
                results, seen_ids = [], []
                if camera_registration_collecting and cam.is_recording:
                    gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
                    results, seen_ids = registration.detect_boards(
                        gray, boards, cam.K, cam.dist)
                valid_registration_observation = False
                for res in results:
                    registration.draw_pose(overlay, res, cam.K, cam.dist, AXIS_LEN_M)
                    if res.has_pose:
                        t = res.tvec.flatten()
                        r = res.rvec.flatten()
                        cv2.putText(overlay,
                                    f"B{res.index}: {t[2]*100:5.1f}cm ({res.n_corners} pts)",
                                    (10, 30 + 30 * res.index),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        if (res.index in camera_observations
                                and res.n_corners >=
                                args.camera_registration_min_corners):
                            data = camera_observations[res.index]
                            data['r'].append(r)
                            data['t'].append(t)
                            data['c'].append(res.n_corners)
                            pose_writer.writerow([
                                ts, res.index, res.n_corners,
                                t[0], t[1], t[2], r[0], r[1], r[2]])
                            valid_registration_observation = True
                if valid_registration_observation:
                    last_registration_pair = (ts, left_bgr, right_bgr)
                finish_camera_registration_if_ready()

                status_txt = []
                if recording_pngs:
                    status_txt.append("REC-PNG")
                if cam.is_recording:
                    status_txt.append("REC-SVO")
                if captured_registration_slots:
                    status_txt.append(
                        f"REG={len(captured_registration_slots)}/4")
                if camera_registration_done:
                    status_txt.append('CAM-REG')
                elif camera_registration_collecting:
                    count = max(
                        (len(data['r']) for data in
                         camera_observations.values()), default=0)
                    status_txt.append(
                        f'CAM-REG={count}/{args.camera_registration_frames}')
                cv2.putText(overlay, f"ids={seen_ids}  {' '.join(status_txt)}",
                            (10, overlay.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                if recording_pngs:
                    cv2.imwrite(os.path.join(left_dir, f"{ts}.png"), left_bgr)
                    cv2.imwrite(os.path.join(right_dir, f"{ts}.png"), right_bgr)
                    frame_idx += 1

                if show:
                    cv2.imshow(win, overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                    elif key == ord('r'):
                        recording_pngs = not recording_pngs
                        print(f"[PNG] {'ON' if recording_pngs else 'OFF'} "
                              f"({frame_idx} frames so far)")
                    elif key == ord('v'):
                        if not cam.is_recording:
                            start_svo()
                        else:
                            stop_svo()
                    elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                        slot = int(chr(key))
                        if em_recorder is None or not em_recorder.is_recording:
                            print('[REG] start SVO+EM recording before capture')
                        else:
                            try:
                                result = em_recorder.capture_registration_slot(slot)
                                captured_registration_slots.add(slot)
                                print(
                                    f"[REG] slot {slot}: "
                                    f"mean_mm={result['mean_aurora_mm']} "
                                    f"max_std_mm={result['max_std_mm']:.3f} "
                                    f"n={result['sample_count']}")
                                if result['transform_status'] == 'solved':
                                    fit = result['registration_fit']
                                    print(
                                        '[REG] transform solved: '
                                        f"RMS={fit['rms_residual_mm']:.3f} mm, "
                                        f"max={fit['max_residual_mm']:.3f} mm")
                                    if (not camera_registration_collecting
                                            and not camera_registration_done):
                                        for data in camera_observations.values():
                                            data['r'].clear()
                                            data['t'].clear()
                                            data['c'].clear()
                                        last_registration_pair = None
                                        camera_registration_notice = None
                                        camera_registration_collecting = True
                                        print(
                                            '[CAM-REG] EM registration complete; '
                                            'remove the probe and expose a '
                                            'configured ChArUco board. Collecting '
                                            f'{args.camera_registration_frames} '
                                            'valid poses now.')
                                elif result['registration_complete']:
                                    print(
                                        '[REG] measurements saved; transform not '
                                        f"solved: {result['transform_error']}")
                            except RuntimeError as exc:
                                print(f'[REG] slot {slot} not captured: {exc}')
                    elif key == ord('s'):
                        cv2.imwrite(os.path.join(left_dir, f"snap_{ts}.png"), left_bgr)
                        cv2.imwrite(os.path.join(right_dir, f"snap_{ts}.png"), right_bgr)
                        print(f"[snap] {ts}")
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            pose_csv.close()
            if cam.is_recording:
                stop_svo()
            if show:
                cv2.destroyAllWindows()
            if em_recorder is not None:
                em_recorder.close()
            print(f"Done. Data in {session_dir}")


if __name__ == "__main__":
    main()
