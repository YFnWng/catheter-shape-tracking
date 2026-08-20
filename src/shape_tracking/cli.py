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
import json
import os
import time

import numpy as np
import yaml

from . import boards as boards_mod
from . import registration
from .boards import AXIS_LEN_M, MARKERS_PER_BOARD


CAMERA_CONFIG_KEYS = {
    "resolution", "fps", "sharpness", "exposure", "gain",
    "white_balance_temperature", "white_balance_auto_freeze_s",
    "white_balance_auto_freeze_retries",
    "brightness", "contrast", "hue",
    "saturation", "gamma", "svo_compression",
}


def _default_camera_config_path():
    return os.environ.get(
        "SHAPE_TRACKING_CAMERA_CONFIG",
        os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "camera_config.yaml")))


def _load_camera_config(path):
    """Load and flatten the camera/recording sections of a capture profile."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError("camera config must contain a YAML mapping")
    top_level = set(document) - {"camera", "recording"}
    if top_level:
        raise ValueError(
            "unknown camera config section(s): " + ", ".join(sorted(top_level)))
    camera = document.get("camera", {}) or {}
    recording = document.get("recording", {}) or {}
    if not isinstance(camera, dict) or not isinstance(recording, dict):
        raise ValueError("camera and recording config sections must be mappings")
    values = dict(camera)
    values.update(recording)
    unknown = set(values) - CAMERA_CONFIG_KEYS
    if unknown:
        raise ValueError(
            "unknown camera config setting(s): " + ", ".join(sorted(unknown)))
    return values


def _limit_frame_index(path, frame_count):
    """Atomically retain at most ``frame_count`` rows in an SVO sidecar."""
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = rows[:max(0, int(frame_count))]
    temporary = path + ".finalizing"
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["svo_frame", "timestamp_ns"])
        writer.writerows(
            (index, row["timestamp_ns"]) for index, row in enumerate(rows))
    os.replace(temporary, path)
    return len(rows)


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--camera-config", default=_default_camera_config_path())
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        camera_defaults = _load_camera_config(config_args.camera_config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        config_parser.error(
            f"could not load --camera-config {config_args.camera_config!r}: {exc}")

    ap = argparse.ArgumentParser(
        prog="shape_tracking",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--camera-config", default=config_args.camera_config,
        help="camera/recording YAML profile. CLI settings override YAML.")
    ap.add_argument("--resolution", default=camera_defaults.get("resolution", "HD1080"),
                    choices=["HD2K", "HD1080", "HD720", "VGA"],
                    help="ZED capture resolution (default HD1080).")
    ap.add_argument("--fps", type=int, default=camera_defaults.get("fps", 30),
                    help="capture fps (0 = max).")
    ap.add_argument("--outdir",
                    default=os.environ.get(
                        "CATHETER_SESSION_ROOT",
                        r"D:\robot-dev\catheter_sessions"),
                    help="session root (default D:\\robot-dev\\catheter_sessions).")
    ap.add_argument("--sharpness", type=int,
                    default=camera_defaults.get("sharpness", 4),
                    help="ZED digital sharpness 0..8 (fixed-focus; digital unsharp, "
                         "not real focus). Default 4 (SDK default); try 6-8 for "
                         "crisper edges.")
    ap.add_argument("--exposure", type=int,
                    default=camera_defaults.get("exposure", -1),
                    help="Manual exposure 0..100 (%% of max time); disables auto. "
                         "Lower = less motion blur (needs more light/gain). "
                         "Default -1 = auto.")
    ap.add_argument("--gain", type=int,
                    default=camera_defaults.get("gain", -1),
                    help="Manual gain 0..100; disables auto. Raise to compensate a "
                         "short exposure (adds noise). Default -1 = auto.")
    ap.add_argument(
        "--white-balance-temperature", "--whitebalance-temperature",
        dest="white_balance_temperature", type=int,
        default=camera_defaults.get("white_balance_temperature", -1),
        help="Manual white-balance color temperature in kelvin; disables auto "
             "white balance. Default -1 = auto.")
    ap.add_argument(
        "--white-balance-auto-freeze-s", type=float,
        default=camera_defaults.get("white_balance_auto_freeze_s", 0.0),
        help="Run auto white balance for this many unrecorded seconds, then "
             "disable auto without setting a manual temperature (default 0).")
    ap.add_argument(
        "--white-balance-auto-freeze-retries", type=int,
        default=camera_defaults.get("white_balance_auto_freeze_retries", 2),
        help="freeze retries after a failed stereo color check before falling "
             "back to continuous auto white balance (default 2).")
    ap.add_argument("--brightness", type=int,
                    default=camera_defaults.get("brightness"),
                    help="Fixed ZED brightness; omitted keeps SDK default.")
    ap.add_argument("--contrast", type=int,
                    default=camera_defaults.get("contrast"),
                    help="Fixed ZED contrast; omitted keeps SDK default.")
    ap.add_argument("--hue", type=int, default=camera_defaults.get("hue"),
                    help="Fixed ZED hue; omitted keeps SDK default.")
    ap.add_argument("--saturation", type=int,
                    default=camera_defaults.get("saturation"),
                    help="Fixed ZED saturation; omitted keeps SDK default.")
    ap.add_argument("--gamma", type=int, default=camera_defaults.get("gamma"),
                    help="Fixed ZED gamma; omitted keeps SDK default.")
    ap.add_argument(
        "--svo-compression",
        default=camera_defaults.get("svo_compression", "H264"),
        choices=["H264", "H265", "LOSSLESS", "H264_LOSSLESS", "H265_LOSSLESS"],
        help="SVO2 compression mode (default H264).")
    ap.add_argument("--no-display", action="store_true",
                    help="headless: no preview window.")
    ap.add_argument(
        "--preview-fps", type=float, default=10.0,
        help="image retrieval/preview rate while SVO recording (default 10Hz; "
             "SVO grab and recording remain at --fps). Set 0 to retrieve every "
             "recorded frame.")
    ap.add_argument("--autorecord", action="store_true",
                    help="start unified ZED+Aurora recording immediately.")
    ap.add_argument(
        "--image-only", action="store_true",
        help="record stereo images without Aurora; register only the camera "
             "to the robot-base ChArUco boards.")
    ap.add_argument('--aurora-port', default='',
                    help='Aurora serial port, e.g. COM4; empty disables EM.')
    ap.add_argument('--aurora-expected-tools', type=int, default=3,
                    help='required Aurora tool count (default 3; use 2 in direct '
                         'optical mode if the registration probe is unplugged; '
                         '0 accepts any).')
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
        '--em-registration-dwell-s', type=float, default=0.75,
        help='stationary probe history required for each slot (default 0.75s).')
    ap.add_argument(
        '--em-registration-min-samples', type=int, default=20,
        help='minimum valid probe samples in a slot capture (default 20).')
    ap.add_argument(
        '--em-registration-max-position-deviation-mm',
        type=float, default=0.15,
        help='maximum 95th-percentile probe position deviation (default 0.15mm).')
    ap.add_argument(
        '--em-registration-max-orientation-deviation-deg',
        type=float, default=1.0,
        help='maximum 95th-percentile probe orientation deviation (default 1deg).')
    ap.add_argument(
        '--camera-registration-frames', type=int, default=150,
        help='valid ChArUco poses collected after EM registration.')
    ap.add_argument(
        '--camera-registration-min-corners', type=int, default=6,
        help='minimum ChArUco corners for a camera-registration observation.')
    args = ap.parse_args(argv)
    if args.image_only and args.aurora_port:
        ap.error("--image-only cannot be combined with --aurora-port")
    if (args.exposure >= 0) != (args.gain >= 0):
        ap.error("--exposure and --gain must be provided together, or both left auto")
    if args.exposure < -1 or args.gain < -1:
        ap.error("--exposure and --gain accept -1 for auto or a non-negative value")
    if args.white_balance_temperature < -1:
        ap.error("--white-balance-temperature accepts -1 for auto or kelvin")
    if args.white_balance_temperature >= 0 and (
            not 2800 <= args.white_balance_temperature <= 6500
            or args.white_balance_temperature % 100 != 0):
        ap.error(
            "--white-balance-temperature must be -1 for auto, or an exact "
            "100 K step from 2800 through 6500")
    if args.white_balance_auto_freeze_s < 0:
        ap.error("--white-balance-auto-freeze-s must be non-negative")
    if args.white_balance_auto_freeze_retries < 0:
        ap.error("--white-balance-auto-freeze-retries must be non-negative")
    if (args.white_balance_auto_freeze_s > 0
            and args.white_balance_temperature >= 0):
        ap.error(
            "auto-freeze and a manual white-balance temperature are mutually "
            "exclusive; set white_balance_temperature: -1")
    return args


def main(argv=None):
    import cv2                      # local import: keeps --help fast, errors clear
    from . import register as camera_register
    from .zed_capture import ZedCamera, assess_stereo_color

    args = parse_args(argv)
    field_registration = (
        None if args.image_only else
        camera_register.load_field_generator_config(args.registration_config))

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
            em_registration_dwell_s=args.em_registration_dwell_s,
            em_registration_min_samples=args.em_registration_min_samples,
            em_registration_max_position_deviation_mm=(
                args.em_registration_max_position_deviation_mm),
            em_registration_max_orientation_deviation_deg=(
                args.em_registration_max_orientation_deviation_deg),
            require_probe=(field_registration is None),
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
    metadata_path = os.path.join(session_dir, "session_metadata.json")
    metadata = {
            "schema_version": 2,
            "session_id": session,
            "mode": (
                "image_only" if args.image_only else
                "camera_em" if args.aurora_port else "camera_only"),
            "modalities": {
                "stereo_camera": True,
                "em_tracking": bool(args.aurora_port),
            },
            "requested_resolution": args.resolution,
            "requested_fps": args.fps,
            "preview_fps": args.preview_fps,
            "camera_config": (
                os.path.abspath(args.camera_config) if args.camera_config else None),
            "svo_compression": args.svo_compression,
            "camera_settings_requested": {
                "exposure": args.exposure,
                "gain": args.gain,
                "white_balance_temperature": args.white_balance_temperature,
                "white_balance_auto_freeze_s":
                    args.white_balance_auto_freeze_s,
                "white_balance_auto_freeze_retries":
                    args.white_balance_auto_freeze_retries,
                "brightness": args.brightness,
                "contrast": args.contrast,
                "hue": args.hue,
                "saturation": args.saturation,
                "gamma": args.gamma,
                "sharpness": args.sharpness,
            },
        }
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)

    camera_markers, _ = camera_register.load_config(args.registration_config)
    additional_boards = []
    if field_registration is not None:
        if field_registration['board_index'] in camera_markers:
            raise ValueError(
                'field-generator board index must differ from base boards')
        additional_boards.append((
            field_registration['board_index'],
            field_registration['marker_id_offset']))
    _, boards = boards_mod.build_boards(additional=additional_boards)
    observation_indices = set(camera_markers)
    if field_registration is not None:
        observation_indices.add(field_registration['board_index'])
    camera_observations = {
        index: {
            'r': deque(maxlen=args.camera_registration_frames),
            't': deque(maxlen=args.camera_registration_frames),
            'c': deque(maxlen=args.camera_registration_frames),
        } for index in observation_indices
    }

    with ZedCamera(
            resolution=args.resolution, fps=args.fps,
            sharpness=args.sharpness, exposure=args.exposure, gain=args.gain,
            white_balance_temperature=args.white_balance_temperature,
            white_balance_auto_freeze_s=args.white_balance_auto_freeze_s,
            white_balance_auto_freeze_retries=
                args.white_balance_auto_freeze_retries,
            brightness=args.brightness, contrast=args.contrast, hue=args.hue,
            saturation=args.saturation, gamma=args.gamma,
            svo_compression=args.svo_compression) as cam:
        info = cam.info()
        print(f"ZED serial {info['serial']}, model {info['model']}, SDK {info['sdk']}")
        print(f"Resolution {info['resolution']}  fx={info['fx']:.1f} fy={info['fy']:.1f} "
              f"cx={info['cx']:.1f} cy={info['cy']:.1f}  (fixed focus)")
        print("Camera settings: " + ", ".join(
            f"{key}={value}" for key, value in
            sorted(info["camera_settings"].items())))
        if info["white_balance_freeze"] is not None:
            freeze = info["white_balance_freeze"]
            print(
                "[WB] auto-freeze complete: "
                f"{freeze['frames_grabbed']} warm-up frames, "
                f"{freeze['actual_warmup_s']:.2f}s, "
                f"temperature {freeze['temperature_before_freeze']} -> "
                f"{freeze['temperature_after_freeze']}, "
                f"auto={freeze['whitebalance_auto_after_freeze']}, "
                f"attempts={len(freeze['attempts'])}, "
                f"fallback_auto={freeze['fallback_to_continuous_auto']}")
        print(f"Session dir: {session_dir}")

        metadata["camera"] = info
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)

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
        recording_grabs = 0
        repeated_svo_timestamps = 0
        last_svo_timestamp_ns = None
        last_stereo_color_health = None
        unhealthy_recording_checks = 0
        pending_autorecord = bool(args.autorecord)
        autorecord_deadline = time.monotonic() + 15.0
        frame_index_file = None
        frame_index_writer = None
        preview_stride = (
            1 if args.preview_fps <= 0 or args.fps <= 0 else
            max(1, int(round(args.fps / args.preview_fps))))

        def svo_path():
            return os.path.join(session_dir, f"stereo_{session}.svo2")

        def start_svo(color_health):
            nonlocal frame_index_file, frame_index_writer, svo_frame
            nonlocal recording_grabs, repeated_svo_timestamps
            nonlocal last_svo_timestamp_ns
            nonlocal camera_registration_collecting
            nonlocal camera_registration_done, camera_registration_notice
            nonlocal last_registration_pair
            if color_health is None or not color_health["healthy"]:
                print(
                    "[COLOR] SVO start blocked: stereo color preflight failed; "
                    f"health={color_health}")
                return False
            if not cam.start_recording(svo_path()):
                return False
            metadata["recording_stereo_color_preflight"] = color_health
            with open(metadata_path, "w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2, sort_keys=True)
            # Sidecar mapping each SVO frame -> absolute capture time so offline
            # alignment needs no SVO decode. timestamp_ns is epoch ns on the
            # Windows host clock (ZED TIME_REFERENCE.IMAGE).
            frame_index_file = open(
                os.path.join(session_dir, "frame_index.csv"), "w", newline="")
            frame_index_writer = csv.writer(frame_index_file)
            frame_index_writer.writerow(["svo_frame", "timestamp_ns"])
            svo_frame = 0
            recording_grabs = 0
            repeated_svo_timestamps = 0
            last_svo_timestamp_ns = None
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
            if field_registration is not None or args.image_only:
                for data in camera_observations.values():
                    data['r'].clear()
                    data['t'].clear()
                    data['c'].clear()
                last_registration_pair = None
                camera_registration_notice = None
                camera_registration_done = False
                camera_registration_collecting = True
                if args.image_only:
                    print(
                        '[REG] image-only camera registration: collecting '
                        f'robot-base boards for '
                        f'{args.camera_registration_frames} valid frames')
                else:
                    print(
                        '[REG] optical EM registration: collecting base board '
                        f"and field board IDs "
                        f"{field_registration['marker_id_offset']}-"
                        f"{field_registration['marker_id_offset'] + 7} for "
                        f'{args.camera_registration_frames} valid frames')
            return True

        def stop_svo():
            nonlocal frame_index_file, frame_index_writer, svo_frame
            recording_status = cam.stop_recording()
            if em_recorder is not None and em_recorder.is_recording:
                em_recorder.stop()
                print(
                    f'[EM] stopped: {em_recorder.samples} frames, '
                    f'{em_recorder.valid_rows}/{em_recorder.rows} valid poses')
            if frame_index_file is not None:
                frame_index_file.close()
                frame_index_file = None
                frame_index_writer = None
            playable_count = None
            try:
                playable_count = cam.playable_svo_frame_count(svo_path())
                if playable_count < svo_frame:
                    svo_frame = _limit_frame_index(
                        os.path.join(session_dir, "frame_index.csv"),
                        playable_count)
            except (OSError, RuntimeError) as exc:
                print(f"[warn] could not reconcile SVO frame count: {exc}")
            detail = (
                f" indexed={svo_frame} grabs={recording_grabs} "
                f"repeated_timestamps={repeated_svo_timestamps}")
            if playable_count is not None:
                detail += f" playable={playable_count}"
            if recording_status is not None:
                detail += (
                    " ingested="
                    f"{recording_status['number_frames_ingested']} encoded="
                    f"{recording_status['number_frames_encoded']} avg_encode_ms="
                    f"{recording_status['average_compression_time_ms']:.2f}")
            print(f"[SVO] stopped:{detail}")

        def finish_camera_registration_if_ready():
            nonlocal camera_registration_collecting
            nonlocal camera_registration_done, camera_registration_notice
            if (not camera_registration_collecting or camera_registration_done
                    or last_registration_pair is None):
                return
            counts = {
                index: len(data['r'])
                for index, data in camera_observations.items()}
            if field_registration is not None:
                base_count = max(
                    (counts.get(index, 0) for index in camera_markers),
                    default=0)
                field_count = counts.get(
                    field_registration['board_index'], 0)
                ready = (
                    base_count >= args.camera_registration_frames
                    and field_count >= args.camera_registration_frames)
            else:
                ready = bool(counts) and (
                    max(counts.values()) >= args.camera_registration_frames)
            if not ready:
                notice = (
                    f'waiting for {args.camera_registration_frames} valid '
                    f'ChArUco frames; counts={counts}')
                if notice != camera_registration_notice:
                    print(f'[CAM-REG] {notice}')
                    camera_registration_notice = notice
                return
            image_ts, image_left, image_right = last_registration_pair
            try:
                if em_recorder is None and not args.image_only:
                    raise RuntimeError(
                        'camera registration overlay requires live EM tracking')
                em_tool_poses = (
                    None if args.image_only else
                    em_recorder.tip_coil_poses_at(image_ts))
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
                    em_tool_poses=em_tool_poses,
                    image_only=args.image_only,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
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

        show = not args.no_display
        win = "ZED2 ChArUco tracker"
        if show:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1600, 500)

        if args.image_only:
            print("\nControls: r=PNGs v=SVO s=snapshot q/ESC=quit; "
                  "preview=LEFT|RIGHT\n")
        elif field_registration is not None:
            print("\nControls: r=PNGs v=SVO+EM s=snapshot q/ESC=quit; "
                  "preview=LEFT|RIGHT\n")
        else:
            print("\nControls: r=PNGs v=SVO+EM 1..4=capture probe slot "
                  "s=snapshot q/ESC=quit; preview=LEFT|RIGHT\n")

        try:
            while True:
                retrieve_images = (
                    not cam.is_recording
                    or recording_pngs
                    or recording_grabs % preview_stride == 0)
                grabbed = cam.grab(retrieve=retrieve_images)
                if em_recorder is not None and em_recorder.error is not None:
                    raise RuntimeError(
                        f'Aurora recording failed: {em_recorder.error}'
                    ) from em_recorder.error
                if grabbed is None:
                    continue
                ts, left_bgr, right_bgr = grabbed
                if frame_index_writer is not None:
                    recording_grabs += 1
                    # A successful SDK grab can repeat the preceding camera
                    # timestamp. Such a grab is not a new SVO image and must not
                    # advance the playback index.
                    if ts != last_svo_timestamp_ns:
                        frame_index_writer.writerow([svo_frame, ts])
                        svo_frame += 1
                        last_svo_timestamp_ns = ts
                    else:
                        repeated_svo_timestamps += 1
                if left_bgr is None:
                    continue
                last_stereo_color_health = assess_stereo_color(
                    left_bgr, right_bgr)
                if pending_autorecord:
                    if last_stereo_color_health["healthy"]:
                        if start_svo(last_stereo_color_health):
                            pending_autorecord = False
                    elif time.monotonic() >= autorecord_deadline:
                        raise RuntimeError(
                            "SVO autorecord blocked for 15s by failed stereo "
                            f"color preflight: {last_stereo_color_health}")
                if cam.is_recording:
                    if last_stereo_color_health["healthy"]:
                        unhealthy_recording_checks = 0
                    else:
                        unhealthy_recording_checks += 1
                        if unhealthy_recording_checks >= 3:
                            metadata["stereo_color_fault"] = {
                                "timestamp_ns": int(ts),
                                "health": last_stereo_color_health,
                            }
                            with open(
                                    metadata_path, "w", encoding="utf-8") as stream:
                                json.dump(metadata, stream, indent=2, sort_keys=True)
                            stop_svo()
                            raise RuntimeError(
                                "SVO stopped after 3 consecutive failed stereo "
                                f"color checks: {last_stereo_color_health}")
                overlay = left_bgr.copy()
                right_overlay = right_bgr.copy()
                results, seen_ids = [], []
                if camera_registration_collecting and cam.is_recording:
                    gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
                    results, seen_ids = registration.detect_boards(
                        gray, boards, cam.K, cam.dist)
                valid_registration_observation = False
                valid_registration_indices = set()
                frame_registration_poses = {}
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
                            frame_registration_poses[res.index] = (
                                r, t, res.n_corners)
                if field_registration is not None:
                    field_index = field_registration['board_index']
                    base_indices = (
                        set(camera_markers) & set(frame_registration_poses))
                    if field_index in frame_registration_poses and base_indices:
                        valid_registration_indices = base_indices | {field_index}
                else:
                    valid_registration_indices = set(frame_registration_poses)
                for index in sorted(valid_registration_indices):
                    r, t, corner_count = frame_registration_poses[index]
                    data = camera_observations[index]
                    data['r'].append(r)
                    data['t'].append(t)
                    data['c'].append(corner_count)
                    pose_writer.writerow([
                        ts, index, corner_count,
                        t[0], t[1], t[2], r[0], r[1], r[2]])
                valid_registration_observation = bool(
                    valid_registration_indices)
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
                health_color = (0, 220, 0) if (
                    last_stereo_color_health["healthy"]) else (0, 0, 255)
                left_health = last_stereo_color_health["eyes"]["left"]
                right_health = last_stereo_color_health["eyes"]["right"]
                cv2.putText(
                    overlay,
                    f"LEFT color_dom={left_health['channel_dominance']:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, health_color, 2)
                cv2.putText(
                    right_overlay,
                    f"RIGHT color_dom={right_health['channel_dominance']:.2f} "
                    f"green_clip={right_health['green_clip_fraction']:.1%}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, health_color, 2)
                if pending_autorecord:
                    cv2.putText(
                        right_overlay, "AUTORECORD WAITING FOR COLOR PREFLIGHT",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if recording_pngs:
                    cv2.imwrite(os.path.join(left_dir, f"{ts}.png"), left_bgr)
                    cv2.imwrite(os.path.join(right_dir, f"{ts}.png"), right_bgr)
                    frame_idx += 1

                if show:
                    cv2.imshow(win, np.hstack((overlay, right_overlay)))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                    elif key == ord('r'):
                        recording_pngs = not recording_pngs
                        print(f"[PNG] {'ON' if recording_pngs else 'OFF'} "
                              f"({frame_idx} frames so far)")
                    elif key == ord('v'):
                        if not cam.is_recording:
                            start_svo(last_stereo_color_health)
                        else:
                            stop_svo()
                    elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                        if args.image_only:
                            print('[REG] probe slots disabled in image-only mode')
                            continue
                        if field_registration is not None:
                            print(
                                '[REG] probe slots disabled: using the optical '
                                'field-generator ChArUco board')
                            continue
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
                                    f"pos_p95_mm={result['stationarity']['position_p95_deviation_mm']:.3f} "
                                    f"rot_p95_deg={result['stationarity']['orientation_p95_deviation_deg']:.3f} "
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
