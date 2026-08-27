"""Dual-ZED live capture and single-board robot-base registration."""
from __future__ import annotations

from collections import deque
import copy
import csv
import json
import os
import time

import numpy as np

from . import boards as boards_mod
from . import registration
from .boards import AXIS_LEN_M
from .multi_camera import MultiZedCapture


def _camera_kwargs(args, serial_number):
    return {
        "resolution": args.resolution,
        "fps": args.fps,
        "sharpness": args.sharpness,
        "exposure": args.exposure,
        "gain": args.gain,
        "white_balance_temperature": args.white_balance_temperature,
        "white_balance_auto_freeze_s": args.white_balance_auto_freeze_s,
        "white_balance_auto_freeze_retries":
            args.white_balance_auto_freeze_retries,
        "brightness": args.brightness,
        "contrast": args.contrast,
        "hue": args.hue,
        "saturation": args.saturation,
        "gamma": args.gamma,
        "svo_compression": args.svo_compression,
        "serial_number": serial_number,
    }


def _open_em_recorder(args, field_registration):
    if not args.aurora_port:
        return None
    from .aurora_capture import AuroraRecorder

    print(f"Opening Aurora on {args.aurora_port} ...")
    recorder = AuroraRecorder(
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
        recorder.open()
    except Exception:
        recorder.close()
        raise
    print(f"Aurora ready: {recorder.tool_count} tools")
    for item in sorted(
            recorder.tool_info.values(), key=lambda value: value["role"]):
        print(
            f"  {item['role']}: handle={item['port_handle']} "
            f"serial={item['serial_number']} part={item['part_number']}")
    return recorder


def _observation_store(indices, maxlen):
    return {
        index: {
            "r": deque(maxlen=maxlen),
            "t": deque(maxlen=maxlen),
            "c": deque(maxlen=maxlen),
        }
        for index in indices
    }


def _clear_observations(observations):
    for rig_data in observations.values():
        for board_data in rig_data.values():
            for values in board_data.values():
                values.clear()


def _right_pose_to_left(rvec, tvec, left_camera_T_right_camera):
    """Express a right-eye board pose in the ZED left-camera frame."""
    import cv2

    right_camera_T_board = np.eye(4, dtype=np.float64)
    right_camera_T_board[:3, :3] = cv2.Rodrigues(
        np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    right_camera_T_board[:3, 3] = np.asarray(
        tvec, dtype=np.float64).reshape(3)
    left_camera_T_board = (
        np.asarray(left_camera_T_right_camera, dtype=np.float64)
        @ right_camera_T_board)
    left_rvec = cv2.Rodrigues(left_camera_T_board[:3, :3])[0].reshape(3)
    return left_rvec, left_camera_T_board[:3, 3].copy()


def _camera_registration_ready(
        observations, rig_ids, camera_markers, field_registration,
        minimum_frames):
    counts = {
        rig_id: {
            index: len(values["r"])
            for index, values in observations[rig_id].items()
        }
        for rig_id in rig_ids
    }
    for rig_id in rig_ids:
        base_count = max(
            (counts[rig_id].get(index, 0) for index in camera_markers),
            default=0)
        if base_count < minimum_frames:
            return False, counts
    if field_registration is not None:
        primary = rig_ids[0]
        if counts[primary].get(field_registration["board_index"], 0) < (
                minimum_frames):
            return False, counts
    return True, counts


def _save_dual_registration(
        args, session_dir, session_id, rig_ids, cameras, infos,
        observations, last_frames, boards, field_registration, em_recorder):
    from . import register as camera_register

    primary, secondary = rig_ids
    primary_frame = last_frames[primary]
    secondary_frame = last_frames[secondary]
    registration_path = os.path.join(session_dir, "registration.json")
    if em_recorder is None and not args.image_only:
        raise RuntimeError(
            "camera registration overlay requires live EM tracking")
    em_tool_poses = (
        None if args.image_only else
        em_recorder.tip_coil_poses_at(primary_frame.timestamp_ns))

    def save_rig(
            rig_id, frame, registration_file, image_only,
            em_poses=None):
        camera = cameras[rig_id]
        info = infos[rig_id]
        return camera_register.save_session_camera_registration(
            registration_path=registration_file,
            output_dir=session_dir,
            config_path=args.registration_config,
            collected=observations[rig_id],
            left_bgr=frame.left_bgr,
            right_bgr=frame.right_bgr,
            boards=boards,
            K=camera.K,
            dist=camera.dist,
            K_right=camera.K_right,
            dist_right=camera.dist_right,
            right_camera_T_left_camera=(
                camera.right_camera_T_left_camera),
            resolution=info["resolution"],
            zed_serial=info["serial"],
            baseline_m=info["baseline_m"],
            image_timestamp_ns=frame.timestamp_ns,
            min_frames=args.camera_registration_frames,
            em_tool_poses=em_poses,
            image_only=image_only,
            overlay_prefix=rig_id,
        )

    primary_camera = save_rig(
        primary, primary_frame, registration_path,
        args.image_only, em_tool_poses)
    temporary_registration = os.path.join(
        session_dir, f".{secondary}_registration.json")
    try:
        secondary_camera = save_rig(
            secondary, secondary_frame, temporary_registration, True)
    finally:
        try:
            os.remove(temporary_registration)
        except FileNotFoundError:
            pass

    with open(registration_path, encoding="utf-8") as stream:
        document = json.load(stream)
    for rig_id, camera in (
            (primary, primary_camera), (secondary, secondary_camera)):
        camera["rig_id"] = rig_id
        camera["svo_file"] = f"{rig_id}_{session_id}.svo2"
        camera["frame_index_file"] = f"{rig_id}_frame_index.csv"
    document["schema_version"] = 3
    document["camera_rigs"] = {
        primary: primary_camera,
        secondary: secondary_camera,
    }
    # Keep a primary-camera alias for registration inspection. Existing
    # offline reconstruction remains intentionally single-rig.
    document["camera"] = primary_camera
    document["camera_sync"] = {
        "method": "host_image_timestamp_nearest_neighbor",
        "reference_rig": primary,
        "secondary_rig": secondary,
        "pair_file": "camera_frame_pairs.csv",
        "hardware_triggered": False,
    }
    temporary = registration_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
    os.replace(temporary, registration_path)
    return document


def main_dual(args, rig_specs):
    """Capture two ZEDs and register both from one fixed ChArUco board."""
    import cv2
    from . import register as camera_register
    from .zed_capture import (
        ZedCamera, assess_multi_camera_color, assess_stereo_color)

    rig_ids = tuple(spec.rig_id for spec in rig_specs)
    field_registration = (
        None if args.image_only else
        camera_register.load_field_generator_config(args.registration_config))
    em_recorder = _open_em_recorder(args, field_registration)
    session_id = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.outdir, session_id)
    image_dirs = {}
    for rig_id in rig_ids:
        image_dirs[rig_id] = {}
        for eye in ("left", "right"):
            path = os.path.join(session_dir, "images", rig_id, eye)
            os.makedirs(path, exist_ok=True)
            image_dirs[rig_id][eye] = path
    camera_markers, _ = camera_register.load_config(args.registration_config)
    additional = []
    if field_registration is not None:
        if field_registration["board_index"] in camera_markers:
            raise ValueError("field-generator board must differ from base board")
        additional.append((field_registration["board_index"],
                           field_registration["marker_id_offset"]))
    _, boards = boards_mod.build_boards(additional=additional)
    indices = set(camera_markers)
    if field_registration is not None:
        indices.add(field_registration["board_index"])
    observations = {
        rig: _observation_store(indices, args.camera_registration_frames)
        for rig in rig_ids}
    metadata_path = os.path.join(session_dir, "session_metadata.json")
    metadata = {
        "schema_version": 3, "session_id": session_id,
        "mode": "dual_camera_image_only" if args.image_only else "dual_camera_em",
        "modalities": {"stereo_camera": True, "camera_count": 2,
                       "em_tracking": bool(args.aurora_port)},
        "requested_resolution": args.resolution, "requested_fps": args.fps,
        "preview_fps": args.preview_fps,
        "camera_config": os.path.abspath(args.camera_config),
        "svo_compression": args.svo_compression, "camera_rigs": {},
        "camera_sync": {
            "method": "host_image_timestamp_nearest_neighbor",
            "reference_rig": rig_ids[0], "secondary_rig": rig_ids[1],
            "pair_file": "camera_frame_pairs.csv", "hardware_triggered": False}}
    cameras, capture, pose_file = {}, None, None
    show = not args.no_display
    completed = False
    state = {"png": False, "png_frames": 0,
             "pending": bool(args.autorecord),
             "deadline": time.monotonic() + 20.0,
             "reg_collecting": False, "reg_done": False, "notice": None,
             "bad_color_checks": {rig: 0 for rig in rig_ids},
             "bad_cross_camera_color_checks": 0}
    try:
        for spec in rig_specs:
            camera = ZedCamera(**_camera_kwargs(args, spec.serial_number))
            try:
                if args.white_balance_auto_freeze_s > 0:
                    print(
                        f"[{spec.rig_id}][WB] warming native auto white "
                        f"balance for {args.white_balance_auto_freeze_s:.1f}s; "
                        "preview starts after both cameras are frozen",
                        flush=True)
                camera.open()
            except Exception as exc:
                camera.close()
                message = (
                    f"could not initialize camera rig {spec.rig_id!r} "
                    f"(ZED serial {spec.serial_number}) at "
                    f"{args.resolution}@{args.fps}: {exc}")
                if "ZED open failed" in str(exc):
                    message += (
                        ". Connect the cameras to separate USB 3 host "
                        "controllers or reduce resolution/frame rate; SVO "
                        "compression does not reduce live USB bandwidth.")
                raise RuntimeError(message) from exc
            cameras[spec.rig_id] = camera
            info = camera.info()
            metadata["camera_rigs"][spec.rig_id] = info
            print(f"[{spec.rig_id}] ZED {info['serial']} {info['resolution']}@{args.fps}")
            freeze = info.get("white_balance_freeze")
            if freeze is not None:
                print(
                    f"[{spec.rig_id}][WB] auto-freeze complete: "
                    f"{freeze['frames_grabbed']} frames, "
                    f"{freeze['actual_warmup_s']:.2f}s, "
                    f"temperature read-back "
                    f"{freeze['temperature_before_freeze']} -> "
                    f"{freeze['temperature_after_freeze']}, "
                    f"auto={freeze['whitebalance_auto_after_freeze']}",
                    flush=True)
            np.savez(os.path.join(session_dir, f"{spec.rig_id}_intrinsics.npz"),
                     K_left=camera.K, dist_left=camera.dist,
                     K_right=camera.K_right, dist_right=camera.dist_right,
                     left_camera_T_right_camera=camera.left_camera_T_right_camera,
                     right_camera_T_left_camera=camera.right_camera_T_left_camera)
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
        stride = (max(1, int(round(args.fps / args.preview_fps)))
                  if args.preview_fps > 0 else 1)
        capture = MultiZedCapture(cameras, preview_stride=stride).start()
        state["deadline"] = time.monotonic() + 20.0
        infos = {rig: cameras[rig].info() for rig in rig_ids}
        pose_file = open(os.path.join(session_dir, "board_poses.csv"),
                         "w", newline="", encoding="utf-8")
        pose_writer = csv.writer(pose_file)
        pose_writer.writerow([
            "camera_rig", "detection_eye", "timestamp_ns", "board", "corners",
            "tx_m", "ty_m", "tz_m", "rx_rad", "ry_rad", "rz_rad"])
        last_frames, color_health = {}, {}
        multi_camera_color_health = None
        last_processed = {rig: 0 for rig in rig_ids}
        last_bundle_sequence = 0

        def write_metadata():
            with open(metadata_path, "w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2, sort_keys=True)

        def start_recording():
            nonlocal multi_camera_color_health
            if capture.is_recording:
                return
            bad = {rig: value for rig, value in color_health.items()
                   if not value["healthy"]}
            multi_camera_color_health = assess_multi_camera_color(color_health)
            if (len(color_health) != 2 or bad
                    or not multi_camera_color_health["healthy"]):
                raise RuntimeError(
                    "dual-camera color preflight failed: "
                    f"per_rig={bad}, cross_rig={multi_camera_color_health}")
            metadata["recording_color_preflight"] = {
                "per_rig": copy.deepcopy(color_health),
                "cross_rig": copy.deepcopy(multi_camera_color_health)}
            capture.start_recording(session_dir, session_id)
            try:
                if em_recorder is not None:
                    em_recorder.start(session_dir)
            except Exception:
                capture.stop_recording(session_dir)
                raise
            _clear_observations(observations)
            state.update(reg_collecting=bool(args.image_only or field_registration),
                         reg_done=False, notice=None)
            print("[SVO] recording both cameras")
            if state["reg_collecting"]:
                print("[CAM-REG] collecting the same fixed board independently "
                      f"in both cameras ({args.camera_registration_frames} frames)")

        def stop_recording():
            if not capture.is_recording:
                return
            reports, timing = capture.stop_recording(session_dir)
            if em_recorder is not None and em_recorder.is_recording:
                em_recorder.stop()
            metadata["recording_reports"] = reports
            metadata["camera_sync"].update(timing)
            write_metadata()
            for rig, report in reports.items():
                print(f"[{rig}] stopped: indexed={report['indexed_frames']} "
                      f"grabs={report['recording_grabs']} "
                      f"repeats={report['repeated_timestamps']} "
                      f"playable={report['playable_frames']}")
            print("[SYNC] paired={paired_frames} p95_abs={p95_abs_delta_ms}ms "
                  "max_abs={max_abs_delta_ms}ms".format(**timing))

        def finish_registration_if_ready():
            if not state["reg_collecting"] or state["reg_done"]:
                return
            ready, counts = _camera_registration_ready(
                observations, rig_ids, camera_markers, field_registration,
                args.camera_registration_frames)
            if not ready:
                notice = str(counts)
                if notice != state["notice"]:
                    print(f"[CAM-REG] waiting for "
                          f"{args.camera_registration_frames} frames: {counts}")
                    state["notice"] = notice
                return
            if any(rig not in last_frames for rig in rig_ids):
                return
            try:
                document = _save_dual_registration(
                    args, session_dir, session_id, rig_ids, cameras, infos,
                    observations, last_frames, boards, field_registration,
                    em_recorder)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                notice = str(exc)
                if notice != state["notice"]:
                    print(f"[CAM-REG] not completed: {notice}")
                    state["notice"] = notice
                return
            state.update(reg_collecting=False, reg_done=True, notice=None)
            metadata["registration_file"] = "registration.json"
            metadata["registration_camera_rigs"] = list(document["camera_rigs"])
            write_metadata()
            print("[CAM-REG] solved both cameras from one fixed board; "
                  "saved registration.json and four overlays")

        window = "Dual ZED ChArUco tracker"
        if show:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1600, 900)
        print(f"Session dir: {session_dir}")
        print("\nControls: v=both SVOs r=four-view PNGs s=snapshot "
              "q/ESC=quit; rows=cameras, columns=LEFT|RIGHT\n")

        while True:
            bundle = capture.wait_for_bundle(
                after_sequence=last_bundle_sequence, timeout=1.0)
            if bundle is None:
                continue
            last_bundle_sequence = bundle[rig_ids[0]].sequence
            overlays = {}
            for rig in rig_ids:
                frame = bundle[rig]
                last_frames[rig] = frame
                color_health[rig] = assess_stereo_color(
                    frame.left_bgr, frame.right_bgr)
                left_overlay = frame.left_bgr.copy()
                right_overlay = frame.right_bgr.copy()
                if (state["reg_collecting"] and capture.is_recording
                        and frame.sequence > last_processed[rig]):
                    last_processed[rig] = frame.sequence
                    frame_poses = {}
                    eye_inputs = (
                        ("left", frame.left_bgr, left_overlay,
                         cameras[rig].K, cameras[rig].dist),
                        ("right", frame.right_bgr, right_overlay,
                         cameras[rig].K_right, cameras[rig].dist_right))
                    for eye, image, overlay, K_eye, dist_eye in eye_inputs:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        results, _ = registration.detect_boards(
                            gray, boards, K_eye, dist_eye)
                        for result in results:
                            registration.draw_pose(
                                overlay, result, K_eye, dist_eye, AXIS_LEN_M)
                            if (not result.has_pose
                                    or result.index not in observations[rig]
                                    or result.n_corners <
                                    args.camera_registration_min_corners):
                                continue
                            rvec = result.rvec.flatten()
                            tvec = result.tvec.flatten()
                            if eye == "right":
                                rvec, tvec = _right_pose_to_left(
                                    rvec, tvec,
                                    cameras[rig].left_camera_T_right_camera)
                            candidate = (rvec, tvec, result.n_corners, eye)
                            previous = frame_poses.get(result.index)
                            if previous is None or candidate[2] > previous[2]:
                                frame_poses[result.index] = candidate
                    valid = set(frame_poses)
                    if field_registration is not None and rig == rig_ids[0]:
                        field_index = field_registration["board_index"]
                        base = set(camera_markers) & valid
                        valid = (base | {field_index}
                                 if field_index in valid and base else set())
                    else:
                        valid &= set(camera_markers)
                    for index in sorted(valid):
                        rvec, tvec, corners, eye = frame_poses[index]
                        data = observations[rig][index]
                        data["r"].append(rvec)
                        data["t"].append(tvec)
                        data["c"].append(corners)
                        pose_writer.writerow([rig, eye, frame.timestamp_ns,
                                              index, corners, *tvec, *rvec])
                health = color_health[rig]
                color = (0, 220, 0) if health["healthy"] else (0, 0, 255)
                cv2.putText(left_overlay,
                            f"{rig} LEFT serial={infos[rig]['serial']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
                cv2.putText(right_overlay, f"{rig} RIGHT", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
                overlays[rig] = np.hstack((left_overlay, right_overlay))

            multi_camera_color_health = assess_multi_camera_color(color_health)
            if state["pending"]:
                if (len(color_health) == 2 and all(
                        value["healthy"] for value in color_health.values())
                        and multi_camera_color_health["healthy"]):
                    start_recording()
                    state["pending"] = False
                elif time.monotonic() >= state["deadline"]:
                    raise RuntimeError(
                        "dual-camera autorecord blocked for 20s by color "
                        f"preflight: per_rig={color_health}, "
                        f"cross_rig={multi_camera_color_health}")
            if capture.is_recording:
                for rig, health in color_health.items():
                    if health["runtime_healthy"]:
                        state["bad_color_checks"][rig] = 0
                    else:
                        state["bad_color_checks"][rig] += 1
                    if state["bad_color_checks"][rig] >= 3:
                        metadata["stereo_color_fault"] = {
                            "camera_rig": rig,
                            "timestamp_ns": bundle[rig].timestamp_ns,
                            "health": health}
                        stop_recording()
                        raise RuntimeError(
                            f"both SVOs stopped after 3 consecutive {rig} "
                            f"catastrophic ISP color failures: {health}")
            finish_registration_if_ready()
            if state["png"]:
                for rig, frame in bundle.items():
                    cv2.imwrite(os.path.join(image_dirs[rig]["left"],
                                            f"{frame.timestamp_ns}.png"),
                                frame.left_bgr)
                    cv2.imwrite(os.path.join(image_dirs[rig]["right"],
                                            f"{frame.timestamp_ns}.png"),
                                frame.right_bgr)
                state["png_frames"] += 1
            key = 255
            if show:
                cv2.imshow(window, np.vstack([overlays[rig] for rig in rig_ids]))
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                completed = True
                break
            if key == ord("r"):
                state["png"] = not state["png"]
                print(f"[PNG] {'ON' if state['png'] else 'OFF'} "
                      f"({state['png_frames']} four-view frames so far)")
            elif key == ord("v"):
                stop_recording() if capture.is_recording else start_recording()
            elif key == ord("s"):
                for rig, frame in bundle.items():
                    cv2.imwrite(os.path.join(image_dirs[rig]["left"],
                                            f"snap_{frame.timestamp_ns}.png"),
                                frame.left_bgr)
                    cv2.imwrite(os.path.join(image_dirs[rig]["right"],
                                            f"snap_{frame.timestamp_ns}.png"),
                                frame.right_bgr)
                print("[snap] saved all four views")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if pose_file is not None:
            pose_file.close()
        if capture is not None:
            try:
                if capture.is_recording:
                    stop_recording()
            finally:
                capture.close()
        else:
            for camera in cameras.values():
                camera.close()
        if em_recorder is not None:
            em_recorder.close()
        if show:
            cv2.destroyAllWindows()
        label = "Done" if completed else "Session aborted"
        print(f"{label}. Data in {session_dir}")
