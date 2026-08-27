"""Thin wrapper over the ZED2 python API (pyzed).

Requires the ``[capture]`` extra (ZED SDK python API). Kept out of the package's
top-level import so analysis-only consumers don't need the ZED SDK.

Focus note: the ZED2 has FIXED-FOCUS lenses; the SDK exposes no focus or
focal-length control. The only image-sharpness knob is the digital SHARPNESS
video setting, which open_camera() maxes out. The factory fx/fy are read for
pose estimation.
"""

import time

import numpy as np


_MEAN_CHANNEL_RATIO_LIMIT = 1.12
_NEUTRAL_CHANNEL_RATIO_LIMIT = 1.10
_STEREO_CHROMATICITY_DELTA_LIMIT = 0.010
_MULTI_CAMERA_CHROMATICITY_DELTA_LIMIT = 0.03
_GREEN_BALANCE_MIN = 0.95
_GREEN_BALANCE_MAX = 1.10


def _eye_color_report(name, image):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3 or array.size == 0:
        raise ValueError(f"invalid {name} image for stereo color check")
    pixels = array[::8, ::8, :3].reshape(-1, 3).astype(np.float64)
    mean_bgr = pixels.mean(axis=0)
    mean_ratio = float(mean_bgr.max() / max(1.0, mean_bgr.min()))
    brightness = pixels.mean(axis=1)
    # Use bright mid-tones, not clipped white pixels. A clipped light-box
    # background is [255, 255, 255] regardless of the ISP gains and can hide a
    # very visible cast in the rest of the image.
    bright_floor = np.percentile(brightness, 45.0)
    usable = pixels[
        (brightness >= bright_floor)
        & (pixels.max(axis=1) < 248.0)
        & (pixels.min(axis=1) > 24.0)]
    minimum_usable = max(32, int(0.02 * len(pixels)))
    neutral_pixels = usable if len(usable) >= minimum_usable else pixels
    neutral_bgr = np.median(neutral_pixels, axis=0)
    neutral_sum = float(neutral_bgr.sum())
    chroma = neutral_bgr / neutral_sum if neutral_sum else np.zeros(3)
    ratio = float(neutral_bgr.max() / max(1.0, neutral_bgr.min()))
    dominance = float(
        mean_bgr.max() / max(1.0, (mean_bgr.sum() - mean_bgr.max()) / 2.0))
    green_dominance = float(
        mean_bgr[1] / max(1.0, (mean_bgr[0] + mean_bgr[2]) / 2.0))
    green_clip = float(np.mean(
        (pixels[:, 1] >= 250) & (pixels[:, 0] < 100) & (pixels[:, 2] < 100)))
    gross_healthy = (
        dominance < 3.0 and green_clip < 0.10 and mean_bgr.mean() >= 8.0)
    healthy = (gross_healthy
               and mean_ratio <= _MEAN_CHANNEL_RATIO_LIMIT
               and _GREEN_BALANCE_MIN <= green_dominance
               <= _GREEN_BALANCE_MAX
               and neutral_bgr.mean() >= 40.0
               and ratio <= _NEUTRAL_CHANNEL_RATIO_LIMIT)
    return {
        "eye": name, "mean_bgr": mean_bgr.tolist(),
        "mean_channel_ratio": mean_ratio,
        "mean_channel_ratio_limit": _MEAN_CHANNEL_RATIO_LIMIT,
        "channel_dominance": dominance, "green_dominance": green_dominance,
        "green_balance_min": _GREEN_BALANCE_MIN,
        "green_balance_max": _GREEN_BALANCE_MAX,
        "green_clip_fraction": green_clip, "neutral_bgr": neutral_bgr.tolist(),
        "neutral_chromaticity_bgr": chroma.tolist(),
        "neutral_channel_ratio": ratio,
        "neutral_channel_ratio_limit": _NEUTRAL_CHANNEL_RATIO_LIMIT,
        "gross_healthy": bool(gross_healthy),
        "healthy": bool(healthy)}


def assess_stereo_color(left_bgr, right_bgr):
    """Detect tint/clipping and disagreement between a ZED's two eyes."""
    reports = [
        _eye_color_report("left", left_bgr),
        _eye_color_report("right", right_bgr)]
    left_chroma = np.asarray(reports[0]["neutral_chromaticity_bgr"])
    right_chroma = np.asarray(reports[1]["neutral_chromaticity_bgr"])
    stereo_delta = float(np.max(np.abs(left_chroma - right_chroma)))
    return {
        "healthy": bool(all(report["healthy"] for report in reports)
                        and stereo_delta <= _STEREO_CHROMATICITY_DELTA_LIMIT),
        # Runtime scenes may contain hands, tools, or other large colored
        # objects. Only catastrophic ISP corruption/clipping is meaningful
        # after WB has been frozen and recording has started.
        "runtime_healthy": bool(
            all(report["gross_healthy"] for report in reports)),
        "eyes": {report["eye"]: report for report in reports},
        "stereo_chromaticity_delta": stereo_delta,
        "stereo_chromaticity_delta_limit": _STEREO_CHROMATICITY_DELTA_LIMIT,
    }


def assess_multi_camera_color(reports):
    """Require independently healthy rigs to agree on background color."""
    rig_chromaticity = {}
    for rig, report in reports.items():
        eyes = report.get("eyes", {})
        values = [np.asarray(eyes[eye]["neutral_chromaticity_bgr"])
                  for eye in ("left", "right") if eye in eyes]
        if len(values) == 2:
            rig_chromaticity[rig] = np.mean(values, axis=0)
    max_delta = 0.0
    names = sorted(rig_chromaticity)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            delta = np.abs(rig_chromaticity[first]
                           - rig_chromaticity[second])
            max_delta = max(max_delta, float(np.max(delta)))
    complete = len(rig_chromaticity) == len(reports) and len(reports) >= 2
    healthy = (complete
               and all(item.get("healthy", False) for item in reports.values())
               and max_delta <= _MULTI_CAMERA_CHROMATICITY_DELTA_LIMIT)
    return {
        "healthy": bool(healthy), "complete": bool(complete),
        "rig_chromaticity_bgr": {
            rig: value.tolist() for rig, value in rig_chromaticity.items()},
        "max_rig_chromaticity_delta": max_delta,
        "max_rig_chromaticity_delta_limit":
            _MULTI_CAMERA_CHROMATICITY_DELTA_LIMIT}


try:
    import pyzed.sl as sl
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pyzed not found. Install the ZED python API for THIS interpreter:\n"
        '  python "C:\\Program Files (x86)\\ZED SDK\\get_python_api.py"\n'
        "then pip-install the .whl it downloads (or: pip install shape_tracking[capture])."
    ) from e


class ZedCamera:
    """Context manager that opens the ZED2 and yields time-stamped stereo frames."""

    def __init__(self, resolution="HD1080", fps=30, sharpness=4,
                 exposure=-1, gain=-1, white_balance_temperature=-1,
                 brightness=None, contrast=None, saturation=None, gamma=None,
                 hue=None, svo_compression="H264",
                 white_balance_auto_freeze_s=0.0,
                 white_balance_auto_freeze_retries=2,
                 serial_number=None):
        self.resolution = resolution
        self.fps = fps
        self.sharpness = sharpness      # 0..8 (digital unsharp; 4 = SDK default)
        self.exposure = exposure        # 0..100 (% of max time); -1 = auto (AEC)
        self.gain = gain                # 0..100; -1 = auto (AGC)
        self.white_balance_temperature = white_balance_temperature
        self.white_balance_auto_freeze_s = float(
            white_balance_auto_freeze_s)
        self.white_balance_auto_freeze_retries = int(
            white_balance_auto_freeze_retries)
        self.white_balance_freeze = None
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.gamma = gamma
        self.svo_compression = svo_compression
        self.serial_number = (
            None if serial_number in (None, "", 0, "0")
            else int(serial_number))
        self.zed = sl.Camera()
        self._left = sl.Mat()
        self._right = sl.Mat()
        self._runtime = sl.RuntimeParameters()
        self.K = None
        self.K_right = None
        self.dist = None
        self.dist_right = None
        self.left_cam = None
        self.right_cam = None
        self.left_camera_T_right_camera = None
        self.right_camera_T_left_camera = None
        self._recording = False

    # -- lifecycle ---------------------------------------------------------- #
    def open(self):
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.resolution)
        init.camera_fps = self.fps
        init.depth_mode = sl.DEPTH_MODE.NONE       # only the RGB stream is needed
        init.coordinate_units = sl.UNIT.METER
        if self.serial_number is not None:
            init.set_from_serial_number(self.serial_number)
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")
        try:
            self._apply_camera_settings()
            self._read_intrinsics()
        except Exception:
            # A failed stereo-color preflight must not leave the device open,
            # especially when callers use ``with ZedCamera(...)`` (where
            # __enter__ itself raised and __exit__ will therefore not run).
            self.zed.close()
            raise
        return self

    def _set_camera_setting(self, setting, value):
        """Range-check and apply one integer video setting."""
        if (setting == sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE
                and (value % 100 != 0 or not 2800 <= value <= 6500)):
            raise ValueError(
                "ZED WHITEBALANCE_TEMPERATURE must be an exact 100 K step "
                "from 2800 through 6500")
        range_status, lower, upper = self.zed.get_camera_settings_range(setting)
        if range_status == sl.ERROR_CODE.SUCCESS and not lower <= value <= upper:
            raise ValueError(
                f"ZED {setting.name}={value} outside supported "
                f"range [{lower}, {upper}]")
        status = self.zed.set_camera_settings(setting, int(value))
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED could not set {setting.name}={value}: {status}")

    def _apply_camera_settings(self):
        # Camera settings can be retained across applications. Set each auto
        # state explicitly so repeated sessions do not inherit an old mode.
        manual_exposure = self.exposure >= 0 and self.gain >= 0
        self._set_camera_setting(
            sl.VIDEO_SETTINGS.AEC_AGC, 0 if manual_exposure else 1)
        if manual_exposure:
            self._set_camera_setting(sl.VIDEO_SETTINGS.EXPOSURE, self.exposure)
            self._set_camera_setting(sl.VIDEO_SETTINGS.GAIN, self.gain)

        requested = {
            sl.VIDEO_SETTINGS.SHARPNESS: self.sharpness,
            sl.VIDEO_SETTINGS.BRIGHTNESS: self.brightness,
            sl.VIDEO_SETTINGS.CONTRAST: self.contrast,
            sl.VIDEO_SETTINGS.HUE: self.hue,
            sl.VIDEO_SETTINGS.SATURATION: self.saturation,
            sl.VIDEO_SETTINGS.GAMMA: self.gamma,
        }
        for setting, value in requested.items():
            if value is not None:
                self._set_camera_setting(setting, int(value))

        # Converge auto WB only after all other image controls have their final
        # values. Disabling WHITEBALANCE_AUTO without writing a temperature
        # retains the native ISP's current per-sensor gains. The temperature
        # read-back is coarse and must not be written back as a substitute.
        if self.white_balance_auto_freeze_s > 0:
            self._freeze_auto_white_balance()
        else:
            manual_white_balance = self.white_balance_temperature >= 0
            self._set_camera_setting(
                sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO,
                0 if manual_white_balance else 1)
            if manual_white_balance:
                self._set_camera_setting(
                    sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE,
                    self.white_balance_temperature)

    def _freeze_auto_white_balance(self):
        """Warm auto WB, freeze it, and reject intermittent one-eye tint."""
        attempts = []
        total_frames = 0
        total_started = time.monotonic()
        max_attempts = 1 + self.white_balance_auto_freeze_retries
        for attempt in range(1, max_attempts + 1):
            warm_started = time.monotonic()
            self._set_camera_setting(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
            warm_frames = 0
            while time.monotonic() - warm_started < self.white_balance_auto_freeze_s:
                if self.zed.grab(self._runtime) == sl.ERROR_CODE.SUCCESS:
                    warm_frames += 1
                else:
                    time.sleep(0.005)
            status, before = self.zed.get_camera_settings(
                sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE)
            before = int(before) if status == sl.ERROR_CODE.SUCCESS else None
            self._set_camera_setting(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
            for _ in range(2):
                if self.zed.grab(self._runtime) == sl.ERROR_CODE.SUCCESS:
                    warm_frames += 1
            total_frames += warm_frames
            health = self.current_stereo_color_health()
            settings = self.camera_settings()
            attempts.append({
                "attempt": attempt,
                "warmup_s": time.monotonic() - warm_started,
                "frames_grabbed": warm_frames,
                "temperature_before_freeze": before,
                "temperature_after_freeze": settings.get(
                    "whitebalance_temperature"),
                "freeze_method": "disable_auto_retain_native_isp_gains",
                "health": health,
            })
            if health["healthy"]:
                self.white_balance_freeze = {
                    "requested_warmup_s": self.white_balance_auto_freeze_s,
                    "actual_warmup_s": time.monotonic() - total_started,
                    "frames_grabbed": total_frames,
                    "temperature_before_freeze": before,
                    "temperature_after_freeze": settings.get(
                        "whitebalance_temperature"),
                    "freeze_method": "disable_auto_retain_native_isp_gains",
                    "whitebalance_auto_after_freeze": settings.get(
                        "whitebalance_auto"),
                    "attempts": attempts,
                    "fallback_to_continuous_auto": False,
                    "stereo_color_health": health,
                }
                return
            print(
                f"[WB serial={self.serial_number}] attempt "
                f"{attempt}/{max_attempts} rejected; retrying: "
                f"left_mean={health['eyes']['left']['mean_bgr']}, "
                f"right_mean={health['eyes']['right']['mean_bgr']}, "
                f"stereo_delta={health['stereo_chromaticity_delta']:.4f}",
                flush=True)

        # Continuous auto is unsuitable for color-based segmentation. Restore
        # it only as a safe non-recording state, then block camera startup.
        self._set_camera_setting(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
        raise RuntimeError(
            "ZED stereo color preflight failed after deterministic auto-WB "
            f"freeze retries: {attempts}")

    def current_stereo_color_health(self):
        """Retrieve the current pair and return gross per-eye color health."""
        self.zed.retrieve_image(self._left, sl.VIEW.LEFT)
        self.zed.retrieve_image(self._right, sl.VIEW.RIGHT)
        left = np.asarray(self._left.get_data())[..., :3]
        right = np.asarray(self._right.get_data())[..., :3]
        return assess_stereo_color(left, right)

    def camera_settings(self):
        """Return verified values read back from the opened camera."""
        names = (
            "AEC_AGC", "EXPOSURE", "GAIN", "WHITEBALANCE_AUTO",
            "WHITEBALANCE_TEMPERATURE", "BRIGHTNESS", "CONTRAST",
            "HUE", "SATURATION", "GAMMA", "SHARPNESS")
        values = {}
        for name in names:
            setting = getattr(sl.VIDEO_SETTINGS, name)
            status, value = self.zed.get_camera_settings(setting)
            if status == sl.ERROR_CODE.SUCCESS:
                values[name.lower()] = int(value)
        return values

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._recording:
            self.zed.disable_recording()
            self._recording = False
        self.zed.close()

    # -- intrinsics --------------------------------------------------------- #
    def _read_intrinsics(self):
        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
        lc = calib.left_cam
        rc = calib.right_cam
        self.left_cam = lc
        self.right_cam = rc
        self.K = np.array([[lc.fx, 0.0, lc.cx],
                           [0.0, lc.fy, lc.cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.K_right = np.array([[rc.fx, 0.0, rc.cx],
                                 [0.0, rc.fy, rc.cy],
                                 [0.0, 0.0, 1.0]], dtype=np.float64)
        # Rectified ZED images are undistorted; zeros unless you feed raw frames.
        self.dist = np.zeros((5,), dtype=np.float64)
        self.dist_right = np.zeros((5,), dtype=np.float64)
        self.left_camera_T_right_camera = np.asarray(
            calib.stereo_transform.m, dtype=np.float64).reshape(4, 4).copy()
        self.right_camera_T_left_camera = np.linalg.inv(
            self.left_camera_T_right_camera)
        # Stereo baseline (m). SDK historically returns mm; normalize to meters.
        try:
            b = float(calib.get_camera_baseline())
        except Exception:
            b = 120.0
        self.baseline_m = b / 1000.0 if b > 1.0 else b

    def info(self):
        ci = self.zed.get_camera_information()
        return {
            "serial": ci.serial_number,
            "model": str(ci.camera_model),
            "sdk": self.zed.get_sdk_version(),
            "resolution": self.resolution,
            "fx": self.left_cam.fx, "fy": self.left_cam.fy,
            "cx": self.left_cam.cx, "cy": self.left_cam.cy,
            "right_fx": self.right_cam.fx, "right_fy": self.right_cam.fy,
            "right_cx": self.right_cam.cx, "right_cy": self.right_cam.cy,
            "baseline_m": self.baseline_m,
            "left_camera_T_right_camera":
                self.left_camera_T_right_camera.tolist(),
            "right_camera_T_left_camera":
                self.right_camera_T_left_camera.tolist(),
            "svo_compression": self.svo_compression,
            "camera_settings": self.camera_settings(),
            "white_balance_freeze": self.white_balance_freeze,
        }

    # -- capture ------------------------------------------------------------ #
    def grab(self, retrieve=True):
        """Grab one synchronized pair. Returns (timestamp_ns, left_bgr, right_bgr)
        or None if the grab failed this cycle.

        When ``retrieve`` is false, the camera is still grabbed (and therefore
        still written to an enabled SVO recording), but the expensive full-HD
        CPU image copies are skipped and both image values are ``None``.
        """
        import cv2
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        ts = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        if not retrieve:
            return ts, None, None
        self.zed.retrieve_image(self._left, sl.VIEW.LEFT)
        self.zed.retrieve_image(self._right, sl.VIEW.RIGHT)
        # BGRA (H,W,4) -> BGR; cvtColor copies out of the SDK-owned buffer.
        left_bgr = cv2.cvtColor(self._left.get_data(), cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(self._right.get_data(), cv2.COLOR_BGRA2BGR)
        return ts, left_bgr, right_bgr

    # -- SVO recording ------------------------------------------------------ #
    def start_recording(self, svo_path):
        try:
            compression = getattr(sl.SVO_COMPRESSION_MODE, self.svo_compression)
        except AttributeError as exc:
            raise ValueError(
                f"unsupported SVO compression mode {self.svo_compression!r}") from exc
        rp = sl.RecordingParameters(svo_path, compression)
        err = self.zed.enable_recording(rp)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[warn] enable_recording failed: {err}")
            return False
        self._recording = True
        return True

    def recording_status(self):
        """Return plain recording counters from the installed ZED SDK."""
        status = self.zed.get_recording_status()
        return {
            'is_recording': bool(status.is_recording),
            'status': bool(status.status),
            'number_frames_ingested': int(status.number_frames_ingested),
            'number_frames_encoded': int(status.number_frames_encoded),
            'current_compression_time_ms': float(
                status.current_compression_time),
            'average_compression_time_ms': float(
                status.average_compression_time),
        }

    def stop_recording(self):
        status = None
        if self._recording:
            try:
                status = self.recording_status()
            except Exception:
                pass
            self.zed.disable_recording()
            self._recording = False
        return status

    @staticmethod
    def playable_svo_frame_count(svo_path):
        """Return the number of frame positions that can actually be grabbed.

        ZED SDK 5.4 includes one EOF sentinel in get_svo_number_of_frames().
        Account for it directly; grabbing that sentinel only emits a misleading
        END_OF_SVO_FILE_REACHED error during otherwise-clean shutdown.
        """
        init = sl.InitParameters()
        init.set_from_svo_file(str(svo_path))
        init.svo_real_time_mode = False
        init.depth_mode = sl.DEPTH_MODE.NONE
        playback = sl.Camera()
        status = playback.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED failed to reopen recorded SVO: {status}")
        try:
            reported = int(playback.get_svo_number_of_frames())
            return max(0, reported - 1)
        finally:
            playback.close()

    @property
    def is_recording(self):
        return self._recording
