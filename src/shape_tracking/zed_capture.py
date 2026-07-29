"""Thin wrapper over the ZED2 python API (pyzed).

Requires the ``[capture]`` extra (ZED SDK python API). Kept out of the package's
top-level import so analysis-only consumers don't need the ZED SDK.

Focus note: the ZED2 has FIXED-FOCUS lenses; the SDK exposes no focus or
focal-length control. The only image-sharpness knob is the digital SHARPNESS
video setting, which open_camera() maxes out. The factory fx/fy are read for
pose estimation.
"""

import numpy as np

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
                 exposure=-1, gain=-1):
        self.resolution = resolution
        self.fps = fps
        self.sharpness = sharpness      # 0..8 (digital unsharp; 4 = SDK default)
        self.exposure = exposure        # 0..100 (% of max time); -1 = auto (AEC)
        self.gain = gain                # 0..100; -1 = auto (AGC)
        self.zed = sl.Camera()
        self._left = sl.Mat()
        self._right = sl.Mat()
        self._runtime = sl.RuntimeParameters()
        self.K = None
        self.dist = None
        self.left_cam = None
        self._recording = False

    # -- lifecycle ---------------------------------------------------------- #
    def open(self):
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.resolution)
        init.camera_fps = self.fps
        init.depth_mode = sl.DEPTH_MODE.NONE       # only the RGB stream is needed
        init.coordinate_units = sl.UNIT.METER
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")
        # Fixed-focus lens: no focus/focal control exists. Sharpness is a digital
        # unsharp-mask; exposure/gain fight motion blur (shorten exposure + add
        # light or gain). Setting exposure/gain manually disables AEC/AGC.
        try:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, int(self.sharpness))
            if self.exposure >= 0:
                self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(self.exposure))
            if self.gain >= 0:
                self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(self.gain))
        except Exception as e:  # non-fatal
            print(f"[warn] could not set camera settings: {e}")
        self._read_intrinsics()
        return self

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
        self.left_cam = lc
        self.K = np.array([[lc.fx, 0.0, lc.cx],
                           [0.0, lc.fy, lc.cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        # Rectified ZED images are undistorted; zeros unless you feed raw frames.
        self.dist = np.zeros((5,), dtype=np.float64)
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
            "baseline_m": self.baseline_m,
        }

    # -- capture ------------------------------------------------------------ #
    def grab(self):
        """Grab one synchronized pair. Returns (timestamp_ns, left_bgr, right_bgr)
        or None if the grab failed this cycle."""
        import cv2
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        ts = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        self.zed.retrieve_image(self._left, sl.VIEW.LEFT)
        self.zed.retrieve_image(self._right, sl.VIEW.RIGHT)
        # BGRA (H,W,4) -> BGR; cvtColor copies out of the SDK-owned buffer.
        left_bgr = cv2.cvtColor(self._left.get_data(), cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(self._right.get_data(), cv2.COLOR_BGRA2BGR)
        return ts, left_bgr, right_bgr

    # -- SVO recording ------------------------------------------------------ #
    def start_recording(self, svo_path):
        rp = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H264)
        err = self.zed.enable_recording(rp)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[warn] enable_recording failed: {err}")
            return False
        self._recording = True
        return True

    def stop_recording(self):
        if self._recording:
            self.zed.disable_recording()
            self._recording = False

    @property
    def is_recording(self):
        return self._recording
