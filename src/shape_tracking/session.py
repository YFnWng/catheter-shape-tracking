"""Recorded-session loading and camera/EM time synchronization.

This module deliberately keeps the ZED SDK import lazy.  Registration, EM
fusion, and unit tests therefore work on machines that do not have ``pyzed``.
All public transforms follow the names saved in ``registration.json``:

``left_camera_T_robot_base`` maps metres in the robot-base frame to metres in
the rectified left-camera frame, while ``robot_base_T_aurora`` maps millimetres
in the Aurora frame to millimetres in the robot-base frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


COIL_003 = "coil_part_003"
COIL_07222026 = "coil_part_07222026"


def _coil_key_from_part_number(part_number: str) -> str | None:
    normalized = "".join(
        character for character in str(part_number).upper()
        if character.isalnum())
    if normalized == "003":
        return COIL_003
    if normalized.startswith("07222026"):
        return COIL_07222026
    return None


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to one point or an ``(N,3)`` array."""
    points = np.asarray(points, dtype=np.float64)
    one = points.ndim == 1
    points = np.atleast_2d(points)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    result = (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]
    return result[0] if one else result


def project_points(
        camera_matrix: np.ndarray,
        camera_T_base: np.ndarray,
        points_base_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project robot-base points (mm) into a rectified camera image.

    Returns ``(pixels, in_front)``. Pixels are ``nan`` for points at or behind
    the focal plane.
    """
    points_m = np.asarray(points_base_mm, dtype=np.float64) * 1e-3
    camera_points = transform_points(camera_T_base, points_m)
    camera_points = np.atleast_2d(camera_points)
    in_front = camera_points[:, 2] > 1e-6
    pixels = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    if np.any(in_front):
        normalized = camera_points[in_front] / camera_points[in_front, 2, None]
        pixels[in_front] = (
            np.asarray(camera_matrix, dtype=np.float64) @ normalized.T
        ).T[:, :2]
    return pixels, in_front


@dataclass(frozen=True)
class SessionRegistration:
    """Transforms and camera parameters extracted from a session JSON."""

    path: Path
    K: np.ndarray
    distortion: np.ndarray
    baseline_m: float
    left_camera_T_base: np.ndarray
    right_camera_T_base: np.ndarray
    base_T_aurora: np.ndarray | None
    aurora_T_base: np.ndarray | None
    roi_left_xywh: tuple[int, int, int, int]
    roi_right_xywh: tuple[int, int, int, int]
    workspace_base_m: dict
    zed_serial: str
    resolution: str


def load_session_registration(
        path_or_session: os.PathLike | str,
        require_em: bool = True) -> SessionRegistration:
    """Load and validate the unified ``registration.json``."""
    path = Path(path_or_session)
    if path.is_dir():
        path = path / "registration.json"
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    em = document.get("em") or {}
    camera = document.get("camera") or {}
    em_solved = em.get("transform_status") == "solved"
    if require_em and not em_solved:
        raise ValueError(f"EM registration is not solved in {path}")
    if camera.get("status") != "solved":
        raise ValueError(f"camera registration is not solved in {path}")
    intrinsics = camera["intrinsics"]
    K = np.asarray(intrinsics["K"], dtype=np.float64)
    distortion = np.asarray(intrinsics.get("distortion", np.zeros(5)),
                            dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera intrinsic matrix must be 3x3 in {path}")
    return SessionRegistration(
        path=path,
        K=K,
        distortion=distortion,
        baseline_m=float(intrinsics["baseline_m"]),
        left_camera_T_base=np.asarray(
            camera["left_camera_T_robot_base"], dtype=np.float64),
        right_camera_T_base=np.asarray(
            camera["right_camera_T_robot_base"], dtype=np.float64),
        base_T_aurora=(
            np.asarray(em["robot_base_T_aurora"], dtype=np.float64)
            if em_solved else None),
        aurora_T_base=(
            np.asarray(em["aurora_T_robot_base"], dtype=np.float64)
            if em_solved else None),
        roi_left_xywh=tuple(int(v) for v in camera["roi_left_xywh"]),
        roi_right_xywh=tuple(int(v) for v in camera["roi_right_xywh"]),
        workspace_base_m=dict(camera["workspace_base_m"]),
        zed_serial=str(intrinsics.get("zed_serial", "")),
        resolution=str(intrinsics.get("resolution", "")),
    )


@dataclass(frozen=True)
class ToolPose:
    """One interpolated EM tool pose in the robot-base frame."""

    timestamp_ns: int
    position_base_mm: np.ndarray
    rotation_base: np.ndarray
    quaternion_base_wxyz: np.ndarray
    bracket_span_ms: float
    nearest_offset_ms: float


@dataclass(frozen=True)
class TipPose:
    """Fused pose of the two tip coils at a camera timestamp."""

    timestamp_ns: int
    valid: bool
    position_base_mm: np.ndarray
    z_direction_base: np.ndarray
    quaternion_base_wxyz: np.ndarray
    coil_positions_base_mm: np.ndarray
    coil_separation_mm: float
    max_sync_offset_ms: float
    max_bracket_span_ms: float
    status: str

    @staticmethod
    def invalid(timestamp_ns: int, status: str) -> "TipPose":
        nan3 = np.full(3, np.nan, dtype=np.float64)
        return TipPose(
            timestamp_ns=int(timestamp_ns),
            valid=False,
            position_base_mm=nan3.copy(),
            z_direction_base=nan3.copy(),
            quaternion_base_wxyz=np.full(4, np.nan, dtype=np.float64),
            coil_positions_base_mm=np.full((2, 3), np.nan, dtype=np.float64),
            coil_separation_mm=float("nan"),
            max_sync_offset_ms=float("nan"),
            max_bracket_span_ms=float("nan"),
            status=status,
        )


@dataclass(frozen=True)
class _ToolSeries:
    timestamps_ns: np.ndarray
    positions_aurora_mm: np.ndarray
    quaternions_aurora_wxyz: np.ndarray


def _normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("cannot normalize a near-zero vector")
    return np.asarray(vector, dtype=np.float64) / norm


def _wxyz_to_rotation(quaternion: np.ndarray) -> Rotation:
    q = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    q = rotation.as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _tip_rotation(
        rotation_003: np.ndarray,
        rotation_07222026: np.ndarray,
        position_003: np.ndarray,
        position_07222026: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct a right-handed tip frame.

    The z axis is the normalized mean of the two coil z axes. The physical roll
    convention defines +x from part 07222026_01 toward part 003. The measured
    coil baseline is projected perpendicular to z before normalization so the
    resulting frame remains orthonormal.
    """
    z0 = _normalize(rotation_003[:, 2])
    z1 = _normalize(rotation_07222026[:, 2])
    if np.dot(z0, z1) < 0:
        z1 = -z1
    z = _normalize(z0 + z1)
    x_candidate = position_003 - position_07222026
    x_candidate = x_candidate - z * np.dot(x_candidate, z)
    if np.linalg.norm(x_candidate) < 1e-6:
        raise ValueError("coil baseline is parallel to the averaged tip z axis")
    x = _normalize(x_candidate)
    y = _normalize(np.cross(z, x))
    x = _normalize(np.cross(y, z))
    return np.column_stack([x, y, z]), z


class EmSynchronizer:
    """Interpolate valid EM poses and fuse the two configured tip coils."""

    def __init__(
            self,
            tool_series: dict[str, _ToolSeries],
            base_T_aurora: np.ndarray,
            max_interp_gap_ms: float = 75.0,
            max_nearest_ms: float = 25.0):
        missing = {COIL_003, COIL_07222026} - set(tool_series)
        if missing:
            raise ValueError(
                f"EM CSV lacks required coil part numbers: {sorted(missing)}")
        self.tool_series = tool_series
        self.base_T_aurora = np.asarray(base_T_aurora, dtype=np.float64)
        self.max_interp_gap_ns = int(max_interp_gap_ms * 1e6)
        self.max_nearest_ns = int(max_nearest_ms * 1e6)

    @classmethod
    def from_csv(
            cls,
            csv_path: os.PathLike | str,
            base_T_aurora: np.ndarray,
            max_interp_gap_ms: float = 75.0,
            max_nearest_ms: float = 25.0) -> "EmSynchronizer":
        values: dict[str, list[tuple]] = {}
        with Path(csv_path).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if not row["tool_role"].startswith("tip_coil_"):
                    continue
                coil_key = _coil_key_from_part_number(
                    row.get("part_number", ""))
                if coil_key is None:
                    continue
                if row.get("valid", "0") != "1" or row["sensor_status"] != "01":
                    continue
                try:
                    item = (
                        int(row["timestamp_ns"]),
                        float(row["tx_mm"]), float(row["ty_mm"]),
                        float(row["tz_mm"]),
                        float(row["qw"]), float(row["qx"]),
                        float(row["qy"]), float(row["qz"]),
                    )
                except (TypeError, ValueError):
                    continue
                if np.all(np.isfinite(item[1:])):
                    values.setdefault(coil_key, []).append(item)
        series = {}
        for role, rows in values.items():
            rows.sort(key=lambda item: item[0])
            array = np.asarray(rows, dtype=np.float64)
            series[role] = _ToolSeries(
                timestamps_ns=np.asarray([row[0] for row in rows],
                                         dtype=np.int64),
                positions_aurora_mm=array[:, 1:4],
                quaternions_aurora_wxyz=array[:, 4:8],
            )
        return cls(series, base_T_aurora, max_interp_gap_ms, max_nearest_ms)

    def _interpolate(self, role: str, timestamp_ns: int) -> ToolPose | None:
        series = self.tool_series[role]
        timestamps = series.timestamps_ns
        if len(timestamps) == 0:
            return None
        index = int(np.searchsorted(timestamps, timestamp_ns))

        if index == 0 or index == len(timestamps):
            nearest = 0 if index == 0 else len(timestamps) - 1
            offset = abs(int(timestamps[nearest]) - timestamp_ns)
            if offset > self.max_nearest_ns:
                return None
            position = series.positions_aurora_mm[nearest]
            rotation_aurora = _wxyz_to_rotation(
                series.quaternions_aurora_wxyz[nearest]).as_matrix()
            span_ns = 0
        else:
            lo, hi = index - 1, index
            t0, t1 = int(timestamps[lo]), int(timestamps[hi])
            span_ns = t1 - t0
            if span_ns <= 0 or span_ns > self.max_interp_gap_ns:
                nearest = lo if timestamp_ns - t0 <= t1 - timestamp_ns else hi
                offset = abs(int(timestamps[nearest]) - timestamp_ns)
                if offset > self.max_nearest_ns:
                    return None
                position = series.positions_aurora_mm[nearest]
                rotation_aurora = _wxyz_to_rotation(
                    series.quaternions_aurora_wxyz[nearest]).as_matrix()
                span_ns = 0
            else:
                alpha = (timestamp_ns - t0) / span_ns
                position = (
                    (1.0 - alpha) * series.positions_aurora_mm[lo]
                    + alpha * series.positions_aurora_mm[hi])
                rotations = Rotation.from_quat([
                    np.roll(series.quaternions_aurora_wxyz[lo], -1),
                    np.roll(series.quaternions_aurora_wxyz[hi], -1),
                ])
                rotation_aurora = Slerp(
                    [0.0, 1.0], rotations)([float(alpha)])[0].as_matrix()

        position_base = transform_points(self.base_T_aurora, position)
        rotation_base = self.base_T_aurora[:3, :3] @ rotation_aurora
        nearest_offset = int(np.min(np.abs(timestamps[max(0, index - 1):
                                                       min(len(timestamps), index + 1)]
                                            - timestamp_ns)))
        return ToolPose(
            timestamp_ns=int(timestamp_ns),
            position_base_mm=position_base,
            rotation_base=rotation_base,
            quaternion_base_wxyz=_rotation_to_wxyz(
                Rotation.from_matrix(rotation_base)),
            bracket_span_ms=span_ns / 1e6,
            nearest_offset_ms=nearest_offset / 1e6,
        )

    def tip_pose(self, timestamp_ns: int) -> TipPose:
        """Return the dual-coil tip pose at ``timestamp_ns``."""
        coil_003 = self._interpolate(COIL_003, int(timestamp_ns))
        coil_072 = self._interpolate(COIL_07222026, int(timestamp_ns))
        if coil_003 is None or coil_072 is None:
            missing = []
            if coil_003 is None:
                missing.append("003")
            if coil_072 is None:
                missing.append("07222026_01")
            return TipPose.invalid(
                timestamp_ns, "missing_or_long_gap:" + ",".join(missing))
        positions = np.vstack([
            coil_003.position_base_mm, coil_072.position_base_mm])
        try:
            rotation, z_direction = _tip_rotation(
                coil_003.rotation_base, coil_072.rotation_base,
                positions[0], positions[1])
        except ValueError as exc:
            return TipPose.invalid(timestamp_ns, f"tip_frame:{exc}")
        return TipPose(
            timestamp_ns=int(timestamp_ns),
            valid=True,
            position_base_mm=positions.mean(axis=0),
            z_direction_base=z_direction,
            quaternion_base_wxyz=_rotation_to_wxyz(
                Rotation.from_matrix(rotation)),
            coil_positions_base_mm=positions,
            coil_separation_mm=float(np.linalg.norm(positions[1] - positions[0])),
            max_sync_offset_ms=max(
                coil_003.nearest_offset_ms, coil_072.nearest_offset_ms),
            max_bracket_span_ms=max(
                coil_003.bracket_span_ms, coil_072.bracket_span_ms),
            status="valid",
        )


@dataclass(frozen=True)
class FrameRecord:
    svo_frame: int
    timestamp_ns: int


def normalize_frame_records(records: list[FrameRecord]) -> list[FrameRecord]:
    """Drop repeated timestamps and map retained images to contiguous SVO slots.

    A successful live ZED ``grab`` can occasionally expose the preceding image
    timestamp again. SVO stores only the new image, so a sidecar indexed by
    grab count drifts by one at every repeated timestamp. Playback positions
    instead correspond to the strictly increasing timestamp sequence.
    """
    normalized = []
    last_timestamp_ns = None
    for record in records:
        timestamp_ns = int(record.timestamp_ns)
        if last_timestamp_ns is not None:
            if timestamp_ns == last_timestamp_ns:
                continue
            if timestamp_ns < last_timestamp_ns:
                raise ValueError(
                    "frame index timestamps are not monotonic: "
                    f"{timestamp_ns} follows {last_timestamp_ns}")
        normalized.append(FrameRecord(
            svo_frame=len(normalized), timestamp_ns=timestamp_ns))
        last_timestamp_ns = timestamp_ns
    return normalized


def load_frame_index(path_or_session: os.PathLike | str) -> list[FrameRecord]:
    path = Path(path_or_session)
    if path.is_dir():
        path = path / "frame_index.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append(FrameRecord(
                svo_frame=int(row["svo_frame"]),
                timestamp_ns=int(row["timestamp_ns"])))
    rows = normalize_frame_records(rows)
    if not rows:
        raise ValueError(f"frame index is empty: {path}")
    return rows


def find_svo(session: os.PathLike | str) -> Path:
    paths = sorted(Path(session).glob("*.svo2")) + sorted(Path(session).glob("*.svo"))
    if len(paths) != 1:
        raise ValueError(
            f"expected one SVO/SVO2 in {session}, found {len(paths)}")
    return paths[0]


class SvoReader:
    """Random/sequential rectified stereo reader for a recorded ZED SVO."""

    def __init__(self, svo_path: os.PathLike | str):
        self.path = str(Path(svo_path).resolve())
        self.zed = None
        self._left = None
        self._right = None
        self._runtime = None
        self._next_svo_frame = None

    def open(self) -> "SvoReader":
        try:
            import pyzed.sl as sl
        except ImportError as exc:  # pragma: no cover - hardware SDK
            raise ImportError(
                "SVO playback requires the ZED SDK Python API (pyzed)") from exc
        init = sl.InitParameters()
        init.set_from_svo_file(self.path)
        init.svo_real_time_mode = False
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED failed to open {self.path}: {status}")
        self._left, self._right = sl.Mat(), sl.Mat()
        self._runtime = sl.RuntimeParameters()
        return self

    def __enter__(self) -> "SvoReader":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        if self.zed is None:
            raise RuntimeError("SVO reader is not open")
        return int(self.zed.get_svo_number_of_frames())

    def close(self) -> None:
        if self.zed is not None:
            self.zed.close()
            self.zed = None
            self._next_svo_frame = None

    def read(self, svo_frame: int) -> tuple[int, np.ndarray, np.ndarray]:
        """Read one frame and return ``(image_timestamp_ns, left, right)``."""
        if self.zed is None:
            raise RuntimeError("SVO reader is not open")
        import pyzed.sl as sl
        if self._next_svo_frame != int(svo_frame):
            self.zed.set_svo_position(int(svo_frame))
        status = self.zed.grab(self._runtime)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"SVO grab failed at frame {svo_frame}: {status}")
        self.zed.retrieve_image(self._left, sl.VIEW.LEFT)
        self.zed.retrieve_image(self._right, sl.VIEW.RIGHT)
        timestamp_ns = int(
            self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        self._next_svo_frame = int(svo_frame) + 1
        left = cv2.cvtColor(self._left.get_data(), cv2.COLOR_BGRA2BGR)
        right = cv2.cvtColor(self._right.get_data(), cv2.COLOR_BGRA2BGR)
        return timestamp_ns, left, right

    def iter_records(
            self,
            records: list[FrameRecord]) -> Iterator[
                tuple[FrameRecord, int, np.ndarray, np.ndarray]]:
        for record in records:
            timestamp_ns, left, right = self.read(record.svo_frame)
            yield record, timestamp_ns, left, right
