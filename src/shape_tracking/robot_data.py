"""Load and timestamp-align catheter commands and feedback from a ROS 2 bag."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CONTROL_STREAM_MSG = """
std_msgs/Header header
string cartesian_frame
geometry_msgs/Twist cartesian_delta
geometry_msgs/Pose cartesian_target
float64[] joint_vel
float64[] joint_pos
"""

DEVICE_STREAM_MSG = """
uint8 VEL = 86
uint8 POS = 80
uint8 ENC = 69
std_msgs/Header header
uint8 predicate
float64[] data
"""


@dataclass(frozen=True)
class TimedJointSeries:
    timestamp_ns: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class RobotStreams:
    command_velocity: TimedJointSeries
    measured_position: TimedJointSeries
    raw_encoder: TimedJointSeries


@dataclass(frozen=True)
class AlignedRobotData:
    command_velocity: np.ndarray
    measured_position: np.ndarray
    raw_encoder: np.ndarray
    command_age_ms: np.ndarray
    position_age_ms: np.ndarray
    encoder_age_ms: np.ndarray
    command_valid: np.ndarray
    position_valid: np.ndarray
    encoder_valid: np.ndarray


def _stamp_ns(header, receipt_timestamp_ns: int) -> int:
    """Prefer the message stamp, falling back to the bag receipt timestamp."""
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(receipt_timestamp_ns)
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else int(receipt_timestamp_ns)


def _as_series(rows: list[tuple[int, np.ndarray]]) -> TimedJointSeries:
    if not rows:
        return TimedJointSeries(
            np.empty(0, dtype=np.int64), np.empty((0, 6), dtype=np.float64))
    rows.sort(key=lambda item: item[0])
    timestamps = np.asarray([row[0] for row in rows], dtype=np.int64)
    values = np.asarray([row[1] for row in rows], dtype=np.float64)
    keep = np.r_[True, np.diff(timestamps) > 0]
    return TimedJointSeries(timestamps[keep], values[keep])


def load_robot_streams(session_or_bag: str | Path) -> RobotStreams:
    """Deserialize the three model-relevant streams without a ROS workspace."""
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    except ImportError as exc:  # pragma: no cover - dependency error
        raise ImportError("robot alignment requires `pip install rosbags`") from exc

    path = Path(session_or_bag)
    bag = path / "rosbag" if (path / "rosbag").is_dir() else path
    if not bag.is_dir():
        raise FileNotFoundError(f"ROS bag directory does not exist: {bag}")

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(
        CONTROL_STREAM_MSG, "control_interface/msg/ControlStream"))
    typestore.register(get_types_from_msg(
        DEVICE_STREAM_MSG, "control_interface/msg/DeviceStream"))

    command_rows: list[tuple[int, np.ndarray]] = []
    position_rows: list[tuple[int, np.ndarray]] = []
    encoder_rows: list[tuple[int, np.ndarray]] = []
    with AnyReader([bag], default_typestore=typestore) as reader:
        connections = [
            connection for connection in reader.connections
            if connection.topic in ("/teleop/control", "/device/state")]
        for connection, receipt_ns, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            timestamp_ns = _stamp_ns(message.header, receipt_ns)
            if connection.topic == "/teleop/control":
                values = np.asarray(message.joint_vel, dtype=np.float64)
                if values.shape == (6,):
                    command_rows.append((timestamp_ns, values))
            elif message.predicate in (message.POS, message.ENC):
                values = np.asarray(message.data, dtype=np.float64)
                if values.shape != (6,):
                    continue
                target = position_rows if message.predicate == message.POS else encoder_rows
                target.append((timestamp_ns, values))
    return RobotStreams(
        command_velocity=_as_series(command_rows),
        measured_position=_as_series(position_rows),
        raw_encoder=_as_series(encoder_rows),
    )


def _zoh(
        series: TimedJointSeries,
        query_ns: np.ndarray,
        max_age_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = np.full((len(query_ns), 6), np.nan, dtype=np.float64)
    age_ms = np.full(len(query_ns), np.nan, dtype=np.float64)
    valid = np.zeros(len(query_ns), dtype=np.uint8)
    if len(series.timestamp_ns) == 0:
        return output, age_ms, valid
    indices = np.searchsorted(series.timestamp_ns, query_ns, side="right") - 1
    available = indices >= 0
    safe = np.clip(indices, 0, len(series.timestamp_ns) - 1)
    age_ms[available] = (
        query_ns[available] - series.timestamp_ns[safe[available]]) / 1e6
    accepted = available & (age_ms >= 0.0) & (age_ms <= float(max_age_ms))
    output[accepted] = series.values[safe[accepted]]
    valid[accepted] = 1
    return output, age_ms, valid


def _linear(
        series: TimedJointSeries,
        query_ns: np.ndarray,
        max_bracket_ms: float,
        unwrap_degrees: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = np.full((len(query_ns), 6), np.nan, dtype=np.float64)
    age_ms = np.full(len(query_ns), np.nan, dtype=np.float64)
    valid = np.zeros(len(query_ns), dtype=np.uint8)
    if len(series.timestamp_ns) < 2:
        return output, age_ms, valid
    right = np.searchsorted(series.timestamp_ns, query_ns, side="left")
    bracketed = (right > 0) & (right < len(series.timestamp_ns))
    right_safe = np.clip(right, 1, len(series.timestamp_ns) - 1)
    left = right_safe - 1
    t0, t1 = series.timestamp_ns[left], series.timestamp_ns[right_safe]
    span_ms = (t1 - t0) / 1e6
    nearest_ms = np.minimum(np.abs(query_ns - t0), np.abs(t1 - query_ns)) / 1e6
    age_ms[bracketed] = nearest_ms[bracketed]
    accepted = bracketed & (span_ms <= float(max_bracket_ms))
    if not np.any(accepted):
        return output, age_ms, valid
    values = series.values.copy()
    for index in unwrap_degrees:
        values[:, index] = np.rad2deg(np.unwrap(np.deg2rad(values[:, index])))
    weight = np.clip((query_ns - t0) / np.maximum(t1 - t0, 1), 0.0, 1.0)
    interpolated = values[left] + weight[:, None] * (values[right_safe] - values[left])
    for index in unwrap_degrees:
        interpolated[:, index] = (
            interpolated[:, index] + 180.0) % 360.0 - 180.0
    output[accepted] = interpolated[accepted]
    valid[accepted] = 1
    return output, age_ms, valid


def align_robot_streams(
        streams: RobotStreams,
        query_timestamp_ns: np.ndarray,
        max_command_age_ms: float = 30.0,
        max_feedback_gap_ms: float = 30.0) -> AlignedRobotData:
    """Align commands/feedback to camera timestamps with gap-aware validity."""
    query = np.asarray(query_timestamp_ns, dtype=np.int64)
    command, command_age, command_valid = _zoh(
        streams.command_velocity, query, max_command_age_ms)
    position, position_age, position_valid = _linear(
        streams.measured_position, query, max_feedback_gap_ms,
        unwrap_degrees=(1, 4))
    encoder, encoder_age, encoder_valid = _linear(
        streams.raw_encoder, query, max_feedback_gap_ms)
    return AlignedRobotData(
        command_velocity=command,
        measured_position=position,
        raw_encoder=encoder,
        command_age_ms=command_age,
        position_age_ms=position_age,
        encoder_age_ms=encoder_age,
        command_valid=command_valid,
        position_valid=position_valid,
        encoder_valid=encoder_valid,
    )
