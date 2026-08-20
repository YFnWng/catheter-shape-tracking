"""Fuse processed catheter shape, EM tip tracking, and robot actuation.

The command stream is the canonical timeline.  Image observations are matched
by nearest timestamp (never interpolated), while the two EM coils are
interpolated and fused by :class:`shape_tracking.session.EmSynchronizer`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

from .robot_data import AlignedRobotData, RobotStreams, align_robot_streams, load_robot_streams
from .sequence import load_collection_markers
from .session import EmSynchronizer, TipPose, load_session_registration


@dataclass(frozen=True)
class FusionConfig:
    max_image_offset_ms: float = 25.0
    max_command_age_ms: float = 30.0
    max_feedback_gap_ms: float = 30.0
    max_em_interp_gap_ms: float = 75.0
    max_em_nearest_ms: float = 25.0


def select_actuation_timestamps(
        streams: RobotStreams,
        markers: dict[str, dict],
        window: str = "run_and_return",
        start_ns: int | None = None,
        end_ns: int | None = None,
        stride: int = 1,
        max_samples: int | None = None) -> np.ndarray:
    """Select command-message timestamps for the requested collection window."""
    if window not in ("trajectory", "run_and_return", "recording"):
        raise ValueError(f"unsupported window: {window}")
    if start_ns is None and window in ("trajectory", "run_and_return"):
        start_ns = (markers.get("run_start") or {}).get("stamp_ns")
    if end_ns is None and window == "trajectory":
        end_ns = (markers.get("return_start") or {}).get("stamp_ns")
    if end_ns is None and window == "run_and_return":
        end_ns = (markers.get("run_end") or {}).get("stamp_ns")

    timestamps = np.asarray(streams.command_velocity.timestamp_ns, dtype=np.int64)
    if len(timestamps) == 0:
        raise ValueError("ROS bag has no valid /teleop/control joint commands")
    keep = np.ones(len(timestamps), dtype=bool)
    if start_ns is not None:
        keep &= timestamps >= int(start_ns)
    if end_ns is not None:
        keep &= timestamps < int(end_ns)
    selected = timestamps[keep][::max(1, int(stride))]
    if max_samples is not None:
        selected = selected[:max(0, int(max_samples))]
    if len(selected) == 0:
        raise ValueError(
            "no actuation samples selected; verify collection markers/time bounds")
    return selected


def nearest_sample_indices(
        source_timestamp_ns: np.ndarray,
        query_timestamp_ns: np.ndarray,
        max_offset_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest source index, signed offset, and gap-valid flag."""
    source = np.asarray(source_timestamp_ns, dtype=np.int64)
    query = np.asarray(query_timestamp_ns, dtype=np.int64)
    if len(source) == 0:
        return (
            np.full(len(query), -1, dtype=np.int64),
            np.full(len(query), np.nan, dtype=np.float64),
            np.zeros(len(query), dtype=np.uint8),
        )
    if np.any(np.diff(source) <= 0):
        raise ValueError("image timestamps must be strictly increasing")
    right = np.searchsorted(source, query, side="left")
    hi = np.clip(right, 0, len(source) - 1)
    lo = np.clip(right - 1, 0, len(source) - 1)
    choose_hi = np.abs(source[hi] - query) < np.abs(source[lo] - query)
    index = np.where(choose_hi, hi, lo).astype(np.int64)
    offset_ms = (source[index] - query) / 1e6
    valid = (np.abs(offset_ms) <= float(max_offset_ms)).astype(np.uint8)
    return index, offset_ms, valid


def _create_numeric_dataset(group, name: str, data: np.ndarray) -> None:
    data = np.asarray(data)
    chunks = True if data.ndim > 0 and data.size > 0 else None
    kwargs = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    if chunks is None:
        kwargs = {}
    group.create_dataset(name, data=data, chunks=chunks, **kwargs)


def _gather_dataset(dataset, indices: np.ndarray) -> np.ndarray:
    """Gather repeated sorted HDF5 rows without h5py's unique-index limit."""
    unique, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


def _write_robot(group, aligned: AlignedRobotData) -> None:
    values = {
        "joint_velocity_command": aligned.command_velocity,
        "joint_position_measured": aligned.measured_position,
        "encoder_raw": aligned.raw_encoder,
        "command_age_ms": aligned.command_age_ms,
        "position_age_ms": aligned.position_age_ms,
        "encoder_age_ms": aligned.encoder_age_ms,
        "command_valid": aligned.command_valid,
        "position_valid": aligned.position_valid,
        "encoder_valid": aligned.encoder_valid,
    }
    for name, value in values.items():
        _create_numeric_dataset(group, name, value)
    _create_numeric_dataset(
        group, "feedback_valid",
        (aligned.position_valid.astype(bool) & aligned.encoder_valid.astype(bool))
        .astype(np.uint8))


def _write_image(
        group,
        image_h5: Path | None,
        query_ns: np.ndarray,
        enabled: bool,
        max_offset_ms: float) -> np.ndarray:
    import h5py

    count = len(query_ns)
    group.attrs["enabled"] = np.uint8(enabled)
    if not enabled:
        _create_numeric_dataset(group, "source_index", np.full(count, -1, np.int64))
        _create_numeric_dataset(group, "valid", np.zeros(count, np.uint8))
        return np.ones(count, dtype=np.uint8)
    if image_h5 is None or not image_h5.is_file():
        raise FileNotFoundError(f"enabled image input does not exist: {image_h5}")

    with h5py.File(image_h5, "r") as source:
        required = ("frames/timestamp_ns", "frames/valid", "full", "distal")
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"image HDF5 lacks required entries: {missing}")
        image_ns = np.asarray(source["frames/timestamp_ns"], dtype=np.int64)
        if len(image_ns) == 0 or np.any(image_ns <= 0):
            raise ValueError(
                f"image HDF5 is empty or incomplete: {image_h5}; wait for image "
                "processing to finish")
        index, offset_ms, gap_valid = nearest_sample_indices(
            image_ns, query_ns, max_offset_ms)
        source_valid = _gather_dataset(source["frames/valid"], index).astype(np.uint8)
        valid = (gap_valid.astype(bool) & source_valid.astype(bool)).astype(np.uint8)
        source_timestamp = image_ns[index]
        is_new = np.r_[True, np.diff(index) != 0].astype(np.uint8)

        scalar = {
            "source_index": index,
            "source_timestamp_ns": source_timestamp,
            "source_offset_ms": offset_ms,
            "is_new_sample": is_new,
            "valid": valid,
        }
        if "frames/svo_frame" in source:
            scalar["svo_frame"] = _gather_dataset(source["frames/svo_frame"], index)
        if "frames/svo_timestamp_ns" in source:
            scalar["svo_timestamp_ns"] = _gather_dataset(
                source["frames/svo_timestamp_ns"], index)
        for name, value in scalar.items():
            _create_numeric_dataset(group, name, value)

        string_type = h5py.string_dtype(encoding="utf-8")
        status = _gather_dataset(source["frames/status"], index) if (
            "frames/status" in source) else np.full(count, "", dtype=object)
        group.create_dataset("status", data=status, dtype=string_type)

        # Only model-ready geometry and scalar quality are copied. Raw masks and
        # 2-D prompts remain in the image-processing product referenced below.
        for source_group_name in ("full", "distal", "quality"):
            if source_group_name not in source:
                continue
            destination = group.create_group(source_group_name)
            for name, dataset in source[source_group_name].items():
                if not hasattr(dataset, "shape") or dataset.shape[0] != len(image_ns):
                    continue
                values = _gather_dataset(dataset, index)
                if values.dtype.kind in "OUS":
                    destination.create_dataset(name, data=values, dtype=string_type)
                else:
                    _create_numeric_dataset(destination, name, values)
        group.attrs["source_path"] = str(image_h5.resolve())
        group.attrs["source_schema_version"] = source.attrs.get("schema_version", -1)
    return valid


def _tip_arrays(tips: list[TipPose]) -> dict[str, np.ndarray]:
    return {
        "valid": np.asarray([tip.valid for tip in tips], dtype=np.uint8),
        "position_base_mm": np.asarray([tip.position_base_mm for tip in tips]),
        "z_direction_base": np.asarray([tip.z_direction_base for tip in tips]),
        "quaternion_base_wxyz": np.asarray(
            [tip.quaternion_base_wxyz for tip in tips]),
        "coil_positions_base_mm": np.asarray(
            [tip.coil_positions_base_mm for tip in tips]),
        "coil_separation_mm": np.asarray([tip.coil_separation_mm for tip in tips]),
        "max_sync_offset_ms": np.asarray(
            [tip.max_sync_offset_ms for tip in tips]),
        "max_bracket_span_ms": np.asarray(
            [tip.max_bracket_span_ms for tip in tips]),
    }


def _write_em(group, synchronizer, query_ns: np.ndarray, enabled: bool) -> np.ndarray:
    import h5py

    count = len(query_ns)
    group.attrs["enabled"] = np.uint8(enabled)
    if enabled:
        if synchronizer is None:
            raise ValueError("EM is enabled but no EM synchronizer was supplied")
        tips = [synchronizer.tip_pose(int(timestamp)) for timestamp in query_ns]
    else:
        tips = [TipPose.invalid(int(timestamp), "disabled") for timestamp in query_ns]
    arrays = _tip_arrays(tips)
    for name, value in arrays.items():
        _create_numeric_dataset(group, name, value)
    string_type = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(
        "status", data=np.asarray([tip.status for tip in tips], dtype=object),
        dtype=string_type)
    valid = arrays["valid"]
    return valid if enabled else np.ones(count, dtype=np.uint8)


def write_fused_dataset(
        output_h5: os.PathLike | str,
        query_timestamp_ns: np.ndarray,
        robot: AlignedRobotData,
        *,
        image_h5: os.PathLike | str | None = None,
        use_image: bool = False,
        em_synchronizer=None,
        use_em: bool = False,
        config: FusionConfig | None = None,
        metadata: dict | None = None) -> dict:
    """Write an aligned dataset; exposed separately for deterministic tests."""
    import h5py

    config = config or FusionConfig()
    query_ns = np.asarray(query_timestamp_ns, dtype=np.int64)
    output_path = Path(output_h5).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with h5py.File(output_path, "w") as output:
        output.attrs["schema_version"] = 1
        output.attrs["mode"] = "actuation_clock_sensor_fusion"
        output.attrs["coordinate_frame"] = "robot_base"
        output.attrs["position_units"] = "mm"
        output.attrs["curvature_units"] = "1/mm"
        output.attrs["master_clock"] = "rosbag_joint_velocity_command"
        output.attrs["robot_joint_order_json"] = json.dumps([
            "catheter_lin", "catheter_rot", "catheter_bend",
            "sheath_lin", "sheath_rot", "sheath_bend"])
        output.attrs["robot_joint_units_json"] = json.dumps([
            "mm", "deg", "mm", "mm", "deg", "deg"])
        output.attrs["em_quaternion_order"] = "wxyz"
        output.attrs["image_enabled"] = np.uint8(use_image)
        output.attrs["em_enabled"] = np.uint8(use_em)
        output.attrs["fusion_config_json"] = json.dumps(asdict(config), sort_keys=True)
        output.attrs["metadata_json"] = json.dumps(metadata or {}, sort_keys=True)

        frames = output.create_group("frames")
        _create_numeric_dataset(frames, "timestamp_ns", query_ns)
        robot_group = output.create_group("robot")
        _write_robot(robot_group, robot)
        image_valid = _write_image(
            output.create_group("image"),
            None if image_h5 is None else Path(image_h5), query_ns,
            use_image, config.max_image_offset_ms)
        em_valid = _write_em(
            output.create_group("em"), em_synchronizer, query_ns, use_em)
        actuation_valid = robot.command_valid.astype(np.uint8)
        fusion_valid = (
            actuation_valid.astype(bool) & image_valid.astype(bool)
            & em_valid.astype(bool)).astype(np.uint8)
        _create_numeric_dataset(frames, "actuation_valid", actuation_valid)
        _create_numeric_dataset(frames, "fusion_valid", fusion_valid)

    return {
        "output_h5": str(output_path),
        "sample_count": int(len(query_ns)),
        "actuation_valid_count": int(np.count_nonzero(robot.command_valid)),
        "fusion_valid_count": int(np.count_nonzero(fusion_valid)),
        "image_valid_count": (
            int(np.count_nonzero(image_valid)) if use_image else None),
        "em_valid_count": int(np.count_nonzero(em_valid)) if use_em else None,
        "image_enabled": bool(use_image),
        "em_enabled": bool(use_em),
        "elapsed_s": time.perf_counter() - started,
    }


def _modality_default(session: Path, name: str, fallback_exists: bool) -> bool:
    metadata_path = session / "session_metadata.json"
    if metadata_path.is_file():
        try:
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            key = "stereo_camera" if name == "image" else "em_tracking"
            value = (document.get("modalities") or {}).get(key)
            if value is not None:
                return bool(value) and fallback_exists
        except (OSError, ValueError):
            pass
    return fallback_exists


def fuse_session(
        session_path: os.PathLike | str,
        output_dir: os.PathLike | str | None = None,
        image_h5: os.PathLike | str | None = None,
        em_csv: os.PathLike | str | None = None,
        use_image: bool | None = None,
        use_em: bool | None = None,
        config: FusionConfig | None = None,
        window: str = "run_and_return",
        start_ns: int | None = None,
        end_ns: int | None = None,
        stride: int = 1,
        max_samples: int | None = None) -> dict:
    """Load one session and create its post-hoc fused learning dataset."""
    config = config or FusionConfig()
    session = Path(session_path).resolve()
    output = (Path(output_dir).resolve() if output_dir is not None
              else session / "processed_fusion")
    image_path = (Path(image_h5).resolve() if image_h5 is not None
                  else session / "processed_image" / "processed_shapes.h5")
    em_path = (Path(em_csv).resolve() if em_csv is not None
               else session / "em_poses.csv")
    if use_image is None:
        use_image = _modality_default(session, "image", image_path.is_file())
    if use_em is None:
        use_em = _modality_default(session, "em", em_path.is_file())

    streams = load_robot_streams(session)
    markers = load_collection_markers(session)
    query_ns = select_actuation_timestamps(
        streams, markers, window, start_ns, end_ns, stride, max_samples)
    aligned = align_robot_streams(
        streams, query_ns, config.max_command_age_ms,
        config.max_feedback_gap_ms)
    synchronizer = None
    registration_path = None
    if use_em:
        if not em_path.is_file():
            raise FileNotFoundError(f"enabled EM CSV does not exist: {em_path}")
        registration = load_session_registration(session, require_em=True)
        registration_path = str(registration.path)
        synchronizer = EmSynchronizer.from_csv(
            em_path, registration.base_T_aurora,
            config.max_em_interp_gap_ms, config.max_em_nearest_ms)

    metadata = {
        "session": session.name,
        "session_path": str(session),
        "window": window,
        "start_ns": int(query_ns[0]),
        "end_ns": int(query_ns[-1]),
        "stride": int(stride),
        "collection_markers": markers,
        "image_path": str(image_path) if use_image else None,
        "em_csv_path": str(em_path) if use_em else None,
        "registration_path": registration_path,
    }
    summary = write_fused_dataset(
        output / "fused_dataset.h5", query_ns, aligned,
        image_h5=image_path, use_image=use_image,
        em_synchronizer=synchronizer, use_em=use_em,
        config=config, metadata=metadata)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fusion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--image-h5", default=None)
    parser.add_argument("--em-csv", default=None)
    parser.add_argument(
        "--image-data", action=argparse.BooleanOptionalAction, default=None,
        help="enable/disable processed image shape (default: session auto-detect)")
    parser.add_argument(
        "--em-data", action=argparse.BooleanOptionalAction, default=None,
        help="enable/disable EM tip tracking (default: session auto-detect)")
    parser.add_argument(
        "--window", choices=("trajectory", "run_and_return", "recording"),
        default="run_and_return")
    parser.add_argument("--start-ns", type=int, default=None)
    parser.add_argument("--end-ns", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-image-offset-ms", type=float, default=25.0)
    parser.add_argument("--max-command-age-ms", type=float, default=30.0)
    parser.add_argument("--max-feedback-gap-ms", type=float, default=30.0)
    parser.add_argument("--max-em-interp-gap-ms", type=float, default=75.0)
    parser.add_argument("--max-em-nearest-ms", type=float, default=25.0)
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    config = FusionConfig(
        max_image_offset_ms=args.max_image_offset_ms,
        max_command_age_ms=args.max_command_age_ms,
        max_feedback_gap_ms=args.max_feedback_gap_ms,
        max_em_interp_gap_ms=args.max_em_interp_gap_ms,
        max_em_nearest_ms=args.max_em_nearest_ms,
    )
    summary = fuse_session(
        args.session, args.outdir, args.image_h5, args.em_csv,
        args.image_data, args.em_data, config, args.window,
        args.start_ns, args.end_ns, args.stride, args.max_samples)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
