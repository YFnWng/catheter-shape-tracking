"""Multimodal offline processing for recorded catheter sessions.

Example pilot:

    python -m shape_tracking.sequence --session SESSION --stride 300 \
        --max-frames 20 --write-video --write-3d-video

Full trajectory:

    python -m shape_tracking.sequence --session SESSION --write-video

The processor reads the SVO directly, synchronizes both EM tip coils, segments
the dark proximal and brighter distal blue shaft, reconstructs in stereo,
connects the observed curve to the registered base and EM tip, and writes fixed
arc-sampled shapes and curvature to HDF5.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sqlite3
import struct
import time

import cv2
import numpy as np

from .geometry import (
    AssembledShape,
    assemble_anchored_shape,
    curve_geometry,
    stereo_condition_score,
)
from .materials import (
    MaterialCenterline,
    draw_material_overlay,
    extract_material_centerline,
)
from .sequence_reconstruction import (
    project_camera_points,
    reconstruct_disparity_anchored,
)
from .session import (
    EmSynchronizer,
    FrameRecord,
    SvoReader,
    TipPose,
    find_svo,
    load_frame_index,
    load_session_registration,
    project_points,
    transform_points,
)


@dataclass(frozen=True)
class ProcessingConfig:
    full_samples: int = 128
    distal_samples: int = 64
    stereo_samples: int = 96
    disparity_order: int = 3
    smooth_2d: float | None = None
    curvature_smoothing_mm: float = 1.0
    max_interp_gap_ms: float = 75.0
    max_nearest_em_ms: float = 25.0
    min_centerline_points: int = 20
    min_material_confidence: float = 0.35
    max_mean_reprojection_px: float = 5.0
    max_p95_reprojection_px: float = 12.0
    video_scale: float = 0.5
    tip_axis_length_mm: float = 20.0
    boundary_ema_alpha: float = 0.25
    boundary_max_rate_per_s: float = 0.3


def _decode_std_string(serialized: bytes) -> str:
    """Decode a ROS 2 CDR ``std_msgs/String`` without importing ROS."""
    if len(serialized) < 8:
        raise ValueError("serialized String is too short")
    # CDR encapsulation 0x0001 is little-endian; rosbag2 Humble uses it here.
    little = serialized[1] == 1
    endian = "<" if little else ">"
    length = struct.unpack_from(endian + "I", serialized, 4)[0]
    if length < 1 or 8 + length > len(serialized):
        raise ValueError("invalid serialized String length")
    return serialized[8:8 + length - 1].decode("utf-8")


def load_collection_markers(session: Path) -> dict[str, dict]:
    """Read collection JSON markers from a copied rosbag, when available."""
    # Current dual-camera recorder sessions use ``robot_bag``; retain the
    # historical ``rosbag`` name for older recordings.
    databases = sorted({
        *session.glob("rosbag/*.db3"),
        *session.glob("robot_bag/*.db3"),
    })
    markers = {}
    for database in databases:
        connection = sqlite3.connect(str(database))
        try:
            row = connection.execute(
                "SELECT id FROM topics WHERE name='/collection/events'"
            ).fetchone()
            if row is None:
                continue
            for receipt_ns, blob in connection.execute(
                    "SELECT timestamp,data FROM messages "
                    "WHERE topic_id=? ORDER BY timestamp", (row[0],)):
                try:
                    marker = json.loads(_decode_std_string(blob))
                except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                    continue
                marker["receipt_timestamp_ns"] = int(receipt_ns)
                markers[marker.get("event", f"unknown_{receipt_ns}")] = marker
        finally:
            connection.close()
    return markers


def select_frame_records(
        records: list[FrameRecord],
        markers: dict[str, dict],
        window: str,
        start_ns: int | None,
        end_ns: int | None,
        stride: int,
        max_frames: int | None) -> list[FrameRecord]:
    if start_ns is None and window in ("trajectory", "run_and_return"):
        start_ns = (markers.get("run_start") or {}).get("stamp_ns")
    if end_ns is None and window == "trajectory":
        end_ns = (markers.get("return_start") or {}).get("stamp_ns")
    if end_ns is None and window == "run_and_return":
        end_ns = (markers.get("run_end") or {}).get("stamp_ns")
    selected = [
        record for record in records
        if (start_ns is None or record.timestamp_ns >= int(start_ns))
        and (end_ns is None or record.timestamp_ns < int(end_ns))]
    selected = selected[::max(1, int(stride))]
    if max_frames is not None:
        selected = selected[:max(0, int(max_frames))]
    if not selected:
        raise ValueError(
            "no camera frames selected; verify collection markers/time bounds")
    return selected


class H5SequenceWriter:
    """Preallocated fixed-shape HDF5 writer."""

    def __init__(
            self,
            path: Path,
            frame_count: int,
            config: ProcessingConfig,
            metadata: dict):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "sequence output requires h5py; install the project again after "
                "adding its updated dependencies") from exc
        self.h5py = h5py
        self.path = path
        self.file = h5py.File(path, "w")
        self.file.attrs["schema_version"] = 1
        self.file.attrs["coordinate_frame"] = "robot_base"
        self.file.attrs["position_units"] = "mm"
        self.file.attrs["curvature_units"] = "1/mm"
        self.file.attrs["tip_roll_convention"] = (
            "z=sign-aligned mean coil z; x=part 07222026_01 to part 003, "
            "projected perpendicular to z")
        self.file.attrs["tip_coil_position_order_json"] = json.dumps([
            "003", "07222026_01"])
        self.file.attrs["observation_class_json"] = json.dumps({
            0: "invalid_or_unset",
            1: "image_observed_proximal",
            2: "image_observed_distal",
            3: "base_bridge_modelled",
            4: "tip_bridge_modelled",
        }, sort_keys=True)
        self.file.attrs["curvature_validity"] = (
            "NaN at open-curve derivative edges; class 3/4 is modelled, not "
            "directly image-observed")
        self.file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        self.file.attrs["processing_config_json"] = json.dumps(
            asdict(config), sort_keys=True)
        self.frame_count = frame_count
        self._create_datasets(frame_count, config)

    def _dataset(self, name, shape, dtype=np.float32, fillvalue=None):
        kwargs = {"shape": shape, "dtype": dtype, "compression": "gzip",
                  "compression_opts": 4, "shuffle": True}
        if fillvalue is not None:
            kwargs["fillvalue"] = fillvalue
        return self.file.create_dataset(name, **kwargs)

    def _create_datasets(self, count: int, config: ProcessingConfig) -> None:
        frames = self.file.create_group("frames")
        frames.create_dataset("svo_frame", shape=(count,), dtype=np.int32)
        frames.create_dataset("timestamp_ns", shape=(count,), dtype=np.int64)
        frames.create_dataset(
            "svo_timestamp_ns", shape=(count,), dtype=np.int64)
        frames.create_dataset("valid", shape=(count,), dtype=np.uint8)
        string_type = self.h5py.string_dtype(encoding="utf-8")
        frames.create_dataset("status", shape=(count,), dtype=string_type)

        tip = self.file.create_group("tip")
        self._dataset("tip/position_base_mm", (count, 3), fillvalue=np.nan)
        self._dataset("tip/z_direction_base", (count, 3), fillvalue=np.nan)
        self._dataset("tip/quaternion_base_wxyz", (count, 4), fillvalue=np.nan)
        self._dataset("tip/coil_positions_base_mm", (count, 2, 3),
                      fillvalue=np.nan)
        self._dataset("tip/coil_separation_mm", (count,), fillvalue=np.nan)
        self._dataset("tip/max_sync_offset_ms", (count,), fillvalue=np.nan)
        self._dataset("tip/max_bracket_span_ms", (count,), fillvalue=np.nan)
        tip.create_dataset("valid", shape=(count,), dtype=np.uint8)

        for name, samples in (
                ("full", config.full_samples),
                ("distal", config.distal_samples)):
            self.file.create_group(name)
            self._dataset(f"{name}/points_base_mm", (count, samples, 3),
                          fillvalue=np.nan)
            self._dataset(f"{name}/s_mm", (count, samples), fillvalue=np.nan)
            self._dataset(f"{name}/tangent_base", (count, samples, 3),
                          fillvalue=np.nan)
            self._dataset(f"{name}/curvature_per_mm", (count, samples),
                          fillvalue=np.nan)
            self._dataset(
                f"{name}/observation_class", (count, samples),
                dtype=np.uint8, fillvalue=0)

        quality = self.file.create_group("quality")
        for name in (
                "reprojection_left_px", "reprojection_right_px",
                "reprojection_max_px", "reprojection_p95_px",
                "stereo_condition",
                "material_boundary_fraction", "material_boundary_confidence",
                "material_boundary_contrast", "distal_boundary_s_mm",
                "tip_bridge_length_mm", "base_bridge_length_mm",
                "visible_arc_length_mm"):
            self._dataset(f"quality/{name}", (count,), fillvalue=np.nan)
        quality.create_dataset(
            "material_boundary_observed", shape=(count,), dtype=np.uint8)

    def write_identity(
            self, index: int, record: FrameRecord, svo_timestamp_ns: int,
            tip: TipPose) -> None:
        self.file["frames/svo_frame"][index] = record.svo_frame
        self.file["frames/timestamp_ns"][index] = record.timestamp_ns
        self.file["frames/svo_timestamp_ns"][index] = svo_timestamp_ns
        self.file["tip/valid"][index] = int(tip.valid)
        self.file["tip/position_base_mm"][index] = tip.position_base_mm
        self.file["tip/z_direction_base"][index] = tip.z_direction_base
        self.file["tip/quaternion_base_wxyz"][index] = (
            tip.quaternion_base_wxyz)
        self.file["tip/coil_positions_base_mm"][index] = (
            tip.coil_positions_base_mm)
        self.file["tip/coil_separation_mm"][index] = tip.coil_separation_mm
        self.file["tip/max_sync_offset_ms"][index] = tip.max_sync_offset_ms
        self.file["tip/max_bracket_span_ms"][index] = tip.max_bracket_span_ms

    def write_failure(self, index: int, status: str) -> None:
        self.file["frames/valid"][index] = 0
        self.file["frames/status"][index] = status

    def write_success(
            self,
            index: int,
            assembled: AssembledShape,
            full_geometry,
            distal_geometry,
            metrics: dict) -> None:
        self.file["frames/valid"][index] = 1
        self.file["frames/status"][index] = "valid"
        for name, shape, geometry in (
                ("full", assembled, full_geometry),
                ("distal", assembled, distal_geometry)):
            if name == "full":
                s = shape.full_s_mm
                labels = shape.full_observation_class
            else:
                s = shape.distal_s_mm
                labels = shape.distal_observation_class
            self.file[f"{name}/points_base_mm"][index] = geometry.points_mm
            self.file[f"{name}/s_mm"][index] = s
            self.file[f"{name}/tangent_base"][index] = geometry.tangent
            self.file[f"{name}/curvature_per_mm"][index] = (
                geometry.curvature_per_mm)
            self.file[f"{name}/observation_class"][index] = labels
        for key, value in metrics.items():
            path = f"quality/{key}"
            if path in self.file:
                self.file[path][index] = value

    def close(self) -> None:
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None

    def __enter__(self) -> "H5SequenceWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class OverlayVideoWriter:
    def __init__(
            self, output_dir: Path, fps: float, scale: float,
            enabled: bool):
        self.output_dir = output_dir
        self.fps = fps
        self.scale = scale
        self.enabled = enabled
        self.left = None
        self.right = None

    def write(self, left: np.ndarray, right: np.ndarray) -> None:
        if not self.enabled:
            return
        if self.scale != 1.0:
            left = cv2.resize(left, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        if self.left is None:
            height, width = left.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.left = cv2.VideoWriter(
                str(self.output_dir / "overlay_left.mp4"), fourcc, self.fps,
                (width, height))
            self.right = cv2.VideoWriter(
                str(self.output_dir / "overlay_right.mp4"), fourcc, self.fps,
                (width, height))
            if not self.left.isOpened() or not self.right.isOpened():
                raise RuntimeError("OpenCV failed to open overlay video output")
        self.left.write(left)
        self.right.write(right)

    def close(self) -> None:
        if self.left is not None:
            self.left.release()
            self.right.release()
            self.left = self.right = None


class Plot3DVideoWriter:
    def __init__(
            self,
            path: Path,
            fps: float,
            enabled: bool,
            workspace_base_m: dict):
        self.path = path
        self.fps = fps
        self.enabled = enabled
        self.workspace = workspace_base_m
        self.writer = None
        self.last_image = None

    def _render(
            self,
            full_points: np.ndarray,
            distal_points: np.ndarray,
            curvature: np.ndarray,
            tip: TipPose,
            timestamp_s: float) -> np.ndarray:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7.2, 6.0), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(
            full_points[:, 0], full_points[:, 1], full_points[:, 2],
            color="0.45", linewidth=2.0, label="full shaft")
        finite = np.isfinite(curvature)
        if np.any(finite):
            high = float(np.nanpercentile(curvature[finite], 95))
            high = max(high, 1e-6)
            color = np.clip(curvature / high, 0.0, 1.0)
            axis.scatter(
                distal_points[:, 0], distal_points[:, 1], distal_points[:, 2],
                c=color, cmap="turbo", s=12, vmin=0.0, vmax=1.0,
                label="distal / curvature")
        axis.scatter(
            tip.coil_positions_base_mm[:, 0],
            tip.coil_positions_base_mm[:, 1],
            tip.coil_positions_base_mm[:, 2],
            color=("magenta", "orange"), s=30)
        p = tip.position_base_mm
        z = tip.z_direction_base * 20.0
        axis.quiver(p[0], p[1], p[2], z[0], z[1], z[2],
                    color="red", linewidth=2.0)
        axis.scatter([0], [0], [0], color="green", s=45, label="base")
        axis.set_xlim(np.asarray(self.workspace["x"]) * 1000.0)
        axis.set_ylim(np.asarray(self.workspace["y"]) * 1000.0)
        axis.set_zlim(np.asarray(self.workspace["z"]) * 1000.0)
        axis.set_xlabel("base x (mm)")
        axis.set_ylabel("base y (mm)")
        axis.set_zlabel("base z (mm)")
        axis.set_title(f"Catheter shape, t={timestamp_s:.3f} s")
        axis.view_init(elev=18, azim=-55)
        axis.legend(loc="upper left", fontsize=8)
        figure.tight_layout()
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def write(
            self,
            full_points: np.ndarray,
            distal_points: np.ndarray,
            curvature: np.ndarray,
            tip: TipPose,
            timestamp_s: float) -> None:
        if not self.enabled:
            return
        image = self._render(
            full_points, distal_points, curvature, tip, timestamp_s)
        self.last_image = image
        if self.writer is None:
            height, width = image.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                (width, height))
            if not self.writer.isOpened():
                raise RuntimeError("OpenCV failed to open 3D video output")
        self.writer.write(image)

    def close(self, preview_path: Path | None = None) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if preview_path is not None and self.last_image is not None:
            cv2.imwrite(str(preview_path), self.last_image)


def _camera_point_from_base(
        camera_T_base: np.ndarray,
        point_base_mm: np.ndarray) -> np.ndarray:
    return transform_points(camera_T_base, np.asarray(point_base_mm) * 1e-3)


def _base_points_from_camera(
        camera_T_base: np.ndarray,
        points_camera_m: np.ndarray) -> np.ndarray:
    return transform_points(
        np.linalg.inv(camera_T_base), points_camera_m) * 1000.0


def _project_shape(
        registration,
        points_base_mm: np.ndarray,
        right: bool = False) -> np.ndarray:
    transform = (
        registration.right_camera_T_base if right
        else registration.left_camera_T_base)
    return project_points(registration.K, transform, points_base_mm)[0]


def _failure_overlay(
        image: np.ndarray, status: str, tip_pixel: np.ndarray | None) -> np.ndarray:
    output = image.copy()
    cv2.putText(output, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 3, cv2.LINE_AA)
    if tip_pixel is not None and np.all(np.isfinite(tip_pixel)):
        cv2.circle(output, tuple(np.rint(tip_pixel).astype(int)), 7,
                   (0, 0, 255), 2, cv2.LINE_AA)
    return output


def process_session(
        session_path: os.PathLike | str,
        output_dir: os.PathLike | str | None = None,
        config: ProcessingConfig | None = None,
        window: str = "trajectory",
        start_ns: int | None = None,
        end_ns: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
        write_video: bool = False,
        write_3d_video: bool = False) -> dict:
    """Process one recorded session and return its summary."""
    config = config or ProcessingConfig()
    session = Path(session_path).resolve()
    output = (
        Path(output_dir).resolve() if output_dir is not None
        else session / "processed")
    output.mkdir(parents=True, exist_ok=True)
    registration = load_session_registration(session)
    markers = load_collection_markers(session)
    records = select_frame_records(
        load_frame_index(session), markers, window, start_ns, end_ns,
        stride, max_frames)
    synchronizer = EmSynchronizer.from_csv(
        session / "em_poses.csv", registration.base_T_aurora,
        max_interp_gap_ms=config.max_interp_gap_ms,
        max_nearest_ms=config.max_nearest_em_ms)
    median_period_s = float(np.median(np.diff([
        record.timestamp_ns for record in records]))) * 1e-9 if len(records) > 1 else 1 / 30
    output_fps = float(np.clip(1.0 / max(median_period_s, 1e-6), 1.0, 60.0))
    metadata = {
        "session": session.name,
        "session_path": str(session),
        "registration_path": str(registration.path),
        "svo_path": str(find_svo(session)),
        "window": window,
        "start_ns": records[0].timestamp_ns,
        "end_ns": records[-1].timestamp_ns,
        "stride": stride,
        "source_frame_count": len(records),
        "collection_markers": markers,
    }
    with (output / "processing_config.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "metadata": metadata,
            "config": asdict(config),
        }, stream, indent=2, sort_keys=True)

    base_mm = np.zeros(3, dtype=np.float64)
    base_left = project_points(
        registration.K, registration.left_camera_T_base, base_mm)[0][0]
    base_right = project_points(
        registration.K, registration.right_camera_T_base, base_mm)[0][0]
    summary_rows = []
    counts: dict[str, int] = {}
    boundary_state = None
    boundary_timestamp_ns = None
    last_valid_plot = None
    started = time.perf_counter()

    overlay_writer = OverlayVideoWriter(
        output, output_fps, config.video_scale, write_video)
    plot_writer = Plot3DVideoWriter(
        output / "shape_3d.mp4", output_fps, write_3d_video,
        registration.workspace_base_m)
    writer = H5SequenceWriter(
        output / "processed_shapes.h5", len(records), config, metadata)
    try:
        with SvoReader(find_svo(session)) as svo:
            for index, (record, svo_timestamp_ns, left, right) in enumerate(
                    svo.iter_records(records)):
                tip = synchronizer.tip_pose(record.timestamp_ns)
                writer.write_identity(index, record, svo_timestamp_ns, tip)
                status = "valid"
                metrics = {}
                left_tip = right_tip = left_tip_axis = right_tip_axis = None
                left_material = right_material = None
                assembled = full_geometry = distal_geometry = None
                try:
                    if not tip.valid:
                        raise ValueError("em_" + tip.status)
                    left_tip = _project_shape(
                        registration, tip.position_base_mm[None, :])[0]
                    right_tip = _project_shape(
                        registration, tip.position_base_mm[None, :], True)[0]
                    axis_point = (
                        tip.position_base_mm
                        + config.tip_axis_length_mm * tip.z_direction_base)
                    left_tip_axis = _project_shape(
                        registration, axis_point[None, :])[0]
                    right_tip_axis = _project_shape(
                        registration, axis_point[None, :], True)[0]

                    left_material = extract_material_centerline(
                        left, registration.roi_left_xywh, base_left, left_tip)
                    right_material = extract_material_centerline(
                        right, registration.roi_right_xywh, base_right, right_tip)
                    if left_material is None or len(left_material) < config.min_centerline_points:
                        raise ValueError("segmentation_left")
                    if right_material is None or len(right_material) < config.min_centerline_points:
                        raise ValueError("segmentation_right")

                    material_candidates = [
                        material for material in (left_material, right_material)
                        if material.material_valid]
                    boundary_observed = bool(material_candidates)
                    if material_candidates:
                        weights = np.asarray([
                            max(material.boundary_confidence, 1e-3)
                            for material in material_candidates])
                        fractions = np.asarray([
                            material.distal_boundary_fraction
                            for material in material_candidates])
                        observed_boundary = float(
                            np.average(fractions, weights=weights))
                        if boundary_state is None:
                            boundary_state = observed_boundary
                        else:
                            alpha = config.boundary_ema_alpha
                            proposed_delta = alpha * (
                                observed_boundary - boundary_state)
                            elapsed_s = max(
                                (record.timestamp_ns - boundary_timestamp_ns)
                                * 1e-9, 0.0)
                            max_delta = (
                                config.boundary_max_rate_per_s * elapsed_s)
                            boundary_state += float(np.clip(
                                proposed_delta, -max_delta, max_delta))
                        boundary_timestamp_ns = record.timestamp_ns
                    if boundary_state is None:
                        raise ValueError("material_boundary_unresolved")
                    boundary_confidence = float(max(
                        left_material.boundary_confidence,
                        right_material.boundary_confidence))
                    boundary_contrast = float(max(
                        left_material.boundary_contrast,
                        right_material.boundary_contrast))

                    base_camera_m = _camera_point_from_base(
                        registration.left_camera_T_base, base_mm)
                    tip_camera_m = _camera_point_from_base(
                        registration.left_camera_T_base,
                        tip.position_base_mm)
                    reconstruction = reconstruct_disparity_anchored(
                        left_material.centerline,
                        right_material.centerline,
                        registration.K,
                        registration.baseline_m,
                        base_camera_m,
                        tip_camera_m,
                        n_samples=config.stereo_samples,
                        disparity_order=config.disparity_order,
                        smooth_2d=config.smooth_2d)
                    visible_base_mm = _base_points_from_camera(
                        registration.left_camera_T_base,
                        reconstruction["points_camera_m"])
                    assembled = assemble_anchored_shape(
                        visible_base_mm,
                        boundary_state,
                        tip.position_base_mm,
                        tip.z_direction_base,
                        full_count=config.full_samples,
                        distal_count=config.distal_samples,
                        base_position_base_mm=base_mm)
                    full_geometry = curve_geometry(
                        assembled.full_points_mm,
                        config.curvature_smoothing_mm)
                    distal_geometry = curve_geometry(
                        assembled.distal_points_mm,
                        config.curvature_smoothing_mm)
                    stereo_score = min(
                        stereo_condition_score(left_material.points),
                        stereo_condition_score(right_material.points))
                    metrics = {
                        "reprojection_left_px": reconstruction[
                            "reprojection_left_px"],
                        "reprojection_right_px": reconstruction[
                            "reprojection_right_px"],
                        "reprojection_max_px": reconstruction[
                            "reprojection_max_px"],
                        "reprojection_p95_px": reconstruction[
                            "reprojection_p95_px"],
                        "stereo_condition": stereo_score,
                        "material_boundary_fraction": boundary_state,
                        "material_boundary_confidence": boundary_confidence,
                        "material_boundary_contrast": boundary_contrast,
                        "material_boundary_observed": int(boundary_observed),
                        "distal_boundary_s_mm": assembled.distal_boundary_s_mm,
                        "tip_bridge_length_mm": assembled.tip_bridge_length_mm,
                        "base_bridge_length_mm": assembled.base_bridge_length_mm,
                        "visible_arc_length_mm": reconstruction[
                            "visible_arc_length_mm"],
                    }
                    worst_mean = max(
                        reconstruction["reprojection_left_px"],
                        reconstruction["reprojection_right_px"])
                    if (worst_mean > config.max_mean_reprojection_px
                            or reconstruction["reprojection_p95_px"]
                            > config.max_p95_reprojection_px):
                        raise ValueError(
                            "reprojection_exceeds_threshold:"
                            f"mean={worst_mean:.2f}px,"
                            f"p95={reconstruction['reprojection_p95_px']:.2f}px")
                    writer.write_success(
                        index, assembled, full_geometry, distal_geometry,
                        metrics)
                except (ArithmeticError, RuntimeError, ValueError,
                        np.linalg.LinAlgError) as exc:
                    status = str(exc)
                    writer.write_failure(index, status)

                counts[status] = counts.get(status, 0) + 1
                shape_left = shape_right = None
                if full_geometry is not None:
                    shape_left = _project_shape(
                        registration, full_geometry.points_mm)
                    shape_right = _project_shape(
                        registration, full_geometry.points_mm, True)
                quality_text = (
                    f"frame {record.svo_frame} {status}"
                    + (f" L/R={metrics.get('reprojection_left_px', np.nan):.1f}/"
                       f"{metrics.get('reprojection_right_px', np.nan):.1f}px"
                       if metrics else ""))
                if left_material is not None:
                    left_overlay = draw_material_overlay(
                        left, left_material, shape_left, left_tip, left_tip_axis,
                        quality_text)
                else:
                    left_overlay = _failure_overlay(left, quality_text, left_tip)
                if right_material is not None:
                    right_overlay = draw_material_overlay(
                        right, right_material, shape_right, right_tip,
                        right_tip_axis, quality_text)
                else:
                    right_overlay = _failure_overlay(right, quality_text, right_tip)
                overlay_writer.write(left_overlay, right_overlay)
                if status == "valid":
                    last_valid_plot = (
                        full_geometry.points_mm.copy(),
                        distal_geometry.points_mm.copy(),
                        distal_geometry.curvature_per_mm.copy(),
                        tip,
                        (record.timestamp_ns - records[0].timestamp_ns) * 1e-9,
                    )
                    plot_writer.write(
                        full_geometry.points_mm,
                        distal_geometry.points_mm,
                        distal_geometry.curvature_per_mm,
                        tip,
                        (record.timestamp_ns - records[0].timestamp_ns) * 1e-9)

                row = {
                    "output_index": index,
                    "svo_frame": record.svo_frame,
                    "timestamp_ns": record.timestamp_ns,
                    "svo_timestamp_ns": svo_timestamp_ns,
                    "svo_timestamp_error_ms": (
                        svo_timestamp_ns - record.timestamp_ns) / 1e6,
                    "valid": int(status == "valid"),
                    "status": status,
                    "tip_valid": int(tip.valid),
                    "tip_sync_offset_ms": tip.max_sync_offset_ms,
                    **metrics,
                }
                summary_rows.append(row)
                if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(records):
                    elapsed = time.perf_counter() - started
                    rate = (index + 1) / max(elapsed, 1e-9)
                    print(
                        f"[{index + 1}/{len(records)}] {status}; "
                        f"{rate:.2f} frames/s", flush=True)
    finally:
        writer.close()
        overlay_writer.close()
        if last_valid_plot is not None and plot_writer.last_image is None:
            plot_writer.last_image = plot_writer._render(*last_valid_plot)
        plot_writer.close(output / "shape_3d_preview.png")

    fieldnames = []
    for row in summary_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output / "frame_summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(summary_rows)
    valid_count = counts.get("valid", 0)
    summary = {
        "session": session.name,
        "output_dir": str(output),
        "frame_count": len(records),
        "valid_frame_count": valid_count,
        "valid_fraction": valid_count / len(records),
        "status_counts": counts,
        "elapsed_s": time.perf_counter() - started,
        "output_fps": output_fps,
    }
    with (output / "processing_summary.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--window", choices=("trajectory", "run_and_return", "recording"),
        default="trajectory")
    parser.add_argument("--start-ns", type=int, default=None)
    parser.add_argument("--end-ns", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--full-samples", type=int, default=128)
    parser.add_argument("--distal-samples", type=int, default=64)
    parser.add_argument("--stereo-samples", type=int, default=96)
    parser.add_argument("--disp-order", type=int, default=3)
    parser.add_argument("--smooth2d", type=float, default=None)
    parser.add_argument("--curvature-smoothing-mm", type=float, default=1.0)
    parser.add_argument("--max-em-gap-ms", type=float, default=75.0)
    parser.add_argument("--max-em-offset-ms", type=float, default=25.0)
    parser.add_argument("--max-mean-reprojection-px", type=float, default=5.0)
    parser.add_argument("--max-p95-reprojection-px", type=float, default=12.0)
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--write-3d-video", action="store_true")
    parser.add_argument("--video-scale", type=float, default=0.5)
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    config = ProcessingConfig(
        full_samples=args.full_samples,
        distal_samples=args.distal_samples,
        stereo_samples=args.stereo_samples,
        disparity_order=args.disp_order,
        smooth_2d=args.smooth2d,
        curvature_smoothing_mm=args.curvature_smoothing_mm,
        max_interp_gap_ms=args.max_em_gap_ms,
        max_nearest_em_ms=args.max_em_offset_ms,
        max_mean_reprojection_px=args.max_mean_reprojection_px,
        max_p95_reprojection_px=args.max_p95_reprojection_px,
        video_scale=args.video_scale,
    )
    summary = process_session(
        args.session,
        output_dir=args.outdir,
        config=config,
        window=args.window,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        stride=args.stride,
        max_frames=args.max_frames,
        write_video=args.write_video,
        write_3d_video=args.write_3d_video)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
