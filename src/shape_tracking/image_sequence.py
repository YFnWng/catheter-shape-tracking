"""Offline SAM 2 processing for image-only stereo catheter sessions.

This entry point deliberately does not use EM tracking. It segments each
rectified view, reconstructs the complete visible catheter (including its
yellow tip), reports the dark-blue/cyan material transition as the distal base,
and aligns robot commands and feedback to the camera timestamps.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .geometry import (
    AssembledShape,
    assemble_image_only_shape,
    cumulative_arclength,
    curve_geometry,
    stereo_condition_score,
)
from .interface_smoothing import smooth_interface_hdf5
from .joint_spline_reconstruction import (
    distal_image_observation,
    fit_joint_two_view_spline,
    project_point_to_polyline,
)
from .materials import (
    MaterialCenterline,
    _sample_brightness,
    detect_distal_boundary,
    draw_material_overlay,
)
from .marked_segmentation import (
    catheter_color_likelihood,
    extract_marked_chromatic_result,
    refine_marked_stereo_pair,
    reroute_marked_centerline,
    enforce_stereo_epipolar_sweep,
    center_marked_route_on_mask,
    _smooth_marker_route,
)
from .robot_data import AlignedRobotData, align_robot_streams, load_robot_streams
from .temporal_markers import repair_stereo_marker_tracks
from .sam_segmentation import (
    AutomaticPromptResult,
    PromptSet,
    Sam2CatheterSegmenter,
    SamMaterialResult,
    _automatic_prompts,
    load_prompt_overrides,
    material_centerline_from_mask,
)
from .segmentation import Centerline, resample_arclength
from .sequence import load_collection_markers, select_frame_records
from .sequence_reconstruction import (
    epipolar_mask_ambiguity_fraction,
    project_camera_points,
    reconstruct_disparity_anchored,
)
from .stereo_smoothing import smooth_stereo_disparity_hdf5
from .spline_temporal_smoothing import (
    interpolate_short_spline_gaps_hdf5,
    smooth_distal_spline_coefficients_hdf5,
)
from .session import (
    FrameRecord,
    SvoReader,
    find_svo,
    load_frame_index,
    load_session_registration,
    project_points,
    transform_points,
)


@dataclass(frozen=True)
class ImageProcessingConfig:
    full_samples: int = 128
    distal_samples: int = 64
    stereo_samples: int = 96
    image_centerline_samples: int = 256
    smooth_2d: float | None = None
    disparity_first_difference_weight: float = 0.25
    disparity_second_difference_weight: float = 10.0
    disparity_huber_delta_px: float = 1.5
    temporal_disparity_weight: float = 2.0
    temporal_disparity_max_gap_ms: float = 1000.0
    temporal_disparity_decay_ms: float = 500.0
    stereo_reference_hysteresis_score: float = 3.0
    overlap_aware_reconstruction: bool = True
    overlap_row_search_px: int = 2
    overlap_self_fraction_threshold: float = 0.05
    overlap_self_distance_px: float = 8.0
    overlap_min_arclength_separation: float = 0.12
    centerline_tip_depth_weight: float = 5.0
    max_centerline_tip_epipolar_error_px: float = 8.0
    terminal_disparity_refinement: bool = True
    terminal_disparity_fraction: float = 0.20
    terminal_disparity_smoothness_scale: float = 0.10
    terminal_disparity_observation_weight: float = 4.0
    terminal_disparity_tip_weight: float = 20.0
    terminal_disparity_max_p95_degradation_px: float = 0.25
    stereo_offline_cutoff_hz: float = 2.0
    stereo_offline_huber_delta_px: float = 1.5
    temporal_shape_cutoff_hz: float = 0.0
    distal_length_mm: float = 60.0
    marked_distal_length_mm: float = 57.0
    distal_length_prior_sigma_mm: float = 4.0
    interface_color_sigma_mm: float = 3.0
    interface_temporal_sigma_mm: float = 1.5
    interface_temporal_cutoff_hz: float = 4.0
    interface_projection_window_mm: float = 15.0
    interface_offline_cutoff_hz: float = 2.0
    interface_offline_huber_delta_mm: float = 2.0
    interface_offline_iterations: int = 4
    interface_length_gate_mm: float = 4.0
    interface_session_prior_relative_weight: float = 0.5
    spline_temporal_cutoff_hz: float = 3.0
    spline_temporal_terminal_cutoff_hz: float = 5.0
    spline_temporal_terminal_basis_count: int = 4
    spline_temporal_huber_delta_mm: float = 1.0
    spline_temporal_iterations: int = 4
    spline_temporal_max_gap_ms: float = 500.0
    spline_temporal_outlier_sigma: float = 4.5
    spline_temporal_outlier_floor_mm: float = 0.75
    spline_temporal_frame_outlier_fraction: float = 0.5
    spline_temporal_terminal_outlier_samples: int = 4
    spline_temporal_observation_blend: float = 0.35
    spline_temporal_terminal_observation_blend: float = 0.65
    spline_interpolation_max_gap_ms: float = 650.0
    spline_interpolation_max_good_island_frames: int = 2
    distal_boundary_search_half_width_mm: float = 12.0
    curvature_smoothing_mm: float = 0.25
    curvature_spline_bases: int = 20
    max_base_endpoint_distance_mm: float = 15.0
    max_stereo_centerline_length_ratio: float = 1.8
    marked_stereo_guided_retry_ratio: float = 1.23
    marked_ill_epipolar_ambiguity_fraction: float = 0.30
    stereo_ill_switch_confirm_frames: int = 5
    max_inferred_centerline_temporal_p95_px: float = 8.0
    max_inferred_centerline_temporal_max_px: float = 14.0
    max_inferred_sharp_turn_clusters: int = 1
    ill_eye_min_observation_weight: float = 0.20
    ill_eye_temporal_shape_sigma_mm: float = 6.0
    stereo_reference_switch_ratio: float = 1.15
    min_centerline_points: int = 20
    min_sam_iou: float = 0.25
    min_seed_recall: float = 0.75
    max_mask_area_fraction: float = 0.20
    max_mask_effective_width_px: float = 20.0
    marked_max_mask_effective_width_px: float = 28.0
    max_temporal_prompt_gap_ms: float = 100.0
    max_mean_reprojection_px: float = 5.0
    max_p95_reprojection_px: float = 12.0
    max_tip_epipolar_error_px: float = 5.0
    max_tip_endpoint_error_px: float = 8.0
    marker_tip_endpoint_width_scale: float = 0.40
    marker_tip_endpoint_width_cap_px: float = 12.0
    max_final_mask_outside_fraction: float = 0.20
    max_final_mask_distance_p95_px: float = 6.0
    chromatic_min_saturation: int = 55
    chromatic_min_value: int = 30
    chromatic_background_subtraction: bool = True
    chromatic_background_samples: int = 21
    chromatic_background_difference: int = 18
    marker_min_confidence: float = 0.50
    marked_epipolar_sweep_deficit_px: float = 4.0
    marked_epipolar_sweep_row_half_width_px: int = 3
    marker_disparity_weight: float = 8.0
    marker_interface_weight_scale: float = 2.0
    marker_tip_weight_scale: float = 3.0
    joint_spline_basis_count: int = 20
    joint_spline_max_nfev: int = 35
    max_marker_epipolar_error_px: float = 6.0
    marked_distal_length_prior_sigma_mm: float = 3.0
    marked_interface_marker_sigma_mm: float = 0.75
    marked_sheath_exclusion_radius_mm: float = 7.0
    marked_sheath_exclusion_length_mm: float = 60.0
    min_temporal_centerline_coverage: float = 0.55
    temporal_centerline_tolerance_px: float = 18.0
    max_temporal_centerline_p95_px: float = 35.0
    max_command_age_ms: float = 30.0
    max_feedback_gap_ms: float = 30.0
    video_scale: float = 0.5
    sam_frame_batch_size: int = 1
    sam_postprocess_workers: int = 2
    prompt_workers: int = 4
    chromatic_eye_workers: int = 2
    preprocess_chunk_size: int = 16
    prefetch_frames: int = 16
    hdf_buffer_frames: int = 128
    hdf_queue_chunks: int = 2
    store_masks: bool = True


def _automatic_prompt_result(
        image: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point: np.ndarray) -> AutomaticPromptResult:
    prompt, seed, tip = _automatic_prompts(image, roi, base_point)
    return AutomaticPromptResult(prompt=prompt, seed=seed, tip=tip)


class _BackgroundPrefetch:
    """Bounded producer that overlaps SVO/prompt work with main-thread SAM."""

    _END = object()

    def __init__(self, source, max_items: int):
        self.source = source
        self.items: queue.Queue = queue.Queue(maxsize=max(1, int(max_items)))
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._produce, name="image-prompt-prefetch", daemon=True)
        self.thread.start()

    def _put(self, item) -> bool:
        while not self.stop.is_set():
            try:
                self.items.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self) -> None:
        try:
            for item in self.source:
                if not self._put(item):
                    return
        except BaseException as exc:  # forwarded to the consumer
            self._put(exc)
        finally:
            self._put(self._END)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.items.get()
        if item is self._END:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.stop.set()
        close = getattr(self.source, "close", None)
        if close is not None:
            try:
                close()
            except (RuntimeError, ValueError):
                pass
        while self.thread.is_alive():
            try:
                self.items.get_nowait()
            except queue.Empty:
                pass
            self.thread.join(timeout=0.1)


def _external_mask_result(
        image: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point: np.ndarray,
        mask: np.ndarray,
        source: str,
        confidence: float = 1.0) -> SamMaterialResult:
    try:
        prompt, seed, tip = _automatic_prompts(image, roi, base_point)
    except ValueError:
        # A propagated SAM mask must remain usable in precisely the frames in
        # which the colour heuristic disappears.  Recover geometry directly
        # from video memory first, then synthesize a diagnostic prompt from
        # that geometry instead of making HSV a hidden hard prerequisite.
        seed = np.zeros_like(mask, dtype=np.uint8)
        tip = None
        provisional = material_centerline_from_mask(
            image, roi, np.uint8(mask > 0) * 255, base_point, None)
        if provisional is None:
            raise ValueError(f"{source}_centerline_missing")
        points = provisional.centerline.points
        x, y, width, height = roi
        margin = 8.0
        box = np.array([
            max(float(x), float(points[:, 0].min() - margin)),
            max(float(y), float(points[:, 1].min() - margin)),
            min(float(x + width - 1), float(points[:, 0].max() + margin)),
            min(float(y + height - 1), float(points[:, 1].max() + margin)),
        ])
        indices = np.unique(np.rint(np.linspace(
            0, len(points) - 1, min(6, len(points)))).astype(int))
        corners = np.array([
            [box[0], box[1]], [box[2], box[1]],
            [box[0], box[3]], [box[2], box[3]]])
        prompt = PromptSet(
            box_xyxy=box,
            positive_xy=points[indices],
            negative_xy=corners,
            source=f"{source}_mask_geometry")
    material = material_centerline_from_mask(
        image, roi, np.uint8(mask > 0) * 255, base_point, tip)
    if material is None:
        raise ValueError(f"{source}_centerline_missing")
    # External/cached masks are already fixed observations; their displayed
    # prompt must describe the accepted mask geometry rather than a fresh HSV
    # seed that may contain only the proximal component.
    points = material.points
    x, y, width, height = roi
    margin = 18.0
    box = np.array([
        max(x, float(np.min(points[:, 0]) - margin)),
        max(y, float(np.min(points[:, 1]) - margin)),
        min(x + width - 1, float(np.max(points[:, 0]) + margin)),
        min(y + height - 1, float(np.max(points[:, 1]) + margin)),
    ])
    indices = np.unique(np.rint(
        np.linspace(0, len(points) - 1, min(8, len(points)))).astype(int))
    corners = np.array([
        [box[0], box[1]], [box[2], box[1]],
        [box[0], box[3]], [box[2], box[3]]], dtype=np.float64)
    prompt = PromptSet(
        box_xyxy=box,
        positive_xy=points[indices],
        negative_xy=corners,
        source=f"{source}_mask_geometry")
    seed_bool = seed > 0
    area = int(np.count_nonzero(mask))
    seed_area = int(np.count_nonzero(seed_bool))
    recall = (float(np.count_nonzero((mask > 0) & seed_bool) / seed_area)
              if seed_area else 1.0)
    return SamMaterialResult(
        material=material,
        prompt=prompt,
        sam_iou=float(confidence),
        selection_score=float(confidence),
        seed_recall=recall,
        mask_area_px=area,
        yellow_tip_xy=tip)


def _ordered_material_boundary_index(
        points_xy: np.ndarray, boundary_point_xy: np.ndarray,
        boundary_fraction: float) -> int:
    """Map an interface to route order without cross-branch nearest snapping."""
    points = np.asarray(points_xy, dtype=float)
    if np.isfinite(boundary_fraction):
        return int(np.clip(round(
            float(boundary_fraction) * max(len(points) - 1, 1)),
            0, len(points) - 1))
    return int(np.argmin(np.linalg.norm(
        points - np.asarray(boundary_point_xy, dtype=float), axis=1)))


class PropagatedMaskCache:
    """Read propagation masks or masks from a completed image HDF5."""

    def __init__(self, path: os.PathLike | str, records: list[FrameRecord]):
        import h5py
        self.file = h5py.File(Path(path).resolve(), "r")
        expected = np.asarray([record.svo_frame for record in records], np.int64)
        if "frames/svo_frame" in self.file:
            self.layout = "processed_image"
            actual = np.asarray(self.file["frames/svo_frame"], np.int64)
        else:
            self.layout = "propagated"
            actual = np.asarray(self.file["svo_frame"], np.int64)
        positions = np.searchsorted(actual, expected)
        safe = np.clip(positions, 0, max(len(actual) - 1, 0))
        if (len(actual) == 0 or np.any(positions >= len(actual))
                or not np.array_equal(actual[safe], expected)):
            self.file.close()
            raise ValueError("mask cache does not contain the requested frame sequence")
        self.source_indices = safe.astype(np.int64)
        self.temporal_marker_tracks = None
        self.temporal_marker_reroute = np.zeros(len(actual), dtype=bool)
        if (self.layout == "processed_image"
                and "frames/timestamp_ns" in self.file
                and all(
                    f"images/{view}/marker_centers_px" in self.file
                    for view in ("left", "right"))):
            self.temporal_marker_tracks = repair_stereo_marker_tracks(
                {view: self.file[
                    f"images/{view}/marker_centers_px"][:]
                 for view in ("left", "right")},
                {view: self.file[
                    f"images/{view}/marker_widths_px"][:]
                 for view in ("left", "right")},
                {view: self.file[
                    f"images/{view}/marker_confidence"][:]
                 for view in ("left", "right")},
                {view: self.file[
                    f"images/{view}/marker_observed"][:]
                 for view in ("left", "right")},
                self.file["frames/timestamp_ns"][:])
            changed = np.zeros(len(actual), dtype=bool)
            for view in ("left", "right"):
                prefix = f"images/{view}"
                original_centers = np.asarray(
                    self.file[f"{prefix}/marker_centers_px"][:], float)
                original_observed = np.asarray(
                    self.file[f"{prefix}/marker_observed"][:], bool)
                repaired = self.temporal_marker_tracks[view]
                repaired_centers = np.asarray(repaired["centers"], float)
                repaired_observed = np.asarray(repaired["observed"], bool)
                both = (np.all(np.isfinite(original_centers), axis=2)
                        & np.all(np.isfinite(repaired_centers), axis=2))
                displacement = np.zeros(both.shape, dtype=float)
                displacement[both] = np.linalg.norm(
                    original_centers[both] - repaired_centers[both], axis=1)
                changed |= np.any(
                    (original_observed != repaired_observed)
                    | np.asarray(repaired["interpolated"], bool)
                    | (displacement > 0.25), axis=1)
            # Routing is temporally regularized, so also revisit a short halo
            # around every repaired marker instead of creating a seam at the
            # exact repair boundary.
            changed_indices = np.flatnonzero(changed)
            for offset in range(-2, 3):
                indices = changed_indices + offset
                indices = indices[(indices >= 0) & (indices < len(changed))]
                self.temporal_marker_reroute[indices] = True

    def marker_track_requires_reroute(self, index: int) -> bool:
        """Whether repaired marker evidence warrants rebuilding the 2D path."""
        return bool(self.temporal_marker_reroute[self.source_indices[index]])

    def result(
            self, index: int, view: str, image: np.ndarray,
            roi: tuple[int, int, int, int],
            base_point: np.ndarray) -> SamMaterialResult:
        if self.layout == "processed_image":
            dataset = self.file[f"images/{view}/mask_packbits"]
            source = "sam2_completed_h5_cache"
            stored_roi = tuple(int(value) for value in dataset.attrs["roi_xywh"])
        else:
            dataset = self.file[f"{view}/mask_packbits"]
            source = "sam2_video_propagated"
            stored_roi = tuple(
                int(value) for value in self.file[view].attrs["roi_xywh"])
        if stored_roi != tuple(roi):
            raise ValueError(f"propagated {view} ROI does not match registration")
        _, _, width, height = roi
        packed = np.asarray(dataset[self.source_indices[index]])
        mask = np.unpackbits(
            packed, bitorder="little", count=width * height).reshape(height, width)
        if (self.layout == "processed_image"
                and f"images/{view}/marker_centers_px" in self.file):
            source_index = self.source_indices[index]
            prefix = f"images/{view}"
            observation_name = (
                f"{prefix}/observed_centerline_px"
                if f"{prefix}/observed_centerline_px" in self.file
                else f"{prefix}/centerline_px")
            points = np.asarray(
                self.file[observation_name][source_index], dtype=np.float64)
            finite_points = points[np.all(np.isfinite(points), axis=1)]
            track = (
                None if self.temporal_marker_tracks is None
                else self.temporal_marker_tracks[view])
            marker_confidence = np.asarray(
                track["confidence"][source_index] if track is not None
                else self.file[f"{prefix}/marker_confidence"][source_index])
            if len(finite_points) < 2:
                # A failed source reconstruction can still contain a complete
                # packed chromatic mask. Rebuild only the initial image route;
                # repaired markers and the normal marked rerouting stages will
                # subsequently recover material order. This preserves useful
                # image evidence instead of forcing the 3D gap bridge to work
                # from masks that were never exposed to reconstruction.
                rebuilt = _external_mask_result(
                    image, roi, base_point, mask,
                    source="cached_mask_route_rebuilt", confidence=1.0)
                stored_tip = np.asarray(
                    self.file[f"{prefix}/yellow_tip_px"][source_index])
                return replace(
                    rebuilt,
                    prompt=replace(
                        rebuilt.prompt,
                        source=rebuilt.prompt.source + "+temporal_markers"),
                    yellow_tip_xy=(
                        stored_tip if np.all(np.isfinite(stored_tip)) else None),
                    marker_centers_xy=np.asarray(
                        track["centers"][source_index] if track is not None
                        else self.file[
                            f"{prefix}/marker_centers_px"][source_index]),
                    marker_widths_px=np.asarray(
                        track["widths"][source_index] if track is not None
                        else self.file[
                            f"{prefix}/marker_widths_px"][source_index]),
                    marker_confidence=marker_confidence,
                    marker_observed=np.asarray(
                        track["observed"][source_index] if track is not None
                        else self.file[
                            f"{prefix}/marker_observed"][source_index]),
                    marker_interpolated=(
                        np.asarray(track["interpolated"][source_index])
                        if track is not None else np.zeros(4, dtype=np.uint8)),
                    marker_raw_cluster_count=int(self.file[
                        f"{prefix}/marker_raw_cluster_count"][source_index]),
                    tip_source="stored_marked_observation")
            points = finite_points
            boundary_point = np.asarray(
                self.file[f"{prefix}/distal_boundary_px"][source_index],
                dtype=np.float64)
            # In a projected V, the interface and a distal branch can occupy
            # nearly the same pixel. Euclidean nearest-point lookup can then
            # place the color split at the tip and draw the whole catheter as
            # proximal blue. The stored material fraction preserves route
            # order and is the correct coordinate for the observed overlay.
            fraction_name = "quality/material_boundary_fraction"
            boundary_fraction = (
                float(self.file[fraction_name][source_index])
                if fraction_name in self.file else float("nan"))
            boundary = _ordered_material_boundary_index(
                points, boundary_point, boundary_fraction)
            confidence = float(marker_confidence[0])
            material = MaterialCenterline(
                centerline=Centerline(
                    points, roi, np.uint8(mask > 0) * 255,
                    float(np.count_nonzero(mask) / max(2.0 * len(points), 1.0))),
                distal_boundary_index=boundary,
                distal_boundary_fraction=float(
                    boundary / max(len(points) - 1, 1)),
                boundary_confidence=confidence,
                boundary_contrast=float("nan"),
                material_valid=bool(np.isfinite(confidence)),
                brightness_profile=np.full(len(points), np.nan))
            box = np.asarray(
                self.file[f"{prefix}/prompt_box_xyxy"][source_index])
            positive_count = int(self.file[
                f"{prefix}/prompt_positive_count"][source_index])
            negative_count = int(self.file[
                f"{prefix}/prompt_negative_count"][source_index])
            prompt_source = self.file[f"{prefix}/prompt_source"][source_index]
            if isinstance(prompt_source, bytes):
                prompt_source = prompt_source.decode("utf-8")
            prompt = PromptSet(
                box_xyxy=box,
                positive_xy=np.asarray(self.file[
                    f"{prefix}/prompt_positive_px"][source_index, :positive_count]),
                negative_xy=np.asarray(self.file[
                    f"{prefix}/prompt_negative_px"][source_index, :negative_count]),
                source=str(prompt_source))
            stored_tip = np.asarray(
                self.file[f"{prefix}/yellow_tip_px"][source_index])
            tip = stored_tip if np.all(np.isfinite(stored_tip)) else None
            if track is not None:
                prompt = replace(
                    prompt, source=prompt.source + "+temporal_markers")
            return SamMaterialResult(
                material=material,
                prompt=prompt,
                sam_iou=float(self.file[f"{prefix}/sam_iou"][source_index]),
                selection_score=float(self.file[
                    f"{prefix}/selection_score"][source_index]),
                seed_recall=float(self.file[f"{prefix}/seed_recall"][source_index]),
                mask_area_px=int(np.count_nonzero(mask)),
                yellow_tip_xy=tip,
                marker_centers_xy=np.asarray(
                    track["centers"][source_index] if track is not None
                    else self.file[f"{prefix}/marker_centers_px"][source_index]),
                marker_widths_px=np.asarray(
                    track["widths"][source_index] if track is not None
                    else self.file[f"{prefix}/marker_widths_px"][source_index]),
                marker_confidence=marker_confidence,
                marker_observed=np.asarray(
                    track["observed"][source_index] if track is not None
                    else self.file[f"{prefix}/marker_observed"][source_index]),
                marker_interpolated=(
                    np.asarray(track["interpolated"][source_index])
                    if track is not None else np.zeros(4, dtype=np.uint8)),
                marker_raw_cluster_count=int(self.file[
                    f"{prefix}/marker_raw_cluster_count"][source_index]),
                tip_source="stored_marked_observation")
        result = _external_mask_result(
            image, roi, base_point, mask,
            source=source, confidence=1.0)
        if self.layout == "processed_image":
            source_index = self.source_indices[index]
            prefix = f"images/{view}"
            if f"{prefix}/marker_centers_px" in self.file:
                result = replace(
                    result,
                    marker_centers_xy=np.asarray(
                        self.file[f"{prefix}/marker_centers_px"][source_index]),
                    marker_widths_px=np.asarray(
                        self.file[f"{prefix}/marker_widths_px"][source_index]),
                    marker_confidence=np.asarray(
                        self.file[f"{prefix}/marker_confidence"][source_index]),
                    marker_observed=np.asarray(
                        self.file[f"{prefix}/marker_observed"][source_index]),
                    marker_raw_cluster_count=int(self.file[
                        f"{prefix}/marker_raw_cluster_count"][source_index]))
            if f"{prefix}/yellow_tip_px" in self.file:
                stored_tip = np.asarray(
                    self.file[f"{prefix}/yellow_tip_px"][source_index])
                if np.all(np.isfinite(stored_tip)):
                    result = replace(result, yellow_tip_xy=stored_tip)
        return result

    def close(self) -> None:
        self.file.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sam_provenance(checkpoint: Path, config: str) -> dict:
    module = importlib.import_module("sam2")
    root = Path(module.__file__).resolve().parent.parent
    commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True)
        commit = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "implementation": "facebookresearch/sam2",
        "package_root": str(root),
        "git_commit": commit,
        "config": config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }


def _repository_provenance() -> dict:
    """Identify the exact local processing source, including dirty edits."""
    root = Path(__file__).resolve().parents[2]
    commit = "unknown"
    status: list[str] = []
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True)
        commit = commit_result.stdout.strip()
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--short"], check=True,
            capture_output=True, text=True)
        status = status_result.stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    digest = hashlib.sha256()
    source_root = root / "src" / "shape_tracking"
    for path in sorted(source_root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "repository_root": str(root),
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_short": status,
        "shape_tracking_source_sha256": digest.hexdigest(),
    }


def _camera_point_from_base(camera_T_base: np.ndarray, point_base_mm: np.ndarray) -> np.ndarray:
    return transform_points(
        camera_T_base, np.asarray(point_base_mm, dtype=np.float64) * 1e-3)


def _base_points_from_camera(camera_T_base: np.ndarray, points_camera_m: np.ndarray) -> np.ndarray:
    return transform_points(
        np.linalg.inv(camera_T_base), points_camera_m) * 1000.0


def _project_base_points(registration, points_base_mm: np.ndarray, right: bool = False) -> np.ndarray:
    transform = (
        registration.right_camera_T_base if right
        else registration.left_camera_T_base)
    camera = transform_points(transform, np.asarray(points_base_mm) * 1e-3)
    return project_camera_points(
        camera, registration.K, registration.baseline_m, right=False)


def _joint_temporal_stereo_prior(
        registration, points_base_mm: np.ndarray,
        sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return material-indexed disparity and left-camera points for time t+1.

    The disparity initializer is followed by a two-view joint optimization.
    Carrying the pre-joint disparity forward can therefore preserve a branch
    that the joint fit just corrected.  Resampling the finalized 3D curve here
    makes the next frame's temporal prior describe the actual stored shape.
    """
    points = np.asarray(points_base_mm, dtype=np.float64)
    source_s = cumulative_arclength(points)
    if len(points) < 2 or source_s[-1] <= 1e-9:
        raise ValueError("joint temporal prior has zero arc length")
    target_s = np.linspace(0.0, source_s[-1], int(sample_count))
    sampled = np.column_stack([
        np.interp(target_s, source_s, points[:, coordinate])
        for coordinate in range(3)])
    left = _project_base_points(registration, sampled, right=False)
    right = _project_base_points(registration, sampled, right=True)
    disparity = left[:, 0] - right[:, 0]
    camera = transform_points(
        registration.left_camera_T_base, sampled * 1e-3)
    if (not np.all(np.isfinite(disparity))
            or np.any(disparity <= 0.25)
            or not np.all(np.isfinite(camera))):
        raise ValueError("joint temporal prior is not projectable")
    return disparity, camera


def _replace_material_path(
        result: SamMaterialResult,
        ordered_points_xy: np.ndarray) -> SamMaterialResult:
    """Replace a collapsed cyan route with an inferred mask-supported route."""
    material = result.material
    points = np.asarray(ordered_points_xy, dtype=np.float64)
    if len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("inferred material path is invalid")
    mask = np.asarray(material.mask) > 0
    support_rows, support_columns = np.where(mask)
    if not len(support_columns):
        raise ValueError("inferred material path has no target-mask support")
    roi_x, roi_y, _, _ = material.roi
    support = np.column_stack([
        support_columns + roi_x, support_rows + roi_y]).astype(np.float64)
    local = np.rint(points - [roi_x, roi_y]).astype(int)
    in_bounds = (
        (local[:, 0] >= 0) & (local[:, 0] < mask.shape[1])
        & (local[:, 1] >= 0) & (local[:, 1] < mask.shape[0]))
    supported = np.zeros(len(points), dtype=bool)
    supported[in_bounds] = mask[
        local[in_bounds, 1], local[in_bounds, 0]]
    support_tree = cKDTree(support)
    for index in np.flatnonzero(~supported):
        # Complete a missing DP row on that same epipolar line whenever
        # possible. A global nearest-mask snap can cross to the other V arm.
        row = int(np.rint(points[index, 1] - roi_y))
        columns = (
            np.flatnonzero(mask[row])
            if 0 <= row < mask.shape[0] else np.empty(0, dtype=int))
        if len(columns):
            column = int(columns[np.argmin(np.abs(
                columns + roi_x - points[index, 0]))])
            points[index] = [column + roi_x, row + roi_y]
        else:
            points[index] = support[int(support_tree.query(
                points[index], k=1)[1])]
    # Do not resample/smooth/snap the inferred corridor here. Those operations
    # are not topology-aware and previously undid the DP branch choices. The
    # downstream 3D B-spline supplies spatial smoothness in material order.
    old_boundary = material.points[int(np.clip(
        material.distal_boundary_index, 0, len(material.points) - 1))]
    boundary = int(np.argmin(np.linalg.norm(points - old_boundary, axis=1)))
    old_profile = np.asarray(material.brightness_profile, dtype=np.float64)
    if len(old_profile) >= 2:
        brightness = np.interp(
            np.linspace(0.0, 1.0, len(points)),
            np.linspace(0.0, 1.0, len(old_profile)), old_profile)
    else:
        brightness = np.full(len(points), np.nan)
    inferred = MaterialCenterline(
        centerline=Centerline(
            points, material.roi, material.mask,
            material.centerline.radius_px),
        distal_boundary_index=boundary,
        distal_boundary_fraction=float(boundary / max(len(points) - 1, 1)),
        boundary_confidence=material.boundary_confidence,
        boundary_contrast=material.boundary_contrast,
        material_valid=material.material_valid,
        brightness_profile=brightness)
    return replace(result, material=inferred)


def _distal_medial_ridge_evidence(
        image: np.ndarray,
        result: SamMaterialResult,
        max_points: int = 1200) -> np.ndarray:
    """Return unordered distal-color medial support in full-image pixels.

    The final topology optimizer must see the alternative arm of a tight V;
    feeding it the already ordered cyan curve would merely preserve that
    curve's branch decision.  These samples therefore come directly from the
    independently formed chromatic mask.  Dark proximal blue is excluded by
    channel direction, while cyan, red rings, and the small yellow end remain.
    """
    material = result.material
    roi_x, roi_y, width, height = material.roi
    crop = np.asarray(image[roi_y:roi_y + height, roi_x:roi_x + width])
    mask = np.asarray(material.mask) > 0
    if crop.shape[:2] != mask.shape:
        raise ValueError("ridge evidence image/mask shape mismatch")
    b, g, r = [crop[:, :, channel].astype(np.int16) for channel in range(3)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    chromatic = (hsv[:, :, 1] >= 38) & (hsv[:, :, 2] >= 25)
    cyan = (b - r >= 24) & (g - r >= 18) & (g >= 42)
    red_ring = (r - g >= 20) & (r >= 48)
    yellow = (r - b >= 20) & (g - b >= 12) & (g >= 45)
    distal_support = mask & chromatic & (cyan | red_ring | yellow)
    # Centering must be defined by the complete silhouette.  Using the
    # thresholded paint subset makes the ridge follow whichever side is more
    # illuminated, which produced the persistent upper-edge bias.  Color is
    # retained only to remove clearly proximal ridge samples below.
    distance = cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 5)
    local_maximum = cv2.dilate(
        distance, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ridge = mask & (distance >= 0.9) & (
        distance >= local_maximum - 0.20)
    if np.count_nonzero(distal_support) >= 12:
        color_distance = cv2.distanceTransform(
            np.uint8(~distal_support), cv2.DIST_L2, 5)
        ridge &= color_distance <= 5.0
    rows, columns = np.where(ridge)
    if len(columns) < 12:
        # Retain operation under glare/color dropout.  This fallback is still
        # mask-derived and is used only as a point-to-support term, so proximal
        # samples cannot impose ordering on the distal spline.
        distance = cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 5)
        local_maximum = cv2.dilate(
            distance, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        rows, columns = np.where(
            mask & (distance >= 0.9) & (distance >= local_maximum - 0.20))
    if not len(columns):
        raise ValueError("distal medial ridge evidence is empty")
    if len(columns) > int(max_points):
        selected = np.linspace(
            0, len(columns) - 1, int(max_points)).round().astype(int)
        rows, columns = rows[selected], columns[selected]
    return np.column_stack([
        columns + roi_x, rows + roi_y]).astype(np.float64)


def _resample_material_corridor(
        points_xy: np.ndarray, sample_count: int,
        material_s: np.ndarray | None = None) -> np.ndarray:
    """Resample stereo observations at uniform reconstructed arclength."""
    points = np.asarray(points_xy, dtype=np.float64)
    if material_s is None:
        source = np.linspace(0.0, 1.0, len(points))
    else:
        source = np.asarray(material_s, dtype=np.float64)
        if len(source) != len(points) or not np.all(np.isfinite(source)):
            raise ValueError("corridor material coordinate is invalid")
        source = source - source[0]
        if source[-1] <= 1e-9 or np.any(np.diff(source) < 0.0):
            raise ValueError("corridor material coordinate is not monotone")
        source /= source[-1]
        # Consecutive triangulated samples can coincide. np.interp requires a
        # strictly increasing abscissa for a well-defined material mapping.
        keep = np.r_[True, np.diff(source) > 1e-9]
        source = source[keep]
        points = points[keep]
    target = np.linspace(0.0, 1.0, int(sample_count))
    corridor = np.column_stack([
        np.interp(target, source, points[:, coordinate])
        for coordinate in range(2)])
    corridor = gaussian_filter1d(
        corridor, sigma=0.7, axis=0, mode="nearest")
    corridor[0], corridor[-1] = points[0], points[-1]
    return corridor


def _center_ordered_mask_corridor(
        points_xy: np.ndarray,
        result: SamMaterialResult,
        sample_count: int,
        material_s: np.ndarray | None = None,
        maximum_shift_px: float = 9.0) -> np.ndarray:
    """Center one ordered topology inside its contiguous mask branch."""
    corridor = _resample_material_corridor(
        points_xy, sample_count, material_s)
    tangent = np.gradient(corridor, axis=0)
    tangent /= np.maximum(
        np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    mask = np.asarray(result.material.mask) > 0
    distance = cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 5)
    roi_x, roi_y, width, height = result.material.roi
    offsets = np.arange(-12.0, 12.01, 0.5)
    centered = corridor.copy()
    for index, (point, direction) in enumerate(zip(corridor, normal)):
        samples = point + offsets[:, None] * direction
        pixel = np.rint(samples - [roi_x, roi_y]).astype(int)
        inside = (
            (pixel[:, 0] >= 0) & (pixel[:, 0] < width)
            & (pixel[:, 1] >= 0) & (pixel[:, 1] < height))
        supported = np.zeros(len(offsets), dtype=bool)
        supported[inside] = mask[pixel[inside, 1], pixel[inside, 0]]
        supported_indices = np.flatnonzero(supported)
        if not len(supported_indices):
            continue
        # Split the normal slice into mask runs and retain the run nearest the
        # incoming ordered topology. This projects small off-mask errors back
        # onto their chosen arm without switching across a V-shaped gap.
        breaks = np.flatnonzero(np.diff(supported_indices) > 1) + 1
        runs = np.split(supported_indices, breaks)
        run = min(runs, key=lambda values: float(np.min(
            np.abs(offsets[values]))))
        if float(np.min(np.abs(offsets[run]))) > maximum_shift_px:
            continue
        lower, upper = int(run[0]), int(run[-1])
        run_width = float(offsets[upper] - offsets[lower])
        # A normal slice wider than the catheter is probably a merged V lobe.
        if run_width > 20.0:
            continue
        run_pixels = pixel[run]
        run_distance = distance[run_pixels[:, 1], run_pixels[:, 0]]
        maximum_distance = float(np.max(run_distance))
        medial = run[np.flatnonzero(
            run_distance >= maximum_distance - 0.15)]
        midpoint = 0.5 * (offsets[lower] + offsets[upper])
        selected = int(medial[np.argmin(np.abs(offsets[medial] - midpoint))])
        shift = float(np.clip(
            offsets[selected], -maximum_shift_px, maximum_shift_px))
        centered[index] = point + shift * direction
    return centered


def _center_paired_mask_corridors(
        left_points_xy: np.ndarray,
        right_points_xy: np.ndarray,
        left_result: SamMaterialResult,
        right_result: SamMaterialResult,
        sample_count: int,
        material_s: np.ndarray,
        maximum_shift_px: float = 9.0,
        epipolar_tolerance_px: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Move paired routes to mask medial support without breaking epipolarity."""
    left = _resample_material_corridor(
        left_points_xy, sample_count, material_s)
    right = _resample_material_corridor(
        right_points_xy, sample_count, material_s)
    offsets = np.arange(-12.0, 12.01, 0.5)

    def geometry(points, result):
        tangent = np.gradient(points, axis=0)
        tangent /= np.maximum(
            np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        mask = np.asarray(result.material.mask) > 0
        distance = cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 5)
        roi = np.asarray(result.material.roi[:2], dtype=float)
        height, width = mask.shape
        return normal, mask, distance, roi, width, height

    left_geometry = geometry(left, left_result)
    right_geometry = geometry(right, right_result)

    def candidates(point, direction, values):
        _, mask, distance, roi, width, height = values
        samples = point + offsets[:, None] * direction
        pixel = np.rint(samples - roi).astype(int)
        inside = (
            (pixel[:, 0] >= 0) & (pixel[:, 0] < width)
            & (pixel[:, 1] >= 0) & (pixel[:, 1] < height))
        supported = np.zeros(len(offsets), dtype=bool)
        supported[inside] = mask[pixel[inside, 1], pixel[inside, 0]]
        indices = np.flatnonzero(supported)
        if not len(indices):
            return None
        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        runs = np.split(indices, breaks)
        run = min(runs, key=lambda values: float(np.min(
            np.abs(offsets[values]))))
        if (float(np.min(np.abs(offsets[run]))) > maximum_shift_px
                or offsets[run[-1]] - offsets[run[0]] > 20.0):
            return None
        pixels = pixel[run]
        return (
            samples[run], offsets[run],
            distance[pixels[:, 1], pixels[:, 0]])

    centered_left = left.copy()
    centered_right = right.copy()
    for index in range(len(left)):
        left_candidates = candidates(
            left[index], left_geometry[0][index], left_geometry)
        right_candidates = candidates(
            right[index], right_geometry[0][index], right_geometry)
        if left_candidates is None or right_candidates is None:
            continue
        left_samples, left_offsets, left_distance = left_candidates
        right_samples, right_offsets, right_distance = right_candidates
        row_error = np.abs(
            left_samples[:, None, 1] - right_samples[None, :, 1])
        feasible = row_error <= float(epipolar_tolerance_px)
        if not np.any(feasible):
            continue
        # Prefer thick medial support, with small displacement and residual
        # row mismatch only as tie-breakers. The feasibility constraint keeps
        # every selected pair triangulatable by the calibrated stereo model.
        cost = (
            -(left_distance[:, None] + right_distance[None, :])
            + 0.02 * (
                np.square(left_offsets[:, None])
                + np.square(right_offsets[None, :]))
            + 0.5 * np.square(row_error))
        cost[~feasible] = np.inf
        left_index, right_index = np.unravel_index(
            int(np.argmin(cost)), cost.shape)
        centered_left[index] = left_samples[left_index]
        centered_right[index] = right_samples[right_index]
    return centered_left, centered_right


def _stored_material_result(output, index: int, view: str):
    """Expose persisted mask geometry to the corridor centering helper."""
    dataset = output[f"images/{view}/mask_packbits"]
    shape = tuple(int(value) for value in dataset.attrs["unpacked_shape"])
    mask = np.unpackbits(
        np.asarray(dataset[index]), bitorder="little",
        count=shape[0] * shape[1]).reshape(shape).astype(bool)
    roi = tuple(int(value) for value in dataset.attrs["roi_xywh"])
    return SimpleNamespace(material=SimpleNamespace(mask=mask, roi=roi))


def _projected_turn_fraction(points_xy: np.ndarray) -> tuple[float | None, float]:
    """Locate the sole strong projected fold without constructing routes."""
    points = resample_arclength(points_xy, 96, smooth_window=1)
    points = gaussian_filter1d(points, sigma=1.2, axis=0, mode="nearest")
    span = 4
    before = points[span:-span] - points[:-2 * span]
    after = points[2 * span:] - points[span:-span]
    denominator = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    cosine = np.divide(
        np.sum(before * after, axis=1), denominator,
        out=np.ones_like(denominator), where=denominator > 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    if not len(angle):
        return None, 0.0
    fractions = (np.arange(len(angle)) + span) / (len(points) - 1)
    interior = (fractions >= 0.12) & (fractions <= 0.88)
    if not np.any(interior):
        return None, 0.0
    candidates = np.flatnonzero(interior)
    index = int(candidates[np.argmax(angle[candidates])])
    maximum = float(angle[index])
    if maximum < 28.0:
        return None, maximum
    return float((index + span) / (len(points) - 1)), maximum


def _epipolar_sweep_extremum_fraction(
        points_xy: np.ndarray) -> float | None:
    """Return the material coordinate of the farthest epipolar image row.

    ZED images are rectified, so corresponding epipolar lines have equal
    image y.  In an end-on view the distal route can reverse at the material
    sample whose row is farthest from the interface.  The well-conditioned
    eye identifies that sample without needing to exhibit a visible cusp.
    An endpoint extremum means the sweep is monotone and needs no interior
    turn window.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return None
    source = np.linspace(0.0, 1.0, len(points))
    target = np.linspace(0.0, 1.0, 96)
    points = np.column_stack([
        np.interp(target, source, points[:, coordinate])
        for coordinate in range(2)])
    displacement = np.abs(points[:, 1] - points[0, 1])
    index = int(np.argmax(displacement))
    fraction = float(index / max(len(points) - 1, 1))
    if fraction < 0.08 or fraction > 0.92:
        return None
    return fraction


def _replace_distal_with_joint_projection(
        result: SamMaterialResult,
        projected_distal_xy: np.ndarray) -> SamMaterialResult:
    """Make the final joint 3D topology the displayed/stored distal cyan path."""
    material = result.material
    projected = np.asarray(projected_distal_xy, dtype=np.float64)
    projected = projected[np.all(np.isfinite(projected), axis=1)]
    if len(projected) < 2:
        raise ValueError("joint distal projection is empty")
    boundary = int(np.clip(
        material.distal_boundary_index, 0, len(material.points) - 1))
    proximal = np.asarray(material.points[:boundary], dtype=np.float64)
    points = np.vstack([proximal, projected]) if len(proximal) else projected
    new_boundary = len(proximal)
    old_profile = np.asarray(material.brightness_profile, dtype=np.float64)
    if len(old_profile) >= 2:
        brightness = np.interp(
            np.linspace(0.0, 1.0, len(points)),
            np.linspace(0.0, 1.0, len(old_profile)), old_profile)
    else:
        brightness = np.full(len(points), np.nan)
    unified = MaterialCenterline(
        centerline=Centerline(
            points, material.roi, material.mask,
            material.centerline.radius_px),
        distal_boundary_index=new_boundary,
        distal_boundary_fraction=float(
            new_boundary / max(len(points) - 1, 1)),
        boundary_confidence=material.boundary_confidence,
        boundary_contrast=material.boundary_contrast,
        material_valid=material.material_valid,
        brightness_profile=brightness)
    return replace(
        result, material=unified,
        prompt=replace(
            result.prompt,
            source=f"{result.prompt.source}+unified_stereo_topology"))


def _inferred_material_path_quality(
        observed_result: SamMaterialResult,
        inferred_points_xy: np.ndarray,
        previous_observed_points_xy: np.ndarray | None,
        max_temporal_p95_px: float,
        max_temporal_max_px: float,
        max_sharp_turn_clusters: int,
) -> tuple[SamMaterialResult, bool, dict[str, float | int]]:
    """Validate a stereo-inferred cyan path before it replaces an observation.

    Marker overlap may make indices equal, but their material order may never
    reverse. Temporal motion is measured after removing the median image
    translation, so normal rigid motion does not look like a topology change.
    """
    candidate = _replace_material_path(observed_result, inferred_points_xy)
    topology = _marked_centerline_topology_metrics(candidate)
    rejection_code = 0
    # 1 marker reversal, 2 terminal mismatch, 4 excess turns,
    # 8 temporal p95, 16 temporal maximum, 32 mask support, 64 non-medial.
    markers = observed_result.marker_centers_xy
    observed = observed_result.marker_observed
    confidence = observed_result.marker_confidence
    points = np.asarray(candidate.material.points, dtype=np.float64)
    marker_order_valid = True
    terminal_error = float("nan")
    if markers is not None and observed is not None:
        marker_points = np.asarray(markers, dtype=np.float64)
        usable = np.asarray(observed, dtype=bool)
        if confidence is not None:
            usable &= np.asarray(confidence, dtype=np.float64) >= 0.25
        usable &= np.all(np.isfinite(marker_points), axis=1)
        ids = np.flatnonzero(usable)
        indices = np.asarray([
            int(np.argmin(np.linalg.norm(points - marker_points[index], axis=1)))
            for index in ids], dtype=int)
        # Up to two resampled indices of uncertainty permits coincident rings
        # in the ill projection without permitting an interval reversal.
        marker_order_valid = bool(
            len(indices) < 2 or np.all(np.diff(indices) >= -2))
        if not marker_order_valid:
            rejection_code |= 1
        if 3 in ids:
            terminal_error = float(np.linalg.norm(
                points[-1] - marker_points[3]))
            widths = observed_result.marker_widths_px
            width = (
                float(np.asarray(widths)[3])
                if widths is not None and np.isfinite(np.asarray(widths)[3])
                else 12.0)
            if terminal_error > max(12.0, 0.85 * width):
                rejection_code |= 2
    sharp_turn_clusters = int(round(
        topology["centerline_sharp_turn_cluster_count"]))
    if sharp_turn_clusters > min(max(int(max_sharp_turn_clusters), 0), 1):
        rejection_code |= 4

    temporal_p95 = float("nan")
    temporal_max = float("nan")
    if previous_observed_points_xy is not None:
        previous = resample_arclength(
            np.asarray(previous_observed_points_xy, dtype=np.float64),
            len(points), smooth_window=1)
        displacement = points - previous
        displacement -= np.median(displacement, axis=0, keepdims=True)
        magnitude = np.linalg.norm(displacement, axis=1)
        temporal_p95 = float(np.percentile(magnitude, 95))
        temporal_max = float(np.max(magnitude))
        if temporal_p95 > float(max_temporal_p95_px):
            rejection_code |= 8
        if temporal_max > float(max_temporal_max_px):
            rejection_code |= 16
    outside = _centerline_outside_mask_fraction(candidate)
    if outside > 0.02:
        rejection_code |= 32
    mask = np.asarray(candidate.material.mask > 0, dtype=np.uint8)
    medial = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    roi_x, roi_y, width, height = candidate.material.roi
    candidate_local = np.rint(points - [roi_x, roi_y]).astype(int)
    candidate_local[:, 0] = np.clip(candidate_local[:, 0], 0, width - 1)
    candidate_local[:, 1] = np.clip(candidate_local[:, 1], 0, height - 1)
    candidate_radius = float(np.median(medial[
        candidate_local[:, 1], candidate_local[:, 0]]))
    observed_points = np.asarray(
        observed_result.material.points, dtype=np.float64)
    observed_local = np.rint(observed_points - [roi_x, roi_y]).astype(int)
    observed_local[:, 0] = np.clip(observed_local[:, 0], 0, width - 1)
    observed_local[:, 1] = np.clip(observed_local[:, 1], 0, height - 1)
    observed_radius = float(np.median(medial[
        observed_local[:, 1], observed_local[:, 0]]))
    medial_ratio = candidate_radius / max(observed_radius, 1.0)
    if candidate_radius < 1.5 or medial_ratio < 0.55:
        rejection_code |= 64
    accepted = rejection_code == 0
    if accepted:
        candidate = replace(
            candidate,
            prompt=replace(
                candidate.prompt,
                source=candidate.prompt.source
                + "+validated_stereo_topology"))
    return candidate, accepted, {
        "stereo_inferred_rejection_code": rejection_code,
        "stereo_inferred_marker_order_valid": int(marker_order_valid),
        "stereo_inferred_terminal_error_px": terminal_error,
        "stereo_inferred_sharp_turn_clusters": sharp_turn_clusters,
        "stereo_inferred_temporal_p95_px": temporal_p95,
        "stereo_inferred_temporal_max_px": temporal_max,
        "stereo_inferred_outside_mask_fraction": outside,
        "stereo_inferred_medial_radius_ratio": medial_ratio,
    }


def _fitted_curve_reprojection_metrics(
        registration,
        points_base_mm: np.ndarray,
        left_centerline: np.ndarray,
        right_centerline: np.ndarray) -> dict[str, float]:
    """Measure the final fitted curve, not only its pre-fit reconstruction."""
    distances = []
    means = []
    for right, centerline in (
            (False, left_centerline), (True, right_centerline)):
        projected = _project_base_points(
            registration, points_base_mm, right=right)
        observed = np.asarray(centerline, dtype=np.float64)
        observed = observed[np.all(np.isfinite(observed), axis=1)]
        if len(observed) < 2:
            raise ValueError("fitted_reprojection_centerline_missing")
        distance = np.min(np.linalg.norm(
            projected[:, None, :] - observed[None, :, :], axis=2), axis=1)
        distances.append(distance)
        means.append(float(np.mean(distance)))
    combined = np.concatenate(distances)
    return {
        "fitted_reprojection_left_px": means[0],
        "fitted_reprojection_right_px": means[1],
        "fitted_reprojection_max_px": float(np.max(combined)),
        "fitted_reprojection_p95_px": float(np.percentile(combined, 95)),
    }


def _stereo_tip_camera_point(
        left_tip_xy: np.ndarray | None,
        right_tip_xy: np.ndarray | None,
        camera_matrix: np.ndarray,
        baseline_m: float,
        max_epipolar_error_px: float = 5.0) -> tuple[np.ndarray | None, float]:
    """Triangulate a rectified yellow-tip observation from both views."""
    if left_tip_xy is None or right_tip_xy is None:
        return None, float("nan")
    left = np.asarray(left_tip_xy, dtype=np.float64)
    right = np.asarray(right_tip_xy, dtype=np.float64)
    if left.shape != (2,) or right.shape != (2,) or not (
            np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return None, float("nan")
    epipolar_error = float(abs(left[1] - right[1]))
    disparity = float(left[0] - right[0])
    if epipolar_error > float(max_epipolar_error_px) or disparity <= 0.5:
        return None, epipolar_error
    K = np.asarray(camera_matrix, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth = float(fx * float(baseline_m) / disparity)
    if not np.isfinite(depth) or not 0.05 <= depth <= 2.0:
        return None, epipolar_error
    y = 0.5 * (left[1] + right[1])
    point = np.array([
        (left[0] - cx) * depth / fx,
        (y - cy) * depth / fy,
        depth,
    ], dtype=np.float64)
    return point, epipolar_error


def _usable_marker_observations(
        result: SamMaterialResult,
        minimum_confidence: float) -> tuple[np.ndarray, np.ndarray]:
    centers = np.full((4, 2), np.nan, dtype=np.float64)
    confidence = np.zeros(4, dtype=np.float64)
    if result.marker_centers_xy is None:
        return centers, confidence
    source_centers = np.asarray(result.marker_centers_xy, dtype=np.float64)
    source_confidence = (
        np.ones(len(source_centers), dtype=np.float64)
        if result.marker_confidence is None else
        np.asarray(result.marker_confidence, dtype=np.float64))
    source_observed = (
        np.ones(len(source_centers), dtype=bool)
        if result.marker_observed is None else
        np.asarray(result.marker_observed, dtype=bool))
    count = min(4, len(source_centers), len(source_confidence), len(source_observed))
    valid = (
        source_observed[:count]
        & np.all(np.isfinite(source_centers[:count]), axis=1)
        & (source_confidence[:count] >= float(minimum_confidence)))
    centers[:count][valid] = source_centers[:count][valid]
    confidence[:count][valid] = source_confidence[:count][valid]
    return centers, confidence


def _stereo_marker_camera_points(
        left_markers: np.ndarray,
        right_markers: np.ndarray,
        K: np.ndarray,
        baseline_m: float,
        max_epipolar_error_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.full((4, 3), np.nan, dtype=np.float64)
    observed = np.zeros(4, dtype=np.uint8)
    errors = np.full(4, np.nan, dtype=np.float64)
    for marker_id in range(4):
        point, error = _stereo_tip_camera_point(
            left_markers[marker_id], right_markers[marker_id], K, baseline_m,
            max_epipolar_error_px=max_epipolar_error_px)
        errors[marker_id] = error
        if point is not None:
            points[marker_id] = point
            observed[marker_id] = 1
    return points, observed, errors


def _stereo_candidate_score(
        reconstruction: dict,
        reference_view: str,
        left_length_px: float,
        right_length_px: float,
        previous_points_camera_m: np.ndarray | None = None,
        previous_reference_view: str | None = None) -> tuple[float, dict[str, float]]:
    """Score one ordered stereo hypothesis using both views and time."""
    worst_mean = max(
        float(reconstruction["reprojection_left_px"]),
        float(reconstruction["reprojection_right_px"]))
    reprojection = worst_mean + float(reconstruction["reprojection_p95_px"])
    terminal_reprojection = max(
        float(reconstruction.get("terminal_reprojection_left_px", 0.0)),
        float(reconstruction.get("terminal_reprojection_right_px", 0.0)))
    reference_length = (
        left_length_px if reference_view == "left" else right_length_px)
    best_length = max(left_length_px, right_length_px)
    foreshortening = max(0.0, best_length / max(reference_length, 1e-9) - 1.0)
    temporal_rms_mm = 0.0
    if previous_points_camera_m is not None:
        previous = np.asarray(previous_points_camera_m, dtype=np.float64)
        current = np.asarray(reconstruction["points_camera_m"], dtype=np.float64)
        if len(previous) != len(current):
            parameter = np.linspace(0.0, 1.0, len(previous))
            target = np.linspace(0.0, 1.0, len(current))
            previous = np.column_stack([
                np.interp(target, parameter, previous[:, axis])
                for axis in range(3)])
        temporal_rms_mm = float(np.sqrt(np.mean(np.sum(
            (current - previous) ** 2, axis=1))) * 1000.0)
    switch_penalty = float(
        previous_reference_view is not None
        and reference_view != previous_reference_view)
    score = (
        reprojection
        + 1.5 * terminal_reprojection
        + 8.0 * foreshortening
        + 0.20 * min(temporal_rms_mm, 50.0)
        + 0.25 * switch_penalty)
    return score, {
        "reprojection_score": reprojection,
        "terminal_reprojection_score": terminal_reprojection,
        "foreshortening_penalty": foreshortening,
        "temporal_rms_mm": temporal_rms_mm,
        "switch_penalty": switch_penalty,
    }


def _select_stereo_reference(
        candidates: dict[str, tuple[float, dict]],
        previous_reference_view: str | None,
        hysteresis_score: float,
        ill_view: str | None = None) -> str:
    """Choose a non-overlapped reference and suppress score chatter."""
    if not candidates:
        raise ValueError("stereo_both_reference_candidates_failed")
    # Once one projection is observably shortened, it is not a valid source of
    # material order.  It remains an image observation for the subsequent joint
    # fit, but must not win merely because a shortcut has a low reprojection
    # cost or because reference-view hysteresis keeps the previous choice.
    observable = {
        view: value for view, value in candidates.items()
        if view != ill_view}
    if observable:
        candidates = observable
    eligible = {
        view: value for view, value in candidates.items()
        if (value[1]["reference_eye_self_overlap_fraction"]
            <= value[1]["other_eye_self_overlap_fraction"] + 0.02)}
    if not eligible:
        eligible = candidates
    selected = min(eligible, key=lambda view: eligible[view][0])
    if (previous_reference_view in eligible
            and eligible[previous_reference_view][0]
            <= eligible[selected][0] + float(hysteresis_score)):
        selected = previous_reference_view
    return selected


def _fuse_distal_length(
        nominal_length_mm: float,
        observed_lengths_mm: list[float],
        observed_confidences: list[float],
        previous_length_mm: float | None,
        prior_sigma_mm: float,
        color_sigma_mm: float,
        temporal_sigma_mm: float,
        robust_delta_mm: float = 8.0) -> tuple[float, float, float]:
    """Fuse color, nominal length, and temporal evidence without equality."""
    nominal = float(nominal_length_mm)
    prior_weight = 1.0 / max(float(prior_sigma_mm), 1e-6) ** 2
    values = [nominal]
    weights = [prior_weight]
    for value, confidence in zip(observed_lengths_mm, observed_confidences):
        if not np.isfinite(value):
            continue
        residual = abs(float(value) - nominal)
        robust = min(1.0, float(robust_delta_mm) / max(residual, 1e-9))
        weight = (
            max(float(confidence), 0.05) * robust
            / max(float(color_sigma_mm), 1e-6) ** 2)
        values.append(float(value))
        weights.append(weight)
    raw = float(np.average(values, weights=weights))
    filtered_values = list(values)
    filtered_weights = list(weights)
    if previous_length_mm is not None and np.isfinite(previous_length_mm):
        filtered_values.append(float(previous_length_mm))
        filtered_weights.append(
            1.0 / max(float(temporal_sigma_mm), 1e-6) ** 2)
    filtered = float(np.average(filtered_values, weights=filtered_weights))
    uncertainty = float(np.sqrt(1.0 / np.sum(filtered_weights)))
    return raw, filtered, uncertainty


def _project_point_to_polyline_s(
        points: np.ndarray,
        point: np.ndarray,
        center_s_mm: float,
        half_window_mm: float) -> float:
    """Project a prior 3D interface onto a local part of the current curve."""
    curve = np.asarray(points, dtype=np.float64)
    target = np.asarray(point, dtype=np.float64)
    s = cumulative_arclength(curve)
    best_distance = float("inf")
    best_s = float(np.clip(center_s_mm, 0.0, s[-1]))
    lo = float(center_s_mm) - max(float(half_window_mm), 0.0)
    hi = float(center_s_mm) + max(float(half_window_mm), 0.0)
    for index, (start, end) in enumerate(zip(curve[:-1], curve[1:])):
        if s[index + 1] < lo or s[index] > hi:
            continue
        segment = end - start
        denominator = float(np.dot(segment, segment))
        alpha = 0.0 if denominator <= 1e-12 else float(np.clip(
            np.dot(target - start, segment) / denominator, 0.0, 1.0))
        candidate = start + alpha * segment
        distance = float(np.linalg.norm(candidate - target))
        if distance < best_distance:
            best_distance = distance
            best_s = float(s[index] + alpha * (s[index + 1] - s[index]))
    return best_s


def _point_at_arclength(points: np.ndarray, target_s_mm: float) -> np.ndarray:
    curve = np.asarray(points, dtype=np.float64)
    s = cumulative_arclength(curve)
    target = float(np.clip(target_s_mm, 0.0, s[-1]))
    return np.array([
        np.interp(target, s, curve[:, axis]) for axis in range(3)],
        dtype=np.float64)


def _temporal_centerline_metrics(
        previous_points: np.ndarray,
        current_points: np.ndarray,
        tolerance_px: float) -> tuple[float, float]:
    previous = np.asarray(previous_points, dtype=np.float64)
    current = np.asarray(current_points, dtype=np.float64)
    previous_to_current = np.min(np.linalg.norm(
        previous[:, None, :] - current[None, :, :], axis=2), axis=1)
    current_to_previous = np.min(np.linalg.norm(
        current[:, None, :] - previous[None, :, :], axis=2), axis=1)
    return (
        float(min(
            np.mean(previous_to_current <= float(tolerance_px)),
            np.mean(current_to_previous <= float(tolerance_px)))),
        float(max(
            np.percentile(previous_to_current, 95),
            np.percentile(current_to_previous, 95))))


def _ill_eye_observation_weights(
        active_ill_view: str | None,
        shorter_view: str,
        left_length_px: float,
        right_length_px: float,
        minimum_weight: float) -> tuple[float, float, str | None]:
    """Downweight only when the hysteretic label matches current evidence."""
    if active_ill_view not in ("left", "right"):
        return 1.0, 1.0, None
    if active_ill_view != shorter_view:
        # During a pending switch/release, suppress neither eye. Applying the
        # current short/long ratio to yesterday's label downweights the good
        # eye and was the direct cause of branch alternation.
        return 1.0, 1.0, None
    short_length = min(float(left_length_px), float(right_length_px))
    long_length = max(float(left_length_px), float(right_length_px))
    weight = float(np.clip(max(
        float(minimum_weight),
        (short_length / max(long_length, 1e-9)) ** 4), 0.0, 1.0))
    return (
        (weight, 1.0, "left") if active_ill_view == "left"
        else (1.0, weight, "right"))


def _update_ill_view_hysteresis(
        active: str | None,
        pending: str | None,
        pending_count: int,
        observed: str | None,
        confirmation_frames: int,
) -> tuple[str | None, str | None, int]:
    """Confirm entry, release, and eye switches with the same persistence.

    A single ambiguity-threshold crossing must not replace an independently
    well-conditioned image route. The previous implementation confirmed only
    release/switches but entered recovery immediately, which made marginal
    ambiguity noise toggle the reconstruction objective.
    """
    required = max(1, int(confirmation_frames))
    if observed == active:
        return active, None, 0
    if observed == pending:
        count = int(pending_count) + 1
    else:
        pending = observed
        count = 1
    if count >= required:
        return observed, None, 0
    return active, pending, count


def _joint_fit_supports_bijective_stereo(result) -> bool:
    """Accept an image-supported ordinary fit before topology recovery.

    The active ill-eye label has release hysteresis and can persist for several
    frames after both cyan observations become bijective again. At that point
    low two-view model and coverage errors are direct evidence that ordinary
    stereo has recovered. A projected-turn counter is diagnostic here; using
    it as an additional hard gate caused accurate post-ill fits to disappear.
    """
    values = np.asarray([
        result.left_model_mean_px, result.right_model_mean_px,
        result.left_model_p95_px, result.right_model_p95_px,
        result.left_coverage_mean_px, result.right_coverage_mean_px,
    ], dtype=float)
    return bool(
        result.optimizer_success
        and np.all(np.isfinite(values))
        and max(values[:2]) <= 3.0
        and max(values[2:4]) <= 8.0
        and max(values[4:]) <= 5.0)


class ImageSequenceWriter:
    """Fixed-shape output written as bounded contiguous background batches."""

    _END = object()

    def __init__(
            self,
            path: Path,
            frame_count: int,
            config: ImageProcessingConfig,
            registration,
            metadata: dict):
        import h5py

        self.image_centerline_samples = int(config.image_centerline_samples)
        self.store_masks = bool(config.store_masks)
        self.hdf_buffer_frames = max(1, int(config.hdf_buffer_frames))
        self.file = h5py.File(path, "w")
        self.string_type = h5py.string_dtype(encoding="utf-8")
        self.file.attrs["schema_version"] = 15
        self.file.attrs["mode"] = "image_only_shape_tracking"
        self.file.attrs["coordinate_frame"] = "robot_base"
        self.file.attrs["position_units"] = "mm"
        self.file.attrs["curvature_units"] = "1/mm"
        self.file.attrs["observation_class_json"] = json.dumps({
            0: "invalid_or_unset",
            1: "image_observed_proximal",
            2: "image_observed_distal",
            3: "reserved_modelled_base_bridge_not_used_in_image_only_mode",
        }, sort_keys=True)
        self.file.attrs["curvature_validity"] = (
            "NaN at open-curve derivative edges; image-only mode does not "
            "insert a modelled base bridge")
        self.file.attrs["robot_joint_order_json"] = json.dumps([
            "catheter_lin", "catheter_rot", "catheter_bend",
            "sheath_lin", "sheath_rot", "sheath_bend"])
        self.file.attrs["robot_joint_units_json"] = json.dumps([
            "mm", "deg", "mm", "mm", "deg", "deg"])
        self.file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        self.file.attrs["processing_config_json"] = json.dumps(
            asdict(config), sort_keys=True)
        self._create(frame_count, config, registration)
        self.dataset_paths: set[str] = set()
        self.file.visititems(
            lambda name, item: self.dataset_paths.add(name)
            if hasattr(item, "shape") else None)
        self._pending: dict[int, dict[str, object]] = {}
        self._staging: list[tuple[int, dict[str, object]]] = []
        self._write_queue: queue.Queue = queue.Queue(
            maxsize=max(1, int(config.hdf_queue_chunks)))
        self._writer_error: BaseException | None = None
        self.write_time_s = 0.0
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="image-hdf5-writer", daemon=True)
        self._writer_thread.start()

    def _dataset(self, name, shape, dtype=np.float32, fillvalue=None, **extra):
        dtype = np.dtype(dtype)
        row_elements = int(np.prod(shape[1:], dtype=np.int64))
        row_bytes = max(1, row_elements * dtype.itemsize)
        chunk_rows = max(1, min(
            int(shape[0]), self.hdf_buffer_frames,
            max(1, (1024 * 1024) // row_bytes)))
        kwargs = dict(
            shape=shape, dtype=dtype, compression="gzip",
            compression_opts=1, shuffle=True,
            chunks=(chunk_rows, *shape[1:]))
        if fillvalue is not None:
            kwargs["fillvalue"] = fillvalue
        kwargs.update(extra)
        return self.file.create_dataset(name, **kwargs)

    def _create(self, count: int, config: ImageProcessingConfig, registration) -> None:
        calibration = self.file.create_group("calibration")
        calibration["camera_matrix"] = np.asarray(registration.K, np.float64)
        calibration["distortion"] = np.asarray(
            getattr(registration, "distortion", np.empty(0)), np.float64)
        calibration["left_camera_T_base"] = np.asarray(
            registration.left_camera_T_base, np.float64)
        calibration["right_camera_T_base"] = np.asarray(
            registration.right_camera_T_base, np.float64)
        calibration["roi_left_xywh"] = np.asarray(
            registration.roi_left_xywh, np.int32)
        calibration["roi_right_xywh"] = np.asarray(
            registration.roi_right_xywh, np.int32)
        calibration.attrs["baseline_m"] = float(registration.baseline_m)
        calibration.attrs["zed_serial"] = str(
            getattr(registration, "zed_serial", "unknown"))
        calibration.attrs["resolution"] = str(
            getattr(registration, "resolution", "unknown"))
        frames = self.file.create_group("frames")
        frames.create_dataset("svo_frame", shape=(count,), dtype=np.int32)
        frames.create_dataset("timestamp_ns", shape=(count,), dtype=np.int64)
        frames.create_dataset("svo_timestamp_ns", shape=(count,), dtype=np.int64)
        frames.create_dataset("valid", shape=(count,), dtype=np.uint8)
        frames.create_dataset(
            "observation_valid", shape=(count,), dtype=np.uint8)
        frames.create_dataset("status", shape=(count,), dtype=self.string_type)

        for view, roi in (
                ("left", registration.roi_left_xywh),
                ("right", registration.roi_right_xywh)):
            group = self.file.create_group(f"images/{view}")
            _, _, width, height = roi
            packed_size = (width * height + 7) // 8
            if config.store_masks:
                mask = self._dataset(
                    f"images/{view}/mask_packbits", (count, packed_size),
                    dtype=np.uint8, fillvalue=0)
                mask.attrs["roi_xywh"] = roi
                mask.attrs["unpacked_shape"] = (height, width)
                mask.attrs["packbits_bitorder"] = "little"
            self._dataset(
                f"images/{view}/centerline_px",
                (count, config.image_centerline_samples, 2), fillvalue=np.nan)
            self._dataset(
                f"images/{view}/observed_centerline_px",
                (count, config.image_centerline_samples, 2), fillvalue=np.nan)
            self._dataset(
                f"images/{view}/distal_boundary_px", (count, 2), fillvalue=np.nan)
            self._dataset(f"images/{view}/sam_iou", (count,), fillvalue=np.nan)
            self._dataset(f"images/{view}/selection_score", (count,), fillvalue=np.nan)
            self._dataset(f"images/{view}/seed_recall", (count,), fillvalue=np.nan)
            self._dataset(f"images/{view}/mask_area_px", (count,), dtype=np.int32)
            self._dataset(
                f"images/{view}/prompt_box_xyxy", (count, 4), fillvalue=np.nan)
            self._dataset(
                f"images/{view}/prompt_positive_px", (count, 16, 2), fillvalue=np.nan)
            self._dataset(
                f"images/{view}/prompt_negative_px", (count, 16, 2), fillvalue=np.nan)
            self._dataset(
                f"images/{view}/prompt_positive_count", (count,), dtype=np.uint8)
            self._dataset(
                f"images/{view}/prompt_negative_count", (count,), dtype=np.uint8)
            self._dataset(
                f"images/{view}/yellow_tip_px", (count, 2), fillvalue=np.nan)
            group.create_dataset("prompt_source", shape=(count,), dtype=self.string_type)
            group.create_dataset("tip_source", shape=(count,), dtype=self.string_type)
            self._dataset(
                f"images/{view}/marker_centers_px", (count, 4, 2),
                fillvalue=np.nan)
            self._dataset(
                f"images/{view}/marker_widths_px", (count, 4),
                fillvalue=np.nan)
            self._dataset(
                f"images/{view}/marker_confidence", (count, 4),
                fillvalue=np.nan)
            self._dataset(
                f"images/{view}/marker_observed", (count, 4),
                dtype=np.uint8, fillvalue=0)
            self._dataset(
                f"images/{view}/marker_interpolated", (count, 4),
                dtype=np.uint8, fillvalue=0)
            self._dataset(
                f"images/{view}/marker_raw_cluster_count", (count,),
                dtype=np.uint8, fillvalue=0)

        for name, samples in (
                ("full", config.full_samples),
                ("distal", config.distal_samples)):
            self.file.create_group(name)
            self._dataset(f"{name}/points_base_mm", (count, samples, 3), fillvalue=np.nan)
            self._dataset(f"{name}/s_mm", (count, samples), fillvalue=np.nan)
            self._dataset(f"{name}/tangent_base", (count, samples, 3), fillvalue=np.nan)
            self._dataset(f"{name}/curvature_per_mm", (count, samples), fillvalue=np.nan)
            self._dataset(
                f"{name}/observation_class", (count, samples),
                dtype=np.uint8, fillvalue=0)
        self.file.create_group("stereo")
        self._dataset(
            "stereo/visible_points_base_mm",
            (count, config.stereo_samples, 3), fillvalue=np.nan)
        self._dataset(
            "stereo/causal_visible_points_base_mm",
            (count, config.stereo_samples, 3), fillvalue=np.nan)
        for name in ("ordered_left_px", "ordered_right_px"):
            self._dataset(
                f"stereo/{name}", (count, config.stereo_samples, 2),
                fillvalue=np.nan)
        for name in ("fitted_disparity_px", "smoothed_disparity_px"):
            self._dataset(
                f"stereo/{name}", (count, config.stereo_samples),
                fillvalue=np.nan)
        self.file.create_group("joint")
        self._dataset(
            "joint/coefficients_base_mm",
            (count, config.joint_spline_basis_count, 3), fillvalue=np.nan)
        self._dataset("distal/base_position_base_mm", (count, 3), fillvalue=np.nan)
        self._dataset(
            "distal/raw_base_position_base_mm", (count, 3), fillvalue=np.nan)
        self._dataset(
            "distal/causal_base_position_base_mm", (count, 3), fillvalue=np.nan)
        self.file.create_group("markers")
        self._dataset(
            "markers/points_base_mm", (count, 4, 3), fillvalue=np.nan)
        self._dataset(
            "markers/stereo_observed", (count, 4), dtype=np.uint8, fillvalue=0)

        self.file.create_group("robot")
        for name in ("joint_velocity_command", "joint_position_measured", "encoder_raw"):
            self._dataset(f"robot/{name}", (count, 6), fillvalue=np.nan)
        for name in ("command_age_ms", "position_age_ms", "encoder_age_ms"):
            self._dataset(f"robot/{name}", (count,), fillvalue=np.nan)
        for name in ("command_valid", "position_valid", "encoder_valid"):
            self._dataset(f"robot/{name}", (count,), dtype=np.uint8, fillvalue=0)

        self.file.create_group("quality")
        for name in (
                "reprojection_left_px", "reprojection_right_px",
                "reprojection_max_px", "reprojection_p95_px",
                "fitted_reprojection_left_px",
                "fitted_reprojection_right_px",
                "fitted_reprojection_max_px",
                "fitted_reprojection_p95_px",
                "tip_epipolar_error_px", "tip_anchor_endpoint_distance_mm",
                "tip_endpoint_left_px", "tip_endpoint_right_px",
                "stereo_candidate_score_left", "stereo_candidate_score_right",
                "stereo_candidate_margin", "stereo_candidate_temporal_rms_left_mm",
                "stereo_candidate_temporal_rms_right_mm",
                "stereo_epipolar_ambiguity_left",
                "stereo_epipolar_ambiguity_right",
                "stereo_other_eye_self_overlap_left",
                "stereo_other_eye_self_overlap_right",
                "overlap_aware_used", "overlap_one_turn_enforced",
                "overlap_expected_turn_index",
                "centerline_tip_anchor_used",
                "centerline_tip_epipolar_error_px",
                "terminal_reprojection_left_px",
                "terminal_reprojection_right_px",
                "terminal_refinement_used",
                "terminal_refinement_improvement_px",
                "terminal_refinement_start_fraction",
                "marker_anchor_count", "marker_interval_boundary_count",
                "marker_epipolar_error_max_px",
                "marker_interval_route_used_left",
                "marker_interval_route_used_right",
                "marker_path_order_valid_left",
                "marker_path_order_valid_right",
                "marker_path_max_centroid_distance_left_px",
                "marker_path_max_centroid_distance_right_px",
                "marker_path_observed_count_left",
                "marker_path_observed_count_right",
                "centerline_sharp_turn_cluster_count_left",
                "centerline_sharp_turn_cluster_count_right",
                "centerline_max_local_turn_left_deg",
                "centerline_max_local_turn_right_deg",
                "epipolar_sweep_left_px", "epipolar_sweep_right_px",
                "epipolar_sweep_deficit_px",
                "epipolar_sweep_enforced_view",
                "epipolar_sweep_anchor_used",
                "temporal_centerline_coverage_left",
                "temporal_centerline_coverage_right",
                "temporal_centerline_p95_left_px",
                "temporal_centerline_p95_right_px",
                "stereo_condition", "material_boundary_fraction",
                "material_boundary_color_fraction",
                "material_boundary_prior_error_mm",
                "material_boundary_observed_distal_length_mm",
                "distal_length_raw_mm", "distal_length_filtered_mm",
                "distal_length_smoothed_mm",
                "distal_length_residual_mm", "interface_uncertainty_mm",
                "interface_raw_s_mm", "interface_filtered_s_mm",
                "interface_smoothed_s_mm",
                "interface_smoothing_adjustment_mm",
                "interface_smoothing_weight",
                "interface_temporal_alpha",
                "material_boundary_confidence", "material_boundary_contrast",
                "distal_boundary_s_mm", "base_bridge_length_mm",
                "base_endpoint_distance_mm", "visible_arc_length_mm",
                "centerline_length_left_px", "centerline_length_right_px",
                "centerline_outside_mask_fraction_left",
                "centerline_outside_mask_fraction_right",
                "mask_effective_width_left_px",
                "mask_effective_width_right_px",
                "stereo_centerline_length_ratio_initial",
                "stereo_centerline_length_ratio",
                "stereo_retry_used", "stereo_retry_view",
                "stereo_retry_attempted", "stereo_retry_status_code",
                "stereo_reference_view",
                "stereo_ill_view", "stereo_observation_weight_left",
                "stereo_observation_weight_right",
                "overlap_aware_forced", "stereo_inferred_view",
                "stereo_raw_ill_view", "stereo_ill_view_switch_pending",
                "stereo_inferred_candidate_view",
                "stereo_inferred_accepted",
                "stereo_inferred_rejection_code",
                "stereo_inferred_marker_order_valid",
                "stereo_inferred_terminal_error_px",
                "stereo_inferred_sharp_turn_clusters",
                "stereo_inferred_temporal_p95_px",
                "stereo_inferred_temporal_max_px",
                "stereo_inferred_outside_mask_fraction",
                "stereo_inferred_medial_radius_ratio",
                "stereo_ill_detection_ambiguity",
                "temporal_mask_inconsistency_bypassed_view",
                "final_mask_outside_fraction_left",
                "final_mask_outside_fraction_right",
                "final_mask_distance_p95_left_px",
                "final_mask_distance_p95_right_px",
                "matched_epipolar", "disparity_robust_inlier_count",
                "full_spline_basis_count", "full_spline_internal_knot_count",
                "full_spline_rms_residual_mm", "distal_spline_basis_count",
                "distal_spline_internal_knot_count",
                "distal_spline_rms_residual_mm",
                "distal_spline_arc_length_mm", "temporal_shape_alpha",
                "joint_left_model_mean_px", "joint_right_model_mean_px",
                "joint_left_model_p95_px", "joint_right_model_p95_px",
                "joint_left_coverage_mean_px",
                "joint_right_coverage_mean_px",
                "joint_initial_symmetric_mean_px",
                "joint_final_symmetric_mean_px",
                "joint_arc_length_mm", "joint_length_residual_mm",
                "joint_optimizer_cost", "joint_optimizer_evaluations",
                "joint_optimizer_success", "joint_turn_fraction",
                "joint_left_turn_angle_deg", "joint_right_turn_angle_deg",
                "joint_topology_recovery_used",
                "joint_left_sharp_turn_clusters",
                "joint_right_sharp_turn_clusters"):
            self._dataset(f"quality/{name}", (count,), fillvalue=np.nan)
        self._dataset(
            "quality/material_boundary_observed", (count,),
            dtype=np.uint8, fillvalue=0)
        self._dataset(
            "quality/material_boundary_stereo_consistent", (count,),
            dtype=np.uint8, fillvalue=0)
        self._dataset(
            "quality/tip_stereo_observed", (count,),
            dtype=np.uint8, fillvalue=0)

    def _record(self, index: int) -> dict[str, object]:
        return self._pending.setdefault(int(index), {})

    @staticmethod
    def _copy_value(value):
        return value.copy() if isinstance(value, np.ndarray) else value

    def _check_writer(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("background HDF5 writer failed") from self._writer_error

    def _queue_put(self, item) -> None:
        while True:
            self._check_writer()
            try:
                self._write_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _flush_staging(self) -> None:
        if self._staging:
            chunk = tuple(self._staging)
            self._staging.clear()
            self._queue_put(chunk)

    def _finalize_record(self, index: int) -> None:
        record = self._pending.pop(int(index))
        if self._staging and index != self._staging[-1][0] + 1:
            self._flush_staging()
        self._staging.append((int(index), record))
        if len(self._staging) >= self.hdf_buffer_frames:
            self._flush_staging()

    def _filled_batch(self, dataset, count: int):
        shape = (count, *dataset.shape[1:])
        if dataset.dtype.kind in ("O", "S", "U"):
            output = np.empty(shape, dtype=object)
            output[...] = ""
            return output
        fill = dataset.fillvalue
        if fill is None:
            fill = np.nan if dataset.dtype.kind == "f" else 0
        return np.full(shape, fill, dtype=dataset.dtype)

    def _write_batch(self, records: tuple[tuple[int, dict[str, object]], ...]) -> None:
        indices = [item[0] for item in records]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("HDF5 batch indices must be contiguous")
        paths = sorted({path for _, record in records for path in record})
        start, stop = indices[0], indices[-1] + 1
        for path in paths:
            dataset = self.file[path]
            batch = self._filled_batch(dataset, len(records))
            for offset, (_, record) in enumerate(records):
                if path in record:
                    batch[offset] = record[path]
            dataset[start:stop] = batch

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self._write_queue.get()
                if item is self._END:
                    return
                started = time.perf_counter()
                self._write_batch(item)
                self.write_time_s += time.perf_counter() - started
        except BaseException as exc:
            self._writer_error = exc

    def write_identity(
            self,
            index: int,
            record: FrameRecord,
            svo_timestamp_ns: int,
            robot: AlignedRobotData) -> None:
        output = self._record(index)
        output.update({
            "frames/svo_frame": int(record.svo_frame),
            "frames/timestamp_ns": int(record.timestamp_ns),
            "frames/svo_timestamp_ns": int(svo_timestamp_ns),
        })
        mappings = {
            "joint_velocity_command": robot.command_velocity,
            "joint_position_measured": robot.measured_position,
            "encoder_raw": robot.raw_encoder,
            "command_age_ms": robot.command_age_ms,
            "position_age_ms": robot.position_age_ms,
            "encoder_age_ms": robot.encoder_age_ms,
            "command_valid": robot.command_valid,
            "position_valid": robot.position_valid,
            "encoder_valid": robot.encoder_valid,
        }
        for name, values in mappings.items():
            output[f"robot/{name}"] = self._copy_value(values[index])

    def write_view(self, index: int, view: str, result: SamMaterialResult) -> None:
        output = self._record(index)
        material = result.material
        mask_path = f"images/{view}/mask_packbits"
        if self.store_masks:
            packed = np.packbits((material.mask > 0).reshape(-1), bitorder="little")
            output[mask_path] = packed
        sampled = resample_arclength(
            material.points, self.image_centerline_samples,
            smooth_window=1)
        boundary = int(np.clip(
            material.distal_boundary_index, 0, len(material.points) - 1))
        positive_count = min(16, len(result.prompt.positive_xy))
        negative_count = min(16, len(result.prompt.negative_xy))
        positive = np.full((16, 2), np.nan, dtype=np.float32)
        negative = np.full((16, 2), np.nan, dtype=np.float32)
        if positive_count:
            positive[:positive_count] = result.prompt.positive_xy[:positive_count]
        if negative_count:
            negative[:negative_count] = result.prompt.negative_xy[:negative_count]
        output.update({
            f"images/{view}/centerline_px": sampled,
            f"images/{view}/distal_boundary_px": material.points[boundary].copy(),
            f"images/{view}/sam_iou": float(result.sam_iou),
            f"images/{view}/selection_score": float(result.selection_score),
            f"images/{view}/seed_recall": float(result.seed_recall),
            f"images/{view}/mask_area_px": int(result.mask_area_px),
            f"images/{view}/prompt_source": result.prompt.source,
            f"images/{view}/tip_source": result.tip_source,
            f"images/{view}/prompt_box_xyxy": result.prompt.box_xyxy.copy(),
            f"images/{view}/prompt_positive_px": positive,
            f"images/{view}/prompt_negative_px": negative,
            f"images/{view}/prompt_positive_count": positive_count,
            f"images/{view}/prompt_negative_count": negative_count,
        })
        if result.yellow_tip_xy is not None:
            output[f"images/{view}/yellow_tip_px"] = result.yellow_tip_xy.copy()
        if result.marker_centers_xy is not None:
            output[f"images/{view}/marker_centers_px"] = np.asarray(
                result.marker_centers_xy).copy()
        if result.marker_widths_px is not None:
            output[f"images/{view}/marker_widths_px"] = np.asarray(
                result.marker_widths_px).copy()
        if result.marker_confidence is not None:
            output[f"images/{view}/marker_confidence"] = np.asarray(
                result.marker_confidence).copy()
        if result.marker_observed is not None:
            output[f"images/{view}/marker_observed"] = np.asarray(
                result.marker_observed, dtype=np.uint8).copy()
        if result.marker_interpolated is not None:
            output[f"images/{view}/marker_interpolated"] = np.asarray(
                result.marker_interpolated, dtype=np.uint8).copy()
        output[f"images/{view}/marker_raw_cluster_count"] = int(
            result.marker_raw_cluster_count)

    def write_observed_centerline(
            self, index: int, view: str, result: SamMaterialResult) -> None:
        """Preserve independent image evidence before unified 3D projection."""
        self._record(index)[f"images/{view}/observed_centerline_px"] = (
            resample_arclength(
                result.material.points, self.image_centerline_samples,
                smooth_window=1))

    def write_success(
            self,
            index: int,
            assembled: AssembledShape,
            full_geometry,
            distal_geometry,
            metrics: dict,
            raw_interface_base_mm: np.ndarray | None = None,
            visible_points_base_mm: np.ndarray | None = None,
            reconstruction: dict | None = None,
            joint_reconstruction=None,
            marker_points_base_mm: np.ndarray | None = None,
            marker_stereo_observed: np.ndarray | None = None) -> None:
        output = self._record(index)
        for name, shape, geometry in (
                ("full", assembled, full_geometry),
                ("distal", assembled, distal_geometry)):
            prefix = "full" if name == "full" else "distal"
            output[f"{prefix}/points_base_mm"] = geometry.points_mm.copy()
            output[f"{prefix}/s_mm"] = cumulative_arclength(
                geometry.points_mm)
            output[f"{prefix}/tangent_base"] = geometry.tangent.copy()
            output[f"{prefix}/curvature_per_mm"] = (
                geometry.curvature_per_mm.copy())
            output[f"{prefix}/observation_class"] = getattr(
                shape, f"{prefix}_observation_class").copy()
        output["distal/base_position_base_mm"] = (
            distal_geometry.points_mm[0].copy())
        if raw_interface_base_mm is not None:
            output["distal/raw_base_position_base_mm"] = (
                np.asarray(raw_interface_base_mm).copy())
        if visible_points_base_mm is not None:
            output["stereo/visible_points_base_mm"] = np.asarray(
                visible_points_base_mm).copy()
            output["stereo/causal_visible_points_base_mm"] = np.asarray(
                visible_points_base_mm).copy()
        if reconstruction is not None:
            for name in (
                    "ordered_left_px", "ordered_right_px",
                    "fitted_disparity_px"):
                output[f"stereo/{name}"] = np.asarray(
                    reconstruction[name]).copy()
        if joint_reconstruction is not None:
            output["joint/coefficients_base_mm"] = np.asarray(
                joint_reconstruction.coefficients_base_mm).copy()
        if marker_points_base_mm is not None:
            output["markers/points_base_mm"] = np.asarray(
                marker_points_base_mm).copy()
        if marker_stereo_observed is not None:
            output["markers/stereo_observed"] = np.asarray(
                marker_stereo_observed, dtype=np.uint8).copy()
        self.write_metrics(index, metrics)
        output["frames/observation_valid"] = 1
        output["frames/valid"] = 1
        output["frames/status"] = "valid"
        self._finalize_record(index)

    def write_metrics(self, index: int, metrics: dict) -> None:
        output = self._record(index)
        for name, value in metrics.items():
            path = f"quality/{name}"
            if path in self.dataset_paths:
                output[path] = self._copy_value(value)

    def mark_observation_valid(self, index: int) -> None:
        self._record(index)["frames/observation_valid"] = 1

    def write_failure(self, index: int, status: str, metrics: dict | None = None) -> None:
        output = self._record(index)
        if metrics:
            self.write_metrics(index, metrics)
        output["frames/valid"] = 0
        output["frames/status"] = status
        self._finalize_record(index)

    def write_observations(
            self, index: int, left: SamMaterialResult,
            right: SamMaterialResult, metrics: dict | None = None) -> None:
        """Finalize a mask/centerline/marker cache without running 3D stages."""
        self.write_observed_centerline(index, "left", left)
        self.write_observed_centerline(index, "right", right)
        self.write_view(index, "left", left)
        self.write_view(index, "right", right)
        if metrics:
            self.write_metrics(index, metrics)
        output = self._record(index)
        output["frames/observation_valid"] = 1
        output["frames/valid"] = 0
        output["frames/status"] = "observations_only"
        self._finalize_record(index)

    def close(self) -> None:
        unfinished = sorted(self._pending)
        for index in unfinished:
            output = self._record(index)
            output["frames/valid"] = 0
            output["frames/status"] = "hdf_writer_closed_before_frame_finalized"
            self._finalize_record(index)
        try:
            self._flush_staging()
            self._queue_put(self._END)
            self._writer_thread.join()
            self._check_writer()
        finally:
            if self._writer_error is not None:
                self._writer_thread.join()
            self.file.close()
        if unfinished:
            raise RuntimeError(
                f"HDF5 writer closed with unfinished frames: {unfinished[:5]}")


class OverlayWriter:
    def __init__(
            self, output: Path, fps: float, scale: float,
            video_enabled: bool, snapshots_enabled: bool = False,
            snapshot_indices: set[int] | None = None):
        self.output = output
        self.fps = fps
        self.scale = scale
        self.video_enabled = video_enabled
        self.snapshots_enabled = snapshots_enabled
        self.snapshot_indices = snapshot_indices
        self.writers: dict[str, cv2.VideoWriter] = {}
        if snapshots_enabled:
            for view in ("left", "right"):
                (output / "overlay_snapshots" / view).mkdir(
                    parents=True, exist_ok=True)

    def _writer(self, view: str, image: np.ndarray) -> cv2.VideoWriter:
        if view not in self.writers:
            height, width = image.shape[:2]
            size = (int(round(width * self.scale)), int(round(height * self.scale)))
            writer = cv2.VideoWriter(
                str(self.output / f"overlay_{view}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
            if not writer.isOpened():
                raise RuntimeError(f"failed to open overlay video for {view}")
            self.writers[view] = writer
        return self.writers[view]

    def write(
            self,
            view: str,
            image: np.ndarray,
            result: SamMaterialResult | None,
            projected_shape: np.ndarray | None,
            projected_boundary: np.ndarray | None,
            status: str,
            frame_index: int | None = None,
            svo_frame: int | None = None) -> None:
        write_snapshot = bool(
            self.snapshots_enabled
            and frame_index is not None
            and (self.snapshot_indices is None
                 or frame_index in self.snapshot_indices))
        if not self.video_enabled and not write_snapshot:
            return
        if result is not None:
            output = draw_material_overlay(
                image, result.material, projected_shape=projected_shape,
                projected_boundary=projected_boundary,
                quality_text=(
                    f"{status} SAM={result.sam_iou:.2f} "
                    f"prompt={result.prompt.source}"))
            for point in result.prompt.positive_xy:
                cv2.circle(output, tuple(np.rint(point).astype(int)), 4, (0, 255, 0), -1)
            for point in result.prompt.negative_xy:
                cv2.drawMarker(output, tuple(np.rint(point).astype(int)), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 8, 2)
            if result.marker_centers_xy is not None:
                observed = (
                    np.ones(4, dtype=bool) if result.marker_observed is None
                    else np.asarray(result.marker_observed, dtype=bool))
                for marker_id, point in enumerate(result.marker_centers_xy):
                    if (marker_id >= len(observed) or not observed[marker_id]
                            or not np.all(np.isfinite(point))):
                        continue
                    center = tuple(np.rint(point).astype(int))
                    cv2.circle(output, center, 7, (255, 0, 255), 2,
                               cv2.LINE_AA)
                    cv2.putText(
                        output, str(marker_id), (center[0] + 6, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2,
                        cv2.LINE_AA)
            if (result.yellow_tip_xy is not None
                    and np.all(np.isfinite(result.yellow_tip_xy))):
                cv2.circle(
                    output, tuple(np.rint(result.yellow_tip_xy).astype(int)),
                    6, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            output = image.copy()
            cv2.putText(output, status, (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if self.video_enabled:
            resized = cv2.resize(output, None, fx=self.scale, fy=self.scale)
            self._writer(view, output).write(resized)
        if write_snapshot:
            source = frame_index if svo_frame is None else svo_frame
            path = (
                self.output / "overlay_snapshots" / view
                / f"frame_{frame_index:06d}_svo_{source:06d}.png")
            if not cv2.imwrite(str(path), output):
                raise RuntimeError(f"failed to write overlay snapshot {path}")

    def close(self) -> None:
        for writer in self.writers.values():
            writer.release()


def _write_final_overlays(
        session: Path,
        output: Path,
        records: list[FrameRecord],
        registration,
        mask_h5: Path,
        fps: float,
        scale: float,
        write_video: bool,
        write_snapshots: bool,
        snapshot_indices: set[int]) -> None:
    """Render overlays after the non-causal interface pass has finalized."""
    import h5py

    overlay = OverlayWriter(
        output, fps, scale, video_enabled=write_video,
        snapshots_enabled=write_snapshots,
        snapshot_indices=snapshot_indices)
    cache = PropagatedMaskCache(mask_h5, records)
    base_left = project_points(
        registration.K, registration.left_camera_T_base,
        np.zeros(3, dtype=np.float64))[0][0]
    base_right = project_points(
        registration.K, registration.right_camera_T_base,
        np.zeros(3, dtype=np.float64))[0][0]
    try:
        with h5py.File(output / "processed_shapes.h5", "r") as shapes, (
                SvoReader(find_svo(session))) as svo:
            valid = shapes["frames/valid"][:].astype(bool)
            shape_supported = (
                shapes["quality/shape_temporal_supported"][:].astype(bool)
                if "quality/shape_temporal_supported" in shapes
                else valid.copy())
            if "frames/curve_temporally_interpolated" in shapes:
                interpolated = shapes[
                    "frames/curve_temporally_interpolated"][:].astype(bool)
                shape_supported |= interpolated
            else:
                interpolated = np.zeros(len(valid), dtype=bool)
            if "frames/learning_valid" in shapes:
                shape_supported &= (
                    shapes["frames/learning_valid"][:].astype(bool)
                    | interpolated)
            statuses = shapes["frames/status"][:]
            for index, record in enumerate(records):
                _, left_image, right_image = svo.read(record.svo_frame)
                status_value = statuses[index]
                status = (
                    status_value.decode() if isinstance(status_value, bytes)
                    else str(status_value))
                if valid[index] and not shape_supported[index]:
                    status = "learning_rejected_final_geometry"
                elif interpolated[index]:
                    status = "temporally_interpolated_3d"
                for view, image, roi, base_point in (
                        ("left", left_image, registration.roi_left_xywh,
                         base_left),
                        ("right", right_image, registration.roi_right_xywh,
                         base_right)):
                    try:
                        result = cache.result(
                            index, view, image, roi, base_point)
                    except (RuntimeError, ValueError):
                        result = None
                    projected = boundary = None
                    if valid[index] and shape_supported[index] and result is not None:
                        points = shapes["distal/points_base_mm"][index]
                        projected = _project_base_points(
                            registration, points, view == "right")
                        boundary = projected[0]
                        result = _replace_distal_with_joint_projection(
                            result, projected)
                    overlay.write(
                        view, image, result, projected, boundary, status,
                        frame_index=index, svo_frame=record.svo_frame)
    finally:
        cache.close()
        overlay.close()


def _refresh_final_geometry_metrics(path: Path, registration) -> None:
    """Make stored reprojection diagnostics describe the final distal curve."""
    import h5py

    with h5py.File(path, "r+") as output:
        valid = output["frames/valid"][:].astype(bool)
        if "quality/shape_temporal_supported" in output:
            supported = output[
                "quality/shape_temporal_supported"][:].astype(bool)
            if "frames/curve_temporally_interpolated" in output:
                supported |= output[
                    "frames/curve_temporally_interpolated"][:].astype(bool)
            valid &= supported
        count = len(valid)

        def dataset(name: str):
            if name in output:
                return output[name]
            return output.create_dataset(
                name, shape=(count,), dtype=np.float32, fillvalue=np.nan,
                compression="gzip", compression_opts=1, shuffle=True)

        terminal_left = dataset(
            "quality/final_terminal_reprojection_left_px")
        terminal_right = dataset(
            "quality/final_terminal_reprojection_right_px")
        mask_outside = {
            view: dataset(f"quality/final_mask_outside_fraction_{view}")
            for view in ("left", "right")}
        mask_p95 = {
            view: dataset(f"quality/final_mask_distance_p95_{view}_px")
            for view in ("left", "right")}
        for index in np.flatnonzero(valid):
            points = output["distal/points_base_mm"][index]
            projected_views = []
            distances = []
            means = []
            terminal = []
            for view, right in (("left", False), ("right", True)):
                projected = _project_base_points(
                    registration, points, right=right)
                observed_name = f"images/{view}/observed_centerline_px"
                centerline = output[
                    observed_name
                    if observed_name in output
                    else f"images/{view}/centerline_px"][index]
                centerline = centerline[np.all(np.isfinite(centerline), axis=1)]
                distance = (
                    np.min(np.linalg.norm(
                        projected[:, None, :] - centerline[None, :, :], axis=2),
                        axis=1)
                    if len(centerline) else np.full(len(projected), np.nan))
                projected_views.append(projected)
                distances.append(distance)
                means.append(float(np.mean(distance)))
                terminal.append(float(np.mean(distance[-8:])))
                mask_name = f"images/{view}/mask_packbits"
                if mask_name in output:
                    mask_ds = output[mask_name]
                    mask_shape = tuple(int(value) for value in
                                       mask_ds.attrs["unpacked_shape"])
                    roi = tuple(int(value) for value in
                                mask_ds.attrs["roi_xywh"])
                    mask = np.unpackbits(
                        np.asarray(mask_ds[index]), bitorder="little",
                        count=mask_shape[0] * mask_shape[1]
                    ).reshape(mask_shape).astype(bool)
                    local = projected - np.asarray(roi[:2], dtype=float)
                    pixel = np.rint(local).astype(int)
                    inside = (
                        (pixel[:, 0] >= 0)
                        & (pixel[:, 0] < mask_shape[1])
                        & (pixel[:, 1] >= 0)
                        & (pixel[:, 1] < mask_shape[0]))
                    supported = np.zeros(len(pixel), dtype=bool)
                    supported[inside] = mask[
                        pixel[inside, 1], pixel[inside, 0]]
                    distance_to_mask = np.full(len(pixel), np.inf)
                    if np.any(mask):
                        distance_field = cv2.distanceTransform(
                            np.uint8(~mask), cv2.DIST_L2, 5)
                        distance_to_mask[inside] = distance_field[
                            pixel[inside, 1], pixel[inside, 0]]
                    mask_outside[view][index] = float(
                        1.0 - np.mean(supported))
                    mask_p95[view][index] = float(np.percentile(
                        distance_to_mask, 95))
            combined = np.concatenate(distances)
            finite_combined = combined[np.isfinite(combined)]
            output["quality/fitted_reprojection_left_px"][index] = means[0]
            output["quality/fitted_reprojection_right_px"][index] = means[1]
            output["quality/fitted_reprojection_max_px"][index] = (
                float(np.max(finite_combined)) if len(finite_combined)
                else float("nan"))
            output["quality/fitted_reprojection_p95_px"][index] = (
                float(np.percentile(finite_combined, 95))
                if len(finite_combined) else float("nan"))
            terminal_left[index], terminal_right[index] = terminal
            for view_index, view in enumerate(("left", "right")):
                terminal_observation = output[
                    f"images/{view}/yellow_tip_px"][index]
                if (f"images/{view}/marker_centers_px" in output
                        and f"images/{view}/marker_observed" in output
                        and output[f"images/{view}/marker_observed"][index, 3]):
                    marker = output[
                        f"images/{view}/marker_centers_px"][index, 3]
                    if np.all(np.isfinite(marker)):
                        terminal_observation = marker
                error = (
                    float(np.linalg.norm(
                        projected_views[view_index][-1]
                        - terminal_observation))
                    if np.all(np.isfinite(terminal_observation))
                    else float("nan"))
                output[f"quality/tip_endpoint_{view}_px"][index] = error

                # Schema 14 keeps the independent image route separately, so
                # it is safe to make centerline_px represent the temporally
                # finalized, unified topology.  This prevents final overlays
                # (and downstream consumers) from mixing a causal cyan route
                # with a zero-phase-filtered yellow 3D curve.  Older files do
                # not have observed_centerline_px and remain read-only here.
                evidence_name = f"images/{view}/observed_centerline_px"
                if evidence_name in output:
                    view_projected = projected_views[view_index]
                    stored_ds = output[f"images/{view}/centerline_px"]
                    stored = np.asarray(stored_ds[index], dtype=float)
                    stored = stored[np.all(np.isfinite(stored), axis=1)]
                    if len(stored) >= 2:
                        old_boundary = np.asarray(output[
                            f"images/{view}/distal_boundary_px"][index], float)
                        boundary_index = int(np.argmin(np.linalg.norm(
                            stored - old_boundary, axis=1)))
                        proximal = stored[:boundary_index]
                        unified = (
                            np.vstack([proximal, view_projected])
                            if len(proximal) else view_projected)
                        stored_ds[index] = resample_arclength(
                            unified, stored_ds.shape[1], smooth_window=1)
                        output[f"images/{view}/distal_boundary_px"][index] = (
                            view_projected[0])
        output.flush()


def _apply_final_learning_quality(
        path: Path,
        max_mean_reprojection_px: float,
        max_p95_reprojection_px: float,
        max_tip_endpoint_error_px: float,
        marker_tip_width_scale: float = 0.40,
        marker_tip_width_cap_px: float = 12.0,
        max_mask_outside_fraction: float = 0.20,
        max_mask_distance_p95_px: float = 6.0) -> tuple[int, int]:
    """Reject temporally finalized curves that no longer fit the 2D evidence."""
    import h5py

    final_reprojection_flag = np.uint16(1 << 5)
    with h5py.File(path, "r+") as output:
        valid = output["frames/valid"][:].astype(bool)
        left_mean = output["quality/fitted_reprojection_left_px"][:]
        right_mean = output["quality/fitted_reprojection_right_px"][:]
        p95 = output["quality/fitted_reprojection_p95_px"][:]
        left_tip = output["quality/tip_endpoint_left_px"][:]
        right_tip = output["quality/tip_endpoint_right_px"][:]
        left_tip_threshold = np.full(
            len(valid), float(max_tip_endpoint_error_px))
        right_tip_threshold = left_tip_threshold.copy()
        for view, threshold in (
                ("left", left_tip_threshold),
                ("right", right_tip_threshold)):
            width_name = f"images/{view}/marker_widths_px"
            observed_name = f"images/{view}/marker_observed"
            if width_name not in output:
                continue
            widths = output[width_name][:, 3].astype(float)
            observed = (
                output[observed_name][:, 3].astype(bool)
                if observed_name in output else np.isfinite(widths))
            uncertainty = np.minimum(
                float(marker_tip_width_cap_px),
                float(marker_tip_width_scale) * widths)
            use = observed & np.isfinite(uncertainty)
            threshold[use] = np.maximum(threshold[use], uncertainty[use])
        bad_tip = (
            (np.isfinite(left_tip)
             & (left_tip > left_tip_threshold))
            | (np.isfinite(right_tip)
               & (right_tip > right_tip_threshold)))
        bad_mask = np.zeros(len(valid), dtype=bool)
        for view in ("left", "right"):
            outside_name = f"quality/final_mask_outside_fraction_{view}"
            distance_name = f"quality/final_mask_distance_p95_{view}_px"
            if outside_name in output and distance_name in output:
                outside = output[outside_name][:]
                mask_distance = output[distance_name][:]
                bad_mask |= (
                    (np.isfinite(outside)
                     & (outside > float(max_mask_outside_fraction)))
                    | (np.isfinite(mask_distance)
                       & (mask_distance > float(max_mask_distance_p95_px))))
        image_rejected = valid & (
            (left_mean > float(max_mean_reprojection_px))
            | (right_mean > float(max_mean_reprojection_px))
            | (p95 > float(max_p95_reprojection_px))
            | bad_tip | bad_mask)
        # A single barely-good frame inside a rejected interval is not useful
        # for dynamics learning and produces distracting good/bad overlay
        # chatter. Bridge only short, bounded good gaps; never turn a rejected
        # observation into an accepted one.
        timestamps_ns = output["frames/timestamp_ns"][:]
        image_rejected = _bridge_short_good_gaps(
            image_rejected, valid, timestamps_ns, maximum_gap_ms=100.0)
        interpolated = (
            output["frames/curve_temporally_interpolated"][:].astype(bool)
            if "frames/curve_temporally_interpolated" in output
            else np.zeros(len(valid), dtype=bool))
        # Current masks are least trustworthy in the short V-overlap interval
        # for which this time-symmetric bridge exists. The bridge has already
        # passed duration, finite-anchor, point-step and arc-length checks, so
        # retain it even when the contaminated per-frame mask disagrees.
        overridden = image_rejected & interpolated
        rejected = image_rejected & ~interpolated
        override_name = "quality/interpolated_curve_image_qc_overridden"
        if override_name in output:
            override_ds = output[override_name]
        else:
            override_ds = output.create_dataset(
                override_name, shape=(len(valid),), dtype=np.uint8,
                fillvalue=0, compression="gzip", compression_opts=1,
                shuffle=True)
        override_ds[:] = overridden.astype(np.uint8)
        flags = output["frames/learning_rejection_flags"][:].astype(np.uint16)
        # Final metrics are recomputed after downstream interpolation/resumes;
        # do not retain a stale final-image-fit bit from the preceding pass.
        flags &= np.uint16(~int(final_reprojection_flag) & 0xffff)
        flags[rejected] |= final_reprojection_flag
        output["frames/learning_rejection_flags"][:] = flags
        output["frames/learning_valid"][:] = (flags == 0).astype(np.uint8)
        mapping = json.loads(output.attrs.get(
            "learning_rejection_flags_json", "{}"))
        mapping["32"] = "final_curve_reprojection_tip_or_mask"
        output.attrs["learning_rejection_flags_json"] = json.dumps(
            mapping, separators=(",", ":"))
        output.flush()
        return (
            int(np.count_nonzero(rejected)),
            int(np.count_nonzero(flags == 0)))


def _resume_joint_fit_from_stereo(
        path: Path, registration, config: ImageProcessingConfig,
        marker_backend_active: bool) -> dict[str, object]:
    """Refit per-frame joint splines from persisted 2D/stereo observations."""
    import h5py

    fitted = failed = 0
    failure_reasons: dict[str, int] = {}
    previous_points: np.ndarray | None = None
    previous_timestamp_ns: int | None = None
    previous_turn_fraction: float | None = None
    nominal_length_mm = (
        config.marked_distal_length_mm
        if marker_backend_active else config.distal_length_mm)
    with h5py.File(path, "r+") as output:
        coefficient_ds = output["joint/coefficients_base_mm"]
        if coefficient_ds.shape[1] != config.joint_spline_basis_count:
            raise ValueError(
                "--resume-from stereo cannot change --joint-spline-basis-count "
                f"({coefficient_ds.shape[1]} stored, "
                f"{config.joint_spline_basis_count} requested); resume from "
                "observations to rebuild the HDF5 schema")
        if "distal/pre_temporal_points_base_mm" in output:
            pre_temporal = output["distal/pre_temporal_points_base_mm"]
        else:
            pre_temporal = output.create_dataset(
                "distal/pre_temporal_points_base_mm",
                shape=output["distal/points_base_mm"].shape,
                dtype=np.float32, fillvalue=np.nan,
                compression="gzip", compression_opts=1, shuffle=True)

        valid = output["frames/valid"][:].astype(bool)
        metric_names = (
            "joint_left_model_mean_px", "joint_right_model_mean_px",
            "joint_left_model_p95_px", "joint_right_model_p95_px",
            "joint_left_coverage_mean_px", "joint_right_coverage_mean_px",
            "joint_initial_symmetric_mean_px",
            "joint_final_symmetric_mean_px", "joint_arc_length_mm",
            "joint_length_residual_mm", "joint_optimizer_cost",
            "joint_optimizer_evaluations", "joint_optimizer_success",
            "joint_topology_recovery_used",
            "joint_left_sharp_turn_clusters",
            "joint_right_sharp_turn_clusters")
        for name in metric_names:
            full_name = f"quality/{name}"
            if full_name not in output:
                output.create_dataset(
                    full_name, shape=(len(valid),), dtype=np.float32,
                    fillvalue=np.nan, compression="gzip",
                    compression_opts=1, shuffle=True)

        for index in np.flatnonzero(valid):
            try:
                visible = np.asarray(
                    output["stereo/visible_points_base_mm"][index], float)
                visible = visible[np.all(np.isfinite(visible), axis=1)]
                if len(visible) < 4:
                    raise ValueError("stored stereo initializer is missing")
                visible_s = cumulative_arclength(visible)
                interface = np.asarray(
                    output["distal/raw_base_position_base_mm"][index], float)
                if not np.all(np.isfinite(interface)):
                    interface = np.asarray(
                        output["distal/base_position_base_mm"][index], float)
                interface_s = _project_point_to_polyline_s(
                    visible, interface, float(visible_s[-1] - nominal_length_mm),
                    float("inf"))
                target_fraction = float(np.clip(
                    interface_s / max(float(visible_s[-1]), 1e-9), 0.0, 1.0))
                initial = assemble_image_only_shape(
                    visible, target_fraction,
                    full_count=config.full_samples,
                    distal_count=config.distal_samples,
                    base_position_base_mm=np.zeros(3), bridge_base=False,
                    distal_length_mm=max(
                        float(visible_s[-1] - interface_s), 1e-3)
                ).distal_points_mm

                left_name = (
                    "images/left/observed_centerline_px"
                    if "images/left/observed_centerline_px" in output
                    else "images/left/centerline_px")
                right_name = (
                    "images/right/observed_centerline_px"
                    if "images/right/observed_centerline_px" in output
                    else "images/right/centerline_px")
                left_centerline = np.asarray(output[left_name][index], float)
                right_centerline = np.asarray(output[right_name][index], float)
                left_centerline = left_centerline[
                    np.all(np.isfinite(left_centerline), axis=1)]
                right_centerline = right_centerline[
                    np.all(np.isfinite(right_centerline), axis=1)]
                left_markers = np.asarray(
                    output["images/left/marker_centers_px"][index], float)
                right_markers = np.asarray(
                    output["images/right/marker_centers_px"][index], float)
                left_boundary = np.asarray(
                    output["images/left/distal_boundary_px"][index], float)
                right_boundary = np.asarray(
                    output["images/right/distal_boundary_px"][index], float)

                def endpoint(markers, marker_index, fallback):
                    point = markers[marker_index]
                    return point if np.all(np.isfinite(point)) else fallback

                left_distal, left_endpoints = distal_image_observation(
                    left_centerline,
                    endpoint(left_markers, 0, left_boundary),
                    endpoint(left_markers, 3, left_centerline[-1]),
                    count=max(64, config.stereo_samples))
                right_distal, right_endpoints = distal_image_observation(
                    right_centerline,
                    endpoint(right_markers, 0, right_boundary),
                    endpoint(right_markers, 3, right_centerline[-1]),
                    count=max(64, config.stereo_samples))

                left_observation_weight = 1.0
                right_observation_weight = 1.0
                observation_ill_view = None
                if marker_backend_active:
                    left_image_length = float(cumulative_arclength(
                        left_centerline)[-1])
                    right_image_length = float(cumulative_arclength(
                        right_centerline)[-1])
                    image_length_ratio = max(
                        left_image_length, right_image_length) / max(
                            min(left_image_length, right_image_length), 1e-9)
                    if (image_length_ratio
                            > config.marked_stereo_guided_retry_ratio):
                        shorter_view = (
                            "left" if left_image_length < right_image_length
                            else "right")
                        active_code = int(output[
                            "quality/stereo_ill_view"][index]) if (
                                "quality/stereo_ill_view" in output) else 0
                        active_view = (
                            "left" if active_code == 1 else
                            "right" if active_code == 2 else shorter_view)
                        (left_observation_weight,
                         right_observation_weight,
                         observation_ill_view) = (
                            _ill_eye_observation_weights(
                                active_view, shorter_view,
                                left_image_length, right_image_length,
                                config.ill_eye_min_observation_weight))

                ordered_left = np.asarray(
                    output["stereo/ordered_left_px"][index], float)
                ordered_right = np.asarray(
                    output["stereo/ordered_right_px"][index], float)
                boundary_sample = int(np.clip(round(
                    target_fraction * (len(ordered_left) - 1)),
                    1, len(ordered_left) - 2))
                left_corridor = ordered_left[boundary_sample:]
                right_corridor = ordered_right[boundary_sample:]
                corridor_s = visible_s[boundary_sample:]
                left_corridor, right_corridor = _center_paired_mask_corridors(
                    left_corridor, right_corridor,
                    _stored_material_result(output, index, "left"),
                    _stored_material_result(output, index, "right"),
                    config.distal_samples, corridor_s)
                left_turn, left_angle = _projected_turn_fraction(left_corridor)
                right_turn, right_angle = _projected_turn_fraction(right_corridor)
                if observation_ill_view == "left":
                    turn_fraction = _epipolar_sweep_extremum_fraction(
                        right_corridor)
                elif observation_ill_view == "right":
                    turn_fraction = _epipolar_sweep_extremum_fraction(
                        left_corridor)
                else:
                    turn_fraction = None
                timestamp_ns = int(output["frames/timestamp_ns"][index])
                temporal_adjacent = bool(
                    previous_points is not None
                    and previous_timestamp_ns is not None
                    and 0 < (timestamp_ns - previous_timestamp_ns) * 1e-6
                    <= config.max_temporal_prompt_gap_ms)
                if turn_fraction is None and observation_ill_view is not None:
                    turn_fraction = (
                        previous_turn_fraction if temporal_adjacent else None)
                elif (turn_fraction is not None and temporal_adjacent
                      and previous_turn_fraction is not None):
                    turn_fraction = float(
                        0.75 * np.clip(
                            turn_fraction,
                            previous_turn_fraction - 0.08,
                            previous_turn_fraction + 0.08)
                        + 0.25 * previous_turn_fraction)

                common_fit = dict(
                    left_axial_markers_xy=left_markers,
                    right_axial_markers_xy=right_markers,
                    output_samples=config.distal_samples,
                    basis_count=config.joint_spline_basis_count,
                    fit_samples=config.distal_samples,
                    length_sigma_mm=(
                        config.marked_distal_length_prior_sigma_mm
                        if marker_backend_active
                        else config.distal_length_prior_sigma_mm),
                    temporal_prior_points_base_mm=(
                        previous_points if temporal_adjacent else None),
                    temporal_prior_sigma_mm=max(
                        2.5, 0.65 * config.ill_eye_temporal_shape_sigma_mm),
                    max_nfev=config.joint_spline_max_nfev)
                result = None
                topology_recovery_used = False
                if observation_ill_view is not None:
                    ordinary = fit_joint_two_view_spline(
                        initial, left_distal, right_distal, registration.K,
                        registration.left_camera_T_base,
                        registration.right_camera_T_base, nominal_length_mm,
                        left_observation_weight=1.0,
                        right_observation_weight=1.0,
                        coverage_weight=0.30, **common_fit)
                    if _joint_fit_supports_bijective_stereo(ordinary):
                        result = ordinary
                if result is None:
                    topology_recovery_used = observation_ill_view is not None
                    result = fit_joint_two_view_spline(
                        initial, left_distal, right_distal, registration.K,
                        registration.left_camera_T_base,
                        registration.right_camera_T_base, nominal_length_mm,
                        left_ordered_corridor_xy=(
                            left_corridor if topology_recovery_used else None),
                        right_ordered_corridor_xy=(
                            right_corridor if topology_recovery_used else None),
                        left_observation_weight=left_observation_weight,
                        right_observation_weight=right_observation_weight,
                        turn_fraction=(
                            turn_fraction if topology_recovery_used else None),
                        coverage_weight=(
                            0.0 if topology_recovery_used else 0.30),
                        **common_fit)
                if (topology_recovery_used
                        and (result.left_sharp_turn_clusters > 1
                             or result.right_sharp_turn_clusters > 1)):
                    raise ValueError("stored joint fit violates one-turn rule")
                geometry = curve_geometry(
                    result.points_base_mm, config.curvature_smoothing_mm,
                    config.curvature_spline_bases)
                coefficient_ds[index] = result.coefficients_base_mm
                pre_temporal[index] = geometry.points_mm
                output["distal/points_base_mm"][index] = geometry.points_mm
                output["distal/s_mm"][index] = cumulative_arclength(
                    geometry.points_mm)
                output["distal/tangent_base"][index] = geometry.tangent
                output["distal/curvature_per_mm"][index] = (
                    geometry.curvature_per_mm)
                output["distal/base_position_base_mm"][index] = (
                    geometry.points_mm[0])
                values = (
                    result.left_model_mean_px, result.right_model_mean_px,
                    result.left_model_p95_px, result.right_model_p95_px,
                    result.left_coverage_mean_px,
                    result.right_coverage_mean_px,
                    result.initial_symmetric_mean_px,
                    result.final_symmetric_mean_px, result.arc_length_mm,
                    result.length_residual_mm, result.optimizer_cost,
                    result.optimizer_evaluations,
                    int(result.optimizer_success),
                    int(topology_recovery_used),
                    result.left_sharp_turn_clusters,
                    result.right_sharp_turn_clusters)
                for name, value in zip(metric_names, values):
                    output[f"quality/{name}"][index] = value
                fitted += 1
                previous_points = result.points_base_mm.copy()
                previous_timestamp_ns = timestamp_ns
                previous_turn_fraction = turn_fraction
            except (ArithmeticError, RuntimeError, ValueError,
                    np.linalg.LinAlgError) as error:
                failed += 1
                reason = str(error).split(":", 1)[0].strip()
                if not reason:
                    reason = type(error).__name__
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        output.flush()
    return {
        "joint_refit_count": fitted,
        "joint_refit_failed_count": failed,
        "joint_refit_failure_reasons": failure_reasons,
    }


def resume_image_session(
        session_path: os.PathLike | str,
        source_h5: os.PathLike | str,
        output_dir: os.PathLike | str,
        resume_from: str,
        config: ImageProcessingConfig,
        write_video: bool = False,
        write_snapshots: bool = False,
        snapshot_count: int = 12,
        snapshot_frames: list[int] | None = None) -> dict:
    """Resume at a persisted 3D boundary without decoding or segmenting SVO."""
    import h5py

    if resume_from not in ("stereo", "joint"):
        raise ValueError("downstream resume must start from stereo or joint")
    started = time.perf_counter()
    session = Path(session_path).resolve()
    source = Path(source_h5).resolve()
    output = Path(output_dir).resolve()
    destination = output / "processed_shapes.h5"
    if source == destination:
        raise ValueError("resume output must differ from its source HDF5")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"resume destination already exists: {destination}; choose a new "
            "--outdir so an earlier result is not overwritten")
    shutil.copy2(source, destination)
    registration = load_session_registration(session, require_em=False)
    with h5py.File(destination, "r+") as shapes:
        required = (
            "images/left/centerline_px", "images/right/centerline_px",
            "stereo/visible_points_base_mm", "joint/coefficients_base_mm",
            "distal/points_base_mm")
        missing = [name for name in required if name not in shapes]
        if missing:
            raise ValueError("resume HDF5 is missing: " + ", ".join(missing))
        marker_backend_active = (
            "images/left/marker_centers_px" in shapes
            and np.any(np.isfinite(shapes[
                "images/left/marker_centers_px"][:])))
        metadata = json.loads(shapes.attrs.get("metadata_json", "{}"))
        metadata.update({
            "resume_from": resume_from,
            "resume_source_h5": str(source),
            "session_path": str(session),
        })
        shapes.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        shapes.attrs["processing_config_json"] = json.dumps(
            asdict(config), sort_keys=True)

    summary: dict[str, object] = {
        "resume_from": resume_from,
        "resume_source_h5": str(source),
        "output_h5": str(destination),
    }
    if resume_from == "stereo":
        summary.update(_resume_joint_fit_from_stereo(
            destination, registration, config, marker_backend_active))

    if config.spline_temporal_cutoff_hz > 0.0:
        summary["spline_temporal_smoothing"] = (
            smooth_distal_spline_coefficients_hdf5(
                destination,
                cutoff_hz=config.spline_temporal_cutoff_hz,
                huber_delta_mm=config.spline_temporal_huber_delta_mm,
                iterations=config.spline_temporal_iterations,
                maximum_gap_ms=config.spline_temporal_max_gap_ms,
                basis_count=config.curvature_spline_bases,
                terminal_cutoff_hz=(
                    config.spline_temporal_terminal_cutoff_hz),
                terminal_basis_count=(
                    config.spline_temporal_terminal_basis_count),
                outlier_sigma=config.spline_temporal_outlier_sigma,
                outlier_floor_mm=config.spline_temporal_outlier_floor_mm,
                frame_outlier_fraction=(
                    config.spline_temporal_frame_outlier_fraction),
                terminal_outlier_sample_count=(
                    config.spline_temporal_terminal_outlier_samples),
                max_learning_mask_width_px=(
                    config.marked_max_mask_effective_width_px
                    if marker_backend_active
                    else config.max_mask_effective_width_px),
                observation_blend=config.spline_temporal_observation_blend,
                terminal_observation_blend=(
                    config.spline_temporal_terminal_observation_blend),
                reject_recovered_outliers=False))
    _refresh_final_geometry_metrics(destination, registration)
    rejected, learning_valid = _apply_final_learning_quality(
        destination, config.max_mean_reprojection_px,
        config.max_p95_reprojection_px, config.max_tip_endpoint_error_px,
        config.marker_tip_endpoint_width_scale,
        config.marker_tip_endpoint_width_cap_px,
        config.max_final_mask_outside_fraction,
        config.max_final_mask_distance_p95_px)
    interpolation = (
        interpolate_short_spline_gaps_hdf5(
            destination,
            maximum_gap_ms=config.spline_interpolation_max_gap_ms,
            maximum_good_island_frames=(
                config.spline_interpolation_max_good_island_frames))
        if config.spline_temporal_cutoff_hz > 0.0 else {
            "interpolated_frame_count": 0,
            "interpolated_runs": [],
            "maximum_gap_ms": config.spline_interpolation_max_gap_ms})
    if interpolation["interpolated_frame_count"]:
        _refresh_final_geometry_metrics(destination, registration)
        rejected, learning_valid = _apply_final_learning_quality(
            destination, config.max_mean_reprojection_px,
            config.max_p95_reprojection_px,
            config.max_tip_endpoint_error_px,
            config.marker_tip_endpoint_width_scale,
            config.marker_tip_endpoint_width_cap_px,
            config.max_final_mask_outside_fraction,
            config.max_final_mask_distance_p95_px)
    summary["spline_gap_interpolation"] = interpolation
    summary["final_reprojection_rejected_count"] = rejected
    summary["learning_valid_count"] = learning_valid

    if write_video or write_snapshots:
        with h5py.File(destination, "r") as shapes:
            svo_frames = shapes["frames/svo_frame"][:]
            timestamps = shapes["frames/timestamp_ns"][:]
            if "images/left/mask_packbits" not in shapes:
                raise ValueError(
                    "resume overlays require packed masks in the source HDF5")
        records_by_frame = {
            record.svo_frame: record for record in load_frame_index(session)}
        records = [records_by_frame[int(frame)] for frame in svo_frames]
        if snapshot_frames:
            snapshot_indices = {
                int(index) for index in snapshot_frames
                if 0 <= int(index) < len(records)}
        else:
            snapshot_indices = set(np.unique(np.rint(np.linspace(
                0, len(records) - 1, min(max(1, snapshot_count), len(records))
            )).astype(int)).tolist())
        period_s = (float(np.median(np.diff(timestamps))) * 1e-9
                    if len(timestamps) > 1 else 1 / 30)
        _write_final_overlays(
            session, output, records, registration, destination,
            float(np.clip(1.0 / max(period_s, 1e-6), 1.0, 60.0)),
            config.video_scale, write_video=write_video,
            write_snapshots=write_snapshots,
            snapshot_indices=snapshot_indices)

    summary["elapsed_s"] = time.perf_counter() - started
    with (output / "resume_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    with (output / "processing_config.json").open(
            "w", encoding="utf-8") as stream:
        json.dump({
            "metadata": {
                "session_path": str(session), "resume_from": resume_from,
                "resume_source_h5": str(source)},
            "config": asdict(config)}, stream, indent=2, sort_keys=True)
    return summary


def _bridge_short_good_gaps(
        rejected: np.ndarray,
        valid: np.ndarray,
        timestamps_ns: np.ndarray,
        maximum_gap_ms: float) -> np.ndarray:
    """Reject short good islands bounded by rejected frames, non-causally."""
    output = np.asarray(rejected, dtype=bool).copy()
    valid = np.asarray(valid, dtype=bool)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    candidates = valid & ~output
    indices = np.flatnonzero(candidates)
    if not len(indices):
        return output
    splits = np.flatnonzero(np.diff(indices) != 1) + 1
    for run in np.split(indices, splits):
        if not len(run):
            continue
        before, after = int(run[0]) - 1, int(run[-1]) + 1
        if (before < 0 or after >= len(output)
                or not output[before] or not output[after]):
            continue
        duration_ms = (timestamps[after] - timestamps[before]) * 1e-6
        if duration_ms <= float(maximum_gap_ms):
            output[run] = True
    return output


def _apply_distal_boundary_prior(
        result: SamMaterialResult,
        target_fraction: float,
        minimum_fraction: float,
        maximum_fraction: float,
        target_pixel_xy: np.ndarray,
        ) -> tuple[SamMaterialResult, float, bool]:
    """Place the displayed boundary at the known-length location.

    Color contrast is still evaluated, but only inside the 3D-length-derived
    interval. Its detected fraction is returned as a diagnostic; the material
    split used for learning is the geometric target so distal length is stable.
    """
    material = result.material
    points = material.points
    pixel_s = cumulative_arclength(points)
    if pixel_s[-1] <= 1e-9:
        raise ValueError("material_centerline_zero_length")

    def fraction_index(fraction: float) -> int:
        target_s = float(np.clip(fraction, 0.0, 1.0)) * pixel_s[-1]
        return int(np.clip(
            np.searchsorted(pixel_s, target_s), 1, len(points) - 2))

    target_index = int(np.argmin(np.linalg.norm(
        points - np.asarray(target_pixel_xy, dtype=np.float64), axis=1)))
    target_index = int(np.clip(target_index, 1, len(points) - 2))
    lo_index = fraction_index(minimum_fraction)
    hi_index = fraction_index(maximum_fraction)
    lo_index, hi_index = sorted((lo_index, hi_index))
    detected, confidence, contrast, valid = detect_distal_boundary(
        material.brightness_profile,
        min_fraction=lo_index / len(points),
        max_fraction=hi_index / len(points))
    detected_fraction = float(detected / max(len(points) - 1, 1))
    selected_index = int(detected) if valid else target_index
    updated_material = replace(
        material,
        distal_boundary_index=selected_index,
        distal_boundary_fraction=float(
            selected_index / max(len(points) - 1, 1)),
        boundary_confidence=float(confidence),
        boundary_contrast=float(contrast),
        material_valid=bool(valid))
    return replace(result, material=updated_material), detected_fraction, bool(valid)


def _stereo_retry_prompt(
        source: SamMaterialResult,
        target_roi: tuple[int, int, int, int],
        horizontal_shift_px: float) -> PromptSet:
    """Transfer a complete centerline into the other rectified view."""
    source_points = source.material.points
    indices = np.unique(np.rint(
        np.linspace(0, len(source_points) - 1, 8)).astype(int))
    positive = source_points[indices].copy()
    positive[:, 0] += float(horizontal_shift_px)
    x, y, width, height = target_roi
    positive[:, 0] = np.clip(positive[:, 0], x + 2, x + width - 3)
    positive[:, 1] = np.clip(positive[:, 1], y + 2, y + height - 3)
    margin = 35.0
    box = np.array([
        max(x, float(np.min(positive[:, 0]) - margin)),
        max(y, float(np.min(positive[:, 1]) - margin)),
        min(x + width - 1, float(np.max(positive[:, 0]) + margin)),
        min(y + height - 1, float(np.max(positive[:, 1]) + margin)),
    ])
    negative = np.array([
        [box[0], box[1]], [box[2], box[1]],
        [box[0], box[3]], [box[2], box[3]]], dtype=np.float64)
    return PromptSet(
        box_xyxy=box,
        positive_xy=positive,
        negative_xy=negative,
        source="stereo_guided_retry")


def _centerline_prompt(
        result: SamMaterialResult,
        roi: tuple[int, int, int, int],
        source: str,
        margin_px: float = 18.0) -> PromptSet:
    """Build a full-shaft prompt from a previously accepted centerline."""
    points = result.material.points
    indices = np.unique(np.rint(
        np.linspace(0, len(points) - 1, 8)).astype(int))
    positive = points[indices].copy()
    x, y, width, height = roi
    box = np.array([
        max(x, float(np.min(points[:, 0]) - margin_px)),
        max(y, float(np.min(points[:, 1]) - margin_px)),
        min(x + width - 1, float(np.max(points[:, 0]) + margin_px)),
        min(y + height - 1, float(np.max(points[:, 1]) + margin_px)),
    ])
    negative = np.array([
        [box[0], box[1]], [box[2], box[1]],
        [box[0], box[3]], [box[2], box[3]],
    ], dtype=np.float64)
    return PromptSet(
        box_xyxy=box,
        positive_xy=positive,
        negative_xy=negative,
        source=source)


def _mask_effective_width(result: SamMaterialResult) -> float:
    marked_path = (
        result.prompt.source.startswith("marked_chromatic")
        or (result.prompt.source == "stereo_guided_ridge"
            and result.marker_centers_xy is not None))
    if marked_path:
        # area / projected-length is not a width when the catheter overlaps
        # itself or points toward the camera: the denominator shrinks while
        # the correct union mask remains finite. Measure the robust local
        # support diameter along the selected cyan path instead.
        mask = np.asarray(result.material.mask > 0, dtype=np.uint8)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        x, y, width, height = result.material.centerline.roi
        points = np.rint(
            result.material.points - np.array([x, y])).astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        radii = distance[points[:, 1], points[:, 0]]
        radii = radii[radii > 0.0]
        if len(radii):
            return float(2.0 * np.median(radii))
    length = float(cumulative_arclength(result.material.points)[-1])
    return float(result.mask_area_px / max(length, 1e-9))


def _centerline_outside_mask_fraction(result: SamMaterialResult) -> float:
    """Fraction of rounded cyan samples unsupported by the green mask."""
    mask = np.asarray(result.material.mask > 0, dtype=bool)
    x, y, width, height = result.material.centerline.roi
    points = np.rint(
        result.material.points - np.array([x, y])).astype(int)
    in_bounds = (
        (points[:, 0] >= 0) & (points[:, 0] < width)
        & (points[:, 1] >= 0) & (points[:, 1] < height))
    supported = np.zeros(len(points), dtype=bool)
    supported[in_bounds] = mask[
        points[in_bounds, 1], points[in_bounds, 0]]
    return float(1.0 - np.mean(supported))


def _marked_centerline_topology_metrics(
        result: SamMaterialResult) -> dict[str, float]:
    """Summarize marker order and separated sharp turns in one cyan path."""
    points = resample_arclength(
        result.material.points, 128, smooth_window=1)
    span = 4
    incoming = points[span:-span] - points[:-2 * span]
    outgoing = points[2 * span:] - points[span:-span]
    incoming /= np.maximum(
        np.linalg.norm(incoming, axis=1, keepdims=True), 1e-9)
    outgoing /= np.maximum(
        np.linalg.norm(outgoing, axis=1, keepdims=True), 1e-9)
    turn = np.degrees(np.arccos(np.clip(
        np.sum(incoming * outgoing, axis=1), -1.0, 1.0)))
    sharp = np.flatnonzero(turn >= 35.0)
    clusters = (
        np.split(sharp, np.flatnonzero(np.diff(sharp) > 8) + 1)
        if len(sharp) else [])
    marker_order_valid = float("nan")
    marker_max_distance = float("nan")
    observed_count = 0
    if (result.marker_centers_xy is not None
            and result.marker_observed is not None):
        markers = np.asarray(result.marker_centers_xy, dtype=np.float64)
        observed = np.asarray(result.marker_observed, dtype=bool)
        usable = observed & np.all(np.isfinite(markers), axis=1)
        observed_count = int(np.count_nonzero(usable))
        if observed_count:
            tree = cKDTree(points)
            distance, indices = tree.query(markers[usable])
            marker_max_distance = float(np.max(distance))
            marker_order_valid = float(
                np.all(np.diff(indices) > 0) if len(indices) >= 2 else True)
    return {
        "marker_interval_route_used": float(
            "+marker_interval_route" in result.prompt.source),
        "marker_path_order_valid": marker_order_valid,
        "marker_path_max_centroid_distance_px": marker_max_distance,
        "marker_path_observed_count": float(observed_count),
        "centerline_sharp_turn_cluster_count": float(len(clusters)),
        "centerline_max_local_turn_deg": (
            float(np.max(turn)) if len(turn) else 0.0),
    }


def _basic_segmentation_error(
        view: str,
        result: SamMaterialResult,
        roi: tuple[int, int, int, int],
        config: ImageProcessingConfig) -> str | None:
    marked_chromatic = result.prompt.source.startswith("marked_chromatic")
    source = "chromatic" if marked_chromatic else "sam"
    if len(result.material) < config.min_centerline_points:
        return f"centerline_{view}_too_short"
    if result.sam_iou < config.min_sam_iou:
        return f"{source}_{view}_low_iou:{result.sam_iou:.3f}"
    if result.seed_recall < config.min_seed_recall:
        return f"{source}_{view}_low_seed_recall:{result.seed_recall:.3f}"
    if result.mask_area_px > config.max_mask_area_fraction * roi[2] * roi[3]:
        return f"{source}_{view}_mask_too_large"
    effective_width = _mask_effective_width(result)
    maximum_width = (
        config.marked_max_mask_effective_width_px
        if marked_chromatic
        else config.max_mask_effective_width_px)
    if effective_width > maximum_width:
        return f"{source}_{view}_mask_too_wide:{effective_width:.1f}px"
    return None


def _stereo_guided_retry_result(
        result: SamMaterialResult,
        source: SamMaterialResult,
        target_image: np.ndarray,
        horizontal_shift_px: float,
        reference_width_px: float,
        maximum_width_px: float,
        disparity_prior_px: np.ndarray | None = None,
        allow_temporal_completion: bool = False) -> SamMaterialResult:
    """Recover a target path near the transferred opposite-view centerline.

    A SAM retry may contain the catheter but skeletonize through a much larger
    adjacent surface. Rectification gives the target path's image row at every
    source point. We therefore search only its horizontal coordinate, favoring
    blue/cyan pixels inside or next to SAM's mask, and smooth that disparity
    correction rather than accepting SAM's potentially branched skeleton.
    """
    material = result.material
    x, y, width, height = material.roi
    mask = material.mask
    guide = resample_arclength(source.material.points, 256, smooth_window=1)
    if disparity_prior_px is None:
        horizontal_offset = np.full(len(guide), float(horizontal_shift_px))
    else:
        prior = np.asarray(disparity_prior_px, dtype=np.float64)
        prior = np.interp(
            np.linspace(0.0, 1.0, len(guide)),
            np.linspace(0.0, 1.0, len(prior)), prior)
        direction = 1.0 if horizontal_shift_px >= 0.0 else -1.0
        horizontal_offset = direction * prior
    guide = guide.copy()
    guide[:, 0] += horizontal_offset
    guide[:, 0] = np.clip(guide[:, 0], x + 1, x + width - 2)
    guide[:, 1] = np.clip(guide[:, 1], y + 1, y + height - 2)

    crop = target_image[y:y + height, x:x + width]
    sub = crop.astype(np.float64)
    # Use the same class-specific score as direct marked segmentation.  A
    # generic blue-channel excess accepts demosaicing fringes on ChArUco edges.
    color_likelihood = catheter_color_likelihood(crop).astype(np.float64)
    hsv = cv2.cvtColor(
        crop, cv2.COLOR_BGR2HSV)
    yellow = (
        (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 48)
        & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 55))
    red = (
        ((hsv[:, :, 0] <= 14) | (hsv[:, :, 0] >= 166))
        & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 45))
    foreground = (color_likelihood >= 0.08) | yellow | red
    # The source eye supplies order, but the ill eye must still supply every
    # accepted pixel. Intersect with its independently generated green mask;
    # repeated target pixels remain legal when several material coordinates
    # overlap in this projection.
    target_support = np.asarray(mask) > 0
    appearance_score = 100.0 * color_likelihood
    local_guide = guide - np.array([x, y], dtype=np.float64)
    selected_x = np.empty(len(guide), dtype=np.float64)
    color_supported = np.zeros(len(guide), dtype=bool)
    # With a temporal disparity observation, catheter depth cannot jump forty
    # pixels in one video frame.  The narrower search prevents a ChArUco edge
    # from winning merely because it has stronger local contrast.
    search_x_px = 16 if disparity_prior_px is not None else 32
    search_y_px = 3
    for index, (guide_x, guide_y) in enumerate(local_guide):
        col = int(round(guide_x))
        row = int(round(guide_y))
        x0, x1 = max(0, col - search_x_px), min(width, col + search_x_px + 1)
        y0, y1 = max(0, row - search_y_px), min(height, row + search_y_px + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        chroma = appearance_score[y0:y1, x0:x1]
        local_foreground = target_support[y0:y1, x0:x1]
        color_supported[index] = bool(
            local_foreground.size and np.any(local_foreground))
        supported = local_foreground
        score = (
            1.5 * chroma
            - 0.6 * np.abs(xx - guide_x)
            - 2.0 * np.abs(yy - guide_y))
        score[~supported] = -np.inf
        if np.any(np.isfinite(score)):
            peak = float(np.max(score))
            ridge = np.isfinite(score) & (score >= peak - 15.0)
            weights = np.exp((score[ridge] - peak) / 10.0)
            selected_x[index] = float(np.average(xx[ridge], weights=weights))
        else:
            selected_x[index] = guide_x

    # A transferred curve is a search prior, not an observation. Require target
    # evidence throughout the path, not merely a good global fraction: otherwise
    # a proximal fragment can license a fabricated distal continuation.
    unsupported_runs = np.split(
        np.flatnonzero(~color_supported),
        np.where(np.diff(np.flatnonzero(~color_supported)) != 1)[0] + 1)
    longest_unsupported = max(
        (len(run) for run in unsupported_runs), default=0)
    support_fraction = float(np.mean(color_supported))
    hard_insufficient_support = bool(
        support_fraction < 0.60
        or longest_unsupported > int(np.ceil(0.22 * len(guide))))
    insufficient_support = bool(
        support_fraction < 0.80
        or longest_unsupported > int(np.ceil(0.08 * len(guide))))
    # Temporal completion may bridge a short antialiasing or red-ring gap, but
    # never a long target-eye region with no catheter-colored observation.
    if (hard_insufficient_support
            or (insufficient_support and not (
                allow_temporal_completion
                and disparity_prior_px is not None))):
        raise ValueError("stereo_guided_ridge_insufficient_color_support")

    # Depth/disparity varies smoothly along a continuous catheter, even across
    # a sharp in-plane bend. Smooth only the horizontal correction; preserve
    # the source curve's rectified epipolar rows and its kink geometry.
    x_offset = gaussian_filter1d(selected_x - local_guide[:, 0], sigma=2.0)
    guided_points = guide.copy()
    guided_points[:, 0] = np.clip(
        guide[:, 0] + x_offset, x + 1, x + width - 2)

    # Smoothing disparity can move a valid sample between two thin mask
    # branches. Snap each ordered sample back to the nearest current-frame
    # target support without enforcing uniqueness: self-overlap necessarily
    # maps multiple arc-length values to the same or neighboring pixels.
    support_rows, support_columns = np.where(target_support)
    if not len(support_columns):
        raise ValueError("stereo_guided_ridge_target_mask_empty")
    support_tree = cKDTree(np.column_stack([
        support_columns + x, support_rows + y]))
    snap_distance, snap_index = support_tree.query(guided_points, k=1)
    maximum_snap_px = 8.0 if disparity_prior_px is not None else 14.0
    snapped = np.column_stack([
        support_columns[snap_index] + x,
        support_rows[snap_index] + y]).astype(np.float64)
    snap_supported = (
        (snap_distance <= maximum_snap_px)
        & (np.abs(snapped[:, 1] - guide[:, 1]) <= search_y_px))
    if np.mean(snap_supported) < 0.90:
        raise ValueError("stereo_guided_ridge_too_far_from_target_mask")
    guided_points[snap_supported] = snapped[snap_supported]
    if np.any(~snap_supported):
        raise ValueError("stereo_guided_ridge_has_unsupported_samples")

    corridor = np.zeros_like(mask, dtype=np.uint8)
    local_points = np.rint(
        guided_points - np.array([x, y], dtype=np.float64)).astype(np.int32)
    tube_width = int(round(np.clip(
        1.25 * float(reference_width_px), 7.0, maximum_width_px)))
    cv2.polylines(
        corridor, [local_points], False, 255, tube_width, cv2.LINE_AA)
    cv2.circle(corridor, tuple(local_points[0]), max(1, tube_width // 2), 255, -1)
    cv2.circle(corridor, tuple(local_points[-1]), max(1, tube_width // 2), 255, -1)
    restricted = cv2.bitwise_and(np.uint8(mask), corridor)
    distance = cv2.distanceTransform(np.uint8(restricted > 0), cv2.DIST_L2, 5)
    rows = np.clip(local_points[:, 1], 0, mask.shape[0] - 1)
    cols = np.clip(local_points[:, 0], 0, mask.shape[1] - 1)
    radius = float(np.median(distance[rows, cols]))
    centerline = Centerline(
        guided_points, material.roi, restricted, radius)
    brightness = _sample_brightness(target_image, guided_points)
    boundary, confidence, contrast, valid = detect_distal_boundary(brightness)
    updated_material = replace(
        material,
        centerline=centerline,
        distal_boundary_index=int(boundary),
        distal_boundary_fraction=float(boundary / max(len(guided_points) - 1, 1)),
        boundary_confidence=float(confidence),
        boundary_contrast=float(contrast),
        material_valid=bool(valid),
        brightness_profile=brightness)
    prompt_indices = np.unique(np.rint(
        np.linspace(0, len(guided_points) - 1, 8)).astype(int))
    positive = guided_points[prompt_indices]
    prompt_margin = max(12.0, float(tube_width))
    prompt_box = np.array([
        max(x, float(np.min(guided_points[:, 0]) - prompt_margin)),
        max(y, float(np.min(guided_points[:, 1]) - prompt_margin)),
        min(x + width - 1,
            float(np.max(guided_points[:, 0]) + prompt_margin)),
        min(y + height - 1,
            float(np.max(guided_points[:, 1]) + prompt_margin)),
    ])
    negative = np.array([
        [prompt_box[0], prompt_box[1]], [prompt_box[2], prompt_box[1]],
        [prompt_box[0], prompt_box[3]], [prompt_box[2], prompt_box[3]],
    ], dtype=np.float64)
    guided_prompt = PromptSet(
        box_xyxy=prompt_box,
        positive_xy=positive,
        negative_xy=negative,
        source="stereo_guided_ridge")
    return replace(
        result,
        material=updated_material,
        prompt=guided_prompt,
        mask_area_px=int(np.count_nonzero(restricted)),
        yellow_tip_xy=result.yellow_tip_xy,
        marker_centers_xy=result.marker_centers_xy,
        marker_widths_px=result.marker_widths_px,
        marker_confidence=result.marker_confidence,
        marker_observed=result.marker_observed,
        tip_source=(
            "stereo_temporal_inferred" if insufficient_support
            else result.tip_source))


def _iter_batched_sam_frames(
        record_iterator,
        segmenter: Sam2CatheterSegmenter,
        prompts: dict[str, dict[int, PromptSet]],
        left_roi: tuple[int, int, int, int],
        right_roi: tuple[int, int, int, int],
        base_left: np.ndarray,
        base_right: np.ndarray,
        frame_batch_size: int,
        prompt_workers: int):
    """Decode bounded chunks and batch frame-local SAM work.

    Automatic colour prompts do not depend on temporal state, so both views of
    several frames are prepared concurrently before one GPU encoder call.  Any
    missing/failed view is returned as an exception for the chronological
    reducer to retry from its preceding accepted frame.
    """
    batch_size = max(1, int(frame_batch_size))
    workers = max(1, int(prompt_workers))
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        while True:
            decode_started = time.perf_counter()
            decoded = []
            for _ in range(batch_size):
                try:
                    decoded.append(next(record_iterator))
                except StopIteration:
                    break
            if not decoded:
                return
            decode_per_frame = (
                time.perf_counter() - decode_started) / len(decoded)

            images = []
            rois = []
            bases = []
            overrides = []
            for record, _, left_image, right_image in decoded:
                images.extend([left_image, right_image])
                rois.extend([left_roi, right_roi])
                bases.extend([base_left, base_right])
                overrides.extend([
                    prompts["left"].get(record.svo_frame),
                    prompts["right"].get(record.svo_frame)])

            prompt_started = time.perf_counter()
            futures = [
                executor.submit(_automatic_prompt_result, image, roi, base)
                for image, roi, base in zip(images, rois, bases)]
            automatic: list[AutomaticPromptResult | Exception] = []
            for future in futures:
                try:
                    automatic.append(future.result())
                except (ArithmeticError, RuntimeError, ValueError) as exc:
                    automatic.append(exc)
            prompt_wall = time.perf_counter() - prompt_started

            results: list[SamMaterialResult | Exception] = [
                ValueError("automatic_prompt_missing") for _ in images]
            valid_indices = [
                index for index, (auto, override) in enumerate(
                    zip(automatic, overrides))
                if not isinstance(auto, Exception) or override is not None]
            sam_timing: dict[str, float] = {}
            if valid_indices:
                selected = segmenter.segment_batch(
                    [images[i] for i in valid_indices],
                    [rois[i] for i in valid_indices],
                    [bases[i] for i in valid_indices],
                    [overrides[i] for i in valid_indices],
                    [automatic[i] for i in valid_indices],
                    allow_finish_errors=True)
                for target, result in zip(valid_indices, selected):
                    results[target] = result
                sam_timing = dict(segmenter.last_timing_s)
            # _prepare still performs image conversion and prompt tensor
            # assembly; include concurrent colour processing in the same stage.
            sam_timing["sam_prompt_preprocess"] = (
                sam_timing.get("sam_prompt_preprocess", 0.0) + prompt_wall)
            timing_per_frame = {
                key: value / len(decoded) for key, value in sam_timing.items()}
            timing_per_frame["svo_decode"] = decode_per_frame
            for local_index, decoded_frame in enumerate(decoded):
                yield (
                    *decoded_frame,
                    results[2 * local_index],
                    results[2 * local_index + 1],
                    dict(timing_per_frame))
    finally:
        executor.shutdown(wait=True)


def _iter_preprocessed_frames(
        record_iterator,
        left_roi: tuple[int, int, int, int],
        right_roi: tuple[int, int, int, int],
        base_left: np.ndarray,
        base_right: np.ndarray,
        chunk_size: int,
        prompt_workers: int):
    """Prepare independent colour prompts in bounded CPU-parallel chunks."""
    count = max(1, int(chunk_size))
    executor = ThreadPoolExecutor(max_workers=max(1, int(prompt_workers)))
    try:
        while True:
            decode_started = time.perf_counter()
            decoded = []
            for _ in range(count):
                try:
                    decoded.append(next(record_iterator))
                except StopIteration:
                    break
            if not decoded:
                return
            decode_per_frame = (
                time.perf_counter() - decode_started) / len(decoded)
            tasks = []
            for _, _, left_image, right_image in decoded:
                tasks.extend([
                    (left_image, left_roi, base_left),
                    (right_image, right_roi, base_right)])
            prompt_started = time.perf_counter()
            futures = [executor.submit(_automatic_prompt_result, *task)
                       for task in tasks]
            automatic: list[AutomaticPromptResult | Exception] = []
            for future in futures:
                try:
                    automatic.append(future.result())
                except (ArithmeticError, RuntimeError, ValueError) as exc:
                    automatic.append(exc)
            prompt_per_frame = (
                time.perf_counter() - prompt_started) / len(decoded)
            for local_index, decoded_frame in enumerate(decoded):
                yield (
                    *decoded_frame,
                    automatic[2 * local_index],
                    automatic[2 * local_index + 1],
                    {"svo_decode": decode_per_frame,
                     "sam_prompt_preprocess": prompt_per_frame})
    finally:
        executor.shutdown(wait=True)


def _estimate_chromatic_backgrounds(
        session: Path,
        registration,
        sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the fixed scene from sparse samples of the complete SVO.

    The camera and light box are fixed, while the catheter moves.  A temporal
    median therefore retains the ChArUco board (including its colored edge
    fringes) but removes the moving catheter at almost every pixel.
    """
    all_records = load_frame_index(session)
    if not all_records:
        raise ValueError("cannot estimate background without SVO frame records")
    count = int(np.clip(int(sample_count), 5, len(all_records)))
    indices = np.unique(np.rint(np.linspace(
        0, len(all_records) - 1, count)).astype(int))
    left_crops: list[np.ndarray] = []
    right_crops: list[np.ndarray] = []
    left_roi = registration.roi_left_xywh
    right_roi = registration.roi_right_xywh
    with SvoReader(find_svo(session)) as reader:
        for index in indices:
            _, left, right = reader.read(all_records[int(index)].svo_frame)
            for image, roi, output in (
                    (left, left_roi, left_crops),
                    (right, right_roi, right_crops)):
                x, y, width, height = roi
                output.append(image[y:y + height, x:x + width].copy())

    def median_uint8(images: list[np.ndarray]) -> np.ndarray:
        stack = np.stack(images, axis=0)
        middle = len(stack) // 2
        # Integer order statistics avoid np.median's much larger float64
        # temporary for two HD regions of interest.
        return np.partition(stack, middle, axis=0)[middle]

    return median_uint8(left_crops), median_uint8(right_crops)


def _projected_workspace_roi_mask(
        registration, right: bool,
        sheath_exclusion_radius_mm: float = 0.0,
        sheath_exclusion_length_mm: float = 0.0) -> np.ndarray:
    """Return workspace silhouette minus the proximal base-z sheath."""
    workspace = registration.workspace_base_m
    corners_mm = 1000.0 * np.asarray([
        [x, y, z]
        for x in workspace["x"]
        for y in workspace["y"]
        for z in workspace["z"]
    ], dtype=np.float64)
    transform = (
        registration.right_camera_T_base
        if right else registration.left_camera_T_base)
    pixels, in_front = project_points(registration.K, transform, corners_mm)
    pixels = pixels[in_front & np.all(np.isfinite(pixels), axis=1)]
    if len(pixels) < 3:
        raise ValueError("workspace projection has fewer than three visible corners")
    roi = registration.roi_right_xywh if right else registration.roi_left_xywh
    x, y, width, height = roi
    local = np.rint(pixels - [x, y]).astype(np.int32)
    hull = cv2.convexHull(local)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_8)
    radius = max(float(sheath_exclusion_radius_mm), 0.0)
    length = max(float(sheath_exclusion_length_mm), 0.0)
    if radius > 0.0 and length > 0.0:
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        # The catheter workspace begins at base z=0 and extends toward +z.
        # The rigid sheath is proximal, along -z.  Project a short physical
        # cylinder rather than guessing a camera-specific image rectangle.
        cylinder_mm = np.asarray([
            [radius * np.cos(angle), radius * np.sin(angle), z]
            for z in (-length, -0.5)
            for angle in angles
        ], dtype=np.float64)
        sheath_pixels, sheath_front = project_points(
            registration.K, transform, cylinder_mm)
        sheath_pixels = sheath_pixels[
            sheath_front & np.all(np.isfinite(sheath_pixels), axis=1)]
        if len(sheath_pixels) >= 3:
            sheath_local = np.rint(
                sheath_pixels - [x, y]).astype(np.int32)
            sheath_hull = cv2.convexHull(sheath_local)
            cv2.fillConvexPoly(mask, sheath_hull, 0, cv2.LINE_8)
    return mask


class _ObservationsOnlyFrame(Exception):
    """Internal control flow after a frame-local observation is committed."""


def process_image_session(
        session_path: os.PathLike | str,
        sam_checkpoint: os.PathLike | str | None,
        sam_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        output_dir: os.PathLike | str | None = None,
        prompt_json: os.PathLike | str | None = None,
        config: ImageProcessingConfig | None = None,
        window: str = "run_and_return",
        start_ns: int | None = None,
        end_ns: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
        write_video: bool = False,
        write_snapshots: bool = False,
        snapshot_count: int = 12,
        snapshot_frames: list[int] | None = None,
        device: str = "cuda",
        segmentation_backend: str = "sam",
        reconstruction_backend: str = "disparity",
        propagated_mask_h5: os.PathLike | str | None = None,
        cached_mask_h5: os.PathLike | str | None = None,
        observations_only: bool = False) -> dict:
    initialization_started = time.perf_counter()
    config = config or ImageProcessingConfig()
    session = Path(session_path).resolve()
    output = (
        Path(output_dir).resolve() if output_dir is not None
        else session / "processed_image")
    output.mkdir(parents=True, exist_ok=True)
    registration = load_session_registration(session, require_em=False)
    markers = load_collection_markers(session)
    records = select_frame_records(
        load_frame_index(session), markers, window, start_ns, end_ns,
        stride, max_frames)
    snapshot_count = int(max(1, snapshot_count))
    if snapshot_frames:
        snapshot_indices = {
            int(index) for index in snapshot_frames
            if 0 <= int(index) < len(records)}
    else:
        snapshot_indices = set(np.unique(np.rint(np.linspace(
            0, len(records) - 1, min(snapshot_count, len(records))
        )).astype(int)).tolist())
    query_ns = np.asarray([record.timestamp_ns for record in records], dtype=np.int64)
    robot = align_robot_streams(
        load_robot_streams(session), query_ns,
        config.max_command_age_ms, config.max_feedback_gap_ms)
    prompts = load_prompt_overrides(prompt_json)
    if segmentation_backend not in (
            "sam", "propagated", "cached", "hsv", "chromatic_markers"):
        raise ValueError(
            "segmentation_backend must be sam, propagated, cached, hsv, "
            "or chromatic_markers")
    if reconstruction_backend not in ("disparity", "joint_spline"):
        raise ValueError(
            "reconstruction_backend must be disparity or joint_spline")
    if observations_only and segmentation_backend != "chromatic_markers":
        raise ValueError(
            "--observations-only currently requires chromatic_markers")
    if observations_only and not config.store_masks:
        raise ValueError("--observations-only requires --store-masks")
    if segmentation_backend == "propagated" and propagated_mask_h5 is None:
        raise ValueError("propagated backend requires propagated_mask_h5")
    if segmentation_backend == "cached" and cached_mask_h5 is None:
        raise ValueError("cached backend requires cached_mask_h5")
    if (segmentation_backend == "cached"
            and Path(cached_mask_h5).resolve()
            == (output / "processed_shapes.h5").resolve()):
        raise ValueError(
            "cached reconstruction output must differ from its source HDF5; "
            "pass --outdir")

    median_period_s = (
        float(np.median(np.diff(query_ns))) * 1e-9 if len(query_ns) > 1 else 1 / 30)
    output_fps = float(np.clip(1.0 / max(median_period_s, 1e-6), 1.0, 60.0))
    defer_final_overlay = bool(not observations_only and
        (write_video or write_snapshots) and (
            config.interface_offline_cutoff_hz > 0.0
            or config.stereo_offline_cutoff_hz > 0.0)
        and (config.store_masks
             or segmentation_backend in ("cached", "propagated")))
    if segmentation_backend == "sam" and sam_checkpoint is None:
        raise ValueError("SAM backend requires --sam-checkpoint")
    checkpoint = (
        None if sam_checkpoint is None else Path(sam_checkpoint).resolve())
    metadata = {
        "session": session.name,
        "session_path": str(session),
        "svo_path": str(find_svo(session)),
        "registration_path": str(registration.path),
        "window": window,
        "start_ns": int(query_ns[0]),
        "end_ns": int(query_ns[-1]),
        "stride": int(stride),
        "source_frame_count": len(records),
        "sam": (
            None if checkpoint is None else
            _sam_provenance(checkpoint, sam_config)),
        "prompt_json": (
            None if prompt_json is None else str(Path(prompt_json).resolve())),
        "device": device,
        "segmentation_backend": segmentation_backend,
        "reconstruction_backend": reconstruction_backend,
        "observations_only": bool(observations_only),
        "write_snapshots": bool(write_snapshots),
        "snapshot_count": int(snapshot_count),
        "snapshot_frames": sorted(snapshot_indices),
        "propagated_mask_h5": (
            None if propagated_mask_h5 is None
            else str(Path(propagated_mask_h5).resolve())),
        "cached_mask_h5": (
            None if cached_mask_h5 is None
            else str(Path(cached_mask_h5).resolve())),
        "collection_markers": markers,
        "code_provenance": _repository_provenance(),
    }
    with (output / "processing_config.json").open("w", encoding="utf-8") as stream:
        json.dump({"metadata": metadata, "config": asdict(config)}, stream,
                  indent=2, sort_keys=True)

    segmenter = (
        Sam2CatheterSegmenter(
            sam_config, checkpoint, device=device,
            postprocess_workers=config.sam_postprocess_workers)
        if segmentation_backend == "sam" else None)
    propagated_cache = (
        PropagatedMaskCache(
            propagated_mask_h5 if segmentation_backend == "propagated"
            else cached_mask_h5, records)
        if segmentation_backend in ("propagated", "cached") else None)
    marker_backend_active = bool(
        segmentation_backend == "chromatic_markers"
        or (propagated_cache is not None
            and propagated_cache.layout == "processed_image"
            and "images/left/marker_centers_px" in propagated_cache.file))
    nominal_distal_length_mm = (
        config.marked_distal_length_mm
        if marker_backend_active else config.distal_length_mm)
    chromatic_background_left = chromatic_background_right = None
    workspace_mask_left = workspace_mask_right = None
    if (segmentation_backend == "chromatic_markers"
            and config.chromatic_background_subtraction):
        chromatic_background_left, chromatic_background_right = (
            _estimate_chromatic_backgrounds(
                session, registration, config.chromatic_background_samples))
    if segmentation_backend == "chromatic_markers":
        workspace_mask_left = _projected_workspace_roi_mask(
            registration, right=False,
            sheath_exclusion_radius_mm=(
                config.marked_sheath_exclusion_radius_mm),
            sheath_exclusion_length_mm=(
                config.marked_sheath_exclusion_length_mm))
        workspace_mask_right = _projected_workspace_roi_mask(
            registration, right=True,
            sheath_exclusion_radius_mm=(
                config.marked_sheath_exclusion_radius_mm),
            sheath_exclusion_length_mm=(
                config.marked_sheath_exclusion_length_mm))
    writer = ImageSequenceWriter(
        output / "processed_shapes.h5", len(records), config,
        registration, metadata)
    overlay = OverlayWriter(
        output, output_fps, config.video_scale,
        video_enabled=write_video and not defer_final_overlay,
        snapshots_enabled=write_snapshots and not defer_final_overlay,
        snapshot_indices=snapshot_indices)
    base_mm = np.zeros(3, dtype=np.float64)
    base_left = project_points(
        registration.K, registration.left_camera_T_base, base_mm)[0][0]
    base_right = project_points(
        registration.K, registration.right_camera_T_base, base_mm)[0][0]
    registered_base_disparity_px = float(base_left[0] - base_right[0])
    base_camera_m = _camera_point_from_base(
        registration.left_camera_T_base, base_mm)
    summary_rows = []
    counts: dict[str, int] = {}
    timing_totals_s: dict[str, float] = {}
    initialization_s = time.perf_counter() - initialization_started
    started = time.perf_counter()
    previous_valid_results: tuple[SamMaterialResult, SamMaterialResult] | None = None
    previous_valid_timestamp_ns: int | None = None
    previous_stereo_reference_view: str | None = None
    active_ill_view: str | None = None
    pending_ill_view: str | None = None
    pending_ill_view_count = 0
    previous_fitted_disparity_px: np.ndarray | None = None
    previous_reconstructed_camera_m: np.ndarray | None = None
    previous_interface_base_mm: np.ndarray | None = None
    previous_distal_length_mm: float | None = None
    previous_filtered_distal_points: np.ndarray | None = None
    previous_joint_turn_fraction: float | None = None
    background_prefetch: _BackgroundPrefetch | None = None
    chromatic_executor = (
        ThreadPoolExecutor(max_workers=config.chromatic_eye_workers)
        if (segmentation_backend == "chromatic_markers"
            and config.chromatic_eye_workers > 1) else None)
    interface_smoothing_summary: dict[str, float] = {}
    stereo_smoothing_summary: dict[str, float] = {}
    spline_temporal_smoothing_summary: dict[str, float] = {}

    try:
        svo_open_started = time.perf_counter()
        with SvoReader(find_svo(session)) as svo:
            timing_totals_s["svo_open"] = (
                time.perf_counter() - svo_open_started)
            record_iterator = iter(svo.iter_records(records))
            if segmentation_backend == "sam":
                if config.sam_frame_batch_size > 1:
                    frame_iterator = _iter_batched_sam_frames(
                        record_iterator, segmenter, prompts,
                        registration.roi_left_xywh,
                        registration.roi_right_xywh,
                        base_left, base_right,
                        config.sam_frame_batch_size, config.prompt_workers)
                else:
                    preprocessed = _iter_preprocessed_frames(
                        record_iterator,
                        registration.roi_left_xywh,
                        registration.roi_right_xywh,
                        base_left, base_right,
                        config.preprocess_chunk_size,
                        config.prompt_workers)
                    if config.prefetch_frames > 0:
                        background_prefetch = _BackgroundPrefetch(
                            preprocessed, config.prefetch_frames)
                        frame_iterator = background_prefetch
                    else:
                        frame_iterator = preprocessed
            else:
                def decoded_frames():
                    while True:
                        decode_started = time.perf_counter()
                        try:
                            item = next(record_iterator)
                        except StopIteration:
                            return
                        yield (*item, None, None, {
                            "svo_decode": time.perf_counter() - decode_started})
                decoded = decoded_frames()
                if config.prefetch_frames > 0:
                    background_prefetch = _BackgroundPrefetch(
                        decoded, config.prefetch_frames)
                    frame_iterator = background_prefetch
                else:
                    frame_iterator = decoded
            for index in range(len(records)):
                try:
                    (record, svo_timestamp_ns, left_image, right_image,
                     prefetched_left, prefetched_right,
                     frame_timing_s) = next(frame_iterator)
                except StopIteration:
                    break
                frame_started = time.perf_counter()
                prefetched_time_s = sum(frame_timing_s.values())

                stage_started = time.perf_counter()
                writer.write_identity(index, record, svo_timestamp_ns, robot)
                frame_timing_s["hdf5_write"] = (
                    time.perf_counter() - stage_started)
                status = "valid"
                metrics: dict[str, float | int] = {}
                left_result = right_result = None
                full_geometry = distal_geometry = assembled = None
                joint_reconstruction = None
                try:
                    manual_left = prompts["left"].get(record.svo_frame)
                    manual_right = prompts["right"].get(record.svo_frame)
                    explicit_manual_left = manual_left is not None
                    explicit_manual_right = manual_right is not None
                    temporal_gap_ms = (
                        np.inf if previous_valid_timestamp_ns is None
                        else (record.timestamp_ns - previous_valid_timestamp_ns) / 1e6)
                    if (previous_valid_results is not None
                            and 0.0 < temporal_gap_ms
                            <= config.max_temporal_prompt_gap_ms):
                        if manual_left is None:
                            manual_left = _centerline_prompt(
                                previous_valid_results[0],
                                registration.roi_left_xywh,
                                source="temporal_previous_valid")
                        if manual_right is None:
                            manual_right = _centerline_prompt(
                                previous_valid_results[1],
                                registration.roi_right_xywh,
                                source="temporal_previous_valid")
                    if segmentation_backend == "sam":
                        if config.sam_frame_batch_size <= 1:
                            try:
                                sequential_results = segmenter.segment_batch(
                                    [left_image, right_image],
                                    [registration.roi_left_xywh,
                                     registration.roi_right_xywh],
                                    [base_left, base_right],
                                    [manual_left, manual_right],
                                    [prefetched_left, prefetched_right])
                                left_result, right_result = sequential_results
                            finally:
                                for key, value in segmenter.last_timing_s.items():
                                    frame_timing_s[key] = (
                                        frame_timing_s.get(key, 0.0) + value)
                        else:
                            left_result, right_result = (
                                prefetched_left, prefetched_right)
                        if (isinstance(left_result, Exception)
                                or isinstance(right_result, Exception)):
                            if manual_left is None and manual_right is None:
                                error = (
                                    left_result if isinstance(
                                        left_result, Exception)
                                    else right_result)
                                raise error
                            try:
                                retry_results = segmenter.segment_batch(
                                    [left_image, right_image],
                                    [registration.roi_left_xywh,
                                     registration.roi_right_xywh],
                                    [base_left, base_right],
                                    [manual_left, manual_right])
                                left_result, right_result = retry_results
                            finally:
                                for key, value in segmenter.last_timing_s.items():
                                    frame_timing_s[key] = (
                                        frame_timing_s.get(key, 0.0) + value)
                    elif segmentation_backend in ("propagated", "cached"):
                        left_result = propagated_cache.result(
                            index, "left", left_image,
                            registration.roi_left_xywh, base_left)
                        right_result = propagated_cache.result(
                            index, "right", right_image,
                            registration.roi_right_xywh, base_right)
                    elif segmentation_backend == "chromatic_markers":
                        chromatic_started = time.perf_counter()
                        previous_left_points = previous_right_points = None
                        if (previous_valid_results is not None
                                and 0.0 < temporal_gap_ms
                                <= config.max_temporal_prompt_gap_ms):
                            previous_left_points = (
                                previous_valid_results[0].material.points)
                            previous_right_points = (
                                previous_valid_results[1].material.points)
                        extraction_arguments = (
                            (left_image, registration.roi_left_xywh, base_left,
                             previous_left_points, chromatic_background_left,
                             workspace_mask_left),
                            (right_image, registration.roi_right_xywh, base_right,
                             previous_right_points, chromatic_background_right,
                             workspace_mask_right))

                        def extract_eye(arguments):
                            image, roi, base, previous, background, workspace = arguments
                            return extract_marked_chromatic_result(
                                image, roi, base,
                                config.chromatic_min_saturation,
                                config.chromatic_min_value,
                                previous_points_xy=previous,
                                background_bgr=background,
                                minimum_background_difference=(
                                    config.chromatic_background_difference),
                                workspace_mask=workspace)

                        if chromatic_executor is None:
                            left_result, right_result = map(
                                extract_eye, extraction_arguments)
                        else:
                            left_future = chromatic_executor.submit(
                                extract_eye, extraction_arguments[0])
                            right_future = chromatic_executor.submit(
                                extract_eye, extraction_arguments[1])
                            left_result, right_result = (
                                left_future.result(), right_future.result())
                        left_result, right_result = refine_marked_stereo_pair(
                            left_image, right_image,
                            left_result, right_result,
                            registration.roi_left_xywh,
                            registration.roi_right_xywh,
                            registered_base_disparity_px,
                            epipolar_half_width_px=max(
                                10.0, config.max_marker_epipolar_error_px),
                            previous_left_result=(
                                None if previous_left_points is None
                                else previous_valid_results[0]),
                            previous_right_result=(
                                None if previous_right_points is None
                                else previous_valid_results[1]))
                        frame_timing_s["chromatic_segmentation"] = (
                            time.perf_counter() - chromatic_started)
                    else:
                        left_prompt, left_seed, _ = _automatic_prompts(
                            left_image, registration.roi_left_xywh, base_left)
                        right_prompt, right_seed, _ = _automatic_prompts(
                            right_image, registration.roi_right_xywh, base_right)
                        left_result = _external_mask_result(
                            left_image, registration.roi_left_xywh, base_left,
                            left_seed, "hsv_baseline")
                        right_result = _external_mask_result(
                            right_image, registration.roi_right_xywh, base_right,
                            right_seed, "hsv_baseline")
                    reroute_marked_observation = bool(
                        marker_backend_active
                        and (segmentation_backend != "cached"
                             or propagated_cache is None
                             or propagated_cache.layout != "processed_image"
                             or propagated_cache.marker_track_requires_reroute(
                                 index)))
                    if reroute_marked_observation:
                        marker_routing_started = time.perf_counter()
                        previous_left_points = previous_right_points = None
                        if (previous_valid_results is not None
                                and 0.0 < temporal_gap_ms
                                <= config.max_temporal_prompt_gap_ms):
                            previous_left_points = (
                                previous_valid_results[0].material.points)
                            previous_right_points = (
                                previous_valid_results[1].material.points)
                        def reroute_eye(result, image, roi, previous):
                            return reroute_marked_centerline(
                                result, image, roi,
                                previous_points_xy=previous,
                                minimum_marker_confidence=(
                                    config.marker_min_confidence))

                        if chromatic_executor is None:
                            left_result = reroute_eye(
                                left_result, left_image,
                                registration.roi_left_xywh,
                                previous_left_points)
                            right_result = reroute_eye(
                                right_result, right_image,
                                registration.roi_right_xywh,
                                previous_right_points)
                        else:
                            left_future = chromatic_executor.submit(
                                reroute_eye, left_result, left_image,
                                registration.roi_left_xywh,
                                previous_left_points)
                            right_future = chromatic_executor.submit(
                                reroute_eye, right_result, right_image,
                                registration.roi_right_xywh,
                                previous_right_points)
                            left_result, right_result = (
                                left_future.result(), right_future.result())
                        left_result, right_result, sweep_metrics = (
                            enforce_stereo_epipolar_sweep(
                                left_result, right_result,
                                left_image, right_image,
                                registration.roi_left_xywh,
                                registration.roi_right_xywh,
                                base_left, base_right,
                                previous_left_points_xy=previous_left_points,
                                previous_right_points_xy=previous_right_points,
                                minimum_marker_confidence=(
                                    config.marker_min_confidence),
                                minimum_sweep_deficit_px=(
                                    config.marked_epipolar_sweep_deficit_px),
                                row_half_width_px=(
                                    config.marked_epipolar_sweep_row_half_width_px)))
                        metrics.update(sweep_metrics)
                        # Local normal centering is independent of marker
                        # availability. Rings select order/topology; they must
                        # not decide whether a fallback route receives basic
                        # medial refinement.
                        if chromatic_executor is None:
                            left_result = center_marked_route_on_mask(left_result)
                            right_result = center_marked_route_on_mask(right_result)
                        else:
                            left_future = chromatic_executor.submit(
                                center_marked_route_on_mask, left_result)
                            right_future = chromatic_executor.submit(
                                center_marked_route_on_mask, right_result)
                            left_result, right_result = (
                                left_future.result(), right_future.result())
                        for view, result in (
                                ("left", left_result),
                                ("right", right_result)):
                            topology = _marked_centerline_topology_metrics(
                                result)
                            metrics.update({
                                f"marker_interval_route_used_{view}": (
                                    topology["marker_interval_route_used"]),
                                f"marker_path_order_valid_{view}": (
                                    topology["marker_path_order_valid"]),
                                f"marker_path_max_centroid_distance_{view}_px": (
                                    topology[
                                        "marker_path_max_centroid_distance_px"]),
                                f"marker_path_observed_count_{view}": (
                                    topology["marker_path_observed_count"]),
                                f"centerline_sharp_turn_cluster_count_{view}": (
                                    topology[
                                        "centerline_sharp_turn_cluster_count"]),
                                f"centerline_max_local_turn_{view}_deg": (
                                    topology["centerline_max_local_turn_deg"]),
                            })
                        frame_timing_s["marker_routing"] = (
                            time.perf_counter() - marker_routing_started)
                    frame_timing_s.setdefault("sam_mask_postprocess", 0.0)

                    # Batched inference uses frame-local prompts so it is not
                    # serialized by the preceding frame.  Preserve the former
                    # temporal behavior as a selective rescue only when the
                    # initial mask fails local QC.
                    initial_errors = [
                        _basic_segmentation_error(
                            "left", left_result,
                            registration.roi_left_xywh, config),
                        _basic_segmentation_error(
                            "right", right_result,
                            registration.roi_right_xywh, config)]
                    can_temporal_retry = (
                        segmentation_backend == "sam"
                        and config.sam_frame_batch_size > 1
                        and previous_valid_results is not None
                        and 0.0 < temporal_gap_ms
                        <= config.max_temporal_prompt_gap_ms
                        and (not explicit_manual_left
                             or not explicit_manual_right))
                    if any(initial_errors) and can_temporal_retry:
                        try:
                            retry_results = segmenter.segment_batch(
                                [left_image, right_image],
                                [registration.roi_left_xywh,
                                 registration.roi_right_xywh],
                                [base_left, base_right],
                                [manual_left, manual_right])
                            retry_errors = [
                                _basic_segmentation_error(
                                    "left", retry_results[0],
                                    registration.roi_left_xywh, config),
                                _basic_segmentation_error(
                                    "right", retry_results[1],
                                    registration.roi_right_xywh, config)]
                            if sum(error is not None for error in retry_errors) < sum(
                                    error is not None for error in initial_errors):
                                left_result, right_result = retry_results
                                initial_errors = retry_errors
                        except (RuntimeError, ValueError):
                            pass
                        finally:
                            for key, value in segmenter.last_timing_s.items():
                                frame_timing_s[key] = (
                                    frame_timing_s.get(key, 0.0) + value)

                    stage_started = time.perf_counter()
                    for view, result, roi in (
                            ("left", left_result, registration.roi_left_xywh),
                            ("right", right_result, registration.roi_right_xywh)):
                        error = _basic_segmentation_error(
                            view, result, roi, config)
                        if error is not None:
                            raise ValueError(error)
                    left_length_px = float(cumulative_arclength(
                        left_result.material.points)[-1])
                    right_length_px = float(cumulative_arclength(
                        right_result.material.points)[-1])
                    left_outside_mask_fraction = (
                        _centerline_outside_mask_fraction(left_result))
                    right_outside_mask_fraction = (
                        _centerline_outside_mask_fraction(right_result))
                    if (previous_valid_results is not None
                            and 0.0 < temporal_gap_ms
                            <= config.max_temporal_prompt_gap_ms):
                        left_coverage, left_temporal_p95 = (
                            _temporal_centerline_metrics(
                                previous_valid_results[0].material.points,
                                left_result.material.points,
                                config.temporal_centerline_tolerance_px))
                        right_coverage, right_temporal_p95 = (
                            _temporal_centerline_metrics(
                                previous_valid_results[1].material.points,
                                right_result.material.points,
                                config.temporal_centerline_tolerance_px))
                        metrics.update({
                            "temporal_centerline_coverage_left": left_coverage,
                            "temporal_centerline_coverage_right": right_coverage,
                            "temporal_centerline_p95_left_px": left_temporal_p95,
                            "temporal_centerline_p95_right_px": right_temporal_p95,
                        })
                    centerline_length_ratio = max(
                        left_length_px, right_length_px) / max(
                            min(left_length_px, right_length_px), 1e-9)
                    initial_centerline_length_ratio = centerline_length_ratio
                    if observations_only:
                        metrics.update({
                            "centerline_length_left_px": left_length_px,
                            "centerline_length_right_px": right_length_px,
                            "centerline_outside_mask_fraction_left": (
                                left_outside_mask_fraction),
                            "centerline_outside_mask_fraction_right": (
                                right_outside_mask_fraction),
                            "stereo_centerline_length_ratio_initial": (
                                initial_centerline_length_ratio),
                            "stereo_centerline_length_ratio": (
                                centerline_length_ratio),
                            "material_boundary_fraction": float(np.mean([
                                left_result.material.distal_boundary_fraction,
                                right_result.material.distal_boundary_fraction])),
                        })
                        frame_timing_s["quality_control"] = (
                            time.perf_counter() - stage_started)
                        writer.write_observations(
                            index, left_result, right_result, metrics)
                        previous_valid_results = (left_result, right_result)
                        previous_valid_timestamp_ns = record.timestamp_ns
                        status = "observations_only"
                        raise _ObservationsOnlyFrame()
                    if (centerline_length_ratio
                            > config.max_stereo_centerline_length_ratio
                            and can_temporal_retry):
                        try:
                            temporal_results = segmenter.segment_batch(
                                [left_image, right_image],
                                [registration.roi_left_xywh,
                                 registration.roi_right_xywh],
                                [base_left, base_right],
                                [manual_left, manual_right])
                            temporal_errors = [
                                _basic_segmentation_error(
                                    "left", temporal_results[0],
                                    registration.roi_left_xywh, config),
                                _basic_segmentation_error(
                                    "right", temporal_results[1],
                                    registration.roi_right_xywh, config)]
                            if not any(temporal_errors):
                                temporal_left_length = float(
                                    cumulative_arclength(
                                        temporal_results[0].material.points)[-1])
                                temporal_right_length = float(
                                    cumulative_arclength(
                                        temporal_results[1].material.points)[-1])
                                temporal_ratio = max(
                                    temporal_left_length,
                                    temporal_right_length) / max(
                                        min(temporal_left_length,
                                            temporal_right_length), 1e-9)
                                if temporal_ratio < centerline_length_ratio:
                                    left_result, right_result = temporal_results
                                    left_length_px = temporal_left_length
                                    right_length_px = temporal_right_length
                                    centerline_length_ratio = temporal_ratio
                        except (RuntimeError, ValueError):
                            pass
                        finally:
                            for key, value in segmenter.last_timing_s.items():
                                frame_timing_s[key] = (
                                    frame_timing_s.get(key, 0.0) + value)
                    frame_timing_s["quality_control"] = (
                        time.perf_counter() - stage_started)
                    stereo_retry_view = 0
                    stereo_retry_attempted = 0
                    # 0: not attempted, 1: accepted, 2: ridge construction
                    # failed, 3: candidate failed quality, 4: no geometric
                    # improvement.  Numeric codes keep the buffered HDF5 path
                    # compact while making failed recovery attempts auditable.
                    stereo_retry_status_code = 0
                    guided_retry_ratio = (
                        config.marked_stereo_guided_retry_ratio
                        if marker_backend_active
                        else config.max_stereo_centerline_length_ratio)
                    outside_retry_threshold = 0.15
                    outside_retry = bool(
                        marker_backend_active and max(
                            left_outside_mask_fraction,
                            right_outside_mask_fraction)
                        > outside_retry_threshold)
                    if ((segmenter is not None or marker_backend_active)
                            and (centerline_length_ratio > guided_retry_ratio
                                 or outside_retry)):
                        stereo_retry_attempted = 1
                        if (abs(left_outside_mask_fraction
                               - right_outside_mask_fraction) > 0.05):
                            retry_is_left = (
                                left_outside_mask_fraction
                                > right_outside_mask_fraction)
                        else:
                            retry_is_left = left_length_px < right_length_px
                        retry_image = left_image if retry_is_left else right_image
                        retry_roi = (
                            registration.roi_left_xywh if retry_is_left
                            else registration.roi_right_xywh)
                        retry_base = base_left if retry_is_left else base_right
                        original_target_result = (
                            left_result if retry_is_left else right_result)
                        source_result = right_result if retry_is_left else left_result
                        shift = (
                            registered_base_disparity_px if retry_is_left
                            else -registered_base_disparity_px)
                        retry_result = None
                        if segmenter is not None:
                            retry_prompt = _stereo_retry_prompt(
                                source_result, retry_roi, shift)
                            try:
                                retry_result = segmenter.segment(
                                    retry_image, retry_roi, retry_base,
                                    retry_prompt)
                            except (RuntimeError, ValueError):
                                pass
                            finally:
                                for key, value in segmenter.last_timing_s.items():
                                    frame_timing_s[key] = (
                                        frame_timing_s.get(key, 0.0) + value)
                        # Even when the second SAM call fails, the original
                        # partial target mask plus target-image color can support
                        # a complete stereo-guided ridge. The helper rejects a
                        # transferred curve with insufficient target evidence.
                        guided_input = (
                            retry_result if retry_result is not None
                            else original_target_result)
                        try:
                            retry_result = _stereo_guided_retry_result(
                                guided_input,
                                source=source_result,
                                target_image=retry_image,
                                horizontal_shift_px=shift,
                                reference_width_px=_mask_effective_width(
                                    source_result),
                                maximum_width_px=(
                                    config.marked_max_mask_effective_width_px
                                    if marker_backend_active else
                                    config.max_mask_effective_width_px),
                                disparity_prior_px=(
                                    previous_fitted_disparity_px
                                    if (previous_fitted_disparity_px is not None
                                        and temporal_gap_ms <=
                                        config.temporal_disparity_max_gap_ms)
                                    else None),
                                allow_temporal_completion=marker_backend_active)
                        except ValueError:
                            retry_result = None
                            stereo_retry_status_code = 2
                        if retry_result is not None:
                            # A cross-view prompt can recover the complete path
                            # while SAM also includes a broad adjacent surface.
                            # Preserve that path, but retain only a catheter-width
                            # part of the predicted mask around it.
                            retry_length_px = float(cumulative_arclength(
                                retry_result.material.points)[-1])
                            other_length_px = (
                                right_length_px if retry_is_left
                                else left_length_px)
                            retry_ratio = max(
                                retry_length_px, other_length_px) / max(
                                    min(retry_length_px, other_length_px), 1e-9)
                            retry_roi_area = retry_roi[2] * retry_roi[3]
                            retry_effective_width_px = _mask_effective_width(
                                retry_result)
                            retry_outside_mask_fraction = (
                                _centerline_outside_mask_fraction(retry_result))
                            original_outside_mask_fraction = (
                                left_outside_mask_fraction if retry_is_left
                                else right_outside_mask_fraction)
                            retry_quality_ok = (
                                len(retry_result.material)
                                >= config.min_centerline_points
                                and (marker_backend_active or (
                                    retry_result.sam_iou >= config.min_sam_iou
                                    and retry_result.seed_recall
                                    >= config.min_seed_recall))
                                and retry_result.mask_area_px
                                <= config.max_mask_area_fraction * retry_roi_area
                                and retry_effective_width_px
                                <= (config.marked_max_mask_effective_width_px
                                    if marker_backend_active else
                                    config.max_mask_effective_width_px))
                            retry_improves_geometry = bool(
                                retry_ratio < centerline_length_ratio
                                or retry_outside_mask_fraction + 0.02
                                < original_outside_mask_fraction)
                            if retry_quality_ok and retry_improves_geometry:
                                if retry_is_left:
                                    left_result = retry_result
                                    left_length_px = retry_length_px
                                    left_outside_mask_fraction = (
                                        retry_outside_mask_fraction)
                                    stereo_retry_view = 1
                                else:
                                    right_result = retry_result
                                    right_length_px = retry_length_px
                                    right_outside_mask_fraction = (
                                        retry_outside_mask_fraction)
                                    stereo_retry_view = 2
                                centerline_length_ratio = retry_ratio
                                stereo_retry_status_code = 1
                            elif not retry_quality_ok:
                                stereo_retry_status_code = 3
                            else:
                                stereo_retry_status_code = 4
                    ill_view = None
                    observation_ill_view = None
                    observation_weight_left = 1.0
                    observation_weight_right = 1.0
                    ill_detection_ambiguity = float("nan")
                    if marker_backend_active:
                        shorter_view = (
                            "left" if left_length_px < right_length_px
                            else "right")
                        longer_view = (
                            "right" if shorter_view == "left" else "left")
                        if (centerline_length_ratio
                                > config.marked_stereo_guided_retry_ratio):
                            ill_view = shorter_view
                        elif (centerline_length_ratio
                                > config.stereo_reference_switch_ratio):
                            longer_result = (
                                left_result if longer_view == "left"
                                else right_result)
                            shorter_result = (
                                left_result if shorter_view == "left"
                                else right_result)
                            ill_detection_ambiguity = (
                                epipolar_mask_ambiguity_fraction(
                                    longer_result.material.centerline,
                                    shorter_result.material.centerline,
                                    longer_view,
                                    n_samples=config.stereo_samples,
                                    smooth_2d=config.smooth_2d,
                                    row_search_px=config.overlap_row_search_px))
                            if (ill_detection_ambiguity
                                    >= config.marked_ill_epipolar_ambiguity_fraction):
                                ill_view = shorter_view
                        raw_ill_view = ill_view
                        # Entry, release, and left/right switches all require
                        # persistent evidence. A one-frame ambiguity excursion
                        # must not replace a usable independent cyan route.
                        (active_ill_view, pending_ill_view,
                         pending_ill_view_count) = (
                            _update_ill_view_hysteresis(
                                active_ill_view, pending_ill_view,
                                pending_ill_view_count, raw_ill_view,
                                config.stereo_ill_switch_confirm_frames))
                        ill_view = active_ill_view
                        metrics.update({
                            "stereo_raw_ill_view": (
                                1 if raw_ill_view == "left" else
                                2 if raw_ill_view == "right" else 0),
                            "stereo_ill_view_switch_pending": int(
                                pending_ill_view_count > 0),
                        })
                        (observation_weight_left,
                         observation_weight_right,
                         observation_ill_view) = _ill_eye_observation_weights(
                            ill_view, shorter_view,
                            left_length_px, right_length_px,
                            config.ill_eye_min_observation_weight)
                    stage_started = time.perf_counter()
                    left_effective_width_px = _mask_effective_width(left_result)
                    right_effective_width_px = _mask_effective_width(right_result)
                    metrics.update({
                        "centerline_length_left_px": left_length_px,
                        "centerline_length_right_px": right_length_px,
                        "centerline_outside_mask_fraction_left": (
                            left_outside_mask_fraction),
                        "centerline_outside_mask_fraction_right": (
                            right_outside_mask_fraction),
                        "mask_effective_width_left_px": left_effective_width_px,
                        "mask_effective_width_right_px": right_effective_width_px,
                        "stereo_centerline_length_ratio_initial": (
                            initial_centerline_length_ratio),
                        "stereo_centerline_length_ratio": centerline_length_ratio,
                        "stereo_retry_used": int(stereo_retry_view != 0),
                        "stereo_retry_view": stereo_retry_view,
                        "stereo_retry_attempted": stereo_retry_attempted,
                        "stereo_retry_status_code": stereo_retry_status_code,
                        "stereo_ill_detection_ambiguity": (
                            ill_detection_ambiguity),
                    })
                    temporal_mask_bypassed_view = 0
                    if (previous_valid_results is not None
                            and 0.0 < temporal_gap_ms
                            <= config.max_temporal_prompt_gap_ms):
                        for view, current, previous, length in (
                                ("left", left_result, previous_valid_results[0],
                                 left_length_px),
                                ("right", right_result, previous_valid_results[1],
                                 right_length_px)):
                            coverage, temporal_p95 = _temporal_centerline_metrics(
                                previous.material.points,
                                current.material.points,
                                config.temporal_centerline_tolerance_px)
                            previous_length = float(cumulative_arclength(
                                previous.material.points)[-1])
                            metrics[f"temporal_centerline_coverage_{view}"] = coverage
                            metrics[f"temporal_centerline_p95_{view}_px"] = temporal_p95
                            collapsed = length < 0.80 * previous_length
                            displaced = (
                                coverage < config.min_temporal_centerline_coverage
                                and temporal_p95
                                > config.max_temporal_centerline_p95_px)
                            if ((collapsed and (
                                    coverage
                                    < config.min_temporal_centerline_coverage
                                    or temporal_p95
                                    > config.max_temporal_centerline_p95_px))
                                    or displaced):
                                if marker_backend_active and view == ill_view:
                                    temporal_mask_bypassed_view = (
                                        1 if view == "left" else 2)
                                    continue
                                raise ValueError(
                                    f"temporal_{view}_mask_inconsistent:"
                                    f"coverage={coverage:.2f},"
                                    f"length_ratio={length / max(previous_length, 1e-9):.2f}")
                    metrics["temporal_mask_inconsistency_bypassed_view"] = (
                        temporal_mask_bypassed_view)
                    frame_timing_s["quality_control"] += (
                        time.perf_counter() - stage_started)
                    if (not marker_backend_active and centerline_length_ratio
                            > config.max_stereo_centerline_length_ratio):
                        raise ValueError(
                            "stereo_centerline_length_mismatch:"
                            f"ratio={centerline_length_ratio:.2f}")
                    # Preserve independently extracted observations. A later
                    # overlap solve may provide a better displayed/inference
                    # path, but it must never become the next frame's image
                    # segmentation prior without passing a separate gate.
                    observed_results = (left_result, right_result)
                    stage_started = time.perf_counter()
                    tip_camera_m, tip_epipolar_error_px = (
                        _stereo_tip_camera_point(
                            left_result.yellow_tip_xy,
                            right_result.yellow_tip_xy,
                            registration.K, registration.baseline_m,
                            config.max_tip_epipolar_error_px))
                    left_markers, left_marker_confidence = (
                        _usable_marker_observations(
                            left_result, config.marker_min_confidence))
                    right_markers, right_marker_confidence = (
                        _usable_marker_observations(
                            right_result, config.marker_min_confidence))
                    if marker_backend_active:
                        marker_scales = np.array([
                            config.marker_interface_weight_scale,
                            1.0, 1.0,
                            config.marker_tip_weight_scale,
                        ], dtype=np.float64)
                        left_marker_confidence *= marker_scales
                        right_marker_confidence *= marker_scales
                    marker_camera_m, marker_stereo_observed, marker_errors = (
                        _stereo_marker_camera_points(
                            left_markers, right_markers,
                            registration.K, registration.baseline_m,
                            config.max_marker_epipolar_error_px))
                    marker_points_base_mm = np.full((4, 3), np.nan)
                    finite_markers = np.all(np.isfinite(marker_camera_m), axis=1)
                    if np.any(finite_markers):
                        marker_points_base_mm[finite_markers] = (
                            _base_points_from_camera(
                                registration.left_camera_T_base,
                                marker_camera_m[finite_markers]))
                    candidates = {}
                    candidate_details = {}
                    disparity_prior_available = bool(
                        previous_fitted_disparity_px is not None
                        and temporal_gap_ms
                        <= config.temporal_disparity_max_gap_ms)
                    disparity_prior_weight = (
                        config.temporal_disparity_weight * np.exp(
                            -max(temporal_gap_ms, 0.0)
                            / max(config.temporal_disparity_decay_ms, 1e-6))
                        if disparity_prior_available else 0.0)
                    for candidate_view in ("left", "right"):
                        try:
                            force_overlap = bool(
                                ill_view is not None
                                and candidate_view != ill_view)
                            candidate = reconstruct_disparity_anchored(
                                left_result.material.centerline,
                                right_result.material.centerline,
                                registration.K, registration.baseline_m,
                                base_camera_m, tip_camera_m,
                                n_samples=config.stereo_samples,
                                smooth_2d=config.smooth_2d,
                                disparity_first_difference_weight=(
                                    config.disparity_first_difference_weight),
                                disparity_second_difference_weight=(
                                    config.disparity_second_difference_weight),
                                disparity_huber_delta_px=(
                                    config.disparity_huber_delta_px),
                                reference_view=candidate_view,
                                disparity_prior_px=(
                                    previous_fitted_disparity_px
                                    if disparity_prior_available
                                    else None),
                                disparity_prior_weight=(
                                    disparity_prior_weight),
                                overlap_aware=(
                                    config.overlap_aware_reconstruction
                                    and (not marker_backend_active
                                         or force_overlap)),
                                force_overlap_aware=force_overlap,
                                overlap_row_search_px=(
                                    config.overlap_row_search_px),
                                overlap_self_fraction_threshold=(
                                    config.overlap_self_fraction_threshold),
                                overlap_self_distance_px=(
                                    config.overlap_self_distance_px),
                                overlap_min_arclength_separation=(
                                    config.overlap_min_arclength_separation),
                                centerline_tip_depth_weight=(
                                    config.centerline_tip_depth_weight),
                                max_centerline_tip_epipolar_error_px=(
                                    config.max_centerline_tip_epipolar_error_px),
                                terminal_refinement=(
                                    config.terminal_disparity_refinement
                                    and not marker_backend_active),
                                terminal_refinement_fraction=(
                                    config.terminal_disparity_fraction),
                                terminal_refinement_smoothness_scale=(
                                    config.terminal_disparity_smoothness_scale),
                                terminal_refinement_observation_weight=(
                                    config.terminal_disparity_observation_weight),
                                terminal_refinement_tip_weight=(
                                    config.terminal_disparity_tip_weight),
                                terminal_refinement_max_p95_degradation_px=(
                                    config.terminal_disparity_max_p95_degradation_px),
                                marker_left_px=left_markers,
                                marker_right_px=right_markers,
                                marker_confidence_left=left_marker_confidence,
                                marker_confidence_right=right_marker_confidence,
                                marker_disparity_weight=(
                                    config.marker_disparity_weight),
                                max_marker_epipolar_error_px=(
                                    config.max_marker_epipolar_error_px))
                            score, details = _stereo_candidate_score(
                                candidate, candidate_view,
                                left_length_px, right_length_px,
                                previous_reconstructed_camera_m if (
                                    temporal_gap_ms
                                    <= config.max_temporal_prompt_gap_ms)
                                else None,
                                previous_stereo_reference_view)
                            candidates[candidate_view] = (score, candidate)
                            candidate_details[candidate_view] = details
                        except (ArithmeticError, RuntimeError, ValueError,
                                np.linalg.LinAlgError):
                            continue
                    reference_view = _select_stereo_reference(
                        candidates, previous_stereo_reference_view,
                        config.stereo_reference_hysteresis_score,
                        ill_view=ill_view)
                    reconstruction = candidates[reference_view][1]
                    left_score = (
                        candidates["left"][0] if "left" in candidates
                        else float("nan"))
                    right_score = (
                        candidates["right"][0] if "right" in candidates
                        else float("nan"))
                    metrics.update({
                        "stereo_candidate_score_left": left_score,
                        "stereo_candidate_score_right": right_score,
                        "stereo_candidate_margin": (
                            abs(left_score - right_score)
                            if np.isfinite(left_score) and np.isfinite(right_score)
                            else float("nan")),
                        "stereo_ill_view": (
                            1 if ill_view == "left" else
                            2 if ill_view == "right" else 0),
                        "stereo_observation_weight_left": (
                            observation_weight_left),
                        "stereo_observation_weight_right": (
                            observation_weight_right),
                        "stereo_candidate_temporal_rms_left_mm": (
                            candidate_details.get("left", {}).get(
                                "temporal_rms_mm", float("nan"))),
                        "stereo_candidate_temporal_rms_right_mm": (
                            candidate_details.get("right", {}).get(
                                "temporal_rms_mm", float("nan"))),
                        "stereo_epipolar_ambiguity_left": (
                            candidates.get("left", (None, {}))[1].get(
                                "epipolar_ambiguity_fraction", float("nan"))),
                        "stereo_epipolar_ambiguity_right": (
                            candidates.get("right", (None, {}))[1].get(
                                "epipolar_ambiguity_fraction", float("nan"))),
                        "stereo_other_eye_self_overlap_left": (
                            candidates.get("left", (None, {}))[1].get(
                                "other_eye_self_overlap_fraction", float("nan"))),
                        "stereo_other_eye_self_overlap_right": (
                            candidates.get("right", (None, {}))[1].get(
                                "other_eye_self_overlap_fraction", float("nan"))),
                    })
                    metrics["stereo_reference_view"] = (
                        1 if reference_view == "left" else 2)
                    metrics["overlap_aware_used"] = int(
                        reconstruction["overlap_aware_used"])
                    metrics["overlap_aware_forced"] = int(
                        reconstruction.get("overlap_aware_forced", 0))
                    metrics["overlap_one_turn_enforced"] = int(
                        reconstruction.get("overlap_one_turn_enforced", 0))
                    metrics["overlap_expected_turn_index"] = int(
                        reconstruction.get("overlap_expected_turn_index", -1))
                    metrics["centerline_tip_anchor_used"] = int(
                        reconstruction["centerline_tip_anchor_used"])
                    metrics["centerline_tip_epipolar_error_px"] = (
                        reconstruction["centerline_tip_epipolar_error_px"])
                    metrics["marker_anchor_count"] = int(
                        reconstruction["marker_anchor_count"])
                    metrics["marker_interval_boundary_count"] = int(
                        reconstruction.get("marker_interval_boundary_count", 0))
                    metrics["marker_epipolar_error_max_px"] = (
                        reconstruction["marker_epipolar_error_max_px"])
                    metrics["terminal_refinement_used"] = int(
                        reconstruction["terminal_refinement_used"])
                    metrics["terminal_refinement_improvement_px"] = (
                        reconstruction["terminal_refinement_improvement_px"])
                    metrics["terminal_refinement_start_fraction"] = (
                        reconstruction["terminal_refinement_start_fraction"])
                    metrics["terminal_reprojection_left_px"] = (
                        reconstruction["terminal_reprojection_left_px"])
                    metrics["terminal_reprojection_right_px"] = (
                        reconstruction["terminal_reprojection_right_px"])
                    inferred_view = 0
                    inferred_candidate_view = 0
                    inferred_accepted = 0
                    inferred_metrics: dict[str, float | int] = {
                        "stereo_inferred_rejection_code": 0,
                        "stereo_inferred_marker_order_valid": np.nan,
                        "stereo_inferred_terminal_error_px": np.nan,
                        "stereo_inferred_sharp_turn_clusters": np.nan,
                        "stereo_inferred_temporal_p95_px": np.nan,
                        "stereo_inferred_temporal_max_px": np.nan,
                        "stereo_inferred_outside_mask_fraction": np.nan,
                        "stereo_inferred_medial_radius_ratio": np.nan,
                    }
                    if reconstruction.get("overlap_aware_forced", 0):
                        if ill_view == "left":
                            inferred_candidate_view = 1
                            previous_observed = (
                                None if previous_valid_results is None else
                                previous_valid_results[0].material.points)
                            (candidate_result, accepted,
                             inferred_metrics) = _inferred_material_path_quality(
                                left_result,
                                reconstruction["ordered_left_px"],
                                previous_observed,
                                config.max_inferred_centerline_temporal_p95_px,
                                config.max_inferred_centerline_temporal_max_px,
                                config.max_inferred_sharp_turn_clusters)
                            if accepted:
                                left_result = candidate_result
                                inferred_view = 1
                                inferred_accepted = 1
                        elif ill_view == "right":
                            inferred_candidate_view = 2
                            previous_observed = (
                                None if previous_valid_results is None else
                                previous_valid_results[1].material.points)
                            (candidate_result, accepted,
                             inferred_metrics) = _inferred_material_path_quality(
                                right_result,
                                reconstruction["ordered_right_px"],
                                previous_observed,
                                config.max_inferred_centerline_temporal_p95_px,
                                config.max_inferred_centerline_temporal_max_px,
                                config.max_inferred_sharp_turn_clusters)
                            if accepted:
                                right_result = candidate_result
                                inferred_view = 2
                                inferred_accepted = 1
                    metrics.update(inferred_metrics)
                    metrics.update({
                        "stereo_inferred_candidate_view": inferred_candidate_view,
                        "stereo_inferred_accepted": inferred_accepted,
                        "stereo_inferred_view": inferred_view,
                    })
                    # These diagnostics describe the actual cyan paths written
                    # to HDF5/overlay, including any accepted stereo repair.
                    for view, result in (
                            ("left", left_result), ("right", right_result)):
                        topology = _marked_centerline_topology_metrics(result)
                        metrics.update({
                            f"marker_interval_route_used_{view}": (
                                topology["marker_interval_route_used"]),
                            f"marker_path_order_valid_{view}": (
                                topology["marker_path_order_valid"]),
                            f"marker_path_max_centroid_distance_{view}_px": (
                                topology[
                                    "marker_path_max_centroid_distance_px"]),
                            f"marker_path_observed_count_{view}": (
                                topology["marker_path_observed_count"]),
                            f"centerline_sharp_turn_cluster_count_{view}": (
                                topology[
                                    "centerline_sharp_turn_cluster_count"]),
                            f"centerline_max_local_turn_{view}_deg": (
                                topology["centerline_max_local_turn_deg"]),
                            f"centerline_outside_mask_fraction_{view}": (
                                _centerline_outside_mask_fraction(result)),
                            f"centerline_length_{view}_px": float(
                                cumulative_arclength(
                                    result.material.points)[-1]),
                        })
                    metrics["tip_stereo_observed"] = int(
                        tip_camera_m is not None)
                    metrics["tip_epipolar_error_px"] = (
                        tip_epipolar_error_px)
                    metrics["tip_anchor_endpoint_distance_mm"] = (
                        float("nan") if tip_camera_m is None else float(
                            np.linalg.norm(
                                reconstruction["points_camera_m"][-1]
                                - tip_camera_m) * 1000.0))
                    frame_timing_s["stereo_reconstruction"] = (
                        time.perf_counter() - stage_started)

                    stage_started = time.perf_counter()
                    reconstructed_s_mm = cumulative_arclength(
                        reconstruction["points_camera_m"]) * 1000.0
                    visible_length_mm = float(reconstructed_s_mm[-1])
                    if (not marker_backend_active
                            and visible_length_mm + 1e-6
                            < nominal_distal_length_mm):
                        metrics["visible_arc_length_mm"] = visible_length_mm
                        raise ValueError(
                            "visible_curve_shorter_than_distal:"
                            f"{visible_length_mm:.2f}mm")
                    parameter = np.linspace(0.0, 1.0, len(reconstructed_s_mm))
                    target_boundary_s_mm = (
                        visible_length_mm - nominal_distal_length_mm)
                    search_half = config.distal_boundary_search_half_width_mm
                    target_fraction = float(np.interp(
                        target_boundary_s_mm, reconstructed_s_mm, parameter))
                    boundary_camera_m = np.array([
                        np.interp(
                            target_boundary_s_mm, reconstructed_s_mm,
                            reconstruction["points_camera_m"][:, coordinate])
                        for coordinate in range(3)
                    ])
                    boundary_left_px = project_camera_points(
                        boundary_camera_m[None], registration.K,
                        registration.baseline_m, right=False)[0]
                    boundary_right_px = project_camera_points(
                        boundary_camera_m[None], registration.K,
                        registration.baseline_m, right=True)[0]
                    minimum_fraction = float(np.interp(
                        max(0.0, target_boundary_s_mm - search_half),
                        reconstructed_s_mm, parameter))
                    maximum_fraction = float(np.interp(
                        min(visible_length_mm, target_boundary_s_mm + search_half),
                        reconstructed_s_mm, parameter))
                    boundary_indices = []
                    boundary_weights = []
                    if marker_backend_active:
                        left_color_fraction = right_color_fraction = float("nan")
                        left_boundary_valid = bool(np.all(np.isfinite(
                            left_markers[0])))
                        right_boundary_valid = bool(np.all(np.isfinite(
                            right_markers[0])))
                        for valid, point, confidence, ordered in (
                                (left_boundary_valid, left_markers[0],
                                 left_marker_confidence[0],
                                 reconstruction["ordered_left_px"]),
                                (right_boundary_valid, right_markers[0],
                                 right_marker_confidence[0],
                                 reconstruction["ordered_right_px"])):
                            if valid:
                                boundary_indices.append(int(np.argmin(
                                    np.linalg.norm(ordered - point, axis=1))))
                                boundary_weights.append(max(
                                    float(confidence), 1e-3))
                    else:
                        left_result, left_color_fraction, left_boundary_valid = (
                            _apply_distal_boundary_prior(
                                left_result, target_fraction,
                                minimum_fraction, maximum_fraction,
                                boundary_left_px))
                        right_result, right_color_fraction, right_boundary_valid = (
                            _apply_distal_boundary_prior(
                                right_result, target_fraction,
                                minimum_fraction, maximum_fraction,
                                boundary_right_px))
                        if left_boundary_valid:
                            point = left_result.material.points[
                                left_result.material.distal_boundary_index]
                            boundary_indices.append(int(np.argmin(np.linalg.norm(
                                reconstruction["ordered_left_px"] - point,
                                axis=1))))
                            boundary_weights.append(max(
                                left_result.material.boundary_confidence, 1e-3))
                        if right_boundary_valid:
                            point = right_result.material.points[
                                right_result.material.distal_boundary_index]
                            boundary_indices.append(int(np.argmin(np.linalg.norm(
                                reconstruction["ordered_right_px"] - point,
                                axis=1))))
                            boundary_weights.append(max(
                                right_result.material.boundary_confidence, 1e-3))
                    boundary_observed = bool(boundary_indices)
                    boundary_stereo_consistent = bool(
                        boundary_observed and (
                            len(boundary_indices) < 2
                            or abs(boundary_indices[0] - boundary_indices[1])
                            <= max(6, config.stereo_samples // 8)))
                    observed_lengths_mm = [
                        float(visible_length_mm - reconstructed_s_mm[index])
                        for index in boundary_indices]
                    if boundary_observed:
                        observed_sample = int(np.clip(round(np.average(
                            boundary_indices, weights=boundary_weights)),
                            1, config.stereo_samples - 2))
                        observed_boundary_s_mm = float(
                            reconstructed_s_mm[observed_sample])
                        observed_distal_length_mm = float(
                            visible_length_mm - observed_boundary_s_mm)
                        color_fraction = float(parameter[observed_sample])
                        boundary_prior_error_mm = float(
                            observed_boundary_s_mm - target_boundary_s_mm)
                    else:
                        observed_distal_length_mm = float("nan")
                        color_fraction = float("nan")
                        boundary_prior_error_mm = float("nan")
                    raw_distal_length_mm, fused_distal_length_mm, (
                        interface_uncertainty_mm) = _fuse_distal_length(
                            nominal_distal_length_mm,
                            observed_lengths_mm, boundary_weights,
                            previous_distal_length_mm if (
                                temporal_gap_ms
                                <= config.max_temporal_prompt_gap_ms)
                            else None,
                            (config.marked_distal_length_prior_sigma_mm
                             if marker_backend_active
                             else config.distal_length_prior_sigma_mm),
                            (config.marked_interface_marker_sigma_mm
                             if marker_backend_active
                             else config.interface_color_sigma_mm),
                            config.interface_temporal_sigma_mm)
                    raw_interface_s_mm = float(np.clip(
                        visible_length_mm - raw_distal_length_mm,
                        0.0, visible_length_mm))
                    fused_interface_s_mm = float(np.clip(
                        visible_length_mm - fused_distal_length_mm,
                        0.0, visible_length_mm))

                    visible_base_mm = _base_points_from_camera(
                        registration.left_camera_T_base,
                        reconstruction["points_camera_m"])
                    raw_interface_base_mm = _point_at_arclength(
                        visible_base_mm, raw_interface_s_mm)
                    interface_temporal_alpha = 1.0
                    filtered_interface_s_mm = fused_interface_s_mm
                    if (previous_interface_base_mm is not None
                            and 0.0 < temporal_gap_ms
                            <= config.max_temporal_prompt_gap_ms
                            and config.interface_temporal_cutoff_hz > 0.0):
                        projected_previous_s_mm = _project_point_to_polyline_s(
                            visible_base_mm, previous_interface_base_mm,
                            fused_interface_s_mm,
                            config.interface_projection_window_mm)
                        dt_s = max(temporal_gap_ms * 1e-3, 1e-6)
                        interface_temporal_alpha = float(
                            1.0 - np.exp(
                                -2.0 * np.pi
                                * config.interface_temporal_cutoff_hz * dt_s))
                        filtered_interface_s_mm = float(
                            interface_temporal_alpha * fused_interface_s_mm
                            + (1.0 - interface_temporal_alpha)
                            * projected_previous_s_mm)
                    filtered_interface_s_mm = float(np.clip(
                        filtered_interface_s_mm, 0.0, visible_length_mm))
                    filtered_distal_length_mm = float(
                        visible_length_mm - filtered_interface_s_mm)
                    target_fraction = float(np.interp(
                        filtered_interface_s_mm,
                        reconstructed_s_mm, parameter))
                    boundary_sample = int(np.clip(round(
                        target_fraction * (config.stereo_samples - 1)),
                        1, config.stereo_samples - 2))
                    boundary_camera_m = reconstruction[
                        "points_camera_m"][boundary_sample]
                    boundary_left_px = reconstruction[
                        "ordered_left_px"][boundary_sample]
                    boundary_right_px = reconstruction[
                        "ordered_right_px"][boundary_sample]
                    # Both displayed 2D splits now represent the same fused 3D
                    # material observation.
                    left_index = int(np.argmin(np.linalg.norm(
                        left_result.material.points - boundary_left_px,
                        axis=1)))
                    right_index = int(np.argmin(np.linalg.norm(
                        right_result.material.points - boundary_right_px,
                        axis=1)))
                    left_result = replace(left_result, material=replace(
                        left_result.material,
                        distal_boundary_index=left_index,
                        distal_boundary_fraction=float(
                            left_index / max(len(left_result.material) - 1, 1))))
                    right_result = replace(right_result, material=replace(
                        right_result.material,
                        distal_boundary_index=right_index,
                        distal_boundary_fraction=float(
                            right_index / max(len(right_result.material) - 1, 1))))
                    boundary_confidence = float(max(
                        left_result.material.boundary_confidence,
                        right_result.material.boundary_confidence))
                    boundary_contrast = float(max(
                        left_result.material.boundary_contrast,
                        right_result.material.boundary_contrast))
                    frame_timing_s["material_boundary"] = (
                        time.perf_counter() - stage_started)

                    stage_started = time.perf_counter()
                    base_endpoint_distance_mm = float(np.linalg.norm(
                        visible_base_mm[0] - base_mm))
                    metrics["base_endpoint_distance_mm"] = (
                        base_endpoint_distance_mm)
                    if (base_endpoint_distance_mm
                            > config.max_base_endpoint_distance_mm):
                        raise ValueError(
                            "visible_base_endpoint_too_far:"
                            f"{base_endpoint_distance_mm:.2f}mm")
                    assembled = assemble_image_only_shape(
                        visible_base_mm, target_fraction,
                        full_count=config.full_samples,
                        distal_count=config.distal_samples,
                        base_position_base_mm=base_mm,
                        bridge_base=False,
                        distal_length_mm=filtered_distal_length_mm)
                    full_geometry = curve_geometry(
                        assembled.full_points_mm, config.curvature_smoothing_mm,
                        config.curvature_spline_bases)
                    distal_geometry = curve_geometry(
                        assembled.distal_points_mm, config.curvature_smoothing_mm,
                        config.curvature_spline_bases)
                    if reconstruction_backend == "joint_spline":
                        joint_started = time.perf_counter()
                        left_interface_observation = (
                            left_markers[0] if np.all(np.isfinite(
                                left_markers[0])) else
                            left_result.material.points[
                                left_result.material.distal_boundary_index])
                        right_interface_observation = (
                            right_markers[0] if np.all(np.isfinite(
                                right_markers[0])) else
                            right_result.material.points[
                                right_result.material.distal_boundary_index])
                        left_tip_observation = (
                            left_markers[3] if np.all(np.isfinite(
                                left_markers[3])) else
                            left_result.material.points[-1])
                        right_tip_observation = (
                            right_markers[3] if np.all(np.isfinite(
                                right_markers[3])) else
                            right_result.material.points[-1])
                        left_distal_observation, left_endpoint_axial = (
                            distal_image_observation(
                                left_result.material.points,
                                left_interface_observation,
                                left_tip_observation,
                                count=max(64, config.stereo_samples)))
                        right_distal_observation, right_endpoint_axial = (
                            distal_image_observation(
                                right_result.material.points,
                                right_interface_observation,
                                right_tip_observation,
                                count=max(64, config.stereo_samples)))
                        (left_ordered_corridor,
                         right_ordered_corridor) = (
                            _center_paired_mask_corridors(
                                reconstruction[
                                    "ordered_left_px"][boundary_sample:],
                                reconstruction[
                                    "ordered_right_px"][boundary_sample:],
                                left_result, right_result,
                                config.distal_samples,
                                reconstructed_s_mm[boundary_sample:]))
                        left_ridge_evidence = _distal_medial_ridge_evidence(
                            left_image, left_result)
                        right_ridge_evidence = _distal_medial_ridge_evidence(
                            right_image, right_result)
                        left_turn, left_turn_angle = _projected_turn_fraction(
                            left_ordered_corridor)
                        right_turn, right_turn_angle = _projected_turn_fraction(
                            right_ordered_corridor)
                        # The well-conditioned eye defines the shared turn.
                        # Its farthest epipolar row, rather than a visible
                        # corner, transfers the material location at which the
                        # ill eye is allowed to reverse.
                        if observation_ill_view == "left":
                            turn_fraction = _epipolar_sweep_extremum_fraction(
                                right_ordered_corridor)
                        elif observation_ill_view == "right":
                            turn_fraction = _epipolar_sweep_extremum_fraction(
                                left_ordered_corridor)
                        else:
                            turn_fraction = None
                        if (turn_fraction is None
                                and observation_ill_view is not None):
                            turn_fraction = previous_joint_turn_fraction
                        elif (turn_fraction is not None
                              and previous_joint_turn_fraction is not None
                              and temporal_gap_ms
                              <= config.max_temporal_prompt_gap_ms):
                            # This state controls only where curvature is cheap;
                            # it does not blend curve positions or introduce
                            # image-space lag.
                            bounded_turn = float(np.clip(
                                turn_fraction,
                                previous_joint_turn_fraction - 0.08,
                                previous_joint_turn_fraction + 0.08))
                            turn_fraction = float(
                                0.75 * bounded_turn
                                + 0.25 * previous_joint_turn_fraction)
                        common_fit = dict(
                            # Ring centroids remain soft observations.
                            left_axial_markers_xy=left_markers,
                            right_axial_markers_xy=right_markers,
                            left_ridge_evidence_xy=left_ridge_evidence,
                            right_ridge_evidence_xy=right_ridge_evidence,
                            output_samples=config.distal_samples,
                            basis_count=config.joint_spline_basis_count,
                            fit_samples=config.distal_samples,
                            length_sigma_mm=(
                                config.marked_distal_length_prior_sigma_mm
                                if marker_backend_active else
                                config.distal_length_prior_sigma_mm),
                            temporal_prior_points_base_mm=(
                                previous_filtered_distal_points
                                if (previous_filtered_distal_points is not None
                                    and temporal_gap_ms
                                    <= config.max_temporal_prompt_gap_ms)
                                else None),
                            temporal_prior_sigma_mm=max(
                                2.5, 0.65
                                * config.ill_eye_temporal_shape_sigma_mm),
                            max_nfev=config.joint_spline_max_nfev)
                        joint_reconstruction = None
                        topology_recovery_used = False
                        if observation_ill_view is not None:
                            ordinary = fit_joint_two_view_spline(
                                distal_geometry.points_mm,
                                left_distal_observation,
                                right_distal_observation,
                                registration.K,
                                registration.left_camera_T_base,
                                registration.right_camera_T_base,
                                nominal_distal_length_mm,
                                left_observation_weight=1.0,
                                right_observation_weight=1.0,
                                coverage_weight=0.30, **common_fit)
                            if _joint_fit_supports_bijective_stereo(ordinary):
                                joint_reconstruction = ordinary
                        if joint_reconstruction is None:
                            topology_recovery_used = (
                                observation_ill_view is not None)
                            joint_reconstruction = fit_joint_two_view_spline(
                                distal_geometry.points_mm,
                                left_distal_observation,
                                right_distal_observation,
                                registration.K,
                                registration.left_camera_T_base,
                                registration.right_camera_T_base,
                                nominal_distal_length_mm,
                                left_ordered_corridor_xy=(
                                    left_ordered_corridor
                                    if topology_recovery_used else None),
                                right_ordered_corridor_xy=(
                                    right_ordered_corridor
                                    if topology_recovery_used else None),
                                left_observation_weight=(
                                    observation_weight_left),
                                right_observation_weight=(
                                    observation_weight_right),
                                turn_fraction=(
                                    turn_fraction
                                    if topology_recovery_used else None),
                                coverage_weight=(
                                    0.0 if topology_recovery_used else 0.30),
                                **common_fit)
                        if not np.all(np.isfinite(
                                joint_reconstruction.points_base_mm)):
                            raise ValueError("joint_spline_nonfinite")
                        if (topology_recovery_used
                                and (
                                    joint_reconstruction.
                                    left_sharp_turn_clusters > 1
                                    or joint_reconstruction.
                                    right_sharp_turn_clusters > 1)):
                            raise ValueError(
                                "joint_multiple_projected_turns:"
                                f"left={joint_reconstruction.left_sharp_turn_clusters},"
                                f"right={joint_reconstruction.right_sharp_turn_clusters}")
                        distal_geometry = curve_geometry(
                            joint_reconstruction.points_base_mm,
                            config.curvature_smoothing_mm,
                            config.curvature_spline_bases)
                        metrics.update({
                            "joint_left_model_mean_px": (
                                joint_reconstruction.left_model_mean_px),
                            "joint_right_model_mean_px": (
                                joint_reconstruction.right_model_mean_px),
                            "joint_left_model_p95_px": (
                                joint_reconstruction.left_model_p95_px),
                            "joint_right_model_p95_px": (
                                joint_reconstruction.right_model_p95_px),
                            "joint_left_coverage_mean_px": (
                                joint_reconstruction.left_coverage_mean_px),
                            "joint_right_coverage_mean_px": (
                                joint_reconstruction.right_coverage_mean_px),
                            "joint_initial_symmetric_mean_px": (
                                joint_reconstruction.initial_symmetric_mean_px),
                            "joint_final_symmetric_mean_px": (
                                joint_reconstruction.final_symmetric_mean_px),
                            "joint_arc_length_mm": (
                                joint_reconstruction.arc_length_mm),
                            "joint_length_residual_mm": (
                                joint_reconstruction.length_residual_mm),
                            "joint_optimizer_cost": (
                                joint_reconstruction.optimizer_cost),
                            "joint_optimizer_evaluations": (
                                joint_reconstruction.optimizer_evaluations),
                            "joint_optimizer_success": int(
                                joint_reconstruction.optimizer_success),
                            "joint_turn_fraction": (
                                float("nan") if turn_fraction is None
                                else turn_fraction),
                            "joint_left_turn_angle_deg": left_turn_angle,
                            "joint_right_turn_angle_deg": right_turn_angle,
                            "joint_topology_recovery_used": int(
                                topology_recovery_used),
                            "joint_left_sharp_turn_clusters": (
                                joint_reconstruction.left_sharp_turn_clusters),
                            "joint_right_sharp_turn_clusters": (
                                joint_reconstruction.right_sharp_turn_clusters),
                        })
                        frame_timing_s["joint_spline_reconstruction"] = (
                            time.perf_counter() - joint_started)
                    temporal_shape_alpha = 1.0
                    if (previous_filtered_distal_points is not None
                            and temporal_gap_ms
                            <= config.max_temporal_prompt_gap_ms
                            and config.temporal_shape_cutoff_hz > 0.0):
                        dt_s = max(temporal_gap_ms * 1e-3, 1e-6)
                        temporal_shape_alpha = float(
                            1.0 - np.exp(
                                -2.0 * np.pi
                                * config.temporal_shape_cutoff_hz * dt_s))
                        filtered_distal = (
                            temporal_shape_alpha * distal_geometry.points_mm
                            + (1.0 - temporal_shape_alpha)
                            * previous_filtered_distal_points)
                        distal_geometry = curve_geometry(
                            filtered_distal,
                            config.curvature_smoothing_mm,
                            config.curvature_spline_bases)
                    if joint_reconstruction is not None:
                        # The paired spline is the sole final distal topology.
                        # Both cyan projections now come from the same 3D
                        # material coordinate as the yellow reconstruction.
                        left_result = _replace_distal_with_joint_projection(
                            left_result,
                            _project_base_points(
                                registration, distal_geometry.points_mm,
                                right=False))
                        right_result = _replace_distal_with_joint_projection(
                            right_result,
                            _project_base_points(
                                registration, distal_geometry.points_mm,
                                right=True))
                        for view, result in (
                                ("left", left_result),
                                ("right", right_result)):
                            topology = _marked_centerline_topology_metrics(result)
                            metrics.update({
                                f"marker_path_order_valid_{view}": topology[
                                    "marker_path_order_valid"],
                                f"marker_path_max_centroid_distance_{view}_px":
                                    topology[
                                        "marker_path_max_centroid_distance_px"],
                                f"centerline_sharp_turn_cluster_count_{view}":
                                    topology[
                                        "centerline_sharp_turn_cluster_count"],
                                f"centerline_max_local_turn_{view}_deg":
                                    topology["centerline_max_local_turn_deg"],
                                f"centerline_outside_mask_fraction_{view}":
                                    _centerline_outside_mask_fraction(result),
                                f"centerline_length_{view}_px": float(
                                    cumulative_arclength(
                                        result.material.points)[-1]),
                            })
                    stereo_score = min(
                        stereo_condition_score(left_result.material.points),
                        stereo_condition_score(right_result.material.points))
                    fitted_reprojection = _fitted_curve_reprojection_metrics(
                        registration, distal_geometry.points_mm,
                        observed_results[0].material.points,
                        observed_results[1].material.points)
                    fitted_tip_left = _project_base_points(
                        registration, distal_geometry.points_mm[-1:], False)[0]
                    fitted_tip_right = _project_base_points(
                        registration, distal_geometry.points_mm[-1:], True)[0]
                    left_terminal_observation = (
                        left_markers[3]
                        if marker_backend_active
                        and np.all(np.isfinite(left_markers[3]))
                        else left_result.yellow_tip_xy)
                    right_terminal_observation = (
                        right_markers[3]
                        if marker_backend_active
                        and np.all(np.isfinite(right_markers[3]))
                        else right_result.yellow_tip_xy)
                    tip_endpoint_left_px = (
                        float("nan") if left_terminal_observation is None
                        else float(np.linalg.norm(
                            fitted_tip_left - left_terminal_observation)))
                    tip_endpoint_right_px = (
                        float("nan") if right_terminal_observation is None
                        else float(np.linalg.norm(
                            fitted_tip_right - right_terminal_observation)))
                    frame_timing_s["shape_geometry"] = (
                        time.perf_counter() - stage_started)

                    stage_started = time.perf_counter()
                    metrics.update({
                        "reprojection_left_px": reconstruction["reprojection_left_px"],
                        "reprojection_right_px": reconstruction["reprojection_right_px"],
                        "reprojection_max_px": reconstruction["reprojection_max_px"],
                        "reprojection_p95_px": reconstruction["reprojection_p95_px"],
                        "stereo_condition": stereo_score,
                        "material_boundary_fraction": target_fraction,
                        "material_boundary_color_fraction": color_fraction,
                        "material_boundary_prior_error_mm": boundary_prior_error_mm,
                        "material_boundary_observed_distal_length_mm": (
                            observed_distal_length_mm),
                        "distal_length_raw_mm": raw_distal_length_mm,
                        "distal_length_filtered_mm": filtered_distal_length_mm,
                        "distal_length_residual_mm": (
                            filtered_distal_length_mm
                            - nominal_distal_length_mm),
                        "interface_uncertainty_mm": interface_uncertainty_mm,
                        "interface_raw_s_mm": raw_interface_s_mm,
                        "interface_filtered_s_mm": filtered_interface_s_mm,
                        "interface_temporal_alpha": interface_temporal_alpha,
                        "material_boundary_confidence": boundary_confidence,
                        "material_boundary_contrast": boundary_contrast,
                        "material_boundary_observed": int(boundary_observed),
                        "material_boundary_stereo_consistent": int(
                            boundary_stereo_consistent),
                        "distal_boundary_s_mm": assembled.distal_boundary_s_mm,
                        "base_bridge_length_mm": assembled.base_bridge_length_mm,
                        "base_endpoint_distance_mm": base_endpoint_distance_mm,
                        "visible_arc_length_mm": reconstruction["visible_arc_length_mm"],
                        "matched_epipolar": reconstruction["matched_epipolar"],
                        "disparity_robust_inlier_count": (
                            reconstruction["disparity_robust_inlier_count"]),
                        "full_spline_basis_count": full_geometry.spline_basis_count,
                        "full_spline_internal_knot_count": (
                            full_geometry.spline_internal_knot_count),
                        "full_spline_rms_residual_mm": (
                            full_geometry.spline_rms_residual_mm),
                        "distal_spline_basis_count": distal_geometry.spline_basis_count,
                        "distal_spline_internal_knot_count": (
                            distal_geometry.spline_internal_knot_count),
                        "distal_spline_rms_residual_mm": (
                            distal_geometry.spline_rms_residual_mm),
                        "distal_spline_arc_length_mm": (
                            distal_geometry.spline_arc_length_mm),
                        "temporal_shape_alpha": temporal_shape_alpha,
                        "tip_endpoint_left_px": tip_endpoint_left_px,
                        "tip_endpoint_right_px": tip_endpoint_right_px,
                        **fitted_reprojection,
                    })
                    worst_mean = max(
                        reconstruction["reprojection_left_px"],
                        reconstruction["reprojection_right_px"])
                    if (reconstruction_backend != "joint_spline"
                            and (worst_mean > config.max_mean_reprojection_px
                            or reconstruction["reprojection_p95_px"]
                            > config.max_p95_reprojection_px)):
                        raise ValueError(
                            "reprojection_exceeds_threshold:"
                            f"mean={worst_mean:.2f}px,"
                            f"p95={reconstruction['reprojection_p95_px']:.2f}px")
                    fitted_worst_mean = max(
                        fitted_reprojection["fitted_reprojection_left_px"],
                        fitted_reprojection["fitted_reprojection_right_px"])
                    if (reconstruction_backend != "joint_spline"
                            and (fitted_worst_mean
                                 > config.max_mean_reprojection_px
                            or fitted_reprojection["fitted_reprojection_p95_px"]
                            > config.max_p95_reprojection_px)):
                        raise ValueError(
                            "fitted_reprojection_exceeds_threshold:"
                            f"mean={fitted_worst_mean:.2f}px,"
                            "p95="
                            f"{fitted_reprojection['fitted_reprojection_p95_px']:.2f}px")
                    if (reconstruction_backend != "joint_spline"
                            and (left_terminal_observation is not None
                                 or right_terminal_observation is not None)
                            and max(
                            tip_endpoint_left_px, tip_endpoint_right_px)
                            > config.max_tip_endpoint_error_px):
                        raise ValueError(
                            "tip_endpoint_exceeds_threshold:"
                            f"left={tip_endpoint_left_px:.2f}px,"
                            f"right={tip_endpoint_right_px:.2f}px")
                    frame_timing_s["quality_control"] += (
                        time.perf_counter() - stage_started)

                    stage_started = time.perf_counter()
                    writer.write_observed_centerline(
                        index, "left", observed_results[0])
                    writer.write_observed_centerline(
                        index, "right", observed_results[1])
                    writer.write_view(index, "left", left_result)
                    writer.write_view(index, "right", right_result)
                    writer.write_success(
                        index, assembled, full_geometry, distal_geometry, metrics,
                        raw_interface_base_mm=raw_interface_base_mm,
                        visible_points_base_mm=visible_base_mm,
                        reconstruction=reconstruction,
                        joint_reconstruction=joint_reconstruction,
                        marker_points_base_mm=marker_points_base_mm,
                        marker_stereo_observed=marker_stereo_observed)
                    frame_timing_s["hdf5_write"] += (
                        time.perf_counter() - stage_started)
                    # Keep stereo-inferred correspondences out of the next
                    # frame's image-routing prior. They are valid 3D evidence,
                    # not independent 2D observations.
                    previous_valid_results = observed_results
                    previous_valid_timestamp_ns = record.timestamp_ns
                    previous_stereo_reference_view = reference_view
                    if joint_reconstruction is not None:
                        try:
                            (previous_fitted_disparity_px,
                             previous_reconstructed_camera_m) = (
                                _joint_temporal_stereo_prior(
                                    registration,
                                    joint_reconstruction.points_base_mm,
                                    config.stereo_samples))
                        except ValueError:
                            previous_fitted_disparity_px = reconstruction[
                                "fitted_disparity_px"].copy()
                            previous_reconstructed_camera_m = reconstruction[
                                "points_camera_m"].copy()
                    else:
                        previous_fitted_disparity_px = reconstruction[
                            "fitted_disparity_px"].copy()
                        previous_reconstructed_camera_m = reconstruction[
                            "points_camera_m"].copy()
                    previous_interface_base_mm = (
                        distal_geometry.points_mm[0].copy())
                    previous_distal_length_mm = filtered_distal_length_mm
                    previous_filtered_distal_points = (
                        distal_geometry.points_mm.copy())
                    if joint_reconstruction is not None:
                        previous_joint_turn_fraction = turn_fraction
                except _ObservationsOnlyFrame:
                    pass
                except (ArithmeticError, RuntimeError, ValueError,
                        np.linalg.LinAlgError) as exc:
                    status = str(exc)
                    stage_started = time.perf_counter()
                    # Persist a failed frame's final selected masks once for
                    # audit/reconstruction reuse. Successful frames are also
                    # written only once above; the old three-pass overwrite was
                    # the dominant HDF5 cost on external media.
                    if (isinstance(left_result, SamMaterialResult)
                            and isinstance(right_result, SamMaterialResult)):
                        writer.write_view(index, "left", left_result)
                        writer.write_view(index, "right", right_result)
                        writer.mark_observation_valid(index)
                    writer.write_failure(index, status, metrics)
                    frame_timing_s["hdf5_write"] += (
                        time.perf_counter() - stage_started)

                counts[status] = counts.get(status, 0) + 1
                projected_left = projected_right = None
                projected_boundary_left = projected_boundary_right = None
                if status == "valid" and distal_geometry is not None:
                    projected_left = _project_base_points(
                        registration, distal_geometry.points_mm, False)
                    projected_right = _project_base_points(
                        registration, distal_geometry.points_mm, True)
                    boundary_base_mm = distal_geometry.points_mm[0]
                    projected_boundary_left = _project_base_points(
                        registration, boundary_base_mm[None], False)[0]
                    projected_boundary_right = _project_base_points(
                        registration, boundary_base_mm[None], True)[0]
                stage_started = time.perf_counter()
                overlay.write(
                    "left", left_image, left_result, projected_left,
                    projected_boundary_left, status,
                    frame_index=index, svo_frame=record.svo_frame)
                overlay.write(
                    "right", right_image, right_result, projected_right,
                    projected_boundary_right, status,
                    frame_index=index, svo_frame=record.svo_frame)
                frame_timing_s["overlay_video"] = (
                    time.perf_counter() - stage_started)
                frame_timing_s["frame_total"] = (
                    time.perf_counter() - frame_started + prefetched_time_s)
                exclusive_total = sum(
                    value for key, value in frame_timing_s.items()
                    if key != "frame_total")
                frame_timing_s["other"] = max(
                    0.0, frame_timing_s["frame_total"] - exclusive_total)
                for key, value in frame_timing_s.items():
                    timing_totals_s[key] = timing_totals_s.get(key, 0.0) + value
                row = {
                    "output_index": index,
                    "svo_frame": record.svo_frame,
                    "timestamp_ns": record.timestamp_ns,
                    "svo_timestamp_ns": svo_timestamp_ns,
                    "svo_timestamp_error_ms": (
                        svo_timestamp_ns - record.timestamp_ns) / 1e6,
                    "valid": int(status == "valid"),
                    "status": status,
                    "command_valid": int(robot.command_valid[index]),
                    "position_valid": int(robot.position_valid[index]),
                    "encoder_valid": int(robot.encoder_valid[index]),
                    "sam_left_iou": (
                        np.nan if left_result is None else left_result.sam_iou),
                    "sam_right_iou": (
                        np.nan if right_result is None else right_result.sam_iou),
                    **metrics,
                    **{
                        f"timing_{key}_ms": value * 1000.0
                        for key, value in frame_timing_s.items()
                    },
                }
                summary_rows.append(row)
                if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(records):
                    elapsed = time.perf_counter() - started
                    print(
                        f"[{index + 1}/{len(records)}] {status}; "
                        f"{(index + 1) / max(elapsed, 1e-9):.2f} frames/s",
                        flush=True)
            if background_prefetch is not None:
                background_prefetch.close()
                background_prefetch = None
    finally:
        finalize_started = time.perf_counter()
        if background_prefetch is not None:
            background_prefetch.close()
        try:
            writer.close()
            if (not observations_only
                    and reconstruction_backend == "disparity"
                    and config.stereo_offline_cutoff_hz > 0.0):
                smoothing_started = time.perf_counter()
                stereo_smoothing_summary = smooth_stereo_disparity_hdf5(
                    output / "processed_shapes.h5", registration,
                    cutoff_hz=config.stereo_offline_cutoff_hz,
                    huber_delta_px=config.stereo_offline_huber_delta_px,
                    maximum_gap_ms=config.temporal_disparity_max_gap_ms,
                    curvature_smoothing_mm=config.curvature_smoothing_mm,
                    curvature_spline_bases=config.curvature_spline_bases)
                timing_totals_s["stereo_offline_smoothing"] = (
                    time.perf_counter() - smoothing_started)
            if (not observations_only
                    and reconstruction_backend == "disparity"
                    and config.interface_offline_cutoff_hz > 0.0):
                smoothing_started = time.perf_counter()
                interface_smoothing_summary = smooth_interface_hdf5(
                    output / "processed_shapes.h5",
                    cutoff_hz=config.interface_offline_cutoff_hz,
                    huber_delta_mm=config.interface_offline_huber_delta_mm,
                    iterations=config.interface_offline_iterations,
                    maximum_gap_ms=config.max_temporal_prompt_gap_ms,
                    curvature_smoothing_mm=config.curvature_smoothing_mm,
                    curvature_spline_bases=config.curvature_spline_bases,
                    nominal_distal_length_mm=nominal_distal_length_mm,
                    length_gate_mm=(
                        float("inf")
                        if marker_backend_active
                        else config.interface_length_gate_mm),
                    session_prior_relative_weight=(
                        0.0 if marker_backend_active
                        else config.interface_session_prior_relative_weight))
                timing_totals_s["interface_offline_smoothing"] = (
                    time.perf_counter() - smoothing_started)
            if (not observations_only
                    and config.spline_temporal_cutoff_hz > 0.0):
                smoothing_started = time.perf_counter()
                spline_temporal_smoothing_summary = (
                    smooth_distal_spline_coefficients_hdf5(
                        output / "processed_shapes.h5",
                        cutoff_hz=config.spline_temporal_cutoff_hz,
                        huber_delta_mm=(
                            config.spline_temporal_huber_delta_mm),
                        iterations=config.spline_temporal_iterations,
                        maximum_gap_ms=config.spline_temporal_max_gap_ms,
                        basis_count=config.curvature_spline_bases,
                        terminal_cutoff_hz=(
                            config.spline_temporal_terminal_cutoff_hz),
                        terminal_basis_count=(
                            config.spline_temporal_terminal_basis_count),
                        outlier_sigma=(
                            config.spline_temporal_outlier_sigma),
                        outlier_floor_mm=(
                            config.spline_temporal_outlier_floor_mm),
                        frame_outlier_fraction=(
                            config.spline_temporal_frame_outlier_fraction),
                        terminal_outlier_sample_count=(
                            config.spline_temporal_terminal_outlier_samples),
                        max_learning_mask_width_px=(
                            config.marked_max_mask_effective_width_px
                            if marker_backend_active
                            else config.max_mask_effective_width_px),
                        observation_blend=(
                            config.spline_temporal_observation_blend),
                        terminal_observation_blend=(
                            config.spline_temporal_terminal_observation_blend),
                        reject_recovered_outliers=(
                            reconstruction_backend == "disparity")))
                timing_totals_s["spline_temporal_smoothing"] = (
                    time.perf_counter() - smoothing_started)
            if not observations_only:
                metrics_started = time.perf_counter()
                _refresh_final_geometry_metrics(
                    output / "processed_shapes.h5", registration)
                (final_reprojection_rejected,
                 final_learning_valid) = _apply_final_learning_quality(
                    output / "processed_shapes.h5",
                    config.max_mean_reprojection_px,
                    config.max_p95_reprojection_px,
                    config.max_tip_endpoint_error_px,
                    config.marker_tip_endpoint_width_scale,
                    config.marker_tip_endpoint_width_cap_px,
                    config.max_final_mask_outside_fraction,
                    config.max_final_mask_distance_p95_px)
                interpolation_summary = (
                    interpolate_short_spline_gaps_hdf5(
                        output / "processed_shapes.h5",
                        maximum_gap_ms=config.spline_interpolation_max_gap_ms,
                        maximum_good_island_frames=(
                            config.spline_interpolation_max_good_island_frames))
                    if config.spline_temporal_cutoff_hz > 0.0 else {
                        "interpolated_frame_count": 0,
                        "interpolated_runs": [],
                        "maximum_gap_ms": config.spline_interpolation_max_gap_ms})
                if interpolation_summary["interpolated_frame_count"]:
                    _refresh_final_geometry_metrics(
                        output / "processed_shapes.h5", registration)
                    (final_reprojection_rejected,
                     final_learning_valid) = _apply_final_learning_quality(
                        output / "processed_shapes.h5",
                        config.max_mean_reprojection_px,
                        config.max_p95_reprojection_px,
                        config.max_tip_endpoint_error_px,
                        config.marker_tip_endpoint_width_scale,
                        config.marker_tip_endpoint_width_cap_px,
                        config.max_final_mask_outside_fraction,
                        config.max_final_mask_distance_p95_px)
                spline_temporal_smoothing_summary[
                    "spline_gap_interpolation"] = interpolation_summary
                spline_temporal_smoothing_summary[
                    "final_reprojection_rejected_count"] = (
                        final_reprojection_rejected)
                spline_temporal_smoothing_summary[
                    "learning_valid_count"] = final_learning_valid
                timing_totals_s["final_geometry_metrics"] = (
                    time.perf_counter() - metrics_started)
        finally:
            timing_totals_s["hdf5_write_background"] = writer.write_time_s
            overlay.close()
            if segmenter is not None:
                segmenter.close()
            if propagated_cache is not None:
                propagated_cache.close()
            if chromatic_executor is not None:
                chromatic_executor.shutdown(wait=True)
            timing_totals_s["output_finalize"] = (
                timing_totals_s.get("output_finalize", 0.0)
                + time.perf_counter() - finalize_started)

    if interface_smoothing_summary:
        import h5py
        with h5py.File(output / "processed_shapes.h5", "r") as smoothed_file:
            smoothed_fields = (
                "distal_length_smoothed_mm", "interface_smoothed_s_mm",
                "interface_smoothing_adjustment_mm",
                "interface_smoothing_weight")
            values = {
                name: smoothed_file[f"quality/{name}"][:]
                for name in smoothed_fields}
        for index, row in enumerate(summary_rows):
            for name in smoothed_fields:
                row[name] = values[name][index]

    if spline_temporal_smoothing_summary:
        import h5py
        with h5py.File(output / "processed_shapes.h5", "r") as smoothed_file:
            smoothed_fields = (
                "shape_coefficient_innovation_rms_mm",
                "shape_temporal_adjustment_rms_mm",
                "shape_temporal_outlier_fraction",
                "shape_temporal_frame_outlier",
                "shape_temporal_supported",
                "shape_temporal_terminal_outlier",
                "shape_temporal_long_gap_unsupported",
                "mask_width_learning_rejected")
            values = {
                name: smoothed_file[f"quality/{name}"][:]
                for name in smoothed_fields}
            learning_valid = smoothed_file["frames/learning_valid"][:]
            rejection_flags = smoothed_file[
                "frames/learning_rejection_flags"][:]
        for index, row in enumerate(summary_rows):
            for name in smoothed_fields:
                row[name] = values[name][index]
            row["learning_valid"] = int(learning_valid[index])
            row["learning_rejection_flags"] = int(rejection_flags[index])

    if defer_final_overlay:
        overlay_started = time.perf_counter()
        if config.store_masks:
            final_mask_h5 = output / "processed_shapes.h5"
        elif segmentation_backend == "cached":
            final_mask_h5 = Path(cached_mask_h5).resolve()
        elif segmentation_backend == "propagated":
            final_mask_h5 = Path(propagated_mask_h5).resolve()
        else:
            raise ValueError(
                "final smoothed overlay requires stored or cached masks")
        _write_final_overlays(
            session, output, records, registration, final_mask_h5,
            output_fps, config.video_scale,
            write_video=write_video,
            write_snapshots=write_snapshots,
            snapshot_indices=snapshot_indices)
        timing_totals_s["overlay_video_final"] = (
            time.perf_counter() - overlay_started)

    fieldnames = []
    for row in summary_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    finalize_started = time.perf_counter()
    with (output / "frame_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(summary_rows)
    timing_totals_s["output_finalize"] += (
        time.perf_counter() - finalize_started)
    valid_count = counts.get("valid", 0)
    observation_valid_count = counts.get("observations_only", 0)
    processed_count = len(summary_rows)
    elapsed_s = time.perf_counter() - started
    summary = {
        "session": session.name,
        "output_dir": str(output),
        "frame_count": len(records),
        "processed_frame_count": processed_count,
        "valid_frame_count": valid_count,
        "valid_fraction": valid_count / max(len(records), 1),
        "observation_valid_frame_count": observation_valid_count,
        "observation_valid_fraction": (
            observation_valid_count / max(len(records), 1)),
        "status_counts": counts,
        "elapsed_s": elapsed_s,
        "initialization_s": initialization_s,
        "timing": {
            key: {
                "total_s": value,
                "mean_ms_per_frame": 1000.0 * value / max(processed_count, 1),
                "fraction_of_elapsed": value / max(elapsed_s, 1e-9),
            }
            for key, value in sorted(timing_totals_s.items())
        },
        "output_fps": output_fps,
        "interface_smoothing": interface_smoothing_summary,
        "stereo_smoothing": stereo_smoothing_summary,
        "spline_temporal_smoothing": spline_temporal_smoothing_summary,
    }
    with (output / "processing_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument(
        "--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--segmentation-backend",
        choices=("sam", "propagated", "cached", "hsv", "chromatic_markers"),
        default="sam")
    parser.add_argument(
        "--reconstruction-backend",
        choices=("disparity", "joint_spline"), default=None,
        help=("joint_spline refines one 3D B-spline against both cyan "
              "centerlines without choosing a reference eye; observation "
              "resumes inherit the source backend"))
    parser.add_argument("--propagated-mask-h5", default=None)
    parser.add_argument(
        "--cached-mask-h5", default=None,
        help="completed processed_shapes.h5 whose stored SAM masks are reused")
    parser.add_argument(
        "--resume-from", choices=("observations", "stereo", "joint"),
        default=None,
        help=("restart from a completed HDF5: observations reruns stereo and "
              "all 3D stages; stereo reruns the joint 3D fit and temporal "
              "stages; joint reruns only temporal/final geometry stages"))
    parser.add_argument(
        "--resume-h5", default=None,
        help="processed_shapes.h5 used by --resume-from")
    parser.add_argument(
        "--observations-only", action="store_true",
        help=("decode and persist masks, 2D centerlines, and marker tracks, "
              "then stop before stereo/3D fitting; the output can be reused "
              "with --resume-from observations"))
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--prompt-json", default=None)
    parser.add_argument(
        "--window", choices=("trajectory", "run_and_return", "recording"),
        default="run_and_return")
    parser.add_argument("--start-ns", type=int, default=None)
    parser.add_argument("--end-ns", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--full-samples", type=int, default=128)
    parser.add_argument("--distal-samples", type=int, default=64)
    parser.add_argument("--stereo-samples", type=int, default=96)
    parser.add_argument("--smooth2d", type=float, default=None)
    parser.add_argument("--disparity-first-difference-weight", type=float, default=0.25)
    parser.add_argument("--disparity-second-difference-weight", type=float, default=10.0)
    parser.add_argument("--disparity-huber-delta-px", type=float, default=1.5)
    parser.add_argument("--temporal-disparity-weight", type=float, default=2.0)
    parser.add_argument("--temporal-disparity-max-gap-ms", type=float, default=1000.0)
    parser.add_argument("--temporal-disparity-decay-ms", type=float, default=500.0)
    parser.add_argument("--stereo-reference-hysteresis-score", type=float, default=3.0)
    parser.add_argument(
        "--overlap-aware-reconstruction",
        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-row-search-px", type=int, default=2)
    parser.add_argument(
        "--overlap-self-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--overlap-self-distance-px", type=float, default=8.0)
    parser.add_argument(
        "--overlap-min-arclength-separation", type=float, default=0.12)
    parser.add_argument("--centerline-tip-depth-weight", type=float, default=5.0)
    parser.add_argument(
        "--max-centerline-tip-epipolar-error-px", type=float, default=8.0)
    parser.add_argument(
        "--terminal-disparity-refinement",
        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminal-disparity-fraction", type=float, default=0.20)
    parser.add_argument(
        "--terminal-disparity-smoothness-scale", type=float, default=0.10)
    parser.add_argument(
        "--terminal-disparity-observation-weight", type=float, default=4.0)
    parser.add_argument(
        "--terminal-disparity-tip-weight", type=float, default=20.0)
    parser.add_argument(
        "--terminal-disparity-max-p95-degradation-px", type=float, default=0.25)
    parser.add_argument("--stereo-offline-cutoff-hz", type=float, default=2.0)
    parser.add_argument(
        "--stereo-offline-huber-delta-px", type=float, default=1.5)
    parser.add_argument("--temporal-shape-cutoff-hz", type=float, default=0.0)
    parser.add_argument("--distal-length-mm", type=float, default=60.0)
    parser.add_argument("--marked-distal-length-mm", type=float, default=57.0)
    parser.add_argument("--distal-length-prior-sigma-mm", type=float, default=4.0)
    parser.add_argument("--interface-color-sigma-mm", type=float, default=3.0)
    parser.add_argument("--interface-temporal-sigma-mm", type=float, default=1.5)
    parser.add_argument("--interface-temporal-cutoff-hz", type=float, default=4.0)
    parser.add_argument("--interface-projection-window-mm", type=float, default=15.0)
    parser.add_argument("--interface-offline-cutoff-hz", type=float, default=2.0)
    parser.add_argument(
        "--interface-offline-huber-delta-mm", type=float, default=2.0)
    parser.add_argument("--interface-offline-iterations", type=int, default=4)
    parser.add_argument("--interface-length-gate-mm", type=float, default=4.0)
    parser.add_argument(
        "--interface-session-prior-relative-weight", type=float, default=0.5)
    parser.add_argument("--spline-temporal-cutoff-hz", type=float, default=3.0)
    parser.add_argument(
        "--spline-temporal-terminal-cutoff-hz", type=float, default=5.0)
    parser.add_argument(
        "--spline-temporal-terminal-basis-count", type=int, default=4)
    parser.add_argument(
        "--spline-temporal-huber-delta-mm", type=float, default=1.0)
    parser.add_argument("--spline-temporal-iterations", type=int, default=4)
    parser.add_argument(
        "--spline-temporal-max-gap-ms", type=float, default=500.0)
    parser.add_argument(
        "--spline-temporal-outlier-sigma", type=float, default=4.5)
    parser.add_argument(
        "--spline-temporal-outlier-floor-mm", type=float, default=0.75)
    parser.add_argument(
        "--spline-temporal-frame-outlier-fraction", type=float, default=0.5)
    parser.add_argument(
        "--spline-temporal-terminal-outlier-samples", type=int, default=4)
    parser.add_argument(
        "--spline-temporal-observation-blend", type=float, default=0.35)
    parser.add_argument(
        "--spline-temporal-terminal-observation-blend",
        type=float, default=0.65)
    parser.add_argument(
        "--spline-interpolation-max-gap-ms", type=float, default=650.0,
        help="maximum bracketed invalid interval repaired in the full 3D spline")
    parser.add_argument(
        "--spline-interpolation-max-good-island-frames", type=int, default=2,
        help="accepted frames this short inside a rejected interval are bridged")
    parser.add_argument(
        "--distal-boundary-search-half-width-mm", type=float, default=12.0)
    parser.add_argument("--curvature-smoothing-mm", type=float, default=0.25)
    parser.add_argument("--curvature-spline-bases", type=int, default=20)
    parser.add_argument("--max-base-endpoint-distance-mm", type=float, default=15.0)
    parser.add_argument(
        "--max-stereo-centerline-length-ratio", type=float, default=1.8)
    parser.add_argument(
        "--marked-stereo-guided-retry-ratio", type=float, default=1.23,
        help=("trigger good-eye/temporal reconstruction when marked-view "
              "projected centerline lengths differ by this ratio"))
    parser.add_argument(
        "--ill-eye-min-observation-weight", type=float, default=0.20,
        help=("minimum relative joint-fit weight of an observably shortened "
              "eye; the good eye and temporal 3D prior retain material order"))
    parser.add_argument(
        "--ill-eye-temporal-shape-sigma-mm", type=float, default=6.0,
        help=("soft coefficient-space uncertainty of the preceding accepted "
              "3D shape; the unified fit uses 65 percent of this sigma in "
              "all temporally adjacent frames"))
    parser.add_argument(
        "--marked-ill-epipolar-ambiguity-fraction", type=float, default=0.30,
        help=("activate good-eye ordering before a full length collapse when "
              "this fraction of epipolar rows has multiple target branches"))
    parser.add_argument(
        "--stereo-ill-switch-confirm-frames", type=int, default=5,
        help=("consecutive frames required to enter, switch, or release an "
              "ill-eye state"))
    parser.add_argument(
        "--max-inferred-centerline-temporal-p95-px", type=float,
        default=8.0)
    parser.add_argument(
        "--max-inferred-centerline-temporal-max-px", type=float,
        default=14.0)
    parser.add_argument(
        "--max-inferred-sharp-turn-clusters", type=int, default=1,
        help="maximum separated sharp-turn clusters after mask snapping")
    parser.add_argument("--stereo-reference-switch-ratio", type=float, default=1.15)
    parser.add_argument("--max-mask-effective-width-px", type=float, default=20.0)
    parser.add_argument(
        "--marked-max-mask-effective-width-px", type=float, default=28.0)
    parser.add_argument("--max-temporal-prompt-gap-ms", type=float, default=100.0)
    parser.add_argument("--max-mean-reprojection-px", type=float, default=5.0)
    parser.add_argument("--max-p95-reprojection-px", type=float, default=12.0)
    parser.add_argument("--max-tip-epipolar-error-px", type=float, default=5.0)
    parser.add_argument("--max-tip-endpoint-error-px", type=float, default=8.0)
    parser.add_argument(
        "--marker-tip-endpoint-width-scale", type=float, default=0.40,
        help="tip endpoint uncertainty as a fraction of marker-3 image width")
    parser.add_argument(
        "--marker-tip-endpoint-width-cap-px", type=float, default=12.0)
    parser.add_argument(
        "--max-final-mask-outside-fraction", type=float, default=0.20)
    parser.add_argument(
        "--max-final-mask-distance-p95-px", type=float, default=6.0)
    parser.add_argument("--chromatic-min-saturation", type=int, default=55)
    parser.add_argument("--chromatic-min-value", type=int, default=30)
    parser.add_argument(
        "--chromatic-background-subtraction",
        action=argparse.BooleanOptionalAction, default=True,
        help=("remove static saturated scene pixels using a sparse temporal "
              "median of the fixed-camera SVO"))
    parser.add_argument("--chromatic-background-samples", type=int, default=21)
    parser.add_argument(
        "--chromatic-background-difference", type=int, default=18,
        help="minimum max-channel change from the static background")
    parser.add_argument("--marker-min-confidence", type=float, default=0.50)
    parser.add_argument(
        "--marked-epipolar-sweep-deficit-px", type=float, default=4.0,
        help=("minimum rectified-row span mismatch before the good eye "
              "supplies an epipolar-line turn constraint"))
    parser.add_argument(
        "--marked-epipolar-sweep-row-half-width-px", type=int, default=3)
    parser.add_argument("--marker-disparity-weight", type=float, default=8.0)
    parser.add_argument("--joint-spline-basis-count", type=int, default=20)
    parser.add_argument("--joint-spline-max-nfev", type=int, default=35)
    parser.add_argument(
        "--marker-interface-weight-scale", type=float, default=2.0)
    parser.add_argument(
        "--marker-tip-weight-scale", type=float, default=3.0)
    parser.add_argument(
        "--max-marker-epipolar-error-px", type=float, default=6.0)
    parser.add_argument(
        "--marked-distal-length-prior-sigma-mm", type=float, default=3.0,
        help=("soft interface-to-tip length prior for marked recordings; "
              "never enforced as an equality"))
    parser.add_argument(
        "--marked-interface-marker-sigma-mm", type=float, default=0.75,
        help="soft uncertainty of marker-0 interface observations")
    parser.add_argument(
        "--marked-sheath-exclusion-radius-mm", type=float, default=7.0)
    parser.add_argument(
        "--marked-sheath-exclusion-length-mm", type=float, default=60.0)
    parser.add_argument(
        "--min-temporal-centerline-coverage", type=float, default=0.55)
    parser.add_argument(
        "--temporal-centerline-tolerance-px", type=float, default=18.0)
    parser.add_argument(
        "--max-temporal-centerline-p95-px", type=float, default=35.0)
    parser.add_argument(
        "--sam-frame-batch-size", type=int, default=1,
        help=("stereo timestamps per independent SAM encoder batch; values >1 "
              "use automatic-first prompting and selective temporal rescue"))
    parser.add_argument(
        "--prompt-workers", type=int, default=4,
        help="CPU workers for frame-local colour/prompt preprocessing")
    parser.add_argument(
        "--chromatic-eye-workers", type=int, default=2,
        help=("parallel workers for independent left/right chromatic mask and "
              "centerline extraction (default: 2)"))
    parser.add_argument(
        "--sam-postprocess-workers", type=int, default=2,
        help="CPU workers for independent SAM mask finishing")
    parser.add_argument(
        "--preprocess-chunk-size", type=int, default=16,
        help="bounded frame chunk for CPU prompt preprocessing")
    parser.add_argument(
        "--prefetch-frames", type=int, default=16,
        help="bounded decoded/preprocessed frame queue; 0 disables overlap")
    parser.add_argument(
        "--hdf-buffer-frames", type=int, default=128,
        help="frames combined into each contiguous HDF5 write")
    parser.add_argument(
        "--hdf-queue-chunks", type=int, default=2,
        help="bounded completed HDF5 chunks queued to the writer thread")
    parser.add_argument(
        "--store-masks", action=argparse.BooleanOptionalAction, default=None,
        help="store bit-packed masks (default: off for cached, on otherwise)")
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument(
        "--write-snapshots", action="store_true",
        help="write evenly spaced full-resolution PNG overlays")
    parser.add_argument(
        "--snapshot-count", type=int, default=12,
        help="maximum evenly spaced stereo overlay snapshots (default: 12)")
    parser.add_argument(
        "--snapshot-frames", default=None,
        help=("comma-separated output-frame indices to snapshot; overrides "
              "--snapshot-count"))
    parser.add_argument("--video-scale", type=float, default=0.5)
    return parser


def _resume_source_reconstruction_backend(path: str | Path) -> str:
    """Read the 3D backend that produced an observation cache."""
    import h5py

    with h5py.File(Path(path).resolve(), "r") as source:
        metadata = json.loads(source.attrs.get("metadata_json", "{}"))
        backend = metadata.get("reconstruction_backend")
        if backend in ("disparity", "joint_spline"):
            return str(backend)
        if "joint/coefficients_base_mm" in source:
            coefficients = source["joint/coefficients_base_mm"]
            for start in range(0, len(coefficients), 256):
                if np.any(np.isfinite(coefficients[start:start + 256])):
                    return "joint_spline"
    return "disparity"


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    if args.observations_only and args.resume_from is not None:
        raise ValueError(
            "--observations-only creates a cache and cannot be combined with "
            "--resume-from")
    if args.resume_from is not None and args.resume_h5 is None:
        raise ValueError("--resume-from requires --resume-h5")
    if args.reconstruction_backend is None:
        args.reconstruction_backend = (
            _resume_source_reconstruction_backend(args.resume_h5)
            if args.resume_from == "observations"
            else "disparity")
    if args.resume_from == "observations":
        if args.cached_mask_h5 is not None:
            raise ValueError(
                "use either --resume-h5 or --cached-mask-h5, not both")
        args.segmentation_backend = "cached"
        args.cached_mask_h5 = args.resume_h5
    elif args.resume_from in ("stereo", "joint") and args.outdir is None:
        raise ValueError("--resume-from stereo/joint requires --outdir")
    config = ImageProcessingConfig(
        full_samples=args.full_samples,
        distal_samples=args.distal_samples,
        stereo_samples=args.stereo_samples,
        smooth_2d=args.smooth2d,
        disparity_first_difference_weight=args.disparity_first_difference_weight,
        disparity_second_difference_weight=args.disparity_second_difference_weight,
        disparity_huber_delta_px=args.disparity_huber_delta_px,
        temporal_disparity_weight=args.temporal_disparity_weight,
        temporal_disparity_max_gap_ms=args.temporal_disparity_max_gap_ms,
        temporal_disparity_decay_ms=args.temporal_disparity_decay_ms,
        stereo_reference_hysteresis_score=(
            args.stereo_reference_hysteresis_score),
        overlap_aware_reconstruction=args.overlap_aware_reconstruction,
        overlap_row_search_px=args.overlap_row_search_px,
        overlap_self_fraction_threshold=args.overlap_self_fraction_threshold,
        overlap_self_distance_px=args.overlap_self_distance_px,
        overlap_min_arclength_separation=(
            args.overlap_min_arclength_separation),
        centerline_tip_depth_weight=args.centerline_tip_depth_weight,
        max_centerline_tip_epipolar_error_px=(
            args.max_centerline_tip_epipolar_error_px),
        terminal_disparity_refinement=args.terminal_disparity_refinement,
        terminal_disparity_fraction=args.terminal_disparity_fraction,
        terminal_disparity_smoothness_scale=(
            args.terminal_disparity_smoothness_scale),
        terminal_disparity_observation_weight=(
            args.terminal_disparity_observation_weight),
        terminal_disparity_tip_weight=args.terminal_disparity_tip_weight,
        terminal_disparity_max_p95_degradation_px=(
            args.terminal_disparity_max_p95_degradation_px),
        stereo_offline_cutoff_hz=args.stereo_offline_cutoff_hz,
        stereo_offline_huber_delta_px=args.stereo_offline_huber_delta_px,
        temporal_shape_cutoff_hz=args.temporal_shape_cutoff_hz,
        distal_length_mm=args.distal_length_mm,
        marked_distal_length_mm=args.marked_distal_length_mm,
        distal_length_prior_sigma_mm=args.distal_length_prior_sigma_mm,
        interface_color_sigma_mm=args.interface_color_sigma_mm,
        interface_temporal_sigma_mm=args.interface_temporal_sigma_mm,
        interface_temporal_cutoff_hz=args.interface_temporal_cutoff_hz,
        interface_projection_window_mm=args.interface_projection_window_mm,
        interface_offline_cutoff_hz=args.interface_offline_cutoff_hz,
        interface_offline_huber_delta_mm=(
            args.interface_offline_huber_delta_mm),
        interface_offline_iterations=args.interface_offline_iterations,
        interface_length_gate_mm=args.interface_length_gate_mm,
        interface_session_prior_relative_weight=(
            args.interface_session_prior_relative_weight),
        spline_temporal_cutoff_hz=args.spline_temporal_cutoff_hz,
        spline_temporal_terminal_cutoff_hz=(
            args.spline_temporal_terminal_cutoff_hz),
        spline_temporal_terminal_basis_count=(
            args.spline_temporal_terminal_basis_count),
        spline_temporal_huber_delta_mm=(
            args.spline_temporal_huber_delta_mm),
        spline_temporal_iterations=args.spline_temporal_iterations,
        spline_temporal_max_gap_ms=args.spline_temporal_max_gap_ms,
        spline_temporal_outlier_sigma=args.spline_temporal_outlier_sigma,
        spline_temporal_outlier_floor_mm=(
            args.spline_temporal_outlier_floor_mm),
        spline_temporal_frame_outlier_fraction=(
            args.spline_temporal_frame_outlier_fraction),
        spline_temporal_terminal_outlier_samples=(
            args.spline_temporal_terminal_outlier_samples),
        spline_temporal_observation_blend=(
            args.spline_temporal_observation_blend),
        spline_temporal_terminal_observation_blend=(
            args.spline_temporal_terminal_observation_blend),
        spline_interpolation_max_gap_ms=(
            args.spline_interpolation_max_gap_ms),
        spline_interpolation_max_good_island_frames=(
            args.spline_interpolation_max_good_island_frames),
        distal_boundary_search_half_width_mm=(
            args.distal_boundary_search_half_width_mm),
        curvature_smoothing_mm=args.curvature_smoothing_mm,
        curvature_spline_bases=args.curvature_spline_bases,
        max_base_endpoint_distance_mm=args.max_base_endpoint_distance_mm,
        max_stereo_centerline_length_ratio=(
            args.max_stereo_centerline_length_ratio),
        marked_stereo_guided_retry_ratio=(
            args.marked_stereo_guided_retry_ratio),
        ill_eye_min_observation_weight=(
            args.ill_eye_min_observation_weight),
        ill_eye_temporal_shape_sigma_mm=(
            args.ill_eye_temporal_shape_sigma_mm),
        marked_ill_epipolar_ambiguity_fraction=(
            args.marked_ill_epipolar_ambiguity_fraction),
        stereo_ill_switch_confirm_frames=(
            args.stereo_ill_switch_confirm_frames),
        max_inferred_centerline_temporal_p95_px=(
            args.max_inferred_centerline_temporal_p95_px),
        max_inferred_centerline_temporal_max_px=(
            args.max_inferred_centerline_temporal_max_px),
        max_inferred_sharp_turn_clusters=(
            args.max_inferred_sharp_turn_clusters),
        stereo_reference_switch_ratio=args.stereo_reference_switch_ratio,
        max_mask_effective_width_px=args.max_mask_effective_width_px,
        marked_max_mask_effective_width_px=(
            args.marked_max_mask_effective_width_px),
        max_temporal_prompt_gap_ms=args.max_temporal_prompt_gap_ms,
        max_mean_reprojection_px=args.max_mean_reprojection_px,
        max_p95_reprojection_px=args.max_p95_reprojection_px,
        max_tip_epipolar_error_px=args.max_tip_epipolar_error_px,
        max_tip_endpoint_error_px=args.max_tip_endpoint_error_px,
        marker_tip_endpoint_width_scale=(
            args.marker_tip_endpoint_width_scale),
        marker_tip_endpoint_width_cap_px=(
            args.marker_tip_endpoint_width_cap_px),
        max_final_mask_outside_fraction=(
            args.max_final_mask_outside_fraction),
        max_final_mask_distance_p95_px=(
            args.max_final_mask_distance_p95_px),
        chromatic_min_saturation=args.chromatic_min_saturation,
        chromatic_min_value=args.chromatic_min_value,
        chromatic_background_subtraction=(
            args.chromatic_background_subtraction),
        chromatic_background_samples=args.chromatic_background_samples,
        chromatic_background_difference=args.chromatic_background_difference,
        marker_min_confidence=args.marker_min_confidence,
        marked_epipolar_sweep_deficit_px=(
            args.marked_epipolar_sweep_deficit_px),
        marked_epipolar_sweep_row_half_width_px=(
            args.marked_epipolar_sweep_row_half_width_px),
        marker_disparity_weight=args.marker_disparity_weight,
        joint_spline_basis_count=args.joint_spline_basis_count,
        joint_spline_max_nfev=args.joint_spline_max_nfev,
        marker_interface_weight_scale=args.marker_interface_weight_scale,
        marker_tip_weight_scale=args.marker_tip_weight_scale,
        max_marker_epipolar_error_px=args.max_marker_epipolar_error_px,
        marked_distal_length_prior_sigma_mm=(
            args.marked_distal_length_prior_sigma_mm),
        marked_interface_marker_sigma_mm=(
            args.marked_interface_marker_sigma_mm),
        marked_sheath_exclusion_radius_mm=(
            args.marked_sheath_exclusion_radius_mm),
        marked_sheath_exclusion_length_mm=(
            args.marked_sheath_exclusion_length_mm),
        min_temporal_centerline_coverage=(
            args.min_temporal_centerline_coverage),
        temporal_centerline_tolerance_px=(
            args.temporal_centerline_tolerance_px),
        max_temporal_centerline_p95_px=(
            args.max_temporal_centerline_p95_px),
        sam_frame_batch_size=args.sam_frame_batch_size,
        sam_postprocess_workers=args.sam_postprocess_workers,
        prompt_workers=args.prompt_workers,
        chromatic_eye_workers=args.chromatic_eye_workers,
        preprocess_chunk_size=args.preprocess_chunk_size,
        prefetch_frames=args.prefetch_frames,
        hdf_buffer_frames=args.hdf_buffer_frames,
        hdf_queue_chunks=args.hdf_queue_chunks,
        store_masks=(
            args.store_masks if args.store_masks is not None
            else (args.segmentation_backend != "cached"
                  or args.resume_from == "observations")),
        video_scale=args.video_scale)
    parsed_snapshot_frames = (
        None if not args.snapshot_frames else
        [int(value) for value in args.snapshot_frames.split(",")
         if value.strip()])
    if args.resume_from in ("stereo", "joint"):
        summary = resume_image_session(
            args.session, args.resume_h5, args.outdir, args.resume_from,
            config, write_video=args.write_video,
            write_snapshots=args.write_snapshots,
            snapshot_count=args.snapshot_count,
            snapshot_frames=parsed_snapshot_frames)
    else:
        summary = process_image_session(
            args.session,
            sam_checkpoint=args.sam_checkpoint,
            sam_config=args.sam_config,
            output_dir=args.outdir,
            prompt_json=args.prompt_json,
            config=config,
            window=args.window,
            start_ns=args.start_ns,
            end_ns=args.end_ns,
            stride=args.stride,
            max_frames=args.max_frames,
            write_video=args.write_video,
            write_snapshots=args.write_snapshots,
            snapshot_count=args.snapshot_count,
            snapshot_frames=parsed_snapshot_frames,
            device=args.device,
            segmentation_backend=args.segmentation_backend,
            reconstruction_backend=args.reconstruction_backend,
            propagated_mask_h5=args.propagated_mask_h5,
            cached_mask_h5=args.cached_mask_h5,
            observations_only=args.observations_only)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
