"""Offline four-view reconstruction for sessions recorded by two ZED rigs.

The legacy :mod:`shape_tracking.image_sequence` entry point remains the
single-rig implementation. This module creates one reusable observation cache
per rig and then fits one common robot-base B-spline to all available views.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import csv
from itertools import combinations
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import cumulative_arclength, curve_geometry, resample_polyline
from .image_sequence import ImageProcessingConfig, process_image_session
from .joint_spline_reconstruction import (
    _basis, _resample_2d, distal_image_observation,
)
from .multi_view_reconstruction import (
    ViewObservation,
    candidate_symmetric_error,
    fit_multi_view_spline,
    projected_route_topology_weight,
    triangulate_material_curve,
)
from .multi_camera import RecordedFrame, pair_recorded_frames
from .robot_data import align_robot_streams, load_robot_streams
from .sequence import load_collection_markers, select_frame_records
from .session import (
    SvoReader,
    find_svo,
    load_frame_index,
    load_session_registration,
    project_points,
)
from .spline_temporal_smoothing import (
    interpolate_short_spline_gaps_hdf5,
    smooth_distal_spline_coefficients_hdf5,
)
from .temporal_markers import repair_stereo_marker_tracks


RIGS = ("primary", "oblique")
EYES = ("left", "right")
VIEW_IDS = tuple(f"{rig}_{eye}" for rig in RIGS for eye in EYES)
LENGTH_REJECTION_FLAG = np.uint16(1 << 6)
WHOLE_RIG_REJECTION_FLAG = np.uint16(1 << 7)
ABRUPT_OBSERVATION_REJECTION_FLAG = np.uint16(1 << 8)


@dataclass(frozen=True)
class MultiCameraConfig:
    distal_samples: int = 64
    spline_basis_count: int = 20
    fit_samples: int = 64
    coverage_samples: int = 48
    nominal_length_mm: float = 57.0
    length_sigma_mm: float = 3.0
    temporal_prior_sigma_mm: float = 3.0
    max_temporal_gap_ms: float = 150.0
    velocity_update_fraction: float = 0.30
    max_coefficient_speed_mm_s: float = 300.0
    maximum_pair_offset_ms: float = 21.0
    max_nfev: int = 40
    max_final_symmetric_px: float = 12.0
    min_learning_length_mm: float = 50.0
    max_learning_length_mm: float = 65.0
    temporal_cutoff_hz: float = 3.0
    temporal_terminal_cutoff_hz: float = 3.0
    temporal_observation_blend: float = 0.15
    temporal_terminal_observation_blend: float = 0.15
    temporal_max_gap_ms: float = 500.0
    interpolation_max_gap_ms: float = 650.0
    reacquisition_rig_error_px: float = 9.0
    whole_rig_max_symmetric_px: float = 12.0
    marker_cross_view_inlier_px: float = 8.0
    marker_cross_view_outlier_px: float = 12.0
    marker_cross_view_temporal_sigma_mm: float = 4.0
    reacquisition_terminal_error_px: float = 10.0
    reacquisition_coverage_mean_px: float = 6.0
    reacquisition_length_deviation_mm: float = 6.0
    max_terminal_error_px: float = 15.0
    max_rig_coverage_mean_px: float = 10.0
    max_direct_length_deviation_mm: float = 10.0
    max_cross_rig_endpoint_disagreement_mm: float = 8.0
    excluded_rig_weight: float = 0.002
    temporal_polish_max_score_increase: float = 0.50
    ordered_stereo_samples: int = 64
    ordered_stereo_min_topology_weight: float = 0.70
    ordered_stereo_max_epipolar_p95_px: float = 4.0
    ordered_stereo_min_match_fraction: float = 0.65
    ordered_stereo_min_curve_length_mm: float = 40.0
    ordered_stereo_max_curve_length_mm: float = 80.0
    ordered_stereo_sigma_mm: float = 1.5
    ordered_stereo_weight: float = 0.55
    ordered_disparity_basis_count: int = 12
    ordered_disparity_spatial_weight: float = 8.0
    # Do not causally drag the current stereo disparity toward the last
    # accepted frame.  That state can be stale after a rejected candidate and
    # causes a depth hold followed by a snap.  Temporal regularization belongs
    # to the final zero-phase 3-D coefficient pass.
    ordered_disparity_temporal_weight: float = 0.0
    ordered_disparity_huber_px: float = 1.5
    ordered_disparity_max_temporal_gap_ms: float = 200.0
    ordered_disparity_max_right_support_p95_px: float = 8.0
    ordered_disparity_support_sigma_px: float = 5.0
    ordered_stereo_max_innovation_mm: float = 4.0
    refine_cross_rig_registration: bool = True
    cross_rig_registration_min_pairs: int = 200
    cross_rig_registration_max_initial_error_mm: float = 10.0
    cross_rig_registration_max_translation_mm: float = 8.0
    cross_rig_registration_max_rotation_deg: float = 3.0
    reject_abrupt_observation_jumps: bool = False


class ObservationCache:
    """Read one rig's packed observation-only HDF5."""

    def __init__(self, path: os.PathLike | str, rig_id: str):
        import h5py

        self.path = Path(path).resolve()
        self.rig_id = rig_id
        self.file = h5py.File(self.path, "r")
        self.svo_frames = np.asarray(self.file["frames/svo_frame"], np.int64)
        self.timestamps_ns = np.asarray(
            self.file["frames/timestamp_ns"], np.int64)
        self.observation_valid = np.asarray(
            self.file["frames/observation_valid"], dtype=bool)
        self.index_by_svo = {
            int(frame): index for index, frame in enumerate(self.svo_frames)}
        if len(self.observation_valid) != len(self.svo_frames):
            raise ValueError(
                f"observation validity length does not match frames: {path}")
        centers = {
            eye: self.file[f"images/{eye}/marker_centers_px"][:]
            for eye in EYES}
        widths = {
            eye: self.file[f"images/{eye}/marker_widths_px"][:]
            for eye in EYES}
        confidence = {
            eye: self.file[f"images/{eye}/marker_confidence"][:]
            for eye in EYES}
        observed = {
            eye: self.file[f"images/{eye}/marker_observed"][:]
            for eye in EYES}
        self.marker_tracks = repair_stereo_marker_tracks(
            centers, widths, confidence, observed, self.timestamps_ns)

    def close(self):
        self.file.close()

    def view(self, index: int, eye: str, registration,
             timestamp_offset_s: float,
             marker_override: np.ndarray | None = None) -> ViewObservation:
        name = f"images/{eye}/observed_centerline_px"
        centerline = np.asarray(self.file[name][index], float)
        centerline = centerline[np.all(np.isfinite(centerline), axis=1)]
        markers = np.asarray(
            self.marker_tracks[eye]["centers"][index]
            if marker_override is None else marker_override, float)
        interface = markers[0] if np.all(np.isfinite(markers[0])) else None
        tip = markers[3] if np.all(np.isfinite(markers[3])) else None
        distal, axial_endpoints = distal_image_observation(
            centerline, interface, tip, count=96)
        fit_markers = markers.copy()
        # Ring centroids can sit off the apparent catheter axis. The distal
        # crop already projects marker 0/3 onto the extracted centerline; use
        # those axial locations as strong soft endpoint observations.
        fit_markers[0] = axial_endpoints[0]
        fit_markers[3] = axial_endpoints[1]
        topology_weight = projected_route_topology_weight(distal, markers)
        transform = (registration.left_camera_T_base if eye == "left"
                     else registration.right_camera_T_base)
        return ViewObservation(
            view_id=f"{self.rig_id}_{eye}",
            K=registration.K,
            camera_T_base=transform,
            centerline_xy=distal,
            axial_markers_xy=fit_markers,
            weight=topology_weight,
            timestamp_offset_s=float(timestamp_offset_s),
            topology_weight=topology_weight)


def _pair_map(session: Path) -> dict[int, tuple[int, int, int]]:
    """Return primary frame -> (oblique frame, primary ns, oblique ns)."""
    recorded_path = session / "camera_frame_pairs.csv"
    if recorded_path.is_file():
        output = {}
        with recorded_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if not row.get("secondary_frame"):
                    continue
                output[int(row["reference_frame"])] = (
                    int(row["secondary_frame"]),
                    int(row["reference_timestamp_ns"]),
                    int(row["secondary_timestamp_ns"]),
                )
        if output:
            return output
    # Compatibility fallback for early dual-camera recordings that predate the
    # pairing sidecar.
    primary = load_frame_index(session, rig_id="primary")
    oblique = load_frame_index(session, rig_id="oblique")
    generated = pair_recorded_frames(
        [RecordedFrame(item.svo_frame, item.timestamp_ns) for item in primary],
        [RecordedFrame(item.svo_frame, item.timestamp_ns) for item in oblique])
    return {
        int(item.reference_frame): (
            int(item.secondary_frame), int(item.reference_timestamp_ns),
            int(item.secondary_timestamp_ns))
        for item in generated if item.secondary_frame is not None}


def _build_one_observation_cache(arguments) -> tuple[str, dict]:
    (session, rig_output, image_config, window, start_ns, end_ns,
     stride, max_frames, rig_id) = arguments
    summary = process_image_session(
        session, sam_checkpoint=None, output_dir=rig_output,
        config=replace(
            image_config, store_masks=True,
            dual_camera_observation_profile=True),
        window=window, start_ns=start_ns, end_ns=end_ns, stride=stride,
        max_frames=max_frames, segmentation_backend="chromatic_markers",
        reconstruction_backend="joint_spline", observations_only=True,
        rig_id=rig_id)
    return rig_id, summary


def build_observation_caches(
        session: Path, output: Path, image_config: ImageProcessingConfig,
        window: str, start_ns: int | None, end_ns: int | None,
        stride: int, max_frames: int | None,
        rig_workers: int = 2) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    arguments = [
        (session, output / f"observations_{rig_id}", image_config,
         window, start_ns, end_ns, stride, max_frames, rig_id)
        for rig_id in RIGS]
    summaries = {}
    workers = int(np.clip(int(rig_workers), 1, len(RIGS)))
    if workers == 1:
        for item in arguments:
            rig_id, summary = _build_one_observation_cache(item)
            summaries[rig_id] = summary
    else:
        # Spawn, rather than fork, because each worker owns an independent ZED
        # decoder/CUDA context. Both SVOs and output files are also disjoint.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
                max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(_build_one_observation_cache, item): item[-1]
                for item in arguments}
            for future in as_completed(futures):
                rig_id, summary = future.result()
                summaries[rig_id] = summary
    for rig_id in RIGS:
        summary = summaries[rig_id]
        if summary["processed_frame_count"] != summary["frame_count"]:
            raise RuntimeError(
                f"{rig_id} observation cache is incomplete: {summary}")
        paths[rig_id] = str(
            output / f"observations_{rig_id}" / "processed_shapes.h5")
    return paths


def _initializer_candidates(
        observations: list[ViewObservation],
        previous_points: np.ndarray | None,
        nominal_length_mm: float) -> list[tuple[float, np.ndarray, str]]:
    candidates = []
    groups = [
        ("all", observations),
        *[(rig, [item for item in observations
                 if item.view_id.startswith(rig + "_")]) for rig in RIGS],
    ]
    for name, group in groups:
        if len(group) < 2:
            continue
        try:
            points = triangulate_material_curve(group, samples=64)
            length = float(cumulative_arclength(points)[-1])
            score = candidate_symmetric_error(points, observations)
            score += abs(length - nominal_length_mm) / 3.0
            candidates.append((score, points, name))
        except (ArithmeticError, RuntimeError, ValueError,
                np.linalg.LinAlgError):
            pass
    if previous_points is not None:
        score = candidate_symmetric_error(previous_points, observations)
        length = float(cumulative_arclength(previous_points)[-1])
        score += abs(length - nominal_length_mm) / 3.0
        candidates.append((score, previous_points.copy(), "temporal"))
    return sorted(candidates, key=lambda item: item[0])


def _adaptive_view_weights(
        initial: np.ndarray,
        observations: list[ViewObservation]) -> list[ViewObservation]:
    """Downweight a route that disagrees with the multi-view consensus."""
    output = []
    for observation in observations:
        projected = project_points(
            observation.K, observation.camera_T_base, initial)[0]
        observed = np.asarray(observation.centerline_xy, float)
        from scipy.spatial import cKDTree
        model = cKDTree(observed).query(projected)[0]
        coverage = cKDTree(projected).query(observed)[0]
        error = 0.5 * (float(np.mean(model)) + float(np.mean(coverage)))
        consensus_weight = float(np.clip(
            np.exp(-(error / 8.0) ** 2), 0.12, 1.0))
        lower = min(0.04, max(float(observation.weight), 1e-6))
        weight = float(np.clip(
            observation.weight * consensus_weight, lower,
            max(float(observation.weight), lower)))
        output.append(replace(observation, weight=weight))
    return output


def _projection_matrix(observation: ViewObservation) -> np.ndarray:
    return (np.asarray(observation.K, dtype=np.float64)
            @ np.asarray(observation.camera_T_base, dtype=np.float64)[:3])


def _triangulate_marker_pair(
        first_xy: np.ndarray, first_projection: np.ndarray,
        second_xy: np.ndarray, second_projection: np.ndarray
) -> np.ndarray | None:
    rows = np.asarray([
        first_xy[0] * first_projection[2] - first_projection[0],
        first_xy[1] * first_projection[2] - first_projection[1],
        second_xy[0] * second_projection[2] - second_projection[0],
        second_xy[1] * second_projection[2] - second_projection[1],
    ])
    _, _, vt = np.linalg.svd(rows, full_matrices=False)
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-10:
        return None
    point = homogeneous[:3] / homogeneous[3] * 1000.0
    return point if np.all(np.isfinite(point)) else None


def _rigid_alignment(
        source: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform mapping ``source`` onto ``target``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("rigid-alignment inputs must be matching (N, 3) arrays")
    if len(source) < 3:
        raise ValueError("rigid alignment requires at least three points")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _robust_rigid_alignment(
        source: np.ndarray, target: np.ndarray,
        maximum_initial_error_mm: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IRLS-like trimmed rigid alignment for cross-rig marker positions."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.all(np.isfinite(source), axis=1) & np.all(
        np.isfinite(target), axis=1)
    finite &= np.linalg.norm(source - target, axis=1) <= float(
        maximum_initial_error_mm)
    inliers = finite.copy()
    for _ in range(6):
        if np.count_nonzero(inliers) < 3:
            break
        rotation, translation = _rigid_alignment(
            source[inliers], target[inliers])
        residual = np.linalg.norm(
            (rotation @ source.T).T + translation - target, axis=1)
        center = float(np.median(residual[inliers]))
        mad = float(np.median(np.abs(residual[inliers] - center)))
        cutoff = min(float(maximum_initial_error_mm),
                     max(1.0, center + 3.5 * 1.4826 * mad))
        updated = finite & (residual <= cutoff)
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    if np.count_nonzero(inliers) < 3:
        raise ValueError("cross-rig rigid alignment has too few inliers")
    rotation, translation = _rigid_alignment(
        source[inliers], target[inliers])
    return rotation, translation, inliers


def _refine_cross_rig_registration(
        registrations: dict[str, object], marker_centers: np.ndarray,
        marker_inferred: np.ndarray,
        marker_temporal_outliers: dict[str, np.ndarray],
        config: MultiCameraConfig,
) -> tuple[dict[str, object], dict]:
    """Align oblique stereo triangulations to the primary base coordinates.

    The board registration remains the initial estimate and the primary rig
    remains the robot-base anchor. Only a small, robust SE(3) correction is
    permitted. Cross-camera-inferred markers and temporally corrupt rig rows
    are excluded so repaired observations cannot calibrate the cameras.
    """
    diagnostic = {
        "enabled": bool(config.refine_cross_rig_registration),
        "applied": False,
        "anchor_rig": "primary",
        "corrected_rig": "oblique",
    }
    if not config.refine_cross_rig_registration:
        return registrations, diagnostic
    templates = {}
    for rig in RIGS:
        registration = registrations[rig]
        for eye in EYES:
            transform = (registration.left_camera_T_base if eye == "left"
                         else registration.right_camera_T_base)
            templates[(rig, eye)] = _projection_matrix(ViewObservation(
                view_id=f"{rig}_{eye}", K=registration.K,
                camera_T_base=transform,
                centerline_xy=np.empty((0, 2), dtype=np.float64)))
    points = {rig: [] for rig in RIGS}
    count = len(marker_centers)
    view_index = {name: index for index, name in enumerate(VIEW_IDS)}
    for frame in range(count):
        if any(marker_temporal_outliers[rig][frame] > 0 for rig in RIGS):
            continue
        for marker in range(4):
            if np.any(marker_inferred[frame, :, marker]):
                continue
            pair = {}
            for rig in RIGS:
                left = marker_centers[
                    frame, view_index[f"{rig}_left"], marker]
                right = marker_centers[
                    frame, view_index[f"{rig}_right"], marker]
                if not (np.all(np.isfinite(left))
                        and np.all(np.isfinite(right))):
                    break
                pair[rig] = _triangulate_marker_pair(
                    left, templates[(rig, "left")],
                    right, templates[(rig, "right")])
                if pair[rig] is None:
                    break
            if len(pair) == len(RIGS):
                for rig in RIGS:
                    points[rig].append(pair[rig])
    primary = np.asarray(points["primary"], dtype=np.float64)
    oblique = np.asarray(points["oblique"], dtype=np.float64)
    diagnostic["candidate_pair_count"] = int(len(primary))
    if len(primary) < config.cross_rig_registration_min_pairs:
        diagnostic["reason"] = "insufficient_marker_pairs"
        return registrations, diagnostic
    rotation, translation_mm, inliers = _robust_rigid_alignment(
        oblique, primary,
        config.cross_rig_registration_max_initial_error_mm)
    rotation_deg = float(np.degrees(np.arccos(np.clip(
        (np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))))
    translation_norm_mm = float(np.linalg.norm(translation_mm))
    before = np.linalg.norm(oblique[inliers] - primary[inliers], axis=1)
    after = np.linalg.norm(
        (rotation @ oblique[inliers].T).T + translation_mm
        - primary[inliers], axis=1)
    diagnostic.update({
        "inlier_pair_count": int(np.count_nonzero(inliers)),
        "rotation_matrix": rotation.tolist(),
        "rotation_angle_deg": rotation_deg,
        "translation_mm": translation_mm.tolist(),
        "translation_norm_mm": translation_norm_mm,
        "before_median_mm": float(np.median(before)),
        "before_p95_mm": float(np.percentile(before, 95)),
        "after_median_mm": float(np.median(after)),
        "after_p95_mm": float(np.percentile(after, 95)),
    })
    if (rotation_deg > config.cross_rig_registration_max_rotation_deg
            or translation_norm_mm
            > config.cross_rig_registration_max_translation_mm):
        diagnostic["reason"] = "correction_exceeds_safety_limit"
        return registrations, diagnostic
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = rotation
    correction[:3, 3] = translation_mm * 1e-3
    base_correction_inverse = np.linalg.inv(correction)
    oblique_registration = registrations["oblique"]
    refined = dict(registrations)
    refined["oblique"] = replace(
        oblique_registration,
        left_camera_T_base=(
            oblique_registration.left_camera_T_base
            @ base_correction_inverse),
        right_camera_T_base=(
            oblique_registration.right_camera_T_base
            @ base_correction_inverse))
    diagnostic["applied"] = True
    diagnostic["reason"] = "robust_marker_alignment"
    return refined, diagnostic


def _fit_reference_from_available_rigs(
        trusted_rig: str | None, available_rig_mask: int,
) -> tuple[str | None, set[str]]:
    """Use all healthy rigs jointly; return one reference only as fallback."""
    available = {
        rig for rig, bit in (("primary", 1), ("oblique", 2))
        if int(available_rig_mask) & bit}
    if trusted_rig in available:
        return trusted_rig, {trusted_rig}
    if len(available) == 1:
        rig = next(iter(available))
        return rig, {rig}
    if len(available) == 2:
        return None, available
    return None, set()


def _monotonic_epipolar_curve_matches(
        left_xy: np.ndarray, right_xy: np.ndarray, samples: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match an order-preserving rectified stereo curve with dynamic time warping.

    The two routes are already oriented from marker 0 toward marker 3. Dynamic
    programming uses epipolar row agreement while allowing perspective to
    stretch their projected arclengths differently. Returned pairs are unique
    in the left route and monotonic in the right route.
    """
    left = _resample_2d(left_xy, samples)
    right = _resample_2d(right_xy, samples)
    count = len(left)
    epipolar = np.abs(left[:, None, 1] - right[None, :, 1])
    order = np.abs(
        np.arange(count)[:, None] - np.arange(count)[None, :]
    ) / max(count - 1, 1)
    local_cost = epipolar / 1.5 + 0.75 * order
    accumulated = np.full((count, count), np.inf, dtype=np.float64)
    predecessor = np.zeros((count, count), dtype=np.uint8)
    accumulated[0, 0] = local_cost[0, 0]
    skip_penalty = 0.20
    for first in range(count):
        for second in range(count):
            if first == 0 and second == 0:
                continue
            options = []
            if first and second:
                options.append((accumulated[first - 1, second - 1], 0))
            if first:
                options.append((
                    accumulated[first - 1, second] + skip_penalty, 1))
            if second:
                options.append((
                    accumulated[first, second - 1] + skip_penalty, 2))
            value, code = min(options, key=lambda item: item[0])
            accumulated[first, second] = value + local_cost[first, second]
            predecessor[first, second] = code
    path = []
    first = second = count - 1
    while True:
        path.append((first, second))
        if first == 0 and second == 0:
            break
        code = int(predecessor[first, second])
        if code == 0:
            first -= 1
            second -= 1
        elif code == 1:
            first -= 1
        else:
            second -= 1
    path.reverse()
    by_left: dict[int, list[int]] = {}
    for first, second in path:
        by_left.setdefault(first, []).append(second)
    first_indices = []
    second_indices = []
    previous_second = -1
    for first in sorted(by_left):
        choices = np.asarray(by_left[first], dtype=int)
        choices = choices[choices >= previous_second]
        if not len(choices):
            continue
        costs = local_cost[first, choices]
        second = int(choices[int(np.argmin(costs))])
        first_indices.append(first)
        second_indices.append(second)
        previous_second = second
    first_indices = np.asarray(first_indices, dtype=int)
    second_indices = np.asarray(second_indices, dtype=int)
    residual = np.abs(
        left[first_indices, 1] - right[second_indices, 1])
    return (
        left[first_indices], right[second_indices], residual,
        first_indices)


def _fit_ordered_disparity_spline(
        left_xy: np.ndarray, right_xy: np.ndarray,
        sample_indices: np.ndarray, sample_count: int,
        config: MultiCameraConfig,
        marker_rows: list[tuple[int, float, float]] | None = None,
        previous_coefficients: np.ndarray | None = None,
) -> dict:
    """Robustly fit one spatially smooth rectified disparity field."""
    basis_count = int(config.ordered_disparity_basis_count)
    full_design = _basis(sample_count, basis_count)
    indices = np.asarray(sample_indices, dtype=int)
    design = full_design[indices]
    disparity = np.asarray(left_xy[:, 0] - right_xy[:, 0], float)
    marker_rows = marker_rows or []
    if marker_rows:
        marker_index = np.asarray([item[0] for item in marker_rows], int)
        marker_design = full_design[marker_index]
        marker_disparity = np.asarray(
            [item[1] for item in marker_rows], float)
        marker_weight = np.asarray([item[2] for item in marker_rows], float)
    else:
        marker_design = np.empty((0, basis_count))
        marker_disparity = np.empty(0)
        marker_weight = np.empty(0)
    second_difference = np.diff(np.eye(basis_count), n=2, axis=0)
    spatial = float(config.ordered_disparity_spatial_weight)
    temporal = float(config.ordered_disparity_temporal_weight)
    previous = (
        np.asarray(previous_coefficients, float)
        if previous_coefficients is not None else None)
    if (temporal <= 0.0 or previous is not None
            and previous.shape != (basis_count,)):
        previous = None
    robust_weight = np.ones(len(disparity), dtype=np.float64)
    coefficients = np.linalg.lstsq(design, disparity, rcond=None)[0]
    for _ in range(5):
        normal = (
            design.T @ (robust_weight[:, None] * design)
            + spatial * (second_difference.T @ second_difference)
            + 1e-8 * np.eye(basis_count))
        rhs = design.T @ (robust_weight * disparity)
        if len(marker_design):
            normal += marker_design.T @ (
                marker_weight[:, None] * marker_design)
            rhs += marker_design.T @ (marker_weight * marker_disparity)
        if previous is not None:
            normal += temporal * np.eye(basis_count)
            rhs += temporal * previous
        coefficients = np.linalg.solve(normal, rhs)
        residual = design @ coefficients - disparity
        scale = max(float(config.ordered_disparity_huber_px), 1e-6)
        robust_weight = np.minimum(
            1.0, scale / np.maximum(np.abs(residual), 1e-9))
    fitted = design @ coefficients
    residual = fitted - disparity
    return {
        "coefficients_px": coefficients,
        "disparity_px": fitted,
        "residual_p95_px": float(np.percentile(np.abs(residual), 95)),
        "temporal_used": previous is not None,
    }


def _ordered_stereo_curve_candidate(
        observations: list[ViewObservation], rig: str,
        config: MultiCameraConfig,
        previous_disparity_coefficients: np.ndarray | None = None,
        previous_timestamp_ns: int | None = None,
        timestamp_ns: int | None = None) -> dict | None:
    """Triangulate a soft 3D curve observation from one well-posed ZED rig."""
    pair = sorted(
        [item for item in observations if item.view_id.startswith(rig + "_")],
        key=lambda item: item.view_id)
    if len(pair) != 2 or not all(
            item.topology_weight
            >= config.ordered_stereo_min_topology_weight for item in pair):
        return None
    left, right = pair
    if not (left.view_id.endswith("_left")
            and right.view_id.endswith("_right")):
        return None
    try:
        left_xy, right_xy, epipolar, sample_indices = (
            _monotonic_epipolar_curve_matches(
            left.centerline_xy, right.centerline_xy,
            samples=config.ordered_stereo_samples))
    except ValueError:
        return None
    maximum_epipolar = config.ordered_stereo_max_epipolar_p95_px
    retained = epipolar <= maximum_epipolar
    minimum = int(np.ceil(
        config.ordered_stereo_min_match_fraction
        * config.ordered_stereo_samples))
    if np.count_nonzero(retained) < minimum:
        return None
    left_xy = left_xy[retained]
    right_xy = right_xy[retained]
    epipolar = epipolar[retained]
    sample_indices = sample_indices[retained]
    temporal_adjacent = bool(
        previous_disparity_coefficients is not None
        and previous_timestamp_ns is not None
        and timestamp_ns is not None
        and 0 < (int(timestamp_ns) - int(previous_timestamp_ns)) * 1e-6
        <= config.ordered_disparity_max_temporal_gap_ms)
    full_left = _resample_2d(
        left.centerline_xy, config.ordered_stereo_samples)
    marker_rows = []
    if left.axial_markers_xy is not None and right.axial_markers_xy is not None:
        left_markers = np.asarray(left.axial_markers_xy, float)
        right_markers = np.asarray(right.axial_markers_xy, float)
        for marker in range(min(len(left_markers), len(right_markers), 4)):
            if not (np.all(np.isfinite(left_markers[marker]))
                    and np.all(np.isfinite(right_markers[marker]))):
                continue
            index = int(np.argmin(np.linalg.norm(
                full_left - left_markers[marker], axis=1)))
            marker_rows.append((
                index,
                float(left_markers[marker, 0] - right_markers[marker, 0]),
                0.60 if marker in (0, 3) else 0.25))
    disparity_fit = _fit_ordered_disparity_spline(
        left_xy, right_xy, sample_indices,
        config.ordered_stereo_samples, config,
        marker_rows=marker_rows,
        previous_coefficients=(
            previous_disparity_coefficients if temporal_adjacent else None))
    right_xy = np.column_stack([
        left_xy[:, 0] - disparity_fit["disparity_px"],
        left_xy[:, 1],
    ])
    right_support = cKDTree(np.asarray(
        right.centerline_xy, float)).query(right_xy)[0]
    right_support_p95 = float(np.percentile(right_support, 95))
    if (not np.isfinite(right_support_p95)
            or right_support_p95
            > config.ordered_disparity_max_right_support_p95_px):
        return None
    support_confidence = float(np.exp(
        -0.5 * (right_support_p95 / max(
            config.ordered_disparity_support_sigma_px, 1e-6)) ** 2))
    projections = (_projection_matrix(left), _projection_matrix(right))
    points = []
    kept_epipolar = []
    for first_xy, second_xy, row_error in zip(
            left_xy, right_xy, epipolar):
        point = _triangulate_marker_pair(
            first_xy, projections[0], second_xy, projections[1])
        if point is None:
            continue
        first_reprojection, first_front = project_points(
            left.K, left.camera_T_base, point[None])
        second_reprojection, second_front = project_points(
            right.K, right.camera_T_base, point[None])
        if not (bool(first_front[0]) and bool(second_front[0])):
            continue
        reprojection = max(
            float(np.linalg.norm(first_reprojection[0] - first_xy)),
            float(np.linalg.norm(second_reprojection[0] - second_xy)))
        if reprojection > 2.5:
            continue
        points.append(point)
        kept_epipolar.append(float(row_error))
    if len(points) < minimum:
        return None
    points = np.asarray(points, dtype=np.float64)
    epipolar = np.asarray(kept_epipolar, dtype=np.float64)
    curve_length = float(cumulative_arclength(points)[-1])
    if (not np.isfinite(curve_length)
            or curve_length < config.ordered_stereo_min_curve_length_mm
            or curve_length > config.ordered_stereo_max_curve_length_mm):
        return None
    # A reference score contains no camera-name preference. Topology and
    # epipolar agreement alone determine which rig supplies depth.
    topology = float(np.mean([item.topology_weight for item in pair]))
    epipolar_p95 = float(np.percentile(epipolar, 95))
    match_fraction = len(points) / float(config.ordered_stereo_samples)
    score = (
        topology * match_fraction
        * np.exp(-epipolar_p95 / max(maximum_epipolar, 1e-6))
        * np.exp(-abs(curve_length - config.nominal_length_mm) / 15.0)
        * support_confidence)
    return {
        "rig": rig,
        "points_base_mm": points,
        "timestamp_offset_s": float(np.mean([
            item.timestamp_offset_s for item in pair])),
        "match_count": len(points),
        "epipolar_p95_px": epipolar_p95,
        "curve_length_mm": curve_length,
        "disparity_coefficients_px": disparity_fit["coefficients_px"],
        "disparity_residual_p95_px": disparity_fit["residual_p95_px"],
        "right_support_p95_px": right_support_p95,
        "support_confidence": support_confidence,
        "disparity_temporal_used": disparity_fit["temporal_used"],
        "score": float(score),
    }


def _select_ordered_stereo_curve(
        observations: list[ViewObservation], trusted_rig: str | None,
        previous_rig: str | None, config: MultiCameraConfig,
        disparity_states: dict[str, dict] | None = None,
        rig_timestamps_ns: dict[str, int] | None = None,
        corrupted_rigs: set[str] | None = None) -> dict | None:
    disparity_states = disparity_states or {}
    rig_timestamps_ns = rig_timestamps_ns or {}
    corrupted_rigs = set() if corrupted_rigs is None else set(corrupted_rigs)
    candidates = {
        rig: candidate for rig in RIGS
        if rig not in corrupted_rigs
        if (candidate := _ordered_stereo_curve_candidate(
            observations, rig, config,
            previous_disparity_coefficients=(
                disparity_states.get(rig, {}).get("coefficients_px")),
            previous_timestamp_ns=(
                disparity_states.get(rig, {}).get("timestamp_ns")),
            timestamp_ns=rig_timestamps_ns.get(rig))) is not None}
    if not candidates:
        return None
    # Cross-rig endpoint consistency is advisory. If that rig's current image
    # is corrupt or topologically unusable, immediately fall back to the other
    # independently valid stereo rig.
    if trusted_rig is not None and trusted_rig in candidates:
        selected = candidates[trusted_rig]
        selected["available_rig_mask"] = sum(
            (1 if rig == "primary" else 2) for rig in candidates)
        return selected
    best = max(candidates.values(), key=lambda item: item["score"])
    previous = candidates.get(previous_rig)
    # Both surviving candidates have already passed topology, epipolar,
    # opposite-eye support, and length gates.  Keep the current physical rig
    # while it remains valid; small score changes between two healthy rigs
    # must not change the objective (and therefore the depth basin) from one
    # frame to the next.  An explicit cross-rig inconsistency decision above
    # still overrides this latch immediately.
    selected = previous if previous is not None else best
    selected["available_rig_mask"] = sum(
        (1 if rig == "primary" else 2) for rig in candidates)
    return selected


def _marker_hypothesis_residuals(
        point_base_mm: np.ndarray, centers: np.ndarray,
        observations: list[ViewObservation]) -> np.ndarray:
    residuals = np.full(len(observations), np.inf, dtype=np.float64)
    for view, (center, observation) in enumerate(zip(centers, observations)):
        if not np.all(np.isfinite(center)):
            continue
        projected, in_front = project_points(
            observation.K, observation.camera_T_base,
            np.asarray(point_base_mm, float)[None])
        if bool(in_front[0]):
            residuals[view] = float(np.linalg.norm(projected[0] - center))
    return residuals


def _repair_cross_camera_marker_centers(
        centers: np.ndarray, observations: list[ViewObservation],
        timestamps_ns: np.ndarray, config: MultiCameraConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject cross-camera marker identities that cannot share one 3D point.

    Each frame/marker is explained by pairwise DLT hypotheses across the four
    calibrated eyes.  A weak 3D motion prior only breaks otherwise ambiguous
    two-against-two hypotheses; it never imposes the approximate ring spacing.
    Rejected samples are subsequently repaired bidirectionally in image space.
    """
    values = np.asarray(centers, dtype=np.float64).copy()
    if values.ndim != 4 or values.shape[1:] != (len(observations), 4, 2):
        raise ValueError("cross-camera marker centers have unexpected shape")
    projections = [_projection_matrix(item) for item in observations]
    rejected = np.zeros(values.shape[:3], dtype=bool)
    previous_3d = np.full((4, 3), np.nan, dtype=np.float64)
    previous_ns = np.zeros(4, dtype=np.int64)
    inlier_px = float(config.marker_cross_view_inlier_px)
    outlier_px = float(config.marker_cross_view_outlier_px)
    temporal_sigma = max(
        float(config.marker_cross_view_temporal_sigma_mm), 1e-6)
    for frame in range(len(values)):
        for marker in range(4):
            sample = values[frame, :, marker]
            finite = np.flatnonzero(np.all(np.isfinite(sample), axis=1))
            if len(finite) < 2:
                continue
            hypotheses = []
            for ordinal, first in enumerate(finite[:-1]):
                for second in finite[ordinal + 1:]:
                    point = _triangulate_marker_pair(
                        sample[first], projections[first],
                        sample[second], projections[second])
                    if point is None:
                        continue
                    residuals = _marker_hypothesis_residuals(
                        point, sample, observations)
                    inliers = residuals <= inlier_px
                    represented_rigs = len({
                        observations[index].view_id.rsplit("_", 1)[0]
                        for index in np.flatnonzero(inliers)})
                    clipped_cost = float(np.sum(
                        np.minimum(residuals[finite], 2.0 * outlier_px) ** 2))
                    temporal_cost = 0.0
                    if np.all(np.isfinite(previous_3d[marker])):
                        elapsed_ms = (
                            int(timestamps_ns[frame]) - previous_ns[marker]
                        ) * 1e-6
                        if 0.0 < elapsed_ms <= 500.0:
                            temporal_cost = float(np.linalg.norm(
                                point - previous_3d[marker]) / temporal_sigma)
                    # Inlier count and representation of both physical rigs
                    # dominate; temporal continuity only resolves ambiguity.
                    score = (
                        -1000.0 * int(np.count_nonzero(inliers))
                        -100.0 * represented_rigs
                        + clipped_cost + temporal_cost)
                    hypotheses.append((score, point, residuals, inliers))
            if not hypotheses:
                continue
            _, point, residuals, inliers = min(
                hypotheses, key=lambda item: item[0])
            inlier_indices = np.flatnonzero(inliers)
            represented_rigs = {
                observations[index].view_id.rsplit("_", 1)[0]
                for index in inlier_indices}
            # Only reject a label when a hypothesis is supported by at least
            # three eyes, or by both physical rigs. This avoids inventing
            # evidence during a genuinely ambiguous overlap.
            supported = (
                len(inlier_indices) >= 3
                or (len(inlier_indices) >= 2 and len(represented_rigs) >= 2))
            if supported:
                bad = np.isfinite(residuals) & (residuals > outlier_px)
                rejected[frame, bad, marker] = True
                values[frame, bad, marker] = np.nan
                previous_3d[marker] = point
                previous_ns[marker] = int(timestamps_ns[frame])

    view_names = [item.view_id for item in observations]
    centers_by_view = {
        name: values[:, view].copy() for view, name in enumerate(view_names)}
    widths = {
        name: np.full(values.shape[:1] + (4,), np.nan)
        for name in view_names}
    confidence = {
        name: np.where(
            np.all(np.isfinite(centers_by_view[name]), axis=2), 1.0, 0.0)
        for name in view_names}
    observed = {
        name: np.all(np.isfinite(centers_by_view[name]), axis=2).astype(np.uint8)
        for name in view_names}
    repaired = repair_stereo_marker_tracks(
        centers_by_view, widths, confidence, observed, timestamps_ns,
        maximum_gap_ms=500.0, maximum_chord_residual_px=12.0)
    output = np.stack([
        repaired[name]["centers"] for name in view_names], axis=1)
    return output, rejected


def _rig_marker_candidates(
        centers: np.ndarray, widths: np.ndarray, confidence: np.ndarray,
        view_offset: int, observations: list[ViewObservation],
        duplicate_px: float = 12.0) -> list[dict]:
    """Return ordered, deduplicated red components for one stereo rig."""
    candidates = []
    for source_slot in range(4):
        eye_centers = np.asarray(centers[:, source_slot], float)
        finite = np.all(np.isfinite(eye_centers), axis=1)
        if not np.any(finite):
            continue
        point = None
        if (np.all(finite)
                and abs(float(eye_centers[0, 1] - eye_centers[1, 1]))
                <= 8.0):
            try:
                point = _triangulate_marker_pair(
                    eye_centers[0], _projection_matrix(
                        observations[view_offset]),
                    eye_centers[1], _projection_matrix(
                        observations[view_offset + 1]))
            except np.linalg.LinAlgError:
                point = None
        candidates.append({
            "source_slot": source_slot,
            "centers": eye_centers,
            "finite": finite,
            "point": point,
            "width": float(np.nanmean(widths[:, source_slot])),
            "confidence": float(np.nanmean(confidence[:, source_slot])),
            "quality": float(np.count_nonzero(finite))
            + (2.0 if point is not None else 0.0)
            + float(np.nanmean(confidence[:, source_slot])),
        })
    # A collapsed detector commonly emits one physical ring twice under
    # adjacent source labels. Do not let either a monocular or a
    # stereo-consistent duplicate make an incomplete rig appear complete.
    kept = []
    for candidate in candidates:
        duplicate = None
        for ordinal, other in enumerate(kept):
            common = candidate["finite"] & other["finite"]
            pixel_duplicate = (
                np.any(common)
                and np.min(np.linalg.norm(
                    candidate["centers"][common]
                    - other["centers"][common], axis=1))
                < float(duplicate_px))
            stereo_duplicate = (
                candidate["point"] is not None
                and other["point"] is not None
                and np.linalg.norm(
                    candidate["point"] - other["point"]) < 4.0)
            weak_duplicate = (
                candidate["point"] is None or other["point"] is None)
            if pixel_duplicate and (stereo_duplicate or weak_duplicate):
                duplicate = ordinal
                break
        if duplicate is None:
            kept.append(candidate)
        elif candidate["quality"] > kept[duplicate]["quality"]:
            kept[duplicate] = candidate
    return sorted(kept, key=lambda item: item["source_slot"])


def _marker_assignment_hypotheses(
        candidates: list[dict], previous_points: np.ndarray,
        temporal_sigma_mm: float = 6.0) -> list[tuple[float, dict[int, dict]]]:
    count = len(candidates)
    if count > 4:
        return []
    output = []
    for physical_ids in combinations(range(4), count):
        assignment = dict(zip(physical_ids, candidates))
        cost = 0.0
        for physical_id, candidate in assignment.items():
            # Existing labels are only a weak prior. Monotonic material order,
            # cross-rig geometry, and temporal 3D continuity dominate.
            cost += 0.15 * (
                candidate["source_slot"] - physical_id) ** 2
            point = candidate["point"]
            if (point is not None
                    and np.all(np.isfinite(previous_points[physical_id]))):
                distance = float(np.linalg.norm(
                    point - previous_points[physical_id]))
                cost += min((distance / temporal_sigma_mm) ** 2, 100.0)
        output.append((cost, assignment))
    return output


def _cross_rig_candidate_cost(
        first: dict, second: dict, first_offset: int, second_offset: int,
        observations: list[ViewObservation]) -> float:
    if first["point"] is not None and second["point"] is not None:
        distance = float(np.linalg.norm(first["point"] - second["point"]))
        return min((distance / 3.0) ** 2, 225.0)
    costs = []
    for source, target, target_offset in (
            (first, second, second_offset),
            (second, first, first_offset)):
        if source["point"] is None:
            continue
        for eye in range(2):
            if not target["finite"][eye]:
                continue
            observation = observations[target_offset + eye]
            projected = project_points(
                observation.K, observation.camera_T_base,
                source["point"][None])[0][0]
            costs.append(float(np.linalg.norm(
                projected - target["centers"][eye])) / 6.0)
    return float(np.sum(np.square(costs))) if costs else 4.0


def _associate_cross_camera_marker_identities(
        centers: np.ndarray, widths: np.ndarray, confidence: np.ndarray,
        observations: list[ViewObservation], timestamps_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Symmetrically associate physical ring IDs across two stereo rigs.

    No rig is preferred initially. If exactly one rig contains four unique
    stereo components, its complete assignment provides the frame-local 3D
    reference for the incomplete rig. Otherwise both assignments are scored
    jointly using cross-rig geometry and temporal marker motion.
    """
    values = np.asarray(centers, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (4, 4, 2):
        raise ValueError("marker identity observations have unexpected shape")
    assigned = np.full_like(values, np.nan)
    assigned_widths = np.full(values.shape[:3], np.nan, dtype=np.float64)
    source_slots = np.full(values.shape[:3], -1, dtype=np.int8)
    inferred = np.zeros(values.shape[:3], dtype=np.uint8)
    reference_code = np.zeros(len(values), dtype=np.uint8)
    previous_points = np.full((4, 3), np.nan, dtype=np.float64)
    rig_offsets = {"primary": 0, "oblique": 2}
    for frame in range(len(values)):
        candidates = {
            rig: _rig_marker_candidates(
                values[frame, offset:offset + 2],
                widths[frame, offset:offset + 2],
                confidence[frame, offset:offset + 2], offset, observations)
            for rig, offset in rig_offsets.items()}
        hypotheses = {
            rig: _marker_assignment_hypotheses(
                candidates[rig], previous_points)
            for rig in RIGS}
        if not all(hypotheses[rig] for rig in RIGS):
            continue
        complete = {
            rig: (len(candidates[rig]) == 4
                  and all(item["point"] is not None
                          for item in candidates[rig]))
            for rig in RIGS}
        if complete["primary"] and not complete["oblique"]:
            reference_code[frame] = 1
        elif complete["oblique"] and not complete["primary"]:
            reference_code[frame] = 2
        elif not complete["primary"] and not complete["oblique"]:
            reference_code[frame] = 3
        joint = []
        for primary_cost, primary_assignment in hypotheses["primary"]:
            for oblique_cost, oblique_assignment in hypotheses["oblique"]:
                cost = primary_cost + oblique_cost
                common = set(primary_assignment) & set(oblique_assignment)
                for physical_id in common:
                    cost += _cross_rig_candidate_cost(
                        primary_assignment[physical_id],
                        oblique_assignment[physical_id], 0, 2, observations)
                # A complete rig is the frame-local reference. Missing target
                # IDs are allowed; shifting all later identities is not.
                if complete["primary"]:
                    cost += 8.0 * len(
                        set(oblique_assignment) - set(primary_assignment))
                if complete["oblique"]:
                    cost += 8.0 * len(
                        set(primary_assignment) - set(oblique_assignment))
                joint.append((cost, {
                    "primary": primary_assignment,
                    "oblique": oblique_assignment}))
        _, selected = min(joint, key=lambda item: item[0])
        for rig, offset in rig_offsets.items():
            for physical_id, candidate in selected[rig].items():
                for eye in range(2):
                    if candidate["finite"][eye]:
                        assigned[frame, offset + eye, physical_id] = (
                            candidate["centers"][eye])
                        assigned_widths[frame, offset + eye, physical_id] = (
                            widths[frame, offset + eye,
                                   candidate["source_slot"]])
                        source_slots[frame, offset + eye, physical_id] = (
                            candidate["source_slot"])
        # If one rig observes a physical marker in stereo and the other does
        # not, project it into the missing rig. This is an inferred marker, not
        # a relabeled neighboring component.
        for physical_id in range(4):
            rig_points = {
                rig: selected[rig].get(physical_id, {}).get("point")
                for rig in RIGS}
            for rig, offset in rig_offsets.items():
                if physical_id in selected[rig]:
                    continue
                other = "oblique" if rig == "primary" else "primary"
                point = rig_points[other]
                if point is None:
                    continue
                for eye in range(2):
                    observation = observations[offset + eye]
                    projected, in_front = project_points(
                        observation.K, observation.camera_T_base,
                        point[None])
                    if bool(in_front[0]):
                        assigned[frame, offset + eye, physical_id] = projected[0]
                        inferred[frame, offset + eye, physical_id] = 1
        # Update the shared temporal state symmetrically. With one complete
        # rig, use it; otherwise average mutually consistent stereo estimates.
        for physical_id in range(4):
            points = []
            for rig in RIGS:
                candidate = selected[rig].get(physical_id)
                if candidate is not None and candidate["point"] is not None:
                    points.append((rig, candidate["point"]))
            if not points:
                continue
            preferred = None
            if reference_code[frame] == 1:
                preferred = "primary"
            elif reference_code[frame] == 2:
                preferred = "oblique"
            if preferred is not None:
                match = [point for rig, point in points if rig == preferred]
                if match:
                    previous_points[physical_id] = match[0]
                    continue
            if len(points) == 2 and np.linalg.norm(
                    points[0][1] - points[1][1]) <= 8.0:
                previous_points[physical_id] = 0.5 * (
                    points[0][1] + points[1][1])
            elif np.all(np.isfinite(previous_points[physical_id])):
                previous_points[physical_id] = min(
                    (point for _, point in points),
                    key=lambda point: np.linalg.norm(
                        point - previous_points[physical_id]))
            else:
                previous_points[physical_id] = points[0][1]

    view_names = [item.view_id for item in observations]
    centers_by_view = {
        name: assigned[:, view].copy()
        for view, name in enumerate(view_names)}
    widths_by_view = {
        name: assigned_widths[:, view].copy()
        for view, name in enumerate(view_names)}
    confidence_by_view = {
        name: np.where(np.all(np.isfinite(
            centers_by_view[name]), axis=2), 0.65, 0.0)
        for name in view_names}
    observed_by_view = {
        name: np.all(np.isfinite(
            centers_by_view[name]), axis=2).astype(np.uint8)
        for name in view_names}
    repaired = repair_stereo_marker_tracks(
        centers_by_view, widths_by_view, confidence_by_view,
        observed_by_view, timestamps_ns, maximum_gap_ms=500.0,
        maximum_chord_residual_px=12.0)
    output = np.stack([
        repaired[name]["centers"] for name in view_names], axis=1)
    return output, source_slots, inferred, reference_code, assigned_widths


def _result_rig_errors(result) -> dict[str, float]:
    """Unweighted symmetric image error for each physical camera rig."""
    errors: dict[str, list[float]] = {rig: [] for rig in RIGS}
    for view in VIEW_IDS:
        if view not in result.view_model_mean_px:
            continue
        errors[view.rsplit("_", 1)[0]].append(float(0.5 * (
            result.view_model_mean_px[view]
            + result.view_coverage_mean_px[view])))
    return {
        rig: (float(np.mean(values)) if values else float("inf"))
        for rig, values in errors.items()}


def _result_rig_coverage_errors(result) -> dict[str, float]:
    errors: dict[str, list[float]] = {rig: [] for rig in RIGS}
    for view in VIEW_IDS:
        if view in result.view_coverage_mean_px:
            errors[view.rsplit("_", 1)[0]].append(float(
                result.view_coverage_mean_px[view]))
    return {
        rig: (float(np.mean(values)) if values else float("inf"))
        for rig, values in errors.items()}


def _result_terminal_error(result) -> float:
    values = [
        *result.view_terminal_start_px.values(),
        *result.view_terminal_end_px.values(),
    ]
    return float(np.max(values)) if values else float("inf")


def _result_terminal_error_for_rigs(
        result, trusted_rigs: set[str] | None = None) -> float:
    trusted = set(RIGS) if trusted_rigs is None else set(trusted_rigs)
    values = []
    for mapping in (
            result.view_terminal_start_px, result.view_terminal_end_px):
        values.extend(
            float(value) for view, value in mapping.items()
            if view.rsplit("_", 1)[0] in trusted)
    return float(np.max(values)) if values else float("inf")


def _stereo_terminal_points(
        observations: list[ViewObservation]) -> dict[str, np.ndarray]:
    """Triangulate the axial tip independently in each physical stereo rig."""
    output = {}
    for rig in RIGS:
        pair = [item for item in observations
                if item.view_id.startswith(rig + "_")]
        if len(pair) != 2:
            continue
        centers = []
        projections = []
        for item in pair:
            markers = np.asarray(item.axial_markers_xy, float)
            if len(markers) < 4 or not np.all(np.isfinite(markers[3])):
                break
            centers.append(markers[3])
            projections.append(_projection_matrix(item))
        if len(centers) != 2:
            continue
        try:
            point = _triangulate_marker_pair(
                centers[0], projections[0], centers[1], projections[1])
        except np.linalg.LinAlgError:
            point = None
        if point is not None:
            output[rig] = point
    return output


def _select_consistent_rig(
        observations: list[ViewObservation],
        previous_terminal_base_mm: np.ndarray | None,
        maximum_disagreement_mm: float,
) -> tuple[str | None, dict[str, np.ndarray], float]:
    """Select one stereo rig when their axial tips cannot be simultaneous."""
    endpoints = _stereo_terminal_points(observations)
    if not all(rig in endpoints for rig in RIGS):
        return None, endpoints, float("nan")
    disagreement = float(np.linalg.norm(
        endpoints["primary"] - endpoints["oblique"]))
    if disagreement <= float(maximum_disagreement_mm):
        return None, endpoints, disagreement
    scores = {}
    for rig in RIGS:
        rig_observations = [item for item in observations
                            if item.view_id.startswith(rig + "_")]
        topology_penalty = 5.0 * (1.0 - float(np.mean([
            item.topology_weight for item in rig_observations])))
        temporal = (
            0.0 if previous_terminal_base_mm is None
            else float(np.linalg.norm(
                endpoints[rig] - previous_terminal_base_mm)))
        scores[rig] = temporal + topology_penalty
    return min(scores, key=scores.get), endpoints, disagreement


def _rig_restricted_observations(
        observations: list[ViewObservation], trusted_rig: str | None,
        excluded_weight: float) -> list[ViewObservation]:
    if trusted_rig is None:
        return observations
    return [
        item if item.view_id.startswith(trusted_rig + "_") else
        replace(item, weight=min(float(item.weight), float(excluded_weight)))
        for item in observations]


def _result_requires_reacquisition(
        result, config: MultiCameraConfig,
        trusted_rigs: set[str] | None = None) -> bool:
    """Return whether the currently trusted image evidence needs multi-start.

    A deliberately excluded rig must not force reacquisition forever. Its
    residual remains available in diagnostics, but only the rig or rigs that
    are currently allowed to determine the shape participate in this gate.
    """
    trusted = set(RIGS) if trusted_rigs is None else set(trusted_rigs)
    rig_errors = _result_rig_errors(result)
    coverage_errors = _result_rig_coverage_errors(result)
    return bool(
        not result.optimizer_success
        or not np.all(np.isfinite(result.points_base_mm))
        or max(rig_errors[rig] for rig in trusted)
        > config.reacquisition_rig_error_px
        or max(coverage_errors[rig] for rig in trusted)
        > config.reacquisition_coverage_mean_px
        or _result_terminal_error_for_rigs(result, trusted)
        > config.reacquisition_terminal_error_px
        or abs(float(result.length_residual_mm))
        > config.reacquisition_length_deviation_mm)


def _result_selection_score(
        result, nominal_length_mm: float,
        previous_points: np.ndarray | None = None,
        trusted_rigs: set[str] | None = None) -> float:
    """Cross-rig score that cannot hide a failed rig behind low weights."""
    trusted = set(RIGS) if trusted_rigs is None else set(trusted_rigs)
    rig_error_mapping = _result_rig_errors(result)
    rig_errors = np.asarray(
        [rig_error_mapping[rig] for rig in RIGS if rig in trusted], float)
    if not np.all(np.isfinite(rig_errors)):
        return float("inf")
    score = float(np.max(rig_errors) + 0.25 * np.mean(rig_errors))
    score += abs(float(result.arc_length_mm) - nominal_length_mm) / 4.0
    score += 0.50 * _result_terminal_error_for_rigs(result, trusted)
    coverage_mapping = _result_rig_coverage_errors(result)
    coverage = np.asarray(
        [coverage_mapping[rig] for rig in RIGS if rig in trusted], float)
    if not np.all(np.isfinite(coverage)):
        return float("inf")
    score += 0.20 * float(np.max(coverage))
    if previous_points is not None:
        try:
            current, _, _ = resample_polyline(
                np.asarray(result.points_base_mm, float), 64)
            previous, _, _ = resample_polyline(
                np.asarray(previous_points, float), 64)
            motion = float(np.sqrt(np.mean(np.sum(
                (current - previous) ** 2, axis=1))))
            score += min(motion / 20.0, 0.75)
        except ValueError:
            return float("inf")
    return score


def _create_output(
        path: Path, count: int, config: MultiCameraConfig,
        registrations: dict, metadata: dict):
    import h5py

    output = h5py.File(path, "w")
    output.attrs["schema_version"] = 26
    output.attrs["mode"] = "dual_zed_four_view_shape_tracking"
    output.attrs["coordinate_frame"] = "robot_base"
    output.attrs["position_units"] = "mm"
    output.attrs["curvature_units"] = "1/mm"
    output.attrs["canonical_clock"] = "paired_camera_timestamp_midpoint"
    output.attrs["inter_rig_time_model"] = (
        "causal_coefficient_velocity_about_timestamp_midpoint")
    output.attrs["processing_config_json"] = json.dumps(
        asdict(config), sort_keys=True)
    output.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    if "cross_rig_registration_refinement" in metadata:
        output.attrs["cross_rig_registration_refinement_json"] = json.dumps(
            metadata["cross_rig_registration_refinement"], sort_keys=True)
    output.attrs["learning_rejection_flags_json"] = json.dumps({
        1: "reconstruction_invalid",
        2: "temporal_long_gap_unsupported",
        4: "whole_shape_temporal_outlier",
        8: "terminal_temporal_outlier",
        16: "mask_effective_width",
        32: "final_image_fit",
        64: "distal_arc_length_outside_learning_range",
        128: "whole_camera_image_fit",
    }, sort_keys=True)

    def ds(name, shape, dtype=np.float32, fillvalue=np.nan):
        return output.create_dataset(
            name, shape=shape, dtype=dtype, fillvalue=fillvalue,
            compression="gzip", compression_opts=1, shuffle=True)

    frames = output.create_group("frames")
    ds("frames/timestamp_ns", (count,), np.int64, 0)
    ds("frames/valid", (count,), np.uint8, 0)
    ds("frames/learning_valid", (count,), np.uint8, 0)
    ds("frames/learning_rejection_flags", (count,), np.uint16, 0)
    ds("frames/observation_valid", (count,), np.uint8, 0)
    for rig in RIGS:
        ds(f"frames/{rig}_observation_valid", (count,), np.uint8, 0)
    ds("frames/primary_svo_frame", (count,), np.int32, -1)
    ds("frames/oblique_svo_frame", (count,), np.int32, -1)
    ds("frames/primary_timestamp_ns", (count,), np.int64, 0)
    ds("frames/oblique_timestamp_ns", (count,), np.int64, 0)
    ds("frames/inter_rig_offset_ms", (count,))
    frames.create_dataset(
        "status", shape=(count,), dtype=h5py.string_dtype("utf-8"))

    distal = output.create_group("distal")
    ds("distal/points_base_mm", (count, config.distal_samples, 3))
    ds("distal/pre_temporal_points_base_mm",
       (count, config.distal_samples, 3))
    ds("distal/s_mm", (count, config.distal_samples))
    ds("distal/tangent_base", (count, config.distal_samples, 3))
    ds("distal/curvature_per_mm", (count, config.distal_samples))
    ds("distal/base_position_base_mm", (count, 3))
    ds("distal/observation_class", (count, config.distal_samples),
       np.uint8, 0)
    # ``full`` is a hard link because this pipeline intentionally reconstructs
    # only the marker-0 to marker-3 learning segment.
    output["full"] = distal
    output.create_group("multi_view")
    ds("multi_view/coefficients_base_mm",
       (count, config.spline_basis_count, 3))
    ds("multi_view/coefficient_velocity_base_mm_s",
       (count, config.spline_basis_count, 3))
    ds("multi_view/active_view_count", (count,), np.uint8, 0)
    ds("multi_view/initializer_code", (count,), np.uint8, 0)
    ds("multi_view/reacquisition_attempted", (count,), np.uint8, 0)
    ds("multi_view/reacquisition_candidate_count", (count,), np.uint8, 0)
    ds("multi_view/reacquisition_reason_code", (count,), np.uint8, 0)
    ds("multi_view/reacquisition_temporal_polished", (count,), np.uint8, 0)
    ds("multi_view/trusted_rig_code", (count,), np.uint8, 0)
    ds("multi_view/fit_reference_rig_code", (count,), np.uint8, 0)
    ds("multi_view/ordered_stereo_reference_code", (count,), np.uint8, 0)
    ds("multi_view/ordered_stereo_disparity_coefficients_px",
       (count, config.ordered_disparity_basis_count))
    ds("multi_view/marker_reference_code", (count,), np.uint8, 0)
    ds("multi_view/marker_temporal_outlier_count", (count,), np.uint8, 0)
    ds("multi_view/ordered_stereo_available_rig_mask",
       (count,), np.uint8, 0)
    for rig in RIGS:
        ds(f"multi_view/{rig}_marker_temporal_outlier_count",
           (count,), np.uint8, 0)
        ds(f"multi_view/{rig}_stereo_terminal_base_mm", (count, 3))
    for view_id in VIEW_IDS:
        ds(f"multi_view/{view_id}_marker_centers_px", (count, 4, 2))
        ds(f"multi_view/{view_id}_axial_endpoints_px", (count, 2, 2))
        ds(f"multi_view/{view_id}_marker_cross_camera_rejected",
           (count, 4), np.uint8, 0)
        ds(f"multi_view/{view_id}_marker_source_slot",
           (count, 4), np.int8, -1)
        ds(f"multi_view/{view_id}_marker_cross_camera_inferred",
           (count, 4), np.uint8, 0)
    output.create_group("quality")
    for name in (
            "reprojection_p95_px", "fitted_reprojection_p95_px",
            "stereo_condition", "joint_final_symmetric_mean_px",
            "multi_view_initial_symmetric_mean_px",
            "multi_view_arc_length_mm", "multi_view_length_residual_mm",
            "multi_view_ill_conditioned_view_count",
            "multi_view_optimizer_cost", "multi_view_optimizer_evaluations",
            "multi_view_optimizer_success", "distal_spline_arc_length_mm",
            "multi_view_primary_rig_symmetric_mean_px",
            "multi_view_oblique_rig_symmetric_mean_px",
            "multi_view_max_rig_symmetric_mean_px",
            "multi_view_whole_rig_failure",
            "multi_view_cross_camera_marker_rejection_count",
            "multi_view_cross_camera_marker_reassignment_count",
            "multi_view_cross_camera_marker_inferred_count",
            "multi_view_max_terminal_error_px",
            "multi_view_max_rig_coverage_mean_px",
            "multi_view_terminal_failure",
            "multi_view_coverage_failure",
            "multi_view_length_failure",
            "multi_view_direct_quality_failure",
            "multi_view_cross_rig_terminal_disagreement_mm",
            "multi_view_ordered_stereo_match_count",
            "multi_view_ordered_stereo_epipolar_p95_px",
            "multi_view_ordered_stereo_curve_length_mm",
            "multi_view_ordered_stereo_score",
            "multi_view_ordered_stereo_disparity_residual_p95_px",
            "multi_view_ordered_stereo_right_support_p95_px",
            "multi_view_ordered_stereo_support_confidence",
            "multi_view_ordered_stereo_innovation_mm",
            "multi_view_ordered_stereo_innovation_rejected",
            "multi_view_ordered_stereo_disparity_temporal_used",
            "abrupt_observation_jump_rejected",
            "distal_spline_basis_count", "distal_spline_internal_knot_count",
            "distal_spline_rms_residual_mm"):
        ds(f"quality/{name}", (count,))
    for view_id in VIEW_IDS:
        for suffix in (
                "model_mean_px", "model_p95_px", "coverage_mean_px",
                "coverage_p95_px", "terminal_start_px", "terminal_end_px",
                "weight", "topology_weight", "sharp_turn_clusters"):
            ds(f"quality/{view_id}_{suffix}", (count,))
    output.create_group("robot")
    for name in (
            "joint_velocity_command", "joint_position_measured", "encoder_raw"):
        ds(f"robot/{name}", (count, 6))
    for name in ("command_age_ms", "position_age_ms", "encoder_age_ms"):
        ds(f"robot/{name}", (count,))
    for name in ("command_valid", "position_valid", "encoder_valid"):
        ds(f"robot/{name}", (count,), np.uint8, 0)

    calibration = output.create_group("calibration")
    for rig, registration in registrations.items():
        group = calibration.create_group(rig)
        group["camera_matrix"] = registration.K
        group["left_camera_T_base"] = registration.left_camera_T_base
        group["right_camera_T_base"] = registration.right_camera_T_base
        group["roi_left_xywh"] = registration.roi_left_xywh
        group["roi_right_xywh"] = registration.roi_right_xywh
        group.attrs["baseline_m"] = registration.baseline_m
        group.attrs["zed_serial"] = registration.zed_serial
    return output


def _registrations_with_saved_calibration(
        session: Path, shapes,
) -> dict[str, object]:
    """Load the exact refined transforms used to create a shape HDF5."""
    registrations = {
        rig: load_session_registration(
            session, require_em=False, rig_id=rig) for rig in RIGS}
    if "calibration" not in shapes:
        return registrations
    for rig in RIGS:
        path = f"calibration/{rig}"
        if path not in shapes:
            continue
        saved = shapes[path]
        registrations[rig] = replace(
            registrations[rig],
            K=np.asarray(saved["camera_matrix"], dtype=np.float64),
            left_camera_T_base=np.asarray(
                saved["left_camera_T_base"], dtype=np.float64),
            right_camera_T_base=np.asarray(
                saved["right_camera_T_base"], dtype=np.float64))
    return registrations


def reconstruct_multi_camera_session(
        session_path: os.PathLike | str,
        output_dir: os.PathLike | str,
        observation_h5: dict[str, os.PathLike | str],
        config: MultiCameraConfig | None = None,
        write_snapshots: bool = False,
        snapshot_count: int = 16) -> dict:
    config = config or MultiCameraConfig()
    session = Path(session_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    caches = {
        rig: ObservationCache(observation_h5[rig], rig) for rig in RIGS}
    registrations = {
        rig: load_session_registration(
            session, require_em=False, rig_id=rig) for rig in RIGS}
    pairs = _pair_map(session)
    primary = caches["primary"]
    rows = []
    for primary_index, primary_frame in enumerate(primary.svo_frames):
        pair = pairs.get(int(primary_frame))
        if pair is None:
            continue
        oblique_frame, primary_ns, oblique_ns = pair
        oblique_index = caches["oblique"].index_by_svo.get(oblique_frame)
        if oblique_index is None:
            continue
        offset_ms = (oblique_ns - primary_ns) * 1e-6
        if abs(offset_ms) > config.maximum_pair_offset_ms:
            continue
        midpoint_ns = int(round(0.5 * (primary_ns + oblique_ns)))
        rows.append((primary_index, oblique_index, int(primary_frame),
                     oblique_frame, primary_ns, oblique_ns, midpoint_ns))
    if not rows:
        for cache in caches.values():
            cache.close()
        raise ValueError("no paired rows overlap the two observation caches")

    timestamps = np.asarray([row[6] for row in rows], np.int64)
    marker_templates = []
    for rig in RIGS:
        registration = registrations[rig]
        for eye in EYES:
            transform = (registration.left_camera_T_base if eye == "left"
                         else registration.right_camera_T_base)
            marker_templates.append(ViewObservation(
                view_id=f"{rig}_{eye}", K=registration.K,
                camera_T_base=transform,
                centerline_xy=np.empty((0, 2), dtype=np.float64)))
    marker_centers = []
    marker_widths = []
    marker_confidence = []
    marker_temporal_outlier_count = []
    marker_temporal_outlier_count_by_rig = {rig: [] for rig in RIGS}
    for primary_index, oblique_index, *_ in rows:
        per_view = []
        per_view_widths = []
        per_view_confidence = []
        temporal_outliers_by_rig = {rig: 0 for rig in RIGS}
        for rig, source_index in (
                ("primary", primary_index), ("oblique", oblique_index)):
            if not caches[rig].observation_valid[source_index]:
                # A decoded row that failed observation extraction is a
                # rig-local corruption event. Marker repair may provide
                # finite values for temporal continuity, but it must not make
                # this camera eligible as the frame's stereo reference.
                temporal_outliers_by_rig[rig] += 1
            for eye in EYES:
                per_view.append(np.asarray(
                    caches[rig].marker_tracks[eye]["centers"][source_index],
                    dtype=np.float64))
                per_view_widths.append(np.asarray(
                    caches[rig].marker_tracks[eye]["widths"][source_index],
                    dtype=np.float64))
                per_view_confidence.append(np.asarray(
                    caches[rig].marker_tracks[eye]["confidence"][source_index],
                    dtype=np.float64))
                temporal_outliers_by_rig[rig] += int(np.count_nonzero(
                    caches[rig].marker_tracks[eye]["outlier"][source_index]))
        marker_centers.append(per_view)
        marker_widths.append(per_view_widths)
        marker_confidence.append(per_view_confidence)
        marker_temporal_outlier_count.append(sum(
            temporal_outliers_by_rig.values()))
        for rig in RIGS:
            marker_temporal_outlier_count_by_rig[rig].append(
                temporal_outliers_by_rig[rig])
    (marker_centers, marker_source_slots, marker_inferred,
     marker_reference_code, _) = _associate_cross_camera_marker_identities(
        np.asarray(marker_centers), np.asarray(marker_widths),
        np.asarray(marker_confidence), marker_templates, timestamps)
    physical_slots = np.arange(4, dtype=np.int8)[None, None, :]
    marker_reassigned = (
        (marker_source_slots >= 0)
        & (marker_source_slots != physical_slots))
    cross_marker_rejected = marker_reassigned
    for rig in RIGS:
        marker_temporal_outlier_count_by_rig[rig] = np.asarray(
            marker_temporal_outlier_count_by_rig[rig], dtype=np.uint8)
    registrations, cross_rig_registration = _refine_cross_rig_registration(
        registrations, marker_centers, marker_inferred,
        marker_temporal_outlier_count_by_rig, config)
    robot = align_robot_streams(load_robot_streams(session), timestamps)
    metadata = {
        "session": session.name,
        "session_path": str(session),
        "observation_h5": {
            rig: str(Path(path).resolve()) for rig, path in observation_h5.items()},
        "rigs": list(RIGS),
        "view_ids": list(VIEW_IDS),
        "cross_rig_registration_refinement": cross_rig_registration,
    }
    h5_path = output_dir / "processed_shapes.h5"
    output = _create_output(
        h5_path, len(rows), config, registrations, metadata)
    output["multi_view/marker_temporal_outlier_count"][:] = np.asarray(
        marker_temporal_outlier_count, dtype=np.uint8)
    for rig in RIGS:
        output[
            f"multi_view/{rig}_marker_temporal_outlier_count"
        ][:] = marker_temporal_outlier_count_by_rig[rig]
    for view, view_id in enumerate(VIEW_IDS):
        output[f"multi_view/{view_id}_marker_centers_px"][:] = (
            marker_centers[:, view])
        output[f"multi_view/{view_id}_marker_cross_camera_rejected"][:] = (
            cross_marker_rejected[:, view].astype(np.uint8))
        output[f"multi_view/{view_id}_marker_source_slot"][:] = (
            marker_source_slots[:, view])
        output[f"multi_view/{view_id}_marker_cross_camera_inferred"][:] = (
            marker_inferred[:, view])
    output["multi_view/marker_reference_code"][:] = marker_reference_code
    previous_points = None
    previous_coefficients = None
    previous_timestamp_ns = None
    previous_terminal_base_mm = None
    accepted_fit_reference_rig = None
    accepted_ordered_stereo_rig = None
    ordered_disparity_states: dict[str, dict] = {}
    coefficient_velocity = np.zeros(
        (config.spline_basis_count, 3), dtype=np.float64)
    status_counts = {}
    started = time.perf_counter()
    initializer_codes = {"all": 1, "primary": 2, "oblique": 3, "temporal": 4}
    try:
        for index, row in enumerate(rows):
            (primary_index, oblique_index, primary_frame, oblique_frame,
             primary_ns, oblique_ns, midpoint_ns) = row
            output["frames/timestamp_ns"][index] = midpoint_ns
            output["frames/primary_svo_frame"][index] = primary_frame
            output["frames/oblique_svo_frame"][index] = oblique_frame
            output["frames/primary_timestamp_ns"][index] = primary_ns
            output["frames/oblique_timestamp_ns"][index] = oblique_ns
            output["frames/inter_rig_offset_ms"][index] = (
                (oblique_ns - primary_ns) * 1e-6)
            rig_observation_valid = {
                "primary": bool(
                    caches["primary"].observation_valid[primary_index]),
                "oblique": bool(
                    caches["oblique"].observation_valid[oblique_index]),
            }
            for rig in RIGS:
                output[f"frames/{rig}_observation_valid"][index] = int(
                    rig_observation_valid[rig])
            output["frames/observation_valid"][index] = int(any(
                rig_observation_valid.values()))
            offsets = {
                "primary": (primary_ns - midpoint_ns) * 1e-9,
                "oblique": (oblique_ns - midpoint_ns) * 1e-9,
            }
            try:
                observations = []
                view_ordinal = 0
                for rig, source_index in (
                        ("primary", primary_index),
                        ("oblique", oblique_index)):
                    for eye in EYES:
                        if rig_observation_valid[rig]:
                            observations.append(caches[rig].view(
                                source_index, eye, registrations[rig],
                                offsets[rig], marker_override=marker_centers[
                                    index, view_ordinal]))
                            output[
                                f"multi_view/{rig}_{eye}_axial_endpoints_px"
                            ][index] = observations[-1].axial_markers_xy[[0, 3]]
                        view_ordinal += 1
                (trusted_rig, stereo_terminals,
                 terminal_disagreement_mm) = _select_consistent_rig(
                    observations, previous_terminal_base_mm,
                    config.max_cross_rig_endpoint_disagreement_mm)
                corrupted_rigs = {
                    rig for rig in RIGS
                    if marker_temporal_outlier_count_by_rig[rig][index] > 0}
                ordered_stereo = _select_ordered_stereo_curve(
                    observations, trusted_rig,
                    accepted_ordered_stereo_rig, config,
                    disparity_states=ordered_disparity_states,
                    rig_timestamps_ns={
                        "primary": int(primary_ns),
                        "oblique": int(oblique_ns),
                    },
                    corrupted_rigs=corrupted_rigs)
                output[
                    "multi_view/ordered_stereo_available_rig_mask"
                ][index] = (
                    0 if ordered_stereo is None else
                    ordered_stereo["available_rig_mask"])
                usable_trusted_rig = (
                    trusted_rig
                    if trusted_rig is not None
                    and trusted_rig not in corrupted_rigs else None)
                available_rig_mask = (
                    0 if ordered_stereo is None else
                    int(ordered_stereo["available_rig_mask"]))
                fit_reference_rig, fit_rigs = (
                    _fit_reference_from_available_rigs(
                        usable_trusted_rig, available_rig_mask))
                if not fit_rigs:
                    # No rig passed the ordered-stereo topology gates. Retain
                    # any decoded, non-corrupt image evidence for diagnostics
                    # and possible temporal recovery, but final learning QC
                    # still rejects the frame through the zero available mask.
                    fit_rigs = {
                        rig for rig in RIGS
                        if rig not in corrupted_rigs
                        and rig_observation_valid[rig]}
                    if len(fit_rigs) == 1:
                        fit_reference_rig = next(iter(fit_rigs))
                fit_observations = _rig_restricted_observations(
                    observations, fit_reference_rig,
                    config.excluded_rig_weight)
                output["multi_view/trusted_rig_code"][index] = (
                    0 if trusted_rig is None else
                    (1 if trusted_rig == "primary" else 2))
                output["multi_view/fit_reference_rig_code"][index] = (
                    0 if fit_reference_rig is None else
                    (1 if fit_reference_rig == "primary" else 2))
                for rig, terminal in stereo_terminals.items():
                    output[f"multi_view/{rig}_stereo_terminal_base_mm"][
                        index] = terminal
                if usable_trusted_rig is not None:
                    previous_terminal_base_mm = stereo_terminals[
                        usable_trusted_rig]
                elif (ordered_stereo is not None
                      and ordered_stereo["rig"] in stereo_terminals):
                    previous_terminal_base_mm = stereo_terminals[
                        ordered_stereo["rig"]]
                elif (not corrupted_rigs
                      and all(rig in stereo_terminals for rig in RIGS)):
                    previous_terminal_base_mm = 0.5 * (
                        stereo_terminals["primary"]
                        + stereo_terminals["oblique"])
                temporal_adjacent = bool(
                    previous_points is not None
                    and previous_timestamp_ns is not None
                    and 0 < (midpoint_ns - previous_timestamp_ns) * 1e-6
                    <= config.max_temporal_gap_ms)
                if not temporal_adjacent:
                    coefficient_velocity.fill(0.0)
                ordered_stereo_innovation = np.nan
                ordered_stereo_innovation_rejected = False
                if (ordered_stereo is not None and temporal_adjacent
                        and previous_points is not None):
                    stereo_points = ordered_stereo["points_base_mm"]
                    ordered_stereo_innovation = 0.5 * (
                        float(np.mean(cKDTree(previous_points).query(
                            stereo_points)[0]))
                        + float(np.mean(cKDTree(stereo_points).query(
                            previous_points)[0])))
                    if (not np.isfinite(ordered_stereo_innovation)
                            or ordered_stereo_innovation
                            > config.ordered_stereo_max_innovation_mm):
                        ordered_stereo = None
                        ordered_stereo_innovation_rejected = True
                candidates = _initializer_candidates(
                    observations,
                    previous_points if temporal_adjacent else None,
                    config.nominal_length_mm)
                if not candidates:
                    raise ValueError("no finite multi-view initializer")
                _, initial, initializer = candidates[0]

                def fit_hypothesis(
                        hypothesis, hypothesis_observations,
                        use_temporal_prior):
                    return fit_multi_view_spline(
                        hypothesis, hypothesis_observations,
                        nominal_length_mm=config.nominal_length_mm,
                        output_samples=config.distal_samples,
                        basis_count=config.spline_basis_count,
                        fit_samples=config.fit_samples,
                        coverage_samples=config.coverage_samples,
                        length_sigma_mm=config.length_sigma_mm,
                        temporal_prior_points_base_mm=(
                            previous_points
                            if use_temporal_prior and temporal_adjacent
                            else None),
                        temporal_prior_sigma_mm=config.temporal_prior_sigma_mm,
                        coefficient_velocity_base_mm_s=coefficient_velocity,
                        stereo_curve_points_base_mm=(
                            None if ordered_stereo is None else
                            ordered_stereo["points_base_mm"]),
                        stereo_curve_timestamp_offset_s=(
                            0.0 if ordered_stereo is None else
                            ordered_stereo["timestamp_offset_s"]),
                        stereo_curve_sigma_mm=config.ordered_stereo_sigma_mm,
                        stereo_curve_weight=(
                            0.0 if ordered_stereo is None else
                            config.ordered_stereo_weight
                            * ordered_stereo["support_confidence"]),
                        stereo_curve_quadratic=(ordered_stereo is not None),
                        max_nfev=config.max_nfev)

                weighted_observations = _adaptive_view_weights(
                    initial, fit_observations)
                result = fit_hypothesis(
                    initial, weighted_observations, True)
                result_observations = weighted_observations
                successful = [(result, initializer, result_observations)]
                reference_rig_changed = (
                    fit_reference_rig != accepted_fit_reference_rig)
                quality_requires_reacquisition = (
                    _result_requires_reacquisition(
                        result, config, trusted_rigs=fit_rigs))
                reacquisition_reason = (
                    int(reference_rig_changed)
                    | (int(quality_requires_reacquisition) << 1))
                reacquisition = bool(reacquisition_reason)
                attempted_count = 0
                if reacquisition:
                    # Refit every geometric hypothesis with topology-only
                    # weights and no temporal coefficient attraction. A stale
                    # temporal curve therefore cannot declare recovered image
                    # evidence to be an outlier before it is tested.
                    successful = []
                    for _, hypothesis, name in candidates:
                        attempted_count += 1
                        try:
                            candidate_result = fit_hypothesis(
                                hypothesis, fit_observations, False)
                        except (ArithmeticError, RuntimeError, ValueError,
                                np.linalg.LinAlgError):
                            continue
                        if (candidate_result.optimizer_success
                                and np.all(np.isfinite(
                                    candidate_result.points_base_mm))):
                            successful.append(
                                (candidate_result, name, fit_observations))
                    # Retain the ordinary result as a candidate if all-view
                    # robust fitting could not improve it.
                    if (result.optimizer_success
                            and np.all(np.isfinite(result.points_base_mm))):
                        successful.append(
                            (result, initializer, result_observations))
                if not successful:
                    raise ValueError("multi-view optimizer failed")
                result, initializer, result_observations = min(
                    successful, key=lambda item: _result_selection_score(
                        item[0], config.nominal_length_mm,
                        previous_points if temporal_adjacent else None,
                        trusted_rigs=fit_rigs))
                temporal_polished = False
                if reacquisition and temporal_adjacent:
                    # Multi-start candidates intentionally omit temporal
                    # attraction so stale state cannot prevent recovery. Once
                    # the image-supported basin is selected, give that same
                    # basin one common velocity-aware temporal refinement.
                    # This removes initializer-dependent frame alternation
                    # without allowing a stale curve to win the search.
                    selected_score = _result_selection_score(
                        result, config.nominal_length_mm, previous_points,
                        trusted_rigs=fit_rigs)
                    try:
                        polished = fit_hypothesis(
                            result.points_base_mm, fit_observations, True)
                        polished_score = _result_selection_score(
                            polished, config.nominal_length_mm,
                            previous_points, trusted_rigs=fit_rigs)
                        if (polished.optimizer_success
                                and np.all(np.isfinite(
                                    polished.points_base_mm))
                                and polished_score <= selected_score
                                + config.temporal_polish_max_score_increase):
                            result = polished
                            result_observations = fit_observations
                            temporal_polished = True
                    except (ArithmeticError, RuntimeError, ValueError,
                            np.linalg.LinAlgError):
                        pass
                rig_errors = _result_rig_errors(result)
                rig_coverage_errors = _result_rig_coverage_errors(result)
                maximum_rig_error = max(
                    rig_errors[rig] for rig in fit_rigs)
                maximum_rig_coverage = max(
                    rig_coverage_errors[rig] for rig in fit_rigs)
                maximum_terminal_error = _result_terminal_error_for_rigs(
                    result, fit_rigs)
                whole_rig_failure = bool(
                    not np.isfinite(maximum_rig_error)
                    or maximum_rig_error
                    > config.whole_rig_max_symmetric_px)
                terminal_failure = bool(
                    not np.isfinite(maximum_terminal_error)
                    or maximum_terminal_error
                    > config.max_terminal_error_px)
                coverage_failure = bool(
                    not np.isfinite(maximum_rig_coverage)
                    or maximum_rig_coverage
                    > config.max_rig_coverage_mean_px)
                length_failure = bool(
                    abs(float(result.length_residual_mm))
                    > config.max_direct_length_deviation_mm)
                direct_quality_failure = bool(
                    whole_rig_failure or terminal_failure
                    or coverage_failure or length_failure)
                direct_valid = bool(
                    result.optimizer_success
                    and np.all(np.isfinite(result.points_base_mm))
                    and result.final_symmetric_mean_px
                    <= config.max_final_symmetric_px
                    and not direct_quality_failure)
                geometry = curve_geometry(
                    result.points_base_mm, smoothing_mm=0.25,
                    basis_count=config.spline_basis_count)
                output["distal/points_base_mm"][index] = geometry.points_mm
                output["distal/pre_temporal_points_base_mm"][index] = (
                    geometry.points_mm)
                output["distal/s_mm"][index] = cumulative_arclength(
                    geometry.points_mm)
                output["distal/tangent_base"][index] = geometry.tangent
                output["distal/curvature_per_mm"][index] = (
                    geometry.curvature_per_mm)
                output["distal/base_position_base_mm"][index] = (
                    geometry.points_mm[0])
                output["distal/observation_class"][index] = 2
                output["multi_view/coefficients_base_mm"][index] = (
                    result.coefficients_base_mm)
                output["multi_view/coefficient_velocity_base_mm_s"][index] = (
                    coefficient_velocity)
                active_observations = [
                    item for item in result_observations
                    if item.weight >= 0.25]
                output["multi_view/active_view_count"][index] = len(
                    active_observations)
                output["multi_view/initializer_code"][index] = (
                    initializer_codes[initializer])
                output["multi_view/reacquisition_attempted"][index] = int(
                    reacquisition)
                output["multi_view/reacquisition_candidate_count"][index] = (
                    attempted_count)
                output["multi_view/reacquisition_reason_code"][index] = (
                    reacquisition_reason)
                output[
                    "multi_view/reacquisition_temporal_polished"
                ][index] = int(temporal_polished)
                output["multi_view/ordered_stereo_reference_code"][index] = (
                    0 if ordered_stereo is None else
                    (1 if ordered_stereo["rig"] == "primary" else 2))
                if ordered_stereo is not None:
                    output[
                        "multi_view/ordered_stereo_disparity_coefficients_px"
                    ][index] = ordered_stereo[
                        "disparity_coefficients_px"]
                    trusted_p95 = [
                        result.view_model_p95_px[item.view_id]
                        for item in active_observations]
                maximum_p95 = max(
                    trusted_p95 or result.view_model_p95_px.values())
                quality = {
                    "reprojection_p95_px": maximum_p95,
                    "fitted_reprojection_p95_px": maximum_p95,
                    "stereo_condition": 1.0,
                    "joint_final_symmetric_mean_px": (
                        result.final_symmetric_mean_px),
                    "multi_view_initial_symmetric_mean_px": (
                        result.initial_symmetric_mean_px),
                    "multi_view_arc_length_mm": result.arc_length_mm,
                    "multi_view_length_residual_mm": result.length_residual_mm,
                    "multi_view_ill_conditioned_view_count": (
                        len(observations) - len(active_observations)),
                    "multi_view_optimizer_cost": result.optimizer_cost,
                    "multi_view_optimizer_evaluations": (
                        result.optimizer_evaluations),
                    "multi_view_optimizer_success": int(result.optimizer_success),
                    "multi_view_primary_rig_symmetric_mean_px": (
                        rig_errors["primary"]),
                    "multi_view_oblique_rig_symmetric_mean_px": (
                        rig_errors["oblique"]),
                    "multi_view_max_rig_symmetric_mean_px": maximum_rig_error,
                    "multi_view_whole_rig_failure": int(whole_rig_failure),
                    "multi_view_max_terminal_error_px": maximum_terminal_error,
                    "multi_view_max_rig_coverage_mean_px": (
                        maximum_rig_coverage),
                    "multi_view_terminal_failure": int(terminal_failure),
                    "multi_view_coverage_failure": int(coverage_failure),
                    "multi_view_length_failure": int(length_failure),
                    "multi_view_direct_quality_failure": int(
                        direct_quality_failure),
                    "multi_view_cross_rig_terminal_disagreement_mm": (
                        terminal_disagreement_mm),
                    "multi_view_ordered_stereo_match_count": (
                        0 if ordered_stereo is None else
                        ordered_stereo["match_count"]),
                    "multi_view_ordered_stereo_epipolar_p95_px": (
                        np.nan if ordered_stereo is None else
                        ordered_stereo["epipolar_p95_px"]),
                    "multi_view_ordered_stereo_curve_length_mm": (
                        np.nan if ordered_stereo is None else
                        ordered_stereo["curve_length_mm"]),
                    "multi_view_ordered_stereo_score": (
                        0.0 if ordered_stereo is None else
                        ordered_stereo["score"]),
                    "multi_view_ordered_stereo_disparity_residual_p95_px": (
                        np.nan if ordered_stereo is None else
                        ordered_stereo["disparity_residual_p95_px"]),
                    "multi_view_ordered_stereo_right_support_p95_px": (
                        np.nan if ordered_stereo is None else
                        ordered_stereo["right_support_p95_px"]),
                    "multi_view_ordered_stereo_support_confidence": (
                        0.0 if ordered_stereo is None else
                        ordered_stereo["support_confidence"]),
                    "multi_view_ordered_stereo_innovation_mm": (
                        ordered_stereo_innovation),
                    "multi_view_ordered_stereo_innovation_rejected": int(
                        ordered_stereo_innovation_rejected),
                    "multi_view_ordered_stereo_disparity_temporal_used": (
                        0 if ordered_stereo is None else int(
                            ordered_stereo["disparity_temporal_used"])),
                    "multi_view_cross_camera_marker_rejection_count": int(
                        np.count_nonzero(cross_marker_rejected[index])),
                    "multi_view_cross_camera_marker_reassignment_count": int(
                        np.count_nonzero(marker_reassigned[index])),
                    "multi_view_cross_camera_marker_inferred_count": int(
                        np.count_nonzero(marker_inferred[index])),
                }
                for observation in result_observations:
                    view = observation.view_id
                    quality.update({
                        f"{view}_model_mean_px": (
                            result.view_model_mean_px[view]),
                        f"{view}_model_p95_px": result.view_model_p95_px[view],
                        f"{view}_coverage_mean_px": (
                            result.view_coverage_mean_px[view]),
                        f"{view}_coverage_p95_px": (
                            result.view_coverage_p95_px[view]),
                        f"{view}_terminal_start_px": (
                            result.view_terminal_start_px[view]),
                        f"{view}_terminal_end_px": (
                            result.view_terminal_end_px[view]),
                        f"{view}_weight": observation.weight,
                        f"{view}_topology_weight": (
                            observation.topology_weight),
                        f"{view}_sharp_turn_clusters": (
                            result.view_sharp_turn_clusters[view]),
                    })
                for name, value in quality.items():
                    output[f"quality/{name}"][index] = value
                output["frames/valid"][index] = int(direct_valid)
                status = (
                    "valid" if direct_valid
                    else "multi-view direct quality exceeds threshold")
                output["frames/status"][index] = status
                if (direct_valid and previous_coefficients is not None
                        and temporal_adjacent):
                    delta_s = (midpoint_ns - previous_timestamp_ns) * 1e-9
                    measured_velocity = (
                        result.coefficients_base_mm - previous_coefficients
                    ) / max(delta_s, 1e-9)
                    speed = np.linalg.norm(measured_velocity, axis=1)
                    scale = np.minimum(
                        1.0, config.max_coefficient_speed_mm_s
                        / np.maximum(speed, 1e-9))
                    measured_velocity *= scale[:, None]
                    alpha = float(np.clip(
                        config.velocity_update_fraction, 0.0, 1.0))
                    coefficient_velocity = (
                        (1.0 - alpha) * coefficient_velocity
                        + alpha * measured_velocity)
                elif direct_valid:
                    coefficient_velocity.fill(0.0)
                if direct_valid:
                    previous_points = result.points_base_mm.copy()
                    previous_coefficients = result.coefficients_base_mm.copy()
                    previous_timestamp_ns = midpoint_ns
                    accepted_fit_reference_rig = fit_reference_rig
                    accepted_ordered_stereo_rig = (
                        None if ordered_stereo is None else
                        ordered_stereo["rig"])
                    if ordered_stereo is not None:
                        rig = ordered_stereo["rig"]
                        ordered_disparity_states[rig] = {
                            "coefficients_px": ordered_stereo[
                                "disparity_coefficients_px"].copy(),
                            "timestamp_ns": int(
                                primary_ns if rig == "primary"
                                else oblique_ns),
                        }
            except (ArithmeticError, RuntimeError, ValueError,
                    np.linalg.LinAlgError) as error:
                status = str(error)
                output["frames/status"][index] = status
            status_counts[status] = status_counts.get(status, 0) + 1
            if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(rows):
                elapsed = time.perf_counter() - started
                print(f"[{index + 1}/{len(rows)}] {status}; "
                      f"{(index + 1) / max(elapsed, 1e-9):.2f} frames/s",
                      flush=True)

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
            output[f"robot/{name}"][:] = values
        output.flush()
    finally:
        output.close()
        for cache in caches.values():
            cache.close()

    smoothing = smooth_distal_spline_coefficients_hdf5(
        h5_path, cutoff_hz=config.temporal_cutoff_hz,
        maximum_gap_ms=config.temporal_max_gap_ms,
        basis_count=config.spline_basis_count,
        terminal_cutoff_hz=config.temporal_terminal_cutoff_hz,
        observation_blend=config.temporal_observation_blend,
        terminal_observation_blend=(
            config.temporal_terminal_observation_blend),
        max_learning_mask_width_px=float("inf"),
        reject_recovered_outliers=False)
    interpolation = interpolate_short_spline_gaps_hdf5(
        h5_path, maximum_gap_ms=config.interpolation_max_gap_ms)
    import h5py
    with h5py.File(h5_path, "r+") as output:
        points = output["distal/points_base_mm"][:]
        length = np.sum(np.linalg.norm(np.diff(points, axis=1), axis=2), axis=1)
        flags = output["frames/learning_rejection_flags"][:].astype(np.uint16)
        implausible = (
            output["frames/valid"][:].astype(bool)
            & ((length < config.min_learning_length_mm)
               | (length > config.max_learning_length_mm)))
        flags[implausible] |= LENGTH_REJECTION_FLAG
        whole_rig_failure = output[
            "quality/multi_view_whole_rig_failure"][:].astype(bool)
        interpolated = (
            output["frames/curve_temporally_interpolated"][:].astype(bool)
            if "frames/curve_temporally_interpolated" in output
            else np.zeros(len(flags), dtype=bool))
        flags[whole_rig_failure & ~interpolated] |= WHOLE_RIG_REJECTION_FLAG
        primary_corrupt = output[
            "multi_view/primary_marker_temporal_outlier_count"][:] > 0
        oblique_corrupt = output[
            "multi_view/oblique_marker_temporal_outlier_count"][:] > 0
        both_rigs_corrupt = primary_corrupt & oblique_corrupt
        no_usable_stereo_rig = (
            output["multi_view/ordered_stereo_available_rig_mask"][:] == 0)
        # Corruption is camera-specific.  One bad rig must not invalidate a
        # frame reconstructed from the other independently usable stereo rig.
        # Reject only simultaneous corruption, or a frame for which every
        # non-corrupt rig is topologically/epipolarly unusable (including a
        # good camera viewing an ill-posed configuration).
        abrupt_jump = both_rigs_corrupt | no_usable_stereo_rig
        abrupt_rejected = (
            abrupt_jump if config.reject_abrupt_observation_jumps else
            np.zeros(len(flags), dtype=bool))
        flags[abrupt_rejected] |= ABRUPT_OBSERVATION_REJECTION_FLAG
        output["quality/abrupt_observation_jump_rejected"][:] = (
            abrupt_rejected.astype(np.uint8))
        final_terminal = np.full(
            (len(points), len(VIEW_IDS), 2), np.nan, dtype=np.float32)
        for view_ordinal, view_id in enumerate(VIEW_IDS):
            rig, eye = view_id.rsplit("_", 1)
            registration = registrations[rig]
            transform = (registration.left_camera_T_base if eye == "left"
                         else registration.right_camera_T_base)
            endpoints = output[
                f"multi_view/{view_id}_axial_endpoints_px"][:]
            finite_frames = np.flatnonzero(np.all(
                np.isfinite(points), axis=(1, 2)))
            if len(finite_frames):
                projected = project_points(
                    registration.K, transform,
                    points[finite_frames][:, [0, -1]].reshape(-1, 3)
                )[0].reshape(len(finite_frames), 2, 2)
                distance = np.linalg.norm(
                    projected - endpoints[finite_frames], axis=2)
                distance[~np.all(np.isfinite(
                    endpoints[finite_frames]), axis=2)] = np.nan
                final_terminal[finite_frames, view_ordinal] = distance
            name = f"quality/{view_id}_final_terminal_start_px"
            if name not in output:
                output.create_dataset(
                    name, shape=(len(points),), dtype=np.float32,
                    fillvalue=np.nan, compression="gzip",
                    compression_opts=1, shuffle=True)
                output.create_dataset(
                    f"quality/{view_id}_final_terminal_end_px",
                    shape=(len(points),), dtype=np.float32,
                    fillvalue=np.nan, compression="gzip",
                    compression_opts=1, shuffle=True)
            output[name][:] = final_terminal[:, view_ordinal, 0]
            output[f"quality/{view_id}_final_terminal_end_px"][:] = (
                final_terminal[:, view_ordinal, 1])
        trusted_code = output["multi_view/fit_reference_rig_code"][:]
        trusted_terminal = final_terminal.copy()
        trusted_terminal[trusted_code == 1, 2:] = np.nan
        trusted_terminal[trusted_code == 2, :2] = np.nan
        finite_terminal = np.any(
            np.isfinite(trusted_terminal), axis=(1, 2))
        maximum_final_terminal = np.full(len(points), np.nan, np.float32)
        maximum_final_terminal[finite_terminal] = np.nanmax(
            trusted_terminal[finite_terminal], axis=(1, 2))
        final_name = "quality/final_multi_view_max_terminal_error_px"
        if final_name not in output:
            output.create_dataset(
                final_name, shape=(len(points),), dtype=np.float32,
                fillvalue=np.nan, compression="gzip",
                compression_opts=1, shuffle=True)
        output[final_name][:] = maximum_final_terminal
        final_terminal_bad = (
            output["frames/valid"][:].astype(bool)
            & finite_terminal
            & (maximum_final_terminal > config.max_terminal_error_px))
        flags[final_terminal_bad] |= np.uint16(1 << 5)
        output["frames/learning_rejection_flags"][:] = flags
        output["frames/learning_valid"][:] = (flags == 0).astype(np.uint8)
        # The generic temporal smoother knows flags 1--16 and rewrites this
        # attribute. Restore the dual-camera pipeline's additional length flag.
        output.attrs["learning_rejection_flags_json"] = json.dumps({
            1: "reconstruction_invalid",
            2: "temporal_long_gap_unsupported",
            4: "whole_shape_temporal_outlier",
            8: "terminal_temporal_outlier",
            16: "mask_effective_width",
            32: "final_image_fit",
            64: "distal_arc_length_outside_learning_range",
            128: "whole_camera_image_fit",
            256: "abrupt_observation_jump",
        }, sort_keys=True)
        output.flush()
        valid_count = int(np.count_nonzero(output["frames/valid"][:]))
        learning_count = int(np.count_nonzero(
            output["frames/learning_valid"][:]))

    if write_snapshots:
        _write_snapshots(
            session, output_dir, h5_path, observation_h5,
            registrations, snapshot_count)
    summary = {
        "session": session.name,
        "output_h5": str(h5_path),
        "paired_frame_count": len(rows),
        "valid_frame_count": valid_count,
        "learning_valid_count": learning_count,
        "status_counts": status_counts,
        "cross_rig_registration_refinement": cross_rig_registration,
        "temporal_smoothing": smoothing,
        "interpolation": interpolation,
        "elapsed_s": time.perf_counter() - started,
    }
    (output_dir / "processing_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _unpack_mask(dataset, index: int) -> np.ndarray:
    _, _, width, height = tuple(int(v) for v in dataset.attrs["roi_xywh"])
    return np.unpackbits(
        np.asarray(dataset[index]), bitorder="little",
        count=width * height).reshape(height, width)


def validate_cross_rig_registration(
        session: Path, shape_h5: Path, output_dir: Path,
        snapshot_count: int = 24,
) -> dict:
    """Audit the marker-derived SE(3) correction without spline fitting."""
    import h5py

    session = Path(session).resolve()
    shape_h5 = Path(shape_h5).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original = {
        rig: load_session_registration(
            session, require_em=False, rig_id=rig) for rig in RIGS}
    with h5py.File(shape_h5, "r") as shapes:
        centers = np.stack([
            shapes[f"multi_view/{view}_marker_centers_px"][:]
            for view in VIEW_IDS], axis=1)
        inferred = np.stack([
            shapes[f"multi_view/{view}_marker_cross_camera_inferred"][:]
            for view in VIEW_IDS], axis=1).astype(bool)
        temporal_outliers = {
            rig: np.asarray(shapes[
                f"multi_view/{rig}_marker_temporal_outlier_count"][:])
            for rig in RIGS}
        timestamps_ns = np.asarray(shapes["frames/timestamp_ns"][:], np.int64)
        source_frames = {
            rig: np.asarray(
                shapes[f"frames/{rig}_svo_frame"][:], np.int64)
            for rig in RIGS}
    refined, diagnostic = _refine_cross_rig_registration(
        original, centers, inferred, temporal_outliers,
        MultiCameraConfig())
    if not diagnostic.get("applied"):
        report = {"shape_h5": str(shape_h5), **diagnostic}
        (output_dir / "cross_rig_registration_validation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report

    view_index = {name: index for index, name in enumerate(VIEW_IDS)}
    primary_projections = {}
    for eye in EYES:
        transform = (original["primary"].left_camera_T_base
                     if eye == "left" else
                     original["primary"].right_camera_T_base)
        primary_projections[eye] = (
            original["primary"].K @ transform[:3])
    per_frame_before = np.full(len(centers), np.nan, dtype=np.float64)
    per_frame_after = np.full(len(centers), np.nan, dtype=np.float64)
    primary_points = np.full(
        (len(centers), 4, 3), np.nan, dtype=np.float64)
    for frame in range(len(centers)):
        before_errors = []
        after_errors = []
        for marker in range(4):
            left = centers[
                frame, view_index["primary_left"], marker]
            right = centers[
                frame, view_index["primary_right"], marker]
            if not (np.all(np.isfinite(left))
                    and np.all(np.isfinite(right))
                    and not np.any(inferred[frame, :, marker])):
                continue
            point = _triangulate_marker_pair(
                left, primary_projections["left"],
                right, primary_projections["right"])
            if point is None:
                continue
            primary_points[frame, marker] = point
            for eye in EYES:
                observed = centers[
                    frame, view_index[f"oblique_{eye}"], marker]
                if not np.all(np.isfinite(observed)):
                    continue
                old_transform = (
                    original["oblique"].left_camera_T_base
                    if eye == "left" else
                    original["oblique"].right_camera_T_base)
                new_transform = (
                    refined["oblique"].left_camera_T_base
                    if eye == "left" else
                    refined["oblique"].right_camera_T_base)
                old_xy = project_points(
                    original["oblique"].K, old_transform,
                    point[None])[0][0]
                new_xy = project_points(
                    refined["oblique"].K, new_transform,
                    point[None])[0][0]
                if np.all(np.isfinite(old_xy)):
                    before_errors.append(float(np.linalg.norm(
                        old_xy - observed)))
                if np.all(np.isfinite(new_xy)):
                    after_errors.append(float(np.linalg.norm(
                        new_xy - observed)))
        if before_errors:
            per_frame_before[frame] = float(np.median(before_errors))
        if after_errors:
            per_frame_after[frame] = float(np.median(after_errors))
    finite = np.isfinite(per_frame_before) & np.isfinite(per_frame_after)
    report = {
        "shape_h5": str(shape_h5),
        **diagnostic,
        "validation_frame_count": int(np.count_nonzero(finite)),
        "cross_reprojection_before_median_px": float(np.median(
            per_frame_before[finite])),
        "cross_reprojection_before_p95_px": float(np.percentile(
            per_frame_before[finite], 95)),
        "cross_reprojection_after_median_px": float(np.median(
            per_frame_after[finite])),
        "cross_reprojection_after_p95_px": float(np.percentile(
            per_frame_after[finite], 95)),
    }
    with (output_dir / "cross_rig_registration_frames.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_index", "timestamp_ns",
                         "before_median_px", "after_median_px"))
        for index in np.flatnonzero(finite):
            writer.writerow((int(index), int(timestamps_ns[index]),
                             float(per_frame_before[index]),
                             float(per_frame_after[index])))

    valid_indices = np.flatnonzero(finite)
    count = min(max(0, int(snapshot_count)), len(valid_indices))
    if count:
        uniform = valid_indices[np.linspace(
            0, len(valid_indices) - 1, max(1, count // 2),
            dtype=int)]
        worst = valid_indices[np.argsort(
            per_frame_before[valid_indices])[-(count - len(uniform)):]] \
            if count > len(uniform) else np.empty(0, dtype=int)
        selected = np.unique(np.concatenate([uniform, worst]))
        if len(selected) > count:
            selected = selected[:count]
        snapshot_dir = output_dir / "cross_rig_registration_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        with SvoReader(find_svo(session, rig_id="primary")) as primary_svo, \
                SvoReader(find_svo(
                    session, rig_id="oblique")) as oblique_svo:
            readers = {"primary": primary_svo, "oblique": oblique_svo}
            for index in selected:
                panels = []
                for rig in RIGS:
                    _, left_image, right_image = readers[rig].read(
                        int(source_frames[rig][index]))
                    for eye, source_image in (
                            ("left", left_image), ("right", right_image)):
                        panel = source_image.copy()
                        for marker in range(4):
                            observed = centers[
                                index, view_index[f"{rig}_{eye}"], marker]
                            if np.all(np.isfinite(observed)):
                                cv2.circle(panel, tuple(np.rint(
                                    observed).astype(int)), 7,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                            point = primary_points[index, marker]
                            if (rig != "oblique"
                                    or not np.all(np.isfinite(point))):
                                continue
                            for registration, color in (
                                    (original[rig], (0, 0, 255)),
                                    (refined[rig], (255, 255, 0))):
                                transform = (
                                    registration.left_camera_T_base
                                    if eye == "left" else
                                    registration.right_camera_T_base)
                                xy = project_points(
                                    registration.K, transform,
                                    point[None])[0][0]
                                if np.all(np.isfinite(xy)):
                                    x, y = tuple(np.rint(xy).astype(int))
                                    cv2.drawMarker(
                                        panel, (x, y), color,
                                        cv2.MARKER_CROSS, 15, 2,
                                        cv2.LINE_AA)
                        cv2.putText(
                            panel, f"{rig} {eye}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 0), 2, cv2.LINE_AA)
                        panels.append(cv2.resize(
                            panel, (960, 540),
                            interpolation=cv2.INTER_AREA))
                mosaic = np.vstack([
                    np.hstack(panels[:2]), np.hstack(panels[2:])])
                cv2.putText(
                    mosaic,
                    ("green=observed  red=board registration  "
                     "cyan=refined registration"),
                    (20, 1070), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2, cv2.LINE_AA)
                path = snapshot_dir / f"frame_{int(index):06d}.png"
                if not cv2.imwrite(str(path), mosaic):
                    raise OSError(f"failed to write {path}")
    report["snapshot_count"] = int(count)
    report["snapshot_dir"] = str(
        output_dir / "cross_rig_registration_snapshots")
    (output_dir / "cross_rig_registration_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _render_four_view_mosaic(
        index: int, shapes, sources: dict, readers: dict,
        source_maps: dict, registrations: dict,
        panel_size: tuple[int, int] = (960, 540)) -> np.ndarray:
    """Render the same four-panel overlay used by snapshots and videos."""
    panels = []
    points = np.asarray(shapes["distal/points_base_mm"][index], float)
    learning_valid = bool(shapes["frames/learning_valid"][index])
    for rig in RIGS:
        svo_frame = int(shapes[f"frames/{rig}_svo_frame"][index])
        _, left, right = readers[rig].read(svo_frame)
        source_index = source_maps[rig][svo_frame]
        for eye, frame_image in (("left", left), ("right", right)):
            registration = registrations[rig]
            roi = (registration.roi_left_xywh if eye == "left"
                   else registration.roi_right_xywh)
            x, y, width, height = roi
            mask_ds = sources[rig][f"images/{eye}/mask_packbits"]
            mask = _unpack_mask(mask_ds, source_index)
            overlay = frame_image.copy()
            crop = overlay[y:y + height, x:x + width]
            green = np.zeros_like(crop)
            green[..., 1] = 255
            crop[:] = np.where(
                mask[..., None] > 0,
                cv2.addWeighted(crop, 0.65, green, 0.35, 0), crop)
            observed = np.asarray(sources[rig][
                f"images/{eye}/observed_centerline_px"][source_index])
            observed = observed[np.all(np.isfinite(observed), axis=1)]
            if len(observed) > 1:
                cv2.polylines(
                    overlay, [np.rint(observed).astype(np.int32)],
                    False, (255, 255, 0), 2, cv2.LINE_AA)
            transform = (registration.left_camera_T_base
                         if eye == "left"
                         else registration.right_camera_T_base)
            if np.all(np.isfinite(points)):
                projected = project_points(
                    registration.K, transform, points)[0]
                cv2.polylines(
                    overlay, [np.rint(projected).astype(np.int32)],
                    False, (0, 255, 255), 3, cv2.LINE_AA)
            label_color = (0, 0, 255) if learning_valid else (0, 0, 180)
            cv2.putText(
                overlay, f"{rig} {eye} frame {index}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
            if not learning_valid:
                cv2.putText(
                    overlay, "LEARNING REJECTED", (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            panels.append(cv2.resize(overlay, panel_size))
    return np.vstack([
        np.hstack(panels[:2]), np.hstack(panels[2:])])


def _write_snapshots(
        session: Path, output_dir: Path, shape_h5: Path,
        observation_h5: dict, registrations: dict,
        snapshot_count: int) -> None:
    import h5py

    snapshot_dir = output_dir / "overlay_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(shape_h5, "r") as shapes, \
            h5py.File(observation_h5["primary"], "r") as primary_obs, \
            h5py.File(observation_h5["oblique"], "r") as oblique_obs, \
            SvoReader(find_svo(session, rig_id="primary")) as primary_svo, \
            SvoReader(find_svo(session, rig_id="oblique")) as oblique_svo:
        sources = {"primary": primary_obs, "oblique": oblique_obs}
        readers = {"primary": primary_svo, "oblique": oblique_svo}
        count = len(shapes["frames/timestamp_ns"])
        indices = np.unique(np.rint(np.linspace(
            0, count - 1, min(max(1, snapshot_count), count))).astype(int))
        source_maps = {
            rig: {int(frame): index for index, frame in enumerate(
                source["frames/svo_frame"][:])}
            for rig, source in sources.items()}
        for index in indices:
            mosaic = _render_four_view_mosaic(
                index, shapes, sources, readers, source_maps, registrations)
            path = snapshot_dir / f"frame_{index:06d}.png"
            if not cv2.imwrite(str(path), mosaic):
                raise OSError(f"failed to write {path}")


def write_four_view_overlay_video(
        session: Path, shape_h5: Path, observation_h5: dict,
        output_path: Path, registrations: dict | None = None,
        scale: float = 1.0, stride: int = 1,
        maximum_frames: int | None = None) -> dict:
    """Render a four-view MP4 from existing results without reconstruction."""
    import h5py

    supplied_registrations = registrations is not None
    stride = max(1, int(stride))
    scale = float(np.clip(scale, 0.25, 1.0))
    panel_size = (
        max(2, int(round(960 * scale)) // 2 * 2),
        max(2, int(round(540 * scale)) // 2 * 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with h5py.File(shape_h5, "r") as shapes, \
            h5py.File(observation_h5["primary"], "r") as primary_obs, \
            h5py.File(observation_h5["oblique"], "r") as oblique_obs, \
            SvoReader(find_svo(session, rig_id="primary")) as primary_svo, \
            SvoReader(find_svo(session, rig_id="oblique")) as oblique_svo:
        if not supplied_registrations:
            registrations = _registrations_with_saved_calibration(
                session, shapes)
        sources = {"primary": primary_obs, "oblique": oblique_obs}
        readers = {"primary": primary_svo, "oblique": oblique_svo}
        source_maps = {
            rig: {int(frame): index for index, frame in enumerate(
                source["frames/svo_frame"][:])}
            for rig, source in sources.items()}
        count = len(shapes["frames/timestamp_ns"])
        indices = np.arange(0, count, stride, dtype=int)
        if maximum_frames is not None:
            indices = indices[:max(0, int(maximum_frames))]
        if not len(indices):
            raise ValueError("overlay selection contains no frames")
        timestamps = np.asarray(shapes["frames/timestamp_ns"][:], np.int64)
        positive_delta = np.diff(timestamps)
        positive_delta = positive_delta[positive_delta > 0]
        source_fps = (
            30.0 if not len(positive_delta)
            else 1e9 / float(np.median(positive_delta)))
        output_fps = source_fps / stride
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps,
            (2 * panel_size[0], 2 * panel_size[1]))
        if not writer.isOpened():
            raise OSError(f"failed to open video writer for {output_path}")
        try:
            for ordinal, index in enumerate(indices):
                mosaic = _render_four_view_mosaic(
                    int(index), shapes, sources, readers, source_maps,
                    registrations, panel_size=panel_size)
                writer.write(mosaic)
                if ordinal == 0 or (ordinal + 1) % 200 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"[overlay {ordinal + 1}/{len(indices)}] "
                        f"{(ordinal + 1) / max(elapsed, 1e-9):.2f} frames/s",
                        flush=True)
        finally:
            writer.release()
    return {
        "output_video": str(output_path),
        "frame_count": int(len(indices)),
        "fps": float(output_fps),
        "width": int(2 * panel_size[0]),
        "height": int(2 * panel_size[1]),
        "elapsed_s": float(time.perf_counter() - started),
    }


def resmooth_multi_camera_hdf5(
        session_path: os.PathLike | str,
        source_h5: os.PathLike | str,
        output_dir: os.PathLike | str,
        config: MultiCameraConfig | None = None) -> dict:
    """Copy and re-filter a completed four-view reconstruction.

    The independent per-frame splines are retained in
    ``distal/pre_temporal_points_base_mm``.  This mode therefore changes only
    the zero-phase temporal result and refreshes the final terminal/length QC;
    image observations and per-frame reconstruction are not repeated.
    """
    config = config or MultiCameraConfig()
    session = Path(session_path).resolve()
    source = Path(source_h5).resolve()
    destination_dir = Path(output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "processed_shapes.h5"
    if source == destination:
        raise ValueError(
            "resmooth output must differ from the source HDF5")
    started = time.perf_counter()
    shutil.copy2(source, destination)

    import h5py
    with h5py.File(destination, "r") as output:
        preserved_flags = output[
            "frames/learning_rejection_flags"][:].astype(np.uint16)

    smoothing = smooth_distal_spline_coefficients_hdf5(
        destination, cutoff_hz=config.temporal_cutoff_hz,
        maximum_gap_ms=config.temporal_max_gap_ms,
        basis_count=config.spline_basis_count,
        terminal_cutoff_hz=config.temporal_terminal_cutoff_hz,
        observation_blend=config.temporal_observation_blend,
        terminal_observation_blend=(
            config.temporal_terminal_observation_blend),
        max_learning_mask_width_px=float("inf"),
        reject_recovered_outliers=False)

    with h5py.File(destination, "r+") as output:
        registrations = _registrations_with_saved_calibration(
            session, output)
        points = output["distal/points_base_mm"][:]
        valid = output["frames/valid"][:].astype(bool)
        flags = preserved_flags.copy()
        # Recompute the two flags affected by the changed final curve while
        # retaining reconstruction/interpolation and whole-camera decisions.
        flags &= np.uint16(0xFFFF ^ (32 | 64))
        length = np.sum(
            np.linalg.norm(np.diff(points, axis=1), axis=2), axis=1)
        implausible = valid & (
            (length < config.min_learning_length_mm)
            | (length > config.max_learning_length_mm))
        flags[implausible] |= LENGTH_REJECTION_FLAG

        final_terminal = np.full(
            (len(points), len(VIEW_IDS), 2), np.nan, dtype=np.float32)
        finite_frames = np.flatnonzero(np.all(
            np.isfinite(points), axis=(1, 2)))
        for view_ordinal, view_id in enumerate(VIEW_IDS):
            rig, eye = view_id.rsplit("_", 1)
            registration = registrations[rig]
            transform = (registration.left_camera_T_base if eye == "left"
                         else registration.right_camera_T_base)
            endpoints = output[
                f"multi_view/{view_id}_axial_endpoints_px"][:]
            if len(finite_frames):
                projected = project_points(
                    registration.K, transform,
                    points[finite_frames][:, [0, -1]].reshape(-1, 3)
                )[0].reshape(len(finite_frames), 2, 2)
                distance = np.linalg.norm(
                    projected - endpoints[finite_frames], axis=2)
                distance[~np.all(np.isfinite(
                    endpoints[finite_frames]), axis=2)] = np.nan
                final_terminal[finite_frames, view_ordinal] = distance
            output[f"quality/{view_id}_final_terminal_start_px"][:] = (
                final_terminal[:, view_ordinal, 0])
            output[f"quality/{view_id}_final_terminal_end_px"][:] = (
                final_terminal[:, view_ordinal, 1])

        reference_name = (
            "multi_view/fit_reference_rig_code"
            if "multi_view/fit_reference_rig_code" in output else
            "multi_view/trusted_rig_code")
        reference_code = output[reference_name][:]
        reference_terminal = final_terminal.copy()
        reference_terminal[reference_code == 1, 2:] = np.nan
        reference_terminal[reference_code == 2, :2] = np.nan
        finite_terminal = np.any(
            np.isfinite(reference_terminal), axis=(1, 2))
        maximum_terminal = np.full(len(points), np.nan, np.float32)
        maximum_terminal[finite_terminal] = np.nanmax(
            reference_terminal[finite_terminal], axis=(1, 2))
        output["quality/final_multi_view_max_terminal_error_px"][:] = (
            maximum_terminal)
        terminal_bad = (
            valid & finite_terminal
            & (maximum_terminal > config.max_terminal_error_px))
        flags[terminal_bad] |= np.uint16(32)
        output["frames/learning_rejection_flags"][:] = flags
        output["frames/learning_valid"][:] = (flags == 0).astype(np.uint8)
        output.attrs["learning_rejection_flags_json"] = json.dumps({
            1: "reconstruction_invalid",
            2: "temporal_long_gap_unsupported",
            4: "whole_shape_temporal_outlier",
            8: "terminal_temporal_outlier",
            16: "mask_effective_width",
            32: "final_image_fit",
            64: "distal_arc_length_outside_learning_range",
            128: "whole_camera_image_fit",
            256: "abrupt_observation_jump",
        }, sort_keys=True)
        learning_valid = int(np.count_nonzero(flags == 0))
        output.flush()

    summary = {
        "source_h5": str(source),
        "output_h5": str(destination),
        "learning_valid_count": learning_valid,
        "temporal_smoothing": smoothing,
        "elapsed_s": float(time.perf_counter() - started),
    }
    (destination_dir / "resmooth_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--stage", choices=(
            "observations", "registration", "reconstruct", "resmooth",
            "overlay", "all"),
        default="all")
    parser.add_argument("--primary-observations-h5", default=None)
    parser.add_argument("--oblique-observations-h5", default=None)
    parser.add_argument(
        "--shape-h5", default=None,
        help=("existing four-view shape HDF5 for --stage "
              "registration/overlay/resmooth"))
    parser.add_argument(
        "--overlay-video", default=None,
        help="MP4 path for --stage overlay; defaults inside --outdir")
    parser.add_argument(
        "--overlay-scale", type=float, default=1.0,
        help="four-view video scale in [0.25, 1.0]; 1.0 is 1920x1080")
    parser.add_argument(
        "--window", choices=("trajectory", "run_and_return", "recording"),
        default="run_and_return")
    parser.add_argument("--start-ns", type=int, default=None)
    parser.add_argument("--end-ns", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--write-snapshots", action="store_true")
    parser.add_argument("--snapshot-count", type=int, default=16)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument(
        "--reacquisition-rig-error-px", type=float, default=9.0,
        help=("run topology-only multi-start fitting when either physical "
              "camera exceeds this symmetric mean residual"))
    parser.add_argument(
        "--whole-rig-max-symmetric-px", type=float, default=12.0,
        help=("reject direct reconstruction when an entire physical camera "
              "exceeds this unweighted symmetric mean residual"))
    parser.add_argument(
        "--reject-abrupt-observation-jumps", action="store_true",
        help=("exclude a frame from learning only when both camera rigs are "
              "temporally corrupt or no non-corrupt, well-conditioned rig "
              "can provide stereo depth; retain diagnostics and overlays"))
    parser.add_argument(
        "--no-cross-rig-registration-refinement", action="store_true",
        help=("disable the robust marker-based SE(3) refinement of the "
              "oblique registration; intended only for calibration audits"))
    parser.add_argument("--chromatic-eye-workers", type=int, default=2)
    parser.add_argument(
        "--rig-workers", type=int, default=2,
        help=("number of ZED observation passes to run concurrently; use 1 "
              "for serial debugging"))
    parser.add_argument("--prefetch-frames", type=int, default=16)
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    session = Path(args.session).resolve()
    output = Path(args.outdir).resolve()
    image_config = ImageProcessingConfig(
        chromatic_eye_workers=args.chromatic_eye_workers,
        prefetch_frames=args.prefetch_frames, store_masks=True)
    observation_paths = {
        "primary": args.primary_observations_h5,
        "oblique": args.oblique_observations_h5,
    }
    if args.stage == "registration":
        shape_h5 = Path(
            args.shape_h5 or output / "processed_shapes.h5").resolve()
        summary = validate_cross_rig_registration(
            session, shape_h5, output,
            snapshot_count=args.snapshot_count)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.stage == "overlay":
        for rig in RIGS:
            if observation_paths[rig] is None:
                observation_paths[rig] = str(
                    output / f"observations_{rig}" / "processed_shapes.h5")
        shape_h5 = Path(
            args.shape_h5 or output / "processed_shapes.h5").resolve()
        video_path = Path(
            args.overlay_video
            or output / "overlay_four_view.mp4").resolve()
        summary = write_four_view_overlay_video(
            session, shape_h5, observation_paths, video_path,
            scale=args.overlay_scale, stride=args.stride,
            maximum_frames=args.max_frames)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.stage == "resmooth":
        if args.shape_h5 is None:
            raise ValueError("resmooth stage requires --shape-h5")
        summary = resmooth_multi_camera_hdf5(
            session, args.shape_h5, output, MultiCameraConfig())
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.stage in ("observations", "all"):
        observation_paths = build_observation_caches(
            session, output, image_config, args.window,
            args.start_ns, args.end_ns, args.stride, args.max_frames,
            rig_workers=args.rig_workers)
        if args.stage == "observations":
            print(json.dumps(observation_paths, indent=2, sort_keys=True))
            return
    if any(value is None for value in observation_paths.values()):
        raise ValueError(
            "reconstruct stage requires --primary-observations-h5 and "
            "--oblique-observations-h5")
    summary = reconstruct_multi_camera_session(
        session, output, observation_paths,
        MultiCameraConfig(
            max_nfev=args.max_nfev,
            reacquisition_rig_error_px=args.reacquisition_rig_error_px,
            whole_rig_max_symmetric_px=args.whole_rig_max_symmetric_px,
            refine_cross_rig_registration=(
                not args.no_cross_rig_registration_refinement),
            reject_abrupt_observation_jumps=(
                args.reject_abrupt_observation_jumps)),
        write_snapshots=args.write_snapshots,
        snapshot_count=args.snapshot_count)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
