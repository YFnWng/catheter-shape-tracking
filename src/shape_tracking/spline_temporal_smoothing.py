"""Robust offline temporal filtering of a common 3D B-spline basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import median_filter

from .geometry import cumulative_arclength
from .interface_smoothing import _valid_blocks, robust_zero_phase_smooth


LEARNING_REJECT_RECONSTRUCTION = 1 << 0
LEARNING_REJECT_TEMPORAL_UNSUPPORTED = 1 << 1
LEARNING_REJECT_SHAPE_OUTLIER = 1 << 2
LEARNING_REJECT_TERMINAL_OUTLIER = 1 << 3
LEARNING_REJECT_MASK_WIDTH = 1 << 4
LEARNING_REJECT_FINAL_IMAGE_FIT = 1 << 5


def _long_rejected_runs(
        rejected: np.ndarray,
        valid: np.ndarray,
        timestamps_ns: np.ndarray,
        maximum_gap_ms: float,
        minimum_sample_fraction: float = 0.0) -> np.ndarray:
    """Mark frames with too much long-gap material support missing.

    A local correspondence failure can persist at one material coordinate
    without making the other spline bases unobservable.  ``minimum_sample_fraction``
    therefore lets callers require a substantial portion of the curve to be in
    long rejected runs before invalidating the entire frame.  Zero preserves
    the historical any-sample behavior for direct callers.
    """
    rejected = np.asarray(rejected, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if rejected.ndim != 2 or rejected.shape[0] != len(valid):
        raise ValueError("rejected observations must have shape (frames, samples)")
    long_rejected = np.zeros_like(rejected, dtype=bool)
    for sample in range(rejected.shape[1]):
        indices = np.flatnonzero(valid & rejected[:, sample])
        if len(indices) == 0:
            continue
        splits = np.flatnonzero(
            (np.diff(indices) != 1)
            | (np.diff(timestamps_ns[indices]) * 1e-6
               > float(maximum_gap_ms))) + 1
        for run in np.split(indices, splits):
            if len(run) == 0:
                continue
            local_dt = np.diff(timestamps_ns[run]) * 1e-6
            local_dt = local_dt[np.isfinite(local_dt) & (local_dt > 0.0)]
            frame_ms = float(np.median(local_dt)) if len(local_dt) else 0.0
            duration_ms = (
                (timestamps_ns[run[-1]] - timestamps_ns[run[0]]) * 1e-6
                + frame_ms)
            if duration_ms > float(maximum_gap_ms):
                long_rejected[run, sample] = True
    fraction = float(np.clip(minimum_sample_fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return np.any(long_rejected, axis=1)
    required = max(1, int(np.ceil(fraction * rejected.shape[1])))
    return np.count_nonzero(long_rejected, axis=1) >= required


def _common_basis(sample_count: int, basis_count: int, degree: int = 3):
    bases = int(np.clip(int(basis_count), degree + 2, sample_count))
    if bases < 8:
        raise ValueError("temporal spline requires at least eight basis functions")
    internal_count = bases - degree - 1
    internal = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    knots = np.concatenate([
        np.zeros(degree + 1), internal, np.ones(degree + 1)])
    u = np.linspace(0.0, 1.0, sample_count)
    design = BSpline.design_matrix(u, knots, degree).toarray()
    return u, knots, design


def _fit_coefficients(points: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Fit the common basis while preserving each observed curve endpoint."""
    # The source has already received its spatial spline regularization.  This
    # solve changes representation only.  Clamped endpoint coefficients are
    # direct observations and are then zero-phase filtered like every other
    # coefficient.  Solving only the interior prevents a least-squares
    # representation change from pulling the near-tip segment off an otherwise
    # accurate observed endpoint.
    basis_count = design.shape[1]
    fixed = np.array([0, basis_count - 1])
    free = np.arange(1, basis_count - 1)
    coefficients = np.zeros(
        (len(points), basis_count, points.shape[2]), dtype=np.float64)
    coefficients[:, 0] = points[:, 0]
    coefficients[:, -1] = points[:, -1]
    adjusted = points - np.einsum(
        "sf,tfd->tsd", design[:, fixed], coefficients[:, fixed])
    free_design = design[:, free]
    normal = free_design.T @ free_design + 1e-9 * np.eye(len(free))
    projector = np.linalg.solve(normal, free_design.T)
    coefficients[:, free] = np.einsum("ks,tsd->tkd", projector, adjusted)
    return coefficients


def _evaluate_geometry(
        coefficients: np.ndarray,
        knots: np.ndarray,
        degree: int,
        u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    spline = BSpline(knots, coefficients, degree, axis=0)
    points = np.asarray(spline(u), dtype=np.float64)
    first = np.asarray(spline.derivative(1)(u), dtype=np.float64)
    second = np.asarray(spline.derivative(2)(u), dtype=np.float64)
    speed = np.linalg.norm(first, axis=1)
    tangent = first / np.clip(speed, 1e-9, None)[:, None]
    curvature = (
        np.linalg.norm(np.cross(first, second), axis=1)
        / np.clip(speed, 1e-9, None) ** 3)
    curvature[~np.isfinite(curvature)] = np.nan
    edge = max(2, int(np.ceil(0.03 * len(points))))
    curvature[:edge] = np.nan
    curvature[-edge:] = np.nan
    return points, tangent, curvature, float(cumulative_arclength(points)[-1])


def _short_good_islands_as_bad(
        bad: np.ndarray, timestamps_ns: np.ndarray,
        maximum_gap_ms: float, maximum_good_frames: int) -> np.ndarray:
    """Join bad intervals separated only by a tiny accepted island."""
    result = np.asarray(bad, dtype=bool).copy()
    good = np.flatnonzero(~result)
    if not len(good):
        return result
    splits = np.flatnonzero(np.diff(good) != 1) + 1
    for run in np.split(good, splits):
        if (not len(run) or len(run) > int(maximum_good_frames)
                or run[0] == 0 or run[-1] == len(result) - 1):
            continue
        if not (result[run[0] - 1] and result[run[-1] + 1]):
            continue
        duration_ms = (
            timestamps_ns[run[-1] + 1] - timestamps_ns[run[0] - 1]) * 1e-6
        if duration_ms <= float(maximum_gap_ms):
            result[run] = True
    return result


def _hermite_coefficient_bridge(
        coefficients: np.ndarray, timestamps_ns: np.ndarray,
        left: int, right: int, targets: np.ndarray) -> np.ndarray:
    """Symmetric cubic interpolation of every coefficient in one operation."""
    t0 = float(timestamps_ns[left]) * 1e-9
    t1 = float(timestamps_ns[right]) * 1e-9
    duration = max(t1 - t0, 1e-9)
    p0 = coefficients[left]
    p1 = coefficients[right]
    v0 = (p1 - p0) / duration
    v1 = v0.copy()
    previous = left - 1
    if previous >= 0 and np.all(np.isfinite(coefficients[previous])):
        dt = t0 - float(timestamps_ns[previous]) * 1e-9
        if dt > 0.0:
            v0 = (p0 - coefficients[previous]) / dt
    following = right + 1
    if following < len(coefficients) and np.all(
            np.isfinite(coefficients[following])):
        dt = float(timestamps_ns[following]) * 1e-9 - t1
        if dt > 0.0:
            v1 = (coefficients[following] - p1) / dt
    alpha = np.clip(
        (timestamps_ns[targets].astype(float) * 1e-9 - t0) / duration,
        0.0, 1.0)[:, None, None]
    h00 = 2.0 * alpha**3 - 3.0 * alpha**2 + 1.0
    h10 = alpha**3 - 2.0 * alpha**2 + alpha
    h01 = -2.0 * alpha**3 + 3.0 * alpha**2
    h11 = alpha**3 - alpha**2
    bridged = (
        h00 * p0[None] + h10 * duration * v0[None]
        + h01 * p1[None] + h11 * duration * v1[None])
    # Neighboring velocities can be noisy.  Keep the cubic within a modest
    # envelope of its two anchors; this retains smooth velocity without a
    # spatial overshoot during a short unobserved interval.
    span = np.abs(p1 - p0)
    margin = 0.25 * span + 0.5
    lower = np.minimum(p0, p1) - margin
    upper = np.maximum(p0, p1) + margin
    return np.clip(bridged, lower[None], upper[None])


def interpolate_short_spline_gaps_hdf5(
        path: str | Path, maximum_gap_ms: float = 650.0,
        maximum_good_island_frames: int = 2,
        maximum_point_step_mm: float = 8.0,
        maximum_arc_length_deviation_mm: float = 4.0) -> dict[str, object]:
    """Fill short invalid intervals in the complete 3D spline trajectory.

    The common material B-spline coefficients are bridged together, so the
    repair cannot independently reorder or kink individual sampled points.
    Original reconstruction validity is retained in a separate dataset and
    every synthesized frame is explicitly labelled.
    """
    path = Path(path)
    with h5py.File(path, "r+") as output:
        required = (
            "distal/filtered_spline_coefficients_base_mm",
            "frames/timestamp_ns", "frames/valid", "frames/learning_valid")
        missing = [name for name in required if name not in output]
        if missing:
            raise ValueError("spline gap interpolation requires: "
                             + ", ".join(missing))
        coefficients = output[
            "distal/filtered_spline_coefficients_base_mm"][:].astype(float)
        timestamps_ns = output["frames/timestamp_ns"][:].astype(np.int64)
        original_valid = output["frames/valid"][:].astype(bool)
        learning_valid = output["frames/learning_valid"][:].astype(bool)
        finite = np.all(np.isfinite(coefficients), axis=(1, 2))
        degree = int(output.attrs.get("distal_temporal_spline_degree", 3))
        basis_count = coefficients.shape[1]
        sample_count = output["distal/points_base_mm"].shape[1]
        u, knots, design = _common_basis(sample_count, basis_count, degree)
        bad = _short_good_islands_as_bad(
            ~learning_valid | ~finite, timestamps_ns, maximum_gap_ms,
            maximum_good_island_frames)
        interpolated = np.zeros(len(bad), dtype=bool)
        physical_max_step = np.full(len(bad), np.nan, dtype=np.float32)
        physical_length_deviation = np.full(
            len(bad), np.nan, dtype=np.float32)
        interpolation_method = np.zeros(len(bad), dtype=np.uint8)
        physical_rejected_runs = 0
        linear_run_count = 0
        hermite_run_count = 0
        indices = np.flatnonzero(bad)
        splits = np.flatnonzero(np.diff(indices) != 1) + 1
        for run in np.split(indices, splits):
            if not len(run):
                continue
            left = int(run[0] - 1)
            right = int(run[-1] + 1)
            if (left < 0 or right >= len(bad) or bad[left] or bad[right]
                    or not finite[left] or not finite[right]):
                continue
            duration_ms = (
                timestamps_ns[right] - timestamps_ns[left]) * 1e-6
            if duration_ms > float(maximum_gap_ms):
                continue
            hermite = _hermite_coefficient_bridge(
                coefficients, timestamps_ns, left, right, run)
            alpha = np.clip(
                ((timestamps_ns[run] - timestamps_ns[left])
                 / (timestamps_ns[right] - timestamps_ns[left])),
                0.0, 1.0)[:, None, None]
            linear = (
                (1.0 - alpha) * coefficients[left][None]
                + alpha * coefficients[right][None])

            def bridge_quality(candidate):
                block_coefficients = np.concatenate([
                    coefficients[left:left + 1], candidate,
                    coefficients[right:right + 1]], axis=0)
                block_points = np.einsum(
                    "sk,tkd->tsd", design, block_coefficients)
                step = float(np.max(np.linalg.norm(
                    np.diff(block_points, axis=0), axis=2)))
                lengths = np.array([
                    cumulative_arclength(points)[-1]
                    for points in block_points])
                expected = np.linspace(lengths[0], lengths[-1], len(lengths))
                length_error = float(np.max(np.abs(lengths - expected)))
                score = max(
                    step / max(float(maximum_point_step_mm), 1e-9),
                    length_error / max(
                        float(maximum_arc_length_deviation_mm), 1e-9))
                return score, step, length_error

            hermite_quality = bridge_quality(hermite)
            linear_quality = bridge_quality(linear)
            if linear_quality[0] < hermite_quality[0]:
                bridged = linear
                _, maximum_step, length_deviation = linear_quality
                method = 1
            else:
                bridged = hermite
                _, maximum_step, length_deviation = hermite_quality
                method = 2
            physical_max_step[run] = maximum_step
            physical_length_deviation[run] = length_deviation
            if (not np.isfinite(maximum_step)
                    or maximum_step > float(maximum_point_step_mm)
                    or not np.isfinite(length_deviation)
                    or length_deviation
                    > float(maximum_arc_length_deviation_mm)):
                physical_rejected_runs += 1
                continue
            coefficients[run] = bridged
            interpolated[run] = True
            interpolation_method[run] = method
            linear_run_count += int(method == 1)
            hermite_run_count += int(method == 2)

        def dataset(name: str, dtype=np.uint8):
            if name in output:
                return output[name]
            return output.create_dataset(
                name, shape=(len(bad),), dtype=dtype, fillvalue=0,
                compression="gzip", compression_opts=1, shuffle=True)

        pre_valid = dataset("frames/pre_interpolation_valid")
        # This source validity is immutable across repeated downstream resumes.
        if not np.any(pre_valid[:]):
            pre_valid[:] = original_valid.astype(np.uint8)
        interpolated_ds = dataset("frames/curve_temporally_interpolated")
        interpolated_ds[:] = interpolated.astype(np.uint8)
        step_ds = dataset(
            "quality/curve_interpolation_max_point_step_mm", dtype=np.float32)
        length_ds = dataset(
            "quality/curve_interpolation_arc_length_deviation_mm",
            dtype=np.float32)
        method_ds = dataset(
            "quality/curve_interpolation_method", dtype=np.uint8)
        step_ds[:] = physical_max_step
        length_ds[:] = physical_length_deviation
        method_ds[:] = interpolation_method
        output["distal/filtered_spline_coefficients_base_mm"][:] = coefficients

        for index in np.flatnonzero(interpolated):
            points, tangent, curvature, arc_length = _evaluate_geometry(
                coefficients[index], knots, degree, u)
            output["distal/points_base_mm"][index] = points
            output["distal/s_mm"][index] = cumulative_arclength(points)
            output["distal/tangent_base"][index] = tangent
            output["distal/curvature_per_mm"][index] = curvature
            output["distal/base_position_base_mm"][index] = points[0]
            output["quality/distal_spline_arc_length_mm"][index] = arc_length
            output["quality/distal_spline_basis_count"][index] = basis_count
            output["quality/distal_spline_internal_knot_count"][index] = (
                basis_count - degree - 1)

        # These frames now have a usable final curve.  Preserve the raw state
        # above and clear prior reasons. Final image/mask disagreement remains
        # diagnostic; a bridge that passed the physical gates is accepted by
        # the final quality stage because V-overlap masks are unreliable.
        valid = original_valid | interpolated
        output["frames/valid"][:] = valid.astype(np.uint8)
        flags = output["frames/learning_rejection_flags"][:].astype(np.uint16)
        flags[interpolated] = 0
        output["frames/learning_rejection_flags"][:] = flags
        output["frames/learning_valid"][:] = (flags == 0).astype(np.uint8)
        output.attrs["distal_short_gap_interpolation"] = (
            "adaptive_symmetric_linear_or_cubic_common_basis_coefficients")
        output.attrs["curve_interpolation_method_json"] = (
            '{"0":"not_interpolated","1":"linear","2":"cubic_hermite"}')
        output.attrs["distal_short_gap_interpolation_maximum_ms"] = float(
            maximum_gap_ms)
        output.attrs["distal_short_gap_maximum_point_step_mm"] = float(
            maximum_point_step_mm)
        output.attrs["distal_short_gap_maximum_arc_length_deviation_mm"] = (
            float(maximum_arc_length_deviation_mm))
        output.flush()
        runs = []
        repaired = np.flatnonzero(interpolated)
        repaired_splits = np.flatnonzero(np.diff(repaired) != 1) + 1
        for run in np.split(repaired, repaired_splits):
            if len(run):
                runs.append([int(run[0]), int(run[-1])])
        return {
            "interpolated_frame_count": int(np.count_nonzero(interpolated)),
            "interpolated_runs": runs,
            "maximum_gap_ms": float(maximum_gap_ms),
            "physical_rejected_run_count": int(physical_rejected_runs),
            "linear_run_count": int(linear_run_count),
            "cubic_hermite_run_count": int(hermite_run_count),
        }


def _smooth_coefficients(
        observed: np.ndarray,
        timestamps_ns: np.ndarray,
        valid: np.ndarray,
        weights: np.ndarray,
        cutoff_hz: float,
        huber_delta_mm: float,
        iterations: int,
        maximum_gap_ms: float,
        terminal_cutoff_hz: float | None = None,
        terminal_basis_count: int = 4) -> np.ndarray:
    smoothed = np.full_like(observed, np.nan)
    basis_cutoffs = np.full(observed.shape[1], float(cutoff_hz))
    terminal_count = int(np.clip(
        terminal_basis_count, 0, observed.shape[1]))
    if (terminal_count > 0 and terminal_cutoff_hz is not None
            and float(terminal_cutoff_hz) > float(cutoff_hz)):
        basis_cutoffs[-terminal_count:] = np.linspace(
            float(cutoff_hz), float(terminal_cutoff_hz), terminal_count)
    for block in _valid_blocks(valid, timestamps_ns, maximum_gap_ms):
        for basis in range(observed.shape[1]):
            for coordinate in range(3):
                smoothed[block, basis, coordinate] = robust_zero_phase_smooth(
                    observed[block, basis, coordinate], timestamps_ns[block],
                    weights[block, basis], cutoff_hz=basis_cutoffs[basis],
                    huber_delta=huber_delta_mm, iterations=iterations)
    return smoothed


def _rolling_median_prediction(
        observed: np.ndarray,
        timestamps_ns: np.ndarray,
        valid: np.ndarray,
        maximum_gap_ms: float) -> np.ndarray:
    """Predict coefficients without following short snap-and-return runs."""
    predicted = np.full_like(observed, np.nan)
    for block in _valid_blocks(valid, timestamps_ns, maximum_gap_ms):
        if len(block) < 3:
            predicted[block] = observed[block]
            continue
        dt_ms = np.diff(timestamps_ns[block]) * 1e-6
        dt_ms = dt_ms[np.isfinite(dt_ms) & (dt_ms > 0.0)]
        if len(dt_ms) == 0:
            predicted[block] = observed[block]
            continue
        # The window spans one maximum repair interval on either side.  A bad
        # run shorter than that interval cannot become the window majority,
        # while a smooth monotonic trajectory retains its central value.
        half_width = max(1, int(round(
            float(maximum_gap_ms) / float(np.median(dt_ms)))))
        size = min(2 * half_width + 1, len(block) | 1)
        if size % 2 == 0:
            size -= 1
        size = max(size, 3)
        predicted[block] = median_filter(
            observed[block], size=(size, 1, 1), mode="nearest")
    return predicted


def _local_motion_prediction(
        observed: np.ndarray,
        timestamps_ns: np.ndarray,
        valid: np.ndarray,
        maximum_gap_ms: float) -> np.ndarray:
    """Predict each observation from its immediate constant-velocity chord.

    Unlike a wide rolling position median, this residual is insensitive to
    sustained physical motion. An instantaneous snap or snap-return still has
    a large local second difference.
    """
    predicted = np.full_like(observed, np.nan)
    for block in _valid_blocks(valid, timestamps_ns, maximum_gap_ms):
        predicted[block] = observed[block]
        if len(block) < 3:
            continue
        previous = block[:-2]
        current = block[1:-1]
        following = block[2:]
        duration = (
            timestamps_ns[following] - timestamps_ns[previous]).astype(float)
        alpha = np.divide(
            timestamps_ns[current] - timestamps_ns[previous], duration,
            out=np.full(len(current), 0.5), where=duration > 0.0)
        predicted[current] = (
            (1.0 - alpha)[:, None, None] * observed[previous]
            + alpha[:, None, None] * observed[following])
    return predicted


def _bridge_short_point_outlier_runs(
        outlier: np.ndarray,
        valid: np.ndarray,
        timestamps_ns: np.ndarray,
        maximum_gap_ms: float) -> np.ndarray:
    """Fill isolated paired snap/return edges without chain bridging.

    Reusing every edge as both the end of one bridge and the start of another
    can turn alternating reconstruction modes into a many-second unsupported
    block. Each consecutive edge cluster therefore participates in at most one
    short bridge.
    """
    bridged = np.asarray(outlier, dtype=bool).copy()
    bridge_limit_ms = min(float(maximum_gap_ms), 200.0)
    for sample in range(bridged.shape[1]):
        indices = np.flatnonzero(valid & bridged[:, sample])
        if len(indices) < 2:
            continue
        splits = np.flatnonzero(np.diff(indices) > 1) + 1
        clusters = [run for run in np.split(indices, splits) if len(run)]
        cursor = 0
        while cursor + 1 < len(clusters):
            first = int(clusters[cursor][-1])
            second = int(clusters[cursor + 1][0])
            duration_ms = (
                timestamps_ns[second] - timestamps_ns[first]) * 1e-6
            if second > first + 1 and duration_ms <= bridge_limit_ms:
                bridged[first:second + 1, sample] = True
                cursor += 2
            else:
                cursor += 1
    return bridged


def _confirmed_support_runs(
        supported: np.ndarray,
        valid: np.ndarray,
        timestamps_ns: np.ndarray,
        maximum_gap_ms: float,
        minimum_frames: int = 3) -> np.ndarray:
    """Require persistent image evidence before temporal reacquisition."""
    confirmed = np.zeros(len(supported), dtype=bool)
    candidates = np.flatnonzero(np.asarray(supported, bool) & valid)
    if not len(candidates):
        return confirmed
    splits = np.flatnonzero(
        (np.diff(candidates) != 1)
        | (np.diff(timestamps_ns[candidates]) * 1e-6
           > float(maximum_gap_ms))) + 1
    for run in np.split(candidates, splits):
        if len(run) >= max(1, int(minimum_frames)):
            confirmed[run] = True
    return confirmed


def _trusted_joint_image_support(
        output, valid: np.ndarray, timestamps_ns: np.ndarray,
        maximum_gap_ms: float, minimum_frames: int = 3) -> np.ndarray:
    """Find persistent per-frame joint fits that justify reacquisition."""
    names = (
        "joint_left_model_mean_px", "joint_right_model_mean_px",
        "joint_left_model_p95_px", "joint_right_model_p95_px",
        "joint_left_coverage_mean_px", "joint_right_coverage_mean_px",
        "joint_optimizer_success")
    if any(f"quality/{name}" not in output for name in names):
        return np.zeros(len(valid), dtype=bool)
    values = {
        name: output[f"quality/{name}"][:].astype(float) for name in names}
    supported = (
        valid
        & (values["joint_optimizer_success"] > 0.5)
        & (values["joint_left_model_mean_px"] <= 3.0)
        & (values["joint_right_model_mean_px"] <= 3.0)
        & (values["joint_left_model_p95_px"] <= 8.0)
        & (values["joint_right_model_p95_px"] <= 8.0)
        & (values["joint_left_coverage_mean_px"] <= 5.0)
        & (values["joint_right_coverage_mean_px"] <= 5.0))
    return _confirmed_support_runs(
        supported, valid, timestamps_ns, maximum_gap_ms, minimum_frames)


def smooth_distal_spline_coefficients_hdf5(
        path: str | Path,
        cutoff_hz: float = 3.0,
        huber_delta_mm: float = 1.0,
        iterations: int = 4,
        maximum_gap_ms: float = 500.0,
        basis_count: int = 20,
        terminal_cutoff_hz: float = 5.0,
        terminal_basis_count: int = 4,
        outlier_sigma: float = 4.5,
        outlier_floor_mm: float = 0.75,
        frame_outlier_fraction: float = 0.5,
        terminal_outlier_sample_count: int = 4,
        max_learning_mask_width_px: float = 20.0,
        observation_blend: float = 0.35,
        terminal_observation_blend: float = 0.65,
        reject_recovered_outliers: bool = True) -> dict[str, float]:
    """Filter distal shape trajectories in a fixed material spline basis.

    A preliminary robust coefficient trajectory supplies a leave-no-phase
    temporal prediction.  Point observations inconsistent with that trajectory
    are downweighted locally, so a false tip does not invalidate the reliable
    proximal part of the same frame.  A second robust solve produces the final
    coefficients, from which position, tangent, and curvature are evaluated
    directly without a temporally independent refit.
    """
    path = Path(path)
    with h5py.File(path, "r+") as output:
        valid = output["frames/valid"][:].astype(bool)
        timestamps = output["frames/timestamp_ns"][:]
        # Reuse the pre-temporal representation when refining an existing HDF5.
        # This makes the operation idempotent instead of smoothing an already
        # filtered curve for a second time.
        source_name = (
            "distal/pre_temporal_points_base_mm"
            if "distal/pre_temporal_points_base_mm" in output
            else "distal/points_base_mm")
        observed_points = output[source_name][:].astype(float)
        finite = np.all(np.isfinite(observed_points), axis=(1, 2))
        valid &= finite
        count, sample_count, _ = observed_points.shape
        degree = 3
        u, knots, design = _common_basis(
            sample_count, basis_count, degree=degree)

        observed_coefficients = np.full(
            (count, design.shape[1], 3), np.nan, dtype=np.float64)
        observed_coefficients[valid] = _fit_coefficients(
            observed_points[valid], design)

        reprojection = output["quality/reprojection_p95_px"][:].astype(float)
        # For joint two-view reconstruction the disparity curve is only an
        # initializer. Weight the observed coefficient trajectory using the
        # actual per-frame joint fit whenever that diagnostic is available.
        if "quality/fitted_reprojection_p95_px" in output:
            fitted_reprojection = output[
                "quality/fitted_reprojection_p95_px"][:].astype(float)
            use_fitted = np.isfinite(fitted_reprojection)
            reprojection[use_fitted] = fitted_reprojection[use_fitted]
        condition = output["quality/stereo_condition"][:].astype(float)
        condition_weight = np.clip(condition, 0.05, 1.0)
        # The classical stereo condition number describes the independent
        # two-ray triangulation.  It remains low in exactly the overlap case
        # for which the joint spline uses good-eye ordering, mask support and
        # temporal material correspondence.  Once that joint fit has a finite
        # image-space score, use its fitted reprojection (above) as the quality
        # weight instead of penalizing the same frame a second time for ray
        # geometry.  Otherwise the offline smoother can average a valid arm
        # back through the empty middle of a tight V.
        if "quality/joint_final_symmetric_mean_px" in output:
            joint_score = output[
                "quality/joint_final_symmetric_mean_px"][:].astype(float)
            condition_weight[np.isfinite(joint_score)] = 1.0
        frame_weight = (
            np.exp(-np.clip(reprojection, 0.0, 50.0) / 12.0)
            * condition_weight)
        frame_weight[~np.isfinite(frame_weight)] = 0.05
        frame_weight[~valid] = 0.0
        initial_weights = np.repeat(
            frame_weight[:, None], design.shape[1], axis=1)
        preliminary = _smooth_coefficients(
            observed_coefficients, timestamps, valid, initial_weights,
            cutoff_hz, huber_delta_mm, iterations, maximum_gap_ms,
            terminal_cutoff_hz, terminal_basis_count)

        outlier_prediction = _local_motion_prediction(
            observed_coefficients, timestamps, valid, maximum_gap_ms)
        # The local chord supplies motion-invariant snap detection. The robust
        # spline trajectory remains a fallback only at missing predictions.
        prediction_coefficients = np.where(
            np.isfinite(outlier_prediction), outlier_prediction, preliminary)
        predicted_points = np.full_like(observed_points, np.nan)
        predicted_points[valid] = np.einsum(
            "sk,tkd->tsd", design, prediction_coefficients[valid])
        innovation = np.linalg.norm(
            observed_points - predicted_points, axis=2)

        point_threshold = np.full(sample_count, float(outlier_floor_mm))
        for sample in range(sample_count):
            values = innovation[valid, sample]
            values = values[np.isfinite(values)]
            if len(values) < 5:
                continue
            center = float(np.median(values))
            scale = 1.4826 * float(np.median(np.abs(values - center)))
            point_threshold[sample] = max(
                float(outlier_floor_mm),
                center + float(outlier_sigma) * max(scale, 1e-6))
        point_outlier = (
            valid[:, None] & np.isfinite(innovation)
            & (innovation > point_threshold[None, :]))
        point_outlier = _bridge_short_point_outlier_runs(
            point_outlier, valid, timestamps, maximum_gap_ms)
        trusted_image_support = _trusted_joint_image_support(
            output, valid, timestamps, maximum_gap_ms, minimum_frames=3)
        reacquired = trusted_image_support & np.any(point_outlier, axis=1)
        # Persistent, independently image-supported joint fits define a new
        # temporal block after an ill interval. Do not let an earlier rejected
        # trajectory veto that reacquisition.
        point_outlier[reacquired] = False
        outlier_fraction = np.mean(point_outlier, axis=1)
        frame_outlier = valid & (
            outlier_fraction >= float(frame_outlier_fraction))

        # Map point confidence to each local B-spline coefficient using squared
        # basis influence.  Thus a bad distal suffix primarily suppresses the
        # distal coefficients rather than discarding the whole observation.
        influence = design**2
        influence /= np.clip(np.sum(influence, axis=0, keepdims=True), 1e-12, None)
        point_confidence = (~point_outlier).astype(np.float64)
        coefficient_confidence = point_confidence @ influence
        coefficient_confidence[frame_outlier] = 0.0
        final_weights = frame_weight[:, None] * coefficient_confidence
        terminal_count = int(np.clip(
            terminal_outlier_sample_count, 1, sample_count))
        terminal_outlier = np.any(point_outlier[:, -terminal_count:], axis=1)
        rejected_samples = point_outlier | frame_outlier[:, None]
        temporal_unsupported = _long_rejected_runs(
            rejected_samples, valid, timestamps, maximum_gap_ms,
            minimum_sample_fraction=frame_outlier_fraction)
        final_valid = valid & ~temporal_unsupported
        final_coefficients = _smooth_coefficients(
            observed_coefficients, timestamps, final_valid, final_weights,
            cutoff_hz, huber_delta_mm, iterations, maximum_gap_ms,
            terminal_cutoff_hz, terminal_basis_count)
        # Robust low-pass filtering removes coefficient jitter, but it can also
        # pull a genuinely observed curve away from both image centerlines. Add
        # a bounded measurement update after the zero-phase solve. Local
        # coefficient confidence makes this update vanish for snap outliers;
        # well-supported terminal coefficients receive more authority because
        # small near-tip errors are especially visible after projection.
        measurement_gain = (
            np.clip(float(observation_blend), 0.0, 1.0)
            * frame_weight[:, None] * coefficient_confidence)
        terminal_basis = int(np.clip(
            terminal_basis_count, 0, design.shape[1]))
        if terminal_basis:
            measurement_gain[:, -terminal_basis:] = (
                np.clip(float(terminal_observation_blend), 0.0, 1.0)
                * frame_weight[:, None]
                * coefficient_confidence[:, -terminal_basis:])
        measurement_gain = np.clip(measurement_gain, 0.0, 1.0)
        measurement_supported = (
            final_valid[:, None, None]
            & np.isfinite(observed_coefficients)
            & np.isfinite(final_coefficients))
        corrected = final_coefficients + measurement_gain[:, :, None] * (
            observed_coefficients - final_coefficients)
        final_coefficients[measurement_supported] = corrected[
            measurement_supported]

        def dataset(name: str, shape, dtype=np.float32, fillvalue=np.nan):
            if name in output:
                return output[name]
            return output.create_dataset(
                name, shape=shape, dtype=dtype, fillvalue=fillvalue,
                compression="gzip", compression_opts=1, shuffle=True)

        pre_points_ds = dataset(
            "distal/pre_temporal_points_base_mm", observed_points.shape)
        observed_coeff_ds = dataset(
            "distal/observed_spline_coefficients_base_mm",
            observed_coefficients.shape)
        filtered_coeff_ds = dataset(
            "distal/filtered_spline_coefficients_base_mm",
            final_coefficients.shape)
        outlier_mask_ds = dataset(
            "distal/temporal_outlier_mask", (count, sample_count),
            dtype=np.uint8, fillvalue=0)
        innovation_ds = dataset(
            "quality/shape_coefficient_innovation_rms_mm", (count,))
        adjustment_ds = dataset(
            "quality/shape_temporal_adjustment_rms_mm", (count,))
        outlier_fraction_ds = dataset(
            "quality/shape_temporal_outlier_fraction", (count,))
        frame_outlier_ds = dataset(
            "quality/shape_temporal_frame_outlier", (count,),
            dtype=np.uint8, fillvalue=0)
        supported_ds = dataset(
            "quality/shape_temporal_supported", (count,),
            dtype=np.uint8, fillvalue=0)
        terminal_outlier_ds = dataset(
            "quality/shape_temporal_terminal_outlier", (count,),
            dtype=np.uint8, fillvalue=0)
        unsupported_ds = dataset(
            "quality/shape_temporal_long_gap_unsupported", (count,),
            dtype=np.uint8, fillvalue=0)
        reacquired_ds = dataset(
            "quality/shape_temporal_reacquired", (count,),
            dtype=np.uint8, fillvalue=0)
        mask_width_rejected_ds = dataset(
            "quality/mask_width_learning_rejected", (count,),
            dtype=np.uint8, fillvalue=0)
        learning_valid_ds = dataset(
            "frames/learning_valid", (count,), dtype=np.uint8, fillvalue=0)
        rejection_flags_ds = dataset(
            "frames/learning_rejection_flags", (count,),
            dtype=np.uint16, fillvalue=0)

        pre_points_ds[:] = observed_points
        observed_coeff_ds[:] = observed_coefficients
        filtered_coeff_ds[:] = final_coefficients
        outlier_mask_ds[:] = point_outlier.astype(np.uint8)
        outlier_fraction_ds[:] = outlier_fraction
        frame_outlier_ds[:] = frame_outlier.astype(np.uint8)
        terminal_outlier_ds[:] = terminal_outlier.astype(np.uint8)
        unsupported_ds[:] = temporal_unsupported.astype(np.uint8)
        reacquired_ds[:] = reacquired.astype(np.uint8)
        mask_width_rejected = np.zeros(count, dtype=bool)
        if ("quality/mask_effective_width_left_px" in output
                and "quality/mask_effective_width_right_px" in output):
            maximum_width = np.maximum(
                output["quality/mask_effective_width_left_px"][:],
                output["quality/mask_effective_width_right_px"][:])
            mask_width_rejected = (
                valid & np.isfinite(maximum_width)
                & (maximum_width > float(max_learning_mask_width_px)))
        mask_width_rejected_ds[:] = mask_width_rejected.astype(np.uint8)
        supported = final_valid & np.all(
            np.isfinite(final_coefficients), axis=(1, 2))
        supported_ds[:] = supported.astype(np.uint8)
        rejection_flags = np.zeros(count, dtype=np.uint16)
        rejection_flags[~valid] |= LEARNING_REJECT_RECONSTRUCTION
        rejection_flags[temporal_unsupported] |= (
            LEARNING_REJECT_TEMPORAL_UNSUPPORTED)
        # A direct joint two-view fit may move substantially from its disparity
        # initializer and therefore look like an observation outlier.  If the
        # zero-phase coefficient solve recovers it and the final two-view
        # geometry check passes, that innovation is diagnostic rather than a
        # reason to discard an otherwise image-supported learning sample.
        if reject_recovered_outliers:
            rejection_flags[frame_outlier] |= LEARNING_REJECT_SHAPE_OUTLIER
            rejection_flags[terminal_outlier] |= LEARNING_REJECT_TERMINAL_OUTLIER
        rejection_flags[mask_width_rejected] |= LEARNING_REJECT_MASK_WIDTH
        rejection_flags_ds[:] = rejection_flags
        learning_valid_ds[:] = (rejection_flags == 0).astype(np.uint8)

        innovations = np.sqrt(np.mean(
            (observed_coefficients - preliminary) ** 2, axis=(1, 2)))
        innovation_ds[:] = innovations
        adjustments = np.full(count, np.nan)
        updated = 0
        for index in np.flatnonzero(supported):
            points, tangent, curvature, arc_length = _evaluate_geometry(
                final_coefficients[index], knots, degree, u)
            adjustments[index] = float(np.sqrt(np.mean(np.sum(
                (points - observed_points[index]) ** 2, axis=1))))
            output["distal/points_base_mm"][index] = points
            output["distal/s_mm"][index] = cumulative_arclength(points)
            output["distal/tangent_base"][index] = tangent
            output["distal/curvature_per_mm"][index] = curvature
            output["distal/base_position_base_mm"][index] = points[0]
            output["quality/distal_spline_arc_length_mm"][index] = arc_length
            output["quality/distal_spline_basis_count"][index] = design.shape[1]
            output["quality/distal_spline_internal_knot_count"][index] = (
                design.shape[1] - degree - 1)
            output["quality/distal_spline_rms_residual_mm"][index] = (
                adjustments[index])
            updated += 1
        adjustment_ds[:] = adjustments

        output.attrs["distal_final_representation"] = (
            "robust_zero_phase_common_basis_spline_coefficients")
        output.attrs["distal_temporal_spline_degree"] = degree
        output.attrs["distal_temporal_spline_basis_count"] = design.shape[1]
        output.attrs["distal_temporal_cutoff_hz"] = float(cutoff_hz)
        output.attrs["distal_temporal_terminal_cutoff_hz"] = float(
            terminal_cutoff_hz)
        output.attrs["distal_temporal_terminal_basis_count"] = int(
            terminal_basis_count)
        output.attrs["distal_temporal_observation_blend"] = float(
            observation_blend)
        output.attrs["distal_temporal_terminal_observation_blend"] = float(
            terminal_observation_blend)
        output.attrs["distal_temporal_huber_delta_mm"] = float(huber_delta_mm)
        output.attrs["distal_temporal_outlier_sigma"] = float(outlier_sigma)
        output.attrs["distal_temporal_outlier_floor_mm"] = float(
            outlier_floor_mm)
        output.attrs["learning_max_mask_effective_width_px"] = float(
            max_learning_mask_width_px)
        output.attrs["learning_rejection_flags_json"] = (
            '{"1":"reconstruction_invalid","2":"temporal_long_gap_unsupported",'
            '"4":"whole_shape_temporal_outlier","8":"terminal_temporal_outlier",'
            '"16":"mask_effective_width"}')
        output.flush()

        finite_adjustment = adjustments[np.isfinite(adjustments)]
        finite_innovation = innovations[np.isfinite(innovations)]
        return {
            "updated_frames": int(updated),
            "frame_outlier_count": int(np.count_nonzero(frame_outlier)),
            "terminal_outlier_count": int(np.count_nonzero(
                terminal_outlier & valid)),
            "long_gap_unsupported_count": int(np.count_nonzero(
                temporal_unsupported)),
            "reacquired_frame_count": int(np.count_nonzero(reacquired)),
            "mask_width_rejected_count": int(np.count_nonzero(
                mask_width_rejected)),
            "learning_valid_count": int(np.count_nonzero(
                rejection_flags == 0)),
            "point_outlier_fraction": float(np.mean(point_outlier[valid]))
            if np.any(valid) else float("nan"),
            "coefficient_innovation_median_mm": float(np.median(
                finite_innovation)) if len(finite_innovation) else float("nan"),
            "temporal_adjustment_median_mm": float(np.median(
                finite_adjustment)) if len(finite_adjustment) else float("nan"),
            "temporal_adjustment_p95_mm": float(np.percentile(
                finite_adjustment, 95)) if len(finite_adjustment) else float("nan"),
        }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--cutoff-hz", type=float, default=3.0)
    parser.add_argument("--terminal-cutoff-hz", type=float, default=5.0)
    parser.add_argument("--terminal-basis-count", type=int, default=4)
    parser.add_argument("--huber-delta-mm", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--maximum-gap-ms", type=float, default=500.0)
    parser.add_argument("--basis-count", type=int, default=20)
    parser.add_argument("--outlier-sigma", type=float, default=4.5)
    parser.add_argument("--outlier-floor-mm", type=float, default=0.75)
    parser.add_argument("--frame-outlier-fraction", type=float, default=0.5)
    parser.add_argument("--terminal-outlier-samples", type=int, default=4)
    parser.add_argument("--max-learning-mask-width-px", type=float, default=20.0)
    parser.add_argument("--observation-blend", type=float, default=0.35)
    parser.add_argument(
        "--terminal-observation-blend", type=float, default=0.65)
    args = parser.parse_args(argv)
    summary = smooth_distal_spline_coefficients_hdf5(
        args.h5, cutoff_hz=args.cutoff_hz,
        terminal_cutoff_hz=args.terminal_cutoff_hz,
        terminal_basis_count=args.terminal_basis_count,
        huber_delta_mm=args.huber_delta_mm, iterations=args.iterations,
        maximum_gap_ms=args.maximum_gap_ms, basis_count=args.basis_count,
        outlier_sigma=args.outlier_sigma,
        outlier_floor_mm=args.outlier_floor_mm,
        frame_outlier_fraction=args.frame_outlier_fraction,
        terminal_outlier_sample_count=args.terminal_outlier_samples,
        max_learning_mask_width_px=args.max_learning_mask_width_px,
        observation_blend=args.observation_blend,
        terminal_observation_blend=args.terminal_observation_blend)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
