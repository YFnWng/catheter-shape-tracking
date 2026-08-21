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


def _long_rejected_runs(
        rejected: np.ndarray,
        valid: np.ndarray,
        timestamps_ns: np.ndarray,
        maximum_gap_ms: float) -> np.ndarray:
    """Mark rejected runs too long to replace by temporal interpolation."""
    rejected = np.asarray(rejected, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if rejected.ndim != 2 or rejected.shape[0] != len(valid):
        raise ValueError("rejected observations must have shape (frames, samples)")
    unsupported = np.zeros(len(valid), dtype=bool)
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
                unsupported[run] = True
    return unsupported


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
        frame_weight = (
            np.exp(-np.clip(reprojection, 0.0, 50.0) / 12.0)
            * np.clip(condition, 0.05, 1.0))
        frame_weight[~np.isfinite(frame_weight)] = 0.05
        frame_weight[~valid] = 0.0
        initial_weights = np.repeat(
            frame_weight[:, None], design.shape[1], axis=1)
        preliminary = _smooth_coefficients(
            observed_coefficients, timestamps, valid, initial_weights,
            cutoff_hz, huber_delta_mm, iterations, maximum_gap_ms,
            terminal_cutoff_hz, terminal_basis_count)

        outlier_prediction = _rolling_median_prediction(
            observed_coefficients, timestamps, valid, maximum_gap_ms)
        # Blend the motion-preserving median predictor with the low-pass
        # estimate.  The median supplies hard snap detection; the robust spline
        # trajectory supplies sub-frame precision for normally moving shapes.
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
            rejected_samples, valid, timestamps, maximum_gap_ms)
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
