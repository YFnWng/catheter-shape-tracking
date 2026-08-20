"""Offline, zero-phase smoothing of the catheter material interface."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .geometry import (
    OBSERVED_DISTAL,
    OBSERVED_PROXIMAL,
    cumulative_arclength,
    curve_geometry,
    resample_polyline,
)


def robust_zero_phase_smooth(
        values: np.ndarray,
        timestamps_ns: np.ndarray,
        weights: np.ndarray | None = None,
        cutoff_hz: float = 2.0,
        huber_delta: float = 2.0,
        iterations: int = 4) -> np.ndarray:
    """Smooth scalar samples with a symmetric robust second-difference fit.

    The batch solve is non-causal and therefore introduces no phase lag.  Its
    second-difference penalty is scaled to an approximate -3 dB cutoff for the
    median sample rate; confidence weights and Huber IRLS suppress isolated
    material-boundary or reconstruction errors.
    """
    observed = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps_ns, dtype=np.float64)
    if observed.ndim != 1 or timestamps.shape != observed.shape:
        raise ValueError("values and timestamps must be equal-length vectors")
    count = len(observed)
    if count < 5 or cutoff_hz <= 0.0:
        return observed.copy()
    base_weight = (
        np.ones(count, dtype=np.float64) if weights is None
        else np.asarray(weights, dtype=np.float64).copy())
    if base_weight.shape != observed.shape:
        raise ValueError("weights must match values")
    finite = np.isfinite(observed) & np.isfinite(base_weight) & (base_weight > 0.0)
    if np.count_nonzero(finite) < 5:
        return observed.copy()
    index = np.arange(count)
    filled = np.interp(index, index[finite], observed[finite])
    dt = np.diff(timestamps[finite]) * 1e-9
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        return observed.copy()
    sample_rate_hz = 1.0 / float(np.median(dt))
    normalized = min(max(float(cutoff_hz) / sample_rate_hz, 1e-4), 0.49)
    response = 2.0 * np.sin(np.pi * normalized)
    smoothing_lambda = (np.sqrt(2.0) - 1.0) / max(response**4, 1e-12)
    second = diags(
        [np.ones(count - 2), -2.0 * np.ones(count - 2),
         np.ones(count - 2)], [0, 1, 2], shape=(count - 2, count),
        format="csr")
    penalty = smoothing_lambda * (second.T @ second)
    median_weight = float(np.median(base_weight[finite]))
    base_weight = np.where(
        finite, np.clip(base_weight / max(median_weight, 1e-12), 0.03, 20.0),
        0.0)
    estimate = filled.copy()
    delta = max(float(huber_delta), 1e-6)
    for _ in range(max(1, int(iterations))):
        residual = estimate - filled
        robust = np.ones(count, dtype=np.float64)
        large = np.abs(residual) > delta
        robust[large] = delta / np.maximum(np.abs(residual[large]), 1e-12)
        effective = base_weight * robust
        normal = diags(effective, format="csr") + penalty
        estimate = np.asarray(spsolve(
            normal + 1e-9 * diags(np.ones(count)), effective * filled))
    return estimate


def _valid_blocks(
        valid: np.ndarray,
        timestamps_ns: np.ndarray,
        maximum_gap_ms: float,
        split_on_missing: bool = True) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return []
    split_condition = (
        np.diff(timestamps_ns[indices]) * 1e-6 > maximum_gap_ms)
    if split_on_missing:
        split_condition |= np.diff(indices) != 1
    split = np.flatnonzero(split_condition) + 1
    return [block for block in np.split(indices, split) if len(block)]


def _suffix_from_tip_length(points: np.ndarray, length_mm: float) -> np.ndarray:
    curve = np.asarray(points, dtype=np.float64)
    s = cumulative_arclength(curve)
    boundary_s = float(np.clip(s[-1] - float(length_mm), 0.0, s[-1]))
    right = int(np.clip(np.searchsorted(s, boundary_s, side="right"),
                        1, len(curve) - 1))
    left = right - 1
    alpha = (boundary_s - s[left]) / max(s[right] - s[left], 1e-12)
    boundary = (1.0 - alpha) * curve[left] + alpha * curve[right]
    return np.vstack([boundary, curve[right:]])


def smooth_interface_hdf5(
        path: str | Path,
        cutoff_hz: float = 2.0,
        huber_delta_mm: float = 2.0,
        iterations: int = 4,
        maximum_gap_ms: float = 100.0,
        curvature_smoothing_mm: float = 0.25,
        curvature_spline_bases: int = 20,
        nominal_distal_length_mm: float = 60.0,
        length_gate_mm: float = 4.0,
        session_prior_relative_weight: float = 0.5) -> dict[str, float]:
    """Apply final interface smoothing and rebuild distal geometry in-place."""
    path = Path(path)
    with h5py.File(path, "r+") as output:
        valid = output["frames/valid"][:].astype(bool)
        timestamps = output["frames/timestamp_ns"][:]
        causal_length = output["quality/distal_length_filtered_mm"][:].astype(float)
        uncertainty = output["quality/interface_uncertainty_mm"][:].astype(float)
        confidence = output["quality/material_boundary_confidence"][:].astype(float)
        stereo_consistent = output[
            "quality/material_boundary_stereo_consistent"][:].astype(bool)
        reprojection = output["quality/reprojection_p95_px"][:].astype(float)
        weights = (
            np.clip(confidence, 0.05, None)
            / np.clip(uncertainty, 0.25, None) ** 2
            * np.where(stereo_consistent, 1.0, 0.35)
            * np.exp(-np.clip(reprojection, 0.0, 50.0) / 12.0))
        length_gate_rejected = (
            np.abs(causal_length - float(nominal_distal_length_mm))
            > max(float(length_gate_mm), 0.0))
        weights[~valid | ~np.isfinite(causal_length)] = 0.0
        weights[length_gate_rejected] = 0.0
        trusted = valid & np.isfinite(causal_length) & (weights > 0.0)
        if np.any(trusted):
            order = np.argsort(causal_length[trusted])
            ordered_values = causal_length[trusted][order]
            ordered_weights = weights[trusted][order]
            cumulative = np.cumsum(ordered_weights)
            session_length_mm = float(ordered_values[np.searchsorted(
                cumulative, 0.5 * cumulative[-1])])
        else:
            session_length_mm = float(nominal_distal_length_mm)
        smoothing_observation = causal_length.copy()
        smoothing_observation[length_gate_rejected] = session_length_mm
        trusted_weight_scale = (
            float(np.median(weights[trusted])) if np.any(trusted) else 1.0)
        weights[length_gate_rejected & valid] = (
            max(float(session_prior_relative_weight), 0.0)
            * trusted_weight_scale)
        smoothed = np.full_like(causal_length, np.nan)
        for block in _valid_blocks(valid, timestamps, maximum_gap_ms):
            smoothed[block] = robust_zero_phase_smooth(
                smoothing_observation[block], timestamps[block], weights[block],
                cutoff_hz=cutoff_hz, huber_delta=huber_delta_mm,
                iterations=iterations)

        def dataset(name: str, shape, dtype=np.float32, fillvalue=np.nan):
            if name in output:
                return output[name]
            return output.create_dataset(
                name, shape=shape, dtype=dtype, fillvalue=fillvalue,
                compression="gzip", compression_opts=1, shuffle=True)

        count = len(valid)
        causal_base = dataset(
            "distal/causal_base_position_base_mm", (count, 3))
        causal_base[:] = output["distal/base_position_base_mm"][:]
        smooth_length_ds = dataset(
            "quality/distal_length_smoothed_mm", (count,))
        smooth_s_ds = dataset("quality/interface_smoothed_s_mm", (count,))
        adjustment_ds = dataset(
            "quality/interface_smoothing_adjustment_mm", (count,))
        weight_ds = dataset("quality/interface_smoothing_weight", (count,))
        rejected_ds = dataset(
            "quality/interface_length_gate_rejected", (count,),
            dtype=np.uint8, fillvalue=0)
        smooth_length_ds[:] = smoothed
        adjustment_ds[:] = smoothed - causal_length
        weight_ds[:] = weights
        rejected_ds[:] = length_gate_rejected.astype(np.uint8)

        source_points = output[
            "stereo/visible_points_base_mm"
            if "stereo/visible_points_base_mm" in output
            else "full/points_base_mm"]
        distal_count = output["distal/points_base_mm"].shape[1]
        updated = 0
        for index in np.flatnonzero(valid & np.isfinite(smoothed)):
            full = np.asarray(source_points[index], dtype=np.float64)
            full_length = float(cumulative_arclength(full)[-1])
            length = float(np.clip(smoothed[index], 1e-3, full_length))
            smoothed[index] = length
            smooth_length_ds[index] = length
            adjustment_ds[index] = length - causal_length[index]
            suffix = _suffix_from_tip_length(full, length)
            sampled, _, _ = resample_polyline(suffix, distal_count)
            geometry = curve_geometry(
                sampled, curvature_smoothing_mm, curvature_spline_bases)
            output["distal/points_base_mm"][index] = geometry.points_mm
            output["distal/s_mm"][index] = cumulative_arclength(
                geometry.points_mm)
            output["distal/tangent_base"][index] = geometry.tangent
            output["distal/curvature_per_mm"][index] = (
                geometry.curvature_per_mm)
            output["distal/observation_class"][index] = OBSERVED_DISTAL
            output["distal/base_position_base_mm"][index] = geometry.points_mm[0]
            interface_s = full_length - length
            smooth_s_ds[index] = interface_s
            fitted_full_s = cumulative_arclength(
                output["full/points_base_mm"][index])
            output["full/observation_class"][index] = np.where(
                fitted_full_s >= max(0.0, fitted_full_s[-1] - length),
                OBSERVED_DISTAL, OBSERVED_PROXIMAL).astype(np.uint8)
            output["quality/distal_boundary_s_mm"][index] = interface_s
            output["quality/material_boundary_fraction"][index] = (
                interface_s / max(full_length, 1e-12))
            output["quality/distal_spline_basis_count"][index] = (
                geometry.spline_basis_count)
            output["quality/distal_spline_internal_knot_count"][index] = (
                geometry.spline_internal_knot_count)
            output["quality/distal_spline_rms_residual_mm"][index] = (
                geometry.spline_rms_residual_mm)
            output["quality/distal_spline_arc_length_mm"][index] = (
                geometry.spline_arc_length_mm)
            updated += 1
        output.attrs["interface_final_representation"] = (
            "robust_zero_phase_tip_back_arclength")
        output.attrs["interface_offline_cutoff_hz"] = float(cutoff_hz)
        output.attrs["interface_offline_huber_delta_mm"] = float(huber_delta_mm)
        output.attrs["interface_session_distal_length_mm"] = session_length_mm
        output.attrs["interface_length_gate_mm"] = float(length_gate_mm)
        output.attrs["interface_session_prior_relative_weight"] = float(
            session_prior_relative_weight)
        output.flush()
        finite_adjustment = np.abs((smoothed - causal_length)[valid])
        finite_adjustment = finite_adjustment[np.isfinite(finite_adjustment)]
        return {
            "updated_frames": int(updated),
            "adjustment_median_mm": float(np.median(finite_adjustment))
            if len(finite_adjustment) else float("nan"),
            "adjustment_p95_mm": float(np.percentile(finite_adjustment, 95))
            if len(finite_adjustment) else float("nan"),
            "session_distal_length_mm": session_length_mm,
            "length_gate_rejected_frames": int(np.count_nonzero(
                length_gate_rejected & valid)),
        }
