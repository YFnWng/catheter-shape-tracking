"""Offline zero-phase temporal smoothing of rectified stereo disparity."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .geometry import cumulative_arclength, curve_geometry, resample_polyline
from .interface_smoothing import _valid_blocks, robust_zero_phase_smooth
from .session import transform_points


def smooth_stereo_disparity_hdf5(
        path: str | Path,
        registration,
        cutoff_hz: float = 2.0,
        huber_delta_px: float = 1.5,
        iterations: int = 4,
        maximum_gap_ms: float = 1000.0,
        curvature_smoothing_mm: float = 0.25,
        curvature_spline_bases: int = 20) -> dict[str, float]:
    """Smooth d(s,t), reconstruct 3D, and rebuild the full fitted curve."""
    path = Path(path)
    with h5py.File(path, "r+") as output:
        required = (
            "stereo/fitted_disparity_px", "stereo/ordered_left_px",
            "stereo/ordered_right_px", "stereo/visible_points_base_mm")
        missing = [name for name in required if name not in output]
        if missing:
            raise ValueError(
                "offline disparity smoothing requires new-schema stereo data: "
                + ", ".join(missing))
        valid = output["frames/valid"][:].astype(bool)
        timestamps = output["frames/timestamp_ns"][:]
        disparity = output["stereo/fitted_disparity_px"][:].astype(float)
        fx, fy = float(registration.K[0, 0]), float(registration.K[1, 1])
        cx, cy = float(registration.K[0, 2]), float(registration.K[1, 2])
        baseline = float(registration.baseline_m)
        reference = output["quality/stereo_reference_view"][:]
        p95 = output["quality/reprojection_p95_px"][:].astype(float)
        condition = output["quality/stereo_condition"][:].astype(float)
        weights = (
            np.exp(-np.clip(p95, 0.0, 50.0) / 12.0)
            * np.clip(condition, 0.05, 1.0))
        weights[~valid] = 0.0
        smoothed = np.full_like(disparity, np.nan)
        for block in _valid_blocks(
                valid, timestamps, maximum_gap_ms, split_on_missing=False):
            for sample in range(disparity.shape[1]):
                smoothed[block, sample] = robust_zero_phase_smooth(
                    disparity[block, sample], timestamps[block], weights[block],
                    cutoff_hz=cutoff_hz, huber_delta=huber_delta_px,
                    iterations=iterations)
        # The registered base is an absolute depth anchor. The distal endpoint
        # remains smoothed and is constrained by the cyan/yellow endpoint
        # disparity during causal reconstruction.
        base_camera = transform_points(
            registration.left_camera_T_base, np.zeros(3, dtype=np.float64))
        if base_camera[2] > 1e-6:
            smoothed[valid, 0] = fx * baseline / base_camera[2]
        else:
            smoothed[valid, 0] = disparity[valid, 0]
        output["stereo/smoothed_disparity_px"][:] = smoothed

        def dataset(name: str, shape, dtype=np.float32, fillvalue=np.nan):
            if name in output:
                return output[name]
            return output.create_dataset(
                name, shape=shape, dtype=dtype, fillvalue=fillvalue,
                compression="gzip", compression_opts=1, shuffle=True)

        count = len(valid)
        adjustment_ds = dataset(
            "quality/stereo_disparity_smoothing_rms_px", (count,))
        shape_adjustment_ds = dataset(
            "quality/stereo_shape_smoothing_rms_mm", (count,))
        camera_to_base = np.linalg.inv(registration.left_camera_T_base)
        updated = 0
        adjustments = []
        shape_adjustments = []
        for index in np.flatnonzero(valid):
            fitted = smoothed[index]
            if not np.all(np.isfinite(fitted)) or np.any(fitted <= 0.25):
                continue
            left = output["stereo/ordered_left_px"][index].astype(float)
            right = output["stereo/ordered_right_px"][index].astype(float)
            if int(round(reference[index])) == 1:
                right = np.column_stack([left[:, 0] - fitted, left[:, 1]])
            else:
                left = np.column_stack([right[:, 0] + fitted, right[:, 1]])
            depth = fx * baseline / fitted
            camera = np.column_stack([
                (left[:, 0] - cx) * depth / fx,
                (left[:, 1] - cy) * depth / fy,
                depth])
            points_base_mm = transform_points(
                camera_to_base, camera) * 1000.0
            causal = output["stereo/causal_visible_points_base_mm"][index]
            shape_adjustment = float(np.sqrt(np.mean(np.sum(
                (points_base_mm - causal) ** 2, axis=1))))
            disparity_adjustment = float(np.sqrt(np.mean(
                (fitted - disparity[index]) ** 2)))
            output["stereo/visible_points_base_mm"][index] = points_base_mm
            sampled, _, _ = resample_polyline(
                points_base_mm, output["full/points_base_mm"].shape[1])
            geometry = curve_geometry(
                sampled, curvature_smoothing_mm, curvature_spline_bases)
            output["full/points_base_mm"][index] = geometry.points_mm
            output["full/s_mm"][index] = cumulative_arclength(
                geometry.points_mm)
            output["full/tangent_base"][index] = geometry.tangent
            output["full/curvature_per_mm"][index] = geometry.curvature_per_mm
            output["quality/visible_arc_length_mm"][index] = float(
                cumulative_arclength(points_base_mm)[-1])
            output["quality/full_spline_basis_count"][index] = (
                geometry.spline_basis_count)
            output["quality/full_spline_internal_knot_count"][index] = (
                geometry.spline_internal_knot_count)
            output["quality/full_spline_rms_residual_mm"][index] = (
                geometry.spline_rms_residual_mm)
            adjustment_ds[index] = disparity_adjustment
            shape_adjustment_ds[index] = shape_adjustment

            distances = []
            means = []
            for view, projected in (("left", left), ("right", right)):
                centerline = output[f"images/{view}/centerline_px"][index]
                centerline = centerline[np.all(np.isfinite(centerline), axis=1)]
                distance = cKDTree(centerline).query(projected)[0]
                distances.append(distance)
                means.append(float(np.mean(distance)))
            combined = np.concatenate(distances)
            output["quality/reprojection_left_px"][index] = means[0]
            output["quality/reprojection_right_px"][index] = means[1]
            output["quality/reprojection_max_px"][index] = float(
                np.max(combined))
            output["quality/reprojection_p95_px"][index] = float(
                np.percentile(combined, 95))
            adjustments.append(disparity_adjustment)
            shape_adjustments.append(shape_adjustment)
            updated += 1
        output.attrs["stereo_final_representation"] = (
            "robust_zero_phase_disparity")
        output.attrs["stereo_offline_cutoff_hz"] = float(cutoff_hz)
        output.flush()
        return {
            "updated_frames": int(updated),
            "disparity_adjustment_median_px": float(np.median(adjustments))
            if adjustments else float("nan"),
            "disparity_adjustment_p95_px": float(np.percentile(
                adjustments, 95)) if adjustments else float("nan"),
            "shape_adjustment_median_mm": float(np.median(shape_adjustments))
            if shape_adjustments else float("nan"),
            "shape_adjustment_p95_mm": float(np.percentile(
                shape_adjustments, 95)) if shape_adjustments else float("nan"),
        }
