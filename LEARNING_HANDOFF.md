# Learning handoff: marked-catheter real trajectory

Last audited: 2026-08-22

This document is the contract between `catheter-shape-tracking` and the agent
responsible for dynamics learning. It describes the final real-data product,
not the older August 2 SAM experiment or the SOFA trajectory schema used by
`cr_meta_lnn`.

## Canonical files

Session:

```text
/media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260820_111551
```

Use this file for learning:

```text
processed_fusion_final/fused_dataset.h5
```

Supporting products:

```text
processed_image_final/processed_shapes.h5
processed_image_final/processing_summary.json
processed_image_final/frame_summary.csv
processed_image_final_observations/processed_shapes.h5
```

The fused HDF5 is schema version 2, uses the camera/image clock, and contains
exactly one row per processed stereo frame. The observation cache is retained
so image-space extraction does not need to be repeated if reconstruction or
temporal filtering is revised.

## Audit summary

- 10,911 stereo samples spanning 366.278 s.
- Timestamps are strictly increasing; median interval is 33.495 ms and the
  largest interval is 67.145 ms. Always use the stored timestamps rather than
  assuming exactly 30 Hz.
- All 10,911 rows have unique image source indices and zero image-clock offset.
- 10,744 rows have a per-frame 3D reconstruction.
- 10,468 rows pass image-side `learning_valid` quality control.
- 10,462 rows pass `frames/fusion_valid`; the difference is six camera rows
  without a sufficiently recent actuation command.
- 161 curves were reconstructed by bounded temporal interpolation. All 161
  currently pass `learning_valid`, but they remain explicitly labeled.
- EM was disabled for this recording.
- The final distal geometry is the robust, zero-phase, common-basis spline
  representation. Filtered and observed spline coefficients are both retained.

## Coordinate and unit contract

Root HDF5 attributes are authoritative:

- Coordinate frame: registered robot base.
- Position and arc length: millimetres.
- Curvature: `1/mm`.
- Timestamp: integer nanoseconds.
- Robot joint order:
  `catheter_lin, catheter_rot, catheter_bend, sheath_lin, sheath_rot,
  sheath_bend`.
- Corresponding position units:
  `mm, deg, mm, mm, deg, deg`.

The command dataset contains joint velocities. Its units are therefore the
listed position units per second. In this recording the sheath commands are
zero.

The distal material segment runs from red marker 0, centered at the
proximal/distal interface, to the center of wide red marker 3 near the probe.
Its approximate physical length is 57 mm. This length was a soft reconstruction
prior, not a hard equality constraint.

## Model-facing datasets

All following paths are relative to `fused_dataset.h5`.

Time and validity:

```text
frames/timestamp_ns
frames/fusion_valid
frames/actuation_valid
image/reconstruction_valid
image/learning_valid
image/pre_interpolation_valid
image/curve_temporally_interpolated
image/learning_rejection_flags
image/source_index
image/svo_frame
```

Final distal shape:

```text
image/distal/points_base_mm                       (N, 64, 3)
image/distal/s_mm                                 (N, 64)
image/distal/tangent_base                         (N, 64, 3)
image/distal/curvature_per_mm                     (N, 64)
image/distal/base_position_base_mm                (N, 3)
image/distal/filtered_spline_coefficients_base_mm (N, 20, 3)
```

Diagnostics and alternative representations:

```text
image/distal/observed_spline_coefficients_base_mm
image/distal/pre_temporal_points_base_mm
image/distal/raw_base_position_base_mm
image/distal/causal_base_position_base_mm
image/distal/temporal_outlier_mask
image/quality/*
```

Actuation and feedback:

```text
robot/joint_velocity_command
robot/joint_position_measured
robot/encoder_raw
robot/command_valid
robot/position_valid
robot/encoder_valid
robot/feedback_valid
robot/command_age_ms
robot/position_age_ms
robot/encoder_age_ms
```

Commands are zero-order held from the most recent ROS command, with a 30 ms
maximum age. Measured positions and encoders are linearly interpolated only
when their bracketing messages are no more than 30 ms apart. Feedback is absent
more often than command or image shape, so do not require `feedback_valid`
unless the chosen model actually consumes feedback.

## Curvature and spline caveats

`curvature_per_mm` is scalar centerline curvature magnitude. Samples 0, 1, 62,
and 63 are intentionally NaN because open-curve derivative estimates are not
trusted at the spline boundaries. Use `curvature_per_mm[:, 2:-2]`, or evaluate
the stored filtered B-spline coefficients at interior collocation points.

The filtered coefficients are usually the cleanest learning target because
they are a fixed-dimensional, temporally filtered representation. They are
linear Cartesian control coefficients, not Cosserat strain coordinates.

Stereo centerlines do not observe material-frame roll or torsional strain. The
red rings are rotationally symmetric and do not resolve this ambiguity. Do not
construct three-component material strain by silently declaring a camera-frame
normal to be the material normal. A learning model must either:

1. learn centerline/Cartesian spline dynamics without torsion;
2. treat axial roll as a latent state informed by robot rotation; or
3. introduce an additional orientation measurement or non-axisymmetric marker.

## Required training mask

Start with `frames/fusion_valid`, but do not use it alone. The present image QC
does not reject every physically implausible arc length. A few accepted
intervals contract far below the approximate 57 mm material length or expand
above 70 mm while retaining low reprojection error in an ill-conditioned view.

For the first baseline, use this conservative mask:

```python
import h5py
import numpy as np

with h5py.File("fused_dataset.h5", "r") as h5:
    points = h5["image/distal/points_base_mm"][:]
    arc_length_mm = np.sum(
        np.linalg.norm(np.diff(points, axis=1), axis=2), axis=1)
    training_valid = (
        h5["frames/fusion_valid"][:].astype(bool)
        & (arc_length_mm >= 50.0)
        & (arc_length_mm <= 65.0)
        & ~h5["image/curve_temporally_interpolated"][:].astype(bool)
    )
```

This retains 10,193 rows. The 50--65 mm interval is a conservative data-quality
gate, not a geometric constraint to apply inside reconstruction or a claim of
exact catheter length. Keep the raw validity fields and arc length available
for sensitivity studies.

After a direct-data baseline is stable, run a separate experiment that includes
the 161 temporally interpolated rows. Never mix the policies without recording
which was used.

The most conspicuous accepted length failures are:

| Output frames | Time from first fused frame | Failure |
|---|---:|---|
| 5528--5548 | 185.681--186.350 s | curve contracts to 29--44 mm |
| 8119--8135 | 272.685--273.220 s | curve contracts as low as 12.839 mm |
| 6940--6951 | 233.147--233.515 s | curve expands to roughly 69--71 mm |

There are also isolated length outliers. Recompute the mask from the stored
curve rather than maintaining a hard-coded frame list.

## Sequence construction

Dynamics windows must be contiguous in both time and validity. Break a window
when any of the following occurs:

- `training_valid` is false;
- timestamp separation exceeds the chosen maximum gap;
- the interpolation policy changes;
- a collection/session boundary is crossed.

Do not remove invalid rows and then concatenate the remaining samples: that
would create artificial one-step transitions across rejected intervals.

Controls should be paired using an explicitly documented convention. A natural
first convention for the zero-order-held command is
`shape[t], command[t], dt[t] -> shape[t+1]`, with
`dt[t] = (timestamp[t+1] - timestamp[t]) * 1e-9`. Compare this with a one-frame
command delay during validation because sensing, communication, and mechanical
response latency have not yet been identified.

This session is one correlated sinusoidal run. Do not randomly split individual
frames. Use contiguous time blocks for preliminary debugging, with guard gaps
larger than the rollout horizon. For claims about generalization, collect more
sessions and split by complete session/trajectory rather than time-adjacent
windows from this run.

## `cr_meta_lnn` integration gap

`cr_meta_lnn` does not currently load this schema. Its legacy loaders expect
SOFA-style root datasets such as:

```text
timestamps
frame_poses
strain_coords
joint_commands
contact_force_body
```

Do not rename or fabricate these fields just to make the existing loader run.
In particular, centerline curvature magnitude is not equivalent to the
three-component SOFA `strain_coords` field, and centerline points do not provide
material-frame quaternion poses.

Implement a dedicated real-data adapter in `cr_meta_lnn` that:

1. reads the image-clock fused schema directly;
2. applies and records the validity/length/interpolation policy;
3. constructs contiguous windows without crossing rejected samples;
4. converts nanosecond timestamps to relative seconds while preserving `dt`;
5. converts degrees to radians only at a clearly documented model boundary;
6. selects either Cartesian spline coefficients/points or a mathematically
   justified observable strain representation;
7. stores session path, source schema version, processing provenance, frame
   indices, and QC policy in every training artifact.

The adapter should expose image quality and interpolation flags even if the
first model ignores them. They are needed for ablations and failure analysis.

## Minimal integrity check

Before every training run, assert at least:

```python
import h5py
import numpy as np

with h5py.File(path, "r") as h5:
    t = h5["frames/timestamp_ns"][:]
    assert h5.attrs["schema_version"] == 2
    assert h5.attrs["master_clock"] == "processed_image_frame"
    assert np.all(np.diff(t) > 0)
    assert len(np.unique(h5["image/source_index"][:])) == len(t)
    assert np.max(np.abs(h5["image/source_offset_ms"][:])) == 0
    assert h5["image/distal/points_base_mm"].shape == (len(t), 64, 3)
    assert h5["image/distal/filtered_spline_coefficients_base_mm"].shape == (
        len(t), 20, 3)
```

## Provenance and patching

The fused `image` group copies all source image-processing attributes under a
`source_` prefix, including the processing configuration, code hash, temporal
filter parameters, interpolation method, and rejection-flag mapping. The source
image HDF5 remains the authoritative visual/reconstruction audit product.

If downstream inspection identifies a short invalid interval that should be
temporally patched, rerun from the final image HDF5 with `--resume-from joint`
into a new directory, inspect the overlay, and generate a new fused file. Never
overwrite the current final data or silently change labels after a model has
been trained.
