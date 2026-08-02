# shape_tracking

3D shape tracking of a steerable catheter centerline in free space.

## Current handoff: image-only SAM work (2026-08-02)

The immediate goal is to build a robust offline catheter segmentation pipeline
from rectified ZED2 stereo video. EM tracking is deliberately excluded from this
variant so the coil housing can be removed and the complete exposed catheter is
visible. Keep the existing EM/stereo sequence processor intact; add the SAM
workflow as a separate image-only processing path.

The latest image-only dataset is:

```
D:\robot-dev\catheter_sessions\20260802_134726
```

It was captured without the coil housing, with the joint-1 velocity limit restored
to 40 deg/s. Its important properties are:

- `session_metadata.json` reports `mode: image_only`, stereo enabled, and EM
  disabled.
- `stereo_20260802_134726.svo2` contains both rectified HD1080 views.
- `frame_index.csv` maps each SVO frame to its ZED image timestamp. Always retain
  `svo_frame` and `timestamp_ns` in derived SAM outputs.
- `registration.json` contains the solved left/right camera transforms relative to
  the robot base and the registered workspace ROIs. It intentionally has
  `em: null`; load it with `load_session_registration(..., require_em=False)`.
- Base boards 0 and 1 were both used. The registration overlay images are the
  quickest visual check that the base axes, workspace, and ROIs are sensible.
- The matching ROS bag is currently in WSL at
  `/home/wangyf/catheter_sessions/20260802_134726/rosbag`. Copy that directory
  into the Windows session as `<session>/rosbag` before transferring the dataset
  to another workstation.

Audit on 2026-08-02 found 12,661 playable, contiguously indexed frames over
429.967 s (29.444 Hz effective). Timestamps are strictly increasing, and the SVO
playable-frame count exactly matches `frame_index.csv`. There are 150 intervals
over 50 ms, mostly isolated 66.9 ms intervals and concentrated near startup; the
maximum interval is 401.4 ms. Treat timestamp differences as authoritative rather
than assuming an exact 30 Hz sample period. Within the marked 6-minute motion
trajectory, 10,736 frames cover 359.990 s at 29.820 Hz, with 26 intervals over
50 ms and a 100.35 ms maximum interval. Camera registration used 148/150 board-0
samples and 145/150 board-1 samples. Their two base estimates differ by 2.77 mm
and 1.23 degrees; within-board pose scatter is small, but this inter-board
agreement is the practical registration-accuracy warning for downstream results.

The matching ROS bag contains a complete 360.002 s seed-1 sinusoidal trajectory
plus its position-control return. Commands were delivered at 100.01 Hz and stayed
within `[-10,10]` mm/s, `[-37.85,40]` deg/s, and `[-1,1]` mm/s for joints 0-2.
Measured position ranges were approximately `[-0.038,39.598]` mm,
`[-137.055,175.125]` degrees, and `[-0.0002,9.786]` mm, so there is no meaningful
limit excursion. One axis-0 stall was suspected and recovered about 11 ms later;
there was no confirmed or latched hardware fault. Return-to-zero succeeded with
final absolute errors `[0.0014 mm, 0.0371 deg, 0.00018 mm]`. Camera frames match
the ROS `run_start`, `return_start`, and `run_end` markers within 2.2, 10.1, and
6.6 ms respectively.

### Moving to the SAM workstation

Transfer the entire `shape_tracking` repository and the entire session directory;
do not transfer `.venv`. On the new Windows workstation:

```powershell
cd D:\robot-dev\shape_tracking
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m unittest discover -s tests -q
```

Reading SVO2 directly also requires the ZED SDK and its `pyzed` wheel installed
for that Python 3.11 environment. A workstation without the SDK cannot decode the
SVO merely by installing this repository. Install PyTorch and the chosen SAM
implementation according to that workstation's CUDA version rather than adding a
machine-specific CUDA wheel to this package's base dependencies.

Recommended first SAM milestone:

1. Add an image-only processor that reads stereo frames through `SvoReader`, uses
   the registered left/right ROIs, and runs the same model independently on both
   rectified views.
2. Start with sparse frames and explicit positive/negative point or box prompts.
   Propagate masks only after single-frame behavior is reliable; periodically
   re-prompt to prevent video-tracker drift.
3. Preserve the complete distal flexible blue segment while excluding the dark
   blue proximal stiff segment, tape, wires, fixtures, and background. Store the
   full mask first, then derive an ordered base-to-tip centerline and the
   distal/proximal material boundary as separate outputs.
4. Write per-view masks, centerlines, prompt/model provenance, confidence and
   rejection reasons keyed by `svo_frame,timestamp_ns`. Never silently fill a
   failed frame.
5. Produce sparse overlay videos before bulk inference. Stereo consistency and
   temporal mask/centerline continuity should be quality checks, not assumptions.

The current `shape_tracking.sequence` command requires EM and is not the entry
point for this dataset. The HSV/skeleton code in `segmentation.py` is a useful
baseline and centerline utility, but no SAM integration has been implemented yet.

Sensors:
- **NDI Aurora** EM coils at the tip → 6-DOF pose feedback.
- **ZED2** stereo camera → time-stamped binocular video for offline centerline
  extraction.

Base registration: a **ChArUco** board (rigid to the catheter base) registers
the base to the camera; an EM probe registers the same base to the Aurora.

## Package layout

Installable as `shape_tracking` (src layout). The top-level import pulls in only
numpy + OpenCV, **not** the ZED SDK, so analysis-only projects can reuse the
board/pose code without `pyzed`:

```
src/shape_tracking/
  boards.py         # board geometry, dictionary, ids     (numpy + cv2.aruco)
  registration.py   # detect boards + solvePnP pose        (numpy + cv2)
  segmentation.py   # ROI -> blue mask -> ordered centerline (skimage)
  reconstruction.py # stereo curve fit (Croom+Lu hybrid)   (scipy)
  zed_capture.py    # ZedCamera wrapper                    (needs pyzed / [capture])
  cli.py            # record + live-track loop     -> python -m shape_tracking
  offline.py        # saved pair -> 3D centerline  -> python -m shape_tracking.offline
```

```python
from shape_tracking import boards, registration     # no ZED SDK needed
_, bs = boards.build_boards()
results, seen_ids = registration.detect_boards(gray, bs, K, dist)
```

The capture CLI opens the ZED2, captures time-stamped rectified stereo images
(and/or a native SVO2 recording), and estimates the ChArUco base pose live.

## Unified ZED + Aurora recording without 3D Slicer

The Windows capture process can own the Aurora serial port directly and record
both sensors through one entry point. Aurora polling runs in a dedicated worker
thread, so it is not gated by ZED frame retrieval, preview, or image processing.

```powershell
d:\robot-dev\shape_tracking\.venv\Scripts\python.exe -m shape_tracking `
    --outdir D:\robot-dev\catheter_sessions `
    --resolution HD1080 --fps 30 --aurora-port COM4 `
    --aurora-expected-tools 3 --autorecord
```

The Aurora serial port is exclusive: stop AuroraTracker in Slicer before using
this recorder. `--autorecord` starts both streams after both devices initialize;
in interactive mode the `v` key starts and stops both together. Acquisition is
asynchronous and both streams use the Windows epoch clock for offline alignment.

Each session contains `stereo_<session>.svo2`, `frame_index.csv`,
`em_poses.csv`, `em_metadata.json`, and `board_poses.csv`. The EM CSV has
one row per tool per
batched hardware frame. Positions are native millimetres; quaternions are native
scalar-first `qw,qx,qy,qz`. Aurora frame number, validity/status, tracking error,
request time, and complete-frame receipt time are retained.

At startup, PHINF identity is queried for all three tools. The part number
`610175 T6E0-S00923` identifies `base_probe`; the other two tools are assigned
stable `tip_coil_0` and `tip_coil_1` roles by their PHINF serial numbers. Every
CSV row includes the role, part number, serial number, manufacturer, revision,
tool type, and transient port handle. The complete mapping is repeated in
`em_metadata.json`; downstream code must use `tool_role` or `serial_number`, not
the discovery index.

### Image-only recording

To remove the EM coils and tip housing and collect stereo images for offline
segmentation, use the explicit image-only mode:

```powershell
d:\robot-dev\shape_tracking\.venv\Scripts\python.exe -m shape_tracking `
    --outdir D:\robot-dev\catheter_sessions `
    --resolution HD1080 --fps 30 --preview-fps 10 `
    --registration-config .\registration_config.yaml `
    --image-only --autorecord
```

Do not pass `--aurora-port` in this mode. The Aurora driver is not opened and
the session contains no `em_poses.csv` or `em_metadata.json`. Camera
registration starts automatically and uses only the configured robot-base
boards; the field-generator board and `field_generator_registration` entry are
ignored. The resulting `registration.json` has `mode: image_only`, `em: null`,
and a solved camera-to-base transform. `session_metadata.json` records the
enabled modalities independently of registration completion.

### Direct optical field-generator registration

The preferred workflow uses the existing base ChArUco board and the 100 x
100 mm field-generator board (`DICT_4X4_50` IDs 17-24) in the same camera frames.
Configure the fixed physical transform in `registration_config.yaml`:

```yaml
field_generator_registration:
  board_index: 2
  marker_id_offset: 17
  aurora_T_marker:              # p_aurora = T @ p_marker
    matrix:
      - [r00, r01, r02, tx_mm]
      - [r10, r11, r12, ty_mm]
      - [r20, r21, r22, tz_mm]
      - [0.0, 0.0, 0.0, 1.0]
```

When unified recording starts with `v` or `--autorecord`, detection begins
automatically. A sample is retained only when the field board and at least one
configured base board both have valid poses in the same image. After the fixed
frame count, the recorder calculates
`camera_T_aurora = camera_T_field_marker @ inv(aurora_T_marker)` and then
`base_T_aurora = inv(camera_T_base) @ camera_T_aurora`. No probe placement or
number-key capture is used in this mode.

The probe may remain connected, preserving the normal
`--aurora-expected-tools 3` command. If it is unplugged, start the recorder with
`--aurora-expected-tools 2`; optical mode still requires and identifies exactly
the two tip coils.

The printed marker origin is the upper-left chessboard corner when viewed
upright; +x is right, +y is down, and +z points into the printed surface. The
configured transform must include the true normal-direction offset and axis
orientation of the Aurora tracking origin. Do not assume that centering the
printed square alone makes the marker and Aurora frames identical.

### Four-slot probe registration (legacy fallback)

Start unified recording with `v`, then seat and hold the base probe stationary
in bracket slot 1 for at least 0.75 seconds before pressing `1`; repeat for slots
2, 3, and 4. Each key evaluates the recent probe history without blocking camera
or EM acquisition. A capture is accepted only when at least 20 valid samples
have a 95th-percentile position deviation no greater than 0.15 mm and orientation
deviation no greater than 1 degree. A rejected capture does not replace an
existing slot. The measurements are written atomically to the `em`
section of `registration.json` in the same session directory, including probe
identity, sample/frame/time bounds, mean point, standard deviation, and tracking
error, plus the complete stationarity diagnostics. Repeating a number replaces
only that slot. Override the defaults with `--em-registration-dwell-s`,
`--em-registration-min-samples`,
`--em-registration-max-position-deviation-mm`, and
`--em-registration-max-orientation-deviation-deg`.

Enter the four known bracket centers in `registration_config.yaml` under
`em_registration.slot_centers_base`; coordinates use the file's top-level
`units` and must be expressed in the robot base frame. Once all four slots are
captured, the recorder automatically fits a rigid transform satisfying
`p_base = robot_base_T_aurora @ p_aurora`. The session JSON stores both transform
directions and per-slot/RMS/max fit residuals. If the YAML entries are still
empty or invalid, all probe measurements remain saved in the correct session and
the JSON reports `transform_status: awaiting_valid_config` with a validation
message.

### Automatic camera registration

In direct optical mode, base-camera and Aurora-camera registration are solved
together immediately after recording starts. In legacy mode, ChArUco detection
starts after the four probe slots are accepted. Both modes collect 150 valid
poses, write the observations to `board_poses.csv`, perform the fit, and disable
ChArUco detection afterward.

The same session `registration.json` contains both nested `em` and `camera`
sections. The camera section includes left/right camera-to-base transforms and
their inverses, intrinsics, stereo baseline, configured marker-to-base
transforms, workspace bounds, left/right ROIs, per-board fit statistics, and the
source image timestamp. Verification images `registration_left.png` and
`registration_right.png` show detected markers/corners, board axes, the large
robot-base coordinate frame, registered Aurora field-generator frame, the
position and +z direction of each physically identified EM tip coil, workspace
box, and green ROI. The time-aligned source coil poses and their camera timestamp
offsets are retained under `camera.em_overlay` in `registration.json`. Override
the defaults with `--camera-registration-frames` and
`--camera-registration-min-corners`.

## Camera ↔ base registration (`shape_tracking.register`)

Averages the ChArUco fixture pose over many static frames and composes the robot
**base** pose from a YAML config, then projects a base-fixed workspace box into
the image as the segmentation ROI.

```powershell
& "…\.venv\Scripts\python.exe" -m shape_tracking.register `
    --config registration_config.yaml --frames 150 --show
```
Copy `registration_config.example.yaml` and fill in, per board, `T_marker_to_base`
(maps marker→base, `p_base = T @ p_marker`) and the `workspace` box (base frame,
metres; `+z` runs from the base toward the tip). If both boards are configured and
seen, the two base estimates are cross-checked for agreement (should be a few mm /
degrees). Output `registration/camera_base_registration.npz` holds `cam_T_base`,
`cam_T_board`, `K`, the workspace box, and the projected `roi_xywh`.

**Verify with `--show`**: the large RGB axes (base frame) should sit at the
catheter base with **+z along the catheter toward the tip**, and the green ROI
should cover the catheter's workspace. If the axes land wrong, the
`T_marker_to_base` convention/values need fixing. Registration is unaffected by
camera roll (portrait is fine); board detection works at any orientation.

## Offline shape reconstruction

`python -m shape_tracking.offline` reconstructs the thin catheter's 3D
centerline from one saved stereo pair and writes overlays + a 3D plot + CSV:

```powershell
d:\robot-dev\shape_tracking\.venv\Scripts\python.exe -m shape_tracking.offline `
    --session D:\robot-dev\recordings\20260704_185544 `
    --roi-left 1050,470,430,135 --roi-right 760,470,345,135 --max-radius 5
```
`--max-radius 5` trims the thicker sheath (radius ~6-8 px) so only the thin
catheter (radius ~4 px) from the collar to the gold tip is reconstructed. Omit
it to keep the whole segmented structure.

**Registration mode (recommended):** pass `--registration camera_base_registration.npz`
(from `shape_tracking.register`) instead of `--roi-left/--roi-right`. It then:
projects the workspace box to **auto-crop** the ROI in both views; **selects the
blue component nearest the registered base** rather than the largest — the
catheter emerges from the base, so this rejects other blue objects in the ROI
(e.g. blue 3D-printed rail clamps) even when they are larger; orders the
centerline base→tip using the **registered base position** (so `--base-hint` and
the catheter's image orientation don't matter — works for the portrait/vertical
mount); and outputs the centerline **in the robot base frame** (`+z` along the
catheter). `centerline_3d.csv` gains a `frame` column (`base`/`camera`).

> Verified end-to-end: base lands at ≈[0,0,0] in the base frame with `+z` along
> the catheter, sub-pixel reprojection. Without `--registration` it falls back to
> the largest-component + `--base-hint` behavior.
```powershell
d:\robot-dev\shape_tracking\.venv\Scripts\python.exe -m shape_tracking.offline `
    --session <session> --registration <path>\camera_base_registration.npz --max-radius 5
```
Outputs land in `<session>/reconstruction/`: `overlay_left.png`,
`overlay_right.png` (mask + centerline + reprojected 3D fit, with a legend),
`reconstruction_3d.png`, `centerline_3d.csv` (X,Y,Z mm, camera frame). Add
`--show` to also open an interactive, mouse-rotatable 3D plot (uses TkAgg).

Overlay legend: green = catheter mask; blue→red = 2D segmented centerline
(base→tip order); yellow dots = the reconstructed 3D curve reprojected into that
view (yellow on the line ⇒ the 3D fit matches the 2D centerline); cyan/yellow
rings = base/tip endpoints.

Pipeline (a **Croom 2010 stereo + Lu 2023 curve-fit** hybrid):
1. **Segment** each view: ROI → blue HSV threshold → morphology → skeletonize →
   longest geodesic path → ordered centerline. A medial-radius trim (`--max-radius`)
   walks back from the tip and cuts off the thicker sheath, so the base lands at
   the sheath collar and only the thin catheter (to the gold EM tip) remains.
   Rejects the sheath, bracket, and lead filaments.
2. **Reconstruct** (`--method disparity`, default): take the in-plane shape (u,v)
   from the left centerline (spline-smoothed, `--smooth2d`: lower = sharper turn,
   higher = less jitter) so a sharp in-plane turn is preserved. Measure disparity
   by **epipolar correspondence** — for each left point (u,v) the match is the
   right centerline point on the same rectified row v — so the *right* reprojection
   lands on the right centerline (arc-length correspondence biased it and made the
   right view visibly wrong). Regularize the weakly-observed disparity d(s) with a
   low-order polynomial (`--disp-order`). 3D: `Z=fx·B/d`, `X=(u-cx)Z/fx`,
   `Y=(v-cy)Z/fy`. `--method bezier` keeps the older global cubic fit (`--reg`).

> **Geometry caveat:** if the catheter lies nearly parallel to the stereo baseline
> (roughly horizontal in the image), depth along it is ill-conditioned and the arc
> length / depth curvature are uncertain. Orient the catheter **diagonally or
> vertically** in the frame (or rotate the ZED) for accurate depth.

## Multimodal sequence post-processing

`shape_tracking.sequence` processes an SVO2 session directly and combines the
rectified stereo video with the two EM tip coils and the unified
`registration.json`. By default it uses the `run_start` to `return_start`
markers copied into `<session>/rosbag`, so the position-control return is not
part of the training trajectory.

Run a sparse pilot before processing a complete session:

```powershell
D:\robot-dev\shape_tracking\.venv\Scripts\python.exe -m shape_tracking.sequence `
    --session D:\robot-dev\catheter_sessions\20260728_171407 `
    --stride 300 --max-frames 30 --write-video --write-3d-video
```

Process every trajectory frame:

```powershell
D:\robot-dev\shape_tracking\.venv\Scripts\python.exe -m shape_tracking.sequence `
    --session D:\robot-dev\catheter_sessions\20260728_171407 `
    --write-video --write-3d-video
```

Install the extra HDF5 dependency with `pip install -r
requirements-postprocess.txt` if the environment predates this pipeline.

Outputs are written to `<session>/processed/`:

- `processed_shapes.h5`: fixed arc-sampled full/distal curves, curvature,
  tangent, EM tip pose, observation class, and frame-level quality metrics.
- `frame_summary.csv` and `processing_summary.json`: audit and rejection reasons.
- `overlay_left.mp4` / `overlay_right.mp4`: material segmentation, projected
  3D shape, EM midpoint, and tip direction.
- `shape_3d.mp4` and `shape_3d_preview.png`: base-frame visualization with the
  distal section colored by curvature.

EM gaps longer than the configured limit are invalid rather than interpolated.
Observation class 1/2 denotes image-observed proximal/distal shaft, while 3/4
denotes a base/tip bridge. Curvature in a bridge is model-derived and must not be
treated as directly observed. The tip frame uses the sign-aligned mean coil z
axis and defines +x from coil part `07222026_01` toward part `003`, projected
perpendicular to z. The HDF5 coil-position order is `[003, 07222026_01]`.

### Board (decoded from `charuco10_DICT4X4_two_boards_LETTER.pdf`)
| param | value |
|-------|-------|
| layout | **two** independent 4×4 ChArUco boards |
| square length | **25.0 mm** (each board is 100 mm printed at 100%) |
| marker length | **18.75 mm** (0.75 × square) |
| dictionary | `DICT_4X4_50` |
| marker ids | board #1 → 1–8, board #2 → 9–16 |

> Print at **100% / Actual size** (not "fit to page"). Measure a printed square
> with calipers and set `SQUARE_LENGTH_M` at the top of the script — pose scale
> depends entirely on this number.

Generate distinct 100 x 100 mm targets for the Aurora field-generator fixture:

```powershell
D:\robot-dev\shape_tracking\.venv\Scripts\python.exe `
    D:\robot-dev\shape_tracking\scripts\generate_charuco_patterns.py `
    --output-dir D:\robot-dev\shape_tracking\generated_charuco
```

The default output contains two new `DICT_4X4_50` boards using IDs 17-24 and
25-32, so they cannot be confused with the existing base boards. The landscape
Letter PDF is dimensionally controlled; print it at **100% / Actual Size** and
confirm that the chessboard outer edge measures 100 x 100 mm. The PNGs are
inspection/reference files, not dimensionally controlled print files. Use
`--count 1` if the generator fixture only needs one face, or `--page a4` for an
A4 print sheet. Keep these targets semantically separate from the existing base
boards when adding field-generator detection to the registration pipeline.

### About "changing focal length"
The ZED2 has **fixed-focus lenses**; the SDK exposes no focus/focal-length
control. The script maxes the digital `SHARPNESS` setting (the only focus-like
knob) and reads the factory `fx, fy` for pose. If the catheter at ~30 cm looks
soft, that's the lens near-limit — move it to ≥0.4–0.5 m and/or capture at
HD2K/HD1080 so the markers span more pixels.

### Dual-5DOF coil fixture calibration

Capture synchronized stereo ArUco and Aurora samples with:

```powershell
python -m shape_tracking.tool_calibration_capture `
    --aurora-port COM4 `
    --aurora-expected-tools 3
```

Future captures estimate marker 33 from both ZED images, reject snapshots whose
stereo reprojection RMSE exceeds 3 px, and automatically write
`tool_calibration_report.json` after at least six accepted poses. The solver
uses each 5-DOF coil's center and quaternion +z axis; unobservable roll about z
is deliberately ignored. It validates the fitted coil spacing against 3.8 mm
and the fitted axes against OpenCV marker -y (CAD +y).
The initial field-generator lock likewise estimates the field ChArUco board
from both cameras over a stationary multi-frame window. Reports from older
sessions re-estimate that lock from their saved synchronized image pair.

Reprocess an existing capture without hardware using:

```powershell
python -m shape_tracking.tool_calibration_solve `
    D:\robot-dev\catheter_sessions\aruco_em_calibration_YYYYMMDD_HHMMSS
```

## Install (dedicated venv, Python 3.11 to match the ZED wheel)

```powershell
# 1) dedicated venv for this project
py -3.11 -m venv D:\robot-dev\shape_tracking\.venv
D:\robot-dev\shape_tracking\.venv\Scripts\Activate.ps1

# 2) editable-install the package + its deps (numpy, opencv-contrib-python)
#    Do NOT also have plain opencv-python installed — it shadows the aruco module.
pip install -e "D:\robot-dev\shape_tracking[capture]"

# 3) ZED python API (pyzed) into THIS venv — SDK 5.4.0 is installed. pyzed is not
#    on PyPI, so run the SDK helper (it detects the active interpreter):
python "C:\Program Files (x86)\ZED SDK\get_python_api.py"
#    then pip install the .whl it downloads, e.g.:
# pip install pyzed-5.4-cp311-cp311-win_amd64.whl
```

Verify:
```powershell
python -c "import cv2, cv2.aruco, numpy, pyzed.sl; print('ok', cv2.__version__)"
```

### Reusing from another project (all consumers on Python 3.11)
venvs are isolated — you can't point one venv at another's site-packages. Instead
editable-install this package into each consumer venv:
```powershell
# analysis-only project: gets boards/registration, does NOT need the ZED SDK
pip install -e "D:\robot-dev\shape_tracking"
```
`-e` means every consumer tracks your latest edits with no reinstall.

## Run
```powershell
python -m shape_tracking --resolution HD1080 --fps 30
# or the installed console script:
shape-tracking-record --resolution HD1080 --fps 30
```
Keys in the preview window: `r` save PNG pairs, `v` unified SVO+EM record,
`s` snapshot, and
`q`/ESC quit. Data is written under `./recordings/<timestamp>/` (override with
`--outdir`). In legacy probe-registration mode only, `1`..`4` capture the four
registration slots.

### Output (`recordings/<timestamp>/`)
- `left/`, `right/` — PNGs named `<image_timestamp_ns>.png` (stereo pairs share
  the timestamp).
- `stereo_<session>.svo2` — native ZED recording (both views + timestamps +
  calibration) for offline processing.
- `left_intrinsics.npz` — `K`, `dist`, `fx/fy/cx/cy` for the rectified left cam.
- `board_poses.csv` — the fixed post-EM camera-registration pose set:
  `timestamp_ns, board, n_corners, t(xyz) m, r(xyz) rodrigues`.

The on-screen `ids=[...]` HUD shows which marker ids are detected — use it to
confirm both boards (0–7 and 8–15) are recognised. If a board never appears,
adjust `BOARD_ID_OFFSETS` in the script.
