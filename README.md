# shape_tracking

3D shape tracking of a steerable catheter centerline in free space.

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

### Four-slot base registration

Start unified recording with `v`, then hold the base probe stationary in bracket
slot 1 and press `1`; repeat for slots 2, 3, and 4. Each key captures the most
recent 0.5 seconds of valid probe measurements and reports its maximum positional
standard deviation. The measurements are written atomically to the `em`
section of `registration.json` in the same session directory, including probe
identity, sample/frame/time bounds, mean point, standard deviation, and tracking
error. Repeating a number replaces only that slot.

Enter the four known bracket centers in `registration_config.yaml` under
`em_registration.slot_centers_base`; coordinates use the file's top-level
`units` and must be expressed in the robot base frame. Once all four slots are
captured, the recorder automatically fits a rigid transform satisfying
`p_base = robot_base_T_aurora @ p_aurora`. The session JSON stores both transform
directions and per-slot/RMS/max fit residuals. If the YAML entries are still
empty or invalid, all probe measurements remain saved in the correct session and
the JSON reports `transform_status: awaiting_valid_config` with a validation
message.

### Automatic camera registration after EM

ChArUco detection is disabled during normal recording and the manual EM slot
captures, because the probe may block the boards. As soon as the four-slot EM
transform is solved, the recorder starts a fresh camera-registration phase.
Remove the probe, keep at least one configured board visible, and keep the
camera/base fixture stationary. It collects 150 valid board poses, writes those
registration observations to `board_poses.csv`, performs the camera fit, and
then disables ChArUco detection again.

The same session `registration.json` contains both nested `em` and `camera`
sections. The camera section includes left/right camera-to-base transforms and
their inverses, intrinsics, stereo baseline, configured marker-to-base
transforms, workspace bounds, left/right ROIs, per-board fit statistics, and the
source image timestamp. Verification images `registration_left.png` and
`registration_right.png` show detected markers/corners, board axes, the large
robot-base coordinate frame, workspace box, and green ROI. Override the defaults
with `--camera-registration-frames` and
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

### About "changing focal length"
The ZED2 has **fixed-focus lenses**; the SDK exposes no focus/focal-length
control. The script maxes the digital `SHARPNESS` setting (the only focus-like
knob) and reads the factory `fx, fy` for pose. If the catheter at ~30 cm looks
soft, that's the lens near-limit — move it to ≥0.4–0.5 m and/or capture at
HD2K/HD1080 so the markers span more pixels.

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
`1`..`4` capture registration slots, `s` snapshot,
`q`/ESC quit. Data is written under `./recordings/<timestamp>/` (override with
`--outdir`).

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
