# shape_tracking

3D shape tracking of a steerable catheter centerline in free space.

The current real-data schema, quality audit, training mask, and `cr_meta_lnn`
integration requirements are documented in [LEARNING_HANDOFF.md](LEARNING_HANDOFF.md).

## Marked light-box recordings (2026-08-19)

Session `20260819_194906` uses a neutral light-box background and four ordered
red rings: the proximal ring marks the material interface, two thinner rings
provide internal landmarks, and the widest distal ring covers most of the
ablation probe.  The remaining exposed yellow dot is the physical tip.
For shape learning, marker 3's center—not that small yellow dot—is the distal
material endpoint, giving an approximately 57 mm interface-to-end segment.

Use the color-first backend for this and compatible recordings:

```bash
source /home/chen-lab/Yifan/cr-venv/bin/activate
cd /home/chen-lab/Yifan/catheter-shape-tracking
python -m shape_tracking.image_sequence \
  --session /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260819_194906 \
  --segmentation-backend chromatic_markers \
  --reconstruction-backend joint_spline \
  --window run_and_return \
  --outdir /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260819_194906/processed_image_marked \
  --write-video
```

`joint_spline` is the preferred reconstruction for controlled marked
recordings. Disparity and the per-eye graph routes are retained only as a
per-frame 3D initializer; they no longer compete to become the final topology.
One cubic material-coordinate B-spline is optimized against both images at
once. Overlap-aware stereo supplies one paired, material-indexed corridor from
the interface to marker 3. The optimizer uses fixed pointwise correspondence
to those corridors: it cannot reassign a sample to an arbitrary nearest ridge
on the opposite V arm. This fixed-corridor mode is activated only when one eye
is detected as non-bijective; well-conditioned stereo retains the more
calibration-tolerant unordered medial-evidence objective. Corridor samples are
parameterized by reconstructed 3D arclength, not stereo sample index. A joint
normal search centers the two corridors on complete-silhouette distance ridges
while enforcing their rectified epipolar-row agreement. Wide merged runs and
large corrections are skipped so centering cannot switch V arms. Color
thresholds no longer define the medial axis, eliminating illumination-dependent
edge bias. Perspective projection enforces stereo geometry with one common
material coordinate in both eyes.

An ill-eye heuristic no longer switches objectives unconditionally. Entry,
eye switches, and release all require five consecutive supporting frames, so a
single ambiguity-threshold crossing cannot replace a usable image route.
Suspected frames first run the ordinary equal-weight two-view fit;
fixed-corridor recovery is used only when that fit fails strict model-distance
and coverage checks.
Projected-turn count remains diagnostic for this ordinary-fit decision: the
ill-eye label has release hysteresis, so accurate two-view image evidence must
be allowed to reacquire immediately after the overlap ends.
`quality/joint_topology_recovery_used` records the resulting
per-frame decision. This prevents benign projected-length differences from
activating overlap recovery while an eye is still geometrically usable.

Raw red-ring centroids enter only as soft image residuals: rings 0 and 3 use
tighter finite uncertainties than rings 1 and 2, but no centroid is a hard
spline waypoint. The approximately 57 mm length, approximate marker material
coordinates, endpoint locations, local stretch, and temporal coefficient
continuity are likewise finite robust penalties. A strong second-difference
prior applies everywhere except one localized material interval, where high
curvature remains legal. In the two image projections, the localized prior is
instead applied to third differences: ordinary smooth curvature remains legal
everywhere, while abrupt curvature changes are confined to the shared turn
window. A post-fit topology gate requires at most one separated projected
sharp-turn cluster per eye. In an ill-pose interval, its material location is
transferred from the farthest epipolar sweep row in the good eye; the good eye
does not itself need to display a cusp. That location is temporally
rate-limited. The previous accepted 3D spline is used as a soft prior in every
temporally adjacent frame, not only after an ill eye has already been declared.

After optimization, the same 3D spline is projected into both images and its
distal projections become the displayed/stored cyan paths. Thus the cyan and
yellow curves cannot encode different distal topologies. The original
independent image routes remain stored as
`images/<view>/observed_centerline_px`; final reprojection QC compares against
those observations and packed masks rather than comparing the spline with its
own projection. The existing zero-phase coefficient pass operates on the
joint-fit curves and final two-view evidence determines learning validity. Use
`--reconstruction-backend disparity` for the legacy behavior.

The joint optimizer uses an analytic coefficient Jacobian. Cubic B-spline
collocation matrices are cached by sample/basis count, perspective projection
is differentiated in closed form, and the length, local-stretch,
third-difference, projected-curvature-variation, fixed-corridor, marker, and
coefficient-prior blocks are assembled directly. A residual and its Jacobian
share one cached evaluation at each trust-region iterate; no per-coefficient
finite differencing is performed.

In joint mode, disparity is only an initializer and its reprojection threshold
does not reject a finite joint fit. Per-frame joint reprojection weights the
offline coefficient filter, while learning acceptance is decided from the
final temporally filtered curve, including endpoint error to marker 3. This
avoids good/bad chatter when an initializer or raw per-frame p95 value crosses
a threshold by a fraction of a pixel.

Joint-fit coefficients are stored in `joint/coefficients_base_mm`. Its
per-view model and coverage residuals, pre/post symmetric residual, optimized
length, cost, evaluation count, and convergence flag are stored under
`quality/joint_*`. The chosen curvature-relaxation coordinate and the two
measured projected turn angles are stored as `quality/joint_turn_fraction` and
`quality/joint_{left,right}_turn_angle_deg`.

`chromatic_markers` does not run SAM inference. It extracts class-specific
dark-blue/cyan/red/yellow likelihood rather than generic saturation and forms
an independent per-eye mask by hysteresis: strong paint pixels seed connected
regions and weaker edge pixels are admitted by only three one-pixel growth
steps from that strong core. It does not promote a complete weak connected
component or morphologically close it; this prevents blur between two arms of
a tight V from filling the inner wedge. The graph-cut uncertainty band is also
limited to two pixels beyond the color mask. The mask is clipped to the convex image projection of the registered 3D
workspace, rather than merely its axis-aligned crop, so the blue sheath behind
the workspace is excluded. A projected proximal cylinder along base-frame
`-z` removes the remaining rigid sheath, and a color-seeded local graph cut
snaps the mask to image edges. It then detects the distinctive distal ring and selects a
mask-constrained, color-weighted, temporally regularized route from the
registered base to that ring center. The
small exposed yellow dot is deliberately ignored. Catheter-specific color
support uses Lab chromaticity rather than low-value HSV saturation. The route
is a minimum-cost path over the continuous paint likelihood, not a skeleton of
a thresholded connected component: it can cross a short weak-paint/specular
gap, while a long traversal through neutral black or white is expensive. The
decoded rings are projected onto the independently smoothed centerline and do
not become path waypoints. There is no generic HSV/component fallback. The
backend then decodes the ordered ring centers
using order, relative widths, and a weak approximate spacing score. This
marker-anchored route prevents a longer saturated ChArUco edge from becoming
the catheter centerline. The independently detected wide distal ring takes
precedence over any endpoint decoded from the provisional skeleton, so an
incorrect ChArUco branch cannot validate itself. A sparse temporal-median
background removes fixed colored ChArUco fringes. Neither the current nor the
preceding centerline adds pixels to the mask. Matching
left/right rings are finite-weight disparity observations; their image
centroids never pin a 3D spline point because an annulus centroid need not lie
on the catheter axis. Implausible one-frame identity jumps are marked missing
rather than replaced by the preceding pixel coordinate. Current-frame
epipolar candidates from either eye can recover them, so a stale marker cannot
remain frozen indefinitely. When one eye compresses neighboring IDs onto the
same visible ring, the eye with the more distinct ordered layout supplies the
identities for a global one-to-one assignment over current-frame red components
in the other eye. Thus local detector confidence cannot override a gross
same-ID epipolar inconsistency. No marker spacing or nominal length is imposed as a hard
geometric constraint; the approximate 57 mm length is only a soft regularizer. The interface ring
replaces the old blue-brightness change point when observed; missing
observations fall back to temporal evidence and the soft length prior.
When one projection is foreshortened, ordered rings from the longer,
well-conditioned eye recover small or dark red components in the other eye
within a relaxed rectified-epipolar band. These matches correct marker identity
without expanding either eye's mask, and they never force the 2D or 3D curve
through an annulus centroid. A gross epipolar inconsistency can override local
confidence. If this changes the distal-ring identity, the affected cyan path
is rerouted through its existing color mask to the corrected terminal ring
before 3D fitting. The cyan path stops at the
nearest image-path pixel in marker 3's terminal neighborhood; its subpixel ring
centroid is not appended after smoothing, which avoids a hard endpoint snap and
a short artificial terminal kink.

Marked-mask width QC uses the median local support diameter along the cyan path
rather than mask-area divided by projected length. The latter grows without
bound under self-overlap or strong foreshortening even when the mask is correct.

For an ill-conditioned marked view, a large projected-length mismatch (default
1.23) or more than 15% cyan samples outside its mask triggers ordered
good-eye/temporal recovery. The good eye supplies normalized arc-length order
and rectified image rows; the preceding disparity supplies an x-coordinate
prior; the bad eye supplies current green-mask support. Each ordered sample is
snapped to target-mask paint near its epipolar row. Target pixels are explicitly
allowed to repeat, since projected self-overlap is non-bijective. The inferred
curve therefore cannot leave the target mask merely because disparity
smoothing placed it between two thin branches. Diagnostics are stored as
`quality/centerline_outside_mask_fraction_left/right` and
`quality/stereo_retry_view`. If the ordinary retry is not accepted and the
length mismatch remains, the shorter eye is excluded from ordered-reference
selection. The longer eye then drives an unordered full-mask epipolar solve;
the shortened eye is retained in the symmetric joint spline fit with reduced
observation weight (`--ill-eye-min-observation-weight`, default 0.2). The next
frame's temporal disparity prior is projected from the finalized joint spline,
not from its pre-joint disparity initializer. Attempt/result codes and the
selected ill eye are stored under `quality/stereo_retry_*` and
`quality/stereo_ill_view`.
The hysteretic label affects a view's observation weight only while that view
is also the currently shorter projection. During a pending switch or release,
both views retain unit weight; a stale label can no longer suppress the newly
good eye.
An ambiguity trigger (default 30% of epipolar rows, after the projected-length
ratio exceeds 1.15) activates this mode before the cyan path fully collapses.
Once active, temporal cyan-consistency rejection is bypassed only for the
identified ill eye so every frame can reach stereo recovery. Regularized
disparities and the displayed inferred cyan path are snapped back to the
current target mask; temporal priors cannot pull them into unsupported pixels.
If two projected arms merge into one filled mask lobe, the epipolar solver also
retains the mask-supported pixel nearest each preceding material-coordinate
prediction. This prevents the run-midpoint representation from deleting both
arms and creating a false cyan path through the middle of a tight V. The joint
3D spline receives the preceding accepted spline as a soft coefficient prior
in every adjacent frame (0.65 times
`--ill-eye-temporal-shape-sigma-mm`, or 3.9 mm by default). The good eye and
current mask evidence can still move the curve, but a many-to-one eye cannot
abruptly change its branch or material order before the ill-eye detector reacts.

Forced overlap recovery now keeps independently extracted image centerlines
separate from stereo-inferred projections. An inferred path never becomes the
next frame's color-routing prior, eliminating a feedback loop in which a wrong
branch in one frame contaminated subsequent frames. The active ill eye is held
until five consecutive frames support entry, switching, or release, rather than
alternating with small framewise score changes. Within a connected V-shaped
mask run, overlap candidates come from each distance-transform ridge instead
of the run midpoint. A pair-state dynamic program transfers the complete
good-eye epipolar-row sequence with first- and second-order disparity costs.
Forced recovery preserves those selected pixels: only missing epipolar rows
are interpolated, and later disparity smoothing or unrestricted nearest-mask
snapping cannot silently change V branches.
An inferred route that passes mask support, non-reversing marker order,
terminal support, exactly-one-turn, mask-mediality, and tighter
translation-compensated temporal-motion checks may initialize stereo, but it
is not a final topology candidate. The
single joint spline always makes the final decision from both masks, both sets
of soft rings, stereo geometry, smoothness, and temporal continuity. These
initializer diagnostics remain stored under `quality/stereo_inferred_*`;
marker/topology metrics are recomputed from the final joint projections.

Marker 3 identity combines distal position, observed width, and the preceding
tip when available; it is not assigned solely to the largest red component.
Markers 0 and 3 have higher stereo weights than the two intermediate rings,
and marker-0 interface observations use a tighter finite uncertainty. A
temporal identity gate rejects implausible one-frame marker jumps. During an
observation-cache resume, every per-eye marker track is then repaired
bidirectionally: a snap is detected against the chord between trusted past and
future observations, and short missing intervals are interpolated in image
time. Repaired samples are stored in `images/<view>/marker_interpolated`; their
confidence remains finite and lower than a direct observation. This makes
markers available through a brief ill pose without freezing them at a past
pixel coordinate or imposing marker centroids as curve waypoints.

After stereo marker identities are reconciled, marked recordings receive a
second 2D routing pass. The registered base and ordered ring identities divide
the shaft into material intervals, but each ring contributes a local red-mask
support region rather than a hard centroid waypoint. The base-to-marker-0
interval preferentially follows dark-blue paint, preventing the proximal
sheath from jumping onto a nearby cyan distal arm. Interior material-coordinate
supports from the preceding accepted route are transferred separately inside
each marker interval and snapped to current mask evidence. Consequently a
tight projected V retains both arms and its turn location instead of being
reduced to the shortest connection through a merged mask lobe. If current
marker order contradicts the preceding route, those temporal supports are
dropped for that frame. Ring spacing remains an identity regularizer only; it
is not imposed on either the image path or the reconstructed length.
During overlap-aware stereo, exact cyan epipolar crossings are labeled by the
same ring-delimited intervals. A crossing from a different material interval
receives a categorical branch penalty, while multiple samples within the same
interval may still reuse one target pixel. No within-interval disparity or
image arc length is interpolated, since either can be strongly nonlinear under
foreshortening. Forced ill-eye recovery uses only exact rectified rows, so the
selected corridor cannot move outside the current target-eye mask when its
vertical coordinate is transferred from the reference eye. Its dynamic program
has three topology states (before, inside, and after one contiguous sharp-turn
cluster). Adjacent high-angle samples at one physical V therefore count as one
turn, while a later branch snap is infeasible. The good-eye epipolar-sweep
extremum is a soft turn-location cost rather than an exact geometric constraint.
If too few exact-row mask samples exist, the frame is reported unsupported
instead of reverting to ordered correspondences and producing a shortcut.

Before 3D reconstruction, the two marked centerlines also compare their
base-relative rectified-row sweeps. If one cyan path spans at least 4 pixels
less than the other, the fuller eye supplies its extremal epipolar row and the
preceding marker interval. The shortened eye must visit that row in the same
material order, but chooses the column from its own mask-medial color support;
there is no hard cross-eye point match. When the extremum precedes marker 3,
this makes a shortcut path reach the only sharp turn visible in the good eye;
marker 3 continues to provide the separately detected terminal observation. Configure the
gate and calibration tolerance with `--marked-epipolar-sweep-deficit-px` and
`--marked-epipolar-sweep-row-half-width-px`. Per-view sweep, deficit, repaired
view, and anchor-use diagnostics are stored under `quality/epipolar_sweep_*`.

For a compact visual audit without creating videos, add
`--write-snapshots --snapshot-count 12`. Full-resolution stereo overlays are
written under `overlay_snapshots/left/` and `overlay_snapshots/right/`, evenly
spaced across the selected processing window. Use `--snapshot-frames 26,132`
to request exact output-frame indices instead.

Per-eye marker centers, widths, confidence, and visibility are stored under
`images/<view>/marker_*`. Stereo marker positions and observation flags are
stored under `markers/`. Overlays label decoded markers `0` through `3` in
magenta. The previous SAM/cached/HSV backends remain unchanged for unmarked
recordings.

## Historical handoff: image-only SAM work (2026-08-02)

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

### Setting up the SAM environment

Transfer the entire `shape_tracking` repository and the entire session directory;
do not transfer `.venv`. On the Ubuntu 22.04 ROS workstation, reuse the shared
Python 3.10 `cr-venv` environment:

```bash
source /home/chen-lab/Yifan/cr-venv/bin/activate
cd /home/chen-lab/Yifan/catheter-shape-tracking
python -m pip install --upgrade pip
pip install -e .
python -m unittest discover -s tests -q
```

Reading SVO2 directly also requires the ZED SDK and its `pyzed` wheel installed
for that Python 3.10 environment. A workstation without the SDK cannot decode the
SVO merely by installing this repository. On Linux, activate `cr-venv` before
running `/usr/local/zed/get_python_api.py`; the helper detects Python 3.10 and
installs the matching `cp310` wheel into the active environment. ZED SDK 5.4's
wheel requires NumPy 2.x; keep the version selected by `pyzed` rather than
downgrading NumPy afterward. Install PyTorch and the chosen SAM implementation
according to that workstation's CUDA version rather than adding a
machine-specific CUDA wheel to this package's base dependencies.

The image-only implementation is `shape_tracking.image_sequence`. It runs the
same SAM 2.1 model on both registered rectified ROIs as one image batch per
stereo pair. In the default quality-preserving pipeline, bounded 16-frame
chunks are decoded and their frame-local colour prompts are prepared by four
CPU workers. A background producer overlaps the next chunk with main-thread
SAM inference and chronological stereo reconstruction. Automatic
prompts use the projected robot base, the selected blue/cyan shaft, and the
yellow component adjacent to the distal shaft endpoint; optional JSON prompts
can override them on individual frames. SAM masks are stored first, then reduced
to an ordered base-to-tip centerline and a dark-blue/cyan material boundary.

Run a sparse audited pilot before bulk inference (run from any directory after
the official SAM 2 checkout has been installed editable in `cr-venv`):

```bash
source /home/chen-lab/Yifan/cr-venv/bin/activate
python -m shape_tracking.image_sequence \
  --session /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260802_134726 \
  --sam-checkpoint /home/chen-lab/Yifan/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --stride 300 --max-frames 30 --write-video
```

Process every marked trajectory frame after reviewing the pilot overlays:

```bash
python -m shape_tracking.image_sequence \
  --session /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260802_134726 \
  --sam-checkpoint /home/chen-lab/Yifan/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --window run_and_return \
  --write-video
```

`run_and_return` is the default processing window. It includes the commanded
trajectory and the return-to-zero motion, while excluding unrelated recording
before and after the run. Unless `--outdir` is supplied, results are written to
the session's `processed_image/` directory alongside the source data.

After a completed SAM run, geometry/reconstruction changes can be evaluated
without running SAM again. The cached backend decodes the SVO, rebuilds the
material centerlines from the stored masks, and reruns stereo and 3D fitting:

```bash
python -m shape_tracking.image_sequence \
  --session /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260802_134726 \
  --sam-checkpoint /home/chen-lab/Yifan/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --segmentation-backend cached \
  --cached-mask-h5 /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260802_134726/processed_image/processed_shapes.h5 \
  --outdir /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260802_134726/processed_image_refined \
  --write-video
```

The cached source and output must differ. This mode retains genuine mask
failures but can recover frames previously rejected only by downstream stereo,
transition-color, or spline logic. Cached reconstruction does not duplicate the
large source masks unless `--store-masks` is explicitly supplied; the refined
HDF5 still contains pixel centerlines, 3D geometry, quality, and robot data.

The same cache boundaries are exposed explicitly through `--resume-from`:

```bash
# Build the expensive mask/2D/marker cache once, without running stereo or 3D.
python -m shape_tracking.image_sequence --session SESSION \
  --segmentation-backend chromatic_markers --observations-only \
  --outdir OBSERVATION_CACHE

# Reuse masks, cyan centerlines, rings, and other image observations; rerun all 3D stages.
python -m shape_tracking.image_sequence --session SESSION \
  --resume-from observations --resume-h5 OBSERVATION_CACHE/processed_shapes.h5 \
  --reconstruction-backend joint_spline --outdir NEW_FROM_OBSERVATIONS

# Reuse image observations and the stored stereo initializer; rerun the joint fit onward.
python -m shape_tracking.image_sequence --session SESSION \
  --resume-from stereo --resume-h5 OLD/processed_shapes.h5 \
  --outdir NEW_FROM_STEREO

# Keep the per-frame joint fit; rerun coefficient filtering, geometry, and learning QC only.
python -m shape_tracking.image_sequence --session SESSION \
  --resume-from joint --resume-h5 OLD/processed_shapes.h5 \
  --outdir NEW_FROM_JOINT
```

For `--resume-from observations`, omitting `--reconstruction-backend` now
inherits the backend recorded in the source HDF5. This prevents a joint-spline
validation from silently reverting to the disparity-only default.

`observations` decodes the selected SVO frames to support normal overlays but
does not repeat segmentation. `stereo` and `joint` copy the source HDF5 into a
new output directory and do not decode the SVO unless an overlay or snapshot is
requested. The source file is never modified. A stereo-level restart currently
requires the stored and requested joint basis counts to match; use
`observations` when changing the HDF5 spline schema. Full primary runs should
retain packed masks (`--store-masks`) so every restart level and later visual
audit remains available.

The performance controls are:

- `--prompt-workers 4`: CPU workers for independent left/right colour and
  prompt extraction.
- `--chromatic-eye-workers 2`: run the independent left/right chromatic mask,
  edge/centerline, and route-refinement work concurrently.
- `--sam-postprocess-workers 2`: persistent CPU workers for independent SAM
  mask selection, morphology, skeletonization, and centerline extraction.
- `--preprocess-chunk-size 16`: number of decoded frames prepared together.
- `--prefetch-frames 16`: maximum prepared-frame queue; use 0 to disable
  decode/preprocessing overlap.
- `--hdf-buffer-frames 128`: consecutive completed records combined into one
  slice write per dataset. Numeric dataset chunks are aligned to this batching
  up to a one-megabyte chunk target.
- `--hdf-queue-chunks 2`: bounded completed-record chunks held for the
  dedicated HDF5 writer thread. This overlaps compression and external-drive
  writes with subsequent image processing without unbounded RAM growth.
- `--sam-frame-batch-size 1`: preserves previous-accepted-frame SAM prompting.
  Values above 1 batch independent timestamps through the image encoder and
  use temporal prompts only as selective retries. This is experimental because
  independently plausible masks can choose different skeleton branches.

The reconstruction and material-interface defaults are now:

- `--overlap-aware-reconstruction`: when one projected centerline touches a
  nonlocal part of itself, the better-separated eye supplies ordered arc
  length. The overlapped eye is treated as an unordered mask observation. A
  dynamic program selects a smooth disparity path from all mask runs on each
  epipolar row; it never requires other-eye arc order or a one-to-one mapping.
  Multiple good-eye samples may therefore project onto the same bad-eye pixel.
  Ordinary frames retain the ordered epipolar solver. Detection is controlled
  by `--overlap-self-distance-px 8`,
  `--overlap-min-arclength-separation 0.12`, and
  `--overlap-self-fraction-threshold 0.05`. For marked data, a projected-length
  collapse also forces this unordered-mask mode even when the collapsed cyan
  path itself no longer contains enough geometry to report self-overlap.
- A self-overlapped eye is excluded from reference-view selection whenever the
  other eye remains separated. `--stereo-reference-hysteresis-score 3` keeps
  the previous good reference through small score fluctuations. The disparity
  prior survives rejected/overlapped frames for up to
  `--temporal-disparity-max-gap-ms 1000`, with a 500 ms exponential decay.
- The two cyan centerline endpoints provide a strong terminal disparity anchor
  when their rectified-row error is at most 8 px. This keeps the non-reference
  reprojection near the observed tip when the yellow cap detector is missing.
- `--stereo-offline-cutoff-hz 2` robustly smooths disparity at every normalized
  arc sample across time after causal processing. The reference-eye pixels
  remain exact, the registered base disparity is not moved, and the full 3D
  spline is rebuilt before the interface is cut. Set this to `0` to retain only
  causal disparity.
- `--interface-offline-cutoff-hz 2`: after all causal frames have been written,
  robust symmetric second-difference smoothing is applied to the scalar
  tip-back distal length. This is a zero-phase batch operation, so it adds no
  temporal lag and never smooths the interface XYZ away from the catheter.
  `--interface-offline-huber-delta-mm 2` controls outlier rejection. Set the
  cutoff to `0` to disable the final pass.
- Distal-length observations more than `--interface-length-gate-mm 4` from the
  nominal 60 mm are identified as the known false near-tip material boundary.
  They are replaced by a weak robust session-length prior rather than being
  allowed to pull the interface toward 45--50 mm. This is a soft constraint:
  trusted color/reconstruction observations still vary within the gate.
- When `--write-video` is active and masks are available, overlay videos are
  rendered after the offline pass, so the displayed interface and yellow curve
  match the final HDF5 data rather than the causal intermediate result.

The HDF5 preserves all interface stages:

- `distal/raw_base_position_base_mm`: unfiltered per-frame interface.
- `distal/causal_base_position_base_mm`: causal interface used while streaming.
- `distal/base_position_base_mm`: final zero-phase interface used for learning.
- `quality/distal_length_raw_mm`, `distal_length_filtered_mm`, and
  `distal_length_smoothed_mm`: raw, causal, and final tip-back coordinates.
- `quality/interface_smoothing_adjustment_mm` and
  `interface_smoothing_weight`: audit fields for the offline estimator.
- `stereo/visible_points_base_mm`: pre-spline stereo curve retained so the
  final distal segment can be re-cut without transferring the full-spline fit
  error into the interface.
- `quality/overlap_aware_used`, `stereo_epipolar_ambiguity_left/right`, and
  `stereo_other_eye_self_overlap_left/right`: reconstruction-mode diagnostics.
- `stereo/fitted_disparity_px`, `smoothed_disparity_px`, ordered stereo pixels,
  and causal/final visible 3D points: temporal-reconstruction audit fields.

True SAM 2 video-memory propagation is available as an optional two-stage
backend. It propagates each registered stereo ROI both forward and backward in
bounded chunks, refreshes memory with automatic prompts at regular anchors, and
stores a reusable bit-packed mask cache:

```bash
python -m shape_tracking.sam_video_propagation \
  --session /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260802_134726 \
  --sam-checkpoint /home/chen-lab/Yifan/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --chunk-frames 300 --anchor-stride 30 \
  --output propagated_masks.h5

python -m shape_tracking.image_sequence \
  --session /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260802_134726 \
  --sam-checkpoint /home/chen-lab/Yifan/third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --segmentation-backend propagated \
  --propagated-mask-h5 propagated_masks.h5 --write-video
```

Propagation is deliberately not the default: it can bridge weak individual
frames, but a bad video-memory track can drift for several frames and produce
inconsistent left/right centerlines. The same reconstruction can be run with
`--segmentation-backend hsv` for an ablation. HSV remains useful for automatic
prompts and propagation-candidate scoring, but a propagated mask does not
require an HSV detection to be accepted.

Outputs land in `<session>/processed_image/`: `processed_shapes.h5` contains
bit-packed per-view masks, fixed-sample pixel/3D centerlines, tangents,
curvature, distal flexible-segment base position, aligned command/POS/raw-ENC
streams, provenance, and quality metrics. `frame_summary.csv` and
`processing_summary.json` retain every failed frame and its rejection reason;
`overlay_left.mp4` and `overlay_right.mp4` are the visual audit. Per-frame stage
times (milliseconds) are included as `timing_*_ms` columns in
`frame_summary.csv`; `processing_summary.json` reports total time, mean time per
frame, and fraction of elapsed processing time for each stage. With background
prefetch, stage totals describe work rather than a serial critical path, so
their fractions can sum to more than 100%. GPU encoder and
prompt-decoder times use CUDA synchronization at their boundaries.
`hdf5_write` measures main-thread record preparation and queue backpressure;
`hdf5_write_background` measures actual batched compression/write worker time
and intentionally overlaps other stages. Each frame is accumulated only after
its final retry/boundary state is known, and consecutive records are committed
as dataset slices rather than dozens of per-frame HDF5 assignments. Failed
frames use the same batched path. The image-only
processor never invokes EM. The existing `shape_tracking.sequence` command
remains the EM-dependent path for older sessions.

### Post-hoc sensor fusion

Use `shape_tracking.fusion` after image processing to build the learning dataset.
Unlike the legacy `shape_tracking.sequence` path, this stage does not use EM to
alter or anchor the reconstructed image shape. It aligns independently measured
shape, EM tip pose, and robot actuation while preserving separate validity and
timestamp-offset fields.

The default uses the 100 Hz `/teleop/control` joint-command timestamps as the
canonical timeline because actuation is present in every recording. For
image-based learning, `--timeline image` instead emits exactly one row per
camera frame, avoiding repeated shape targets. POS and raw encoder feedback are
aligned onto the selected timeline. Processed image shapes are nearest-neighbor
sampled without interpolation, and `image/source_index`,
`image/source_offset_ms`, and `image/is_new_sample` make repeated camera samples
explicit on the actuation clock. EM coil poses are gap-aware interpolations followed by the existing
dual-coil tip-frame fusion. `frames/fusion_valid` requires the command and every
enabled optional modality; `robot/feedback_valid`, `image/valid`, and `em/valid`
remain available for less restrictive filtering.

For the current image-only session, run this after
`processed_image/processed_shapes.h5` has closed successfully:

```bash
source /home/chen-lab/Yifan/cr-venv/bin/activate
cd /home/chen-lab/Yifan/catheter-shape-tracking
python -m shape_tracking.fusion \
  --session /media/chen-lab/84BABCB7BABCA6D81/Yifan/catheter_sessions/20260802_134726 \
  --image-data --no-em-data --timeline image \
  --window run_and_return
```

For a session with both optional sensors, use `--image-data --em-data`; use
`--no-image-data --em-data` for EM only. Omitting both switches auto-detects
modalities from `session_metadata.json` and file presence, but explicit switches
are recommended for reproducible processing. An enabled missing input is an
error. Actuation is mandatory and has no disable switch.

The default output is `<session>/processed_fusion/fused_dataset.h5`, with a
compact `fusion_summary.json` beside it. Fusion also defaults to the marked
`run_and_return` window. The output copies model-ready 3D full/distal geometry
and scalar quality fields, but not masks or 2D SAM prompts; those remain in the
source image HDF5 named by the fusion metadata. The fused image group also
preserves `observation_valid`, `pre_interpolation_valid`, and
`curve_temporally_interpolated`, plus the source processing configuration and
code provenance, so repaired curves can be included or excluded explicitly.

The image-only stereo path preserves sharp turns by arc-length-resampling the
ordered pixel skeleton with piecewise-linear interpolation by default. Disparity
has one degree of freedom per stereo sample and is estimated with robust Huber
weights plus local first- and second-difference regularization; the registered
base depth is an endpoint observation, not a global polynomial constraint. The
processor constructs both left-ordered and right-ordered stereo hypotheses.
Each is scored using reprojection in both eyes, foreshortening, and distance
from the previous accepted 3D curve; a small switching penalty prevents
reference-view chatter. This lets the well-conditioned eye win on each frame
without making either camera permanently primary. At full frame rate, the
previous accepted disparity field is a local prior
(`--temporal-disparity-weight`, default 2.0), reducing depth and distal-base
jitter. Candidate scores, their margin, and temporal RMS errors are stored in
`quality/stereo_candidate_*`. In addition, the distal points can be
filtered causally with a first-order low-pass
(`--temporal-shape-cutoff-hz`; disabled by default), then refit. This filters
the transition point and the whole shape coherently instead of filtering a
single endpoint independently. It adds frequency-dependent lag, so enable it
only when that tradeoff is appropriate (for example, pass
`--temporal-shape-cutoff-hz 4`). The
per-frame coefficient is stored as `quality/temporal_shape_alpha`. The chosen
stereo view is stored as `quality/stereo_reference_view` (1=left, 2=right).

When the final 20% of both centerlines is unambiguous and neither view is
self-overlapped, a terminal disparity refinement increases the opposite-eye
observation weight and tapers the disparity second-difference penalty toward
the tip. The trusted stereo tip remains strongly anchored. The refinement is
accepted only when terminal reprojection improves without materially degrading
the full-curve p95 error. It is enabled by default with
`--terminal-disparity-refinement`; its extent and strength are controlled by
`--terminal-disparity-fraction`,
`--terminal-disparity-smoothness-scale`, and
`--terminal-disparity-observation-weight`. Usage and improvement are stored in
`quality/terminal_refinement_*`.

No modelled base bridge is inserted: a visible proximal endpoint farther than
`--max-base-endpoint-distance-mm` (15 mm by default) from the registered base
rejects the frame. The nominal distal material length is supplied with
`--distal-length-mm` (60 mm by default), but it is a soft prior rather than an
equality constraint. Dark-blue/cyan transitions from both views are searched
only within `--distal-boundary-search-half-width-mm` of that prior, robustly
weighted by color confidence, and fused with the previous accepted distal
length. The default prior, color, and temporal standard deviations are 4, 3,
and 1.5 mm and can be changed with `--distal-length-prior-sigma-mm`,
`--interface-color-sigma-mm`, and `--interface-temporal-sigma-mm`. Thus short
dark markings near the tip are downweighted without discarding genuine material
color information, and a local spline error cannot force all error into the
transition point.

The previous filtered 3D interface is projected onto a local window of the
current reconstructed curve and its material coordinate is causally filtered
along that curve (4 Hz by default). This avoids independently smoothing x, y,
and z off the shaft. The unfiltered and filtered locations are stored in
`distal/raw_base_position_base_mm` and `distal/base_position_base_mm`; length,
uncertainty, along-curve coordinate, and filter coefficient are stored in
`quality/distal_length_*` and `quality/interface_*`. Set
`--interface-temporal-cutoff-hz 0` to disable this last filter. The final distal
spline follows the fused, variable-length material segment rather than being
forced to exactly 60 mm.
The final fitted curve receives its own reprojection quality check. Its single
3D base point is projected into both overlay
views, so the displayed transition cannot be assigned independently or inherit
the wrong branch of a folded 2D projection. A frame is rejected rather than
labelled when its visible
3D curve is shorter than the distal segment or when the left/right image
centerline-length ratio exceeds `--max-stereo-centerline-length-ratio`.

The printed yellow component supplies an explicit distal endpoint. Its distal
edge is estimated along the local shaft tangent rather than using the blob
centroid, and that point is appended to the mask skeleton when the finite-width
cap makes the skeleton stop internally. Epipolar-consistent left/right tip
observations are triangulated and used as the terminal disparity anchor. The
stored `quality/tip_*` fields expose observation consistency and final endpoint
error; a stereo-observed endpoint farther than
`--max-tip-endpoint-error-px` from either detected tip rejects the frame.
Before rejecting a length mismatch, the processor retries the shorter SAM view
with prompts transferred from the complete view using rectified epipolar rows
and the registered base disparity. The retry is retained only when it improves
the length ratio and passes the normal SAM, base-position, and reprojection
checks. A transferred prompt can occasionally make SAM include a broad adjacent
surface and skeletonize through the background. For a failed view, the reliable
opposite-view centerline supplies the rectified epipolar row at each point; the
retry searches only the horizontal coordinate for the local blue/cyan image
ridge and smooths that disparity correction. The resulting path, rather than
SAM's potentially branched skeleton, defines a catheter-width mask corridor.
If the second SAM call fails completely, the same recovery may use the original
partial mask, but only when at least 75% of the transferred path has direct
blue/cyan or yellow-tip support in the target image and no unsupported run is
longer than 10% of the path. Stereo geometry alone cannot create an accepted
centerline. The accepted retry mask is refined from these material-color pixels
rather than retaining a broad SAM region.
All masks must also pass an area-per-centerline-length check controlled by
`--max-mask-effective-width-px` (20 px by default). The overlay includes a green
mask contour so a correct thin mask remains visible beneath the fitted curves.

For normal full-rate processing, each accepted frame also supplies full-shaft
positive points and a box to the next frame in the same view. This previous
accepted centerline is used directly as the next SAM prompt rather than being
unioned with a potentially degenerate current color box. Current-frame color
remains a mask-selection and completion cue, as in the original stable mask
pipeline. The temporal prompt is used only when the timestamp gap is at most
`--max-temporal-prompt-gap-ms` (100 ms by default), and current-frame color seeds
and all stereo/geometry checks still apply. Manual per-frame prompts take
precedence. Thus 30 Hz motion receives a stable image-space prior, while sparse
pilot frames separated by seconds never reuse stale geometry. Prompt points,
boxes, and their source are stored with the output. Successive accepted
centerlines must also retain adequate previous-path coverage; a path that
collapses or jumps to an adjacent structure is rejected by the temporal
coverage/p95-distance check.

The spatial curves use penalized cubic 3D B-splines with an explicit basis
count (`--curvature-spline-bases`, default 20, giving 16 internal knots), so a
fit cannot collapse to one global cubic.  As a final offline stage, all distal
frames are represented in the same normalized material coordinate and the same
uniform clamped knot vector.  The 3D control-point trajectories are robustly
zero-phase filtered (`--spline-temporal-cutoff-hz`, default 3 Hz).  A local
timestamp-weighted constant-velocity chord detects snap-away/snap-back
observations without treating sustained physical rotation as an outlier; point
outliers reduce only the weights of the spline bases that influence that shaft
region, while a predominantly bad curve rejects the whole frame observation.
Paired snap/return edges may bridge one short interval, but each edge cluster
can participate in only one bridge; alternating reconstruction modes can no
longer chain into a long unsupported block.
Three consecutive joint fits with strong two-view model and coverage evidence
explicitly reacquire after an ill interval and start a new supported temporal
block, rather than allowing the pre-ill motion predictor to veto the recovery.
The final centerline, tangent, and curvature are evaluated directly from the
filtered coefficients, with no independent per-frame refit.  Set the cutoff to
zero to disable this stage. The last four control points use a 5 Hz terminal
bandwidth by default so localized distal bending modes are attenuated less;
configure this with `--spline-temporal-terminal-cutoff-hz` and
`--spline-temporal-terminal-basis-count`.

The HDF5 output preserves the input to this stage in
`distal/pre_temporal_points_base_mm`, and stores observed and filtered control
points in `distal/*_spline_coefficients_base_mm`.  Pointwise outlier masks and
frame-level innovation, adjustment, outlier-fraction, and support diagnostics
are stored under `distal/temporal_outlier_mask` and
`quality/shape_temporal_*`.  Spline basis counts, fitted arc length, and RMS
temporal displacement remain available as quality metrics.  The main tuning
options are `--spline-temporal-huber-delta-mm`,
`--spline-temporal-max-gap-ms`, `--spline-temporal-outlier-sigma`, and
`--spline-temporal-outlier-floor-mm`.

The finalized yellow projection is independently checked against each packed
green mask after coefficient filtering. Frames exceeding either 20% outside
support or a 6 px 95th-percentile mask distance receive final-geometry flag 32;
these limits are configurable with `--max-final-mask-outside-fraction` and
`--max-final-mask-distance-p95-px`.

Short, bracketed invalid intervals are repaired after that image-space QC by
interpolating the entire common-basis 3D coefficient vector with a symmetric
cubic Hermite bridge. This is one whole-curve operation, not independent point
interpolation, so material order and spline topology are preserved. Tiny
accepted islands of at most two frames inside the interval are included to
avoid good/bad chatter. The default limit is 650 ms and can be changed with
`--spline-interpolation-max-gap-ms`. The source reconstruction state is stored
in `frames/pre_interpolation_valid`, synthesized frames are labelled by
`frames/curve_temporally_interpolated`, and final reprojection/mask QC is rerun
after interpolation. Linear and cubic-Hermite coefficient bridges are both
evaluated and the one with the smaller normalized point-step/arc-length error
is selected. A bridge is accepted only below the 8 mm per-frame material-point
step and 4 mm arc-length-deviation gates. Because V-overlap masks are known to
be unreliable inside exactly these short intervals, a physically accepted
bridge is not vetoed by per-frame mask QC; that override is recorded in
`quality/interpolated_curve_image_qc_overridden`. Thus a downstream learner may
use these repaired samples or exclude them explicitly.

A completed run can be patched without repeating image processing or the
per-frame joint fit:

```bash
python -m shape_tracking.image_sequence --session SESSION \
  --resume-from joint --resume-h5 FULL_RUN/processed_shapes.h5 \
  --outdir FULL_RUN_PATCHED \
  --spline-interpolation-max-gap-ms 900
```

The source is copied and never modified. Inspect the reported interpolated and
physically rejected runs before choosing a larger limit; long or unbracketed
intervals remain invalid.

`frames/learning_valid` is the model-facing validity label. It requires a valid
reconstruction, finite temporally filtered coefficients, no whole-shape or
terminal temporal-outlier decision, no predominantly unsupported material
region longer than the configured 500 ms interpolation limit, and acceptable
mask width. A persistent local outlier only downweights its influencing spline
bases; by default at least half of the material samples must be in long rejected
runs before the whole frame is unsupported. Unbracketed or long unsupported
runs are not bridged and their overlays omit the yellow curve.
`frames/learning_rejection_flags` records bitwise reasons:
1=reconstruction, 2=unsupported temporal gap, 4=whole-shape outlier,
8=terminal outlier, 16=mask width, and 32=final image/mask fit. The original
reconstruction state and all rejected shape diagnostics are retained for
inspection. Sensor fusion uses
`frames/learning_valid` when present, while exposing both
`image/reconstruction_valid` and `image/learning_valid`; consequently
`frames/fusion_valid` automatically excludes intermittent image outliers and
ill-conditioned reconstruction failures.

Optional manual prompts are full-image pixels keyed by view and SVO frame:

```json
{
  "left": {
    "1482": {
      "box": [1018, 410, 1098, 592],
      "positive": [[1065, 430], [1042, 555]],
      "negative": [[1018, 410], [1098, 592]]
    }
  },
  "right": {}
}
```

Pass this file with `--prompt-json prompts.json`. Prompt source, SAM repository
commit, configuration, checkpoint path, and checkpoint SHA-256 are recorded in
the HDF5 metadata.

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

#### Two ZED2 cameras, one fixed board

The default `camera_config.yaml` now names both physical ZED2s by serial number:

```yaml
cameras:
  primary: {serial: 20757336}
  oblique: {serial: 26080456}
```

With two entries, the same image-only command opens both cameras and shows a
2-by-2 preview (one camera per row, `LEFT | RIGHT` per row). The single fixed
robot-base ChArUco board must be visible in at least one eye of each camera. The
recorder detects both eyes, keeps the observation with more ChArUco corners, and
uses factory stereo calibration to express right-eye observations in the
left-camera frame. It does not need to be visible simultaneously: each camera
accumulates its own observations of that same stationary board.

```powershell
python -m shape_tracking `
    --camera-config .\camera_config.yaml `
    --registration-config .\registration_config.yaml `
    --resolution HD1080 --fps 30 --preview-fps 10 `
    --image-only --autorecord
```

The session contains `primary_*.svo2`, `oblique_*.svo2`, one frame-index CSV per
camera, and `camera_frame_pairs.csv`. Pairing is timestamp-based because USB
ZED2 cameras do not provide a shared hardware trigger. `registration.json`
stores both camera rigs, and four overlays named
`registration_<rig>_<left|right>.png` verify the result. Offline reconstruction
is still single-camera and does not consume this schema yet.

For bandwidth stability, connect the cameras to separate USB 3 host controllers
when possible. Never depend on USB enumeration order; update the serial numbers
in the YAML if the physical primary/oblique assignments change.

### Fixed camera settings for repeatable segmentation

Auto exposure/gain and auto white balance can change catheter colors and edge
contrast during a trajectory. Tune the ZED2 in ZED Explorer, then pass the
chosen values explicitly to the recorder:

```powershell
python -m shape_tracking `
    --resolution HD1080 --fps 30 --image-only `
    --exposure 25 --gain 15 `
    --white-balance-temperature 5000 `
    --brightness 4 --contrast 4 --saturation 4 --gamma 8 --sharpness 4
```

`--exposure` and `--gain` must be supplied together. Omitting both keeps
AEC/AGC enabled; omitting white-balance temperature keeps auto white balance.
The SDK-supported ranges are checked when the camera opens. Requested and
read-back settings are stored in `session_metadata.json`; reuse the same
values for every dataset intended to share a segmentation model.

The default fixed profile is [`camera_config.yaml`](camera_config.yaml). It
contains the ZED Explorer settings selected for the light box, including manual
exposure/gain and an auto-white-balance warm-up/freeze. Running
`python -m shape_tracking` loads it automatically. Select another profile with:

```powershell
python -m shape_tracking --camera-config .\my_camera_config.yaml
```

Command-line camera options override individual YAML values. Pass
`--camera-config ""` to disable profile loading. The profile also selects the
SVO2 codec; override it with, for example,
`--svo-compression H264_LOSSLESS`.

For the ZED2's coarse manual white-balance control, the default profile instead
uses `white_balance_auto_freeze_s: 5.0`: the camera applies all final image
controls, grabs five seconds of unrecorded frames with auto white balance, and
then disables auto without writing a manual temperature. The console and
`session_metadata.json` record the warm-up duration, frame count, temperature
read-back, and final auto/manual state. Keep the light box and scene stationary
during startup.

The recorder checks both eyes after every freeze attempt. A gross single-eye
color cast or clipping (such as the intermittent all-green right view) causes
another auto-WB convergence attempt. After the configured retry count it falls
back to continuous auto white balance and verifies both eyes again. SVO start is
blocked until the stereo pair passes, both views are shown side by side, and
three consecutive failures during recording stop the SVO with details saved in
`session_metadata.json`. Configure retries with
`white_balance_auto_freeze_retries` (default 2).

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

## Install (Python 3.10 or 3.11 with a matching ZED wheel)

The Ubuntu 22.04 catheter stack uses the shared Python 3.10 `cr-venv` so it can
also access ROS 2 Humble packages:

```bash
# 1) activate the shared catheter/ROS environment
source /home/chen-lab/Yifan/cr-venv/bin/activate

# 2) editable-install the package + its deps (numpy, opencv-contrib-python)
#    Do NOT also have plain opencv-python installed — it shadows the aruco module.
python -m pip install -e /home/chen-lab/Yifan/catheter-shape-tracking

# 3) after installing ZED SDK, install pyzed into THIS venv. pyzed is not on
#    PyPI; the SDK helper detects the active CPython 3.10 interpreter and obtains
#    the matching wheel.
cd /usr/local/zed
python get_python_api.py
```

Verify:
```bash
python -c "import cv2, cv2.aruco, numpy, pyzed.sl; print('ok', cv2.__version__)"
```

### Reusing from another project (consumers on Python 3.10 or 3.11)
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
