"""Offline driver: reconstruct the catheter centerline from a saved stereo pair.

Runs the full pipeline on one left/right frame:
    segment (both views) -> stereo curve fit -> 3D centerline
and writes:
    overlay_left.png / overlay_right.png  (mask + centerline + reprojected 3D fit)
    reconstruction_3d.png                 (matplotlib 3D plot)
    centerline_3d.csv                     (X,Y,Z in mm, camera frame)

Usage:
    python -m shape_tracking.offline --session RECORDINGS\\20260704_173043 \
        --roi-left 840,460,200,105 --roi-right 455,458,420,135
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np

import matplotlib                           # backend chosen in main()

from . import reconstruction as rec
from . import segmentation as seg


def parse_roi(s):
    x, y, w, h = (int(v) for v in s.split(","))
    return (x, y, w, h)


def find_pair(session):
    """Return (left_path, right_path) for the first matching frame in a session."""
    lefts = sorted(glob.glob(os.path.join(session, "left", "*.png")))
    if not lefts:
        raise FileNotFoundError(f"no PNGs in {os.path.join(session, 'left')}")
    lp = lefts[0]
    rp = os.path.join(session, "right", os.path.basename(lp))
    if not os.path.exists(rp):
        raise FileNotFoundError(f"no matching right frame for {os.path.basename(lp)}")
    return lp, rp


def load_calib(session, baseline_arg):
    npz = os.path.join(session, "left_intrinsics.npz")
    d = np.load(npz)
    K = d["K"]
    if baseline_arg is not None:
        baseline = baseline_arg
    elif "baseline_m" in d.files:
        baseline = float(d["baseline_m"])
    else:
        baseline = 0.1198                   # ZED2 SN20757336 fallback (119.8 mm)
    return K, baseline


def draw_legend(img):
    """Draw a translucent legend explaining the overlay markings (BGR)."""
    x0, y0, w, rh = 10, 10, 300, 26
    rows = 4
    panel = img.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + rh * rows + 10), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.5, img, 0.5, 0, img)
    y = y0 + 24
    # 1) mask swatch (green)
    cv2.line(img, (x0 + 10, y - 6), (x0 + 34, y - 6), (0, 170, 0), 5)
    cv2.putText(img, "catheter mask", (x0 + 44, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += rh
    # 2) segmented centerline gradient (blue base -> red tip)
    for i in range(25):
        t = i / 24.0
        cv2.line(img, (x0 + 10 + i, y - 10), (x0 + 10 + i, y - 2),
                 (int(255 * (1 - t)), 0, int(255 * t)), 1)
    cv2.putText(img, "2D centerline (base->tip)", (x0 + 44, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += rh
    # 3) reprojected 3D fit (yellow dots)
    for dx in (10, 18, 26):
        cv2.circle(img, (x0 + dx, y - 6), 2, (0, 255, 255), -1)
    cv2.putText(img, "3D fit reprojected", (x0 + 44, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += rh
    # 4) base / tip endpoint rings (cyan / yellow)
    cv2.circle(img, (x0 + 16, y - 6), 5, (255, 255, 0), 1)
    cv2.circle(img, (x0 + 30, y - 6), 5, (0, 255, 255), 1)
    cv2.putText(img, "base / tip", (x0 + 44, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def overlay_image(full, cl, reproj_pts, out_path, pad=45, scale=3):
    """Draw mask + segmented centerline + reprojected 3D fit, crop, upscale, save."""
    vis = seg.draw_centerline(full, cl)
    for (px, py) in reproj_pts:
        cv2.circle(vis, (int(round(px)), int(round(py))), 1, (0, 255, 255), -1)
    x, y, w, h = cl.roi
    crop = vis[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    crop = cv2.resize(crop, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_NEAREST)
    draw_legend(crop)                        # legend drawn at full scale for crisp text
    cv2.imwrite(out_path, crop)


def plot_3d(X_mm, out_path, arc_len_mm, reproj_px, show=False,
            frame_note="camera frame; Z = depth from camera"):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot(X_mm[:, 0], X_mm[:, 1], X_mm[:, 2], "-", color="tab:blue", lw=2,
            label="catheter centerline")
    ax.scatter(*X_mm[0], color="tab:green", s=60, label="base (collar)")
    ax.scatter(*X_mm[-1], color="tab:red", s=60, label="tip (gold)")
    # equal aspect via a cubic bounding box centered on the data
    c = X_mm.mean(0)
    r = (X_mm.max(0) - X_mm.min(0)).max() / 2 + 5
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Reconstructed catheter — arc {arc_len_mm:.1f} mm, "
                 f"reproj {reproj_px:.2f} px\n({frame_note})")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=18, azim=-60)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    if show:
        print("showing interactive 3D plot — rotate with the mouse, close to finish")
        plt.show()
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="recordings session dir (left/, right/, "
                    "left_intrinsics.npz).")
    ap.add_argument("--left", help="explicit left image (overrides --session frame).")
    ap.add_argument("--right", help="explicit right image.")
    ap.add_argument("--registration", default=None,
                    help="camera_base_registration.npz: auto-crops to the workspace "
                         "ROI, orders base->tip from the registered base, and outputs "
                         "the centerline in the ROBOT BASE frame.")
    ap.add_argument("--roi-left", type=parse_roi, default=None,
                    help="left ROI 'x,y,w,h' (ignored if --registration given).")
    ap.add_argument("--roi-right", type=parse_roi, default=None,
                    help="right ROI 'x,y,w,h' (ignored if --registration given).")
    ap.add_argument("--base-hint", default="left",
                    choices=["left", "right", "top", "bottom"],
                    help="which image side the catheter base is on (used only "
                         "without --registration; default left).")
    ap.add_argument("--max-radius", type=float, default=None,
                    help="trim a thicker sheath: keep only the thin segment at the "
                         "tip with medial radius below this (px). Catheter ~4 px, "
                         "sheath ~6-8 px, so ~5 works. Default off.")
    ap.add_argument("--baseline", type=float, default=None,
                    help="stereo baseline (m); default reads from npz or 0.1198.")
    ap.add_argument("--n-samples", type=int, default=80)
    ap.add_argument("--method", default="disparity",
                    choices=["disparity", "bezier"],
                    help="disparity: lift the exact 2D centerline with a smooth "
                         "disparity profile (preserves sharp in-plane turns, "
                         "regularizes only depth). bezier: global cubic fit. "
                         "Default disparity.")
    ap.add_argument("--disp-order", type=int, default=2,
                    help="polynomial order of the disparity(arc) profile for "
                         "--method disparity (1=monotonic depth, 2=single depth "
                         "bend). Default 2.")
    ap.add_argument("--smooth2d", type=float, default=None,
                    help="2D centerline spline smoothing (scipy splprep s). "
                         "Higher = smoother/less jitter but rounder turns; lower = "
                         "sharper/jitterier. Default ~ number of centerline points.")
    ap.add_argument("--reg", type=float, default=100.0,
                    help="[bezier] depth-smoothness regularization (residual per "
                         "metre of control-point Z deviation from the chord).")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: <session>/reconstruction).")
    ap.add_argument("--show", action="store_true",
                    help="also open an interactive 3D plot (rotate/zoom with the "
                         "mouse). Needs a GUI backend (tkinter/Qt).")
    args = ap.parse_args(argv)

    # Choose the matplotlib backend before pyplot is imported anywhere.
    if args.show:
        for backend in ("TkAgg", "QtAgg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue
        else:
            print("[warn] no interactive backend (Qt/Tk) found; saving PNG only")
            matplotlib.use("Agg")
            args.show = False
    else:
        matplotlib.use("Agg")               # headless: render to file, no GUI

    if args.left and args.right:
        lp, rp = args.left, args.right
        session = args.session or os.path.dirname(os.path.dirname(lp))
    elif args.session:
        lp, rp = find_pair(args.session)
        session = args.session
    else:
        ap.error("provide --session or both --left and --right")

    K, baseline = load_calib(session, args.baseline)
    outdir = args.outdir or os.path.join(session, "reconstruction")
    os.makedirs(outdir, exist_ok=True)

    L = cv2.imread(lp)
    R = cv2.imread(rp)
    print(f"frame: {os.path.basename(lp)}   baseline={baseline*1000:.1f} mm")

    # Registration mode: auto ROI from the workspace box + base-frame output.
    cam_T_base = base_pt_L = base_pt_R = None
    if args.registration:
        from .register import project as ws_project, workspace_roi
        reg = np.load(args.registration)
        # K/ROI/base pose are resolution-specific: the registration and the
        # recording MUST be at the same resolution.
        reg_res = str(reg["resolution"]) if "resolution" in reg.files else "?"
        try:
            sess_res = str(np.load(os.path.join(session,
                                                "left_intrinsics.npz"))["resolution"])
        except Exception:
            sess_res = "?"
        if "?" not in (reg_res, sess_res) and reg_res != sess_res:
            print(f"[WARN] RESOLUTION MISMATCH: registration={reg_res} recording={sess_res}. "
                  "K/ROI/base pose will be wrong — re-register at the recording resolution.")
        cam_T_base = reg["cam_T_base"]
        workspace = {"x": reg["workspace_x"], "y": reg["workspace_y"],
                     "z": reg["workspace_z"], "margin_px": int(reg["workspace_margin_px"])}
        Himg, Wimg = L.shape[:2]
        T_shift = np.eye(4)
        T_shift[0, 3] = -baseline
        cam_T_base_R = T_shift @ cam_T_base
        roiL = workspace_roi(K, cam_T_base, workspace, Wimg, Himg)
        roiR = workspace_roi(K, cam_T_base_R, workspace, Wimg, Himg)
        origin = np.array([[0.0, 0.0, 0.0, 1.0]])
        base_pt_L = ws_project(K, cam_T_base, origin)[0][0]
        base_pt_R = ws_project(K, cam_T_base_R, origin)[0][0]
        print(f"registration: base-frame output; auto ROI L={roiL} R={roiR}")
    else:
        if args.roi_left is None or args.roi_right is None:
            raise SystemExit("provide --registration, or both --roi-left and --roi-right")
        roiL, roiR = args.roi_left, args.roi_right

    clL = seg.extract_centerline(L, roiL, base_hint=args.base_hint,
                                 max_radius_px=args.max_radius, base_point=base_pt_L)
    clR = seg.extract_centerline(R, roiR, base_hint=args.base_hint,
                                 max_radius_px=args.max_radius, base_point=base_pt_R)
    nL = len(clL) if clL is not None else 0
    nR = len(clR) if clR is not None else 0
    if nL < 10 or nR < 10:
        raise SystemExit(
            f"segmentation insufficient (left {nL} pts, right {nR} pts) — the "
            "catheter isn't well segmented in the ROI. Check that it's deployed and "
            "in view, and adjust HSV / --max-radius / ROI.")
    print(f"segmented: left {nL} pts, right {nR} pts")

    if args.method == "disparity":
        out = rec.reconstruct_disparity(clL, clR, K, baseline,
                                        n_samples=args.n_samples,
                                        disp_order=args.disp_order,
                                        smooth2d=args.smooth2d)
    else:
        out = rec.fit_curve_stereo(clL, clR, K, baseline,
                                   n_samples=args.n_samples, reg_weight=args.reg)
    X = out["points_cam"]          # camera frame, metres (used for reprojection)
    print(f"reconstruction: arc {out['arc_length_m']*1000:.1f} mm, "
          f"reproj {out['reproj_error_px']:.2f} px, "
          f"camera depth {X[0,2]*100:.1f}->{X[-1,2]*100:.1f} cm")

    # output frame: robot base if registered, else camera
    if cam_T_base is not None:
        Xh = np.hstack([X, np.ones((len(X), 1))])
        X_out = (np.linalg.inv(cam_T_base) @ Xh.T).T[:, :3]
        frame_note = "robot base frame (+z along catheter)"
        print(f"base frame: base {np.round(X_out[0]*1000,1)} -> "
              f"tip {np.round(X_out[-1]*1000,1)} mm")
    else:
        X_out = X
        frame_note = "camera frame; Z = depth from camera"

    # reproject the fitted 3D curve into both views for the overlays (camera frame)
    P_left, P_right = rec.rectified_projection_matrices(K, baseline)

    def project(P, X):
        Xh = np.hstack([X, np.ones((len(X), 1))])
        p = (P @ Xh.T).T
        return p[:, :2] / p[:, 2:3]

    overlay_image(L, clL, project(P_left, X), os.path.join(outdir, "overlay_left.png"))
    overlay_image(R, clR, project(P_right, X), os.path.join(outdir, "overlay_right.png"))
    plot_3d(X_out * 1000, os.path.join(outdir, "reconstruction_3d.png"),
            out["arc_length_m"] * 1000, out["reproj_error_px"], show=args.show,
            frame_note=frame_note)

    with open(os.path.join(outdir, "centerline_3d.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["idx", "X_mm", "Y_mm", "Z_mm", "frame"])
        frame = "base" if cam_T_base is not None else "camera"
        for i, p in enumerate(X_out * 1000):
            wr.writerow([i, f"{p[0]:.2f}", f"{p[1]:.2f}", f"{p[2]:.2f}", frame])

    print("wrote:", outdir)
    for fn in ("overlay_left.png", "overlay_right.png", "reconstruction_3d.png",
               "centerline_3d.csv"):
        print("  -", fn)


if __name__ == "__main__":
    main()
