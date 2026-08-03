"""Croom-2010-style stereo reconstruction of the catheter centerline.

Given ordered 2D centerlines from the left and right rectified views (see
segmentation.py), correspond points by shared normalized arc length between the
two physically-matched endpoints (sheath-collar base and gold-tip), triangulate
with the ZED rectified stereo model, and fit a smooth 3D curve. Optionally
transform the result from the camera frame into the ChArUco base frame.

Rectified ZED model: both cameras share intrinsics K; the right camera is
translated by the baseline along -x. Corresponding image rows are equal
(epipolar lines are horizontal), which we use as a correspondence-quality check.
"""

import cv2
import numpy as np

from .segmentation import resample_arclength


def rectified_projection_matrices(K, baseline_m):
    """P_left, P_right for a rectified stereo pair (right camera at -baseline x)."""
    P_left = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    t = np.array([[-baseline_m], [0.0], [0.0]])
    P_right = K @ np.hstack([np.eye(3), t])
    return P_left, P_right


def triangulate(pts_left, pts_right, K, baseline_m):
    """Triangulate matched (x,y) pairs. Returns (N,3) in the left-camera frame (m)."""
    P_left, P_right = rectified_projection_matrices(K, baseline_m)
    Xh = cv2.triangulatePoints(P_left, P_right,
                               pts_left.T.astype(np.float64),
                               pts_right.T.astype(np.float64))
    return (Xh[:3] / Xh[3]).T


def epipolar_correspond(pts_left, right_points):
    """Match arc-length-sampled left points to the right centerline.

    In a rectified pair, the epipolar line of a left point (x_L, y_L) is the
    horizontal row y = y_L. We intersect that row with the ordered right
    centerline polyline and, where the curve crosses a row more than once,
    pick the crossing whose normalized arc position is closest to the left
    point's (arc-order disambiguation, per Croom 2010). Returns (pts_right, ok)
    where ok masks left points that had no valid right crossing.
    """
    L = np.asarray(pts_left, dtype=np.float64)
    R = np.asarray(right_points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(R, axis=0), axis=1)
    s_r = np.concatenate([[0], np.cumsum(seg)])
    s_r = s_r / s_r[-1] if s_r[-1] > 0 else s_r
    n = len(L)
    out = np.full((n, 2), np.nan)
    for i in range(n):
        x_l, y_l = L[i]
        a_l = i / max(n - 1, 1)
        best_x, best_cost = None, np.inf
        for j in range(len(R) - 1):
            y0, y1 = R[j, 1], R[j + 1, 1]
            if y0 != y1 and (y0 - y_l) * (y1 - y_l) <= 0:      # row crossed
                tt = (y_l - y0) / (y1 - y0)
                x_r = R[j, 0] + tt * (R[j + 1, 0] - R[j, 0])
                a_r = s_r[j] + tt * (s_r[j + 1] - s_r[j])
                cost = abs(a_r - a_l)
                if cost < best_cost:
                    best_cost, best_x = cost, x_r
        if best_x is not None:
            out[i] = (best_x, y_l)
    return out, ~np.isnan(out[:, 0])


def reconstruct(cl_left, cl_right, K, baseline_m, n_samples=60, smooth_window=5):
    """Full stereo reconstruction from two Centerline objects.

    Left centerline is sampled by arc length; each sample is corresponded to the
    right centerline along its epipolar line (equal row). Returns triangulated
    3D points plus a reprojection-error metric.
    """
    p_left = resample_arclength(cl_left.points, n_samples, smooth_window)
    right_dense = resample_arclength(cl_right.points, max(4 * n_samples, 200),
                                     smooth_window)
    p_right, ok = epipolar_correspond(p_left, right_dense)
    p_left, p_right = p_left[ok], p_right[ok]
    X_cam = triangulate(p_left, p_right, K, baseline_m)
    reproj = reprojection_error(X_cam, p_left, p_right, K, baseline_m)
    return {
        "points_cam": X_cam,             # (N,3) meters, left-camera frame
        "pts_left": p_left,
        "pts_right": p_right,
        "arc_length_m": arc_length(X_cam),
        "reproj_error_px": reproj,
        "n_matched": int(ok.sum()),
    }


def reprojection_error(X_cam, pts_left, pts_right, K, baseline_m):
    """Mean 2D reprojection error over both views (px)."""
    P_left, P_right = rectified_projection_matrices(K, baseline_m)
    Xh = np.hstack([X_cam, np.ones((len(X_cam), 1))])
    errs = []
    for P, pts in ((P_left, pts_left), (P_right, pts_right)):
        p = (P @ Xh.T).T
        p = p[:, :2] / p[:, 2:3]
        errs.append(np.linalg.norm(p - pts, axis=1))
    return float(np.mean(np.concatenate(errs)))


def _bezier(ctrl, t):
    """Cubic Bezier at parameters t (array). ctrl: (4,3)."""
    t = t[:, None]
    b = ((1 - t) ** 3 * ctrl[0] + 3 * (1 - t) ** 2 * t * ctrl[1]
         + 3 * (1 - t) * t ** 2 * ctrl[2] + t ** 3 * ctrl[3])
    return b


def fit_curve_stereo(cl_left, cl_right, K, baseline_m, n_samples=60,
                     reg_weight=100.0):
    """Robust stereo curve fit for a near-baseline-parallel catheter.

    Anchors a cubic Bezier at the directly-triangulated base and tip (both are
    well-conditioned point features at matching rows), then optimizes the two
    interior control points to minimize the distance between the curve's
    projection and each 2D centerline in both views. No per-point disparity, so
    it is robust to the horizontal-curve degeneracy.

    reg_weight adds a Tikhonov penalty pulling the interior control points'
    DEPTH (Z) toward the straight base->tip chord (units: residual per metre of
    Z deviation). Only Z is penalized, so the well-observed in-plane (X,Y) bend
    is left to the 2D data while the weakly-observed depth wiggle is suppressed.
    Set 0 to disable.

    Returns a dict with 'points_cam', 'control_points', 'arc_length_m',
    'reproj_error_px' (mean nearest-centerline distance over both views, the
    penalty excluded), plus 'base_cam'/'tip_cam'.
    """
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree

    P0 = triangulate(cl_left.points[:1], cl_right.points[:1], K, baseline_m)[0]
    P3 = triangulate(cl_left.points[-1:], cl_right.points[-1:], K, baseline_m)[0]
    P_left, P_right = rectified_projection_matrices(K, baseline_m)
    tree_l = cKDTree(cl_left.points)
    tree_r = cKDTree(cl_right.points)
    ts = np.linspace(0, 1, n_samples)
    lin1, lin2 = P0 + (P3 - P0) / 3, P0 + 2 * (P3 - P0) / 3

    def project(P, X):
        Xh = np.hstack([X, np.ones((len(X), 1))])
        p = (P @ Xh.T).T
        return p[:, :2] / p[:, 2:3]

    def data_residuals(params):
        ctrl = np.vstack([P0, params[:3], params[3:], P3])
        X = _bezier(ctrl, ts)
        dl, _ = tree_l.query(project(P_left, X))
        dr, _ = tree_r.query(project(P_right, X))
        return np.concatenate([dl, dr])

    def residuals(params):
        # penalize only the depth (Z) deviation from the chord
        reg = reg_weight * np.array([params[2] - lin1[2], params[5] - lin2[2]])
        return np.concatenate([data_residuals(params), reg])

    init = np.concatenate([lin1, lin2])
    sol = least_squares(residuals, init, method="lm", max_nfev=600)
    ctrl = np.vstack([P0, sol.x[:3], sol.x[3:], P3])
    X = _bezier(ctrl, ts)
    return {
        "points_cam": X,
        "control_points": ctrl,
        "arc_length_m": arc_length(X),
        "reproj_error_px": float(np.mean(np.abs(data_residuals(sol.x)))),
        "base_cam": P0,
        "tip_cam": P3,
    }


def _reproj_to_centerlines(X_cam, cl_left, cl_right, K, baseline_m):
    """Mean nearest-centerline distance over both views (px)."""
    from scipy.spatial import cKDTree
    P_left, P_right = rectified_projection_matrices(K, baseline_m)
    tree_l = cKDTree(cl_left.points)
    tree_r = cKDTree(cl_right.points)

    def project(P, X):
        Xh = np.hstack([X, np.ones((len(X), 1))])
        p = (P @ Xh.T).T
        return p[:, :2] / p[:, 2:3]

    dl, _ = tree_l.query(project(P_left, X_cam))
    dr, _ = tree_r.query(project(P_right, X_cam))
    return float(np.mean(np.concatenate([dl, dr])))


def smooth_polyline_2d(points, n_out, smooth=None):
    """Resample a 2D polyline, with an explicitly requested optional spline.

    The default is piecewise-linear arc-length resampling, which preserves sharp
    turns in the mask skeleton. A positive `smooth` opts into the former cubic
    FITPACK smoothing spline and is retained for explicit experiments.
    """
    p = np.asarray(points, dtype=np.float64)
    keep = np.r_[True, np.any(np.diff(p, axis=0) != 0, axis=1)]
    p = p[keep]
    if len(p) < 2:
        return p
    segment = np.linalg.norm(np.diff(p, axis=0), axis=1)
    source_s = np.concatenate([[0.0], np.cumsum(segment)])
    if source_s[-1] <= 1e-9:
        return np.repeat(p[:1], int(n_out), axis=0)
    target_s = np.linspace(0.0, source_s[-1], int(n_out))
    if smooth is None or float(smooth) <= 0.0:
        return np.column_stack([
            np.interp(target_s, source_s, p[:, axis]) for axis in range(2)])

    from scipy.interpolate import splprep, splev
    tck, _ = splprep(
        [p[:, 0], p[:, 1]], u=source_s / source_s[-1],
        s=float(smooth), k=min(3, len(p) - 1))
    x, y = splev(target_s / source_s[-1], tck)
    return np.column_stack([x, y])


def reconstruct_disparity(cl_left, cl_right, K, baseline_m, n_samples=80,
                          disp_order=2, smooth2d=None):
    """Reconstruct by lifting the smoothed 2D left centerline with a disparity(s).

    The in-plane shape (u, v) is the left centerline (spline-smoothed to remove
    jitter, sharp turns preserved). Disparity is measured by EPIPOLAR
    correspondence -- for each left point (u, v), the matching right point lies on
    the same rectified row v -- so the right reprojection lands on the right
    centerline by construction (arc-length correspondence biased it before). The
    per-point disparity is then fit by a low-order polynomial (disp_order) to
    regularize the weakly-observed depth. 3D: Z=fx*B/d, X=(u-cx)Z/fx, Y=(v-cy)Z/fy.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pL = smooth_polyline_2d(cl_left.points, n_samples, smooth2d)
    right_dense = smooth_polyline_2d(cl_right.points, max(4 * n_samples, 300),
                                     smooth2d)
    pR, ok = epipolar_correspond(pL, right_dense)          # same-row matching
    s = np.linspace(0, 1, n_samples)
    disp_raw = pL[:, 0] - pR[:, 0]
    if ok.sum() < disp_order + 2:                          # too flat: fall back
        pR_arc = smooth_polyline_2d(cl_right.points, n_samples, smooth2d)
        disp_raw = pL[:, 0] - pR_arc[:, 0]
        ok = np.ones(n_samples, dtype=bool)
    order = int(min(disp_order, ok.sum() - 1))
    disp = np.polyval(np.polyfit(s[ok], disp_raw[ok], order), s)
    disp = np.clip(disp, 1e-3, None)
    Z = fx * baseline_m / disp
    X = (pL[:, 0] - cx) * Z / fx
    Y = (pL[:, 1] - cy) * Z / fy
    X_cam = np.column_stack([X, Y, Z])
    return {
        "points_cam": X_cam,
        "arc_length_m": arc_length(X_cam),
        "reproj_error_px": _reproj_to_centerlines(X_cam, cl_left, cl_right,
                                                  K, baseline_m),
        "n_epipolar": int(ok.sum()),
        "disp_raw": disp_raw,
        "disp_fit": disp,
        "base_cam": X_cam[0],
        "tip_cam": X_cam[-1],
    }


def fit_spline_3d(X, n_out=None, smooth=1e-6):
    """Fit a cubic smoothing spline through 3D points, resampled by arc length."""
    from scipy.interpolate import splprep, splev
    n_out = n_out or len(X)
    tck, _ = splprep(X.T, s=smooth, k=min(3, len(X) - 1))
    Y = np.array(splev(np.linspace(0, 1, n_out), tck)).T
    return Y


def arc_length(X):
    return float(np.sum(np.linalg.norm(np.diff(X, axis=0), axis=1)))


def to_board_frame(X_cam, rvec, tvec):
    """Transform camera-frame points into the ChArUco board frame.

    T_cam_board = [R|t] maps board->camera, so board = R^T (cam - t).
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    return (R.T @ (X_cam - t).T).T
