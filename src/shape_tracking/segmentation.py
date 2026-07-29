"""Segment the thin catheter and extract an ordered 2D centerline (base -> tip).

Front end shared by both reference pipelines (Croom 2010 stereo-SOM and Lu 2023
differentiable rendering both start from a binary mask thinned to a centerline).

Target = the thin dark-blue catheter that extrudes from the sheath collar out to
the gold EM tip. The gold tip is not blue, so the centerline ends at the last
blue pixel; anchor the true tip from the EM sensor downstream.

Method: ROI crop -> blue color threshold (HSV) -> morphology -> largest blob ->
skeletonize (Zhang-Suen, scikit-image) -> longest geodesic path through the
skeleton (rejects blobs/spurs like the sheath or a bracket, which are not the
dominant thin structure) -> order base->tip.
"""

from collections import deque

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
except ImportError as e:  # pragma: no cover
    raise ImportError("scikit-image required: pip install scikit-image") from e

# Blue range tuned on the thin catheter (saturated royal/navy blue). The pale
# sheath collar and translucent lead filaments fall outside this range.
DEFAULT_HSV_LO = (95, 90, 20)
DEFAULT_HSV_HI = (135, 255, 235)

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Centerline:
    """Ordered 2D catheter centerline in full-image pixel coordinates."""

    __slots__ = ("points", "roi", "mask", "radius_px")

    def __init__(self, points, roi, mask, radius_px):
        self.points = points          # (N,2) float, [x, y], ordered base->tip
        self.roi = roi                # (x, y, w, h) used
        self.mask = mask              # uint8 mask over the ROI
        self.radius_px = radius_px    # median medial-axis radius (px)

    def __len__(self):
        return len(self.points)


def select_roi(bgr, win="select catheter ROI (Enter=confirm)"):
    """Interactive box selection. Returns (x, y, w, h). Needs a GUI."""
    roi = cv2.selectROI(win, bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)
    return tuple(int(v) for v in roi)


def segment_mask(bgr, roi, hsv_lo=DEFAULT_HSV_LO, hsv_hi=DEFAULT_HSV_HI,
                 close_ksize=7, open_ksize=3, close_iter=2,
                 base_point=None, min_area=40):
    """Blue-threshold the ROI, clean up, keep one component.

    Without base_point: keep the largest component. With base_point (the robot
    base origin in FULL-IMAGE pixels): keep the component with the pixel nearest
    the base -- the catheter emerges from the base, so this rejects other blue
    objects in the ROI (e.g. blue rail clamps) even when they are larger.
    """
    x, y, w, h = roi
    sub = bgr[y:y + h, x:x + w]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    if close_ksize:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize,) * 2),
            iterations=close_iter)
    if open_ksize:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize,) * 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    if base_point is None:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if not cand:
            cand = list(range(1, n))
        bx, by = base_point[0] - x, base_point[1] - y      # ROI coords
        sel = np.isin(lab, cand)
        ys, xs = np.where(sel)
        j = int(np.argmin((xs - bx) ** 2 + (ys - by) ** 2))
        keep = int(lab[ys[j], xs[j]])
    return np.uint8(lab == keep) * 255


def _longest_path(skel):
    """Ordered (row,col) pixels of the longest geodesic path in a skeleton."""
    pts = np.argwhere(skel)
    if len(pts) < 2:
        return pts
    idx = {(r, c): i for i, (r, c) in enumerate(pts)}
    adj = [[] for _ in range(len(pts))]
    for i, (r, c) in enumerate(pts):
        for dr, dc in _NB:
            j = idx.get((r + dr, c + dc))
            if j is not None:
                adj[i].append(j)

    def bfs(src):
        dist = [-1] * len(pts)
        par = [-1] * len(pts)
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
        return int(np.argmax(dist)), par

    a, _ = bfs(0)
    b, par = bfs(a)
    path = []
    u = b
    while u != -1:
        path.append(u)
        u = par[u]
    return pts[path[::-1]]


def _order_base_tip(cl_rc, base_hint):
    """Flip the (row,col) path so index 0 is nearest the base side."""
    if len(cl_rc) < 2:
        return cl_rc
    r0, c0 = cl_rc[0]
    r1, c1 = cl_rc[-1]
    if base_hint == "left":
        flip = c0 > c1
    elif base_hint == "right":
        flip = c0 < c1
    elif base_hint == "top":
        flip = r0 > r1
    elif base_hint == "bottom":
        flip = r0 < r1
    else:
        raise ValueError(f"base_hint must be left/right/top/bottom, got {base_hint}")
    return cl_rc[::-1] if flip else cl_rc


def _trim_by_radius(cl_rc, dt, max_radius_px, smooth=7):
    """Keep only the thin segment contiguous with the tip (last index).

    Walks from the tip backward and stops at the first point whose (smoothed)
    medial radius exceeds max_radius_px, cutting off a thicker sheath. Returns
    the truncated (row,col) array (new base = the sheath/catheter transition).
    """
    r = dt[cl_rc[:, 0], cl_rc[:, 1]]
    if len(r) > smooth:
        r = np.convolve(r, np.ones(smooth) / smooth, "same")
    start = 0
    for i in range(len(r) - 1, -1, -1):
        if r[i] > max_radius_px:
            start = i + 1
            break
    return cl_rc[start:]


def extract_centerline(bgr, roi, base_hint="left",
                       hsv_lo=DEFAULT_HSV_LO, hsv_hi=DEFAULT_HSV_HI,
                       max_radius_px=None, base_point=None, **kw):
    """Full segmentation: returns a Centerline (base->tip) or None if not found.

    base_point: optional (x, y) in FULL-IMAGE pixels (e.g. the projected robot
    base origin). If given, the centerline is ordered so index 0 is the end
    nearest that point, overriding base_hint -- so the tip (far, thin end) is
    always last for the max_radius trim.

    max_radius_px: if set, trims a thicker sheath by keeping only the thin
    segment adjacent to the tip. Typical value is a bit above the catheter's
    medial radius (catheter ~4 px, sheath ~6-8 px here).
    """
    x, y, w, h = roi
    mask = segment_mask(bgr, roi, hsv_lo, hsv_hi, base_point=base_point, **kw)
    if not mask.any():
        return None
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skel = skeletonize(mask > 0)
    if skel.sum() < 2:
        return None
    cl_rc = _longest_path(skel)
    if base_point is not None:
        bp = (base_point[1] - y, base_point[0] - x)             # (row, col) in ROI
        d0 = (cl_rc[0, 0] - bp[0]) ** 2 + (cl_rc[0, 1] - bp[1]) ** 2
        d1 = (cl_rc[-1, 0] - bp[0]) ** 2 + (cl_rc[-1, 1] - bp[1]) ** 2
        if d1 < d0:
            cl_rc = cl_rc[::-1]
    else:
        cl_rc = _order_base_tip(cl_rc, base_hint)
    if max_radius_px is not None:
        cl_rc = _trim_by_radius(cl_rc, dt, max_radius_px)
        if len(cl_rc) < 2:
            return None
    radius = float(np.median(dt[cl_rc[:, 0], cl_rc[:, 1]]))
    # (row,col) ROI -> (x,y) full image
    pts_xy = np.column_stack([cl_rc[:, 1] + x, cl_rc[:, 0] + y]).astype(np.float64)
    return Centerline(pts_xy, roi, mask, radius)


def resample_arclength(points, n_samples, smooth_window=5):
    """Smooth then resample a polyline to n uniformly arc-spaced points."""
    p = np.asarray(points, dtype=np.float64)
    if smooth_window > 1 and len(p) >= smooth_window:
        k = np.ones(smooth_window) / smooth_window
        p = np.column_stack([np.convolve(p[:, 0], k, "same"),
                             np.convolve(p[:, 1], k, "same")])
        p[0], p[-1] = points[0], points[-1]           # keep endpoints fixed
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return p
    su = np.linspace(0, s[-1], n_samples)
    return np.column_stack([np.interp(su, s, p[:, 0]),
                            np.interp(su, s, p[:, 1])])


def draw_centerline(bgr, cl, thickness=1):
    """Return a copy of bgr with the mask tinted and centerline drawn base->tip."""
    vis = bgr.copy()
    x, y, w, h = cl.roi
    sub = vis[y:y + h, x:x + w]
    sub[cl.mask > 0] = (0.6 * sub[cl.mask > 0] +
                        np.array([0, 90, 0]) * 0.4).astype(np.uint8)
    n = len(cl.points)
    for k, (px, py) in enumerate(cl.points):
        t = k / max(n - 1, 1)
        cv2.circle(vis, (int(px), int(py)), thickness,
                   (int(255 * (1 - t)), 0, int(255 * t)), -1)
    cv2.circle(vis, tuple(cl.points[0].astype(int)), 4, (255, 255, 0), 1)   # base
    cv2.circle(vis, tuple(cl.points[-1].astype(int)), 4, (0, 255, 255), 1)  # tip
    return vis
