"""Detect ChArUco boards and estimate camera->board pose.

Pure numpy + OpenCV (no ZED SDK), so this is importable and unit-testable from
any project's venv without the capture hardware stack.
"""

import cv2
import numpy as np

from .boards import MARKERS_PER_BOARD


class BoardPose:
    """Result of detecting one board in one frame."""

    __slots__ = ("index", "offset", "n_corners", "rvec", "tvec",
                 "charuco_corners", "charuco_ids", "marker_corners", "marker_ids")

    def __init__(self, index, offset, n_corners, rvec, tvec,
                 charuco_corners, charuco_ids, marker_corners, marker_ids):
        self.index = index
        self.offset = offset
        self.n_corners = n_corners
        self.rvec = rvec              # (3,1) or None if pose not solved
        self.tvec = tvec             # (3,1) meters, or None
        self.charuco_corners = charuco_corners
        self.charuco_ids = charuco_ids
        self.marker_corners = marker_corners
        self.marker_ids = marker_ids

    @property
    def has_pose(self):
        return self.rvec is not None and self.tvec is not None


def detect_boards(gray, boards, K, dist, min_corners=4):
    """Detect each board in a grayscale image and estimate its pose.

    Returns (results, seen_ids) where results is a list of BoardPose (one per
    board that had any markers) and seen_ids is the sorted set of all detected
    marker ids (handy for verifying BOARD_ID_OFFSETS at runtime).
    """
    results = []
    seen_ids = set()
    for b in boards:
        ch_corners, ch_ids, mk_corners, mk_ids = b.detector.detectBoard(gray)
        if mk_ids is None or len(mk_ids) == 0:
            continue
        seen_ids.update(int(i) for i in mk_ids.flatten())

        rvec = tvec = None
        n_corners = 0 if ch_ids is None else len(ch_ids)
        if ch_ids is not None and n_corners >= min_corners:
            obj_pts, img_pts = b.board.matchImagePoints(ch_corners, ch_ids)
            if obj_pts is not None and len(obj_pts) >= min_corners:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    rvec = tvec = None
        results.append(BoardPose(b.index, b.offset, n_corners, rvec, tvec,
                                 ch_corners, ch_ids, mk_corners, mk_ids))
    return results, sorted(seen_ids)


def draw_pose(img, res, K, dist, axis_len, draw_axes=True):
    """Draw detected markers/corners and pose axes for one BoardPose in place."""
    if res.marker_ids is not None and len(res.marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(img, res.marker_corners, res.marker_ids)
    if res.charuco_ids is not None and len(res.charuco_ids) >= 4:
        # OpenCV 5.0's detectBoard returns corners as (N,2); the draw helper wants
        # (N,1,2) so corners.total()==ids.total(). Reshape for drawing only.
        cc = np.asarray(res.charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
        ci = np.asarray(res.charuco_ids, dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedCornersCharuco(img, cc, ci)
    if draw_axes and res.has_pose:
        cv2.drawFrameAxes(img, K, dist, res.rvec, res.tvec, axis_len)
