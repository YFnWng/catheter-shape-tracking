"""Material-aware 2D catheter segmentation.

The complete shaft is selected as a broad blue component between the projected
robot base and projected EM tip.  A one-change-point model along the ordered
centerline separates the darker proximal segment from the brighter distal
segment. Pale tip housing, beige tape, and black EM wires are intentionally not
part of the blue mask; the 3D stage bridges the last visible blue sample to the
EM-defined tip and labels that bridge as extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize

from .segmentation import Centerline, _longest_path


DEFAULT_BLUE_HSV_LO = (85, 45, 10)
DEFAULT_BLUE_HSV_HI = (145, 255, 255)


@dataclass(frozen=True)
class MaterialCenterline:
    centerline: Centerline
    distal_boundary_index: int
    distal_boundary_fraction: float
    boundary_confidence: float
    boundary_contrast: float
    material_valid: bool
    brightness_profile: np.ndarray

    def __len__(self) -> int:
        return len(self.centerline)

    @property
    def points(self) -> np.ndarray:
        return self.centerline.points

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return self.centerline.roi

    @property
    def mask(self) -> np.ndarray:
        return self.centerline.mask


def _component_distance(
        labels: np.ndarray,
        component: int,
        point_roi: np.ndarray | None) -> float:
    if point_roi is None or not np.all(np.isfinite(point_roi)):
        return 0.0
    ys, xs = np.where(labels == component)
    if len(xs) == 0:
        return float("inf")
    return float(np.sqrt(np.min(
        (xs - point_roi[0]) ** 2 + (ys - point_roi[1]) ** 2)))


def broad_blue_mask(
        bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point: np.ndarray | None = None,
        tip_point: np.ndarray | None = None,
        hsv_lo=DEFAULT_BLUE_HSV_LO,
        hsv_hi=DEFAULT_BLUE_HSV_HI,
        close_ksize: int = 7,
        open_ksize: int = 3,
        min_area: int = 40) -> np.ndarray:
    """Return the blue component best connecting projected base and tip."""
    x, y, w, h = roi
    sub = bgr[y:y + h, x:x + w]
    if sub.size == 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=np.uint8)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.asarray(hsv_lo, dtype=np.uint8),
        np.asarray(hsv_hi, dtype=np.uint8))
    if close_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if open_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    candidates = [
        component for component in range(1, count)
        if stats[component, cv2.CC_STAT_AREA] >= min_area]
    if not candidates:
        candidates = list(range(1, count))
    base_roi = None if base_point is None else np.asarray(
        [base_point[0] - x, base_point[1] - y], dtype=np.float64)
    tip_roi = None if tip_point is None else np.asarray(
        [tip_point[0] - x, tip_point[1] - y], dtype=np.float64)
    scores = {}
    for component in candidates:
        distance = (
            _component_distance(labels, component, base_roi)
            + _component_distance(labels, component, tip_roi))
        area = float(stats[component, cv2.CC_STAT_AREA])
        scores[component] = distance - 0.02 * np.sqrt(area)
    selected = min(scores, key=scores.get)
    return np.uint8(labels == selected) * 255


def _order_with_anchors(
        path_rc: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point: np.ndarray | None,
        tip_point: np.ndarray | None) -> np.ndarray:
    if len(path_rc) < 2:
        return path_rc
    x, y, _, _ = roi
    endpoints_xy = np.array([
        [path_rc[0, 1] + x, path_rc[0, 0] + y],
        [path_rc[-1, 1] + x, path_rc[-1, 0] + y],
    ], dtype=np.float64)
    if base_point is not None and tip_point is not None:
        base = np.asarray(base_point, dtype=np.float64)
        tip = np.asarray(tip_point, dtype=np.float64)
        forward = (
            np.linalg.norm(endpoints_xy[0] - base)
            + np.linalg.norm(endpoints_xy[1] - tip))
        reverse = (
            np.linalg.norm(endpoints_xy[1] - base)
            + np.linalg.norm(endpoints_xy[0] - tip))
        return path_rc if forward <= reverse else path_rc[::-1]
    if base_point is not None:
        base = np.asarray(base_point, dtype=np.float64)
        return (path_rc if np.linalg.norm(endpoints_xy[0] - base)
                <= np.linalg.norm(endpoints_xy[1] - base)
                else path_rc[::-1])
    return path_rc


def _sample_brightness(bgr: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Sample a robust blue brightness profile along a centerline."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]
    values = []
    for px, py in points_xy:
        cx = int(np.clip(round(px), 0, w - 1))
        cy = int(np.clip(round(py), 0, h - 1))
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        lightness = float(np.median(lab[y0:y1, x0:x1, 0]))
        value = float(np.median(hsv[y0:y1, x0:x1, 2]))
        values.append(0.65 * lightness + 0.35 * value)
    profile = np.asarray(values, dtype=np.float64)
    if len(profile) >= 7:
        profile = gaussian_filter1d(profile, sigma=3.0, mode="nearest")
    return profile


def detect_distal_boundary(
        brightness: np.ndarray,
        min_fraction: float = 0.12,
        max_fraction: float = 0.88,
        min_contrast: float = 4.0,
        min_confidence: float = 0.35
) -> tuple[int, float, float, bool]:
    """Fit one dark-to-bright change point along a base-to-tip profile."""
    values = np.asarray(brightness, dtype=np.float64)
    n = len(values)
    lo = max(2, int(np.ceil(min_fraction * n)))
    hi = min(n - 2, int(np.floor(max_fraction * n)))
    if n < 8 or hi <= lo:
        return max(0, n // 2), 0.0, 0.0, False
    candidates = []
    for index in range(lo, hi + 1):
        proximal = values[:index]
        distal = values[index:]
        contrast = float(np.mean(distal) - np.mean(proximal))
        balance = np.sqrt(len(proximal) * len(distal) / n)
        candidates.append((contrast * balance, index, contrast))
    _, index, contrast = max(candidates, key=lambda item: item[0])
    proximal = values[:index]
    distal = values[index:]
    pooled = float(np.sqrt(
        (np.var(proximal) + np.var(distal)) / 2.0))
    confidence = max(0.0, contrast) / max(pooled, 1.0)
    valid = contrast >= min_contrast and confidence >= min_confidence
    return index, confidence, contrast, bool(valid)


def extract_material_centerline(
        bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point: np.ndarray,
        tip_point: np.ndarray,
        hsv_lo=DEFAULT_BLUE_HSV_LO,
        hsv_hi=DEFAULT_BLUE_HSV_HI,
        min_area: int = 40) -> MaterialCenterline | None:
    """Extract the full blue shaft and its proximal/distal transition."""
    x, y, _, _ = roi
    mask = broad_blue_mask(
        bgr, roi, base_point=base_point, tip_point=tip_point,
        hsv_lo=hsv_lo, hsv_hi=hsv_hi, min_area=min_area)
    if not np.any(mask):
        return None
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skeleton = skeletonize(mask > 0)
    if int(np.sum(skeleton)) < 2:
        return None
    path_rc = _longest_path(skeleton)
    path_rc = _order_with_anchors(path_rc, roi, base_point, tip_point)
    points = np.column_stack([
        path_rc[:, 1] + x, path_rc[:, 0] + y]).astype(np.float64)
    brightness = _sample_brightness(bgr, points)
    boundary, confidence, contrast, valid = detect_distal_boundary(brightness)
    radius = float(np.median(distance[path_rc[:, 0], path_rc[:, 1]]))
    centerline = Centerline(points, roi, mask, radius)
    return MaterialCenterline(
        centerline=centerline,
        distal_boundary_index=int(boundary),
        distal_boundary_fraction=float(boundary / max(len(points) - 1, 1)),
        boundary_confidence=float(confidence),
        boundary_contrast=float(contrast),
        material_valid=valid,
        brightness_profile=brightness,
    )


def draw_material_overlay(
        image: np.ndarray,
        material: MaterialCenterline,
        projected_shape: np.ndarray | None = None,
        projected_tip: np.ndarray | None = None,
        projected_tip_direction: np.ndarray | None = None,
        projected_boundary: np.ndarray | None = None,
        quality_text: str = "") -> np.ndarray:
    """Draw proximal/distal segmentation and optional reconstructed shape."""
    output = image.copy()
    x, y, w, h = material.roi
    sub = output[y:y + h, x:x + w]
    selected = material.mask > 0
    sub[selected] = (
        0.7 * sub[selected] + 0.3 * np.array([0, 100, 0])
    ).astype(np.uint8)
    # The translucent fill is difficult to see when the correct mask is only a
    # few pixels wide and is covered by the centerline. An outline makes mask
    # presence and gross mask errors inspectable in both stereo views.
    contours, _ = cv2.findContours(
        np.uint8(selected) * 255, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sub, contours, -1, (0, 180, 0), 1, cv2.LINE_AA)
    if projected_shape is not None:
        curve = np.asarray(projected_shape, dtype=np.float64)
        finite = np.all(np.isfinite(curve), axis=1)
        if np.count_nonzero(finite) >= 2:
            curve_i = np.rint(curve[finite]).astype(np.int32)
            # A wider yellow underlay leaves the measured material centerline
            # visible as a colored core when the two curves agree.
            cv2.polylines(output, [curve_i], False, (0, 255, 255), 5,
                          cv2.LINE_AA)
    points = material.points.astype(np.int32)
    boundary = material.distal_boundary_index
    if boundary >= 2:
        cv2.polylines(output, [points[:boundary]], False, (110, 40, 0), 2,
                      cv2.LINE_AA)
    if len(points) - boundary >= 2:
        cv2.polylines(output, [points[boundary:]], False, (255, 160, 20), 2,
                      cv2.LINE_AA)
    boundary_point = (
        points[boundary] if projected_boundary is None
        else np.rint(projected_boundary).astype(np.int32))
    cv2.circle(output, tuple(boundary_point), 6, (0, 255, 255), 2,
               cv2.LINE_AA)
    if projected_tip is not None and np.all(np.isfinite(projected_tip)):
        tip = tuple(np.rint(projected_tip).astype(int))
        cv2.circle(output, tip, 7, (0, 0, 255), 2, cv2.LINE_AA)
        if (projected_tip_direction is not None
                and np.all(np.isfinite(projected_tip_direction))):
            end = tuple(np.rint(projected_tip_direction).astype(int))
            cv2.arrowedLine(output, tip, end, (0, 0, 255), 2, cv2.LINE_AA,
                            tipLength=0.25)
    if quality_text:
        cv2.putText(output, quality_text, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3,
                    cv2.LINE_AA)
        cv2.putText(output, quality_text, (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 1,
                    cv2.LINE_AA)
    return output
