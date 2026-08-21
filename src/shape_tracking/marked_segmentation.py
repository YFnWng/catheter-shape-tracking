"""Color-first segmentation for catheters carrying ordered red ring markers.

The light-box acquisition has a nearly achromatic background.  This backend
therefore treats chroma as the foreground observation and uses the registered
base only to select the catheter component.  Marker spacing participates only
in the *identity score* below; it is never imposed on the reconstructed curve.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter
from skimage.feature import peak_local_max
from skimage.graph import route_through_array

from .geometry import cumulative_arclength
from .sam_segmentation import (
    PromptSet,
    SamMaterialResult,
)
from .materials import MaterialCenterline
from .segmentation import Centerline, resample_arclength


EXPECTED_MARKER_COORDINATES_MM = np.array([0.0, 10.0, 30.0, 57.0])


def _catheter_color_features(
        bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous paint likelihood and reusable red-ring support.

    Saturation alone is a poor discriminator at black/white calibration-board
    edges: demosaicing and chromatic aberration can give a nominally grey edge
    moderate saturation.  The catheter paint instead has a large, correctly
    directed channel contrast.  Keeping this score continuous also lets path
    extraction prefer the middle of the paint over a weak colored fringe.
    """
    image = np.asarray(bgr, dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    blue, green, red = [
        image[:, :, channel].astype(np.float32) for channel in range(3)]

    blue_contrast = blue - 0.55 * green - 0.45 * red
    red_contrast = red - 0.60 * green - 0.40 * blue
    yellow_contrast = np.minimum(red, green) - blue
    blue_score = np.clip((blue_contrast - 8.0) / 72.0, 0.0, 1.0)
    red_score = np.clip((red_contrast - 10.0) / 90.0, 0.0, 1.0)
    yellow_score = np.clip((yellow_contrast - 8.0) / 90.0, 0.0, 1.0)

    blue_score *= ((hue >= 72) & (hue <= 142))
    red_score *= ((hue <= 22) | (hue >= 150))
    yellow_score *= ((hue >= 12) & (hue <= 48))
    # Lab chroma is small for both black and white, and unlike HSV saturation
    # it does not explode when a dark pixel has a few counts of channel noise.
    # The channel-direction term then distinguishes the intended blue/cyan,
    # red/magenta, and yellow paints from weak chromatic aberration.
    likelihood = np.maximum.reduce([blue_score, red_score, yellow_score])
    likelihood *= np.clip((lab_chroma - 7.0) / 28.0, 0.0, 1.0)
    likelihood *= np.clip((value - 0.055) / 0.20, 0.0, 1.0)
    red_support = (
        ((hue <= 24) | (hue >= 145))
        & (hsv[:, :, 1] >= 40)
        & (hsv[:, :, 2] >= 28)
        & (red_contrast >= 18.0))
    return (
        np.asarray(np.clip(likelihood, 0.0, 1.0), dtype=np.float32),
        red_support)


def catheter_color_likelihood(bgr: np.ndarray) -> np.ndarray:
    """Return a continuous likelihood for the four painted catheter colors."""
    return _catheter_color_features(bgr)[0]


def _continuous_color_path(
        likelihood: np.ndarray,
        roi: tuple[int, int, int, int],
        anchors_xy: list[np.ndarray],
        catheter_mask: np.ndarray,
        previous_points_xy: np.ndarray | None = None) -> np.ndarray:
    """Trace ordered anchors through an independently segmented catheter.

    A short weak-paint/specular gap may still be crossed, but distance outside
    the independently formed mask is penalized strongly.  The distance inside
    the mask favors its medial ridge rather than a wiggly silhouette edge.
    """
    x, y, width, height = roi
    anchors = np.asarray(anchors_xy, dtype=np.float64) - [x, y]
    anchors[:, 0] = np.clip(anchors[:, 0], 0, width - 1)
    anchors[:, 1] = np.clip(anchors[:, 1], 0, height - 1)
    endpoint_distance = float(np.linalg.norm(anchors[-1] - anchors[0]))
    margin = int(np.clip(0.65 * endpoint_distance, 90, 220))
    minimum = np.floor(np.min(anchors, axis=0) - margin).astype(int)
    maximum = np.ceil(np.max(anchors, axis=0) + margin + 1).astype(int)
    x0, y0 = np.maximum(minimum, 0)
    x1, y1 = np.minimum(maximum, [width, height])

    local_likelihood = np.asarray(
        likelihood[y0:y1, x0:x1], dtype=np.float64)
    local_mask = np.uint8(catheter_mask[y0:y1, x0:x1] > 0)
    core_distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(
        np.uint8(local_mask == 0), cv2.DIST_L2, 5)
    cost = (
        1.0
        + 8.0 * (1.0 - local_likelihood) ** 2
        + 10.0 / (core_distance + 0.75)
        + np.where(
            local_mask > 0,
            0.0,
            30.0 + 12.0 * np.clip(outside_distance, 0.0, 6.0)))
    if previous_points_xy is not None:
        previous = np.asarray(previous_points_xy, dtype=np.float64)
        previous = previous[np.all(np.isfinite(previous), axis=1)]
        if len(previous) >= 2:
            prior = np.zeros(local_likelihood.shape, dtype=np.uint8)
            local = np.rint(
                previous - [x + x0, y + y0]).astype(np.int32)
            cv2.polylines(prior, [local], False, 255, 3, cv2.LINE_AA)
            temporal_distance = cv2.distanceTransform(
                np.uint8(prior == 0), cv2.DIST_L2, 5)
            cost += 0.08 * np.clip(temporal_distance - 4.0, 0.0, 25.0)

    output = []
    local_anchors = anchors - [x0, y0]
    for start_xy, end_xy in zip(local_anchors[:-1], local_anchors[1:]):
        start = tuple(np.rint(start_xy[::-1]).astype(int))
        end = tuple(np.rint(end_xy[::-1]).astype(int))
        segment, _ = route_through_array(
            cost, start, end, fully_connected=True, geometric=True)
        segment = np.asarray(segment, dtype=np.int32)
        if output:
            segment = segment[1:]
        output.extend(segment.tolist())
    path = np.asarray(output, dtype=np.int32)
    path[:, 0] += y0
    path[:, 1] += x0
    return path


def _independent_catheter_mask(
        likelihood: np.ndarray,
        strong_threshold: float = 0.12,
        support_threshold: float = 0.055,
        minimum_component_area_px: int = 3) -> np.ndarray:
    """Segment paint without using a path, temporal curve, or opposite eye.

    Strong paint pixels seed a hysteresis segmentation.  A weaker pixel is
    retained only when it belongs to a connected component containing a strong
    seed.  This keeps antialiased shaft edges and dim ring pixels without
    admitting isolated weak chromatic noise from the scene.
    """
    values = np.asarray(likelihood, dtype=np.float32)
    strong = np.uint8(values >= float(strong_threshold))
    support = np.uint8(values >= float(support_threshold))
    support = cv2.morphologyEx(
        support, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
    output = np.zeros_like(support, dtype=np.uint8)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < int(minimum_component_area_px):
            continue
        component_pixels = labels == component
        if np.any(strong[component_pixels]):
            output[component_pixels] = 255
    return output


def _edge_refine_catheter_mask(
        bgr: np.ndarray,
        initial_mask: np.ndarray,
        likelihood: np.ndarray,
        workspace_mask: np.ndarray | None = None) -> np.ndarray:
    """Snap a color-seeded mask to local image edges with one graph-cut pass."""
    initial = np.uint8(initial_mask > 0)
    if np.count_nonzero(initial) < 8:
        return initial * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    band = cv2.dilate(initial, kernel11)
    sure_foreground = np.uint8(
        (np.asarray(likelihood) >= 0.16) & (initial > 0))
    sure_foreground = cv2.erode(sure_foreground, kernel3)
    if np.count_nonzero(sure_foreground) < 3:
        sure_foreground = np.uint8(np.asarray(likelihood) >= 0.20)

    labels = np.full(initial.shape, cv2.GC_BGD, dtype=np.uint8)
    labels[band > 0] = cv2.GC_PR_BGD
    labels[initial > 0] = cv2.GC_PR_FGD
    labels[sure_foreground > 0] = cv2.GC_FGD
    if workspace_mask is not None:
        labels[np.asarray(workspace_mask) == 0] = cv2.GC_BGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    rows, columns = np.where(band > 0)
    margin = 3
    x0 = max(int(columns.min()) - margin, 0)
    y0 = max(int(rows.min()) - margin, 0)
    x1 = min(int(columns.max()) + margin + 1, labels.shape[1])
    y1 = min(int(rows.max()) + margin + 1, labels.shape[0])
    local_labels = labels[y0:y1, x0:x1].copy()
    try:
        cv2.grabCut(
            np.asarray(bgr, dtype=np.uint8)[y0:y1, x0:x1],
            local_labels, None,
            background_model, foreground_model, 1, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return initial * 255
    labels[y0:y1, x0:x1] = local_labels
    refined = np.uint8(
        (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)) * 255
    refined[band == 0] = 0
    if workspace_mask is not None:
        refined[np.asarray(workspace_mask) == 0] = 0
    # Keep only graph-cut components that contain a definite color seed.
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(refined > 0), 8)
    output = np.zeros_like(refined)
    for component in range(1, count):
        pixels = components == component
        if stats[component, cv2.CC_STAT_AREA] >= 3 and np.any(
                sure_foreground[pixels]):
            output[pixels] = 255
    return output if np.any(output) else initial * 255


def _component_nearest_base(
        mask: np.ndarray, base_roi_xy: np.ndarray,
        minimum_area_px: int,
        previous_points_roi_xy: np.ndarray | None = None) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(mask > 0), 8)
    candidates = [
        index for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= int(minimum_area_px)]
    if not candidates:
        return np.zeros_like(mask, dtype=np.uint8)

    def box_distance_squared(component: int) -> float:
        left = float(stats[component, cv2.CC_STAT_LEFT])
        top = float(stats[component, cv2.CC_STAT_TOP])
        right = left + float(stats[component, cv2.CC_STAT_WIDTH]) - 1.0
        bottom = top + float(stats[component, cv2.CC_STAT_HEIGHT]) - 1.0
        dx = max(left - base_roi_xy[0], 0.0, base_roi_xy[0] - right)
        dy = max(top - base_roi_xy[1], 0.0, base_roi_xy[1] - bottom)
        # Bounding-box distance is a cheap shortlist metric.  The catheter
        # touches the registered base while static colored fixtures do not.
        return float(dx * dx + dy * dy)

    # Bounding boxes are only a shortlist.  A long, bent component can have a
    # box containing the base even when every one of its pixels is far away.
    # This was the source of occasional base-only/full-shaft alternation.
    shortlist = sorted(candidates, key=box_distance_squared)[:12]
    prior_support = np.zeros(count, dtype=np.float64)
    if previous_points_roi_xy is not None and len(previous_points_roi_xy) >= 2:
        corridor = np.zeros_like(mask, dtype=np.uint8)
        prior = np.rint(previous_points_roi_xy).astype(np.int32)
        cv2.polylines(corridor, [prior], False, 255, 17, cv2.LINE_AA)
        present = labels[corridor > 0]
        prior_support[:len(np.bincount(present, minlength=count))] = np.bincount(
            present, minlength=count)

    def exact_distance(component: int) -> float:
        left = int(stats[component, cv2.CC_STAT_LEFT])
        top = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        rows, columns = np.where(
            labels[top:top + height, left:left + width] == component)
        if len(columns) == 0:
            return float("inf")
        return float(np.sqrt(np.min(
            (columns + left - base_roi_xy[0]) ** 2
            + (rows + top - base_roi_xy[1]) ** 2)))

    distances = {component: exact_distance(component) for component in shortlist}
    # Temporal overlap is deliberately a tie-breaker, not a propagated mask:
    # the current image still determines every accepted pixel.
    selected = min(
        shortlist,
        key=lambda component: (
            distances[component]
            - 0.050 * min(prior_support[component], 2000.0)
            - 0.002 * min(float(stats[component, cv2.CC_STAT_AREA]), 3000.0)))
    return np.uint8(labels == selected) * 255


def chromatic_foreground_mask(
        bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point_xy: np.ndarray,
        minimum_saturation: int = 55,
        minimum_value: int = 30,
        minimum_area_px: int = 20,
        previous_points_xy: np.ndarray | None = None,
        background_bgr: np.ndarray | None = None,
        minimum_background_difference: int = 18) -> np.ndarray:
    """Extract the base-connected colored object from a neutral background."""
    x, y, width, height = roi
    crop = bgr[y:y + height, x:x + width]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Keep weak antialiased catheter pixels available so the dark proximal
    # paint and thin rings remain connected.  Class-specific likelihood is a
    # *path cost* below, not a brittle deletion threshold here.
    foreground = np.uint8(
        (hsv[:, :, 1] >= int(minimum_saturation))
        & (hsv[:, :, 2] >= int(minimum_value))) * 255
    if background_bgr is not None:
        background = np.asarray(background_bgr, dtype=np.uint8)
        if background.shape != crop.shape:
            raise ValueError("chromatic background shape does not match ROI")
        difference = np.max(cv2.absdiff(crop, background), axis=2)
        changed = difference >= int(minimum_background_difference)
        # The base attachment changes little during a recording and may
        # therefore enter the temporal median.  Preserve only a compact local
        # neighborhood; everywhere else, a static saturated ChArUco fringe is
        # removed even if its hue resembles cyan or magenta paint.
        base_roi = np.asarray(base_point_xy, dtype=np.float64) - [x, y]
        base_support = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            base_support, tuple(np.rint(base_roi).astype(int)), 28, 255, -1)
        foreground[(~changed) & (base_support == 0)] = 0
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    base_roi = np.asarray(base_point_xy, dtype=np.float64) - [x, y]
    previous_roi = (
        None if previous_points_xy is None
        else np.asarray(previous_points_xy, dtype=np.float64) - [x, y])
    return _component_nearest_base(
        foreground, base_roi, minimum_area_px=minimum_area_px,
        previous_points_roi_xy=previous_roi)


def _red_color_support(bgr: np.ndarray) -> np.ndarray:
    """Return the frame-local red-paint predicate before object masking."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    blue, green, red = [
        bgr[:, :, channel].astype(np.float32) for channel in range(3)]
    red_contrast = red - 0.60 * green - 0.40 * blue
    return (
        ((hue <= 24) | (hue >= 145))
        & (hsv[:, :, 1] >= 40)
        & (hsv[:, :, 2] >= 28)
        & (red_contrast >= 18.0))


def _red_mask(
        bgr: np.ndarray,
        mask: np.ndarray,
        red_support: np.ndarray | None = None) -> np.ndarray:
    support = (
        _red_color_support(bgr) if red_support is None
        else np.asarray(red_support, dtype=bool))
    return support & (mask > 0)


def _marker_candidates(
        bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        mask: np.ndarray,
        points_xy: np.ndarray,
        red_support: np.ndarray | None = None,
        ) -> list[dict[str, float | np.ndarray]]:
    x, y, width, height = roi
    crop = bgr[y:y + height, x:x + width]
    red = np.uint8(_red_mask(crop, mask, red_support)) * 255
    red = cv2.morphologyEx(
        red, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    if count <= 1 or len(points_xy) < 2:
        return []
    local_path = np.asarray(points_xy, dtype=np.float64) - [x, y]
    path_s = cumulative_arclength(local_path)
    tree = cKDTree(local_path)
    output = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        component_x = int(stats[component, cv2.CC_STAT_LEFT])
        component_y = int(stats[component, cv2.CC_STAT_TOP])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        local_component = np.uint8(
            labels[
                component_y:component_y + component_height,
                component_x:component_x + component_width] == component)
        local_rows, local_columns = np.where(local_component)
        rows = local_rows + component_y
        columns = local_columns + component_x
        component_distance = cv2.distanceTransform(
            local_component, cv2.DIST_L2, 5)
        local_spatial_peaks = peak_local_max(
            component_distance, min_distance=3, threshold_abs=3.0,
            num_peaks=8)
        spatial_peaks = local_spatial_peaks + [component_y, component_x]
        # In a foreshortened eye, both thin middle rings can touch the wide
        # distal band. Their 2D interior lobes remain distinct even when their
        # projections onto a direct path overlap. Preserve these as separate
        # soft observations; they never become path waypoints.
        if len(spatial_peaks) >= 3:
            for row, column in spatial_peaks:
                distance, index = tree.query([column, row], k=1)
                if distance > 22.0:
                    continue
                radius = float(component_distance[
                    row - component_y, column - component_x])
                output.append({
                    "s_px": float(path_s[index]),
                    "width_px": max(2.0 * radius, 1.0),
                    "center_xy": np.array(
                        [column + x, row + y], dtype=np.float64),
                    "support": float(np.pi * radius ** 2),
                })
            continue
        distances, indices = tree.query(
            np.column_stack([columns, rows]), k=1)
        values = np.sort(path_s[indices[distances <= 10.0]])
        if len(values) < 2:
            continue
        # Under axial foreshortening, two painted rings can touch in the 2D
        # red mask while remaining separate modes along the centerline. Split
        # such a component at valleys of its projected-arclength histogram.
        edges = np.arange(
            np.floor(values[0]), np.ceil(values[-1]) + 2.0, 1.0)
        histogram, _ = np.histogram(values, edges)
        smoothed = np.convolve(
            histogram.astype(np.float64), [0.25, 0.50, 0.25], mode="same")
        peaks, _ = find_peaks(
            smoothed, distance=3,
            prominence=max(1.0, 0.05 * float(np.max(smoothed))))
        if len(peaks) >= 2:
            peak_s = edges[peaks] + 0.5
            boundaries = 0.5 * (peak_s[:-1] + peak_s[1:])
            groups = np.split(values, np.searchsorted(values, boundaries))
        else:
            groups = [values]
        for group in groups:
            if len(group) < 2:
                continue
            center_s = float(np.median(group))
            path_width = float(
                np.percentile(group, 95) - np.percentile(group, 5))
            image_extent = float(max(
                stats[component, cv2.CC_STAT_WIDTH],
                stats[component, cv2.CC_STAT_HEIGHT]))
            width_px = max(
                path_width,
                min(image_extent, 12.0) if len(groups) == 1 else 1.0,
                1.0)
            center_xy = np.array([
                np.interp(center_s, path_s, points_xy[:, coordinate])
                for coordinate in range(2)], dtype=np.float64)
            output.append({
                "s_px": center_s,
                "width_px": width_px,
                "center_xy": center_xy,
                "support": float(len(group)),
            })
    return output


def _distal_red_ring_seed(
        bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        mask: np.ndarray,
        base_point_xy: np.ndarray,
        previous_points_xy: np.ndarray | None = None,
        red_support: np.ndarray | None = None,
        ) -> tuple[np.ndarray | None, float]:
    """Locate marker 3 using width, distal position, and optional history."""
    x, y, width, height = roi
    crop = bgr[y:y + height, x:x + width]
    red = np.uint8(_red_mask(crop, mask, red_support)) * 255
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(red, 8)
    candidates = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        extent = max(
            int(stats[component, cv2.CC_STAT_WIDTH]),
            int(stats[component, cv2.CC_STAT_HEIGHT]))
        if extent < 3:
            continue
        center = centroids[component].astype(np.float64) + [x, y]
        candidates.append({
            "area": float(area), "extent": float(extent),
            "center": center, "component": component})
    if not candidates:
        return None, float("nan")
    areas = np.asarray([item["area"] for item in candidates])
    extents = np.asarray([item["extent"] for item in candidates])
    centers = np.asarray([item["center"] for item in candidates])
    size_score = (
        0.55 * np.log1p(areas / max(float(np.median(areas)), 1.0))
        + 0.25 * extents / max(float(np.max(extents)), 1.0))
    previous_tip = None
    if previous_points_xy is not None:
        previous = np.asarray(previous_points_xy, dtype=np.float64)
        if len(previous) >= 2 and np.all(np.isfinite(previous[-1])):
            previous_tip = previous[-1]
    if previous_tip is not None:
        distance = np.linalg.norm(centers - previous_tip, axis=1)
        score = size_score - distance / 18.0
    else:
        distance = np.linalg.norm(
            centers - np.asarray(base_point_xy, dtype=np.float64), axis=1)
        distal_score = distance / max(float(np.max(distance)), 1.0)
        score = size_score + 2.0 * distal_score
    selected = candidates[int(np.argmax(score))]
    return selected["center"], float(selected["extent"])


def _relaxed_red_observations(
        bgr: np.ndarray, roi: tuple[int, int, int, int]) -> list[dict]:
    """Find small/dark red elements for cross-eye epipolar association."""
    x, y, width, height = roi
    crop = bgr[y:y + height, x:x + width]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    blue, green, red = [crop[:, :, k].astype(np.float32) for k in range(3)]
    contrast = red - 0.60 * green - 0.40 * blue
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    mask = np.uint8(
        (contrast >= 24.0) & (chroma >= 10.0)
        & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 28)
        & ((hsv[:, :, 0] <= 22) | (hsv[:, :, 0] >= 145)))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    output = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        component_x = int(stats[component, cv2.CC_STAT_LEFT])
        component_y = int(stats[component, cv2.CC_STAT_TOP])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        local_component = np.uint8(
            labels[
                component_y:component_y + component_height,
                component_x:component_x + component_width] == component)
        distance = cv2.distanceTransform(local_component, cv2.DIST_L2, 5)
        local_peaks = peak_local_max(
            distance, min_distance=3, threshold_abs=1.2, num_peaks=8)
        peaks = local_peaks + [component_y, component_x]
        if len(peaks) >= 2 and area >= 30:
            for row, column in peaks:
                radius = float(distance[
                    row - component_y, column - component_x])
                output.append({"xy": np.array([column + x, row + y], float),
                               "width": 2.0 * radius, "area": np.pi * radius**2})
        else:
            output.append({"xy": centroids[component] + [x, y],
                           "width": float(max(stats[component, 2], stats[component, 3])),
                           "area": float(area)})
    return output


def _marker_layout_quality(result: SamMaterialResult) -> float:
    """Score whether one eye contains four distinct, ordered ring locations."""
    if result.marker_centers_xy is None:
        return 0.0
    centers = np.asarray(result.marker_centers_xy, dtype=np.float64)
    widths = np.asarray(result.marker_widths_px, dtype=np.float64)
    observed = np.asarray(result.marker_observed, dtype=bool)
    quality = 0.5 * float(np.count_nonzero(observed))
    for first, second in zip(range(3), range(1, 4)):
        if not (observed[first] and observed[second]
                and np.all(np.isfinite(centers[[first, second]]))):
            continue
        scale = max(
            5.0, 0.5 * float(widths[first] + widths[second]))
        separation = float(np.linalg.norm(
            centers[second] - centers[first]))
        # Distinct neighboring rings contribute at most two points. Two IDs
        # assigned to peaks on the same physical ring contribute almost none.
        quality += min(separation / scale, 2.0)
    return quality


def _relabel_epipolar_inconsistent_eye(
        source: SamMaterialResult,
        target: SamMaterialResult,
        target_image: np.ndarray,
        target_roi: tuple[int, int, int, int],
        disparity_shift_px: float,
        epipolar_half_width_px: float) -> SamMaterialResult:
    """Globally relabel target-eye red components from source-eye identities."""
    if (source.marker_centers_xy is None
            or target.marker_centers_xy is None):
        return target
    source_centers = np.asarray(source.marker_centers_xy, dtype=np.float64)
    source_widths = np.asarray(source.marker_widths_px, dtype=np.float64)
    source_observed = np.asarray(source.marker_observed, dtype=bool)
    target_centers = np.asarray(target.marker_centers_xy, dtype=np.float64)
    target_observed = np.asarray(target.marker_observed, dtype=bool)
    common = source_observed & target_observed
    if not np.any(common):
        return target
    same_id_epipolar = np.abs(
        source_centers[:, 1] - target_centers[:, 1])
    inconsistent = bool(np.any(
        common & (same_id_epipolar > float(epipolar_half_width_px))))
    if not inconsistent:
        return target

    candidates = _relaxed_red_observations(target_image, target_roi)
    source_ids = np.flatnonzero(
        source_observed & np.all(np.isfinite(source_centers), axis=1))
    if not len(source_ids) or not candidates:
        return target
    # Real candidates plus one private missing-observation column per source
    # ID. Hungarian assignment enforces a bijection among real image regions.
    candidate_count = len(candidates)
    missing_cost = float(epipolar_half_width_px) + 2.0
    cost = np.full(
        (len(source_ids), candidate_count + len(source_ids)),
        missing_cost, dtype=np.float64)
    for row, marker_id in enumerate(source_ids):
        expected = source_centers[marker_id] + [disparity_shift_px, 0.0]
        for column, candidate in enumerate(candidates):
            dy = abs(float(
                candidate["xy"][1] - source_centers[marker_id, 1]))
            if dy > float(epipolar_half_width_px):
                cost[row, column] = missing_cost + dy
                continue
            width_error = abs(np.log(
                max(float(candidate["width"]), 1.0)
                / max(float(source_widths[marker_id]), 1.0)))
            cost[row, column] = (
                dy + 0.35 * width_error
                + 0.005 * abs(float(candidate["xy"][0] - expected[0])))
        # Only the row's own dummy column represents a missing observation.
        cost[row, candidate_count:] = missing_cost + 100.0
        cost[row, candidate_count + row] = missing_cost

    rows, columns = linear_sum_assignment(cost)
    centers = np.full((4, 2), np.nan, dtype=np.float64)
    widths = np.full(4, np.nan, dtype=np.float64)
    confidence = np.zeros(4, dtype=np.float64)
    observed = np.zeros(4, dtype=np.uint8)
    for row, column in zip(rows, columns):
        marker_id = int(source_ids[row])
        if column >= candidate_count or cost[row, column] >= missing_cost:
            continue
        candidate = candidates[column]
        centers[marker_id] = candidate["xy"]
        widths[marker_id] = candidate["width"]
        confidence[marker_id] = 0.80 if marker_id in (0, 3) else 0.75
        observed[marker_id] = 1
    return replace(
        target, marker_centers_xy=centers, marker_widths_px=widths,
        marker_confidence=confidence, marker_observed=observed)


def _reroute_to_reconciled_tip_marker(
        result: SamMaterialResult,
        image: np.ndarray,
        roi: tuple[int, int, int, int],
        minimum_endpoint_change_px: float = 5.0) -> SamMaterialResult:
    """Rebuild the cyan path when stereo changes the distal-ring identity."""
    if (result.marker_centers_xy is None
            or result.marker_observed is None
            or not bool(result.marker_observed[3])):
        return result
    marker = np.asarray(result.marker_centers_xy[3], dtype=np.float64)
    if (not np.all(np.isfinite(marker))
            or np.linalg.norm(result.material.points[-1] - marker)
            <= float(minimum_endpoint_change_px)):
        return result
    x, y, width, height = roi
    mask = np.asarray(result.material.mask, dtype=np.uint8)
    local_marker = np.rint(marker - [x, y]).astype(int)
    if (local_marker[0] < 0 or local_marker[0] >= width
            or local_marker[1] < 0 or local_marker[1] >= height
            or mask[local_marker[1], local_marker[0]] == 0):
        return result
    likelihood, _ = _catheter_color_features(
        image[y:y + height, x:x + width])
    try:
        path = _continuous_color_path(
            likelihood, roi,
            [result.material.points[0], marker], mask,
            result.material.points)
    except (RuntimeError, ValueError):
        return result
    points = np.column_stack([
        path[:, 1] + x, path[:, 0] + y]).astype(np.float64)
    raw_length = float(cumulative_arclength(points)[-1])
    smooth_count = max(16, int(np.ceil(raw_length)) + 1)
    points = resample_arclength(points, smooth_count, smooth_window=1)
    smooth_window = min(
        11, smooth_count if smooth_count % 2 else smooth_count - 1)
    if smooth_window >= 5:
        endpoints = points[[0, -1]].copy()
        points = savgol_filter(
            points, smooth_window, polyorder=2, axis=0, mode="interp")
        points[0], points[-1] = endpoints
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    local_path = np.rint(points - [x, y]).astype(int)
    local_path[:, 0] = np.clip(local_path[:, 0], 0, width - 1)
    local_path[:, 1] = np.clip(local_path[:, 1], 0, height - 1)
    sampled_radius = distance[local_path[:, 1], local_path[:, 0]]
    positive_radius = sampled_radius[sampled_radius > 0.0]
    radius = (
        float(np.median(positive_radius))
        if len(positive_radius) else result.material.centerline.radius_px)
    boundary = int(np.clip(
        result.material.distal_boundary_index, 0, len(points) - 1))
    if (bool(result.marker_observed[0])
            and np.all(np.isfinite(result.marker_centers_xy[0]))):
        boundary = int(np.argmin(np.linalg.norm(
            points - result.marker_centers_xy[0], axis=1)))
    material = replace(
        result.material,
        centerline=Centerline(points, roi, mask, radius),
        distal_boundary_index=boundary,
        distal_boundary_fraction=float(boundary / max(len(points) - 1, 1)),
        brightness_profile=np.full(len(points), np.nan, dtype=np.float64))
    sample_indices = np.unique(np.rint(np.linspace(
        0, len(points) - 1, min(10, len(points)))).astype(int))
    prompt = replace(
        result.prompt, positive_xy=points[sample_indices].copy(),
        source=result.prompt.source + "+stereo_tip_reroute")
    return replace(result, material=material, prompt=prompt)


def refine_marked_stereo_pair(
        left_image: np.ndarray, right_image: np.ndarray,
        left_result: SamMaterialResult, right_result: SamMaterialResult,
        left_roi: tuple[int, int, int, int], right_roi: tuple[int, int, int, int],
        base_disparity_px: float,
        epipolar_half_width_px: float = 10.0,
        previous_left_result: SamMaterialResult | None = None,
        previous_right_result: SamMaterialResult | None = None,
        temporal_marker_max_displacement_px: float = 18.0,
        ) -> tuple[SamMaterialResult, SamMaterialResult]:
    """Recover bad-eye ring identities from the better eye with soft geometry."""
    def reject_temporal_snaps(
            current: SamMaterialResult,
            previous: SamMaterialResult | None) -> SamMaterialResult:
        """Turn an implausible identity jump into a missing observation.

        Do not copy the previous coordinate into the current observation.  A
        copied coordinate becomes a self-reinforcing stale state: every later
        correct detection is far from it and is rejected forever.  Missing
        observations can instead be recovered from current-frame stereo
        evidence below, and do not constrain the next frame when recovery is
        unavailable.
        """
        if (previous is None or current.marker_centers_xy is None
                or previous.marker_centers_xy is None):
            return current
        centers = np.asarray(
            current.marker_centers_xy, dtype=np.float64).copy()
        widths = np.asarray(
            current.marker_widths_px, dtype=np.float64).copy()
        confidence = np.asarray(
            current.marker_confidence, dtype=np.float64).copy()
        observed = np.asarray(
            current.marker_observed, dtype=np.uint8).copy()
        previous_centers = np.asarray(
            previous.marker_centers_xy, dtype=np.float64)
        previous_observed = np.asarray(
            previous.marker_observed, dtype=bool)
        for marker_id in range(min(4, len(centers), len(previous_centers))):
            if not (observed[marker_id]
                    and previous_observed[marker_id]
                    and np.all(np.isfinite(centers[marker_id]))
                    and np.all(np.isfinite(previous_centers[marker_id]))):
                continue
            displacement = float(np.linalg.norm(
                centers[marker_id] - previous_centers[marker_id]))
            if displacement <= float(temporal_marker_max_displacement_px):
                continue
            centers[marker_id] = np.nan
            widths[marker_id] = np.nan
            confidence[marker_id] = 0.0
            observed[marker_id] = 0
        return replace(
            current, marker_centers_xy=centers, marker_widths_px=widths,
            marker_confidence=confidence, marker_observed=observed)

    # Reject identity snaps before cross-eye reconciliation. This lets a
    # well-conditioned eye repair the other eye in the same frame instead of
    # allowing a stale propagated coordinate to suppress that evidence.
    left_result = reject_temporal_snaps(left_result, previous_left_result)
    right_result = reject_temporal_snaps(right_result, previous_right_result)
    left_layout_quality = _marker_layout_quality(left_result)
    right_layout_quality = _marker_layout_quality(right_result)
    # A local decoder can confidently compress two IDs onto one physical ring
    # after self-overlap. If the other eye has a clearly more distinct layout,
    # use its identities to solve a global one-to-one epipolar assignment over
    # current target-eye red components. This may override a confident but
    # geometrically impossible local label.
    layout_margin = 0.35
    if left_layout_quality > right_layout_quality + layout_margin:
        right_result = _relabel_epipolar_inconsistent_eye(
            left_result, right_result, right_image, right_roi,
            -float(base_disparity_px), epipolar_half_width_px)
    elif right_layout_quality > left_layout_quality + layout_margin:
        left_result = _relabel_epipolar_inconsistent_eye(
            right_result, left_result, left_image, left_roi,
            float(base_disparity_px), epipolar_half_width_px)
    left_length = float(cumulative_arclength(left_result.material.points)[-1])
    right_length = float(cumulative_arclength(right_result.material.points)[-1])
    if right_length >= left_length:
        source, target = right_result, left_result
        target_image, target_roi = left_image, left_roi
        shift = float(base_disparity_px)
        target_is_left = True
    else:
        source, target = left_result, right_result
        target_image, target_roi = right_image, right_roi
        shift = -float(base_disparity_px)
        target_is_left = False
    if source.marker_centers_xy is None:
        return left_result, right_result
    candidates: list[dict] = []
    centers = np.full((4, 2), np.nan); widths = np.full(4, np.nan)
    confidence = np.zeros(4); observed = np.zeros(4, np.uint8)
    used: set[int] = set()
    source_centers = np.asarray(source.marker_centers_xy, float)
    source_widths = np.asarray(source.marker_widths_px, float)
    source_observed = np.asarray(source.marker_observed, bool)

    # A confident observation made in this eye is preferable to a relaxed
    # epipolar candidate imported from the other eye.  This matters most for
    # the wide distal band: its relaxed red component can contain several
    # distance-transform peaks, only one of which is the actual annulus
    # centroid.  Cross-eye matching is a recovery mechanism, not a reason to
    # overwrite a good local measurement.
    if target.marker_centers_xy is not None:
        target_centers = np.asarray(target.marker_centers_xy, dtype=np.float64)
        target_widths = np.asarray(target.marker_widths_px, dtype=np.float64)
        target_confidence = np.asarray(
            target.marker_confidence, dtype=np.float64)
        target_observed = np.asarray(target.marker_observed, dtype=bool)
        for marker_id in range(min(4, len(target_centers))):
            preserve_threshold = 0.55
            if not (target_observed[marker_id]
                    and target_confidence[marker_id] >= preserve_threshold
                    and np.all(np.isfinite(target_centers[marker_id]))):
                continue
            centers[marker_id] = target_centers[marker_id]
            widths[marker_id] = target_widths[marker_id]
            confidence[marker_id] = target_confidence[marker_id]
            observed[marker_id] = 1

    recovery_ids = [
        marker_id for marker_id in (3, 0, 1, 2)
        if not observed[marker_id] and source_observed[marker_id]
        and np.all(np.isfinite(source_centers[marker_id]))]
    if recovery_ids:
        candidates = _relaxed_red_observations(target_image, target_roi)
        # Do not assign a component already explained by a preserved local
        # observation to a missing neighboring marker.
        for marker_id in np.flatnonzero(observed):
            if not candidates:
                break
            distances = np.asarray([
                np.linalg.norm(item["xy"] - centers[marker_id])
                for item in candidates])
            nearest = int(np.argmin(distances))
            reservation_radius = max(
                5.0, 0.75 * float(widths[marker_id]))
            if distances[nearest] <= reservation_radius:
                used.add(nearest)

    # Assign the distinctive wide distal band first, then the three rings.
    for marker_id in (3, 0, 1, 2):
        if observed[marker_id]:
            continue
        if not source_observed[marker_id] or not np.all(np.isfinite(source_centers[marker_id])):
            continue
        expected = source_centers[marker_id] + [shift, 0.0]
        choices = []
        for index, item in enumerate(candidates):
            if index in used:
                continue
            dy = abs(float(item["xy"][1] - source_centers[marker_id, 1]))
            if dy > float(epipolar_half_width_px):
                continue
            width_error = abs(np.log(max(float(item["width"]), 1.0)
                                     / max(float(source_widths[marker_id]), 1.0)))
            score = dy + 0.008 * abs(float(item["xy"][0] - expected[0])) + width_error
            if marker_id == 3:
                score -= 0.04 * min(float(item["width"]), 25.0)
            choices.append((score, index))
        if not choices:
            continue
        _, selected = min(choices)
        used.add(selected); item = candidates[selected]
        centers[marker_id] = item["xy"]; widths[marker_id] = item["width"]
        confidence[marker_id] = 0.75; observed[marker_id] = 1

    # Retain unmatched target observations only where cross-eye recovery had
    # no answer. Low-confidence local observations remain soft evidence; the
    # joint fit projects every ring centroid onto the independently measured
    # cyan path before using it.
    if target.marker_centers_xy is not None:
        for marker_id in range(4):
            if not observed[marker_id] and target.marker_observed[marker_id]:
                centers[marker_id] = target.marker_centers_xy[marker_id]
                widths[marker_id] = target.marker_widths_px[marker_id]
                confidence[marker_id] = min(float(target.marker_confidence[marker_id]), 0.35)
                observed[marker_id] = 1

    # Cross-eye evidence may repair marker identity, but segmentation and its
    # centerline remain observations of this eye alone.  In particular, never
    # paint an epipolar corridor into the target mask.
    target = replace(target,
                     marker_centers_xy=centers, marker_widths_px=widths,
                     marker_confidence=confidence, marker_observed=observed)
    left, right = (
        (target, source) if target_is_left else (source, target))

    def recover_missing_from_other(
            target_result: SamMaterialResult,
            source_result: SamMaterialResult,
            target_image: np.ndarray,
            target_roi: tuple[int, int, int, int],
            disparity_shift_px: float) -> SamMaterialResult:
        """Recover either eye; centerline length must not choose the direction."""
        if (target_result.marker_centers_xy is None
                or source_result.marker_centers_xy is None):
            return target_result
        target_centers = np.asarray(
            target_result.marker_centers_xy, dtype=np.float64).copy()
        target_widths = np.asarray(
            target_result.marker_widths_px, dtype=np.float64).copy()
        target_confidence = np.asarray(
            target_result.marker_confidence, dtype=np.float64).copy()
        target_observed = np.asarray(
            target_result.marker_observed, dtype=np.uint8).copy()
        source_centers = np.asarray(
            source_result.marker_centers_xy, dtype=np.float64)
        source_widths = np.asarray(
            source_result.marker_widths_px, dtype=np.float64)
        source_observed = np.asarray(
            source_result.marker_observed, dtype=bool)
        missing = [
            marker_id for marker_id in (3, 0, 1, 2)
            if not target_observed[marker_id]
            and source_observed[marker_id]
            and np.all(np.isfinite(source_centers[marker_id]))]
        if not missing:
            return target_result
        local_candidates = _relaxed_red_observations(target_image, target_roi)
        used_candidates: set[int] = set()
        for marker_id in np.flatnonzero(target_observed):
            if not local_candidates:
                break
            distances = np.asarray([
                np.linalg.norm(item["xy"] - target_centers[marker_id])
                for item in local_candidates])
            nearest = int(np.argmin(distances))
            if distances[nearest] <= max(
                    5.0, 0.75 * float(target_widths[marker_id])):
                used_candidates.add(nearest)
        for marker_id in missing:
            expected = source_centers[marker_id] + [disparity_shift_px, 0.0]
            choices = []
            for candidate_index, item in enumerate(local_candidates):
                if candidate_index in used_candidates:
                    continue
                dy = abs(float(
                    item["xy"][1] - source_centers[marker_id, 1]))
                if dy > float(epipolar_half_width_px):
                    continue
                width_error = abs(np.log(
                    max(float(item["width"]), 1.0)
                    / max(float(source_widths[marker_id]), 1.0)))
                score = (
                    dy
                    + 0.008 * abs(float(item["xy"][0] - expected[0]))
                    + width_error)
                if marker_id == 3:
                    score -= 0.04 * min(float(item["width"]), 25.0)
                choices.append((score, candidate_index))
            if not choices:
                continue
            _, selected = min(choices)
            used_candidates.add(selected)
            item = local_candidates[selected]
            target_centers[marker_id] = item["xy"]
            target_widths[marker_id] = item["width"]
            target_confidence[marker_id] = 0.75
            target_observed[marker_id] = 1
        return replace(
            target_result, marker_centers_xy=target_centers,
            marker_widths_px=target_widths,
            marker_confidence=target_confidence,
            marker_observed=target_observed)

    # The longer cyan centerline is a useful first recovery source, but it is
    # not necessarily the eye with the correct ring identity. Complete the
    # reconciliation in both directions using only current-frame candidates.
    left = recover_missing_from_other(
        left, right, left_image, left_roi, float(base_disparity_px))
    right = recover_missing_from_other(
        right, left, right_image, right_roi, -float(base_disparity_px))

    # Marker reconciliation happens after each eye's initial color route. If a
    # locally wrong distal identity was corrected, the old cyan path still
    # ends at that wrong ring unless it is rebuilt explicitly.
    left = _reroute_to_reconciled_tip_marker(
        left, left_image, left_roi)
    right = _reroute_to_reconciled_tip_marker(
        right, right_image, right_roi)

    return left, right


def _layout_score(candidates: list[dict], ids: tuple[int, ...]) -> float:
    """Softly score an ordered marker assignment without enforcing geometry."""
    observed = np.asarray([item["s_px"] for item in candidates], np.float64)
    expected = EXPECTED_MARKER_COORDINATES_MM[np.asarray(ids)]
    if len(observed) >= 2 and np.ptp(expected) > 0:
        design = np.column_stack([np.ones(len(expected)), expected])
        coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
        prediction = design @ coefficients
        layout_error = float(np.sqrt(np.mean((observed - prediction) ** 2)))
        layout_error /= max(float(np.ptp(observed)), 1.0)
        if coefficients[1] <= 0.0:
            layout_error += 10.0
    else:
        layout_error = 0.0
    widths = np.asarray([item["width_px"] for item in candidates])
    width_scale = max(float(np.median(widths)), 1.0)
    width_penalty = 0.0
    if 3 in ids:
        tip_width = widths[ids.index(3)]
        width_penalty += max(0.0, float(np.max(widths) - tip_width)) / width_scale
    if 0 in ids:
        interface_width = widths[ids.index(0)]
        for middle_id in (1, 2):
            if middle_id in ids:
                width_penalty += 0.25 * max(
                    0.0, float(widths[ids.index(middle_id)] - interface_width)
                ) / width_scale
    # Geometry is deliberately weak relative to order/width: approximate
    # physical distances merely break otherwise equivalent identity choices.
    return 0.35 * layout_error + 0.20 * width_penalty


def decode_marker_candidates(
        candidates: list[dict], path_length_px: float,
        expected_count: int = 4,
        allowed_ids: tuple[int, ...] | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode up to four ordered rings, retaining missing observations as NaN."""
    centers = np.full((expected_count, 2), np.nan, dtype=np.float64)
    widths = np.full(expected_count, np.nan, dtype=np.float64)
    confidence = np.zeros(expected_count, dtype=np.float64)
    observed = np.zeros(expected_count, dtype=np.uint8)
    if not candidates:
        return centers, widths, confidence, observed
    candidates = sorted(candidates, key=lambda item: item["s_px"])
    ids_available = (
        tuple(range(expected_count)) if allowed_ids is None
        else tuple(int(value) for value in allowed_ids))
    if not ids_available:
        return centers, widths, confidence, observed

    assignments: list[tuple[float, list[dict], tuple[int, ...]]] = []
    if len(candidates) >= len(ids_available):
        for chosen_indices in combinations(
                range(len(candidates)), len(ids_available)):
            chosen = [candidates[index] for index in chosen_indices]
            ids = ids_available
            score = _layout_score(chosen, ids)
            # The wide tip band should lie near the distal end, but this remains
            # a soft image-coordinate cue rather than a geometric constraint.
            score += 0.35 * max(
                0.0, 0.75 - chosen[-1]["s_px"] / max(path_length_px, 1.0))
            assignments.append((score, chosen, ids))
    else:
        for ids in combinations(ids_available, len(candidates)):
            score = _layout_score(candidates, ids)
            distal_fraction = candidates[-1]["s_px"] / max(path_length_px, 1.0)
            if 3 in ids:
                score += 0.35 * max(0.0, 0.75 - distal_fraction)
            elif distal_fraction > 0.82:
                score += 0.35
            assignments.append((score, candidates, ids))
    score, chosen, ids = min(assignments, key=lambda item: item[0])
    base_confidence = (
        0.95 if len(candidates) == len(ids_available)
        else 0.65 if len(candidates) > len(ids_available)
        else 0.40)
    base_confidence *= float(np.exp(-min(score, 4.0)))
    for item, marker_id in zip(chosen, ids):
        centers[marker_id] = item["center_xy"]
        widths[marker_id] = item["width_px"]
        confidence[marker_id] = base_confidence
        observed[marker_id] = 1
    return centers, widths, confidence, observed


def extract_marked_chromatic_result(
        bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point_xy: np.ndarray,
        minimum_saturation: int = 55,
        minimum_value: int = 30,
        previous_points_xy: np.ndarray | None = None,
        background_bgr: np.ndarray | None = None,
        minimum_background_difference: int = 18,
        workspace_mask: np.ndarray | None = None) -> SamMaterialResult:
    """Return a pipeline-compatible color mask, centerline, tip, and markers."""
    x, y, width, height = roi
    crop = bgr[y:y + height, x:x + width]
    likelihood, red_support = _catheter_color_features(crop)
    if workspace_mask is not None:
        workspace = np.asarray(workspace_mask, dtype=np.uint8)
        if workspace.shape != likelihood.shape:
            raise ValueError("workspace mask shape does not match ROI")
        likelihood[workspace == 0] = 0.0
    if background_bgr is not None:
        background = np.asarray(background_bgr, dtype=np.uint8)
        if background.shape != crop.shape:
            raise ValueError("chromatic background shape does not match ROI")
        changed = np.max(cv2.absdiff(crop, background), axis=2)
        base_roi = np.asarray(base_point_xy, dtype=np.float64) - [x, y]
        preserve = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            preserve, tuple(np.rint(base_roi).astype(int)), 28, 255, -1)
        likelihood[(changed < int(minimum_background_difference))
                   & (preserve == 0)] = 0.0

    mask = _independent_catheter_mask(likelihood)
    if not np.any(mask):
        raise ValueError("marked_color_support_missing")
    mask = _edge_refine_catheter_mask(
        bgr[y:y + height, x:x + width], mask, likelihood,
        workspace_mask=workspace_mask)
    # Marker detection uses paint chromaticity inside the edge-refined object.
    # There is no generic HSV component and no route to a neutral background.
    color_support = cv2.bitwise_and(
        np.uint8(likelihood >= 0.055) * 255, mask)
    ring_seed, ring_extent = _distal_red_ring_seed(
        bgr, roi, color_support, base_point_xy, previous_points_xy,
        red_support=red_support)
    distal_anchor = ring_seed
    if distal_anchor is None and previous_points_xy is not None:
        previous = np.asarray(previous_points_xy, dtype=np.float64)
        if len(previous) >= 2 and np.all(np.isfinite(previous[-1])):
            distal_anchor = previous[-1]
    if distal_anchor is None:
        raise ValueError("marked_tip_ring_missing")

    path = _continuous_color_path(
        likelihood, roi,
        [np.asarray(base_point_xy, dtype=np.float64), distal_anchor],
        mask,
        previous_points_xy)
    points = np.column_stack([
        path[:, 1] + x, path[:, 0] + y]).astype(np.float64)
    # Rings 0--2 identify material locations but are not image-space
    # centerline points: a visible annulus centroid can lie off the shaft axis.
    # Do not route through them. Smooth the independently found base-to-tip
    # color path before projecting marker observations onto it.
    raw_length = float(cumulative_arclength(points)[-1])
    smooth_count = max(16, int(np.ceil(raw_length)) + 1)
    points = resample_arclength(points, smooth_count, smooth_window=1)
    smooth_window = min(11, smooth_count if smooth_count % 2 else smooth_count - 1)
    if smooth_window >= 5:
        endpoints = points[[0, -1]].copy()
        points = savgol_filter(
            points, smooth_window, polyorder=2, axis=0, mode="interp")
        points[0], points[-1] = endpoints
    # Marker 3 identifies the terminal neighborhood for marked recordings;
    # the tiny exposed yellow dome is deliberately ignored.  The graph route
    # already terminates at the nearest pixel to the locally observed band
    # center.  Do not append the subpixel centroid after smoothing: that made
    # the marker a hard constraint and could create a short, sharply turning
    # final segment.  Marker 3 remains a stronger *soft* observation in the
    # subsequent joint two-view fit.
    tip = None

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sampled_radius = distance[
        np.clip(path[:, 0], 0, mask.shape[0] - 1),
        np.clip(path[:, 1], 0, mask.shape[1] - 1)]
    positive_radius = sampled_radius[sampled_radius > 0.0]
    radius = float(np.median(positive_radius)) if len(positive_radius) else 1.0
    midpoint = len(points) // 2
    material = MaterialCenterline(
        centerline=Centerline(points, roi, mask, radius),
        distal_boundary_index=midpoint,
        distal_boundary_fraction=float(midpoint / max(len(points) - 1, 1)),
        boundary_confidence=0.0,
        boundary_contrast=0.0,
        material_valid=False,
        brightness_profile=np.full(len(points), np.nan, dtype=np.float64))
    path_length = float(cumulative_arclength(material.points)[-1])
    candidates = _marker_candidates(
        bgr, roi, mask, material.points, red_support=red_support)
    if ring_seed is not None:
        exclusion_radius = max(0.75 * float(ring_extent), 8.0)
        candidates = [
            candidate for candidate in candidates
            if np.linalg.norm(candidate["center_xy"] - ring_seed)
            > exclusion_radius]
    centers, widths, confidence, observed = decode_marker_candidates(
        candidates, path_length, allowed_ids=(0, 1, 2))
    # The wide distal band is detected independently of the path and of the
    # other three rings.  Do not let a missing/thin middle ring lower the
    # endpoint confidence through the decoder's group score.
    if ring_seed is not None:
        centers[3] = ring_seed
        widths[3] = ring_extent
        confidence[3] = max(float(confidence[3]), 0.90)
        observed[3] = 1
    if observed[0]:
        boundary = int(np.argmin(np.linalg.norm(
            material.points - centers[0], axis=1)))
        material = replace(
            material,
            distal_boundary_index=boundary,
            distal_boundary_fraction=float(
                boundary / max(len(material.points) - 1, 1)),
            boundary_confidence=float(confidence[0]),
            boundary_contrast=float(widths[0]),
            material_valid=bool(confidence[0] >= 0.25))

    rows, columns = np.where(mask > 0)
    margin = 12
    box = np.array([
        max(x, x + int(columns.min()) - margin),
        max(y, y + int(rows.min()) - margin),
        min(x + width - 1, x + int(columns.max()) + margin),
        min(y + height - 1, y + int(rows.max()) + margin),
    ], dtype=np.float64)
    sample_indices = np.unique(np.rint(np.linspace(
        0, len(material.points) - 1, min(10, len(material.points)))).astype(int))
    prompt = PromptSet(
        box_xyxy=box,
        positive_xy=material.points[sample_indices].copy(),
        negative_xy=np.empty((0, 2), dtype=np.float64),
        source="marked_chromatic")
    return SamMaterialResult(
        material=material,
        prompt=prompt,
        sam_iou=float("nan"),
        selection_score=1.0,
        seed_recall=1.0,
        mask_area_px=int(np.count_nonzero(mask)),
        yellow_tip_xy=tip,
        marker_centers_xy=centers,
        marker_widths_px=widths,
        marker_confidence=confidence,
        marker_observed=observed,
        marker_raw_cluster_count=len(candidates),
        tip_source="wide_red_ring_center")
