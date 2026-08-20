"""SAM 2 front end for image-only catheter segmentation."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .materials import (
    MaterialCenterline,
    _sample_brightness,
    broad_blue_mask,
    detect_distal_boundary,
)
from .segmentation import Centerline, _longest_path


@dataclass(frozen=True)
class PromptSet:
    """SAM prompts in full-image pixel coordinates."""

    box_xyxy: np.ndarray
    positive_xy: np.ndarray
    negative_xy: np.ndarray
    source: str = "automatic_color_seed"


@dataclass(frozen=True)
class SamMaterialResult:
    material: MaterialCenterline
    prompt: PromptSet
    sam_iou: float
    selection_score: float
    seed_recall: float
    mask_area_px: int
    yellow_tip_xy: np.ndarray | None


@dataclass(frozen=True)
class AutomaticPromptResult:
    """Frame-local colour prompt artifacts that can be computed off-GPU."""

    prompt: PromptSet
    seed: np.ndarray
    tip: np.ndarray | None


@dataclass(frozen=True)
class _PreparedSamInput:
    bgr: np.ndarray
    roi: tuple[int, int, int, int]
    base_point: np.ndarray
    prompt: PromptSet
    seed: np.ndarray
    tip: np.ndarray | None
    crop_rgb: np.ndarray
    points: np.ndarray
    labels: np.ndarray
    box: np.ndarray


def load_prompt_overrides(path: str | Path | None) -> dict[str, dict[int, PromptSet]]:
    """Load optional manual prompts keyed by view and SVO frame.

    JSON format::

      {"left": {"1482": {"box": [x0,y0,x1,y1],
                           "positive": [[x,y], ...],
                           "negative": [[x,y], ...]}}}
    """
    if path is None:
        return {"left": {}, "right": {}}
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    output: dict[str, dict[int, PromptSet]] = {"left": {}, "right": {}}
    for view in output:
        for frame, entry in (document.get(view) or {}).items():
            output[view][int(frame)] = PromptSet(
                box_xyxy=np.asarray(entry["box"], dtype=np.float64),
                positive_xy=np.asarray(entry.get("positive", []), dtype=np.float64).reshape(-1, 2),
                negative_xy=np.asarray(entry.get("negative", []), dtype=np.float64).reshape(-1, 2),
                source="manual_json")
    return output


def _component_nearest_or_farthest(
        mask: np.ndarray,
        reference_xy: np.ndarray,
        farthest: bool = False,
        min_area: int = 5) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(mask > 0), 8)
    candidates = [
        index for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= min_area]
    if not candidates:
        return None, None
    distances = [np.linalg.norm(centroids[index] - reference_xy) for index in candidates]
    selected = candidates[int(np.argmax(distances) if farthest else np.argmin(distances))]
    return np.uint8(labels == selected) * 255, centroids[selected].astype(np.float64)


def _yellow_seed(
        bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point: np.ndarray,
        blue_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    x, y, w, h = roi
    sub = bgr[y:y + h, x:x + w]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, np.array([12, 65, 55]), np.array([48, 255, 255]))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blue_path = _path_with_anchors(
        blue_mask, roi, np.asarray(base_point, dtype=np.float64), None)
    if len(blue_path) < 2:
        return np.zeros((h, w), dtype=np.uint8), None
    distal_roi = np.asarray(
        [blue_path[-1, 1], blue_path[-1, 0]], dtype=np.float64)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(raw > 0), 8)
    best_component = None
    best_distance = float("inf")
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < 5:
            continue
        ys, xs = np.where(labels == component)
        distance = float(np.sqrt(np.min(
            (xs - distal_roi[0]) ** 2 + (ys - distal_roi[1]) ** 2)))
        if distance < best_distance:
            best_component, best_distance = component, distance
    # The printed yellow tip is physically adjacent to the cyan shaft. A more
    # distant yellow component is tape, a marker, or a fixture and must not be
    # used as an endpoint anchor.
    if best_component is None or best_distance > 35.0:
        return np.zeros((h, w), dtype=np.uint8), None
    selected = np.uint8(labels == best_component) * 255
    ys, xs = np.where(selected > 0)
    pixels = np.column_stack([xs, ys]).astype(np.float64)
    # A component centroid lies inside the printed yellow cap and therefore
    # shortens the reconstructed catheter. Estimate the distal edge along the
    # local shaft tangent instead. Averaging the leading decile is less noisy
    # than choosing one extreme segmentation pixel.
    tangent_span = max(2, min(12, len(blue_path) // 4))
    tangent_xy = np.array([
        blue_path[-1, 1] - blue_path[-tangent_span, 1],
        blue_path[-1, 0] - blue_path[-tangent_span, 0],
    ], dtype=np.float64)
    tangent_norm = float(np.linalg.norm(tangent_xy))
    if tangent_norm > 1e-6:
        tangent_xy /= tangent_norm
        score = pixels @ tangent_xy
        threshold = float(np.percentile(score, 90.0))
        tip_roi = np.mean(pixels[score >= threshold], axis=0)
    else:
        tip_roi = np.mean(pixels, axis=0)
    return selected, tip_roi + [x, y]


def _path_with_anchors(
        mask: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point: np.ndarray,
        tip_point: np.ndarray | None) -> np.ndarray:
    """Return an ordered skeleton path, preferring the base-to-tip route."""
    skeleton = skeletonize(mask > 0)
    points = np.argwhere(skeleton)
    if len(points) < 2:
        return points
    if tip_point is None:
        path = _longest_path(skeleton)
        x, y, _, _ = roi
        endpoints = np.array([
            [path[0, 1] + x, path[0, 0] + y],
            [path[-1, 1] + x, path[-1, 0] + y]])
        return path if np.argmin(np.linalg.norm(endpoints - base_point, axis=1)) == 0 else path[::-1]

    x, y, _, _ = roi
    base_rc = np.asarray([base_point[1] - y, base_point[0] - x])
    tip_rc = np.asarray([tip_point[1] - y, tip_point[0] - x])
    source = int(np.argmin(np.sum((points - base_rc) ** 2, axis=1)))
    lookup = {(int(r), int(c)): index for index, (r, c) in enumerate(points)}
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1)]
    degrees = np.asarray([
        sum((int(row + dr), int(col + dc)) in lookup
            for dr, dc in neighbors)
        for row, col in points])
    endpoint_indices = np.flatnonzero(degrees == 1)
    if len(endpoint_indices):
        endpoint_distance_squared = np.sum(
            (points[endpoint_indices] - tip_rc) ** 2, axis=1)
        closest = int(np.argmin(endpoint_distance_squared))
        # A stale temporal point is only an ordering hint.  If it is no longer
        # near a current graph endpoint, fall back to the current longest path.
        if endpoint_distance_squared[closest] > 35.0 ** 2:
            path = _longest_path(skeleton)
            endpoints = np.array([
                [path[0, 1] + x, path[0, 0] + y],
                [path[-1, 1] + x, path[-1, 0] + y]])
            return (path if np.argmin(
                np.linalg.norm(endpoints - base_point, axis=1)) == 0
                    else path[::-1])
        target = int(endpoint_indices[closest])
    else:
        target = int(np.argmin(np.sum((points - tip_rc) ** 2, axis=1)))
    parent = np.full(len(points), -1, dtype=np.int64)
    visited = np.zeros(len(points), dtype=bool)
    visited[source] = True
    queue = deque([source])
    while queue and not visited[target]:
        current = queue.popleft()
        row, col = points[current]
        for dr, dc in neighbors:
            nxt = lookup.get((int(row + dr), int(col + dc)))
            if nxt is not None and not visited[nxt]:
                visited[nxt] = True
                parent[nxt] = current
                queue.append(nxt)
    if not visited[target]:
        return _longest_path(skeleton)
    indices = []
    current = target
    while current >= 0:
        indices.append(current)
        if current == source:
            break
        current = int(parent[current])
    return points[np.asarray(indices[::-1], dtype=np.int64)]


def _automatic_prompts(
        bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        base_point: np.ndarray) -> tuple[PromptSet, np.ndarray, np.ndarray | None]:
    x, y, w, h = roi
    blue = broad_blue_mask(bgr, roi, base_point=base_point, min_area=20)
    yellow, tip = _yellow_seed(bgr, roi, base_point, blue)
    seed = cv2.bitwise_or(blue, yellow)
    seed = cv2.morphologyEx(
        seed, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
    ys, xs = np.where(seed > 0)
    if len(xs) < 10:
        raise ValueError("automatic_prompt_seed_missing")
    margin = 18
    box = np.array([
        max(x, x + int(xs.min()) - margin),
        max(y, y + int(ys.min()) - margin),
        min(x + w - 1, x + int(xs.max()) + margin),
        min(y + h - 1, y + int(ys.max()) + margin),
    ], dtype=np.float64)

    path = _path_with_anchors(blue if np.any(blue) else seed, roi, base_point, tip)
    if len(path) >= 2:
        sample_indices = np.unique(np.rint(
            np.linspace(0, len(path) - 1, min(6, len(path)))).astype(int))
        positive = np.column_stack([
            path[sample_indices, 1] + x,
            path[sample_indices, 0] + y]).astype(np.float64)
    else:
        positive = np.array([[float(np.mean(xs) + x), float(np.mean(ys) + y)]])
    if tip is not None and np.min(np.linalg.norm(positive - tip, axis=1)) > 3.0:
        positive = np.vstack([positive, tip])

    corners = np.array([
        [box[0], box[1]], [box[2], box[1]],
        [box[0], box[3]], [box[2], box[3]]], dtype=np.float64)
    negative = []
    distance = cv2.distanceTransform(np.uint8(seed == 0), cv2.DIST_L2, 5)
    for point in corners:
        col = int(np.clip(round(point[0] - x), 0, w - 1))
        row = int(np.clip(round(point[1] - y), 0, h - 1))
        if distance[row, col] >= 5.0:
            negative.append(point)
    return PromptSet(
        box_xyxy=box,
        positive_xy=positive,
        negative_xy=np.asarray(negative, dtype=np.float64).reshape(-1, 2),
    ), seed, tip


def _select_component(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(mask > 0), 8)
    if count <= 1:
        return np.uint8(mask > 0) * 255
    best, best_score = None, -np.inf
    for component in range(1, count):
        component_mask = labels == component
        overlap = int(np.count_nonzero(component_mask & (seed > 0)))
        area = int(stats[component, cv2.CC_STAT_AREA])
        score = overlap - 0.002 * area
        if score > best_score:
            best, best_score = component, score
    return np.uint8(labels == best) * 255


def material_centerline_from_mask(
        bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        mask: np.ndarray,
        base_point: np.ndarray,
        tip_point: np.ndarray | None) -> MaterialCenterline | None:
    """Convert an ROI-sized external mask into a material-aware centerline."""
    if not np.any(mask):
        return None
    x, y, _, _ = roi
    path = _path_with_anchors(mask, roi, base_point, tip_point)
    if len(path) < 2:
        return None
    points = np.column_stack([path[:, 1] + x, path[:, 0] + y]).astype(np.float64)
    if tip_point is not None and np.all(np.isfinite(tip_point)):
        tip = np.asarray(tip_point, dtype=np.float64)
        endpoint_gap = float(np.linalg.norm(points[-1] - tip))
        # The skeleton terminates near the middle of a finite-width yellow cap.
        # Preserve the observed skeleton and explicitly include the detected
        # distal cap edge as its endpoint when it is locally consistent.
        if 1.0 < endpoint_gap <= 35.0:
            points = np.vstack([points, tip])
    brightness = _sample_brightness(bgr, points)
    boundary, confidence, contrast, valid = detect_distal_boundary(brightness)
    distance = cv2.distanceTransform(np.uint8(mask > 0), cv2.DIST_L2, 5)
    radius = float(np.median(distance[path[:, 0], path[:, 1]]))
    return MaterialCenterline(
        centerline=Centerline(points, roi, np.uint8(mask > 0) * 255, radius),
        distal_boundary_index=int(boundary),
        distal_boundary_fraction=float(boundary / max(len(points) - 1, 1)),
        boundary_confidence=float(confidence),
        boundary_contrast=float(contrast),
        material_valid=bool(valid),
        brightness_profile=brightness)


class Sam2CatheterSegmenter:
    """Load SAM 2 once and segment one image or a batch of stereo views."""

    def __init__(
            self,
            model_config: str,
            checkpoint: str | Path,
            device: str = "cuda",
            postprocess_workers: int = 2):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:  # pragma: no cover - environment error
            raise ImportError(
                "SAM processor requires the official facebookresearch/sam2 checkout") from exc
        self.torch = torch
        self.device = device
        model = build_sam2(
            model_config, str(Path(checkpoint).resolve()), device=device,
            apply_postprocessing=True)
        self.predictor = SAM2ImagePredictor(model)
        self.postprocess_workers = max(1, int(postprocess_workers))
        self._postprocess_executor = (
            ThreadPoolExecutor(
                max_workers=self.postprocess_workers,
                thread_name_prefix="sam-mask-postprocess")
            if self.postprocess_workers > 1 else None)
        self.last_timing_s: dict[str, float] = {}

    def close(self) -> None:
        executor = getattr(self, "_postprocess_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._postprocess_executor = None

    def _synchronize(self) -> None:
        """Make CUDA stage timings include queued GPU work."""
        if str(self.device).startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    @staticmethod
    def _prepare(
            bgr: np.ndarray,
            roi: tuple[int, int, int, int],
            base_point: np.ndarray,
            prompt_override: PromptSet | None,
            automatic: AutomaticPromptResult | Exception | None = None,
            ) -> _PreparedSamInput:
        try:
            if isinstance(automatic, Exception):
                raise automatic
            if automatic is None:
                auto_prompt, seed, tip = _automatic_prompts(
                    bgr, roi, base_point)
            else:
                auto_prompt, seed, tip = (
                    automatic.prompt, automatic.seed, automatic.tip)
        except ValueError:
            if prompt_override is None:
                raise
            x, y, w, h = roi
            seed = np.zeros((h, w), dtype=np.uint8)
            for px, py in prompt_override.positive_xy:
                cv2.circle(
                    seed,
                    (int(round(px - x)), int(round(py - y))),
                    3, 255, -1)
            distances = np.linalg.norm(
                prompt_override.positive_xy - np.asarray(base_point), axis=1)
            tip = (
                None if len(distances) == 0
                else prompt_override.positive_xy[int(np.argmax(distances))])
            auto_prompt = prompt_override
        # The shaft colour can remain visible after the small yellow marker is
        # lost.  Keep the distal end of an ordered temporal/manual prompt as a
        # path-ordering anchor.  _path_with_anchors snaps it to the current SAM
        # skeleton, so this does not copy an old endpoint into the result.
        if tip is None and prompt_override is not None:
            distances = np.linalg.norm(
                prompt_override.positive_xy - np.asarray(base_point), axis=1)
            if len(distances):
                tip = prompt_override.positive_xy[int(np.argmax(distances))]
        prompt = prompt_override or auto_prompt
        x, y, w, h = roi
        crop_rgb = cv2.cvtColor(bgr[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
        points = np.vstack([prompt.positive_xy, prompt.negative_xy]) - [x, y]
        labels = np.concatenate([
            np.ones(len(prompt.positive_xy), dtype=np.int32),
            np.zeros(len(prompt.negative_xy), dtype=np.int32)])
        box = prompt.box_xyxy - [x, y, x, y]
        return _PreparedSamInput(
            bgr=bgr, roi=roi, base_point=np.asarray(base_point), prompt=prompt,
            seed=seed, tip=tip, crop_rgb=crop_rgb, points=points,
            labels=labels, box=box)

    @staticmethod
    def _finish(
            prepared: _PreparedSamInput,
            masks: np.ndarray,
            predicted_iou: np.ndarray) -> SamMaterialResult:
        seed_bool = prepared.seed > 0
        seed_area = max(1, int(np.count_nonzero(seed_bool)))
        dilated_seed = cv2.dilate(
            np.uint8(seed_bool),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        candidates = []
        for index, candidate in enumerate(masks > 0):
            area = max(1, int(np.count_nonzero(candidate)))
            recall = float(np.count_nonzero(candidate & seed_bool) / seed_area)
            precision_prior = float(
                np.count_nonzero(candidate & dilated_seed) / area)
            score = (
                0.45 * float(predicted_iou[index])
                + 0.35 * recall + 0.20 * precision_prior)
            candidates.append((score, index, recall))
        selection_score, selected_index, seed_recall = max(candidates)
        mask = np.uint8(masks[selected_index] > 0) * 255
        mask = cv2.bitwise_or(mask, prepared.seed)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = _select_component(mask, prepared.seed)
        material = material_centerline_from_mask(
            prepared.bgr, prepared.roi, mask, prepared.base_point,
            prepared.tip)
        if material is None:
            raise ValueError("sam_centerline_missing")
        return SamMaterialResult(
            material=material,
            prompt=prepared.prompt,
            sam_iou=float(predicted_iou[selected_index]),
            selection_score=float(selection_score),
            seed_recall=float(seed_recall),
            mask_area_px=int(np.count_nonzero(mask)),
            yellow_tip_xy=prepared.tip)

    def segment_batch(
            self,
            images: Sequence[np.ndarray],
            rois: Sequence[tuple[int, int, int, int]],
            base_points: Sequence[np.ndarray],
            prompt_overrides: Sequence[PromptSet | None] | None = None,
            automatic_prompts: Sequence[
                AutomaticPromptResult | Exception | None] | None = None,
            allow_finish_errors: bool = False,
            ) -> list[SamMaterialResult | Exception]:
        """Segment a batch with one batched image-encoder invocation."""
        count = len(images)
        if len(rois) != count or len(base_points) != count:
            raise ValueError("images, rois, and base_points must have equal lengths")
        overrides = (
            [None] * count if prompt_overrides is None else list(prompt_overrides))
        if len(overrides) != count:
            raise ValueError("prompt_overrides must match the image count")
        automatic = (
            [None] * count
            if automatic_prompts is None else list(automatic_prompts))
        if len(automatic) != count:
            raise ValueError("automatic_prompts must match the image count")
        timing: dict[str, float] = {}
        self.last_timing_s = timing

        started = time.perf_counter()
        prepared = [
            self._prepare(image, roi, base, override, auto)
            for image, roi, base, override, auto in zip(
                images, rois, base_points, overrides, automatic)]
        timing["sam_prompt_preprocess"] = time.perf_counter() - started

        autocast = (
            self.torch.autocast("cuda", dtype=self.torch.bfloat16)
            if str(self.device).startswith("cuda")
            else self.torch.autocast("cpu", enabled=False))
        with self.torch.inference_mode(), autocast:
            self._synchronize()
            started = time.perf_counter()
            self.predictor.set_image_batch([item.crop_rgb for item in prepared])
            self._synchronize()
            timing["sam_image_encoder"] = time.perf_counter() - started

            started = time.perf_counter()
            masks_batch, iou_batch, _ = self.predictor.predict_batch(
                point_coords_batch=[item.points for item in prepared],
                point_labels_batch=[item.labels for item in prepared],
                box_batch=[item.box for item in prepared],
                multimask_output=True)
            self._synchronize()
            timing["sam_prompt_decoder"] = time.perf_counter() - started

        started = time.perf_counter()
        finish_inputs = list(zip(prepared, masks_batch, iou_batch))
        def finish_one(values):
            try:
                return self._finish(*values)
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                return exc

        executor = getattr(self, "_postprocess_executor", None)
        if executor is None and len(finish_inputs) > 1 and getattr(
                self, "postprocess_workers", 2) > 1:
            executor = ThreadPoolExecutor(
                max_workers=int(getattr(self, "postprocess_workers", 2)),
                thread_name_prefix="sam-mask-postprocess")
            self._postprocess_executor = executor
        if executor is None or len(finish_inputs) == 1:
            finished = [finish_one(values) for values in finish_inputs]
        else:
            futures = [executor.submit(finish_one, values)
                       for values in finish_inputs]
            finished = [future.result() for future in futures]
        results = []
        for result in finished:
            try:
                if isinstance(result, Exception):
                    raise result
                results.append(result)
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                if not allow_finish_errors:
                    raise
                results.append(exc)
        timing["sam_mask_postprocess"] = time.perf_counter() - started
        return results

    def segment(
            self,
            bgr: np.ndarray,
            roi: tuple[int, int, int, int],
            base_point: np.ndarray,
            prompt_override: PromptSet | None = None) -> SamMaterialResult:
        return self.segment_batch(
            [bgr], [roi], [base_point], [prompt_override])[0]
