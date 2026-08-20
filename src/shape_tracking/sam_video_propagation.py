"""Chunked bidirectional SAM 2 mask propagation for stereo SVO sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import cv2
import h5py
import numpy as np

from .sam_segmentation import _automatic_prompts
from .sequence import load_collection_markers, select_frame_records
from .session import (
    SvoReader,
    find_svo,
    load_frame_index,
    load_session_registration,
    project_points,
)


def _mask_from_logits(logits) -> np.ndarray:
    array = logits.detach().float().cpu().numpy()
    return np.squeeze(array) > 0.0


def _run_view_chunk(
        predictor,
        torch,
        frame_dir: Path,
        anchor_images: dict[int, np.ndarray],
        base_xy: np.ndarray,
        anchor_stride: int) -> tuple[list[np.ndarray], np.ndarray]:
    frame_count = len(list(frame_dir.glob("*.jpg")))
    state = predictor.init_state(
        video_path=str(frame_dir), offload_video_to_cpu=True,
        offload_state_to_cpu=False, async_loading_frames=False)
    anchor_indices = sorted(set(
        [0, frame_count - 1]
        + list(range(0, frame_count, max(1, anchor_stride)))))
    prompted = []
    for frame_index in anchor_indices:
        image = anchor_images[frame_index]
        try:
            prompt, _, _ = _automatic_prompts(
                image, (0, 0, image.shape[1], image.shape[0]), base_xy)
        except ValueError:
            continue
        points = np.vstack([prompt.positive_xy, prompt.negative_xy])
        labels = np.concatenate([
            np.ones(len(prompt.positive_xy), dtype=np.int32),
            np.zeros(len(prompt.negative_xy), dtype=np.int32)])
        predictor.add_new_points_or_box(
            state, frame_idx=frame_index, obj_id=1,
            points=points.astype(np.float32), labels=labels,
            box=prompt.box_xyxy.astype(np.float32))
        prompted.append(frame_index)
    if not prompted:
        raise ValueError("sam_video_chunk_has_no_valid_anchor")

    forward: dict[int, np.ndarray] = {}
    for frame_index, object_ids, logits in predictor.propagate_in_video(
            state, start_frame_idx=min(prompted),
            max_frame_num_to_track=frame_count, reverse=False):
        object_index = list(object_ids).index(1)
        forward[int(frame_index)] = _mask_from_logits(logits[object_index])
    reverse: dict[int, np.ndarray] = {}
    for frame_index, object_ids, logits in predictor.propagate_in_video(
            state, start_frame_idx=max(prompted),
            max_frame_num_to_track=frame_count, reverse=True):
        object_index = list(object_ids).index(1)
        reverse[int(frame_index)] = _mask_from_logits(logits[object_index])

    masks = []
    agreement = np.full(frame_count, np.nan, dtype=np.float32)
    for index in range(frame_count):
        first = forward.get(index)
        second = reverse.get(index)
        if first is None and second is None:
            height, width = next(iter(anchor_images.values())).shape[:2]
            masks.append(np.zeros((height, width), dtype=bool))
        elif first is None:
            masks.append(second)
        elif second is None:
            masks.append(first)
        else:
            intersection = np.count_nonzero(first & second)
            union = np.count_nonzero(first | second)
            agreement[index] = intersection / max(union, 1)
            image = cv2.imread(str(frame_dir / f"{index:06d}.jpg"))
            try:
                _, seed, _ = _automatic_prompts(
                    image, (0, 0, image.shape[1], image.shape[0]), base_xy)
                seed = seed > 0
            except ValueError:
                seed = np.zeros_like(first, dtype=bool)

            def candidate_score(candidate):
                recall = np.count_nonzero(candidate & seed) / max(
                    np.count_nonzero(seed), 1)
                precision = np.count_nonzero(candidate & seed) / max(
                    np.count_nonzero(candidate), 1)
                return 0.7 * recall + 0.3 * precision

            selected = (
                first if candidate_score(first) >= candidate_score(second)
                else second)
            # Preserve current-frame material pixels while using video memory
            # to bridge glare and low-saturation gaps.
            masks.append(selected | seed)
    predictor.reset_state(state)
    return masks, agreement


def propagate_session_masks(
        session_path: str | Path,
        checkpoint: str | Path,
        output_path: str | Path,
        model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        window: str = "trajectory",
        start_ns: int | None = None,
        end_ns: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
        chunk_frames: int = 300,
        anchor_stride: int = 30,
        device: str = "cuda") -> dict:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    session = Path(session_path).resolve()
    registration = load_session_registration(session, require_em=False)
    records = select_frame_records(
        load_frame_index(session), load_collection_markers(session),
        window, start_ns, end_ns, stride, max_frames)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(
        model_config, str(Path(checkpoint).resolve()), device=device)
    base_mm = np.zeros(3, dtype=np.float64)
    base_points = {
        "left": project_points(
            registration.K, registration.left_camera_T_base, base_mm)[0][0],
        "right": project_points(
            registration.K, registration.right_camera_T_base, base_mm)[0][0],
    }
    rois = {
        "left": registration.roi_left_xywh,
        "right": registration.roi_right_xywh,
    }
    started = time.perf_counter()
    with h5py.File(output, "w") as cache:
        cache.attrs["schema_version"] = 1
        cache.attrs["implementation"] = "sam2_bidirectional_video"
        cache.attrs["model_config"] = model_config
        cache.attrs["checkpoint"] = str(Path(checkpoint).resolve())
        cache.attrs["chunk_frames"] = int(chunk_frames)
        cache.attrs["anchor_stride"] = int(anchor_stride)
        cache.create_dataset(
            "svo_frame", data=np.asarray([r.svo_frame for r in records], np.int64))
        cache.create_dataset(
            "timestamp_ns", data=np.asarray([r.timestamp_ns for r in records], np.int64))
        for view, roi in rois.items():
            group = cache.create_group(view)
            group.attrs["roi_xywh"] = roi
            packed_size = (roi[2] * roi[3] + 7) // 8
            group.create_dataset(
                "mask_packbits", (len(records), packed_size), dtype=np.uint8,
                compression="gzip", compression_opts=1)
            group.create_dataset(
                "forward_reverse_iou", (len(records),), dtype=np.float32,
                fillvalue=np.nan)

        with SvoReader(find_svo(session)) as svo:
            iterator = iter(svo.iter_records(records))
            output_index = 0
            while output_index < len(records):
                count = min(max(2, chunk_frames), len(records) - output_index)
                with tempfile.TemporaryDirectory(prefix="sam2_video_") as temp:
                    root = Path(temp)
                    directories = {view: root / view for view in rois}
                    for directory in directories.values():
                        directory.mkdir()
                    anchors: dict[str, dict[int, np.ndarray]] = {
                        "left": {}, "right": {}}
                    anchor_indices = set(
                        [0, count - 1]
                        + list(range(0, count, max(1, anchor_stride))))
                    for local_index in range(count):
                        _, _, left, right = next(iterator)
                        for view, image in (("left", left), ("right", right)):
                            x, y, width, height = rois[view]
                            crop = image[y:y + height, x:x + width]
                            cv2.imwrite(
                                str(directories[view] / f"{local_index:06d}.jpg"),
                                crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if local_index in anchor_indices:
                                anchors[view][local_index] = crop.copy()
                    for view in ("left", "right"):
                        x, y, _, _ = rois[view]
                        local_base = base_points[view] - np.array([x, y])
                        with torch.inference_mode(), torch.autocast(
                                "cuda", dtype=torch.bfloat16,
                                enabled=str(device).startswith("cuda")):
                            masks, agreement = _run_view_chunk(
                                predictor, torch, directories[view],
                                anchors[view], local_base, anchor_stride)
                        for local_index, mask in enumerate(masks):
                            packed = np.packbits(mask.reshape(-1), bitorder="little")
                            cache[f"{view}/mask_packbits"][
                                output_index + local_index] = packed
                        cache[f"{view}/forward_reverse_iou"][
                            output_index:output_index + count] = agreement
                    output_index += count
                    cache.flush()
                    print(
                        f"[{output_index}/{len(records)}] propagated masks",
                        flush=True)
    return {
        "output": str(output), "frame_count": len(records),
        "elapsed_s": time.perf_counter() - started,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument(
        "--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", default="trajectory")
    parser.add_argument("--start-ns", type=int)
    parser.add_argument("--end-ns", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--chunk-frames", type=int, default=300)
    parser.add_argument("--anchor-stride", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(json.dumps(propagate_session_masks(
        args.session, args.sam_checkpoint, args.output,
        model_config=args.sam_config, window=args.window,
        start_ns=args.start_ns, end_ns=args.end_ns, stride=args.stride,
        max_frames=args.max_frames, chunk_frames=args.chunk_frames,
        anchor_stride=args.anchor_stride, device=args.device), indent=2))


if __name__ == "__main__":
    main()
