"""Concurrent recording and timestamp pairing for multiple USB ZED rigs."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
import threading
import time
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CameraRigSpec:
    rig_id: str
    serial_number: int


@dataclass(frozen=True)
class PreviewFrame:
    rig_id: str
    sequence: int
    timestamp_ns: int
    svo_frame: int | None
    left_bgr: np.ndarray
    right_bgr: np.ndarray


@dataclass(frozen=True)
class RecordedFrame:
    svo_frame: int
    timestamp_ns: int


@dataclass(frozen=True)
class FramePair:
    reference_frame: int
    reference_timestamp_ns: int
    secondary_frame: int | None
    secondary_timestamp_ns: int | None

    @property
    def timestamp_delta_ns(self) -> int | None:
        if self.secondary_timestamp_ns is None:
            return None
        return int(
            self.secondary_timestamp_ns - self.reference_timestamp_ns)


def validate_rig_specs(specs) -> tuple[CameraRigSpec, ...]:
    """Validate stable rig names and unique, nonzero camera serial numbers."""
    normalized = tuple(specs)
    if len(normalized) not in (1, 2):
        raise ValueError("capture requires one or two configured ZED rigs")
    names = [item.rig_id for item in normalized]
    serials = [int(item.serial_number) for item in normalized]
    if any(not name or not name.replace("_", "").isalnum() for name in names):
        raise ValueError(
            "camera rig names must contain only letters, digits, and '_'")
    if len(set(names)) != len(names):
        raise ValueError("camera rig names must be unique")
    if any(serial <= 0 for serial in serials):
        raise ValueError("every camera rig requires a positive serial number")
    if len(set(serials)) != len(serials):
        raise ValueError("camera serial numbers must be unique")
    return normalized


def frame_pair_tolerance_ns(reference, secondary) -> int:
    """Allow slightly over half a frame period for unsynchronized cameras."""
    periods = []
    for records in (reference, secondary):
        if len(records) > 1:
            delta = np.diff([item.timestamp_ns for item in records])
            delta = delta[delta > 0]
            if delta.size:
                periods.append(float(np.median(delta)))
    return int(round(0.6 * max(periods))) if periods else 20_000_000


def pair_recorded_frames(
        reference: list[RecordedFrame],
        secondary: list[RecordedFrame],
        max_delta_ns: int | None = None) -> list[FramePair]:
    """Pair to the nearest unused peer without bridging a dropped frame."""
    if max_delta_ns is None:
        max_delta_ns = frame_pair_tolerance_ns(reference, secondary)
    pairs = []
    next_secondary = 0
    for ref in reference:
        if next_secondary >= len(secondary):
            pairs.append(FramePair(
                ref.svo_frame, ref.timestamp_ns, None, None))
            continue
        candidate = next_secondary
        while candidate + 1 < len(secondary):
            current_error = abs(
                secondary[candidate].timestamp_ns - ref.timestamp_ns)
            next_error = abs(
                secondary[candidate + 1].timestamp_ns - ref.timestamp_ns)
            if next_error > current_error:
                break
            candidate += 1
        match = secondary[candidate]
        if abs(match.timestamp_ns - ref.timestamp_ns) > int(max_delta_ns):
            pairs.append(FramePair(
                ref.svo_frame, ref.timestamp_ns, None, None))
            continue
        pairs.append(FramePair(
            ref.svo_frame, ref.timestamp_ns,
            match.svo_frame, match.timestamp_ns))
        next_secondary = candidate + 1
    return pairs


def write_frame_pairs(path, pairs: list[FramePair]) -> dict:
    """Write the cross-rig pairing sidecar and return timing statistics."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "reference_frame", "reference_timestamp_ns",
            "secondary_frame", "secondary_timestamp_ns",
            "timestamp_delta_ns",
        ])
        for pair in pairs:
            writer.writerow([
                pair.reference_frame,
                pair.reference_timestamp_ns,
                "" if pair.secondary_frame is None else pair.secondary_frame,
                "" if pair.secondary_timestamp_ns is None
                else pair.secondary_timestamp_ns,
                "" if pair.timestamp_delta_ns is None
                else pair.timestamp_delta_ns,
            ])
    deltas = np.asarray([
        pair.timestamp_delta_ns for pair in pairs
        if pair.timestamp_delta_ns is not None
    ], dtype=np.int64)
    if deltas.size:
        absolute_ms = np.abs(deltas.astype(np.float64)) * 1.0e-6
        timing = {
            "paired_frames": int(deltas.size),
            "unpaired_reference_frames": int(len(pairs) - deltas.size),
            "median_delta_ms": float(np.median(deltas) * 1.0e-6),
            "median_abs_delta_ms": float(np.median(absolute_ms)),
            "p95_abs_delta_ms": float(np.percentile(absolute_ms, 95)),
            "max_abs_delta_ms": float(np.max(absolute_ms)),
        }
    else:
        timing = {
            "paired_frames": 0,
            "unpaired_reference_frames": len(pairs),
            "median_delta_ms": None,
            "median_abs_delta_ms": None,
            "p95_abs_delta_ms": None,
            "max_abs_delta_ms": None,
        }
    return timing


class CameraCaptureWorker:
    """Continuously grab one camera without blocking the other USB camera."""

    def __init__(self, rig_id, camera, preview_stride=3):
        self.rig_id = str(rig_id)
        self.camera = camera
        self.preview_stride = max(1, int(preview_stride))
        self._sdk_lock = threading.RLock()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread = None
        self._latest = None
        self._sequence = 0
        self._error = None
        self._frame_index_file = None
        self._frame_index_writer = None
        self._svo_path = None
        self._frame_index_path = None
        self._records = []
        self._recording_grabs = 0
        self._repeated_timestamps = 0
        self._last_recording_timestamp_ns = None

    @property
    def error(self):
        return self._error

    @property
    def is_recording(self):
        return self.camera.is_recording

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"zed-capture-{self.rig_id}",
            daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._stop_event.is_set():
                with self._sdk_lock:
                    retrieve = (
                        not self.camera.is_recording
                        or self._recording_grabs % self.preview_stride == 0)
                    grabbed = self.camera.grab(retrieve=retrieve)
                    if grabbed is None:
                        continue
                    timestamp_ns, left_bgr, right_bgr = grabbed
                    svo_frame = None
                    if self._frame_index_writer is not None:
                        self._recording_grabs += 1
                        if timestamp_ns != self._last_recording_timestamp_ns:
                            svo_frame = len(self._records)
                            record = RecordedFrame(
                                svo_frame=svo_frame,
                                timestamp_ns=int(timestamp_ns))
                            self._records.append(record)
                            self._frame_index_writer.writerow([
                                record.svo_frame, record.timestamp_ns])
                            self._last_recording_timestamp_ns = timestamp_ns
                        else:
                            self._repeated_timestamps += 1
                if left_bgr is not None:
                    with self._condition:
                        self._sequence += 1
                        self._latest = PreviewFrame(
                            rig_id=self.rig_id,
                            sequence=self._sequence,
                            timestamp_ns=int(timestamp_ns),
                            svo_frame=svo_frame,
                            left_bgr=left_bgr,
                            right_bgr=right_bgr)
                        self._condition.notify_all()
        except Exception as exc:  # pragma: no cover - hardware path
            self._error = exc
            with self._condition:
                self._condition.notify_all()

    def latest(self):
        with self._condition:
            return self._latest

    def wait_for_preview(self, after_sequence=0, timeout=1.0):
        with self._condition:
            changed = self._condition.wait_for(
                lambda: (
                    self._error is not None
                    or (self._latest is not None
                        and self._latest.sequence > after_sequence)),
                timeout=float(timeout))
            if self._error is not None:
                raise RuntimeError(
                    f"{self.rig_id} capture failed: {self._error}"
                ) from self._error
            if (not changed or self._latest is None
                    or self._latest.sequence <= after_sequence):
                return None
            return self._latest

    def start_recording(self, svo_path, frame_index_path):
        with self._sdk_lock:
            if self.camera.is_recording:
                raise RuntimeError(f"{self.rig_id} is already recording")
            if not self.camera.start_recording(str(svo_path)):
                return False
            try:
                self._frame_index_file = open(
                    frame_index_path, "w", newline="", encoding="utf-8")
                self._frame_index_writer = csv.writer(
                    self._frame_index_file)
                self._frame_index_writer.writerow([
                    "svo_frame", "timestamp_ns"])
            except Exception:
                self.camera.stop_recording()
                raise
            self._svo_path = str(svo_path)
            self._frame_index_path = str(frame_index_path)
            self._records = []
            self._recording_grabs = 0
            self._repeated_timestamps = 0
            self._last_recording_timestamp_ns = None
            return True

    def stop_recording(self):
        with self._sdk_lock:
            if not self.camera.is_recording:
                return None
            recording_status = self.camera.stop_recording()
            if self._frame_index_file is not None:
                self._frame_index_file.close()
            self._frame_index_file = None
            self._frame_index_writer = None
            playable_count = None
            try:
                playable_count = self.camera.playable_svo_frame_count(
                    self._svo_path)
            except (OSError, RuntimeError):
                pass
            if playable_count is not None and playable_count < len(self._records):
                self._records = self._records[:playable_count]
                self._rewrite_frame_index()
            return {
                "rig_id": self.rig_id,
                "svo_path": self._svo_path,
                "frame_index_path": self._frame_index_path,
                "indexed_frames": len(self._records),
                "recording_grabs": self._recording_grabs,
                "repeated_timestamps": self._repeated_timestamps,
                "playable_frames": playable_count,
                "recording_status": recording_status,
                "records": list(self._records),
            }

    def _rewrite_frame_index(self):
        temporary = self._frame_index_path + ".finalizing"
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["svo_frame", "timestamp_ns"])
            writer.writerows(
                (record.svo_frame, record.timestamp_ns)
                for record in self._records)
        os.replace(temporary, self._frame_index_path)

    def close(self):
        if self.camera.is_recording:
            self.stop_recording()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.camera.close()


class MultiZedCapture:
    """Coordinate two independently grabbing ZED workers."""

    def __init__(
            self,
            cameras: Mapping[str, object],
            preview_stride=3,
            stale_timeout_s=2.0):
        if len(cameras) != 2:
            raise ValueError("MultiZedCapture currently requires exactly two rigs")
        self.workers = {
            rig_id: CameraCaptureWorker(
                rig_id, camera, preview_stride=preview_stride)
            for rig_id, camera in cameras.items()
        }
        self.rig_ids = tuple(self.workers)
        self.reference_rig = self.rig_ids[0]
        self.secondary_rig = self.rig_ids[1]
        self.stale_timeout_s = float(stale_timeout_s)
        now = time.monotonic()
        self._last_advance = {rig_id: now for rig_id in self.rig_ids}
        self._last_sequence = {rig_id: 0 for rig_id in self.rig_ids}

    def start(self):
        for worker in self.workers.values():
            worker.start()
        return self

    def wait_for_bundle(self, after_sequence=0, timeout=1.0):
        reference = self.workers[self.reference_rig].wait_for_preview(
            after_sequence=after_sequence, timeout=timeout)
        now = time.monotonic()
        for worker in self.workers.values():
            if worker.error is not None:
                raise RuntimeError(
                    f"{worker.rig_id} capture failed: {worker.error}"
                ) from worker.error
        if reference is None:
            if now - self._last_advance[self.reference_rig] > self.stale_timeout_s:
                raise RuntimeError(
                    f"{self.reference_rig} preview has not advanced for "
                    f"{self.stale_timeout_s:.1f}s")
            return None
        secondary = self.workers[self.secondary_rig].latest()
        if secondary is None:
            return None
        bundle = {
            self.reference_rig: reference,
            self.secondary_rig: secondary,
        }
        for rig_id, frame in bundle.items():
            if frame.sequence > self._last_sequence[rig_id]:
                self._last_sequence[rig_id] = frame.sequence
                self._last_advance[rig_id] = now
            elif now - self._last_advance[rig_id] > self.stale_timeout_s:
                raise RuntimeError(
                    f"{rig_id} preview has not advanced for "
                    f"{self.stale_timeout_s:.1f}s; stopping before frozen "
                    "camera data can be accepted")
        return bundle

    @property
    def is_recording(self):
        states = [worker.is_recording for worker in self.workers.values()]
        if any(states) and not all(states):
            raise RuntimeError("dual-camera recording state is inconsistent")
        return all(states)

    def start_recording(self, session_dir, session_id):
        started = []
        try:
            for rig_id, worker in self.workers.items():
                svo_path = Path(session_dir) / (
                    f"{rig_id}_{session_id}.svo2")
                index_path = Path(session_dir) / (
                    f"{rig_id}_frame_index.csv")
                if not worker.start_recording(svo_path, index_path):
                    raise RuntimeError(
                        f"could not start SVO recording for {rig_id}")
                started.append(worker)
        except Exception:
            for worker in reversed(started):
                worker.stop_recording()
            raise

    def stop_recording(self, session_dir):
        reports = {
            rig_id: worker.stop_recording()
            for rig_id, worker in self.workers.items()
        }
        reference = reports[self.reference_rig]
        secondary = reports[self.secondary_rig]
        if reference is None or secondary is None:
            raise RuntimeError("both cameras must be recording before stop")
        pairs = pair_recorded_frames(
            reference["records"], secondary["records"])
        pair_path = Path(session_dir) / "camera_frame_pairs.csv"
        timing = write_frame_pairs(pair_path, pairs)
        timing.update({
            "reference_rig": self.reference_rig,
            "secondary_rig": self.secondary_rig,
            "pair_file": pair_path.name,
            "pairing_tolerance_ms": frame_pair_tolerance_ns(
                reference["records"], secondary["records"]) * 1.0e-6,
        })
        for report in reports.values():
            report.pop("records", None)
        return reports, timing

    def close(self):
        for worker in self.workers.values():
            worker.close()
