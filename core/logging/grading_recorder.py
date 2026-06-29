"""
core/logging/grading_recorder.py
================================
Per-apple grading export — cropped annotated patches + CSV.

OpenCV encode and disk I/O run off the GUI thread.  The inference hot path
only enqueues lightweight track snapshots; under load, pending batches are
capped so logging never starves the tracker.

Output layout::

    {session}/Lane{L}/Apple{N}/processed/frame_XXX.jpg   # YOLO-combined input crop + grade label
    {session}/Lane{L}/Apple{N}/source0/frame_XXX.jpg     # raw ch1 (Color) crop — no annotations
    {session}/Lane{L}/Apple{N}/source1/frame_XXX.jpg     # raw ch2 (NIR1)  crop — no annotations
    {session}/Lane{L}/Apple{N}/source2/frame_XXX.jpg     # raw ch3 (NIR2)  crop — no annotations
    {session}/Lane{L}/Apple{N}.csv
"""

from __future__ import annotations

import csv
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.log import get_logger

log = get_logger(__name__)

CLASS_NAMES = ("Fresh", "Processing", "Cull")
_CLASS_COLORS = [
    (46, 204, 113),
    (241, 196, 15),
    (231, 76, 60),
]

# Max in-flight batches (frame + encode).  Extra frames are skipped for logging
# only — tracker / counting are unaffected.
_MAX_PENDING_BATCHES = 2
_DEFAULT_MAX_CROPS = 8
_DEFAULT_HEAVY_THRESHOLD = 12


def _tracks_for_logging(active: list[dict], max_n: int) -> list[dict]:
    """Prefer counted apples (seq_id) when capping crops per frame."""
    with_id = [t for t in active if t.get("seq_id") is not None]
    without = [t for t in active if t.get("seq_id") is None]
    out = with_id[:max_n]
    if len(out) < max_n:
        out.extend(without[: max_n - len(out)])
    return out


@dataclass
class _CsvRow:
    frame_idx: int
    detector_class: str
    confidence: float


@dataclass
class _PendingMeta:
    lane: int
    raw_cls: int
    raw_conf: float
    detected_jpeg:  bytes | None = None   # detected: annotated crop with box + label
    raw_crop_jpegs: tuple[bytes | None, bytes | None, bytes | None] = field(
        default_factory=lambda: (None, None, None)
    )  # ch1/ch2/ch3 crops saved under Apple{N}/ch1/, ch2/, ch3/


@dataclass
class _AppleState:
    seq_id: int
    lane: int
    frame_idx: int = 0
    rows: list[_CsvRow] = field(default_factory=list)
    finalized: bool = False
    final_grade: str | None = None
    final_confidence: float | None = None


@dataclass
class _PreparedTrack:
    track_id: int
    seq_id: int | None
    meta: _PendingMeta


@dataclass
class _WriteJob:
    path: Path
    jpeg_bytes: bytes


class GradingRecorder:
    """
    Ordered command queue: every ``record_batch`` is processed before the
    next ``on_grade_committed`` for that frame, eliminating races.
    """

    def __init__(
        self,
        image_format: str = "jpg",
        jpeg_quality: int = 92,
        save_frames: bool = True,
        save_raw_frames: bool = True,
        crop_padding_frac: float = 0.20,
        crop_max_dim: int = 512,
        raw_crop_max_dim: int = 256,    # smaller than processed — faster encode, less I/O
        raw_frame_stride: int = 1,       # save raw crops every Nth logged frame (1=every frame)
        save_raw_full_frames: bool = False,   # save full-resolution frames (all 3 channels)
        save_detected_frames: bool = False,   # save full-res YOLO composite + detection boxes
        max_pending_batches: int = _MAX_PENDING_BATCHES,
        max_crops_per_batch: int = _DEFAULT_MAX_CROPS,
        heavy_threshold: int = _DEFAULT_HEAVY_THRESHOLD,
    ) -> None:
        self._image_ext = image_format.lower().lstrip(".")
        self._jpeg_quality = jpeg_quality
        self._save_frames = save_frames
        self._save_raw_frames = save_raw_frames
        self._crop_pad = crop_padding_frac
        self._crop_max_dim = crop_max_dim
        self._raw_crop_max_dim = max(64, raw_crop_max_dim)
        self._raw_frame_stride = max(1, raw_frame_stride)
        self._raw_frame_tick = 0          # counts logged frames for stride gating
        self._save_raw_full_frames = save_raw_full_frames
        self._save_detected_frames = save_detected_frames
        self._raw_full_frame_counter = 0  # monotonic counter for raw_frames/ filenames
        self._detected_frame_counter = 0  # monotonic counter for detected_frames/ filenames
        self._dropped_raw_frames = 0
        self._dropped_det_frames = 0
        self._max_crops = max(1, max_crops_per_batch)
        self._heavy_threshold = max(1, heavy_threshold)

        self._lock = threading.Lock()
        self._session_dir: Path | None = None
        self._apples: dict[int, _AppleState] = {}
        self._track_buffer: dict[int, list[_PendingMeta]] = {}
        self._track_to_apple: dict[int, int] = {}
        self._dirs_made: set[Path] = set()
        self._active = False
        self._saved_images = 0
        self._dropped_batches = 0

        self._batch_slots = threading.Semaphore(max_pending_batches)
        self._raw_slots  = threading.Semaphore(6)   # cap in-flight full-res raw encodes
        self._det_slots  = threading.Semaphore(3)   # cap in-flight detected-frame encodes
        self._cmd_q: queue.SimpleQueue = queue.SimpleQueue()
        self._write_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="log-wr")
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def start_session(self, base_dir: Path | str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(base_dir) / ts
        session_dir.mkdir(parents=True, exist_ok=True)

        done = threading.Event()
        self._cmd_q.put(("start", session_dir, done))
        done.wait()
        self._session_dir = session_dir
        self._saved_images = 0
        self._dropped_batches = 0
        self._dropped_raw_frames = 0
        self._dropped_det_frames = 0
        self._active = True
        log.info("GradingRecorder session started: %s", session_dir)
        return session_dir

    def acquire_batch_slot(self) -> bool:
        """Non-blocking.  Caller must submit_batch() or release_batch_slot()."""
        if not self._active:
            return False
        if not self._batch_slots.acquire(blocking=False):
            self._dropped_batches += 1
            if self._dropped_batches in (1, 50, 200):
                log.warning(
                    "GradingRecorder: dropped %d logging batches (worker backlog)",
                    self._dropped_batches,
                )
            return False
        return True

    def release_batch_slot(self) -> None:
        self._batch_slots.release()

    def submit_batch(
        self,
        frame_bgr: np.ndarray,
        tracks: list[dict],
        raw_frames: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None = None,
    ) -> None:
        """
        Enqueue a batch; batch slot must already be acquired.

        raw_frames: optional (ch1, ch2, ch3) numpy arrays — the unprocessed
        source frames captured at the same instant as frame_bgr.  When
        supplied, raw crops are saved to source0/, source1/, source2/ under
        each Apple folder.  Pass None to skip raw-frame saving.
        """
        if not tracks or frame_bgr is None:
            self.release_batch_slot()
            return
        capped = (
            _tracks_for_logging(tracks, self._max_crops)
            if len(tracks) > self._max_crops else tracks
        )
        snaps = tuple(_snapshot(t) for t in capped)
        self._cmd_q.put(("batch", frame_bgr, snaps, raw_frames))

    def record_batch(
        self,
        frame_bgr: np.ndarray,
        tracks: list[dict],
        raw_frames: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None = None,
    ) -> None:
        """Legacy entry — acquires slot then submits (used off hot path only)."""
        if not self._active or frame_bgr is None or not tracks:
            return
        if not self.acquire_batch_slot():
            return
        self.submit_batch(frame_bgr, tracks, raw_frames)

    def submit_raw_frame(
        self,
        ch1: np.ndarray | None,
        ch2: np.ndarray | None,
        ch3: np.ndarray | None,
    ) -> None:
        """
        Enqueue a full-resolution raw frame save from all 3 camera channels.
        Non-blocking — drops silently when the worker is backlogged.
        Thread-safe; may be called from any thread.

        Output layout::
            {session}/raw_frames/ch1/frame_000001.jpg
            {session}/raw_frames/ch2/frame_000001.jpg
            {session}/raw_frames/ch3/frame_000001.jpg
        """
        if not self._active or not self._save_raw_full_frames:
            return
        if not self._raw_slots.acquire(blocking=False):
            self._dropped_raw_frames += 1
            if self._dropped_raw_frames in (1, 50, 200):
                log.warning(
                    "GradingRecorder: dropped %d raw-frame batches (encoder backlog)",
                    self._dropped_raw_frames,
                )
            return
        # Pass references — caller must NOT mutate arrays after this call.
        # VideoWorker and mock camera create new arrays each frame, so this is safe.
        self._cmd_q.put(("raw_frame", ch1, ch2, ch3))

    def submit_detected_frame(
        self,
        frame_bgr: np.ndarray,
        active: list[dict],
    ) -> None:
        """
        Enqueue a full-resolution YOLO composite frame with detection boxes overlaid.
        Non-blocking — drops silently when the worker is backlogged.
        Should be called on the inference thread.

        Output layout::
            {session}/detected_frames/frame_000001.jpg
        """
        if not self._active or not self._save_detected_frames:
            return
        if not active:
            return
        if not self._det_slots.acquire(blocking=False):
            self._dropped_det_frames += 1
            return
        # Copy on inference thread so the worker has exclusive ownership.
        self._cmd_q.put(("det_frame", frame_bgr.copy(), list(active)))

    def set_save_options(
        self,
        save_raw_full_frames: bool | None = None,
        save_detected_frames: bool | None = None,
        save_frames: bool | None = None,
        save_raw_frames: bool | None = None,
    ) -> None:
        """
        Update save flags live without restarting the session.
        Flag changes take effect on the next submitted frame.
        Thread-safe: Python bool assignment is atomic under the GIL.
        """
        if save_raw_full_frames is not None:
            self._save_raw_full_frames = save_raw_full_frames
        if save_detected_frames is not None:
            self._save_detected_frames = save_detected_frames
        if save_frames is not None:
            self._save_frames = save_frames
        if save_raw_frames is not None:
            self._save_raw_frames = save_raw_frames

    def on_grade_committed(
        self,
        seq_id: int,
        lane: int,
        class_name: str,
        confidence: float,
        track_id: int = -1,
    ) -> None:
        if not self._active:
            return
        self._cmd_q.put(("commit", seq_id, lane, class_name, confidence, track_id))

    def stop_session(self) -> None:
        if not self._active:
            return
        done = threading.Event()
        self._cmd_q.put(("stop", done))
        done.wait()
        self._active = False
        log.info(
            "GradingRecorder stopped — %d images saved (%d batches dropped)",
            self._saved_images, self._dropped_batches,
        )

    # ── Worker (single ordered loop) ──────────────────────────────────────────

    def _run_worker(self) -> None:
        while True:
            cmd = self._cmd_q.get()
            kind = cmd[0]
            try:
                if kind == "start":
                    _, session_dir, done = cmd
                    with self._lock:
                        self._on_start(session_dir)
                    done.set()
                elif kind == "batch":
                    _, frame, tracks, raw_frames = cmd
                    try:
                        self._on_batch(frame, tracks, raw_frames)
                    finally:
                        self._batch_slots.release()
                elif kind == "commit":
                    _, seq_id, lane, class_name, confidence, track_id = cmd
                    writes: list[_WriteJob] = []
                    with self._lock:
                        writes.extend(self._on_commit(
                            seq_id, lane, class_name, confidence, track_id,
                        ))
                    self._flush_writes(writes)
                elif kind == "stop":
                    _, done = cmd
                    writes: list[_WriteJob] = []
                    with self._lock:
                        writes.extend(self._on_stop())
                    self._flush_writes(writes)
                    done.set()
                elif kind == "raw_frame":
                    _, c1, c2, c3 = cmd
                    try:
                        self._on_raw_frame(c1, c2, c3)
                    finally:
                        self._raw_slots.release()
                elif kind == "det_frame":
                    _, frame, active = cmd
                    try:
                        self._on_detected_frame(frame, active)
                    finally:
                        self._det_slots.release()
            except Exception:
                log.exception("GradingRecorder worker error on %s", kind)
                if kind == "batch":
                    self._batch_slots.release()
                elif kind == "raw_frame":
                    self._raw_slots.release()
                elif kind == "det_frame":
                    self._det_slots.release()

    def _on_batch(
        self,
        frame: np.ndarray,
        tracks: tuple[dict, ...],
        raw_frames: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None,
    ) -> None:
        # ── Step 1: Encode crops WITHOUT holding the lock ─────────────────────
        # cv2.imencode releases the GIL, so the inference thread is not starved.
        # The lock is only acquired below for fast dict bookkeeping (~1 ms).
        self._raw_frame_tick += 1
        # Processed Frames (ch1/ch2/ch3 crops) respect the stride setting
        save_processed = (
            self._save_frames
            and raw_frames is not None
            and self._raw_frame_tick % self._raw_frame_stride == 0
        )
        prepared: list[_PreparedTrack] = []
        for t in tracks:
            item = self._prepare_track(
                frame, t,
                raw_frames if save_processed else None,
            )
            if item is not None:
                prepared.append(item)

        # ── Step 2: Apply to state structures — fast dict ops under lock ──────
        writes: list[_WriteJob] = []
        with self._lock:
            for item in prepared:
                writes.extend(self._apply_prepared(item))

        self._flush_writes(writes)

    def _prepare_track(
        self,
        frame: np.ndarray,
        t: dict,
        raw_frames: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None = None,
    ) -> _PreparedTrack | None:
        if t.get("box") is None:
            return None
        meta = _PendingMeta(
            lane=int(t["lane"]),
            raw_cls=int(t["raw_class_id"]),
            raw_conf=float(t["raw_conf"]),
        )
        # Processed Frames = ch1/ch2/ch3 raw channel crops (no YOLO composite, no annotation)
        if raw_frames is not None:
            meta.raw_crop_jpegs = tuple(
                self._encode_raw_crop(rf, t) if rf is not None else None
                for rf in raw_frames
            )
        # Detected Frames = annotated YOLO composite crop (bounding box + grade label)
        if self._save_detected_frames:
            meta.detected_jpeg = self._encode_crop(frame, t)
        return _PreparedTrack(
            track_id=int(t["track_id"]),
            seq_id=t.get("seq_id"),
            meta=meta,
        )

    def _apply_prepared(self, item: _PreparedTrack) -> list[_WriteJob]:
        """Must be called under self._lock."""
        track_id = item.track_id
        seq_id = item.seq_id
        meta = item.meta
        writes: list[_WriteJob] = []

        if seq_id is not None:
            sid = int(seq_id)
            self._track_to_apple[track_id] = sid
            if track_id in self._track_buffer:
                for bm in self._track_buffer.pop(track_id):
                    writes.extend(self._append_row(sid, bm))
            writes.extend(self._append_row(sid, meta))
        else:
            self._track_buffer.setdefault(track_id, []).append(meta)

        return writes

    def _append_row(self, seq_id: int, meta: _PendingMeta) -> list[_WriteJob]:
        """Must be called under self._lock.  Returns a list of write jobs."""
        state = self._apples.get(seq_id)
        if state is None:
            state = _AppleState(seq_id=seq_id, lane=meta.lane)
            self._apples[seq_id] = state

        if state.finalized:
            return []

        state.frame_idx += 1

        # Only accumulate CSV rows when Detected Frames (grading) is active
        if self._save_detected_frames:
            cls_name = (
                CLASS_NAMES[meta.raw_cls]
                if 0 <= meta.raw_cls < len(CLASS_NAMES)
                else str(meta.raw_cls)
            )
            state.rows.append(_CsvRow(
                frame_idx=state.frame_idx,
                detector_class=cls_name,
                confidence=meta.raw_conf,
            ))

        jobs: list[_WriteJob] = []
        apple_dir = self._apple_dir(state)
        fname = f"frame_{state.frame_idx:03d}.{self._image_ext}"

        # Detected Frames: annotated crop → Apple{N}/detected/
        if self._save_detected_frames and meta.detected_jpeg is not None:
            jobs.append(_WriteJob(apple_dir / "detected" / fname, meta.detected_jpeg))

        # Processed Frames: ch1/ch2/ch3 raw channel crops → Apple{N}/ch1/, ch2/, ch3/
        _CH_NAMES = ("ch1", "ch2", "ch3")
        for ch_name, raw_jpeg in zip(_CH_NAMES, meta.raw_crop_jpegs):
            if raw_jpeg is not None:
                jobs.append(_WriteJob(apple_dir / ch_name / fname, raw_jpeg))

        return jobs

    def _on_start(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._apples.clear()
        self._track_buffer.clear()
        self._track_to_apple.clear()
        self._dirs_made.clear()
        self._raw_full_frame_counter = 0
        self._detected_frame_counter = 0

    def _on_commit(
        self,
        seq_id: int,
        lane: int,
        class_name: str,
        confidence: float,
        track_id: int,
    ) -> list[_WriteJob]:
        """Must be called under self._lock."""
        writes: list[_WriteJob] = []
        if track_id >= 0 and track_id in self._track_buffer:
            for bm in self._track_buffer.pop(track_id):
                writes.extend(self._append_row(seq_id, bm))

        state = self._apples.get(seq_id)
        if state is None:
            state = _AppleState(seq_id=seq_id, lane=lane)
            self._apples[seq_id] = state

        state.final_grade = class_name
        state.final_confidence = float(confidence)
        state.finalized = True
        if self._save_detected_frames:   # CSV only when grading is active
            self._write_csv(state, finalize=True)
        return writes

    def _on_stop(self) -> list[_WriteJob]:
        writes: list[_WriteJob] = []
        for track_id, buffered in list(self._track_buffer.items()):
            sid = self._track_to_apple.get(track_id)
            if sid is None:
                continue
            for bm in buffered:
                writes.extend(self._append_row(sid, bm))
        self._track_buffer.clear()
        self._track_to_apple.clear()

        for state in self._apples.values():
            if self._save_detected_frames:   # CSV only when grading is active
                self._write_csv(
                    state,
                    finalize=True,
                    incomplete=not state.finalized,
                )

        self._apples.clear()
        log.info(
            "GradingRecorder session data flushed  "
            "(raw_frames_dropped=%d  det_frames_dropped=%d)",
            self._dropped_raw_frames, self._dropped_det_frames,
        )
        return writes

    def _on_raw_frame(
        self,
        ch1: np.ndarray | None,
        ch2: np.ndarray | None,
        ch3: np.ndarray | None,
    ) -> None:
        """
        Encode and schedule disk writes for full-resolution raw frames.
        Called on the worker thread — cv2.imencode releases the GIL.
        Output: {session}/raw_frames/ch1/, ch2/, ch3/
        """
        if self._session_dir is None:
            return
        self._raw_full_frame_counter += 1
        n    = self._raw_full_frame_counter
        fname = f"frame_{n:06d}.{self._image_ext}"
        for ch_name, frame in (("ch1", ch1), ("ch2", ch2), ("ch3", ch3)):
            if frame is None:
                continue
            img = _normalize_to_bgr(frame)
            if img is None:
                continue
            ok, buf = cv2.imencode(
                f".{self._image_ext}", img,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            if ok:
                path = self._session_dir / "raw_frames" / ch_name / fname
                self._write_pool.submit(self._write_jpeg, path, buf.tobytes())

    def _on_detected_frame(
        self,
        frame: np.ndarray,
        active: list[dict],
    ) -> None:
        """
        Draw detection boxes on the YOLO composite and schedule a disk write.
        Called on the worker thread.
        Output: {session}/detected_frames/
        """
        if self._session_dir is None:
            return
        self._detected_frame_counter += 1
        n     = self._detected_frame_counter
        fname = f"frame_{n:06d}.{self._image_ext}"

        annotated = _normalize_to_bgr(frame)
        if annotated is None:
            return

        for t in active:
            box = t.get("box")
            if box is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            cls   = int(t.get("class_id", 0))
            conf  = float(t.get("conf", 0.0))
            seq   = t.get("seq_id")
            lane  = int(t.get("lane", 0))
            color = _CLASS_COLORS[cls % len(_CLASS_COLORS)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            id_part = f"#{seq}" if seq is not None else "?"
            name    = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            label   = f"{id_part} {name} {conf * 100:.0f}% L{lane}"
            fs, thick = 0.45, 1
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            lx = x1
            ly = max(lh + 4, y1 - 4)
            cv2.rectangle(annotated, (lx, ly - lh - 3), (lx + lw + 4, ly + 2), color, -1)
            cv2.putText(
                annotated, label, (lx + 2, ly - 1),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick, cv2.LINE_AA,
            )

        ok, buf = cv2.imencode(
            f".{self._image_ext}", annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if ok:
            path = self._session_dir / "detected_frames" / fname
            self._write_pool.submit(self._write_jpeg, path, buf.tobytes())

    def _flush_writes(self, jobs: list[_WriteJob]) -> None:
        for job in jobs:
            self._write_pool.submit(self._write_jpeg, job.path, job.jpeg_bytes)

    # ── Crop rendering ────────────────────────────────────────────────────────

    def _encode_crop(self, frame: np.ndarray, track: dict) -> bytes | None:
        """Encode the YOLO-combined (processed) crop with annotation overlay."""
        crop = self._render_crop(frame, track)
        if crop is None:
            return None
        ok, buf = cv2.imencode(
            f".{self._image_ext}", crop,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        return buf.tobytes() if ok else None

    def _encode_raw_crop(self, frame: np.ndarray, track: dict) -> bytes | None:
        """
        Encode a clean crop from a raw source frame — no annotation overlay.
        The crop region mirrors the processed crop (same box + padding) so
        all four images (processed + source0/1/2) are spatially aligned.
        """
        x1, y1, x2, y2 = (int(v) for v in track["box"])
        h, w = frame.shape[:2]
        if x2 <= x1 or y2 <= y1:
            return None

        bw, bh = x2 - x1, y2 - y1
        pad = int(self._crop_pad * max(bw, bh))
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        crop = frame[cy1:cy2, cx1:cx2]
        # Ensure 3-channel for consistent JPEG output
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            crop = crop[:, :, :3]

        # Resize if larger than raw_crop_max_dim (smaller default than processed)
        ch, cw = crop.shape[:2]
        max_dim = max(ch, cw)
        if max_dim > self._raw_crop_max_dim:
            scale = self._raw_crop_max_dim / max_dim
            crop = cv2.resize(
                crop,
                (int(cw * scale), int(ch * scale)),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(
            f".{self._image_ext}", crop,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        return buf.tobytes() if ok else None

    def _render_crop(self, frame: np.ndarray, track: dict) -> np.ndarray | None:
        x1, y1, x2, y2 = (int(v) for v in track["box"])
        h, w = frame.shape[:2]
        if x2 <= x1 or y2 <= y1:
            return None

        bw, bh = x2 - x1, y2 - y1
        pad = int(self._crop_pad * max(bw, bh))
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        if frame.ndim == 2:
            crop = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_GRAY2BGR)
        else:
            crop = frame[cy1:cy2, cx1:cx2]

        rx1, ry1 = x1 - cx1, y1 - cy1
        rx2, ry2 = x2 - cx1, y2 - cy1

        cls = int(track["class_id"])
        conf = float(track["conf"])
        seq = track.get("seq_id")
        lane = int(track["lane"])
        eligible = bool(track.get("eligible", True))

        color = _CLASS_COLORS[cls % len(_CLASS_COLORS)]
        draw_color = color if eligible else (120, 120, 120)
        cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), draw_color, 2)

        id_part = f"#{seq}" if seq is not None else "?"
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        label = f"{id_part} {name} {conf * 100:.0f}% L{lane}"
        fs, thick = 0.50, 1
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        lx = max(0, rx1)
        ly = max(lh + 4, ry1 - 4)
        cv2.rectangle(crop, (lx, ly - lh - 3), (lx + lw + 4, ly + 2), draw_color, -1)
        cv2.putText(
            crop, label, (lx + 2, ly - 1),
            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick, cv2.LINE_AA,
        )

        ch, cw = crop.shape[:2]
        max_dim = max(ch, cw)
        if max_dim > self._crop_max_dim:
            scale = self._crop_max_dim / max_dim
            crop = cv2.resize(
                crop,
                (int(cw * scale), int(ch * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return crop

    # ── Filesystem ────────────────────────────────────────────────────────────

    def _apple_dir(self, state: _AppleState) -> Path:
        """Root folder for this apple — subfolders: ch1/, ch2/, ch3/, detected/."""
        assert self._session_dir is not None
        return self._session_dir / f"Lane{state.lane}" / f"Apple{state.seq_id}"

    def _ensure_dir(self, path: Path) -> None:
        parent = path.parent
        if parent not in self._dirs_made:
            parent.mkdir(parents=True, exist_ok=True)
            self._dirs_made.add(parent)

    def _write_jpeg(self, path: Path, jpeg_bytes: bytes) -> None:
        self._ensure_dir(path)
        path.write_bytes(jpeg_bytes)
        self._saved_images += 1

    def _write_csv(
        self,
        state: _AppleState,
        finalize: bool = False,
        incomplete: bool = False,
    ) -> None:
        assert self._session_dir is not None
        lane_dir = self._session_dir / f"Lane{state.lane}"
        if lane_dir not in self._dirs_made:
            lane_dir.mkdir(parents=True, exist_ok=True)
            self._dirs_made.add(lane_dir)
        path = lane_dir / f"Apple{state.seq_id}.csv"

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["apple_id", "lane", "frame_idx", "detector_class", "confidence"])
            for row in state.rows:
                writer.writerow([
                    state.seq_id, state.lane, row.frame_idx,
                    row.detector_class, f"{row.confidence:.4f}",
                ])
            if finalize or incomplete:
                writer.writerow([])
                if state.final_grade is not None:
                    writer.writerow(["final_grade", state.final_grade])
                    writer.writerow(["final_confidence", f"{state.final_confidence:.4f}"])
                elif incomplete:
                    writer.writerow(["final_grade", "incomplete"])
                writer.writerow(["frames_total", len(state.rows)])


def _snapshot(t: dict) -> dict:
    """Lightweight track dict for the worker queue."""
    return {
        "track_id":     int(t["track_id"]),
        "seq_id":       t.get("seq_id"),
        "lane":         int(t["lane"]),
        "raw_class_id": int(t["raw_class_id"]),
        "raw_conf":     float(t["raw_conf"]),
        "class_id":     int(t["class_id"]),
        "conf":         float(t["conf"]),
        "eligible":     bool(t.get("eligible", True)),
        "box":          tuple(int(v) for v in t["box"]),
    }


def _normalize_to_bgr(frame: np.ndarray) -> np.ndarray | None:
    """
    Convert any input frame to a uint8 BGR array suitable for JPEG encoding.
    Handles:  grayscale, BGRA, float / uint16 (normalized to 0–255).
    Returns None if the input is None or has 0 area.
    """
    if frame is None:
        return None
    if frame.size == 0:
        return None
    # Normalize non-uint8 dtypes (float32, uint16, …) to 0–255 range
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # Ensure 3-channel BGR
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]  # strip alpha
    return frame

