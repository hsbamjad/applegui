"""
core/logging/grading_recorder.py
================================
Per-apple grading export — cropped annotated patches + CSV.

OpenCV encode and disk I/O run off the GUI thread.  The inference hot path
only enqueues lightweight track snapshots; under load, pending batches are
capped so logging never starves the tracker.

Output layout::

    {session}/Lane{L}/Apple{N}/frame_XXX.jpg   # cropped apple + grade label
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
    crop_jpeg: bytes | None = None


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
        crop_padding_frac: float = 0.20,
        crop_max_dim: int = 512,
        max_pending_batches: int = _MAX_PENDING_BATCHES,
        max_crops_per_batch: int = _DEFAULT_MAX_CROPS,
        heavy_threshold: int = _DEFAULT_HEAVY_THRESHOLD,
    ) -> None:
        self._image_ext = image_format.lower().lstrip(".")
        self._jpeg_quality = jpeg_quality
        self._save_frames = save_frames
        self._crop_pad = crop_padding_frac
        self._crop_max_dim = crop_max_dim
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
        self._cmd_q: queue.SimpleQueue = queue.SimpleQueue()
        self._write_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="log-wr")
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

    def submit_batch(self, frame_bgr: np.ndarray, tracks: list[dict]) -> None:
        """Enqueue a batch; batch slot must already be acquired."""
        if not tracks or frame_bgr is None:
            self.release_batch_slot()
            return
        capped = (
            _tracks_for_logging(tracks, self._max_crops)
            if len(tracks) > self._max_crops else tracks
        )
        snaps = tuple(_snapshot(t) for t in capped)
        self._cmd_q.put(("batch", frame_bgr, snaps))

    def record_batch(self, frame_bgr: np.ndarray, tracks: list[dict]) -> None:
        """Legacy entry — acquires slot then submits (used off hot path only)."""
        if not self._active or frame_bgr is None or not tracks:
            return
        if not self.acquire_batch_slot():
            return
        self.submit_batch(frame_bgr, tracks)

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
                    _, frame, tracks = cmd
                    try:
                        self._on_batch(frame, tracks)
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
            except Exception:
                log.exception("GradingRecorder worker error on %s", kind)
                if kind == "batch":
                    self._batch_slots.release()

    def _on_batch(self, frame: np.ndarray, tracks: tuple[dict, ...]) -> None:
        # Sequential encode — avoids CPU spikes when many apples are on screen.
        writes: list[_WriteJob] = []
        with self._lock:
            for t in tracks:
                item = self._prepare_track(frame, t)
                if item is not None:
                    writes.extend(self._apply_prepared(item))
        self._flush_writes(writes)

    def _prepare_track(
        self, frame: np.ndarray, t: dict,
    ) -> _PreparedTrack | None:
        if t.get("box") is None:
            return None
        meta = _PendingMeta(
            lane=int(t["lane"]),
            raw_cls=int(t["raw_class_id"]),
            raw_conf=float(t["raw_conf"]),
        )
        if self._save_frames:
            meta.crop_jpeg = self._encode_crop(frame, t)
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
                    job = self._append_row(sid, bm)
                    if job is not None:
                        writes.append(job)
            job = self._append_row(sid, meta)
            if job is not None:
                writes.append(job)
        else:
            self._track_buffer.setdefault(track_id, []).append(meta)

        return writes

    def _append_row(self, seq_id: int, meta: _PendingMeta) -> _WriteJob | None:
        """Must be called under self._lock."""
        state = self._apples.get(seq_id)
        if state is None:
            state = _AppleState(seq_id=seq_id, lane=meta.lane)
            self._apples[seq_id] = state

        if state.finalized:
            return None

        state.frame_idx += 1
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

        if self._save_frames and meta.crop_jpeg is not None:
            path = (
                self._apple_dir(state)
                / f"frame_{state.frame_idx:03d}.{self._image_ext}"
            )
            return _WriteJob(path, meta.crop_jpeg)
        return None

    def _on_start(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._apples.clear()
        self._track_buffer.clear()
        self._track_to_apple.clear()
        self._dirs_made.clear()

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
                job = self._append_row(seq_id, bm)
                if job is not None:
                    writes.append(job)

        state = self._apples.get(seq_id)
        if state is None:
            state = _AppleState(seq_id=seq_id, lane=lane)
            self._apples[seq_id] = state

        state.final_grade = class_name
        state.final_confidence = float(confidence)
        state.finalized = True
        self._write_csv(state, finalize=True)
        return writes

    def _on_stop(self) -> list[_WriteJob]:
        writes: list[_WriteJob] = []
        for track_id, buffered in list(self._track_buffer.items()):
            sid = self._track_to_apple.get(track_id)
            if sid is None:
                continue
            for bm in buffered:
                job = self._append_row(sid, bm)
                if job is not None:
                    writes.append(job)
        self._track_buffer.clear()
        self._track_to_apple.clear()

        for state in self._apples.values():
            self._write_csv(
                state,
                finalize=True,
                incomplete=not state.finalized,
            )

        self._apples.clear()
        log.info("GradingRecorder session data flushed")
        return writes

    def _flush_writes(self, jobs: list[_WriteJob]) -> None:
        for job in jobs:
            self._write_pool.submit(self._write_jpeg, job.path, job.jpeg_bytes)

    # ── Crop rendering ────────────────────────────────────────────────────────

    def _encode_crop(self, frame: np.ndarray, track: dict) -> bytes | None:
        crop = self._render_crop(frame, track)
        if crop is None:
            return None
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
