"""
core/logging/grading_recorder.py
================================
Per-apple grading export - cropped patches + CSV.

Raw full-frame path (critical for live JAI / 10GigE)
----------------------------------------------------
Full-res JPEG encode while streaming kills GigE.  During Save we append each
triplet into per-channel ``stream.raw`` files (buffered sequential I/O) on a
BELOW_NORMAL writer with a bounded queue.  No 90-frame hard cap — sessions run
as long as you leave Save on; if the writer falls behind, frames are dropped
and logged.  JPEG conversion from the raw streams runs only after disconnect.

Detected-crop encodes are small and still run live (capped by batch slots).

Output layout::

    {session}/raw_frames/ch1/frame_XXXXXX.jpg   # raw ch1 (Color) - no annotation
    {session}/raw_frames/ch2/frame_XXXXXX.jpg   # raw ch2 (NIR1)  - no annotation
    {session}/raw_frames/ch3/frame_XXXXXX.jpg   # raw ch3 (NIR2)  - no annotation

    {session}/Lane{L}/Apple{N}/frame_XXX.jpg       # composite YOLO crop + boxes
    {session}/Lane{L}/Apple{N}.csv                          # per-apple detection CSV
"""

from __future__ import annotations

import csv
import ctypes
import json
import os
import queue
import threading
import time
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
# only - tracker / counting are unaffected.
_MAX_PENDING_BATCHES = 2
_DEFAULT_MAX_CROPS = 8
_DEFAULT_HEAVY_THRESHOLD = 12
# Live spill queue depth.  Each slot ≈ 15 MB @ 2048×1536.  Depth 24 ≈ 360 MB
# worst-case RAM - enough burst to absorb disk hiccups without the old hard
# cap of 90 frames, and without multi-GB piles that kill GigE.
_MAX_RAW_SPILL_QUEUE = 24
# 16 MiB stdio buffer per channel stream - sequential appends, not per-frame files.
_RAW_FILE_BUFFER = 16 * 1024 * 1024


def _set_spill_priority() -> None:
    """Spill must nearly keep up with the camera - BELOW_NORMAL, not IDLE."""
    if os.name == "nt":
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            # THREAD_PRIORITY_BELOW_NORMAL = -1
            ctypes.windll.kernel32.SetThreadPriority(handle, -1)
        except Exception:
            pass


def _set_low_priority() -> None:
    """IDLE priority for JPEG convert / crop writers (camera already down or small work)."""
    if os.name == "nt":
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, -15)  # IDLE
        except Exception:
            pass


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
    detected_crop_jpeg: bytes | None = None        # composite YOLO crop + boxes (Detected Frames)


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


# NOTE on OpenCV threading (Concurrency/PPL backend on this system):
# - cv2.setNumThreads() is process-global - NEVER call it from encode workers
#   or Bayer demosaic on the JAI-grab thread drops from ~5 ms to ~20 ms.
# - Raw full-frame JPEG must NOT run while the camera streams (see module doc).
# - Detected-crop imencode is small; keep it live but batch-capped.


class GradingRecorder:
    """
    Ordered command queue: every ``submit_batch`` is processed before the
    next ``on_grade_committed`` for that frame, eliminating races.
    """

    def __init__(
        self,
        image_format: str = "jpg",
        jpeg_quality: int = 92,
        save_detected_crops: bool = False,    # save composite YOLO crop + boxes + CSV
        crop_padding_frac: float = 0.20,
        raw_frame_stride: int = 1,            # save raw full frames every Nth camera frame
        save_max_dim: int = 0,                # 0 = full resolution; else downscale longest side (raw + detected)
        save_raw_full_frames: bool = False,   # save full-resolution frames (all 3 channels)
        max_pending_batches: int = _MAX_PENDING_BATCHES,
        max_crops_per_batch: int = _DEFAULT_MAX_CROPS,
        heavy_threshold: int = _DEFAULT_HEAVY_THRESHOLD,
    ) -> None:
        self._image_ext = image_format.lower().lstrip(".")
        self._jpeg_quality = jpeg_quality
        self._save_detected_crops  = save_detected_crops
        self._crop_pad = crop_padding_frac
        self._raw_frame_stride = max(1, raw_frame_stride)
        self._save_max_dim = max(0, save_max_dim)
        self._raw_frame_tick = 0           # counts logged frames for stride gating
        self._save_raw_full_frames = save_raw_full_frames
        self._raw_full_frame_counter = 0   # monotonic counter for raw_frames/ filenames
        self._dropped_raw_frames = 0
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
        # In-memory JPEG buffer: encoded during recording, written to disk after stop.
        # Keeps ALL disk I/O off the camera streaming path so GigE Vision DPCs
        # are never delayed by disk write DPCs competing at DISPATCH_LEVEL.
        self._write_buffer: list[tuple[Path, bytes]] = []
        self._write_buffer_lock = threading.Lock()

        self._batch_slots = threading.Semaphore(max_pending_batches)
        # Live raw path: bounded queue → sequential stream.raw appends.
        self._spill_q: queue.Queue = queue.Queue(maxsize=_MAX_RAW_SPILL_QUEUE)
        self._spill_thread = threading.Thread(
            target=self._spill_loop, daemon=True, name="log-raw-spill",
        )
        self._spill_stop = threading.Event()
        self._raw_written = 0
        self._raw_files: dict[str, object] | None = None  # ch → binary file obj
        self._raw_index_fh = None
        self._cmd_q: queue.SimpleQueue = queue.SimpleQueue()
        # Write pool only serves detected-crop JPEG buffering (small).
        self._write_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="log-wr",
            initializer=_set_low_priority,
        )
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        self._spill_thread.start()
        self._raw_finalized = False
        self._spill_joined = False

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def start_session(self, base_dir: Path | str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(base_dir) / ts
        session_dir.mkdir(parents=True, exist_ok=True)
        self._dirs_made.add(session_dir)

        # Pre-create raw_frames directories to avoid thread-pool race conditions
        if self._save_raw_full_frames:
            for idx in range(3):
                (session_dir / "raw_frames" / f"ch{idx + 1}").mkdir(parents=True, exist_ok=True)

        # Mark active and store session_dir immediately on the GUI thread -
        # no blocking wait.  The worker will process "start" and reset its
        # internal state before it touches any subsequent commands.
        self._session_dir = session_dir
        self._saved_images = 0
        self._dropped_batches = 0
        self._dropped_raw_frames = 0
        self._active = True
        self._cmd_q.put(("start", session_dir))
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
    ) -> None:
        """Enqueue per-apple detected crops; batch slot must already be acquired."""
        if not tracks or frame_bgr is None:
            self.release_batch_slot()
            return
        capped = (
            _tracks_for_logging(tracks, self._max_crops)
            if len(tracks) > self._max_crops else tracks
        )
        snaps = tuple(_snapshot(t) for t in capped)
        self._cmd_q.put(("batch", frame_bgr, snaps))

    def submit_raw_frame(
        self,
        ch1: np.ndarray | None,
        ch2: np.ndarray | None,
        ch3: np.ndarray | None,
    ) -> None:
        """
        Enqueue a full-res triplet for sequential ``stream.raw`` spill.

        No JPEG during live capture.  Queue is bounded — if the writer falls
        behind, frames are dropped (camera stays up).  There is no 90-frame cap;
        a session keeps saving for as long as Save stays on and the writer keeps up.

        Live on-disk layout::
            {session}/raw_frames/ch1/stream.raw
            {session}/raw_frames/ch2/stream.raw
            {session}/raw_frames/ch3/stream.raw
            {session}/raw_frames/index.jsonl
        JPEG files are produced later in ``finalize_raw_and_disk``.
        """
        if not self._active:
            return
        if not self._save_raw_full_frames:
            return
        self._raw_frame_tick += 1
        if self._raw_frame_tick % self._raw_frame_stride != 0:
            return
        try:
            self._spill_q.put_nowait((ch1, ch2, ch3))
        except queue.Full:
            self._dropped_raw_frames += 1
            if self._dropped_raw_frames in (1, 50, 200, 1000):
                log.warning(
                    "GradingRecorder: dropped %d raw frames "
                    "(spill queue full — writer behind; %d written so far)",
                    self._dropped_raw_frames, self._raw_written,
                )

    def set_save_options(
        self,
        save_raw_full_frames: bool | None = None,
        save_detected_crops: bool | None = None,
    ) -> None:
        """
        Update save flags live without restarting the session.
        Thread-safe: Python bool assignment is atomic under the GIL.
        """
        if save_raw_full_frames is not None:
            self._save_raw_full_frames = save_raw_full_frames
        if save_detected_crops is not None:
            self._save_detected_crops = save_detected_crops

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

    def stop_session(self, *, camera_live: bool = False) -> None:
        if not self._active:
            return
        self._active = False
        self._cmd_q.put(("stop", bool(camera_live)))
        log.info(
            "GradingRecorder stop enqueued - %d raw frames written, "
            "%d dropped, %d crop images (camera_live=%s)",
            self._raw_written, self._dropped_raw_frames, self._saved_images,
            camera_live,
        )

    @property
    def buffered_raw_count(self) -> int:
        """Frames still in the spill queue (not yet on disk)."""
        return self._spill_q.qsize()

    @property
    def npy_written(self) -> int:
        """Backward-compat alias for frames written to stream.raw."""
        return self._raw_written

    def drain_spill(self, timeout: float = 120.0) -> None:
        """Block until the live raw spill thread has finished and files are closed."""
        if self._spill_joined:
            return
        self._spill_stop.set()
        try:
            self._spill_q.put_nowait(None)
        except queue.Full:
            pass
        self._spill_thread.join(timeout=timeout)
        self._close_raw_files()
        self._spill_joined = True

    def finalize_raw_and_disk(self) -> None:
        """
        Convert spilled ``stream.raw`` → per-frame JPEGs and flush crop buffer.

        MUST be called only after the GigE camera is fully disconnected.
        """
        if self._raw_finalized:
            return
        self._raw_finalized = True
        _set_low_priority()
        self.drain_spill()
        self._convert_raw_streams_to_jpeg()
        self._flush_write_buffer_to_disk()
        log.info(
            "GradingRecorder finalize done - %d images saved "
            "(%d crop batches dropped, %d raw frames dropped, %d raw written)",
            self._saved_images, self._dropped_batches,
            self._dropped_raw_frames, self._raw_written,
        )

    # ── Worker (single ordered loop) ──────────────────────────────────────────

    def _run_worker(self) -> None:
        _set_low_priority()
        while True:
            cmd = self._cmd_q.get()
            kind = cmd[0]
            try:
                if kind == "start":
                    _, session_dir = cmd
                    with self._lock:
                        self._on_start(session_dir)
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
                    camera_live = bool(cmd[1]) if len(cmd) > 1 else False
                    writes: list[_WriteJob] = []
                    with self._lock:
                        writes.extend(self._on_stop())
                    self._flush_writes(writes)
                    self._write_pool.shutdown(wait=True)
                    # Drain .npy spill (releases any in-flight array refs).
                    self.drain_spill()
                    if camera_live:
                        log.info(
                            "GradingRecorder: spill drained (%d raw frames on disk) - "
                            "JPEG convert deferred until camera disconnect",
                            self._raw_written,
                        )
                    else:
                        self._convert_raw_streams_to_jpeg()
                        self._flush_write_buffer_to_disk()
                        self._raw_finalized = True
                        log.info(
                            "GradingRecorder stopped - %d images saved "
                            "(%d crop batches dropped, %d raw dropped, %d raw written)",
                            self._saved_images, self._dropped_batches,
                            self._dropped_raw_frames, self._raw_written,
                        )
                    break
            except Exception:
                log.exception("GradingRecorder worker error on %s", kind)
                if kind == "batch":
                    self._batch_slots.release()

    def _on_batch(
        self,
        frame: np.ndarray,
        tracks: tuple[dict, ...],
    ) -> None:
        # ── Step 1: Encode crops WITHOUT holding the lock ─────────────────────
        # cv2.imencode releases the GIL, so the inference thread is not starved.
        # The lock is only acquired below for fast dict bookkeeping (~1 ms).
        prepared: list[_PreparedTrack] = []
        for t in tracks:
            item = self._prepare_track(frame, t)
            if item is not None:
                prepared.append(item)

        # ── Step 2: Apply to state structures - fast dict ops under lock ──────
        writes: list[_WriteJob] = []
        with self._lock:
            for item in prepared:
                writes.extend(self._apply_prepared(item))

        self._flush_writes(writes)

    def _prepare_track(
        self,
        frame: np.ndarray,
        t: dict,
    ) -> _PreparedTrack | None:
        if t.get("box") is None:
            return None
        meta = _PendingMeta(
            lane=int(t["lane"]),
            raw_cls=int(t["raw_class_id"]),
            raw_conf=float(t["raw_conf"]),
        )
        if self._save_detected_crops:
            meta.detected_crop_jpeg = self._encode_detected_crop(frame, t)
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

        # Detected crop: composite YOLO frame + boxes → Apple{N}/
        if self._save_detected_crops and meta.detected_crop_jpeg is not None:
            jobs.append(_WriteJob(apple_dir / fname, meta.detected_crop_jpeg))

        return jobs

    def _on_start(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._apples.clear()
        self._track_buffer.clear()
        self._track_to_apple.clear()
        self._dirs_made.clear()
        self._raw_full_frame_counter = 0
        self._raw_written = 0

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
        # CSV is only written when Detected Frames logging is enabled
        if self._save_detected_crops:
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
            if self._save_detected_crops:
                self._write_csv(
                    state,
                    finalize=True,
                    incomplete=not state.finalized,
                )

        self._apples.clear()
        log.info(
            "GradingRecorder session data flushed (raw_frames_dropped=%d)",
            self._dropped_raw_frames,
        )
        return writes

    def _spill_loop(self) -> None:
        """BELOW_NORMAL thread: append full-res triplets to stream.raw files."""
        _set_spill_priority()
        while True:
            if self._spill_stop.is_set() and self._spill_q.empty():
                return
            try:
                item = self._spill_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if item is None:
                    return
                ch1, ch2, ch3 = item
                self._write_raw_triplet(ch1, ch2, ch3)
            except Exception:
                log.exception("GradingRecorder spill error")
            finally:
                item = None

    def _ensure_raw_files(self) -> bool:
        """Open per-channel stream.raw + index.jsonl (spill thread only)."""
        if self._raw_files is not None:
            return True
        if self._session_dir is None:
            return False
        root = self._session_dir / "raw_frames"
        files: dict[str, object] = {}
        try:
            for ch in ("ch1", "ch2", "ch3"):
                ch_dir = root / ch
                ch_dir.mkdir(parents=True, exist_ok=True)
                files[ch] = open(
                    ch_dir / "stream.raw", "wb", buffering=_RAW_FILE_BUFFER,
                )
            self._raw_index_fh = open(
                root / "index.jsonl", "w", encoding="utf-8", buffering=1024 * 1024,
            )
            self._raw_files = files
            log.info("GradingRecorder: opened stream.raw files for live spill")
            return True
        except Exception:
            log.exception("GradingRecorder: failed to open stream.raw files")
            for fh in files.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._raw_files = None
            self._raw_index_fh = None
            return False

    def _close_raw_files(self) -> None:
        if self._raw_files is not None:
            for fh in self._raw_files.values():
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
            self._raw_files = None
        if self._raw_index_fh is not None:
            try:
                self._raw_index_fh.flush()
                self._raw_index_fh.close()
            except Exception:
                pass
            self._raw_index_fh = None

    def _write_raw_triplet(
        self,
        ch1: np.ndarray | None,
        ch2: np.ndarray | None,
        ch3: np.ndarray | None,
    ) -> None:
        if not self._ensure_raw_files():
            return
        assert self._raw_files is not None and self._raw_index_fh is not None
        channels = (("ch1", ch1), ("ch2", ch2), ("ch3", ch3))
        valid = [(n, f) for n, f in channels if f is not None]
        if not valid:
            return
        self._raw_full_frame_counter += 1
        n = self._raw_full_frame_counter
        entry: dict = {"i": n}
        for ch_name, frame in valid:
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            self._raw_files[ch_name].write(memoryview(frame))
            entry[ch_name] = {
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
            }
        self._raw_index_fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._raw_written = n
        if n == 1 or n % 100 == 0:
            log.info(
                "GradingRecorder: spilled %d raw frames (%d dropped)",
                n, self._dropped_raw_frames,
            )

    def _convert_raw_streams_to_jpeg(self) -> None:
        """Read stream.raw + index.jsonl → per-frame JPEGs. Camera must be down."""
        if self._session_dir is None:
            return
        root = self._session_dir / "raw_frames"
        index_path = root / "index.jsonl"
        if not index_path.exists():
            # Backward compat: older sessions may have per-frame .npy
            self._convert_npy_to_jpeg()
            return

        streams: dict[str, object] = {}
        try:
            for ch in ("ch1", "ch2", "ch3"):
                raw_path = root / ch / "stream.raw"
                if raw_path.exists():
                    streams[ch] = open(raw_path, "rb", buffering=_RAW_FILE_BUFFER)
            if not streams:
                return
            log.info(
                "GradingRecorder: converting stream.raw → JPEG "
                "(%d frames indexed, camera down) …",
                self._raw_written,
            )
            t0 = time.perf_counter()
            converted = 0
            with index_path.open("r", encoding="utf-8") as idx:
                for line in idx:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    n = int(entry["i"])
                    fname = f"frame_{n:06d}.{self._image_ext}"
                    for ch in ("ch1", "ch2", "ch3"):
                        meta = entry.get(ch)
                        if meta is None or ch not in streams:
                            continue
                        shape = tuple(meta["shape"])
                        dtype = np.dtype(meta["dtype"])
                        nbytes = int(np.prod(shape) * dtype.itemsize)
                        buf = streams[ch].read(nbytes)
                        if len(buf) != nbytes:
                            log.warning(
                                "Short read on %s frame %d (%d/%d bytes)",
                                ch, n, len(buf), nbytes,
                            )
                            continue
                        arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
                        img = _normalize_to_bgr(arr)
                        if img is None:
                            continue
                        img = _downscale_max_dim(img, self._save_max_dim)
                        ok, enc = cv2.imencode(
                            f".{self._image_ext}", img,
                            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                        )
                        if ok:
                            path = root / ch / fname
                            self._write_jpeg(path, enc.tobytes())
                            converted += 1
            log.info(
                "GradingRecorder: raw→jpeg done — %d files in %.1f s",
                converted, time.perf_counter() - t0,
            )
        finally:
            for fh in streams.values():
                try:
                    fh.close()
                except Exception:
                    pass
            # Remove bulky intermediates after convert
            for ch in ("ch1", "ch2", "ch3"):
                raw_path = root / ch / "stream.raw"
                try:
                    raw_path.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                index_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _convert_npy_to_jpeg(self) -> None:
        """Legacy: turn per-frame .npy files into JPEGs."""
        if self._session_dir is None:
            return
        root = self._session_dir / "raw_frames"
        if not root.exists():
            return
        npy_files = sorted(root.glob("ch*/frame_*.npy"))
        if not npy_files:
            return
        log.info(
            "GradingRecorder: converting %d .npy → JPEG (camera down) …",
            len(npy_files),
        )
        t0 = time.perf_counter()
        converted = 0
        for npy_path in npy_files:
            try:
                arr = np.load(str(npy_path), allow_pickle=False)
                img = _normalize_to_bgr(arr)
                del arr
                if img is None:
                    npy_path.unlink(missing_ok=True)
                    continue
                img = _downscale_max_dim(img, self._save_max_dim)
                ok, buf = cv2.imencode(
                    f".{self._image_ext}", img,
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                )
                del img
                if ok:
                    jpg_path = npy_path.with_suffix(f".{self._image_ext}")
                    self._write_jpeg(jpg_path, buf.tobytes())
                    converted += 1
                npy_path.unlink(missing_ok=True)
            except Exception as exc:
                log.warning("npy→jpeg failed for %s: %s", npy_path, exc)
        log.info(
            "GradingRecorder: npy→jpeg done — %d files in %.1f s",
            converted, time.perf_counter() - t0,
        )

    def _flush_writes(self, jobs: list[_WriteJob]) -> None:
        for job in jobs:
            self._write_pool.submit(self._write_jpeg, job.path, job.jpeg_bytes)

    # ── Crop rendering ────────────────────────────────────────────────────────

    def _encode_detected_crop(self, frame: np.ndarray, track: dict) -> bytes | None:
        """Encode the YOLO composite crop with boxes + grade label (Detected Frames)."""
        crop = self._render_detected_crop(frame, track)
        if crop is None:
            return None
        ok, buf = cv2.imencode(
            f".{self._image_ext}", crop,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        return buf.tobytes() if ok else None

    def _render_detected_crop(self, frame: np.ndarray, track: dict) -> np.ndarray | None:
        """Render the composite YOLO crop with grade label box (Detected Frames)."""
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
            crop = frame[cy1:cy2, cx1:cx2].copy()

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

        return _downscale_max_dim(crop, self._save_max_dim)

    # ── Filesystem ────────────────────────────────────────────────────────────

    def _apple_dir(self, state: _AppleState) -> Path:
        """Per-apple folder for detected crop JPEGs."""
        assert self._session_dir is not None
        return self._session_dir / f"Lane{state.lane}" / f"Apple{state.seq_id}"

    def _ensure_dir(self, path: Path) -> None:
        parent = path.parent
        with self._lock:
            if parent not in self._dirs_made:
                parent.mkdir(parents=True, exist_ok=True)
                self._dirs_made.add(parent)

    def _write_jpeg(self, path: Path, jpeg_bytes: bytes) -> None:
        # Buffer in RAM - NO disk I/O while the camera is streaming.
        # Disk writes happen only in _flush_write_buffer_to_disk() after stop.
        with self._write_buffer_lock:
            self._write_buffer.append((path, jpeg_bytes))
        self._saved_images += 1

    def _flush_write_buffer_to_disk(self) -> None:
        """Write all buffered (path, jpeg_bytes) pairs to disk.
        Called from the worker thread after stop, never during streaming.
        """
        with self._write_buffer_lock:
            buf, self._write_buffer = self._write_buffer, []
        if not buf:
            return
        log.info("GradingRecorder: writing %d buffered images to disk …", len(buf))
        import time as _time
        t0 = _time.perf_counter()
        errors = 0
        for path, jpeg_bytes in buf:
            try:
                self._ensure_dir(path)
                path.write_bytes(jpeg_bytes)
            except Exception as exc:
                log.warning("Buffer flush write error for %s: %s", path, exc)
                errors += 1
        dt = _time.perf_counter() - t0
        log.info(
            "GradingRecorder: disk flush done — %d files written, %d errors (%.1f s)",
            len(buf) - errors, errors, dt,
        )

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


def _downscale_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Downscale so the longest side is at most *max_dim* px. 0 = no change.

    Uses pure numpy (no cv2.resize) so save-path threads never contend for
    OpenCV's process-global parallel_for_ pool used by live Bayer demosaic.
    """
    if max_dim <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    # Integer-factor box downsample when possible; otherwise stride sample.
    # Good enough for logging previews; avoids OpenCV thread-pool contention.
    y_idx = (np.linspace(0, h - 1, new_h)).astype(np.int32)
    x_idx = (np.linspace(0, w - 1, new_w)).astype(np.int32)
    return img[np.ix_(y_idx, x_idx)].copy()


def _normalize_to_bgr(frame: np.ndarray) -> np.ndarray | None:
    """
    Prepare any input frame for JPEG encoding.
    Handles:  grayscale, BGRA, float / uint16 (normalized to 0-255).
    Returns None if the input is None or has 0 area.

    Grayscale (ndim==2) frames are returned as-is.  cv2.imencode('.jpg', arr)
    handles 2-D uint8 arrays natively, producing a grayscale JPEG without any
    extra channel allocation.  This avoids the previous np.stack([g,g,g]) call
    which was allocating a 9 MB 3-channel copy of every 3 MB Mono8 NIR frame.

    Uses only pure numpy - no OpenCV calls - so write pool threads never
    contend for OpenCV's parallel backend workers.
    """
    if frame is None:
        return None
    if frame.size == 0:
        return None
    # Normalize non-uint8 dtypes (float32, uint16, …) to 0-255
    if frame.dtype != np.uint8:
        arr = frame.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        rng = mx - mn if mx > mn else 1.0
        frame = ((arr - mn) * (255.0 / rng)).clip(0.0, 255.0).astype(np.uint8)
    # Grayscale: return as-is (cv2.imencode handles 2-D uint8 as grayscale JPEG)
    if frame.ndim == 2:
        return frame
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]  # strip alpha (BGRA → BGR)
    return frame

