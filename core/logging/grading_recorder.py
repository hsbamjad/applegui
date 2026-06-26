"""
core/logging/grading_recorder.py
================================
Per-apple grading export -- cropped annotated patches + CSV.

OpenCV encode and disk I/O run off the GUI thread.  The inference hot path
only enqueues lightweight track snapshots; under load, pending batches are
capped so logging never starves the tracker.

Architecture -- write-immediately, rename-at-commit
---------------------------------------------------
JPEG crops are written to disk the moment each frame is processed -- never
accumulated in RAM.  Pre-commit frames go into a temporary staging folder::

    {session}/_tmp/track_{track_id}/processed/frame_XXX.jpg
    {session}/_tmp/track_{track_id}/source0/frame_XXX.jpg
    {session}/_tmp/track_{track_id}/source1/frame_XXX.jpg
    {session}/_tmp/track_{track_id}/source2/frame_XXX.jpg

When the grade is committed the staging folder is renamed atomically::

    {session}/Lane{L}/Apple{N}/processed/
    {session}/Lane{L}/Apple{N}/source0/
    {session}/Lane{L}/Apple{N}/source1/
    {session}/Lane{L}/Apple{N}/source2/
    {session}/Lane{L}/Apple{N}.csv

Result: no RAM burst at the exit zone, no GC spike, no disk-write burst.
"""

from __future__ import annotations

import csv
import queue
import shutil
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
class _TrackState:
    """Lightweight per-track bookkeeping -- NO JPEG bytes stored here."""
    track_id: int
    lane: int
    frame_idx: int = 0
    rows: list[_CsvRow] = field(default_factory=list)
    seq_id: int | None = None
    finalized: bool = False
    final_grade: str | None = None
    final_confidence: float | None = None


@dataclass
class _WriteJob:
    path: Path
    jpeg_bytes: bytes


class GradingRecorder:
    """
    Ordered command queue: every record_batch is processed before the
    next on_grade_committed for that frame, eliminating races.

    Write-immediately design: JPEG bytes go to disk (via the write pool) on
    every frame.  No large in-memory buffer accumulates at the exit zone.
    """

    def __init__(
        self,
        image_format: str = "jpg",
        jpeg_quality: int = 92,
        save_frames: bool = True,
        save_raw_frames: bool = True,
        crop_padding_frac: float = 0.20,
        crop_max_dim: int = 512,
        raw_crop_max_dim: int = 256,
        raw_frame_stride: int = 1,
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
        self._raw_frame_tick = 0
        self._max_crops = max(1, max_crops_per_batch)
        self._heavy_threshold = max(1, heavy_threshold)

        self._lock = threading.Lock()
        self._session_dir: Path | None = None
        self._tmp_dir: Path | None = None
        self._tracks: dict[int, _TrackState] = {}
        self._track_to_seq: dict[int, int] = {}
        self._apples: dict[int, _TrackState] = {}
        self._dirs_made: set[Path] = set()
        self._active = False
        self._saved_images = 0
        self._dropped_batches = 0

        self._batch_slots = threading.Semaphore(max_pending_batches)
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

        raw_frames: optional (ch1, ch2, ch3) numpy arrays.  When supplied,
        raw crops are saved to source0/, source1/, source2/ under each Apple
        folder.  Pass None to skip raw-frame saving.
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
        """Legacy entry -- acquires slot then submits (used off hot path only)."""
        if not self._active or frame_bgr is None or not tracks:
            return
        if not self.acquire_batch_slot():
            return
        self.submit_batch(frame_bgr, tracks, raw_frames)

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
            "GradingRecorder stopped -- %d images saved (%d batches dropped)",
            self._saved_images, self._dropped_batches,
        )

    # -- Worker ----------------------------------------------------------------

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
                    self._on_commit(seq_id, lane, class_name, confidence, track_id)
                elif kind == "stop":
                    _, done = cmd
                    self._on_stop()
                    done.set()
            except Exception:
                log.exception("GradingRecorder worker error on %s", kind)
                if kind == "batch":
                    self._batch_slots.release()

    def _on_batch(
        self,
        frame: np.ndarray,
        tracks: tuple[dict, ...],
        raw_frames: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None,
    ) -> None:
        """
        Step 1: encode crops outside lock (cv2 releases GIL).
        Step 2: update lightweight state under lock (fast dict ops only).
        Step 3: submit write jobs to pool (fire and forget -- bytes go to disk).

        No JPEG bytes remain in RAM after this method returns.
        """
        # -- Step 1: Encode outside lock ---------------------------------------
        self._raw_frame_tick += 1
        save_raw = (
            self._save_raw_frames
            and raw_frames is not None
            and self._raw_frame_tick % self._raw_frame_stride == 0
        )

        encoded: list[tuple[dict, bytes | None, tuple]] = []
        for t in tracks:
            if t.get("box") is None:
                continue
            proc_jpeg = self._encode_crop(frame, t) if self._save_frames else None
            if save_raw:
                raw_jpegs = tuple(
                    self._encode_raw_crop(rf, t) if rf is not None else None
                    for rf in raw_frames
                )
            else:
                raw_jpegs = (None, None, None)
            encoded.append((t, proc_jpeg, raw_jpegs))

        # -- Step 2: Update state + build write paths (lock, fast) -------------
        write_jobs: list[_WriteJob] = []
        with self._lock:
            if self._session_dir is None:
                return
            for t, proc_jpeg, raw_jpegs in encoded:
                tid = int(t["track_id"])
                seq_id = t.get("seq_id")

                ts = self._tracks.get(tid)
                if ts is None:
                    ts = _TrackState(track_id=tid, lane=int(t["lane"]))
                    self._tracks[tid] = ts

                if ts.finalized:
                    continue

                if seq_id is not None and ts.seq_id is None:
                    ts.seq_id = int(seq_id)
                    self._track_to_seq[tid] = ts.seq_id
                    self._apples[ts.seq_id] = ts

                ts.frame_idx += 1
                raw_cls = int(t["raw_class_id"])
                cls_name = (
                    CLASS_NAMES[raw_cls] if 0 <= raw_cls < len(CLASS_NAMES) else str(raw_cls)
                )
                ts.rows.append(_CsvRow(
                    frame_idx=ts.frame_idx,
                    detector_class=cls_name,
                    confidence=float(t["raw_conf"]),
                ))

                dest_root = self._track_root(ts)
                fname = f"frame_{ts.frame_idx:03d}.{self._image_ext}"

                if proc_jpeg is not None:
                    write_jobs.append(_WriteJob(dest_root / "processed" / fname, proc_jpeg))
                _SRC = ("source0", "source1", "source2")
                for src, rj in zip(_SRC, raw_jpegs):
                    if rj is not None:
                        write_jobs.append(_WriteJob(dest_root / src / fname, rj))

        # -- Step 3: Async disk writes -----------------------------------------
        self._flush_writes(write_jobs)

    def _on_commit(
        self,
        seq_id: int,
        lane: int,
        class_name: str,
        confidence: float,
        track_id: int,
    ) -> None:
        """
        Finalize an apple.  All JPEG files are already on disk in the staging
        folder -- just rename staging ? Apple{N}/ and write the CSV.
        Both are done off-lock in a write pool thread.
        """
        with self._lock:
            ts = self._tracks.get(track_id)
            if ts is None:
                ts = _TrackState(track_id=track_id, lane=lane, seq_id=seq_id)
                self._tracks[track_id] = ts
                self._apples[seq_id] = ts

            if ts.seq_id is None:
                ts.seq_id = seq_id
                self._apples[seq_id] = ts

            ts.final_grade = class_name
            ts.final_confidence = float(confidence)
            ts.finalized = True

            staging = self._staging_dir(track_id)
            final   = self._apple_dir(lane, seq_id)

        # Rename + CSV in pool thread -- never blocks inference or GUI
        if self._session_dir is not None:
            self._write_pool.submit(self._rename_and_csv, staging, final, ts)

    def _on_stop(self) -> None:
        """Flush unfinalized tracks at session end."""
        with self._lock:
            pending = [
                (
                    ts,
                    self._staging_dir(ts.track_id),
                    self._apple_dir(ts.lane, ts.seq_id or ts.track_id),
                )
                for ts in self._tracks.values()
                if not ts.finalized
            ]
            self._tracks.clear()
            self._track_to_seq.clear()
            self._apples.clear()

        for ts, staging, final in pending:
            self._rename_and_csv(staging, final, ts, incomplete=True)

        if self._tmp_dir is not None:
            try:
                if self._tmp_dir.exists():
                    shutil.rmtree(self._tmp_dir, ignore_errors=True)
            except Exception:
                pass

        log.info("GradingRecorder session data flushed")

    # -- Path helpers ----------------------------------------------------------

    def _staging_dir(self, track_id: int) -> Path:
        assert self._tmp_dir is not None
        return self._tmp_dir / f"track_{track_id}"

    def _apple_dir(self, lane: int, seq_id: int) -> Path:
        assert self._session_dir is not None
        return self._session_dir / f"Lane{lane}" / f"Apple{seq_id}"

    def _track_root(self, ts: _TrackState) -> Path:
        """Staging dir before commit; final apple dir after commit."""
        if ts.seq_id is not None:
            return self._apple_dir(ts.lane, ts.seq_id)
        return self._staging_dir(ts.track_id)

    # -- Rename + CSV (runs in write pool thread) ------------------------------

    def _rename_and_csv(
        self,
        staging: Path,
        final: Path,
        ts: _TrackState,
        incomplete: bool = False,
    ) -> None:
        """
        Rename staging folder ? final Apple{N}/ and write CSV.
        Runs in write pool thread -- never blocks inference or GUI.
        """
        try:
            if staging.exists():
                final.parent.mkdir(parents=True, exist_ok=True)
                if final.exists():
                    # Merge: move sub-items from staging into existing final dir
                    for sub in staging.iterdir():
                        dest_sub = final / sub.name
                        if dest_sub.exists():
                            for f in sub.iterdir():
                                f.rename(dest_sub / f.name)
                        else:
                            sub.rename(dest_sub)
                    try:
                        staging.rmdir()
                    except OSError:
                        pass
                else:
                    staging.rename(final)
            else:
                final.mkdir(parents=True, exist_ok=True)
        except Exception:
            log.exception("GradingRecorder: rename %s -> %s failed", staging, final)

        self._write_csv(ts, final, incomplete=incomplete)

    # -- Crop encoding ---------------------------------------------------------

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
        Encode a clean crop from a raw source frame -- no annotation overlay.
        Same bbox + padding as processed crop so all images are spatially aligned.
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
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            crop = crop[:, :, :3]

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

    # -- Filesystem ------------------------------------------------------------

    def _on_start(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._tmp_dir = session_dir / "_tmp"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._tracks.clear()
        self._track_to_seq.clear()
        self._apples.clear()
        self._dirs_made.clear()

    def _ensure_dir(self, path: Path) -> None:
        parent = path.parent
        if parent not in self._dirs_made:
            parent.mkdir(parents=True, exist_ok=True)
            self._dirs_made.add(parent)

    def _flush_writes(self, jobs: list[_WriteJob]) -> None:
        for job in jobs:
            self._write_pool.submit(self._write_jpeg, job.path, job.jpeg_bytes)

    def _write_jpeg(self, path: Path, jpeg_bytes: bytes) -> None:
        self._ensure_dir(path)
        path.write_bytes(jpeg_bytes)
        self._saved_images += 1

    def _write_csv(
        self,
        ts: _TrackState,
        apple_dir: Path,
        incomplete: bool = False,
    ) -> None:
        csv_path = apple_dir.parent / f"{apple_dir.name}.csv"
        try:
            apple_dir.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["apple_id", "lane", "frame_idx", "detector_class", "confidence"])
                seq = ts.seq_id or "?"
                for row in ts.rows:
                    writer.writerow([
                        seq, ts.lane, row.frame_idx,
                        row.detector_class, f"{row.confidence:.4f}",
                    ])
                writer.writerow([])
                if ts.final_grade is not None:
                    writer.writerow(["final_grade", ts.final_grade])
                    writer.writerow(["final_confidence", f"{ts.final_confidence:.4f}"])
                elif incomplete:
                    writer.writerow(["final_grade", "incomplete"])
                writer.writerow(["frames_total", len(ts.rows)])
        except Exception:
            log.exception("GradingRecorder: CSV write failed for %s", csv_path)


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
