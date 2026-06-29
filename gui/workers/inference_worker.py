"""
gui/workers/inference_worker.py
================================
Contains two workers:

  MockInferenceWorker  — random grade generator for demo / UI development.
  RealInferenceWorker  — live YOLO inference on camera/video frames (Phase 1+).

Phase 4: MockInferenceWorker will be retired once RealInferenceWorker + Grade
Aggregator are fully wired.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from pathlib import Path
from core.log import get_logger

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = get_logger(__name__)

# ── Mock grade constants ───────────────────────────────────────────────────────

GRADE_WEIGHTS = {"Fresh": 58, "Processing": 27, "Cull": 15}
GRADES        = list(GRADE_WEIGHTS.keys())
WEIGHTS       = list(GRADE_WEIGHTS.values())

CONF_RANGES = {
    "Fresh":      (0.82, 0.99),
    "Processing": (0.65, 0.92),
    "Cull":       (0.70, 0.97),
}

GRADE_OUTLET = {
    "Fresh":      "A",
    "Processing": "B",
    "Cull":       "C",
}

LANES = 3


# ── Mock inference worker ──────────────────────────────────────────────────────

class MockInferenceWorker(QThread):
    """
    Simulates the AI grading pipeline for demo / development.
    Fires one grade event per lane per tick at the configured conveyor speed.
    """

    sig_grade = pyqtSignal(int, int, str, float, str)
    # (apple_id, lane, grade, confidence, outlet)

    def __init__(self, apples_per_sec: int = 1) -> None:
        super().__init__()
        self._aps      = max(1, apples_per_sec)
        self._running  = False
        self._apple_id = 0

    def set_speed(self, apples_per_sec: int) -> None:
        self._aps = max(1, apples_per_sec)

    def run(self) -> None:
        self._running = True
        log.info(f"MockInferenceWorker started at {self._aps} apples/s/lane.")

        while self._running:
            for lane in range(1, LANES + 1):
                self._apple_id += 1
                grade, conf = self._mock_grade()
                outlet = GRADE_OUTLET[grade]
                self.sig_grade.emit(self._apple_id, lane, grade, conf, outlet)
            time.sleep(1.0 / self._aps)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    @staticmethod
    def _mock_grade() -> tuple[str, float]:
        grade = random.choices(GRADES, weights=WEIGHTS, k=1)[0]
        lo, hi = CONF_RANGES[grade]
        return grade, round(random.uniform(lo, hi), 4)


# ── Real YOLO inference worker ─────────────────────────────────────────────────

class RealInferenceWorker(QThread):
    """
    Live YOLO inference worker for Phase 1+.

    Runs on a dedicated QThread. Accepts frames from a thread-safe queue
    (maxsize=2 — drops stale frames if GPU falls behind camera FPS).
    YOLO + conveyor tracker run here; GUI thread stays lightweight.

    Signals:
      sig_preview(thumb_bgr, active)   — coalesced on GUI for model-input panel
      sig_graded(list[GradeRecord])    — grade commits, never dropped
      sig_fps(float)
      sig_status(str, bool)
    """

    sig_preview = pyqtSignal(object, list)
    sig_graded  = pyqtSignal(list)
    sig_fps     = pyqtSignal(float)
    sig_status  = pyqtSignal(str, bool)
    sig_model_ready = pyqtSignal()

    _THUMB_W = 512
    _LOG_HEAVY_THRESHOLD = 12
    _LOG_FRAME_STRIDE = 2

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold:  float = 0.45,
        device: str = "cuda",
        input_mode: str = "RB-nir1",
        tracker=None,
        size_acc=None,
        size_lock: threading.Lock | None = None,
    ) -> None:
        super().__init__()
        self._model_path  = model_path
        self._conf        = conf_threshold
        self._iou         = iou_threshold
        self._device      = device
        self._input_mode  = input_mode
        self._tracker     = tracker
        self._size_acc    = size_acc
        self._size_lock   = size_lock
        self._size_tick   = 0
        self._log_tick    = 0
        self._recorder    = None
        self._running     = False
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._raw_frames: tuple | None = None   # (ch1, ch2, ch3) for current frame — raw source crops
        self._tracker_cfg = str(
            Path(__file__).parent.parent.parent / "bytetrack.yaml"
        )

    def set_grading_recorder(self, recorder) -> None:
        """Live session recorder — logging runs on this thread, not the GUI."""
        self._recorder = recorder

    def enqueue(self, ch1, ch2, ch3) -> None:
        """Non-blocking enqueue; drops oldest triplet when full."""
        if ch1 is None:
            return
        try:
            self._queue.put_nowait((ch1, ch2, ch3))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((ch1, ch2, ch3))
            except queue.Full:
                pass

    def drain_pending(self) -> int:
        """Discard queued frames (e.g. after model-load pause)."""
        n = 0
        while True:
            try:
                self._queue.get_nowait()
                n += 1
            except queue.Empty:
                break
        return n

    def _prepare_input(self, ch1, ch2, ch3) -> np.ndarray:
        def _to_gray(f):
            if f is None:
                return np.zeros(ch1.shape[:2], dtype=np.uint8)
            if f.dtype != np.uint8:
                f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return f[:, :, 0] if f.ndim == 3 else f

        def _ensure_bgr(f):
            if f.dtype != np.uint8:
                f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

        mode = self._input_mode.lower()

        if mode == "rb-nir1":
            return np.stack([ch1[:, :, 2], ch1[:, :, 0], _to_gray(ch2)], axis=2)
        if mode == "rg-nir1":
            return np.stack([ch1[:, :, 2], ch1[:, :, 1], _to_gray(ch2)], axis=2)
        if mode == "r-nir1-nir2":
            return np.stack([ch1[:, :, 2], _to_gray(ch2), _to_gray(ch3)], axis=2)
        if mode == "rgb":
            return _ensure_bgr(ch1)
        return _ensure_bgr(ch1)

    @staticmethod
    def _make_thumb(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        nh = max(1, int(h * (RealInferenceWorker._THUMB_W / w)))
        return cv2.resize(
            frame,
            (RealInferenceWorker._THUMB_W, nh),
            interpolation=cv2.INTER_LINEAR,
        )

    def _maybe_log_batch(self, frame: np.ndarray, active: list) -> None:
        """Copy + enqueue only when a recorder slot is free — never blocks inference."""
        rec = self._recorder
        if rec is None or not active:
            return
        n = len(active)
        self._log_tick += 1
        if n > self._LOG_HEAVY_THRESHOLD and self._log_tick % self._LOG_FRAME_STRIDE != 0:
            return
        if not rec.acquire_batch_slot():
            return
        try:
            # Pass raw source frames so the recorder can save source0/1/2 crops.
            # .copy() is intentionally deferred to the background worker to keep
            # the inference hot path as light as possible; the tuple reference itself
            # is safe here because _raw_frames is replaced (not mutated) each loop.
            raw = self._raw_frames  # tuple[ch1, ch2, ch3] or None
            rec.submit_batch(frame.copy(), active, raw)
        except Exception:
            rec.release_batch_slot()
            raise


    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self.wait(5000)

    def run(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            self.sig_status.emit(f"Import error: {e}", True)
            return

        self.sig_status.emit(f"Loading model: {self._model_path}", False)
        try:
            model = YOLO(self._model_path)
            self.sig_status.emit(
                f"Model loaded successfully  |  Device: {self._device.upper()}", False
            )
        except Exception as e:
            self.sig_status.emit(f"Model load failed: {e}", True)
            log.exception("RealInferenceWorker: model load failed")
            return

        self.sig_model_ready.emit()

        self._running   = True
        frame_count     = 0
        fps_start       = time.perf_counter()

        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            ch1, ch2, ch3 = item
            frame = self._prepare_input(ch1, ch2, ch3)
            # Hold raw frames so _maybe_log_batch can pass them to the recorder.
            # Stored as a tuple of references — no copy on the hot path.
            self._raw_frames = (ch1, ch2, ch3)

            try:
                results = model.track(
                    source  = frame,
                    tracker = self._tracker_cfg,
                    persist = True,
                    conf    = self._conf,
                    iou     = self._iou,
                    device  = self._device,
                    imgsz   = 640,
                    verbose = False,
                    save    = False,
                )
            except Exception as e:
                log.warning("Inference error: %s", e)
                continue

            result = results[0]
            if self._tracker is not None:
                active, graded = self._tracker.update(result, frame.shape)
            else:
                active, graded = [], []

            if self._size_acc is not None and active:
                n = len(active)
                self._size_tick += 1
                if n <= 10 or self._size_tick % 2 == 0:
                    lock = self._size_lock
                    if lock:
                        with lock:
                            self._size_acc.update(result, active)
                    else:
                        self._size_acc.update(result, active)

            thumb = self._make_thumb(frame)
            self._maybe_log_batch(frame, active)
            self.sig_preview.emit(thumb, active)
            if graded:
                self.sig_graded.emit(graded)

            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                self.sig_fps.emit(frame_count / elapsed)
                frame_count = 0
                fps_start   = time.perf_counter()

        self.sig_status.emit("Inference worker stopped", False)
