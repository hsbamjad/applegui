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

import random
import threading
import time
import logging

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

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

    Runs on a dedicated QThread.  Accepts frames via a single-slot atomic
    buffer — enqueue() simply overwrites the slot with the latest frame,
    so the inference thread always processes the MOST RECENT frame and never
    accumulates a backlog that would stall the GUI.

    Signals:
      sig_result(annotated_frame, detections)
           annotated_frame : numpy BGR array with boxes + labels drawn
           detections      : supervision.Detections object (raw, pre-tracker)
      sig_fps(float)   -- inference throughput in frames/sec
      sig_status(str, bool) -- status messages and errors
    """

    sig_result = pyqtSignal(object)   # ultralytics Results object only
    sig_fps    = pyqtSignal(float)
    sig_status = pyqtSignal(str, bool)

    # Box colours per class index (BGR)
    _CLASS_COLORS = [
        (52, 211, 153),   # Fresh     — emerald green
        (251, 191, 36),   # Processing — amber
        (248,  113, 113), # Cull       — red
    ]

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold:  float = 0.45,
        device: str = "cuda",
        input_mode: str = "RB-nir1",
        model_imgsz: int = 640,
    ) -> None:
        super().__init__()
        self._model_path  = model_path
        self._conf        = conf_threshold
        self._iou         = iou_threshold
        self._device      = device
        self._input_mode  = input_mode
        self._model_imgsz = model_imgsz
        self._running     = False
        self._lock        = threading.Lock()
        self._latest      = None
        self._has_frame   = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def enqueue(self, ch1, ch2, ch3) -> None:
        """
        Overwrite the single-slot buffer with the latest frames.
        O(1), non-blocking — stale frames are simply replaced, never queued.
        """
        if ch1 is None:
            return
        with self._lock:
            self._latest = (ch1, ch2, ch3)
        self._has_frame.set()

    def _prepare_input(self, ch1, ch2, ch3) -> np.ndarray:
        """
        Build a 3-channel uint8 numpy array for YOLO based on input_mode.
        Frames are resized to model_imgsz BEFORE stacking to reduce memory
        bandwidth (e.g. 2048×1536 → 640×480 = 23× less data per channel).

        Channel layout from camera:
          ch1 = Source0  BGR color frame  (~660 nm visible)
          ch2 = Source1  grayscale NIR1   (~800 nm)
          ch3 = Source2  grayscale NIR2   (~900 nm)

        Supported input_mode values (match the band combo used for training):
          'RB-nir1'     R, B from ch1  + NIR1 from ch2   <- default / best model
          'RG-nir1'     R, G from ch1  + NIR1 from ch2
          'R-nir1-nir2' R from ch1     + NIR1 ch2 + NIR2 ch3
          'RGB'         Full BGR color frame (ch1 as-is)
          'CH1'         ch1 as-is (fallback)
        """
        # Resize to YOLO input size first — all subsequent ops are on small frames
        imgsz = getattr(self, '_model_imgsz', 640)
        h0, w0 = ch1.shape[:2]
        scale  = imgsz / max(h0, w0)
        if scale < 1.0:
            nw, nh = int(w0 * scale), int(h0 * scale)
            ch1 = cv2.resize(ch1, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if ch2 is not None:
                ch2 = cv2.resize(ch2, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if ch3 is not None:
                ch3 = cv2.resize(ch3, (nw, nh), interpolation=cv2.INTER_LINEAR)

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
            R  = ch1[:, :, 2]
            B  = ch1[:, :, 0]
            N1 = _to_gray(ch2)
            return np.stack([R, B, N1], axis=2)

        elif mode == "rg-nir1":
            R  = ch1[:, :, 2]
            G  = ch1[:, :, 1]
            N1 = _to_gray(ch2)
            return np.stack([R, G, N1], axis=2)

        elif mode == "r-nir1-nir2":
            R  = ch1[:, :, 2]
            N1 = _to_gray(ch2)
            N2 = _to_gray(ch3)
            return np.stack([R, N1, N2], axis=2)

        elif mode == "rgb":
            return _ensure_bgr(ch1)

        else:
            return _ensure_bgr(ch1)

    def stop(self) -> None:
        self._running = False
        self._has_frame.set()   # unblock the wait() call in run()
        self.wait(5000)

    # ── QThread entry point ──────────────────────────────────────────────────

    def run(self) -> None:
        # Import here so the module loads even without ultralytics installed
        try:
            from ultralytics import YOLO
            import supervision as sv
        except ImportError as e:
            self.sig_status.emit(f"Import error: {e}", True)
            return

        # Load model
        self.sig_status.emit(f"Loading model: {self._model_path}", False)
        try:
            model = YOLO(self._model_path)
            class_names = model.names   # {0: 'Fresh', 1: 'Processing', 2: 'Cull'}
            self.sig_status.emit(
                f"Model loaded  |  classes: {list(class_names.values())}  "
                f"|  device: {self._device}", False
            )
        except Exception as e:
            self.sig_status.emit(f"Model load failed: {e}", True)
            log.exception("RealInferenceWorker: model load failed")
            return

        self._running  = True
        frame_count    = 0
        fps_start      = time.perf_counter()

        while self._running:
            # Sleep until a frame arrives (or 0.5 s timeout to check _running)
            if not self._has_frame.wait(timeout=0.5):
                continue
            self._has_frame.clear()

            with self._lock:
                item = self._latest
                self._latest = None

            if item is None:
                continue

            ch1, ch2, ch3 = item

            # Build model input from correct band combination
            frame = self._prepare_input(ch1, ch2, ch3)

            # YOLO tracking with persist=True keeps internal tracker state
            # across frames — far more stable than detect+external-tracker
            try:
                results = model.track(
                    source   = frame,
                    tracker  = "bytetrack.yaml",
                    persist  = True,
                    conf     = self._conf,
                    iou      = self._iou,
                    device   = self._device,
                    imgsz    = 640,
                    verbose  = False,
                    save     = False,
                )
            except Exception as e:
                log.warning("Inference error: %s", e)
                continue

            # Emit only the result object — no heavy frame copies through Qt signal
            self.sig_result.emit(results[0])

            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                self.sig_fps.emit(frame_count / elapsed)
                frame_count = 0
                fps_start   = time.perf_counter()

        self.sig_status.emit("Inference worker stopped", False)
