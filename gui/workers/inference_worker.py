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

    Runs on a dedicated QThread. Accepts frames from a thread-safe queue
    (maxsize=2 — drops stale frames if GPU falls behind camera FPS).
    Emits an annotated frame and raw supervision Detections per pass.

    Signals:
      sig_result(annotated_frame, detections)
           annotated_frame : numpy BGR array with boxes + labels drawn
           detections      : supervision.Detections object (raw, pre-tracker)
      sig_fps(float)   -- inference throughput in frames/sec
      sig_status(str, bool) -- status messages and errors
    """

    sig_result = pyqtSignal(object, object)   # annotated_frame, sv.Detections
    sig_fps    = pyqtSignal(float)
    sig_status = pyqtSignal(str, bool)        # message, is_error

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
        input_channel: int = 0,   # 0=CH1 Color, 1=CH2 NIR1, 2=CH3 NIR2
    ) -> None:
        super().__init__()
        self._model_path    = model_path
        self._conf          = conf_threshold
        self._iou           = iou_threshold
        self._device        = device
        self._input_channel = input_channel
        self._running       = False
        self._queue: queue.Queue = queue.Queue(maxsize=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def enqueue(self, ch1, ch2, ch3) -> None:
        """
        Push a frame triplet to the inference queue.
        Non-blocking: drops the oldest frame if queue is full (always process latest).
        """
        frame = [ch1, ch2, ch3][self._input_channel]
        if frame is None:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()   # discard oldest
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._running = False
        # Unblock the queue.get() call in run() so the thread exits cleanly
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
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
            # Block until a frame arrives (timeout so we can check _running)
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:   # sentinel from stop()
                break

            # Ensure BGR uint8
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Run YOLO inference
            try:
                results = model(
                    source  = frame,
                    conf    = self._conf,
                    iou     = self._iou,
                    device  = self._device,
                    verbose = False,
                    save    = False,
                )
            except Exception as e:
                log.warning("Inference error: %s", e)
                continue

            # Convert to supervision Detections
            detections = sv.Detections.from_ultralytics(results[0])

            # Annotate frame
            annotated = frame.copy()
            for i in range(len(detections)):
                box   = detections.xyxy[i].astype(int)
                cls   = int(detections.class_id[i]) if detections.class_id is not None else 0
                conf  = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                label = f"{class_names.get(cls, str(cls))} {conf:.2f}"
                color = self._CLASS_COLORS[cls % len(self._CLASS_COLORS)]

                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated,
                              (box[0], box[1] - th - 6),
                              (box[0] + tw + 4, box[1]), color, -1)
                cv2.putText(annotated, label,
                            (box[0] + 2, box[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            self.sig_result.emit(annotated, detections)

            # Compute and report FPS every second
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                self.sig_fps.emit(frame_count / elapsed)
                frame_count = 0
                fps_start   = time.perf_counter()

        self.sig_status.emit("Inference worker stopped", False)
