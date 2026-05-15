"""
gui/workers/inference_worker.py
================================
QThread mock inference worker — simulates YOLOv8m-seg grading output.

Fires grade events at the configured conveyor speed (apples/s/lane × lanes).
Grade distribution reflects dataset: ~58% Fresh, 27% Processing, 15% Cull.
Confidence ranges match real model performance from the ASABE paper.

Phase 4: replace _mock_grade() with real YOLOv8 inference on camera frames.

Signals:
  sig_grade(apple_id, lane, grade, confidence)
"""

from __future__ import annotations

import random
import time
import logging

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

# Grade distribution from dataset (Fresh/Processing/Cull)
GRADE_WEIGHTS = {"Fresh": 58, "Processing": 27, "Cull": 15}
GRADES        = list(GRADE_WEIGHTS.keys())
WEIGHTS       = list(GRADE_WEIGHTS.values())

# Realistic confidence ranges per grade class (from paper results)
CONF_RANGES = {
    "Fresh":      (0.82, 0.99),
    "Processing": (0.65, 0.92),
    "Cull":       (0.70, 0.97),
}

# Grade → outlet mapping (must match hardware)
GRADE_OUTLET = {
    "Fresh":      "A",
    "Processing": "B",
    "Cull":       "C",
}

LANES = 3


class MockInferenceWorker(QThread):
    """
    Simulates the AI grading pipeline for demo / development.

    Fires one grade event per lane per tick at the configured conveyor speed.
    Total throughput = apples_per_sec × LANES.
    """

    sig_grade = pyqtSignal(int, int, str, float, str)
    # (apple_id, lane, grade, confidence, outlet)

    def __init__(self, apples_per_sec: int = 1) -> None:
        super().__init__()
        self._aps      = max(1, apples_per_sec)
        self._running  = False
        self._apple_id = 0

    def set_speed(self, apples_per_sec: int) -> None:
        """Update conveyor speed at runtime."""
        self._aps = max(1, apples_per_sec)

    def run(self) -> None:
        self._running = True
        log.info(f"MockInferenceWorker started at {self._aps} apples/s/lane.")

        while self._running:
            # One grade per lane per tick
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
