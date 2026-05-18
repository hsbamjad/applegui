"""
gui/workers/camera_worker.py
=============================
QThread camera worker — drives the real JAI camera or mock backend.

Signals:
  sig_frame(ch1, ch2, ch3, fps)  — new hardware-synchronized frame triplet
  sig_status(message, is_error)  — connection / error events

Mode is controlled by config["mode"]:
  "mock" → synthetic frames (works without camera hardware)
  "jai"  → real JAI FS-3200T via eBUS Python SDK
"""

from __future__ import annotations

import time
import logging

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.camera.camera_interface import CameraInterface

log = logging.getLogger(__name__)


class CameraWorker(QThread):
    """Background QThread: grabs synchronized frame triplets, emits to GUI."""

    sig_frame  = pyqtSignal(object, object, object, float)  # ch1, ch2, ch3, fps
    sig_status = pyqtSignal(str, bool)                       # message, is_error

    def __init__(self, config: dict, display_fps: int = 30) -> None:
        super().__init__()
        self._config      = config
        self._display_fps = display_fps
        self._running     = False

    def run(self) -> None:
        mode = self._config.get("mode", "mock")
        self.sig_status.emit(f"Connecting … (mode={mode})", False)

        camera = CameraInterface(self._config)
        if not camera.connect():
            self.sig_status.emit("Connection failed", True)
            return

        actual_mode = camera.mode
        self.sig_status.emit(
            f"Connected  ·  {actual_mode.upper()}", False
        )
        self._running = True

        frame_count = 0
        fps_start   = time.perf_counter()
        fps         = 0.0
        last_idx    = -1

        while self._running:
            # Blocking grab — waits on Condition until a NEW frame is ready
            # No sleep needed: timing is driven by the camera via process thread
            triplet = camera.grab(last_idx=last_idx)
            if triplet is None:
                continue

            last_idx    = triplet.frame_idx
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                fps_start   = time.perf_counter()

            self.sig_frame.emit(triplet.ch1, triplet.ch2, triplet.ch3, fps)


        camera.disconnect()
        self.sig_status.emit("Disconnected", False)

    def stop(self) -> None:
        self._running = False
        self.wait(5000)   # allow up to 5s for clean shutdown (warmup + drain)
