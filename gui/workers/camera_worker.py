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

    sig_frame            = pyqtSignal(object, object, object, float)  # ch1, ch2, ch3, fps
    sig_status           = pyqtSignal(str, bool)                       # message, is_error
    sig_exposure_readback = pyqtSignal(int)   # actual exposure µs read back from firmware

    def __init__(self, config: dict, display_fps: int = 30) -> None:
        super().__init__()
        self._config      = config
        self._display_fps = display_fps
        self._running     = False
        self._camera: CameraInterface | None = None  # set during run()

    def run(self) -> None:
        mode = self._config.get("mode", "mock")
        self.sig_status.emit(f"Connecting … (mode={mode})", False)

        self._camera = CameraInterface(self._config)
        if not self._camera.connect():
            self.sig_status.emit("Connection failed", True)
            self._camera = None
            return

        actual_mode = self._camera.mode
        self.sig_status.emit(
            f"Connected  ·  {actual_mode.upper()}", False
        )
        self._running = True

        frame_count    = 0
        fps_start      = time.perf_counter()
        fps            = 0.0
        last_frame_idx = -1
        # NOTE: min_interval is NOT cached — read self._display_fps each iteration
        # so that set_fps() takes effect immediately without restarting the thread.

        while self._running:
            t0           = time.perf_counter()
            min_interval = 1.0 / max(self._display_fps, 1)   # dynamic — updated by set_fps()

            triplet = self._camera.grab()

            # Skip if no frame yet or same frame as last emit (cached, no new data)
            if triplet is None or triplet.frame_idx == last_frame_idx:
                sleep = min_interval - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
                continue

            last_frame_idx = triplet.frame_idx
            frame_count   += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                fps_start   = time.perf_counter()

            self.sig_frame.emit(triplet.ch1, triplet.ch2, triplet.ch3, fps)

            sleep = min_interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

        self._camera.disconnect()
        self._camera = None
        self.sig_status.emit("Disconnected", False)

    def stop(self) -> None:
        self._running = False
        self.wait(5000)   # allow up to 5s for clean shutdown (warmup + drain)

    # ── Live camera controls (called from GUI main thread via Qt signal) ───────

    def set_exposure(self, exposure_us: int) -> None:
        """
        Forward exposure change to the camera while streaming.
        Safe to call from the GUI main thread — delegates to CameraInterface
        which issues a single GenICam parameter write (non-blocking, <1ms).

        No effect if camera is not yet connected.
        """
        if self._camera is not None:
            self._camera.set_exposure(exposure_us)
        else:
            log.warning("set_exposure ignored — camera not connected")

    def set_fps(self, fps: float) -> None:
        """
        Set camera acquisition FPS AND update the GUI display rate to match.

        Two things happen:
          1. JAICamera.set_fps() writes AcquisitionFrameRate to the device firmware
          2. self._display_fps is updated so the worker loop emits at the new rate

        After a FPS increase, firmware may silently clamp ExposureTime.
        Reads back actual ExposureTime and emits sig_exposure_readback so the
        GUI exposure spinbox can sync to the real (possibly clamped) value.
        """
        self._display_fps = fps   # update display rate immediately
        if self._camera is not None:
            self._camera.set_fps(fps)
            log.info("CameraWorker: display rate updated to %.0f FPS", fps)
            # Read back actual exposure — firmware may have clamped it
            actual_exp = self._camera.get_exposure()
            if actual_exp > 0:
                self.sig_exposure_readback.emit(actual_exp)
        else:
            log.warning("set_fps ignored — camera not connected")

    def get_exposure(self) -> int:
        """Read current ExposureTime from firmware. Returns -1 if not connected."""
        if self._camera is not None:
            return self._camera.get_exposure()
        return -1
