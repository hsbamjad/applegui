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

import os
import time
import logging
import ctypes

import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from core.camera.camera_interface import CameraInterface

log = logging.getLogger(__name__)

# Display resolution for sig_frame emission.
# The camera grab thread produces full 2048x1536 frames (raw data preserved).
# Before emitting to Qt, frames are resized to this size so the GUI thread
# does not have to scale 9 MP images at 60+ FPS (which causes severe lag).
# Inference and saving always use the full-res FrameTriplet, not these copies.
_DISP_W = 640
_DISP_H = 480


def _resize_for_display(frame: np.ndarray) -> np.ndarray:
    """Resize a frame to display resolution using INTER_AREA (best for downscaling)."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if w == _DISP_W and h == _DISP_H:
        return frame   # already correct size (e.g. mock frames)
    return cv2.resize(frame, (_DISP_W, _DISP_H), interpolation=cv2.INTER_AREA)

# Windows high-resolution timer (1ms precision instead of default 15.625ms).
# Without this, time.sleep() rounds to 15.625ms ticks, causing display FPS
# to snap to exactly 64 or 32 FPS regardless of what was requested.
_winmm = None
if os.name == "nt":
    try:
        _winmm = ctypes.windll.winmm
    except Exception:
        pass

def _set_timer_resolution(period_ms: int) -> None:
    """Set Windows multimedia timer resolution (1 = 1ms, 0 = restore default)."""
    if _winmm:
        if period_ms > 0:
            _winmm.timeBeginPeriod(period_ms)
        else:
            _winmm.timeEndPeriod(1)


class CameraWorker(QThread):
    """Background QThread: grabs synchronized frame triplets, emits to GUI."""

    sig_frame             = pyqtSignal(object, object, object, float)  # ch1, ch2, ch3, display_fps
    sig_status            = pyqtSignal(str, bool)                       # message, is_error
    sig_exposure_readback = pyqtSignal(int)    # actual exposure µs read back from firmware
    sig_gain_readback     = pyqtSignal(float)  # actual gain dB read back from firmware
    sig_cam_fps           = pyqtSignal(float)  # actual camera acquisition FPS (grab thread)

    def __init__(self, config: dict, display_fps: int = 30) -> None:
        super().__init__()
        self._config      = config
        self._display_fps = display_fps
        self._running     = False
        self._camera: CameraInterface | None = None  # set during run()

    def run(self) -> None:
        mode = self._config.get("mode", "mock")
        self.sig_status.emit(f"Connecting … (mode={mode})", False)

        # Set Windows timer to 1ms resolution so time.sleep() is precise.
        # Default is 15.625ms (1/64s) which causes FPS to snap to 64 or 32.
        _set_timer_resolution(1)

        self._camera = CameraInterface(self._config)
        if not self._camera.connect():
            self.sig_status.emit("Connection failed", True)
            self._camera = None
            _set_timer_resolution(0)
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
        cam_fps_t      = time.perf_counter()
        # min_interval is computed each loop iteration from self._display_fps
        # so set_fps() takes effect immediately without restarting the thread.

        while self._running:
            t0           = time.perf_counter()
            min_interval = 1.0 / max(self._display_fps, 1)

            triplet = self._camera.grab()

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
                cam_fps = self._camera.grab_fps()
                if cam_fps > 0:
                    self.sig_cam_fps.emit(cam_fps)

            # Resize to display resolution before emitting — keeps Qt fast at 60+ FPS.
            # Raw full-res data remains in `triplet` for inference / saving.
            d1 = _resize_for_display(triplet.ch1)
            d2 = _resize_for_display(triplet.ch2)
            d3 = _resize_for_display(triplet.ch3)
            self.sig_frame.emit(d1, d2, d3, fps)

            sleep = min_interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

        self._camera.disconnect()
        self._camera = None
        _set_timer_resolution(0)   # restore Windows default timer resolution
        self.sig_status.emit("Disconnected", False)

    def stop(self) -> None:
        self._running = False
        self.wait(5000)   # allow up to 5s for clean shutdown (warmup + drain)

    # ── Live camera controls (called from GUI main thread via Qt signal) ───────

    def set_exposure(self, exposure_us: int) -> None:
        """
        Forward exposure change to all 3 camera sources while streaming.
        Reads back the actual value accepted by firmware and emits sig_exposure_readback
        so the GUI spinbox always shows truth (camera may clamp due to FPS limit).
        """
        if self._camera is not None:
            self._camera.set_exposure(exposure_us)
            # Read back actual value — camera may have clamped due to FPS constraint
            actual = self._camera.get_exposure()
            if actual > 0:
                self.sig_exposure_readback.emit(actual)
        else:
            log.warning("set_exposure ignored — camera not connected")


    def set_fps(self, fps: float) -> None:
        """
        Set camera hardware FPS AND update display emit rate to match.

        Camera FPS  = sensor acquisition speed (firmware)
        Display FPS = how fast sig_frame is emitted to the GUI

        The status bar shows real camera FPS; channel headers show actual display FPS.
        """
        self._display_fps = fps   # display tries to match camera; caps at machine limit
        if self._camera is not None:
            self._camera.set_fps(fps)
            log.info("CameraWorker: hardware=%.0f FPS, display target=%.0f FPS",
                     fps, fps)
            # Read back actual exposure — firmware may have clamped it at new FPS
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

    def set_gain(self, gain_db: float) -> None:
        """
        Forward gain change to all 3 camera sources while streaming.
        Reads back the actual value accepted by firmware and emits sig_gain_readback
        so the GUI spinbox always shows truth (camera may clamp the requested value).
        """
        if self._camera is not None:
            actual = self._camera.set_gain(gain_db)
            if actual >= 0:
                self.sig_gain_readback.emit(actual)
        else:
            log.warning("set_gain ignored — camera not connected")
