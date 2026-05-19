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
from PyQt6.QtCore import QThread, pyqtSignal

from core.camera.camera_interface import CameraInterface

log = logging.getLogger(__name__)

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
    sig_exposure_readback = pyqtSignal(int, int, int) # actual CH1/CH2/CH3 exposures in µs
    sig_gains_readback    = pyqtSignal(float, float, float)  # actual CH1/CH2/CH3 gains in dB
    sig_cam_fps           = pyqtSignal(float)         # actual camera acquisition FPS (grab thread)
    sig_block_ids         = pyqtSignal(bool, int, int, int) # synced (bool), ch1_bid, ch2_bid, ch3_bid
    sig_awb_completed     = pyqtSignal(bool, float, float)  # success (bool), red_ratio, blue_ratio


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

            ch1_bid = getattr(triplet, "ch1_bid", -1)
            ch2_bid = getattr(triplet, "ch2_bid", -1)
            ch3_bid = getattr(triplet, "ch3_bid", -1)
            synced = (ch1_bid == ch2_bid == ch3_bid) and ch1_bid != -1
            self.sig_block_ids.emit(synced, ch1_bid, ch2_bid, ch3_bid)

            self.sig_frame.emit(triplet.ch1, triplet.ch2, triplet.ch3, fps)

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
        Forward global/reset exposure change to all 3 camera sources.
        Reads back the actual value accepted by firmware and emits sig_exposure_readback
        for all 3 channels.
        """
        if self._camera is not None:
            actual = self._camera.set_exposure(exposure_us)
            if actual > 0:
                self.sig_exposure_readback.emit(actual, actual, actual)
        else:
            log.warning("set_exposure ignored — camera not connected")

    def set_exposures(self, ch1_us: int, ch2_us: int, ch3_us: int) -> None:
        """
        Set independent exposure times per channel while streaming.
        ch1_us → Source0 (Color), ch2_us → Source1 (NIR1), ch3_us → Source2 (NIR2).
        Reads back the actual values accepted by firmware and emits sig_exposure_readback.
        """
        if self._camera is not None:
            actuals = self._camera.set_exposures_per_source([ch1_us, ch2_us, ch3_us])
            while len(actuals) < 3:
                actuals.append(actuals[-1] if actuals else ch1_us)
            self.sig_exposure_readback.emit(actuals[0], actuals[1], actuals[2])
        else:
            log.warning("set_exposures ignored — camera not connected")

    def set_fps(self, fps: float) -> None:
        """
        Set camera hardware FPS AND update display emit rate to match.
        Camera FPS  = sensor acquisition speed (firmware)
        Display FPS = how fast sig_frame is emitted to the GUI
        """
        self._display_fps = fps   # display tries to match camera; caps at machine limit
        if self._camera is not None:
            self._camera.set_fps(fps)
            log.info("CameraWorker: hardware=%.0f FPS, display target=%.0f FPS", fps, fps)
            # Read back independent exposures — firmware may have clamped them at new FPS
            actuals = self._camera.get_exposures_per_source()
            if actuals and len(actuals) >= 1:
                while len(actuals) < 3:
                    actuals.append(actuals[-1])
                self.sig_exposure_readback.emit(actuals[0], actuals[1], actuals[2])
        else:
            log.warning("set_fps ignored — camera not connected")

    def get_exposure(self) -> int:
        """Read current ExposureTime from firmware. Returns -1 if not connected."""
        if self._camera is not None:
            return self._camera.get_exposure()
        return -1

    def set_gain(self, gain_db: float) -> None:
        """
        Set same gain on all 3 sources (used by global Reset only).
        Emits sig_gains_readback with the same actual value for all 3 channels.
        """
        if self._camera is not None:
            actual = self._camera.set_gain(gain_db)
            if actual >= 0:
                self.sig_gains_readback.emit(actual, actual, actual)
        else:
            log.warning("set_gain ignored — camera not connected")

    def set_gains(self, ch1_db: float, ch2_db: float, ch3_db: float) -> None:
        """
        Set independent gain per channel while streaming.
        ch1_db → Source0 (Color), ch2_db → Source1 (NIR1), ch3_db → Source2 (NIR2).
        Emits sig_gains_readback with the actual firmware-accepted values.
        """
        if self._camera is not None:
            actuals = self._camera.set_gain_per_source([ch1_db, ch2_db, ch3_db])
            # Pad to 3 values in case fewer sources exist
            while len(actuals) < 3:
                actuals.append(actuals[-1] if actuals else ch1_db)
            self.sig_gains_readback.emit(actuals[0], actuals[1], actuals[2])
        else:
            log.warning("set_gains ignored — camera not connected")

    def trigger_auto_white_balance(self) -> None:
        """
        Trigger One-Push Auto White Balance on the background camera interface.
        Emits sig_awb_completed upon firmware completion.
        """
        if self._camera is not None:
            success = self._camera.auto_white_balance()
            r_ratio, b_ratio = 1.0, 1.0
            if success and self._camera.mode == "jai" and self._camera._backend:
                try:
                    nm = self._camera._backend._device.GetParameters()
                    ratio_sel = nm.GetEnum("BalanceRatioSelector")
                    ratio_val = nm.GetFloat("BalanceRatio")
                    if ratio_sel and ratio_val:
                        ratio_sel.SetValue("Red")
                        _, r_ratio = ratio_val.GetValue()
                        ratio_sel.SetValue("Blue")
                        _, b_ratio = ratio_val.GetValue()
                except Exception:
                    pass
            self.sig_awb_completed.emit(success, r_ratio, b_ratio)
        else:
            log.warning("trigger_auto_white_balance ignored — camera not connected")
            self.sig_awb_completed.emit(False, -1.0, -1.0)

