"""
gui/workers/camera_worker.py
=============================
QThread camera worker — generates mock multispectral frames.

In mock mode produces a moving apple-shaped blob with distinct
spectral signatures per channel (simulates real JAI output visually).

Phase 3: swap _generate_mock_frames() with real Harvesters fetch.

Signals:
  sig_frame(ch1, ch2, ch3, fps)  — new frame triplet ready
  sig_status(message, is_error)  — connection events
"""

from __future__ import annotations

import time
import logging

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.camera.camera_interface import CameraInterface

log = logging.getLogger(__name__)

DISPLAY_W = 480
DISPLAY_H = 360


class CameraWorker(QThread):
    """Background thread: grabs frames, emits to GUI at display FPS."""

    sig_frame  = pyqtSignal(object, object, object, float)  # ch1, ch2, ch3, fps
    sig_status = pyqtSignal(str, bool)                       # message, is_error

    def __init__(self, config: dict, display_fps: int = 30) -> None:
        super().__init__()
        self._config      = config
        self._display_fps = display_fps
        self._running     = False
        self._frame_idx   = 0

    def run(self) -> None:
        self.sig_status.emit("Connecting…", False)

        camera = CameraInterface(self._config)
        if not camera.connect():
            self.sig_status.emit("Connection failed", True)
            return

        self.sig_status.emit("Connected", False)
        self._running = True

        frame_count    = 0
        fps_start      = time.perf_counter()
        fps            = 0.0
        min_interval   = 1.0 / self._display_fps  # throttle to display FPS

        while self._running:
            t0 = time.perf_counter()

            ch1, ch2, ch3 = self._generate_mock_frames(self._frame_idx)
            self._frame_idx += 1
            frame_count    += 1

            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                fps        = frame_count / elapsed
                frame_count = 0
                fps_start  = time.perf_counter()

            self.sig_frame.emit(ch1, ch2, ch3, fps)

            # Throttle to display FPS
            sleep = min_interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

        camera.disconnect()
        self.sig_status.emit("Disconnected", False)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    # ── Mock frame generation ─────────────────────────────────────────────────

    def _generate_mock_frames(
        self, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a synthetic apple blob moving across the frame.
        Each channel shows a different spectral response — visually distinct.

        CH1 (RG ~660nm)  : brighter apple, reddish tones
        CH2 (NIR1 ~800nm): softer, diffuse glow — internal structure
        CH3 (NIR2 ~900nm): dimmer, subtle contrast — water content
        """
        H, W = DISPLAY_H, DISPLAY_W

        # Apple position — moves left to right, repeats every 3 seconds
        t       = frame_idx / self._display_fps
        period  = 3.0
        phase   = (t % period) / period
        cx      = int(phase * (W + 140)) - 70
        cy      = H // 2 + int(18 * np.sin(t * 1.1))

        Y, X = np.mgrid[0:H, 0:W]
        ax, ay = 52, 44   # apple ellipse semi-axes (px)

        dist_sq  = ((X - cx) ** 2 / ax ** 2 + (Y - cy) ** 2 / ay ** 2)
        inside   = (dist_sq < 1.0).astype(np.float32)
        glow     = np.clip(1.8 - dist_sq, 0.0, 1.0) * 0.35

        rng = np.random.default_rng(seed=frame_idx % 200)

        # Background textures (conveyor belt look)
        bg1 = rng.integers(18, 52, (H, W), dtype=np.uint8)
        bg2 = rng.integers(28, 62, (H, W), dtype=np.uint8)
        bg3 = rng.integers(10, 38, (H, W), dtype=np.uint8)

        # Per-channel spectral apple brightness
        ch1 = np.clip(bg1.astype(np.float32) + inside * 170 + glow * 210, 0, 255).astype(np.uint8)
        ch2 = np.clip(bg2.astype(np.float32) + inside * 110 + glow * 160, 0, 255).astype(np.uint8)
        ch3 = np.clip(bg3.astype(np.float32) + inside * 80  + glow * 120, 0, 255).astype(np.uint8)

        return ch1, ch2, ch3
