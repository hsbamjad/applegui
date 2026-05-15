"""
core/camera/camera_interface.py
================================
Camera interface — stub for Phase 3 implementation.

Supports two backends controlled by config["camera"]["mode"]:
  "mock" → MockCamera  (synthetic / replay frames, works on any machine)
  "jai"  → JAICamera   (real hardware, requires JAI eBUS SDK + 10 GigE NIC)

Frame triplet format:
  Each acquisition returns a FrameTriplet of 3 NumPy arrays:
    ch1: np.ndarray shape (1536, 2048) dtype uint8  — RG  ~660nm
    ch2: np.ndarray shape (1536, 2048) dtype uint8  — NIR1 ~800nm
    ch3: np.ndarray shape (1536, 2048) dtype uint8  — NIR2 ~900nm

Threading:
  CameraInterface is NOT thread-safe on its own.
  Use CameraWorker (gui/workers/camera_worker.py — Phase 3) to wrap it
  in a QThread and emit frames via Qt signals.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FrameTriplet:
    """One synchronized capture from all 3 JAI sensors."""
    ch1:        np.ndarray    # RG  ~660nm  shape (H, W) uint8
    ch2:        np.ndarray    # NIR1 ~800nm shape (H, W) uint8
    ch3:        np.ndarray    # NIR2 ~900nm shape (H, W) uint8
    timestamp:  float         # time.time() at acquisition
    frame_idx:  int           # monotonically increasing frame counter


class CameraInterface:
    """
    Abstract camera interface — wraps mock or real JAI camera.

    Phase 3 TODO:
        Implement JAICamera subclass using Harvesters:
            h = Harvester()
            h.add_file(config["camera"]["jai"]["gentl_path"])
            h.update_device_info_list()
            ia = h.create_image_acquirer(0)
            ia.start_image_acquisition()
            with ia.fetch_buffer() as buf:
                ch1 = buf.payload.components[0].data.reshape(1536, 2048)
                ch2 = buf.payload.components[1].data.reshape(1536, 2048)
                ch3 = buf.payload.components[2].data.reshape(1536, 2048)
    """

    def __init__(self, config: dict) -> None:
        self._cfg      = config
        self._mode     = config.get("mode", "mock")
        self._running  = False
        self._frame_idx = 0

    def connect(self) -> bool:
        """Connect to camera. Returns True on success."""
        if self._mode == "mock":
            log.info("MockCamera: connected (synthetic frames).")
            self._running = True
            return True
        else:
            # TODO Phase 3: Harvesters JAI connection
            log.warning("JAI camera not yet implemented — falling back to mock.")
            self._mode    = "mock"
            self._running = True
            return True

    def disconnect(self) -> None:
        self._running = False
        log.info("Camera disconnected.")

    def grab(self) -> Optional[FrameTriplet]:
        """
        Grab the next synchronized frame triplet.
        Returns None if camera is not running.

        Phase 3: replace mock implementation with real Harvesters fetch.
        """
        if not self._running:
            return None

        if self._mode == "mock":
            return self._mock_frame()

        # TODO Phase 3: real Harvesters fetch
        return self._mock_frame()

    def _mock_frame(self) -> FrameTriplet:
        """Generate a synthetic frame triplet for development."""
        cfg  = self._cfg.get("mock", {})
        h, w = cfg.get("resolution", [2048, 1536])[1], cfg.get("resolution", [2048, 1536])[0]
        fps  = cfg.get("fps", 60)

        # Synthetic gradient noise — distinct per channel
        ch1 = np.random.randint(60,  180, (h, w), dtype=np.uint8)
        ch2 = np.random.randint(100, 200, (h, w), dtype=np.uint8)
        ch3 = np.random.randint(80,  160, (h, w), dtype=np.uint8)

        self._frame_idx += 1
        time.sleep(1.0 / fps)

        return FrameTriplet(
            ch1       = ch1,
            ch2       = ch2,
            ch3       = ch3,
            timestamp = time.time(),
            frame_idx = self._frame_idx,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def mode(self) -> str:
        return self._mode
