"""
gui/workers/video_worker.py
============================
QThread video simulation worker — replaces the live CameraWorker
during development and testing (no physical camera required).

Reads 3 pre-recorded video files (one per channel) and emits
synchronized frame triplets at the configured FPS, identical to
CameraWorker so the rest of the pipeline requires zero changes.

Signals (mirror CameraWorker):
  sig_frame(ch1, ch2, ch3, fps)  -- numpy arrays, same as live camera
  sig_status(message, is_error)  -- connection / playback events

Usage:
  worker = VideoWorker(
      path_ch1="videos/Source0/G1/G1.mp4",
      path_ch2="videos/Source1/G1/G1.avi",
      path_ch3="videos/Source2/G1/G1.avi",
      fps=30,
      loop=True,
  )
  worker.sig_frame.connect(self._on_frame)
  worker.sig_status.connect(self._on_status)
  worker.start()
"""

from __future__ import annotations

import time
import logging

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


class VideoWorker(QThread):
    """
    Reads 3 video files frame-by-frame and emits synchronized
    frame triplets at the target FPS, looping indefinitely.

    Drop-in replacement for CameraWorker.sig_frame consumers.
    """

    sig_frame  = pyqtSignal(object, object, object, float)  # ch1, ch2, ch3, fps
    sig_status = pyqtSignal(str, bool)                       # message, is_error

    def __init__(
        self,
        path_ch1: str,
        path_ch2: str,
        path_ch3: str,
        fps: int = 30,
        loop: bool = True,
    ) -> None:
        super().__init__()
        self._paths   = [path_ch1, path_ch2, path_ch3]
        self._fps     = max(1, fps)
        self._loop    = loop
        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_fps(self, fps: int) -> None:
        """Update playback speed at runtime (takes effect on next frame)."""
        self._fps = max(1, fps)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    # ── QThread entry point ──────────────────────────────────────────────────

    def run(self) -> None:
        # Open all 3 captures
        caps = [cv2.VideoCapture(p) for p in self._paths]

        for i, (cap, path) in enumerate(zip(caps, self._paths)):
            if not cap.isOpened():
                self.sig_status.emit(
                    f"Video simulation: cannot open CH{i+1} file: {path}", True
                )
                for c in caps:
                    c.release()
                return

        self.sig_status.emit(
            f"Video simulation: playing {len(caps)} channels  (loop={self._loop})", False
        )
        self._running = True

        frame_count = 0
        fps_start   = time.perf_counter()
        reported_fps = float(self._fps)

        while self._running:
            t0           = time.perf_counter()
            min_interval = 1.0 / self._fps

            frames = []
            end_of_video = False

            for cap in caps:
                ok, frame = cap.read()
                if not ok:
                    end_of_video = True
                    break
                frames.append(frame)

            if end_of_video:
                if self._loop:
                    # Rewind all captures to frame 0
                    for cap in caps:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    log.debug("VideoWorker: looping back to start")
                    continue
                else:
                    log.info("VideoWorker: end of video, stopping")
                    break

            # Compute and report FPS every second
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                reported_fps = frame_count / elapsed
                frame_count  = 0
                fps_start    = time.perf_counter()

            self.sig_frame.emit(frames[0], frames[1], frames[2], reported_fps)

            # Pace to target FPS
            sleep = min_interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

        for cap in caps:
            cap.release()

        self.sig_status.emit("Video simulation: stopped", False)
