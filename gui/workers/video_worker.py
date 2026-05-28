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
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


def _read_and_resize(cap: cv2.VideoCapture, target_h: int) -> tuple[bool, np.ndarray | None]:
    """
    Read one frame from cap and downsample it to target_h height (keeping aspect ratio).
    Returns (ok, frame). Uses INTER_AREA for high-quality downscaling.
    Only rescales if the frame is taller than target_h to avoid upscaling.
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None
    h = frame.shape[0]
    if target_h > 0 and h > target_h:
        scale = target_h / h
        new_w = int(frame.shape[1] * scale)
        frame = cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return True, frame


class VideoWorker(QThread):
    """
    Reads 3 video files frame-by-frame and emits synchronized
    frame triplets at the target FPS, looping indefinitely.

    Drop-in replacement for CameraWorker.sig_frame consumers.

    Performance optimisations (simulation only):
      Fix 1 — frames are downscaled to sim_height before emitting.
               YOLO receives the same effective input (imgsz=640 anyway).
               Real CameraWorker is completely unaffected.
      Fix 2 — all 3 cap.read() calls run in parallel via ThreadPoolExecutor,
               cutting the per-frame wall-time roughly in thirds.
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
        sim_height: int = 768,   # Fix 1: downsample target height (0 = disabled)
    ) -> None:
        super().__init__()
        self._paths      = [path_ch1, path_ch2, path_ch3]
        self._fps        = max(1, fps)
        self._loop       = loop
        self._sim_height = max(0, sim_height)
        self._running    = False

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

        h_label = f"{self._sim_height}p" if self._sim_height > 0 else "native"
        self.sig_status.emit(
            f"Video simulation: playing {len(caps)} channels  "
            f"(loop={self._loop}, display={h_label})", False
        )
        self._running = True

        frame_count   = 0
        fps_start     = time.perf_counter()
        reported_fps  = float(self._fps)
        # Diagnostic accumulators — reset every second
        t_read_total  = 0.0
        t_sleep_total = 0.0

        # Fix 2: shared executor — reused every frame, 3 worker threads (one per channel)
        with ThreadPoolExecutor(max_workers=3) as executor:
            while self._running:
                t0           = time.perf_counter()
                min_interval = 1.0 / self._fps

                # Fix 2: submit all 3 reads in parallel
                t_read_start = time.perf_counter()
                futures = [
                    executor.submit(_read_and_resize, cap, self._sim_height)
                    for cap in caps
                ]
                results = [f.result() for f in futures]  # preserves channel order
                t_read_total += time.perf_counter() - t_read_start

                # Check if any channel hit end-of-video
                end_of_video = any(not ok for ok, _ in results)

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

                frames = [frame for _, frame in results]

                # Compute and report FPS every second
                frame_count += 1
                elapsed = time.perf_counter() - fps_start
                if elapsed >= 1.0:
                    reported_fps = frame_count / elapsed
                    log.info(
                        "VideoWorker: %.1f FPS  |  read=%.0fms/frame  sleep=%.0fms/frame  "
                        "frame_shape=%s",
                        reported_fps,
                        (t_read_total / frame_count) * 1000,
                        (t_sleep_total / frame_count) * 1000,
                        frames[0].shape if frames else "?",
                    )
                    frame_count   = 0
                    fps_start     = time.perf_counter()
                    t_read_total  = 0.0
                    t_sleep_total = 0.0

                self.sig_frame.emit(frames[0], frames[1], frames[2], reported_fps)

                # Pace to target FPS
                sleep = min_interval - (time.perf_counter() - t0)
                if sleep > 0:
                    t_sleep_total += sleep
                    time.sleep(sleep)

        for cap in caps:
            cap.release()

        self.sig_status.emit("Video simulation: stopped", False)
