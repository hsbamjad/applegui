"""
gui/workers/tracker.py
=======================
ConveyorTracker — ByteTrack-based multi-object tracker for 3-lane conveyor.

Architecture decision:
  One sv.ByteTrack instance per lane. Detections are split by Y-center into
  lane zones BEFORE tracking. This makes cross-lane ID swaps impossible by
  design, regardless of how close apples are to each other.

Lane zones (equal thirds of frame height, top-to-bottom):
  Lane 1  →  y in [0,       H/3)
  Lane 2  →  y in [H/3,   2H/3)
  Lane 3  →  y in [2H/3,    H)

Global track ID = lane_offset + local ByteTrack ID:
  Lane 1: 10_001 – 19_999
  Lane 2: 20_001 – 29_999
  Lane 3: 30_001 – 39_999

Returned object:
  ConveyorTracker.update() returns a TrackResult namedtuple with:
    tracked   : sv.Detections  (tracker_id set, data dict has 'lane' key)
    exited    : list[dict]     (tracks that crossed the exit line this frame)
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Suppress ByteTrack FutureWarning (deprecated in sv 0.28 → 0.30 replacement TBD)
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

try:
    import supervision as sv
    _SV_AVAILABLE = True
except ImportError:
    _SV_AVAILABLE = False
    log.warning("supervision not installed — ConveyorTracker disabled")


# ── Lane-level ID offsets ──────────────────────────────────────────────────────
_LANE_ID_OFFSET = [10_000, 20_000, 30_000]  # lane 0/1/2


@dataclass
class ExitedTrack:
    """Metadata for a track that has crossed the exit line."""
    global_id: int
    lane: int          # 1-indexed (1, 2, 3)
    class_id: int
    confidence: float
    last_x: float      # last known X-center (pixels)


class ConveyorTracker:
    """
    3-lane conveyor tracker built on supervision ByteTrack.

    Parameters
    ----------
    n_lanes : int
        Number of physical conveyor lanes (default 3).
    frame_fps : int
        Expected camera/video FPS — used for ByteTrack's lost_track_buffer timing.
    lost_track_buffer : int
        Frames to keep a lost track before dropping it. Increase for slower conveyors
        where apples disappear behind each other briefly.
    minimum_matching_threshold : float
        IoU threshold for matching detections to existing tracks (0–1).
        Lower = more permissive matching; raise if IDs swap frequently.
    track_activation_threshold : float
        Minimum detection confidence to start a new track.
    exit_x_fraction : float
        Normalised X position (0–1) beyond which a track is considered to have
        exited the frame and its grade is committed. Matches grade_line_x in config.
    """

    def __init__(
        self,
        n_lanes: int = 3,
        frame_fps: int = 30,
        lost_track_buffer: int = 45,
        minimum_matching_threshold: float = 0.70,
        track_activation_threshold: float = 0.25,
        exit_x_fraction: float = 0.85,
    ) -> None:
        self._n_lanes   = n_lanes
        self._fps       = frame_fps
        self._exit_x_fr = exit_x_fraction

        if not _SV_AVAILABLE:
            self._trackers = []
            return

        # One ByteTrack per lane — guaranteed zero cross-lane ID confusion
        self._trackers = [
            sv.ByteTrack(
                track_activation_threshold  = track_activation_threshold,
                lost_track_buffer           = lost_track_buffer,
                minimum_matching_threshold  = minimum_matching_threshold,
                frame_rate                  = frame_fps,
            )
            for _ in range(n_lanes)
        ]

        # Global state
        self._lane_of: dict[int, int]   = {}   # global_id → lane (1-indexed)
        self._active_ids: set[int]       = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_fps(self, fps: int) -> None:
        """Update FPS parameter. Takes effect on next tracker reset."""
        self._fps = fps

    def reset(self) -> None:
        """Reset all lane trackers (call between sessions/videos)."""
        for t in self._trackers:
            t.reset()
        self._lane_of.clear()
        self._active_ids.clear()

    def update(
        self, detections: "sv.Detections", frame_shape: tuple[int, int]
    ) -> tuple["sv.Detections", list[ExitedTrack]]:
        """
        Run one tracking step.

        Parameters
        ----------
        detections : sv.Detections
            Raw YOLO detections for this frame (no tracker_id yet).
        frame_shape : (height, width)
            Shape of the frame in pixels (used for lane zone computation).

        Returns
        -------
        tracked : sv.Detections
            Detections with tracker_id set. data['lane'] contains 1-indexed lane.
        exited  : list[ExitedTrack]
            Tracks that crossed the exit line this frame — ready for grade commit.
        """
        if not _SV_AVAILABLE or not self._trackers:
            return detections, []

        frame_h, frame_w = frame_shape[:2]
        zone_h = frame_h / self._n_lanes
        exit_x = frame_w * self._exit_x_fr

        # ── Split detections by lane zone ─────────────────────────────────────
        all_tracked_parts: list[sv.Detections] = []

        for lane_idx in range(self._n_lanes):
            y_min = lane_idx * zone_h
            y_max = (lane_idx + 1) * zone_h

            # Mask: detections whose Y-center falls in this lane's zone
            if len(detections) == 0:
                lane_dets = detections
            else:
                cy = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2.0
                mask = (cy >= y_min) & (cy < y_max)
                lane_dets = detections[mask]

            # Run this lane's ByteTracker
            lane_tracked = self._trackers[lane_idx].update_with_detections(lane_dets)

            if len(lane_tracked) == 0:
                all_tracked_parts.append(lane_tracked)
                continue

            # Remap local tracker IDs → globally unique IDs
            offset = _LANE_ID_OFFSET[lane_idx]
            global_ids = lane_tracked.tracker_id + offset

            # Record lane assignment for each track (sticky — first assignment wins)
            for gid in global_ids:
                if gid not in self._lane_of:
                    self._lane_of[gid] = lane_idx + 1   # 1-indexed

            # Attach global IDs and lane data back to the detection object
            lane_tracked = sv.Detections(
                xyxy       = lane_tracked.xyxy,
                confidence = lane_tracked.confidence,
                class_id   = lane_tracked.class_id,
                tracker_id = global_ids,
                data       = {"lane": np.full(len(lane_tracked), lane_idx + 1, dtype=np.int32)},
            )
            all_tracked_parts.append(lane_tracked)

        # ── Merge all lane results into one Detections object ─────────────────
        merged = _merge_detections(all_tracked_parts)

        # ── Detect exited tracks ──────────────────────────────────────────────
        exited: list[ExitedTrack] = []
        if len(merged) > 0 and merged.tracker_id is not None:
            cx = (merged.xyxy[:, 0] + merged.xyxy[:, 2]) / 2.0
            for i, gid in enumerate(merged.tracker_id):
                if cx[i] >= exit_x and gid in self._active_ids:
                    exited.append(ExitedTrack(
                        global_id  = int(gid),
                        lane       = int(merged.data["lane"][i]),
                        class_id   = int(merged.class_id[i]) if merged.class_id is not None else -1,
                        confidence = float(merged.confidence[i]) if merged.confidence is not None else 0.0,
                        last_x     = float(cx[i]),
                    ))
                    self._active_ids.discard(gid)
                else:
                    self._active_ids.add(gid)

        return merged, exited


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_detections(parts: list) -> "sv.Detections":
    """Concatenate a list of sv.Detections into one, handling empty parts."""
    non_empty = [p for p in parts if p is not None and len(p) > 0]
    if not non_empty:
        return sv.Detections.empty()
    if len(non_empty) == 1:
        return non_empty[0]

    xyxy       = np.concatenate([p.xyxy for p in non_empty], axis=0)
    confidence = np.concatenate([p.confidence for p in non_empty]) if non_empty[0].confidence is not None else None
    class_id   = np.concatenate([p.class_id for p in non_empty])   if non_empty[0].class_id   is not None else None
    tracker_id = np.concatenate([p.tracker_id for p in non_empty]) if non_empty[0].tracker_id  is not None else None
    lane       = np.concatenate([p.data["lane"] for p in non_empty if "lane" in p.data])

    return sv.Detections(
        xyxy       = xyxy,
        confidence = confidence,
        class_id   = class_id,
        tracker_id = tracker_id,
        data       = {"lane": lane},
    )
