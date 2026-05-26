"""
gui/workers/tracker.py
======================
Apple grading tracker — multi-frame confidence voter.

Adapted from the chestnut multispectral pipeline (multi_video_processor_10cm_m345.py).
Key differences from the previous sv.ByteTrack wrapper:
  - Tracking is now done by YOLO's built-in model.track(persist=True) upstream.
    This file only handles: lane assignment, multi-frame voting, grade commit,
    and lost-track ID recovery.
  - Grades are committed only after an apple:
      1. Has been seen for >= min_frames consecutive detections
      2. Crosses the exit_x line (right side of frame)
  - Confidence is accumulated across frames (weighted sum per class), not just
    the single-frame value.
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


# ── Per-lane ID offsets (keeps global IDs unique and readable) ────────────────
_LANE_ID_OFFSET = [10_000, 20_000, 30_000]


@dataclass
class GradeRecord:
    """Result emitted when an apple's grade is committed."""
    global_id:   int
    lane:        int          # 1-indexed
    class_id:    int
    class_name:  str
    confidence:  float        # fraction of accumulated votes
    frames_seen: int


# ── Track history entry (one per YOLO track_id) ───────────────────────────────
def _new_history():
    return {
        "votes":       defaultdict(float),  # class_id -> accumulated confidence
        "frames_seen": 0,
        "lane":        -1,
        "last_pos":    (0, 0),
        "last_frame":  0,
        "committed":   False,
    }


class AppleTracker:
    """
    Multi-frame confidence voter for 3-lane horizontal conveyor.

    Usage:
        tracker = AppleTracker(...)
        active, graded = tracker.update(yolo_result, frame_shape)
    """

    CLASS_NAMES = ["Cull", "Fresh", "Processing"]   # indices from best.pt

    def __init__(
        self,
        n_lanes:           int   = 3,
        exit_x_fraction:   float = 0.85,   # X position (fraction of width) to commit grade
        min_frames:        int   = 8,       # minimum detections before grade commit
        max_lost_frames:   int   = 15,      # frames to keep lost track in buffer
        max_recover_dist:  int   = 120,     # max pixels to re-link a recovered track
    ) -> None:
        self._n_lanes         = n_lanes
        self._exit_x_frac     = exit_x_fraction
        self._min_frames      = min_frames
        self._max_lost        = max_lost_frames
        self._max_recover     = max_recover_dist

        self._history:    dict[int, dict] = defaultdict(_new_history)
        self._lost:       dict[int, dict] = {}   # track_id -> last known state
        self._frame_no:   int  = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        result,               # ultralytics.engine.results.Results
        frame_shape: tuple,   # (H, W, C)
    ) -> tuple[list[dict], list[GradeRecord]]:
        """
        Process one YOLO tracking result.

        Returns
        -------
        active  : list of dicts — one per currently visible track, for annotation
        graded  : list of GradeRecord — apples whose grade was committed this frame
        """
        self._frame_no += 1
        h, w = frame_shape[:2]
        lane_h   = h / self._n_lanes
        exit_x   = int(w * self._exit_x_frac)

        active: list[dict]        = []
        graded: list[GradeRecord] = []

        # No detections this frame
        if result.boxes is None or result.boxes.id is None:
            return active, graded

        boxes = result.boxes.cpu().numpy()

        track_ids = boxes.id.astype(int).tolist()
        cls_ids   = boxes.cls.astype(int).tolist()
        xyxys     = boxes.xyxy.tolist()
        confs     = boxes.conf.tolist()

        # ── Lost-track recovery ───────────────────────────────────────────────
        # New track IDs (never seen before) may be reassigned YOLO IDs for the
        # same physical apple.  Try to re-link them to a recently-lost track.
        for tid, cls_id, xyxy, conf in zip(track_ids, cls_ids, xyxys, confs):
            if self._history[tid]["frames_seen"] == 0:
                cx = int((xyxy[0] + xyxy[2]) / 2)
                cy = int((xyxy[1] + xyxy[3]) / 2)
                best_id, best_dist = None, float("inf")
                stale = []
                for lost_id, lost in self._lost.items():
                    if self._frame_no - lost["last_frame"] > self._max_lost:
                        stale.append(lost_id)
                        continue
                    d = math.dist((cx, cy), lost["last_pos"])
                    if d < self._max_recover and d < best_dist:
                        best_dist, best_id = d, lost_id
                for s in stale:
                    del self._lost[s]
                if best_id is not None:
                    # Merge lost track's history into this new ID
                    self._history[tid] = self._lost.pop(best_id)
                    log.debug("Track %d recovered from lost track %d (dist=%.0f)", tid, best_id, best_dist)

        # ── Main update loop ──────────────────────────────────────────────────
        seen_ids = set()
        for tid, cls_id, xyxy, conf in zip(track_ids, cls_ids, xyxys, confs):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            lane   = min(int(cy / lane_h), self._n_lanes - 1)

            hist = self._history[tid]
            hist["frames_seen"] += 1
            hist["last_pos"]    = (cx, cy)
            hist["last_frame"]  = self._frame_no
            hist["lane"]        = lane
            hist["votes"][cls_id] += float(conf)

            seen_ids.add(tid)

            # Determine current best grade (for live annotation)
            best_cls   = max(hist["votes"], key=hist["votes"].get)
            total_vote = sum(hist["votes"].values())
            best_conf  = hist["votes"][best_cls] / total_vote if total_vote else conf

            active.append({
                "track_id":  tid,
                "global_id": _LANE_ID_OFFSET[lane] + tid,
                "class_id":  best_cls,
                "conf":      best_conf,
                "box":       (x1, y1, x2, y2),
                "center":    (cx, cy),
                "lane":      lane + 1,          # 1-indexed for display
                "frames":    hist["frames_seen"],
            })

            # ── Grade commit ─────────────────────────────────────────────────
            if (
                cx > exit_x
                and not hist["committed"]
                and hist["frames_seen"] >= self._min_frames
            ):
                hist["committed"] = True
                cls_name = self.CLASS_NAMES[best_cls] if best_cls < len(self.CLASS_NAMES) else str(best_cls)
                rec = GradeRecord(
                    global_id   = _LANE_ID_OFFSET[lane] + tid,
                    lane        = lane + 1,
                    class_id    = best_cls,
                    class_name  = cls_name,
                    confidence  = best_conf,
                    frames_seen = hist["frames_seen"],
                )
                graded.append(rec)
                log.info(
                    "Grade committed: #%d  lane=%d  class=%s  conf=%.2f  frames=%d",
                    rec.global_id, rec.lane, rec.class_name, rec.confidence, rec.frames_seen,
                )

        # ── Move disappeared tracks to lost buffer ────────────────────────────
        all_known = set(self._history.keys())
        for tid in all_known - seen_ids:
            hist = self._history.get(tid)
            if hist and hist["frames_seen"] > 0 and not hist["committed"]:
                self._lost[tid] = hist

        return active, graded

    def reset(self) -> None:
        """Call between sessions / video restarts."""
        self._history.clear()
        self._lost.clear()
        self._frame_no = 0

    def set_conveyor_speed(self, apples_per_sec: int) -> None:
        """Stored for future use (velocity priors, dynamic lost buffer, etc.)."""
        self._conveyor_speed_aps = apples_per_sec
