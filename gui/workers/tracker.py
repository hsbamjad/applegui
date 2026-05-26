"""
gui/workers/tracker.py
======================
Apple grading tracker — direct adaptation of the chestnut pipeline
(multi_video_processor_10cm_m345.py) for a horizontal 3-lane conveyor.

Algorithm (identical to chestnut):
  1. Each new YOLO track_id gets a sequential apple #ID when it first
     crosses the exit line.
  2. Multi-frame confidence voting accumulates across ALL frames the
     apple is visible, not just the last one.
  3. Lost-track recovery: if a track disappears and re-appears nearby
     within max_lost_frames, its history is merged into the new ID.
  4. Duplicate suppression at the exit line via proximity buffer.
  5. Grade is committed using the same ratio + hit-count rule as chestnuts.

Key adaptation from chestnuts:
  - Chestnut used a Y-axis counting line (single roller, top-to-bottom).
  - We use an X-axis exit line (horizontal belt, left-to-right).
  - 3 grade classes instead of 2 (Cull / Fresh / Processing).
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ── Grade record emitted when a grade is committed ────────────────────────────

@dataclass
class GradeRecord:
    seq_id:      int      # sequential apple counter (#1, #2, ...)
    lane:        int      # 1-indexed lane (derived from Y position)
    class_id:    int
    class_name:  str
    confidence:  float
    frames_seen: int


class AppleTracker:
    """
    Stateful apple tracker — wraps YOLO's built-in tracker and adds:
      - sequential counting IDs
      - multi-frame grade voting
      - lost-track ID recovery
      - exit-line grade commit with duplicate suppression
    """

    CLASS_NAMES = ["Fresh", "Processing", "Cull"]   # index 0/1/2 from best.pt

    def __init__(
        self,
        n_lanes:              int   = 3,
        exit_x_fraction:      float = 0.85,    # fraction of frame width
        min_frames:           int   = 8,        # min detections before commit
        max_lost_frames:      int   = 15,       # frames to keep lost track alive
        max_recover_dist:     int   = 120,      # pixels for lost-track re-link
        min_count_dist:       int   = 100,      # pixels — duplicate suppression
        count_memory_frames:  int   = 15,       # frames to keep counting buffer
        defect_weight:        float = 1.5,      # boost weight for Cull class (id=0)
        hit_threshold:        int   = 20,       # high-conf Cull hits to force commit
        defect_ratio_threshold: float = 0.55,   # Cull vote fraction to force Cull grade
    ) -> None:
        self._n_lanes               = n_lanes
        self._exit_x_frac           = exit_x_fraction
        self._min_frames            = min_frames
        self._max_lost              = max_lost_frames
        self._max_recover           = max_recover_dist
        self._min_count_dist        = min_count_dist
        self._count_mem             = count_memory_frames
        self._defect_weight         = defect_weight
        self._hit_threshold         = hit_threshold
        self._defect_ratio_thresh   = defect_ratio_threshold

        # Per YOLO track_id
        self._track_history: dict[int, dict] = defaultdict(self._new_history)
        # Lost tracks buffer: track_id -> history snapshot
        self._lost: dict[int, dict] = {}
        # Sequential ID mapper: track_id -> seq_id
        self._tracker_id_to_seq_id: dict[int, int] = {}
        # Proximity buffer at exit line (to prevent duplicate counts)
        self._recently_counted: list[dict] = []
        # Global apple counter
        self._count: int = 0
        self._frame_no: int = 0

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _new_history() -> dict:
        return {
            "sum_conf":        defaultdict(float),  # class_id -> weighted conf sum
            "hit_count_cull":  0,                   # frames with high-conf Cull
            "frames_seen":     0,
            "last_pos":        (0, 0),
            "last_frame":      0,
            "committed":       False,
            "lane":            1,
        }

    @staticmethod
    def _dist(p1: tuple, p2: tuple) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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
        active  : list of dicts for annotation (one per visible apple)
        graded  : list of GradeRecord for committed grades this frame
        """
        self._frame_no += 1
        h, w = frame_shape[:2]
        lane_h = h / self._n_lanes
        exit_x = int(w * self._exit_x_frac)

        active: list[dict]        = []
        graded: list[GradeRecord] = []

        if result.boxes is None or result.boxes.id is None:
            # Clean stale counting memory
            self._recently_counted = [
                r for r in self._recently_counted
                if self._frame_no - r["frame"] < self._count_mem
            ]
            return active, graded

        boxes    = result.boxes.cpu().numpy()
        track_ids = boxes.id.astype(int).tolist()
        cls_ids   = boxes.cls.astype(int).tolist()
        xyxys     = boxes.xyxy.tolist()
        confs     = boxes.conf.tolist()

        # ── 1. Lost-track recovery (identical to chestnut) ────────────────────
        new_track_ids = [
            tid for tid in track_ids
            if self._track_history[tid]["frames_seen"] == 0
        ]
        for tid in new_track_ids:
            idx     = track_ids.index(tid)
            cx      = int((xyxys[idx][0] + xyxys[idx][2]) / 2)
            cy      = int((xyxys[idx][1] + xyxys[idx][3]) / 2)
            new_pos = (cx, cy)

            best_id, best_dist = None, float("inf")
            stale = []
            for lost_id, lost_data in self._lost.items():
                if self._frame_no - lost_data["last_frame"] > self._max_lost:
                    stale.append(lost_id)
                    continue
                d = self._dist(new_pos, lost_data["last_pos"])
                if d < self._max_recover and d < best_dist:
                    best_dist, best_id = d, lost_id
            for s in stale:
                del self._lost[s]

            if best_id is not None:
                # Merge lost history into this new YOLO track_id
                self._track_history[tid] = self._lost.pop(best_id)
                if best_id in self._tracker_id_to_seq_id:
                    self._tracker_id_to_seq_id[tid] = \
                        self._tracker_id_to_seq_id[best_id]
                log.debug(
                    "Track %d recovered from lost %d (dist=%.0f)",
                    tid, best_id, best_dist,
                )

        # ── 2. Update history + voting ────────────────────────────────────────
        seen_ids = set()
        for tid, cls_id, xyxy, conf in zip(track_ids, cls_ids, xyxys, confs):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            lane   = min(int(cy / lane_h), self._n_lanes - 1) + 1  # 1-indexed

            hist = self._track_history[tid]
            hist["frames_seen"] += 1
            hist["last_pos"]    = (cx, cy)
            hist["last_frame"]  = self._frame_no
            hist["lane"]        = lane

            # Weighted confidence accumulation (Cull gets extra weight)
            weight = self._defect_weight if cls_id == 0 else 1.0
            hist["sum_conf"][cls_id] += conf * weight
            if cls_id == 0 and conf > 0.6:
                hist["hist_count_cull"] = hist.get("hist_count_cull", 0) + 1

            seen_ids.add(tid)

            # ── 3. Counting / ID assignment at exit line ──────────────────────
            seq_id = self._tracker_id_to_seq_id.get(tid)

            if cx > exit_x and seq_id is None:
                # Check for duplicates using proximity buffer
                is_dup    = False
                match_seq = None
                for recent in self._recently_counted:
                    if self._dist((cx, cy), recent["pos"]) < self._min_count_dist:
                        is_dup    = True
                        match_seq = recent["seq_id"]
                        break

                if is_dup:
                    self._tracker_id_to_seq_id[tid] = match_seq
                    seq_id = match_seq
                else:
                    self._count += 1
                    seq_id = self._count
                    self._tracker_id_to_seq_id[tid] = seq_id
                    self._recently_counted.append({
                        "pos":    (cx, cy),
                        "frame":  self._frame_no,
                        "seq_id": seq_id,
                    })

            # ── 4. Grade commit ───────────────────────────────────────────────
            if (
                seq_id is not None
                and not hist["committed"]
                and hist["frames_seen"] >= self._min_frames
                and cx > exit_x
            ):
                total = sum(hist["sum_conf"].values())
                if total > 0:
                    cull_ratio = hist["sum_conf"].get(0, 0.0) / total
                    hit_cull   = hist.get("hist_count_cull", 0)

                    # Same decision rule as chestnut: ratio OR sustained hits
                    force_cull = (
                        cull_ratio >= self._defect_ratio_thresh
                        or hit_cull >= self._hit_threshold
                    )

                    if force_cull:
                        best_cls  = 0       # Cull
                        best_conf = cull_ratio
                    else:
                        # Best non-Cull class
                        non_cull = {k: v for k, v in hist["sum_conf"].items() if k != 0}
                        if non_cull:
                            best_cls  = max(non_cull, key=non_cull.get)
                            best_conf = non_cull[best_cls] / total
                        else:
                            best_cls  = max(hist["sum_conf"], key=hist["sum_conf"].get)
                            best_conf = hist["sum_conf"][best_cls] / total

                    hist["committed"] = True
                    cls_name = (
                        self.CLASS_NAMES[best_cls]
                        if best_cls < len(self.CLASS_NAMES)
                        else str(best_cls)
                    )
                    rec = GradeRecord(
                        seq_id      = seq_id,
                        lane        = lane,
                        class_id    = best_cls,
                        class_name  = cls_name,
                        confidence  = float(best_conf),
                        frames_seen = hist["frames_seen"],
                    )
                    graded.append(rec)
                    log.info(
                        "Grade #%d  lane=%d  %s  conf=%.2f  frames=%d",
                        seq_id, lane, cls_name, best_conf, hist["frames_seen"],
                    )

            # Current best grade for live annotation
            if hist["sum_conf"]:
                disp_cls  = max(hist["sum_conf"], key=hist["sum_conf"].get)
                total_v   = sum(hist["sum_conf"].values())
                disp_conf = hist["sum_conf"][disp_cls] / total_v
            else:
                disp_cls, disp_conf = cls_id, conf

            active.append({
                "track_id":  tid,
                "seq_id":    seq_id,          # None until apple crosses exit line
                "class_id":  disp_cls,
                "conf":      disp_conf,
                "box":       (x1, y1, x2, y2),
                "center":    (cx, cy),
                "lane":      lane,
                "frames":    hist["frames_seen"],
            })

        # ── 5. Move disappeared tracks to lost buffer ─────────────────────────
        for tid in set(self._track_history.keys()) - seen_ids:
            hist = self._track_history[tid]
            if hist["frames_seen"] > 0 and not hist["committed"]:
                self._lost[tid] = dict(hist)

        # Clean stale counting memory
        self._recently_counted = [
            r for r in self._recently_counted
            if self._frame_no - r["frame"] < self._count_mem
        ]

        return active, graded

    def reset(self) -> None:
        """Call between sessions / video restarts."""
        self._track_history.clear()
        self._lost.clear()
        self._tracker_id_to_seq_id.clear()
        self._recently_counted.clear()
        self._count    = 0
        self._frame_no = 0

    def set_conveyor_speed(self, apples_per_sec: int) -> None:
        self._conveyor_speed_aps = apples_per_sec
