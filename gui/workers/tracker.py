"""
gui/workers/tracker.py
======================
Apple grading tracker — robust adaptation of the chestnut pipeline.

Key improvements over basic implementations:
  1. NARROW COUNTING BAND (not unbounded zone): apple is counted ONLY when its
     center_x passes through a ±band_half window around exit_x.  Once past the
     band, even a new YOLO track_id for the same physical apple will NOT trigger
     another count because it is already outside the band.

  2. ENTRY ZONE GATE: a new YOLO track_id is only eligible for counting if its
     FIRST detection was in the left 35 % of the frame.  IDs born near the exit
     (due to YOLO re-initialisation after a lost track) are silently ignored.

  3. PROXIMITY BUFFER (proportional): no two counts within 12 % of frame width
     of each other within count_memory_frames frames — same logic as chestnuts
     but scaled to resolution.

  4. LOST-TRACK RECOVERY: when a new YOLO ID appears close to a recently-lost
     track, their histories are merged (same seq_id, same accumulated votes).

  5. 3-CLASS WEIGHTED VOTING: Cull gets a ×weight boost (same as chestnut's
     defect_weight), and the final grade is decided by ratio OR sustained-hit
     rule, producing stable grades even from flickering detections.
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class GradeRecord:
    seq_id:      int
    lane:        int      # 1-indexed  (derived from Y position)
    class_id:    int
    class_name:  str
    confidence:  float
    frames_seen: int


class AppleTracker:
    """Stateful apple tracker for a horizontal 3-lane conveyor."""

    CLASS_NAMES = ["Fresh", "Processing", "Cull"]  # 0 / 1 / 2

    # ── Tuneable defaults ─────────────────────────────────────────────────────
    def __init__(
        self,
        n_lanes:              int   = 3,
        exit_x_frac:          float = 0.85,   # centre of counting band
        band_half_frac:       float = 0.025,  # ± 2.5 % of width → ~51 px @ 2048
        entry_x_frac:         float = 0.35,   # first detection must be LEFT of this
        min_frames:           int   = 5,      # min tracked frames before commit
        max_lost_frames:      int   = 10,     # frames to keep a lost track alive
        max_recover_dist:     int   = 80,     # px — max displacement for recovery
        min_count_dist_frac:  float = 0.12,   # proximity buffer: 12 % of width
        count_memory_frames:  int   = 40,     # frames to keep proximity buffer entries
        cull_weight:          float = 1.5,    # extra weight for Cull (class 2)
        hit_threshold:        int   = 20,     # high-conf Cull frames → force Cull
        cull_ratio_threshold: float = 0.55,   # Cull vote fraction → force Cull
    ) -> None:
        self._n_lanes             = n_lanes
        self._exit_x_frac         = exit_x_frac
        self._band_half_frac      = band_half_frac
        self._entry_x_frac        = entry_x_frac
        self._min_frames          = min_frames
        self._max_lost            = max_lost_frames
        self._max_recover         = max_recover_dist
        self._min_count_dist_frac = min_count_dist_frac
        self._count_mem           = count_memory_frames
        self._cull_weight         = cull_weight
        self._hit_threshold       = hit_threshold
        self._cull_ratio_thresh   = cull_ratio_threshold

        # Per YOLO track_id
        self._history: dict[int, dict]         = defaultdict(self._new_history)
        # Lost tracks: track_id → last history snapshot
        self._lost:    dict[int, dict]         = {}
        # Mapping: YOLO track_id → sequential apple #ID
        self._id_map:  dict[int, int]          = {}
        # Proximity buffer at exit band
        self._recent:  list[dict]              = []
        # Global apple counter
        self._count:   int = 0
        self._frame_no: int = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _new_history() -> dict:
        return {
            "votes":         defaultdict(float),  # class_id → accumulated conf
            "hit_cull":      0,                   # frames with high-conf Cull
            "frames_seen":   0,
            "first_cx":      -1,                  # cx at very first detection
            "last_pos":      (0, 0),
            "last_frame":    0,
            "committed":     False,
            "lane":          1,
        }

    @staticmethod
    def _dist(p1: tuple, p2: tuple) -> float:
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        result,               # ultralytics Results
        frame_shape: tuple,   # (H, W, C)
    ) -> tuple[list[dict], list[GradeRecord]]:
        """
        Returns
        -------
        active  – list of dicts (one per visible apple) for annotation
        graded  – list of GradeRecord for grades committed this frame
        """
        self._frame_no += 1
        h, w = frame_shape[:2]
        lane_h    = h / self._n_lanes
        exit_x    = int(w * self._exit_x_frac)
        band_half = max(30, int(w * self._band_half_frac))
        entry_x   = int(w * self._entry_x_frac)
        min_cdist = max(80, int(w * self._min_count_dist_frac))

        active: list[dict]        = []
        graded: list[GradeRecord] = []

        # ── Purge stale proximity buffer entries ──────────────────────────────
        self._recent = [
            r for r in self._recent
            if self._frame_no - r["frame"] < self._count_mem
        ]

        if result.boxes is None or result.boxes.id is None:
            return active, graded

        boxes     = result.boxes.cpu().numpy()
        track_ids = boxes.id.astype(int).tolist()
        cls_ids   = boxes.cls.astype(int).tolist()
        xyxys     = boxes.xyxy.tolist()
        confs     = boxes.conf.tolist()

        # ── Step 1: Lost-track recovery for brand-new YOLO IDs ───────────────
        for i, tid in enumerate(track_ids):
            if self._history[tid]["frames_seen"] != 0:
                continue   # not new

            cx  = int((xyxys[i][0] + xyxys[i][2]) / 2)
            cy  = int((xyxys[i][1] + xyxys[i][3]) / 2)

            best_id, best_dist = None, float("inf")
            stale = []
            for lost_id, ldata in self._lost.items():
                if self._frame_no - ldata["last_frame"] > self._max_lost:
                    stale.append(lost_id)
                    continue
                d = self._dist((cx, cy), ldata["last_pos"])
                if d < self._max_recover and d < best_dist:
                    best_dist, best_id = d, lost_id
            for s in stale:
                del self._lost[s]

            if best_id is not None:
                # Merge history — same physical apple, continuity preserved
                self._history[tid] = self._lost.pop(best_id)
                if best_id in self._id_map:
                    self._id_map[tid] = self._id_map[best_id]
                log.debug("Track %d ← recovered from %d (dist=%.0f)", tid, best_id, best_dist)

        # ── Step 2: Update vote history ───────────────────────────────────────
        seen_ids: set[int] = set()
        for tid, cls_id, xyxy, conf in zip(track_ids, cls_ids, xyxys, confs):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            lane   = min(int(cy / lane_h), self._n_lanes - 1) + 1  # 1-indexed
            hist   = self._history[tid]

            # Record entry position once
            if hist["frames_seen"] == 0:
                hist["first_cx"] = cx

            hist["frames_seen"] += 1
            hist["last_pos"]    = (cx, cy)
            hist["last_frame"]  = self._frame_no
            hist["lane"]        = lane

            # Weighted vote accumulation (Cull = class 2, gets extra weight)
            weight = self._cull_weight if cls_id == 2 else 1.0
            hist["votes"][cls_id] += conf * weight
            if cls_id == 2 and conf > 0.6:
                hist["hit_cull"] += 1

            seen_ids.add(tid)

            # ── Step 3: Counting gate (NARROW BAND only) ─────────────────────
            #  • Apple center must be INSIDE the band (not just past exit_x)
            #  • Apple's first detection must be LEFT of entry_x (anti-duplicate)
            #  • Must have been tracked for min_frames already
            #  • Must not already have a seq_id
            seq_id = self._id_map.get(tid)
            in_band       = (exit_x - band_half) <= cx <= (exit_x + band_half)
            entered_left  = 0 <= hist["first_cx"] < entry_x
            enough_frames = hist["frames_seen"] >= self._min_frames

            if in_band and entered_left and enough_frames and seq_id is None:
                # Proximity check: is there already a count nearby?
                matched_seq = None
                for recent in self._recent:
                    if self._dist((cx, cy), recent["pos"]) < min_cdist:
                        matched_seq = recent["seq_id"]
                        break

                if matched_seq is not None:
                    # Link to existing count — same physical apple
                    self._id_map[tid] = matched_seq
                    seq_id = matched_seq
                    log.debug("Track %d linked to existing #%d (proximity)", tid, matched_seq)
                else:
                    # New apple!
                    self._count  += 1
                    seq_id        = self._count
                    self._id_map[tid] = seq_id
                    self._recent.append({"pos": (cx, cy), "frame": self._frame_no, "seq_id": seq_id})
                    log.debug("Track %d → NEW apple #%d  (frames=%d  first_cx=%d  cx=%d)",
                              tid, seq_id, hist["frames_seen"], hist["first_cx"], cx)

            # ── Step 4: Grade commit when counted + enough evidence ───────────
            if (
                seq_id is not None
                and not hist["committed"]
                and hist["frames_seen"] >= self._min_frames
            ):
                total = sum(hist["votes"].values())
                if total > 0:
                    cull_ratio = hist["votes"].get(2, 0.0) / total
                    force_cull = (
                        cull_ratio >= self._cull_ratio_thresh
                        or hist["hit_cull"] >= self._hit_threshold
                    )
                    if force_cull:
                        best_cls  = 2
                        best_conf = cull_ratio
                    else:
                        # Best among Fresh (0) and Processing (1)
                        non_cull = {k: v for k, v in hist["votes"].items() if k != 2}
                        if non_cull:
                            best_cls  = max(non_cull, key=non_cull.get)
                            best_conf = non_cull[best_cls] / total
                        else:
                            best_cls  = max(hist["votes"], key=hist["votes"].get)
                            best_conf = hist["votes"][best_cls] / total

                    hist["committed"] = True
                    cls_name = (self.CLASS_NAMES[best_cls]
                                if best_cls < len(self.CLASS_NAMES)
                                else str(best_cls))
                    rec = GradeRecord(
                        seq_id      = seq_id,
                        lane        = lane,
                        class_id    = best_cls,
                        class_name  = cls_name,
                        confidence  = float(best_conf),
                        frames_seen = hist["frames_seen"],
                    )
                    graded.append(rec)
                    log.info("Grade #%d  lane=%d  %s  conf=%.2f  frames=%d",
                             seq_id, lane, cls_name, best_conf, hist["frames_seen"])

            # ── Live display grade (best vote so far) ─────────────────────────
            if hist["votes"]:
                disp_cls  = max(hist["votes"], key=hist["votes"].get)
                total_v   = sum(hist["votes"].values())
                disp_conf = hist["votes"][disp_cls] / total_v
            else:
                disp_cls, disp_conf = cls_id, conf

            active.append({
                "track_id":  tid,
                "seq_id":    seq_id,
                "class_id":  disp_cls,
                "conf":      disp_conf,
                "box":       (x1, y1, x2, y2),
                "center":    (cx, cy),
                "lane":      lane,
                "frames":    hist["frames_seen"],
                "eligible":  entered_left,   # True = apple entered from left
            })

        # ── Step 5: Move disappeared tracks to lost buffer ────────────────────
        for tid in set(self._history.keys()) - seen_ids:
            hist = self._history[tid]
            if hist["frames_seen"] > 0:
                self._lost[tid] = dict(hist)

        return active, graded

    # ── Session management ────────────────────────────────────────────────────

    def reset(self) -> None:
        self._history.clear()
        self._lost.clear()
        self._id_map.clear()
        self._recent.clear()
        self._count    = 0
        self._frame_no = 0

    @property
    def total_counted(self) -> int:
        return self._count
