"""
gui/workers/sweetp_tracker.py
==============================
Sweet potato grading tracker - 2-lane, chestnut-style line-crossing gate.

Key design (matches chestnut video_processor logic):
  - track_id -> seq_id mapping is the ONLY gate for counting.
    Once a track_id has a seq_id, it is NEVER counted again.
  - Grade commits are guarded by a global _committed_seqs set.
    Once a seq_id is in that set, no further commit for it fires.
  - Proximity dedup: distance-only check in the recent buffer (no age
    filter during dedup - only purge by age at the END of each frame).
  - Lost-track recovery: deep-copies votes to avoid shared-state bugs.

Orientation support:
  "LR"  left  -> right   (X increasing)   <- sweet potato default
  "RL"  right -> left    (X decreasing)
  "TB"  top   -> bottom  (Y increasing)
  "BT"  bottom-> top     (Y decreasing)

Produces the same data format as AppleTracker:
  update() -> (active: list[dict], graded: list[GradeRecord])
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from core.log import get_logger

log = get_logger(__name__)

ORIENTATIONS = ("LR", "RL", "TB", "BT")


@dataclass
class GradeRecord:
    seq_id:      int
    lane:        int          # 1-indexed
    class_id:    int
    class_name:  str
    confidence:  float
    frames_seen: int
    track_id:    int = -1     # ByteTrack ID


class SweetPotatoTracker:
    """
    Stateful 2-lane tracker for sweet potato conveyor grading.

    Parameters
    ----------
    n_lanes             : number of lanes (default 2)
    orientation         : conveyor direction ("LR" | "RL" | "TB" | "BT")
    class_names         : list of class names from the YOLO model
    line_frac           : counting gate position as fraction of travel axis
    band_half_frac      : +/- half-width of the counting window (fraction)
    min_frames          : minimum frames seen before a track can be counted
    max_lost_frames     : frames before a lost track is discarded
    max_recover_dist    : pixel radius for lost-track recovery
    min_count_dist      : minimum pixel distance between two unique counts
    count_memory_frames : how many frames the recent-count buffer lives
    defect_weight       : vote multiplier for non-Normal (defect) classes
    hit_threshold       : raw high-conf defect hits to force defect grade
    defect_ratio_threshold : vote ratio to force defect grade
    min_det_conf        : drop YOLO boxes below this confidence
    min_vote_conf       : ignore votes below this confidence
    """

    CLASS_NAMES: list = ["Normal", "Moderate defect", "Severe defect"]

    def __init__(
        self,
        n_lanes:                int   = 2,
        orientation:            str   = "LR",
        class_names:            list | None = None,
        line_frac:              float = 0.70,
        band_half_frac:         float = 0.04,
        min_frames:             int   = 10,
        max_lost_frames:        int   = 15,
        max_recover_dist:       int   = 120,
        min_count_dist:         int   = 100,
        count_memory_frames:    int   = 30,
        defect_weight:          float = 1.5,
        hit_threshold:          int   = 32,
        defect_ratio_threshold: float = 0.58,
        min_det_conf:           float = 0.30,
        min_vote_conf:          float = 0.20,
        # kept for API compatibility - no longer used
        entry_frac:             float = 0.30,
        count_merge_frames:     int   = 5,
    ) -> None:
        assert orientation in ORIENTATIONS, \
            f"orientation must be one of {ORIENTATIONS}, got '{orientation}'"

        self._n_lanes          = n_lanes
        self._ori              = orientation
        self._line_frac        = line_frac
        self._band_half_frac   = band_half_frac
        self._min_frames       = min_frames
        self._max_lost         = max_lost_frames
        self._max_recover      = max_recover_dist
        self._min_count_dist   = min_count_dist
        self._count_mem        = count_memory_frames
        self._defect_weight    = defect_weight
        self._hit_threshold    = hit_threshold
        self._defect_ratio_thresh = defect_ratio_threshold
        self._min_det_conf     = min_det_conf
        self._min_vote_conf    = min_vote_conf

        if class_names:
            self.CLASS_NAMES = list(class_names)

        # ── State ──────────────────────────────────────────────────────────────
        self._history:  dict[int, dict] = {}          # track_id -> history dict
        self._lost:     dict[int, dict] = {}          # lost track buffer
        self._id_map:   dict[int, int]  = {}          # track_id -> seq_id
        self._recent:   list[dict]      = []          # recently-counted positions
        self._committed_seqs: set[int]  = set()       # seq_ids already committed
        self._global_count: int = 0
        self._frame_no:     int = 0

    # ── Geometry ───────────────────────────────────────────────────────────────

    def _travel_and_lane(self, cx: int, cy: int, w: int, h: int) -> tuple:
        """Return (travel_pos, lane_1indexed)."""
        ori = self._ori
        if ori == "LR":
            travel = cx
            lane   = min(int(cy / (h / self._n_lanes)), self._n_lanes - 1) + 1
        elif ori == "RL":
            travel = w - cx
            lane   = min(int(cy / (h / self._n_lanes)), self._n_lanes - 1) + 1
        elif ori == "TB":
            travel = cy
            lane   = min(int(cx / (w / self._n_lanes)), self._n_lanes - 1) + 1
        else:  # "BT"
            travel = h - cy
            lane   = min(int(cx / (w / self._n_lanes)), self._n_lanes - 1) + 1
        return travel, lane

    def _axis_size(self, w: int, h: int) -> int:
        return w if self._ori in ("LR", "RL") else h

    @staticmethod
    def _new_history() -> dict:
        return {
            "votes_defect": 0.0,   # weighted sum for defect classes
            "votes_normal": 0.0,   # weighted sum for normal class
            "hit_defect":   0,     # high-conf defect frame count
            "frames_seen":  0,
            "last_pos":     (0, 0),
            "last_frame":   0,
            "lane":         1,
            # Per-class vote accumulation for multi-class grading
            "class_votes":  None,  # dict[int, float] - created on first vote
        }

    @staticmethod
    def _dist(p1: tuple, p2: tuple) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _is_normal(self, class_id: int) -> bool:
        """Class 0 is the good/normal grade."""
        return class_id == 0

    def _best_grade(self, hist: dict) -> tuple:
        """Compute (class_id, confidence) from accumulated votes.

        Returns the forced-defect class if ratio/hit thresholds are exceeded,
        otherwise the highest-voted class.
        """
        votes = hist.get("class_votes") or {}
        if not votes:
            return 0, 0.0

        total    = hist["votes_defect"] + hist["votes_normal"]
        if total <= 0:
            return 0, 0.0

        defect_ratio = hist["votes_defect"] / total
        force_defect = (
            defect_ratio >= self._defect_ratio_thresh
            or hist["hit_defect"] >= self._hit_threshold
        )

        if force_defect:
            # Pick highest-voted defect class
            defect_votes = {k: v for k, v in votes.items() if not self._is_normal(k)}
            if defect_votes:
                best_cls  = max(defect_votes, key=defect_votes.get)
                best_conf = defect_ratio
            else:
                best_cls, best_conf = 0, hist["votes_normal"] / total
        else:
            best_cls  = max(votes, key=votes.get)
            total_cls = sum(votes.values())
            best_conf = votes[best_cls] / total_cls if total_cls > 0 else 0.0

        return best_cls, float(best_conf)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        result,              # ultralytics Results object
        frame_shape: tuple,  # (H, W, C)
    ) -> tuple:
        """Process one YOLO tracking result.

        Returns
        -------
        active : list[dict]          - tracks for overlay drawing
        graded : list[GradeRecord]   - newly committed grades (may be empty)
        """
        self._frame_no += 1
        h, w = frame_shape[:2]

        axis_sz   = self._axis_size(w, h)
        line_pos  = int(axis_sz * self._line_frac)
        band_half = max(20, int(axis_sz * self._band_half_frac))

        active = []
        graded = []

        if result.boxes is None or result.boxes.id is None:
            return active, graded

        boxes     = result.boxes.cpu().numpy()
        track_ids = boxes.id.astype(int).tolist()
        cls_ids   = boxes.cls.astype(int).tolist()
        xyxys     = boxes.xyxy.tolist()
        confs     = boxes.conf.tolist()

        # ── Filter low-confidence boxes ────────────────────────────────────────
        filtered = [
            (t, c, x, f) for t, c, x, f in zip(track_ids, cls_ids, xyxys, confs)
            if f >= self._min_det_conf
        ]
        if not filtered:
            return active, graded
        track_ids, cls_ids, xyxys, confs = map(list, zip(*filtered))

        # ── Lost-track recovery ────────────────────────────────────────────────
        stale_lost = [
            lid for lid, ld in self._lost.items()
            if self._frame_no - ld["last_frame"] > self._max_lost
        ]
        for s in stale_lost:
            self._lost.pop(s, None)

        for i, tid in enumerate(track_ids):
            if tid in self._history:
                continue   # already known track - skip recovery
            cx = int((xyxys[i][0] + xyxys[i][2]) / 2)
            cy = int((xyxys[i][1] + xyxys[i][3]) / 2)

            best_id, best_dist = None, float("inf")
            for lost_id, ldata in self._lost.items():
                d = self._dist((cx, cy), ldata["last_pos"])
                if d < self._max_recover and d < best_dist:
                    best_dist, best_id = d, lost_id

            if best_id is not None:
                # Deep-copy the recovered history to avoid shared dict refs
                recovered = dict(self._lost.pop(best_id))
                if recovered.get("class_votes"):
                    recovered["class_votes"] = dict(recovered["class_votes"])
                self._history[tid] = recovered
                if best_id in self._id_map:
                    self._id_map[tid] = self._id_map[best_id]
                log.debug("Track %d <- recovered from %d (dist=%.0f)", tid, best_id, best_dist)

        # ── Main update loop ───────────────────────────────────────────────────
        seen_ids: set[int] = set()

        for tid, cls_id, xyxy, conf in zip(track_ids, cls_ids, xyxys, confs):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2
            travel, lane    = self._travel_and_lane(cx, cy, w, h)

            if tid not in self._history:
                self._history[tid] = self._new_history()

            hist = self._history[tid]
            hist["frames_seen"] += 1
            hist["last_pos"]    = (cx, cy)
            hist["last_frame"]  = self._frame_no
            hist["lane"]        = lane

            # ── Vote accumulation ──────────────────────────────────────────────
            if conf >= self._min_vote_conf:
                if hist["class_votes"] is None:
                    hist["class_votes"] = {}
                weight = self._defect_weight if not self._is_normal(cls_id) else 1.0
                hist["class_votes"][cls_id] = hist["class_votes"].get(cls_id, 0.0) + conf * weight
                if self._is_normal(cls_id):
                    hist["votes_normal"] += conf
                else:
                    hist["votes_defect"] += conf * weight
            if not self._is_normal(cls_id) and conf > 0.6:
                hist["hit_defect"] += 1

            seen_ids.add(tid)

            # ── Counting gate ──────────────────────────────────────────────────
            # RULE: if track_id already has a seq_id, NEVER count it again.
            # This is the primary dedup guard (same as chestnut).
            in_band      = (line_pos - band_half) <= travel <= (line_pos + band_half)
            enough_frames = hist["frames_seen"] >= self._min_frames

            if in_band and enough_frames and tid not in self._id_map:
                # Proximity dedup: check recently counted positions by DISTANCE only.
                # Do NOT filter by age here - only purge the buffer at end of frame.
                matched_seq = None
                for recent in self._recent:
                    if self._dist((cx, cy), recent["pos"]) < self._min_count_dist:
                        matched_seq = recent["seq_id"]
                        break

                if matched_seq is not None:
                    # Merge to existing seq_id - do NOT commit again
                    self._id_map[tid] = matched_seq
                    log.debug("Track %d merged -> existing #%d", tid, matched_seq)
                else:
                    # New unique sweet potato
                    self._global_count += 1
                    self._id_map[tid] = self._global_count
                    self._recent.append({
                        "pos":    (cx, cy),
                        "frame":  self._frame_no,
                        "seq_id": self._global_count,
                        "lane":   lane,
                    })
                    log.debug(
                        "NEW #%d  lane=%d  travel=%d  frames=%d",
                        self._global_count, lane, travel, hist["frames_seen"],
                    )

            seq_id = self._id_map.get(tid)

            # ── Grade commit ───────────────────────────────────────────────────
            # RULE: each seq_id is committed EXACTLY ONCE, guarded by
            # _committed_seqs. hist["committed"] alone is not enough because
            # multiple track_ids can map to the same seq_id.
            if (
                seq_id is not None
                and seq_id not in self._committed_seqs
                and hist["frames_seen"] >= self._min_frames
            ):
                best_cls, best_conf = self._best_grade(hist)
                self._committed_seqs.add(seq_id)
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
                    confidence  = best_conf,
                    frames_seen = hist["frames_seen"],
                    track_id    = tid,
                )
                graded.append(rec)
                log.info(
                    "Grade #%d  lane=%d  %s  conf=%.2f  frames=%d  "
                    "defect_ratio=%.2f  hits=%d",
                    seq_id, lane, cls_name, best_conf,
                    hist["frames_seen"],
                    hist["votes_defect"] / max(hist["votes_defect"] + hist["votes_normal"], 1e-9),
                    hist["hit_defect"],
                )

            # ── Live display grade ─────────────────────────────────────────────
            disp_cls, disp_conf = self._best_grade(hist)
            if disp_cls == 0 and disp_conf == 0.0:
                disp_cls, disp_conf = cls_id, conf

            active.append({
                "track_id": tid,
                "seq_id":   seq_id,
                "class_id": disp_cls,
                "conf":     disp_conf,
                "raw_class_id": cls_id,
                "raw_conf":     conf,
                "box":      (x1, y1, x2, y2),
                "center":   (cx, cy),
                "lane":     lane,
                "frames":   hist["frames_seen"],
                "eligible": True,
            })

        # ── Move disappeared tracks to lost buffer ─────────────────────────────
        for tid in list(self._history.keys()):
            if tid not in seen_ids:
                hist = self._history.pop(tid)
                if hist["frames_seen"] > 0:
                    # Deep-copy to avoid shared defaultdict references
                    lost_entry = dict(hist)
                    if lost_entry.get("class_votes"):
                        lost_entry["class_votes"] = dict(lost_entry["class_votes"])
                    self._lost[tid] = lost_entry

        # ── Purge stale recent buffer (age-based, end of frame) ───────────────
        self._recent = [
            r for r in self._recent
            if self._frame_no - r["frame"] < self._count_mem
        ]

        return active, graded

    # ── Session management ─────────────────────────────────────────────────────

    def reset(self) -> None:
        self._history.clear()
        self._lost.clear()
        self._id_map.clear()
        self._recent.clear()
        self._committed_seqs.clear()
        self._global_count = 0
        self._frame_no     = 0

    def set_class_names(self, names: list) -> None:
        """Update class names after model loads."""
        self.CLASS_NAMES = list(names)

    @property
    def total_counted(self) -> int:
        return self._global_count
