"""
gui/workers/sweetp_tracker.py
==============================
Sweet potato grading tracker - 2-lane, chestnut-style line-crossing gate.

Design
------
* 2 lanes separated by the frame midpoint on the orthogonal axis.
* Counting gate: a horizontal (or vertical) line at `line_frac` of the
  travel axis, with a tolerance window of ± `band_half_frac`.
  When the object centre crosses into the band and hasn't been counted yet,
  it gets the next global seq_id (first-seen, first-counted - no lane
  interleaving).
* Grade voting: weighted confidence accumulation over all frames, same
  pattern as the chestnut pipeline.
* Lost-track recovery: distance-based, same pattern as chestnut/apple.

Orientation support (same as AppleTracker):
  "LR"  left  → right   (X increasing)   <- default for sweet potato
  "RL"  right → left    (X decreasing)
  "TB"  top   → bottom  (Y increasing)
  "BT"  bottom→ top     (Y decreasing)

Produces the same data format as AppleTracker so the GUI wiring is
100% compatible:
  update() → (active: list[dict], graded: list[GradeRecord])
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
    line_frac           : position of the counting line as fraction of travel axis
    band_half_frac      : +/- half-width of the counting window (fraction)
    entry_frac          : first detection must be in the entry zone (fraction)
    min_frames          : minimum frames seen before a track can be counted
    max_lost_frames     : frames before a lost track is discarded
    max_recover_dist    : pixel radius for lost-track recovery
    min_count_dist      : minimum pixel distance between two new counts (dedup)
    count_memory_frames : how many frames the recent-count buffer lives
    count_merge_frames  : link new track to existing seq within N frames
    defect_weight       : multiplier applied to non-Normal class votes
    hit_threshold       : raw frame hits above this force the defective grade
    defect_ratio_threshold : vote ratio above this forces the defective grade
    min_vote_conf       : ignore votes below this confidence
    min_det_conf        : drop YOLO boxes below this confidence entirely
    """

    # Default class names (overridden when model loads)
    CLASS_NAMES: list[str] = ["Normal", "Moderate defect", "Severe defect"]

    def __init__(
        self,
        n_lanes:                int   = 2,
        orientation:            str   = "LR",
        class_names:            list | None = None,
        line_frac:              float = 0.70,
        band_half_frac:         float = 0.04,
        entry_frac:             float = 0.30,
        min_frames:             int   = 10,
        max_lost_frames:        int   = 12,
        max_recover_dist:       int   = 100,
        min_count_dist:         int   = 120,
        count_memory_frames:    int   = 30,
        count_merge_frames:     int   = 5,
        defect_weight:          float = 1.5,
        hit_threshold:          int   = 20,
        defect_ratio_threshold: float = 0.58,
        min_vote_conf:          float = 0.20,
        min_det_conf:           float = 0.30,
    ) -> None:
        assert orientation in ORIENTATIONS, \
            f"orientation must be one of {ORIENTATIONS}, got '{orientation}'"

        self._n_lanes              = n_lanes
        self._ori                  = orientation
        self._line_frac            = line_frac
        self._band_half_frac       = band_half_frac
        self._entry_frac           = entry_frac
        self._min_frames           = min_frames
        self._max_lost             = max_lost_frames
        self._max_recover          = max_recover_dist
        self._min_count_dist       = min_count_dist
        self._count_mem            = count_memory_frames
        self._count_merge_frames   = count_merge_frames
        self._defect_weight        = defect_weight
        self._hit_threshold        = hit_threshold
        self._defect_ratio_thresh  = defect_ratio_threshold
        self._min_vote_conf        = min_vote_conf
        self._min_det_conf         = min_det_conf

        if class_names:
            self.CLASS_NAMES = list(class_names)

        # State
        self._history:      dict[int, dict] = defaultdict(self._new_history)
        self._lost:         dict[int, dict] = {}
        self._id_map:       dict[int, int]  = {}   # track_id -> seq_id
        self._recent:       list[dict]      = []   # recently-counted buffer
        self._global_count: int = 0                # global sequential counter
        self._frame_no:     int = 0

    # ── Geometry ───────────────────────────────────────────────────────────────

    def _travel_and_lane(self, cx: int, cy: int, w: int, h: int) -> tuple:
        """Return (travel_pos, lane_1indexed).

        travel_pos increases monotonically as object moves toward exit.
        lane is determined by splitting the orthogonal axis into n_lanes bins.
        """
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
        """Length of the travel axis in pixels."""
        return w if self._ori in ("LR", "RL") else h

    @staticmethod
    def _new_history() -> dict:
        return {
            "votes":        defaultdict(float),  # class_id -> weighted conf sum
            "hit_defect":   0,                   # high-conf defect frame count
            "frames_seen":  0,
            "first_travel": -1,
            "last_pos":     (0, 0),
            "last_frame":   0,
            "committed":    False,
            "lane":         1,
        }

    @staticmethod
    def _dist(p1: tuple, p2: tuple) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _is_normal(self, class_id: int) -> bool:
        """Class 0 ('Normal') is the good grade; everything else is a defect."""
        return class_id == 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        result,              # ultralytics Results object
        frame_shape: tuple,  # (H, W, C)
    ) -> tuple:
        """Process one YOLO tracking result.

        Returns
        -------
        active : list[dict]          - current-frame tracks for overlay drawing
        graded : list[GradeRecord]   - newly committed grades (may be empty)
        """
        self._frame_no += 1
        h, w = frame_shape[:2]

        axis_sz   = self._axis_size(w, h)
        line_pos  = int(axis_sz * self._line_frac)
        band_half = max(20, int(axis_sz * self._band_half_frac))
        entry_pos = int(axis_sz * self._entry_frac)

        active = []
        graded = []

        # Purge stale recent buffer
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

        # ── Filter low-confidence boxes ────────────────────────────────────────
        filtered = [
            (t, c, x, f) for t, c, x, f in zip(track_ids, cls_ids, xyxys, confs)
            if f >= self._min_det_conf
        ]
        if not filtered:
            return active, graded
        track_ids, cls_ids, xyxys, confs = map(list, zip(*filtered))

        # ── Lost-track recovery ────────────────────────────────────────────────
        for i, tid in enumerate(track_ids):
            if self._history[tid]["frames_seen"] != 0:
                continue
            cx = int((xyxys[i][0] + xyxys[i][2]) / 2)
            cy = int((xyxys[i][1] + xyxys[i][3]) / 2)

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
                self._history[tid] = self._lost.pop(best_id)
                if best_id in self._id_map:
                    self._id_map[tid] = self._id_map[best_id]
                log.debug("Track %d <- recovered from %d (dist=%.0f)", tid, best_id, best_dist)

        # ── Main update loop ───────────────────────────────────────────────────
        seen_ids = set()

        for tid, cls_id, xyxy, conf in zip(track_ids, cls_ids, xyxys, confs):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy          = (x1 + x2) // 2, (y1 + y2) // 2
            travel, lane    = self._travel_and_lane(cx, cy, w, h)
            hist            = self._history[tid]

            # Record first travel position
            if hist["frames_seen"] == 0:
                hist["first_travel"] = travel

            hist["frames_seen"] += 1
            hist["last_pos"]    = (cx, cy)
            hist["last_frame"]  = self._frame_no
            hist["lane"]        = lane

            # Vote accumulation
            if conf >= self._min_vote_conf:
                weight = self._defect_weight if not self._is_normal(cls_id) else 1.0
                hist["votes"][cls_id] += conf * weight
            # High-confidence defect hits
            if not self._is_normal(cls_id) and conf > 0.6:
                hist["hit_defect"] += 1

            seen_ids.add(tid)
            seq_id = self._id_map.get(tid)

            # ── Counting gate ──────────────────────────────────────────────────
            in_band       = (line_pos - band_half) <= travel <= (line_pos + band_half)
            entered_start = 0 <= hist["first_travel"] < entry_pos
            enough_frames = hist["frames_seen"] >= self._min_frames

            if in_band and entered_start and enough_frames and seq_id is None:
                # Proximity dedup: don't double-count the same object
                matched_seq = None
                for recent in self._recent:
                    age = self._frame_no - recent["frame"]
                    if age > self._count_merge_frames:
                        continue
                    if self._dist((cx, cy), recent["pos"]) < self._min_count_dist:
                        matched_seq = recent["seq_id"]
                        break

                if matched_seq is not None:
                    self._id_map[tid] = matched_seq
                    seq_id = matched_seq
                    log.debug("Track %d merged to existing #%d", tid, matched_seq)
                else:
                    self._global_count += 1
                    seq_id = self._global_count
                    self._id_map[tid] = seq_id
                    self._recent.append({
                        "pos":    (cx, cy),
                        "frame":  self._frame_no,
                        "seq_id": seq_id,
                        "lane":   lane,
                    })
                    log.debug(
                        "NEW sweet potato #%d  lane=%d  travel=%d  first=%d  frames=%d",
                        seq_id, lane, travel, hist["first_travel"], hist["frames_seen"],
                    )

            # ── Grade commit ───────────────────────────────────────────────────
            if (
                seq_id is not None
                and not hist["committed"]
                and hist["frames_seen"] >= self._min_frames
            ):
                total = sum(hist["votes"].values())
                if total > 0:
                    normal_votes = hist["votes"].get(0, 0.0)
                    defect_votes = total - normal_votes
                    defect_ratio = defect_votes / total

                    force_defect = (
                        defect_ratio >= self._defect_ratio_thresh
                        or hist["hit_defect"] >= self._hit_threshold
                    )

                    if force_defect:
                        # Pick the worst (highest-voted) defect class
                        non_normal = {
                            k: v for k, v in hist["votes"].items()
                            if not self._is_normal(k)
                        }
                        if non_normal:
                            best_cls  = max(non_normal, key=non_normal.get)
                            best_conf = defect_ratio
                        else:
                            best_cls  = 0
                            best_conf = normal_votes / total
                    else:
                        best_cls  = max(hist["votes"], key=hist["votes"].get)
                        best_conf = hist["votes"][best_cls] / total

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
                        track_id    = tid,
                    )
                    graded.append(rec)
                    log.info(
                        "Grade #%d  lane=%d  %s  conf=%.2f  frames=%d  "
                        "defect_ratio=%.2f  hits=%d",
                        seq_id, lane, cls_name, best_conf,
                        hist["frames_seen"], defect_ratio, hist["hit_defect"],
                    )

            # ── Live display grade (mirrors commit logic for consistent labels) ──
            if hist["votes"]:
                total_v  = sum(hist["votes"].values())
                norm_v   = hist["votes"].get(0, 0.0)
                defect_v = total_v - norm_v
                defect_r = defect_v / total_v if total_v > 0 else 0.0
                force_v  = (
                    defect_r >= self._defect_ratio_thresh
                    or hist["hit_defect"] >= self._hit_threshold
                )
                if force_v:
                    non_normal = {
                        k: v for k, v in hist["votes"].items()
                        if not self._is_normal(k)
                    }
                    disp_cls  = max(non_normal, key=non_normal.get) if non_normal else 0
                    disp_conf = defect_r
                else:
                    disp_cls  = max(hist["votes"], key=hist["votes"].get)
                    disp_conf = hist["votes"][disp_cls] / total_v
            else:
                disp_cls, disp_conf = cls_id, conf

            active.append({
                "track_id":     tid,
                "seq_id":       seq_id,
                "class_id":     disp_cls,
                "conf":         disp_conf,
                "raw_class_id": cls_id,
                "raw_conf":     conf,
                "box":          (x1, y1, x2, y2),
                "center":       (cx, cy),
                "lane":         lane,
                "frames":       hist["frames_seen"],
                "eligible":     (0 <= hist["first_travel"] < entry_pos),
            })

        # ── Move disappeared tracks to lost buffer ─────────────────────────────
        for tid in set(self._history.keys()) - seen_ids:
            hist = self._history[tid]
            if hist["frames_seen"] > 0:
                self._lost[tid] = dict(hist)

        return active, graded

    # ── Session management ─────────────────────────────────────────────────────

    def reset(self) -> None:
        self._history.clear()
        self._lost.clear()
        self._id_map.clear()
        self._recent.clear()
        self._global_count = 0
        self._frame_no     = 0

    def set_class_names(self, names: list) -> None:
        """Update class names after model loads."""
        self.CLASS_NAMES = list(names)

    @property
    def total_counted(self) -> int:
        return self._global_count
