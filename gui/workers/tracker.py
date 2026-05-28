"""
gui/workers/tracker.py
======================
Apple grading tracker — orientation-aware, robust against double counting.

Supported conveyor orientations (set in config.yaml → conveyor.orientation):
  "LR"  left  → right   (X increasing)   — horizontal belt
  "RL"  right → left    (X decreasing)   — horizontal belt, reversed
  "TB"  top   → bottom  (Y increasing)   — vertical belt, top entry
  "BT"  bottom→ top     (Y decreasing)   — vertical belt, bottom entry  ← default

For ANY orientation the algorithm reduces the 2-D detection to a single
scalar `travel_pos` that monotonically INCREASES as the apple travels:

  LR → travel_pos = cx
  RL → travel_pos = W - cx
  TB → travel_pos = cy
  BT → travel_pos = H - cy        (0 at bottom, H at top)

All gate logic (entry zone, counting band, proximity) operates on travel_pos.
Lane assignment uses the orthogonal axis (Y for LR/RL, X for TB/BT).
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Supported orientations
ORIENTATIONS = ("LR", "RL", "TB", "BT")


@dataclass
class GradeRecord:
    seq_id:      int
    lane:        int        # 1-indexed
    class_id:    int
    class_name:  str
    confidence:  float
    frames_seen: int


class AppleTracker:
    """
    Stateful apple tracker for conveyor grading.
    Works for any of the four conveyor orientations.
    """

    CLASS_NAMES = ["Fresh", "Processing", "Cull"]   # index 0 / 1 / 2

    def __init__(
        self,
        n_lanes:              int   = 3,
        orientation:          str   = "BT",    # see module docstring
        exit_frac:            float = 0.85,    # exit band centre (fraction of travel axis)
        band_half_frac:       float = 0.025,   # ± 2.5 % of travel axis
        entry_frac:           float = 0.35,    # first detection must be < this (travel axis)
        min_frames:           int   = 5,
        max_lost_frames:      int   = 10,
        max_recover_dist:     int   = 80,
        min_count_dist_frac:  float = 0.12,    # proximity buffer: 12 % of travel axis
        count_memory_frames:  int   = 40,
        cull_weight:          float = 1.5,
        hit_threshold:        int   = 20,
        cull_ratio_threshold: float = 0.55,
    ) -> None:
        assert orientation in ORIENTATIONS, \
            f"orientation must be one of {ORIENTATIONS}, got '{orientation}'"

        self._n_lanes             = n_lanes
        self._ori                 = orientation
        self._exit_frac           = exit_frac
        self._band_half_frac      = band_half_frac
        self._entry_frac          = entry_frac
        self._min_frames          = min_frames
        self._max_lost            = max_lost_frames
        self._max_recover         = max_recover_dist
        self._min_count_dist_frac = min_count_dist_frac
        self._count_mem           = count_memory_frames
        self._cull_weight         = cull_weight
        self._hit_threshold       = hit_threshold
        self._cull_ratio_thresh   = cull_ratio_threshold

        self._history:  dict[int, dict] = defaultdict(self._new_history)
        self._lost:     dict[int, dict] = {}
        self._id_map:   dict[int, int]  = {}
        self._recent:   list[dict]      = []
        self._count:    int = 0
        self._frame_no: int = 0

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _travel_and_lane(self, cx: int, cy: int, w: int, h: int) -> tuple[int, int]:
        """
        Returns (travel_pos, lane_1indexed).

        travel_pos: scalar that increases as apple moves toward exit.
        lane:       1-indexed bin along the orthogonal axis.
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
            travel = h - cy   # 0 at bottom, H at top → increases as apple moves up
            lane   = min(int(cx / (w / self._n_lanes)), self._n_lanes - 1) + 1
        return travel, lane

    def _axis_size(self, w: int, h: int) -> int:
        """Length of the travel axis in pixels."""
        return w if self._ori in ("LR", "RL") else h

    @staticmethod
    def _new_history() -> dict:
        return {
            "votes":       defaultdict(float),
            "hit_cull":    0,
            "frames_seen": 0,
            "first_travel": -1,   # travel_pos at very first detection
            "last_pos":    (0, 0),
            "last_frame":  0,
            "committed":   False,
            "lane":        1,
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
        Process one YOLO tracking result.
        Returns (active_list, graded_list).
        """
        self._frame_no += 1
        h, w = frame_shape[:2]

        axis_size = self._axis_size(w, h)
        exit_pos   = int(axis_size * self._exit_frac)
        band_half  = max(30, int(axis_size * self._band_half_frac))
        entry_pos  = int(axis_size * self._entry_frac)
        min_cdist  = max(80, int(axis_size * self._min_count_dist_frac))

        active: list[dict]        = []
        graded: list[GradeRecord] = []

        # Purge stale proximity buffer
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

        # Extract segmentation mask polygons if the model is a seg model.
        # masks.xy is a list of (N, 2) float arrays in original frame pixel space,
        # one entry per detection, same index order as boxes.
        masks_xy: list = [None] * len(track_ids)
        if result.masks is not None:
            raw = result.masks.xy
            if len(raw) == len(track_ids):
                masks_xy = raw

        # ── Step 1: Lost-track recovery for brand-new YOLO IDs ───────────────
        for i, tid in enumerate(track_ids):
            if self._history[tid]["frames_seen"] != 0:
                continue
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
                self._history[tid] = self._lost.pop(best_id)
                if best_id in self._id_map:
                    self._id_map[tid] = self._id_map[best_id]
                log.debug("Track %d ← recovered from %d (dist=%.0f)", tid, best_id, best_dist)

        # ── Step 2: Update votes + gate check ────────────────────────────────
        seen_ids: set[int] = set()

        for tid, cls_id, xyxy, conf, mask_poly in zip(
            track_ids, cls_ids, xyxys, confs, masks_xy
        ):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            travel, lane = self._travel_and_lane(cx, cy, w, h)
            hist = self._history[tid]

            # Record entry travel_pos on very first detection
            if hist["frames_seen"] == 0:
                hist["first_travel"] = travel

            hist["frames_seen"] += 1
            hist["last_pos"]    = (cx, cy)
            hist["last_frame"]  = self._frame_no
            hist["lane"]        = lane

            # Weighted vote accumulation
            weight = self._cull_weight if cls_id == 2 else 1.0
            hist["votes"][cls_id] += conf * weight
            if cls_id == 2 and conf > 0.6:
                hist["hit_cull"] += 1

            seen_ids.add(tid)
            seq_id = self._id_map.get(tid)

            # ── Counting gate ─────────────────────────────────────────────────
            #  Guard 1: narrow band  — apple inside exit_pos ± band_half
            #  Guard 2: entry zone   — first detection was in the START of travel
            #  Guard 3: min frames   — not a phantom
            in_band       = (exit_pos - band_half) <= travel <= (exit_pos + band_half)
            entered_start = 0 <= hist["first_travel"] < entry_pos
            enough_frames = hist["frames_seen"] >= self._min_frames

            if in_band and entered_start and enough_frames and seq_id is None:
                # Proximity check — same physical apple already counted nearby?
                matched_seq = None
                for recent in self._recent:
                    if self._dist((cx, cy), recent["pos"]) < min_cdist:
                        matched_seq = recent["seq_id"]
                        break

                if matched_seq is not None:
                    self._id_map[tid] = matched_seq
                    seq_id = matched_seq
                    log.debug("Track %d linked to existing #%d", tid, matched_seq)
                else:
                    self._count += 1
                    seq_id = self._count
                    self._id_map[tid] = seq_id
                    self._recent.append({
                        "pos":    (cx, cy),
                        "frame":  self._frame_no,
                        "seq_id": seq_id,
                    })
                    log.debug("NEW apple #%d  ori=%s  travel=%d  first=%d  frames=%d",
                              seq_id, self._ori, travel, hist["first_travel"], hist["frames_seen"])

            # ── Grade commit ──────────────────────────────────────────────────
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
                        best_cls, best_conf = 2, cull_ratio
                    else:
                        non_cull = {k: v for k, v in hist["votes"].items() if k != 2}
                        if non_cull:
                            best_cls  = max(non_cull, key=non_cull.get)
                            best_conf = non_cull[best_cls] / total
                        else:
                            best_cls  = max(hist["votes"], key=hist["votes"].get)
                            best_conf = hist["votes"][best_cls] / total

                    hist["committed"] = True
                    cls_name = (self.CLASS_NAMES[best_cls]
                                if best_cls < len(self.CLASS_NAMES) else str(best_cls))
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

            # Live display grade
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
                "eligible":  entered_start,
                "mask":      mask_poly,   # (N, 2) float array or None
            })

        # ── Step 3: Move disappeared tracks to lost buffer ────────────────────
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
