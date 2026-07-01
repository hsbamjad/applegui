"""
gui/workers/tracker.py
======================
Apple grading tracker - orientation-aware, robust against double counting.

Supported conveyor orientations (set in config.yaml → conveyor.orientation):
  "LR"  left  → right   (X increasing)   - horizontal belt
  "RL"  right → left    (X decreasing)   - horizontal belt, reversed
  "TB"  top   → bottom  (Y increasing)   - vertical belt, top entry
  "BT"  bottom→ top     (Y decreasing)   - vertical belt, bottom entry  ← default

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
from core.log import get_logger
from collections import defaultdict
from dataclasses import dataclass

log = get_logger(__name__)

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
    track_id:    int = -1   # ByteTrack ID - used by AppleSizeAccumulator


class AppleTracker:
    """
    Stateful apple tracker for conveyor grading.
    Works for any of the four conveyor orientations.
    """

    CLASS_NAMES = ["Fresh", "Processing", "Cull"]   # index 0 / 1 / 2

    def __init__(
        self,
        n_lanes:                    int   = 3,
        orientation:                str   = "BT",
        exit_frac:                  float = 0.85,
        band_half_frac:             float = 0.025,
        entry_frac:                 float = 0.35,
        min_frames:                 int   = 25,
        max_lost_frames:            int   = 10,
        max_recover_dist:           int   = 80,
        min_count_dist_frac:        float = 0.12,
        count_memory_frames:        int   = 40,
        count_merge_frames:         int   = 5,    # only merge to existing seq within N frames
        cull_weight:                float = 1.0,
        hit_threshold:              int   = 20,
        cull_ratio_threshold:       float = 0.65,
        min_vote_conf:              float = 0.20,
        min_det_conf:               float = 0.35,
        peak_conf_override:         float = 0.50,
        overwhelming_cull_threshold: int  = 40,   # hit_cull above this bypasses peak protection
        camera_fps:                 float = 30.0, # actual inference/camera FPS
        apple_speed:                float = 1.0,  # apples per second on conveyor
    ) -> None:
        assert orientation in ORIENTATIONS, \
            f"orientation must be one of {ORIENTATIONS}, got '{orientation}'"

        self._n_lanes                  = n_lanes
        self._ori                      = orientation
        self._exit_frac                = exit_frac
        self._band_half_frac           = band_half_frac
        self._entry_frac               = entry_frac
        self._max_lost                 = max_lost_frames
        self._max_recover              = max_recover_dist
        self._min_count_dist_frac      = min_count_dist_frac
        self._count_mem                = count_memory_frames
        self._count_merge_frames       = count_merge_frames
        self._cull_weight              = cull_weight
        self._hit_threshold            = hit_threshold
        self._cull_ratio_thresh        = cull_ratio_threshold
        self._min_vote_conf            = min_vote_conf
        self._min_det_conf             = min_det_conf
        self._peak_conf_override       = peak_conf_override
        self._overwhelming_cull        = overwhelming_cull_threshold
        self._camera_fps               = max(camera_fps, 1.0)
        self._apple_speed              = max(apple_speed, 0.1)

        # ── Speed-adaptive min_frames ──────────────────────────────────────────
        # Budget: frames_per_apple = camera_fps / apple_speed
        # Gate must fire BEFORE the apple exits, so min_frames must be strictly
        # less than the budget.  We cap at the configured value (don't relax
        # below what the user asked for at slow speed) but ALWAYS clamp so we
        # can physically reach the threshold within the observation window.
        frames_budget = self._camera_fps / self._apple_speed
        # Use at most 60% of the budget to leave room for entry detection lag
        safe_max = max(3, int(frames_budget * 0.60))
        self._min_frames = min(min_frames, safe_max)
        if self._min_frames < min_frames:
            log.warning(
                "AppleTracker: min_frames clamped %d → %d  "
                "(camera_fps=%.1f  apple_speed=%.1f  budget=%.1f frames)",
                min_frames, self._min_frames,
                self._camera_fps, self._apple_speed, frames_budget,
            )
        else:
            log.info(
                "AppleTracker: min_frames=%d  camera_fps=%.1f  "
                "apple_speed=%.1f  budget=%.1f frames",
                self._min_frames, self._camera_fps,
                self._apple_speed, frames_budget,
            )

        self._history:     dict[int, dict] = defaultdict(self._new_history)
        self._lost:        dict[int, dict] = {}
        self._id_map:      dict[int, int]  = {}
        self._recent:      list[dict]      = []
        # Per-lane counters so that seq_id = (lane_count - 1) * n_lanes + lane,
        # matching the GT interleaving: Lane1→1,4,7…  Lane2→2,5,8…  Lane3→3,6,9…
        self._lane_counts: dict[int, int]  = {}   # lane (1-indexed) → apples counted so far
        self._frame_no:    int = 0

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
            "votes":        defaultdict(float),
            "peak_conf":    defaultdict(float),  # highest conf seen per class_id
            "hit_cull":     0,
            "frames_seen":  0,
            "first_travel": -1,
            "last_pos":     (0, 0),
            "last_frame":   0,
            "committed":    False,
            "lane":         1,
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

        # ── Layer 1 defence: drop low-confidence YOLO boxes entirely ──────────
        # ByteTrack can pass sub-threshold boxes through secondary association;
        # we discard anything below min_det_conf before it touches our state.
        filtered = [
            (t, c, x, f) for t, c, x, f in zip(track_ids, cls_ids, xyxys, confs)
            if f >= self._min_det_conf
        ]
        if not filtered:
            return active, graded
        track_ids, cls_ids, xyxys, confs = map(list, zip(*filtered))

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

        for tid, cls_id, xyxy, conf in zip(track_ids, cls_ids, xyxys, confs):
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

            # Weighted vote accumulation - ignore very-low-confidence frames
            # (background noise classified as Cull at conf < min_vote_conf would
            # otherwise accumulate hundreds of votes and drown out real grades).
            if conf >= self._min_vote_conf:
                weight = self._cull_weight if cls_id == 2 else 1.0
                hist["votes"][cls_id] += conf * weight
            # Always track peak confidence per class (even below min_vote_conf)
            if conf > hist["peak_conf"][cls_id]:
                hist["peak_conf"][cls_id] = conf
            if cls_id == 2 and conf > 0.6:
                hist["hit_cull"] += 1

            seen_ids.add(tid)
            seq_id = self._id_map.get(tid)

            # ── Counting gate ─────────────────────────────────────────────────
            #  Guard 1: in/past exit band  - apple at or beyond exit_pos - band_half
            #           (we use >= instead of exact band so a fast apple that skips
            #           through the narrow band in one frame still fires the gate)
            #  Guard 2: entry zone   - first detection was in the START of travel
            #  Guard 3: min frames   - not a phantom (adaptive, clamped to budget)
            in_band       = travel >= (exit_pos - band_half)   # fire once past the gate line
            entered_start = 0 <= hist["first_travel"] < entry_pos
            enough_frames = hist["frames_seen"] >= self._min_frames

            if in_band and entered_start and enough_frames and seq_id is None:
                # Proximity check - prevent double-fire on the SAME apple only.
                # Require same lane, very recent count, and close position so that
                # lag/bunching does not assign one seq_id to different apples.
                matched_seq = None
                for recent in self._recent:
                    age = self._frame_no - recent["frame"]
                    if age > self._count_merge_frames:
                        continue
                    if recent.get("lane") != lane:
                        continue
                    if self._dist((cx, cy), recent["pos"]) < min_cdist:
                        matched_seq = recent["seq_id"]
                        break

                if matched_seq is not None:
                    self._id_map[tid] = matched_seq
                    seq_id = matched_seq
                    log.debug("Track %d linked to existing #%d (lane=%d age=%d)",
                              tid, matched_seq, lane, age)
                else:
                    lane_cnt = self._lane_counts.get(lane, 0) + 1
                    self._lane_counts[lane] = lane_cnt
                    # Interleaved GT formula: Apple #1 on Lane1→1, Lane2→2, Lane3→3,
                    # Apple #2 on Lane1→4, Lane2→5, Lane3→6, etc.
                    seq_id = (lane_cnt - 1) * self._n_lanes + lane
                    self._id_map[tid] = seq_id
                    self._recent.append({
                        "pos":    (cx, cy),
                        "frame":  self._frame_no,
                        "seq_id": seq_id,
                        "lane":   lane,
                    })
                    log.debug("NEW apple #%d  lane=%d  lane_cnt=%d  ori=%s  travel=%d  first=%d  frames=%d",
                              seq_id, lane, lane_cnt, self._ori, travel, hist["first_travel"], hist["frames_seen"])

            # ── Grade commit ──────────────────────────────────────────────────
            # Normal path: apple has a seq_id, min frames seen, not yet committed.
            # Early-exit path: apple has passed 90% of travel axis (definitely
            # leaving FOV soon) - commit now with whatever votes we have, provided
            # we have at least half the min_frames budget (avoids phantom commits).
            past_exit      = travel >= int(axis_size * 0.90)
            half_min       = max(3, self._min_frames // 2)
            early_eligible = past_exit and hist["frames_seen"] >= half_min

            if (
                seq_id is not None
                and not hist["committed"]
                and (hist["frames_seen"] >= self._min_frames or early_eligible)
            ):
                total = sum(hist["votes"].values())
                if total > 0:
                    cull_ratio = hist["votes"].get(2, 0.0) / total

                    # Peak-confidence override: if any non-Cull class was ever
                    # seen at >= peak_conf_override, the apple is clearly NOT Cull.
                    # This prevents background noise from forcing Cull on a track
                    # that had a strong Fresh/Processing sighting.
                    max_non_cull_peak = max(
                        hist["peak_conf"].get(cls, 0.0)
                        for cls in range(len(self.CLASS_NAMES))
                        if cls != 2
                    ) if hist["peak_conf"] else 0.0
                    clearly_non_cull = max_non_cull_peak >= self._peak_conf_override

                    # Overwhelming cull: hit_cull so high it can only be a cull apple.
                    # Lifts peak_conf_override protection - a genuine Fresh/Processing apple
                    # will never accumulate this many high-conf Cull hits on a screw conveyor.
                    overwhelming_cull = hist["hit_cull"] >= self._overwhelming_cull

                    force_cull = (
                        cull_ratio >= self._cull_ratio_thresh
                        or hist["hit_cull"] >= self._hit_threshold
                    ) and (not clearly_non_cull or overwhelming_cull)

                    # Non-cull vote split for diagnostics
                    non_cull = {k: v for k, v in hist["votes"].items() if k != 2}
                    non_cull_str = "  ".join(
                        f"{self.CLASS_NAMES[k]}={v/total:.2f}"
                        for k, v in sorted(non_cull.items(), key=lambda x: -x[1])
                    ) if non_cull else "(none)"

                    log.info(
                        "Vote commit: apple=%s lane=%d  cull_ratio=%.2f  hit_cull=%d  "
                        "overwhelming=%s  peak_non_cull=%.2f  clearly_non_cull=%s  "
                        "force_cull=%s  | non_cull split: %s",
                        seq_id, lane, cull_ratio, hist["hit_cull"],
                        overwhelming_cull, max_non_cull_peak,
                        clearly_non_cull, force_cull, non_cull_str,
                    )

                    if force_cull:
                        best_cls, best_conf = 2, cull_ratio
                    else:
                        # Let all classes compete including Cull.
                        # Previously this code used max(non_cull) which stripped Cull out of
                        # the race entirely - so if Cull had 40% of votes (live display showed
                        # Cull) but force_cull=False, Fresh at 35% would win the committed
                        # grade. That caused "tracked as Cull on screen → committed as Fresh."
                        best_cls = max(hist["votes"], key=hist["votes"].get)
                        if best_cls == 2:
                            # Cull won the vote count even without hitting force_cull threshold
                            best_conf = cull_ratio
                        else:
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
                        track_id    = tid,
                    )
                    graded.append(rec)
                    log.info("Grade #%d  lane=%d  %s  conf=%.2f  frames=%d",
                             seq_id, lane, cls_name, best_conf, hist["frames_seen"])

            # Live display grade - mirrors commit force_cull logic so on-screen label
            # always matches what the committed grade will be.
            # Previously used raw max(votes) which showed "Fresh" even while Cull
            # evidence was accumulating, creating a confusing mismatch.
            if hist["votes"]:
                total_v      = sum(hist["votes"].values())
                cull_v       = hist["votes"].get(2, 0)
                cull_ratio_v = cull_v / total_v if total_v > 0 else 0.0
                hit_cull_v   = hist["hit_cull"]

                # Mirror the same guards used at commit time
                overwhelming_v    = hit_cull_v >= self._overwhelming_cull
                non_cull_peak_v   = max(
                    (hist["peak_conf"].get(k, 0.0) for k in [0, 1]), default=0.0
                )
                clearly_non_cull_v = non_cull_peak_v >= self._peak_conf_override
                force_cull_v = (
                    (cull_ratio_v >= self._cull_ratio_thresh
                     or hit_cull_v >= self._hit_threshold)
                    and (not clearly_non_cull_v or overwhelming_v)
                )

                if force_cull_v or overwhelming_v:
                    # Cull evidence is strong enough that commit will override to Cull -
                    # show Cull on screen now so the label matches the final grade.
                    disp_cls  = 2
                    disp_conf = cull_ratio_v
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
                "eligible":     entered_start,
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
        self._lane_counts.clear()
        self._frame_no = 0

    @property
    def total_counted(self) -> int:
        return sum(self._lane_counts.values())
