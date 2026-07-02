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
        processing_weight:          float = 1.0,  # multiplier on Processing class votes
        hit_threshold:              int   = 20,
        cull_ratio_threshold:       float = 0.65,
        proc_ratio_threshold:       float = 0.45, # if Processing votes >= this fraction, force Processing
        hit_proc_threshold:         int   = 99,   # high-conf Processing hits needed to force Processing
        fresh_peak_protect:         float = 0.65, # if Fresh peak conf >= this, block force_processing
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
        self._processing_weight        = processing_weight
        self._hit_threshold            = hit_threshold
        self._cull_ratio_thresh        = cull_ratio_threshold
        self._proc_ratio_thresh        = proc_ratio_threshold
        self._hit_proc_threshold       = hit_proc_threshold
        self._fresh_peak_protect       = fresh_peak_protect
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
        self._lane_counts:      dict[int, int] = {}   # lane (1-indexed) → apples counted so far
        self._committed_seq_ids: set[int]      = set() # seq_ids that have already fired a grade
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
            "hit_proc":     0,   # high-conf Processing detections (conf > 0.50)
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
            hist_i = self._history[tid]

            # ── ID-reuse guard ────────────────────────────────────────────────
            # ByteTrack recycles track IDs: if a committed apple's ID goes into
            # ByteTrack's buffer (track_buffer frames of no detection) and then
            # a NEW apple appears nearby, ByteTrack may re-assign the same ID.
            # Without this guard the committed=True flag from the first apple
            # permanently blocks grading of the second apple, causing every
            # subsequent-wave apple on that lane to go ungraded.
            #
            # Detection: the track is already committed AND the new detection is
            # far from the last known position (> max_recover_dist pixels).
            # Geometry guarantee: a committed apple's last_pos is always near
            # the EXIT zone (~x=1741 for LR). A new apple entering always
            # appears at the ENTRY zone (~x=0). The gap (~1741 px) always
            # exceeds max_recover_dist (160 px), so no time gate is needed.
            # NOTE: The old time gate (frames_absent >= max_lost=20) was
            # removed because consecutive apples on the same lane are only
            # ~10-15 frames apart, causing wave-3 resets to be silently skipped.
            if hist_i["committed"] and hist_i["frames_seen"] > 0:
                cx_i = int((xyxys[i][0] + xyxys[i][2]) / 2)
                cy_i = int((xyxys[i][1] + xyxys[i][3]) / 2)
                dist_from_last = self._dist((cx_i, cy_i), hist_i["last_pos"])
                if dist_from_last > self._max_recover:
                    log.debug(
                        "Track %d ID reuse detected: committed apple, "
                        "new detection %.0f px away → resetting history",
                        tid, dist_from_last,
                    )
                    # Remove stale id_map entry so this apple gets a fresh seq_id
                    self._id_map.pop(tid, None)
                    # Full history reset - this is now a different apple
                    self._history[tid] = self._new_history()
                    hist_i = self._history[tid]

            if hist_i["frames_seen"] != 0:
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
                if cls_id == 2:
                    weight = self._cull_weight
                elif cls_id == 1:
                    weight = self._processing_weight
                else:
                    weight = 1.0   # Fresh - no boost
                hist["votes"][cls_id] += conf * weight
            # Always track peak confidence per class (even below min_vote_conf)
            if conf > hist["peak_conf"][cls_id]:
                hist["peak_conf"][cls_id] = conf
            if cls_id == 2 and conf > 0.6:
                hist["hit_cull"] += 1
            if cls_id == 1 and conf > 0.50:   # Processing hit counter
                hist["hit_proc"] += 1

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
                    if matched_seq in self._committed_seq_ids:
                        # The matched seq_id is already a committed grade.
                        # This is the NEXT real apple on the same lane arriving
                        # at the same exit position - NOT a ghost of the previous
                        # one.  Give it a fresh seq_id instead of merging.
                        log.debug(
                            "Track %d: proximity match seq=%d already committed – "
                            "treating as new apple (same lane, age=%d)",
                            tid, matched_seq, age,
                        )
                        matched_seq = None  # fall through to new-apple branch

                if matched_seq is not None:
                    self._id_map[tid] = matched_seq
                    seq_id = matched_seq
                    hist["committed"] = True   # suppress ghost: this track is a duplicate of
                                               # an already-committed apple (same lane, close
                                               # position) — do not fire a second grade commit
                    log.debug("Track %d linked to existing #%d (lane=%d age=%d) – ghost suppressed",
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
                    # Lifts peak_conf_override protection AND independently forces Cull.
                    # Matches the live display logic (which already shows Cull on overwhelming).
                    # A genuine Fresh/Processing apple will never accumulate this many
                    # high-conf Cull hits on a screw conveyor.
                    overwhelming_cull = hist["hit_cull"] >= self._overwhelming_cull

                    force_cull = (
                        overwhelming_cull  # overwhelming alone is sufficient to force Cull
                        or (
                            (cull_ratio >= self._cull_ratio_thresh
                             or hist["hit_cull"] >= self._hit_threshold)
                            and (not clearly_non_cull or overwhelming_cull)
                        )
                    )

                    # ── Force Processing ──────────────────────────────────────
                    # Symmetric to force_cull: if Processing has a strong enough
                    # presence in the vote, force Processing over Fresh.
                    # Only triggers when force_cull is False (Cull takes priority).
                    #
                    # Guard: clearly_fresh - if Fresh ever peaked at >= fresh_peak_protect,
                    # the apple is clearly Fresh and force_processing is blocked.
                    # A genuine Processing apple will rarely show a strong Fresh peak;
                    # a genuine Fresh apple almost always will.
                    proc_ratio  = hist["votes"].get(1, 0.0) / total
                    fresh_peak  = hist["peak_conf"].get(0, 0.0)
                    clearly_fresh = fresh_peak >= self._fresh_peak_protect
                    force_processing = (
                        not force_cull
                        and not overwhelming_cull  # block force_proc when Cull signal is overwhelming
                        and not clearly_fresh
                        and (
                            proc_ratio >= self._proc_ratio_thresh
                            or hist["hit_proc"] >= self._hit_proc_threshold
                        )
                    )

                    # Non-cull vote split for diagnostics
                    non_cull = {k: v for k, v in hist["votes"].items() if k != 2}
                    non_cull_str = "  ".join(
                        f"{self.CLASS_NAMES[k]}={v/total:.2f}"
                        for k, v in sorted(non_cull.items(), key=lambda x: -x[1])
                    ) if non_cull else "(none)"

                    log.info(
                        "Vote commit: apple=%s lane=%d  cull_ratio=%.2f  proc_ratio=%.2f  "
                        "hit_cull=%d  hit_proc=%d  overwhelming=%s  peak_non_cull=%.2f  "
                        "fresh_peak=%.2f  clearly_fresh=%s  "
                        "clearly_non_cull=%s  force_cull=%s  force_proc=%s  | non_cull: %s",
                        seq_id, lane, cull_ratio, proc_ratio,
                        hist["hit_cull"], hist["hit_proc"],
                        overwhelming_cull, max_non_cull_peak,
                        fresh_peak, clearly_fresh,
                        clearly_non_cull, force_cull, force_processing, non_cull_str,
                    )

                    if force_cull:
                        best_cls, best_conf = 2, cull_ratio
                    elif force_processing:
                        best_cls, best_conf = 1, proc_ratio
                    else:
                        # Let all classes compete on raw weighted vote totals.
                        best_cls = max(hist["votes"], key=hist["votes"].get)
                        if best_cls == 2:
                            best_conf = cull_ratio
                        elif best_cls == 1:
                            best_conf = proc_ratio
                        else:
                            best_conf = hist["votes"][best_cls] / total

                    hist["committed"] = True
                    self._committed_seq_ids.add(seq_id)   # mark this grade as fired
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
                    disp_cls  = 2
                    disp_conf = cull_ratio_v
                else:
                    # Mirror force_processing + clearly_fresh guard for live display
                    proc_v        = hist["votes"].get(1, 0)
                    proc_ratio_v  = proc_v / total_v if total_v > 0 else 0.0
                    fresh_peak_v  = hist["peak_conf"].get(0, 0.0)
                    clearly_fresh_v = fresh_peak_v >= self._fresh_peak_protect
                    force_proc_v  = (
                        not clearly_fresh_v
                        and (
                            proc_ratio_v >= self._proc_ratio_thresh
                            or hist["hit_proc"] >= self._hit_proc_threshold
                        )
                    )
                    if force_proc_v:
                        disp_cls  = 1
                        disp_conf = proc_ratio_v
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
        self._committed_seq_ids.clear()
        self._frame_no = 0

    @property
    def total_counted(self) -> int:
        return sum(self._lane_counts.values())
