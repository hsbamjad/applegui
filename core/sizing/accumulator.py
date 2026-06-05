"""
core/sizing/accumulator.py
===========================
Per-track feature accumulator for live apple sizing.

Integrates with the GUI inference loop:
  - update()  called every frame — extracts mask diameter features
  - commit()  called when a track is graded — runs view_fusion + ML predict
  - discard() called when a track disappears without grading (cleanup)

Zero heavy computation on the critical path.  Per-frame cost: ~0.5ms for
mask_diameter on one apple.  Per-apple commit cost: <1ms.

ID convention
-------------
The accumulator is keyed by ByteTrack track_id (from result.boxes.id),
which is the same ID stored in GradeRecord.track_id after our tracker change.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from core.log import get_logger
from core.sizing.mask_diameter import all_diameters
from core.sizing.view_fusion import fuse_apple

logger = get_logger(__name__)

# Feature columns — must match what the Ridge model was trained on
# (same order as view_fusion.feature_matrix default cols)
_FEATURE_COLS = [
    "d_maxw_wmean", "d_sym_wmean", "d_ell_wmean", "d_area_wmean",
    "d_maxw_peak",  "d_sym_peak",  "d_ell_peak",  "d_area_peak",
    "ell_a",        "ell_b",
    "mean_Q",       "n_central",   "lane",
]


class AppleSizeAccumulator:
    """
    Accumulates per-frame mask measurements for each tracked apple, then
    produces a size estimate (mm) when the track is committed.

    Parameters
    ----------
    model_path  : path to models/size_model.pkl  (Ridge bundle)
    min_frames  : minimum frames required to attempt sizing (default 4)
    """

    def __init__(self, model_path: str | Path, min_frames: int = 4) -> None:
        self._min_frames   = min_frames
        self._tracks:    dict[int, list[dict]] = {}   # track_id → measurements
        self._committed: set[int]              = set() # already sized — skip update
        self._model  = None
        self._scaler = None

        self._load_model(Path(model_path))

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Sizing model not found at %s — sizing disabled", path)
            return
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            self._model  = bundle.get("model")
            self._scaler = bundle.get("scaler")
            logger.info("Sizing model loaded from %s", path)
        except Exception as exc:
            logger.error("Failed to load sizing model from %s: %s", path, exc)

    @property
    def ready(self) -> bool:
        """True if a model is loaded and sizing is possible."""
        return self._model is not None

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, yolo_result, active_tracks: list) -> None:
        """
        Extract mask diameter features for every visible tracked apple.

        Called once per inference frame AFTER tracker.update().

        Parameters
        ----------
        yolo_result   : ultralytics Results object (result[0] from model.track())
        active_tracks : list of track dicts returned by AppleTracker.update()
                        Each dict has 'track_id' (ByteTrack int ID).
        """
        if yolo_result.masks is None:
            return   # detection-only model — no masks available

        boxes = yolo_result.boxes
        if boxes is None or boxes.id is None:
            return   # tracking not initialised yet

        # Build index: ByteTrack_id → position in YOLO result
        track_ids  = boxes.id.cpu().numpy().astype(int)
        tid_to_idx = {int(tid): i for i, tid in enumerate(track_ids)}

        masks_data = yolo_result.masks.data   # tensor (N, H, W)
        boxes_xyxy = boxes.xyxy.cpu().numpy() # (N, 4) — original image coords

        for t in active_tracks:
            tid = t["track_id"]

            if tid in self._committed:
                continue   # already sized — don't waste time

            idx = tid_to_idx.get(tid)
            if idx is None:
                continue   # this track not in current YOLO result (lost frame)

            # ── Extract mask ──────────────────────────────────────────────────
            mask_np = (masks_data[idx].cpu().numpy() * 255).astype(np.uint8)

            # ── Bounding box centre (original image coords → cx for traversal)
            x1, y1, x2, y2 = boxes_xyxy[idx]
            cx_orig = float((x1 + x2) / 2.0)

            # ── Diameter features ─────────────────────────────────────────────
            diams = all_diameters(mask_np)   # ~0.3 ms per apple

            meas = {
                "cx_px":  cx_orig,
                # Rename to match fuse_apple()'s expected frame key names
                "d_maxw": diams["d_maxwidth"],
                "d_sym":  diams["d_symmetry"],
                "d_ell":  diams["d_ellipse"],
                "d_area": diams["d_area"],
                "d_minor":diams["d_minor"],
                "quality":diams["quality"],
            }

            self._tracks.setdefault(tid, []).append(meas)

    # ── Commit (apple exits) ──────────────────────────────────────────────────

    def commit(self, track_id: int, lane: int = 0) -> Optional[float]:
        """
        Run sizing for a completed apple track.

        Pops the accumulated measurements from memory, runs view_fusion and
        Ridge regression, and returns the predicted diameter in mm.

        Parameters
        ----------
        track_id : ByteTrack ID (GradeRecord.track_id)
        lane     : 1-indexed conveyor lane (passed to view_fusion for scale)

        Returns
        -------
        float : predicted diameter in mm, or None if insufficient data/no model
        """
        measurements = self._tracks.pop(track_id, [])
        self._committed.add(track_id)

        if not self.ready:
            return None

        n = len(measurements)
        if n < self._min_frames:
            logger.debug(
                "Track %d: only %d frames (need %d) — sizing skipped",
                track_id, n, self._min_frames,
            )
            return None

        # ── Build consensus ellipse axes (quality-weighted means) ─────────────
        qs  = np.array([m["quality"] for m in measurements], dtype=np.float32)
        qs  = np.clip(qs, 1e-6, None)
        ell = np.array([m["d_ell"]   for m in measurements], dtype=np.float32)
        mnr = np.array([m["d_minor"] for m in measurements], dtype=np.float32)

        ell_a = float(np.sum(qs * ell) / np.sum(qs)) if ell.any() else 0.0
        ell_b = float(np.sum(qs * mnr) / np.sum(qs)) if mnr.any() else 0.0

        # ── Build apple dict compatible with fuse_apple() ─────────────────────
        apple = {
            "frames":           measurements,
            "cx_min":           min(m["cx_px"] for m in measurements),
            "cx_max":           max(m["cx_px"] for m in measurements),
            "lane":             lane - 1,   # fuse_apple uses 0-indexed lane
            "consensus_params": {"axis_a": ell_a, "axis_b": ell_b},
        }

        features = fuse_apple(apple)
        if features is None:
            logger.debug("Track %d: fuse_apple returned None — sizing skipped", track_id)
            return None

        # ── Build feature vector in training column order ─────────────────────
        X = np.array(
            [[float(features.get(c, 0.0)) for c in _FEATURE_COLS]],
            dtype=np.float32,
        )

        if self._scaler is not None:
            X = self._scaler.transform(X)

        size_mm = float(self._model.predict(X)[0])
        size_mm = round(max(40.0, min(120.0, size_mm)), 1)  # sanity clamp

        logger.debug(
            "Track %d: lane=%d  frames=%d  size=%.1fmm",
            track_id, lane, n, size_mm,
        )
        return size_mm

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def discard(self, track_id: int) -> None:
        """Drop accumulated data for a track that left without being graded."""
        self._tracks.pop(track_id, None)

    def clear(self) -> None:
        """Reset all state (call when pipeline stops)."""
        self._tracks.clear()
        self._committed.clear()
        logger.debug("AppleSizeAccumulator cleared")
