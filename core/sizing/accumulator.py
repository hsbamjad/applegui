"""
core/sizing/accumulator.py
===========================
Per-track feature accumulator for live apple sizing.

Integrates with the GUI inference loop:
  - update()  called every frame — extracts mask diameter features
  - commit()  called when a track is graded — runs view_fusion + ML predict
  - discard() called when a track disappears without grading (cleanup)

Performance note
----------------
all_diameters() from mask_diameter.py runs max_width() with 90 warpAffine
rotations — far too slow for real-time (>10ms per apple).
_fast_diameters() below uses angle_step=90 (2 rotations) and n_angles=6,
reducing cost to ~0.3ms per apple with no change to the feature structure.

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
from core.sizing.mask_diameter import (
    max_width, symmetry_diameter, ellipse_diameter,
    ellipse_minor_axis, area_diameter, quality_score,
)
from core.sizing.view_fusion import fuse_apple

logger = get_logger(__name__)


def _fast_diameters(mask: np.ndarray) -> dict:
    """
    Real-time version of all_diameters() with reduced angle resolution.

    max_width uses angle_step=90  → 2 warpAffine ops  (was 90)
    symmetry_diameter uses n_angles=6  → 6 iterations   (was 90)
    Other methods are already O(contour_length) — unchanged.

    Per-apple cost: ~0.3ms  vs ~12ms for full all_diameters().
    Feature key names match the offline pkl format expected by fuse_apple().
    """
    ell = ellipse_diameter(mask)
    return {
        "d_maxwidth": max_width(mask, angle_step=90),        # 2 rotations
        "d_symmetry": symmetry_diameter(mask, n_angles=6),   # 6 angles
        "d_ellipse":  ell,
        "d_area":     area_diameter(mask),
        "d_minor":    ellipse_minor_axis(mask),
        "quality":    quality_score(mask),
    }


# _FEATURE_COLS is loaded from bundle['feature_cols'] at runtime.
# The trained model is a sklearn Pipeline (scaler + Ridge) — feature order
# and count are whatever was used during training, stored in the bundle.


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
        self._min_frames    = min_frames
        self._tracks:    dict[int, list[dict]] = {}   # track_id → measurements
        self._committed: set[int]              = set() # already sized — skip update
        self._model       = None   # sklearn Pipeline (scaler + Ridge)
        self._feature_cols: list[str] = []   # loaded from bundle

        self._load_model(Path(model_path))

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Sizing model not found at %s — sizing disabled", path)
            return
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            self._model        = bundle.get("model")          # sklearn Pipeline
            self._feature_cols = bundle.get("feature_cols", [])
            logger.info(
                "Sizing model loaded from %s  |  %d features  |  "
                "MAE=%.2fmm  R²=%.3f",
                path,
                len(self._feature_cols),
                bundle.get("mae", float("nan")),
                bundle.get("r2",  float("nan")),
            )
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
        track_ids = boxes.id.cpu().numpy().astype(int)
        tid_to_idx = {int(tid): i for i, tid in enumerate(track_ids)}

        # ── ONE batched GPU→CPU transfer for all masks ────────────────────────
        # masks_data: (N, H_mask, W_mask) numpy array
        # Doing masks_data[i].cpu() inside the loop causes N separate GPU→CPU
        # round-trips which dominate cost at high apple counts.
        masks_data_np = yolo_result.masks.data.cpu().numpy()   # single transfer
        boxes_xyxy    = boxes.xyxy.cpu().numpy()               # (N, 4) orig coords

        mask_h, mask_w = masks_data_np.shape[1], masks_data_np.shape[2]
        orig_h, orig_w = yolo_result.orig_shape[:2]

        # Scale factors: original image coordinates → mask pixel coordinates
        sx = mask_w / orig_w
        sy = mask_h / orig_h

        # Log mask resolution once so we can verify scale
        if not hasattr(self, "_mask_shape_logged"):
            logger.debug(
                "Mask shape: %dx%d  orig: %dx%d  scale: %.3fx%.3f",
                mask_w, mask_h, orig_w, orig_h, sx, sy,
            )
            self._mask_shape_logged = True

        for t in active_tracks:
            tid = t["track_id"]

            if tid in self._committed:
                continue   # already sized — skip

            idx = tid_to_idx.get(tid)
            if idx is None:
                continue   # this track not in current YOLO result (lost frame)

            # ── Bounding box centre in original image coords ───────────────────
            x1, y1, x2, y2 = boxes_xyxy[idx]
            cx_orig = float((x1 + x2) / 2.0)

            # ── Crop mask to apple bbox ───────────────────────────────────────
            # Original masks may be 2048×1536 — warpAffine on the full frame
            # is prohibitively slow.  Crop first: typical apple bbox is ~100×100.
            pad  = 4
            mx1  = max(0,      int(x1 * sx) - pad)
            my1  = max(0,      int(y1 * sy) - pad)
            mx2  = min(mask_w, int(x2 * sx) + pad)
            my2  = min(mask_h, int(y2 * sy) + pad)

            mask_crop = (masks_data_np[idx, my1:my2, mx1:mx2] * 255).astype(np.uint8)

            if mask_crop.size == 0 or mask_crop.max() == 0:
                continue   # empty crop — apple not segmented

            # ── Diameter features on the small cropped mask ───────────────────
            diams = _fast_diameters(mask_crop)   # <0.1ms on 100×100 crop

            meas = {
                "cx_px":  cx_orig,
                # Key names must match fuse_apple() frame format
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

        # ── Build feature vector in training column order ──────────────────────
        X = np.array(
            [[float(features.get(c, 0.0)) for c in self._feature_cols]],
            dtype=np.float32,
        )

        # Pipeline handles scaling internally — just call predict
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
