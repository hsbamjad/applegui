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
from core.sizing.mask_diameter import max_width, symmetry_diameter, area_diameter
from core.sizing.view_fusion import fuse_apple

import cv2

logger = get_logger(__name__)


def _hull_diameters(hull_crop: np.ndarray) -> dict:
    """
    Fast diameter estimates on a convex-hull crop.

    Matches extract_frames.py which calls all_diameters(hull_crop, angle_step=5).
    We use angle_step=90 (2 rotations) and n_angles=6 for real-time speed.
    All diameter values are in the same pixel space as the offline PKL
    because hull_crop is built from masks.xy in ORIGINAL image coordinates.
    """
    return {
        "d_maxwidth": max_width(hull_crop, angle_step=90),        # 2 rotations
        "d_symmetry": symmetry_diameter(hull_crop, n_angles=6),   # 6 angles
        "d_area":     area_diameter(hull_crop),
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

        Matches the offline extract_frames.py pipeline exactly:
          - Uses masks.xy (polygon in ORIGINAL image coords, ~2048×1536)
          - Computes convex hull → renders small hull_crop at correct scale
          - Computes quality as axis_ratio × completeness (not perimeter circularity)
          - Uses fitEllipse on hull vertices for ell_a / ell_b

        This ensures diameter features (d_maxw, d_ell, d_area...) are in the
        same pixel space as the PKL training data — essential for correct mm
        prediction by the Ridge model.
        """
        if yolo_result.masks is None:
            return   # detection-only model — no masks

        boxes = yolo_result.boxes
        if boxes is None or boxes.id is None:
            return   # tracking not initialised yet

        track_ids  = boxes.id.cpu().numpy().astype(int)
        tid_to_idx = {int(tid): i for i, tid in enumerate(track_ids)}

        # masks.xy: list of (N_pts, 2) float arrays in ORIGINAL image coords
        # This is the same coordinate space as extract_frames.py (retina_masks=True)
        polys      = yolo_result.masks.xy
        boxes_xyxy = boxes.xyxy.cpu().numpy()   # (N, 4) original coords
        orig_h, orig_w = yolo_result.orig_shape[:2]

        if not hasattr(self, "_mask_shape_logged"):
            logger.info(
                "masks.xy mode  |  orig frame: %dx%d px  |  %d detections this frame",
                orig_w, orig_h, len(polys),
            )
            self._mask_shape_logged = True

        for t in active_tracks:
            tid = t["track_id"]
            if tid in self._committed:
                continue

            idx = tid_to_idx.get(tid)
            if idx is None:
                continue

            poly = polys[idx]
            if poly is None or len(poly) < 5:
                continue   # need ≥5 points for fitEllipse

            x1, y1, x2, y2 = boxes_xyxy[idx]
            cx_orig = float((x1 + x2) / 2.0)
            cy_orig = float((y1 + y2) / 2.0)

            # ── Convex hull in original image coordinates ──────────────────────
            poly_int = poly.astype(np.int32).reshape(-1, 1, 2)
            hull     = cv2.convexHull(poly_int)           # (K, 1, 2)
            hx, hy, hw, hh = cv2.boundingRect(hull)

            # ── Render hull into a small crop (bbox + 4 px pad) ──────────────
            PAD = 4
            mx1 = max(0,       hx - PAD)
            my1 = max(0,       hy - PAD)
            mx2 = min(orig_w,  hx + hw + PAD)
            my2 = min(orig_h,  hy + hh + PAD)
            cw, ch = mx2 - mx1, my2 - my1

            if cw < 10 or ch < 10:
                continue

            hull_crop = np.zeros((ch, cw), dtype=np.uint8)
            hull_shifted = (hull.reshape(-1, 2) - np.array([[mx1, my1]])).astype(np.int32)
            cv2.fillPoly(hull_crop, [hull_shifted], 255)

            if hull_crop.max() == 0:
                continue

            # ── Ellipse fit on hull vertices (same as extract_frames.py) ───────
            hull_pts = hull.reshape(-1, 2).astype(np.float32)
            if len(hull_pts) >= 5:
                (_, _), (ma, mb), _ = cv2.fitEllipse(hull_pts)
                ell_a = float(max(ma, mb))   # major axis in px (orig coords)
                ell_b = float(min(ma, mb))   # minor axis in px (orig coords)
            else:
                ell_a = float(max(hw, hh))
                ell_b = float(min(hw, hh))

            # ── Quality = axis_ratio × completeness (extract_frames.py formula) ──
            axis_ratio   = ell_b / (ell_a + 1e-6)
            half_w, half_h = (x2 - x1) / 2.0, (y2 - y1) / 2.0
            MARGIN = 10
            fully_inside = (
                cx_orig - half_w >= MARGIN and
                cx_orig + half_w <= orig_w - MARGIN and
                cy_orig - half_h >= MARGIN and
                cy_orig + half_h <= orig_h - MARGIN
            )
            quality = axis_ratio * (1.0 if fully_inside else 0.5)

            # ── Remaining diameter estimates on hull_crop ──────────────────────
            diams = _hull_diameters(hull_crop)

            meas = {
                "cx_px":  cx_orig,
                "d_maxw": diams["d_maxwidth"],
                "d_sym":  diams["d_symmetry"],
                "d_ell":  ell_a,               # from fitEllipse on hull
                "d_area": diams["d_area"],
                "d_minor":ell_b,               # from fitEllipse on hull
                "quality":quality,             # axis_ratio × completeness
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

        # Log feature values on first commit to verify scale is correct
        if not hasattr(self, "_feature_logged"):
            feat_str = "  ".join(
                f"{c}={features.get(c, 0.0):.1f}" for c in self._feature_cols
            )
            logger.info("First commit features: %s", feat_str)
            logger.info("First commit raw X: %s", X.tolist())
            self._feature_logged = True

        # Pipeline handles scaling internally — just call predict
        raw_pred = float(self._model.predict(X)[0])
        size_mm  = round(max(40.0, min(120.0, raw_pred)), 1)  # sanity clamp

        logger.info(
            "Track %d: lane=%d  frames=%d  raw=%.1fmm  size=%.1fmm",
            track_id, lane, n, raw_pred, size_mm,
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
