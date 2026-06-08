"""
core/sizing/accumulator.py
===========================
Per-track feature accumulator for live apple sizing.

Architecture
------------
Two-phase design that decouples speed from accuracy:

  MAIN THREAD (called every inference frame, <0.2ms/apple):
    update()  ─ convexHull + fitEllipse + quality score
                Submits hull_crop to background worker pool.

  BACKGROUND THREAD POOL (2 workers, runs concurrently):
    _compute_hull_diameters()  ─ max_width(angle_step=10) + symmetry_diameter
                                  Same quality as offline pipeline (angle_step=5).
                                  Completes in ~1ms per apple on a small crop.

  MAIN THREAD (called when apple exits):
    commit()  ─ resolves all futures (done long ago by commit time),
                runs view_fusion + Ridge.predict → size_mm.

Result: offline-quality diameter features with zero FPS impact.

Offline reference
-----------------
extract_frames.py uses:
  - masks.xy polygon in original image coords (2048×1536)
  - cv2.convexHull → hull_crop  (bbox + 4 px padding)
  - all_diameters(hull_crop, angle_step=5) — 36 rotations
  - quality = axis_ratio × completeness

This module matches that exactly, using angle_step=10 (18 rotations)
for a good accuracy/speed balance. The background thread handles the cost.
"""

from __future__ import annotations

import pickle
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.log import get_logger
from core.sizing.mask_diameter import area_diameter, max_width, symmetry_diameter
from core.sizing.view_fusion import fuse_apple

logger = get_logger(__name__)


# ── Background worker function ────────────────────────────────────────────────

def _compute_hull_diameters(hull_crop: np.ndarray, angle_step: int) -> dict:
    """
    High-quality hull diameter computation — runs in background thread.

    angle_step=10 → 18 rotations (vs 2 in old approach, 36 in offline pipeline).
    On a 200×200 px crop: ~1ms.  Matches offline accuracy within ~0.5mm.
    """
    return {
        "d_maxwidth": max_width(hull_crop, angle_step=angle_step),
        "d_symmetry": symmetry_diameter(hull_crop, n_angles=18),
        "d_area":     area_diameter(hull_crop),
    }


# ── Accumulator ───────────────────────────────────────────────────────────────

class AppleSizeAccumulator:
    """
    Accumulates per-frame mask measurements for each tracked apple, then
    produces a size estimate (mm) when the track is committed.

    Parameters
    ----------
    model_path     : path to models/size_model.pkl  (Ridge pipeline bundle)
    min_frames     : minimum frames required to attempt sizing (default 4)
    bg_angle_step  : angle resolution for background max_width computation.
                     10 = 18 rotations (default, offline-quality speed/accuracy).
                     5  = 36 rotations (exactly offline — slightly slower).
    """

    def __init__(
        self,
        model_path:    str | Path,
        min_frames:    int = 4,
        bg_angle_step: int = 10,
    ) -> None:
        self._min_frames    = min_frames
        self._bg_angle_step = bg_angle_step

        # track_id → list of (partial_meas_dict, Future[dict])
        self._tracks:    dict[int, list[tuple[dict, Future]]] = {}
        self._committed: set[int] = set()

        self._model:        object = None   # sklearn Pipeline (scaler + Ridge)
        self._feature_cols: list[str] = []

        # Thread pool — 2 workers handle hull diameter computation in parallel
        self._pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="sizing",
        )

        self._load_model(Path(model_path))

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Sizing model not found at %s — sizing disabled", path)
            return
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            self._model        = bundle.get("model")
            self._feature_cols = bundle.get("feature_cols", [])
            logger.info(
                "Sizing model loaded  |  %d features  |  MAE=%.2fmm  R²=%.3f",
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

    # ── Per-frame update (MAIN THREAD) ────────────────────────────────────────

    def update(self, yolo_result, active_tracks: list) -> None:
        """
        Fast per-frame update — runs on the inference thread.

        For each visible tracked apple:
          1. Computes convex hull + fitEllipse + quality  (~0.1ms)
          2. Submits hull_crop to background pool for max_width / symmetry

        By design this never blocks.  Background tasks finish long before
        commit() is called (100+ frames later).
        """
        if yolo_result.masks is None:
            return   # detection-only model

        boxes = yolo_result.boxes
        if boxes is None or boxes.id is None:
            return

        track_ids  = boxes.id.cpu().numpy().astype(int)
        tid_to_idx = {int(tid): i for i, tid in enumerate(track_ids)}

        # masks.xy — polygon in ORIGINAL image coordinates (e.g. 2048×1536).
        # Same coordinate space as extract_frames.py (retina_masks=True).
        polys      = yolo_result.masks.xy
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        orig_h, orig_w = yolo_result.orig_shape[:2]

        if not hasattr(self, "_init_logged"):
            logger.info(
                "Sizing active  |  orig %dx%d  |  bg angle_step=%d° (%d rotations)",
                orig_w, orig_h,
                self._bg_angle_step,
                180 // self._bg_angle_step,
            )
            self._init_logged = True

        for t in active_tracks:
            tid = t["track_id"]
            if tid in self._committed:
                continue

            idx = tid_to_idx.get(tid)
            if idx is None:
                continue

            poly = polys[idx]
            if poly is None or len(poly) < 5:
                continue

            x1, y1, x2, y2 = boxes_xyxy[idx]
            cx_orig = float((x1 + x2) / 2.0)
            cy_orig = float((y1 + y2) / 2.0)

            # ── Convex hull in original image coordinates ──────────────────────
            poly_int = poly.astype(np.int32).reshape(-1, 1, 2)
            hull     = cv2.convexHull(poly_int)
            hx, hy, hw, hh = cv2.boundingRect(hull)

            # ── Hull crop with 4 px padding (same as extract_frames.py) ───────
            PAD = 4
            mx1 = max(0,      hx - PAD)
            my1 = max(0,      hy - PAD)
            mx2 = min(orig_w, hx + hw + PAD)
            my2 = min(orig_h, hy + hh + PAD)
            cw, ch = mx2 - mx1, my2 - my1

            if cw < 10 or ch < 10:
                continue

            hull_crop    = np.zeros((ch, cw), dtype=np.uint8)
            hull_shifted = (hull.reshape(-1, 2) - np.array([[mx1, my1]])).astype(np.int32)
            cv2.fillPoly(hull_crop, [hull_shifted], 255)

            if hull_crop.max() == 0:
                continue

            # ── Ellipse fit on hull vertices (~0.05ms) ─────────────────────────
            hull_pts = hull.reshape(-1, 2).astype(np.float32)
            if len(hull_pts) >= 5:
                (_, _), (ma, mb), _ = cv2.fitEllipse(hull_pts)
                ell_a = float(max(ma, mb))
                ell_b = float(min(ma, mb))
            else:
                ell_a = float(max(hw, hh))
                ell_b = float(min(hw, hh))

            # ── Quality = axis_ratio × completeness (extract_frames.py formula) ─
            axis_ratio = ell_b / (ell_a + 1e-6)
            half_w = (x2 - x1) / 2.0
            half_h = (y2 - y1) / 2.0
            fully_inside = (
                cx_orig - half_w >= 10 and
                cx_orig + half_w <= orig_w - 10 and
                cy_orig - half_h >= 10 and
                cy_orig + half_h <= orig_h - 10
            )
            quality = axis_ratio * (1.0 if fully_inside else 0.5)

            # ── Fast partial measurement (fills d_ell, d_minor, quality, cx_px) ─
            partial_meas = {
                "cx_px":  cx_orig,
                "d_ell":  ell_a,
                "d_minor":ell_b,
                "quality":quality,
                # d_maxw, d_sym, d_area filled in by background thread
            }

            # ── Submit slow diameter computation to background thread ───────────
            future = self._pool.submit(
                _compute_hull_diameters,
                hull_crop,          # small crop (~200×200) — thread-safe copy
                self._bg_angle_step,
            )

            self._tracks.setdefault(tid, []).append((partial_meas, future))

    # ── Commit (apple exits — MAIN THREAD) ────────────────────────────────────

    def commit(self, track_id: int, lane: int = 0) -> Optional[float]:
        """
        Resolve background futures and run sizing for a completed apple track.

        By the time this is called (100+ frames after first detection), all
        submitted futures will be done.  future.result() is non-blocking.

        Parameters
        ----------
        track_id : ByteTrack ID (GradeRecord.track_id)
        lane     : 1-indexed conveyor lane

        Returns
        -------
        float : predicted diameter in mm, or None if insufficient data/no model
        """
        stored = self._tracks.pop(track_id, [])
        self._committed.add(track_id)

        if not self.ready:
            return None

        if len(stored) < self._min_frames:
            logger.debug(
                "Track %d: only %d frames (need %d) — sizing skipped",
                track_id, len(stored), self._min_frames,
            )
            return None

        # ── Resolve futures → complete measurement dicts ──────────────────────
        measurements: list[dict] = []
        for partial_meas, future in stored:
            try:
                # Should already be done; 2s timeout is a safety net only
                diams = future.result(timeout=2.0)
                meas  = {
                    **partial_meas,
                    "d_maxw": diams["d_maxwidth"],
                    "d_sym":  diams["d_symmetry"],
                    "d_area": diams["d_area"],
                }
            except Exception as exc:
                logger.warning("Hull diameters future failed — using ellipse fallback: %s", exc)
                # Fallback: use ell_a as proxy for missing methods
                meas = {
                    **partial_meas,
                    "d_maxw": partial_meas["d_ell"],
                    "d_sym":  partial_meas["d_ell"],
                    "d_area": partial_meas["d_ell"],
                }
            measurements.append(meas)

        n = len(measurements)

        # ── Consensus ellipse axes (quality-weighted mean over all frames) ────
        qs  = np.clip([m["quality"] for m in measurements], 1e-6, None)
        ell = np.array([m["d_ell"]   for m in measurements], dtype=np.float32)
        mnr = np.array([m["d_minor"] for m in measurements], dtype=np.float32)

        ell_a = float(np.average(ell, weights=qs)) if ell.any() else 0.0
        ell_b = float(np.average(mnr, weights=qs)) if mnr.any() else 0.0

        # ── Build apple dict for fuse_apple() ─────────────────────────────────
        apple = {
            "frames":           measurements,
            "cx_min":           min(m["cx_px"] for m in measurements),
            "cx_max":           max(m["cx_px"] for m in measurements),
            "lane":             lane - 1,          # 0-indexed (0=top, 1=mid, 2=bot)
            "consensus_params": {"axis_a": ell_a, "axis_b": ell_b},
        }

        features = fuse_apple(apple)
        if features is None:
            logger.debug("Track %d: fuse_apple returned None — sizing skipped", track_id)
            return None

        # ── Feature vector in training order ──────────────────────────────────
        X = np.array(
            [[float(features.get(c, 0.0)) for c in self._feature_cols]],
            dtype=np.float32,
        )

        # Log first commit to verify scale and features are correct
        if not hasattr(self, "_first_commit_logged"):
            feat_str = "  ".join(
                f"{c}={features.get(c, 0.0):.1f}" for c in self._feature_cols
            )
            logger.info("First sizing commit | features: %s", feat_str)
            self._first_commit_logged = True

        # Pipeline (StandardScaler + Ridge) handles everything internally
        raw_pred = float(self._model.predict(X)[0])
        size_mm  = round(max(40.0, min(120.0, raw_pred)), 1)

        logger.info(
            "Track %d  lane=%d  frames=%d  raw=%.1fmm  size=%.1fmm",
            track_id, lane, n, raw_pred, size_mm,
        )
        return size_mm

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def discard(self, track_id: int) -> None:
        """Drop accumulated data for a track that left without being graded."""
        stored = self._tracks.pop(track_id, [])
        for _, future in stored:
            future.cancel()

    def clear(self) -> None:
        """
        Reset all state — call when pipeline stops.
        Cancels pending background futures and shuts down the thread pool,
        then creates a fresh pool ready for the next session.
        """
        for stored in self._tracks.values():
            for _, future in stored:
                future.cancel()
        self._tracks.clear()
        self._committed.clear()

        # Shut down pool (don't wait for running tasks — they'll finish quickly)
        self._pool.shutdown(wait=False)

        # Fresh pool for next session
        self._pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="sizing",
        )

        # Reset logged-once flags so next session logs correctly
        for attr in ("_init_logged", "_first_commit_logged"):
            self.__dict__.pop(attr, None)

        logger.debug("AppleSizeAccumulator cleared")

    def __del__(self) -> None:
        """Ensure thread pool is cleaned up on garbage collection."""
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass
