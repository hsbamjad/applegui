"""
core/sizing/view_fusion.py  —  Step 3: Cross-Frame Fusion
==========================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

PURPOSE
-------
For each apple track (from a session pkl), fuse the per-frame diameter
estimates (d_maxw, d_sym, d_ell, d_area) across all frames into a compact
feature vector suitable for ML regression.

FUSION STRATEGY
---------------
For each method m in {maxw, sym, ell, area}:

  1. Filter to "central" frames only:
       cx in [cx_min + 20%*cx_range,  cx_max - 20%*cx_range]
     (same filter used in compute_consensus_ellipse — excludes partial-view
      entry/exit frames)

  2. Quality-weighted mean:
       D_wmean_m = sum(Q_i * D_m_i) / sum(Q_i)
     Low-quality frames (partial, occluded) contribute less.

  3. Peak (alpha-trimmed max):
       D_peak_m = mean of top 10% D values in central frames
     More stable than a raw max which can be a single lucky frame.

OUTPUT FEATURES (per apple)
---------------------------
  d_maxw_wmean, d_maxw_peak
  d_sym_wmean,  d_sym_peak
  d_ell_wmean,  d_ell_peak
  d_area_wmean, d_area_peak
  mean_Q        (mean quality of central frames)
  n_central     (number of central frames used)
  cx_range      (total traversal in pixels)
  lane          (0/1/2 — systematic scale offset from lens geometry)
  ell_a, ell_b  (consensus ellipse axes — from extract_frames.py)

These 15 features feed directly into the Step 4 ML regressor.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from core.log import get_logger
logger = get_logger(__name__)

# Central fraction: skip first/last 20% of traversal
CENTRAL_FRAC = 0.20

# Alpha-trim fraction for peak estimate (use top 10% of frames)
PEAK_FRAC = 0.10


def fuse_apple(apple: dict, central_frac: float = CENTRAL_FRAC,
               peak_frac: float = PEAK_FRAC) -> dict | None:
    """
    Fuse per-frame diameter estimates for a single apple track.

    Parameters
    ----------
    apple        : one element of pkl["apples"]
    central_frac : fraction of traversal to skip at each end (default 0.20)
    peak_frac    : top fraction of frames to average for 'peak' estimate

    Returns
    -------
    dict of fused features, or None if insufficient data
    """
    frames = apple.get("frames", [])
    if not frames:
        return None

    cx_min = apple.get("cx_min", min(f["cx_px"] for f in frames))
    cx_max = apple.get("cx_max", max(f["cx_px"] for f in frames))
    cx_range = cx_max - cx_min

    # ── Central frame filter ──────────────────────────────────────────────────
    if cx_range > 0:
        lo = cx_min + central_frac * cx_range
        hi = cx_max - central_frac * cx_range
        central = [f for f in frames if lo <= f["cx_px"] <= hi]
    else:
        central = frames

    # Need at least 5 central frames
    if len(central) < 5:
        central = frames   # fall back to all frames
    if not central:
        return None

    methods = ["d_maxw", "d_sym", "d_ell", "d_area"]
    features: dict[str, float] = {}

    qs = np.array([f.get("quality", 0.0) for f in central], dtype=np.float32)
    qs = np.clip(qs, 1e-6, None)   # avoid division by zero

    for m in methods:
        vals = np.array([f.get(m, 0.0) for f in central], dtype=np.float32)

        # Zero values = method failed on that frame; exclude from fusion
        valid = vals > 5.0
        if valid.sum() < 3:
            features[f"{m}_wmean"] = 0.0
            features[f"{m}_peak"]  = 0.0
            continue

        v = vals[valid]
        q = qs[valid]

        # Quality-weighted mean
        features[f"{m}_wmean"] = float(np.sum(q * v) / np.sum(q))

        # Alpha-trimmed peak: mean of top peak_frac values
        k = max(1, int(len(v) * peak_frac))
        top_k = np.sort(v)[-k:]
        features[f"{m}_peak"] = float(top_k.mean())

    # ── Aggregate quality and count ───────────────────────────────────────────
    features["mean_Q"]    = float(qs.mean())
    features["n_central"] = len(central)
    features["cx_range"]  = int(cx_range)

    # ── Lane and position (systematic scale offset) ───────────────────────────
    features["lane"]      = int(apple.get("lane", 0))
    features["pos"]       = int(apple.get("pos_in_lane", 0))

    # ── Consensus ellipse axes (from extract_frames.py) ───────────────────────
    cp = apple.get("consensus_params")
    features["ell_a"] = float(cp["axis_a"]) if cp else 0.0
    features["ell_b"] = float(cp["axis_b"]) if cp else 0.0

    # ── GT label (None if not available) ──────────────────────────────────────
    features["gt_mm"]      = apple.get("gt_mm")
    features["apple_idx"]  = apple.get("apple_idx", -1)
    features["session"]    = apple.get("session", "?")

    return features


def fuse_session(pkl_data: dict, **kwargs) -> list[dict]:
    """
    Fuse all apples in a session pkl.

    Parameters
    ----------
    pkl_data : loaded pkl dict (output of extract_frames.py)
    **kwargs : forwarded to fuse_apple (central_frac, peak_frac)

    Returns
    -------
    List of feature dicts, one per apple (sorted by apple_idx).
    Apples with insufficient data are excluded (logged as warning).
    """
    session = pkl_data.get("session", "?")
    apples  = sorted(pkl_data.get("apples", []), key=lambda a: a["apple_idx"])

    results = []
    for apple in apples:
        # Attach session name (not stored per-apple in pkl)
        apple["session"] = session
        feat = fuse_apple(apple, **kwargs)
        if feat is None:
            idx = apple.get("apple_idx", "?")
            logger.warning(f"Apple {idx} in {session} - no usable frames, skipped.")
            continue
        results.append(feat)

    return results


def feature_matrix(fused_list: list[dict],
                   feature_cols: list[str] | None = None) -> tuple:
    """
    Convert a list of fused feature dicts to numpy arrays.

    Parameters
    ----------
    fused_list   : output of fuse_session() or concatenated sessions
    feature_cols : list of feature keys to include.
                   If None, uses the default Step 2 feature set.

    Returns
    -------
    X : np.ndarray shape (N, F)  — feature matrix
    y : np.ndarray shape (N,)    — GT labels in mm (NaN where unavailable)
    meta : list of dicts         — apple_idx, session, lane, pos per row
    cols : list[str]             — column names matching X
    """
    if feature_cols is None:
        feature_cols = [
            # Per-method quality-weighted means (primary signals)
            "d_maxw_wmean", "d_sym_wmean", "d_ell_wmean", "d_area_wmean",
            # Per-method peaks (upper bound; useful for ML)
            "d_maxw_peak",  "d_sym_peak",  "d_ell_peak",  "d_area_peak",
            # Consensus ellipse (from central-60% median — already validated)
            "ell_a", "ell_b",
            # Quality and coverage
            "mean_Q", "n_central",
            # Spatial (systematic scale offset per lane)
            "lane",
        ]

    X_rows, y_vals, metas = [], [], []
    for r in fused_list:
        row = [float(r.get(c, 0.0)) for c in feature_cols]
        X_rows.append(row)
        gt = r.get("gt_mm")
        y_vals.append(float(gt) if gt is not None else float("nan"))
        metas.append({
            "apple_idx": r.get("apple_idx"),
            "session":   r.get("session"),
            "lane":      r.get("lane"),
            "pos":       r.get("pos"),
        })

    X    = np.array(X_rows, dtype=np.float32)
    y    = np.array(y_vals,  dtype=np.float64)
    return X, y, metas, feature_cols
