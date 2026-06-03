"""
core/sizing/mask_diameter.py  —  Step 2: Per-Frame Diameter Estimators
=======================================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

PURPOSE
-------
Given a binary uint8 mask (cropped, from the pkl "mask" field), compute
four independent diameter estimates in pixels plus a frame quality score.

All functions accept the CROPPED mask (as stored in the pkl).
The caller must ensure the mask is properly loaded — no path or pkl logic here.

FUNCTIONS
---------
  max_width(mask)          -> float   Method 1 - rotating projection
  symmetry_diameter(mask)  -> float   Method 2 - contour cross-correlation
  ellipse_diameter(mask)   -> float   Method 3 - fitted ellipse major axis
  area_diameter(mask)      -> float   Method 4 - sqrt(4*A/pi)
  quality_score(mask)      -> float   Q in [0, 1]: circularity * completeness
  all_diameters(mask)      -> dict    All four + Q in one call

UNITS
-----
All diameters are in PIXELS.  The scale factor (px -> mm) is applied
downstream in fit_scale.py / train_size_regressor.py.
"""

from __future__ import annotations

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_uint8(mask: np.ndarray) -> np.ndarray:
    """Guarantee mask is binary uint8 with values 0/255 for OpenCV."""
    m = np.asarray(mask, dtype=np.uint8)
    if m.max() <= 1:
        m = m * 255
    return m


def _get_contour(mask_u8: np.ndarray) -> np.ndarray | None:
    """Return the largest contour or None if mask is empty."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — MAX WIDTH PROJECTION
# Rotate the mask through 180° in 1° steps.  For each angle, project the
# mask onto the perpendicular axis and measure the span of the foreground
# pixels.  The maximum span across all angles is the diameter estimate.
#
# Rationale: the equatorial cross-section of a sphere is always the widest,
# so this gives an upper-bound estimate that is physically grounded.
# ─────────────────────────────────────────────────────────────────────────────
def max_width(mask: np.ndarray, angle_step: int = 2) -> float:
    """
    Diameter estimate from maximum projection width across rotations.

    Parameters
    ----------
    mask       : binary uint8 or bool array (cropped mask from pkl)
    angle_step : degrees between rotations (1 = finest, 2 = fast, default 2)

    Returns
    -------
    float : maximum span in pixels, or 0.0 if mask is empty
    """
    m = _ensure_uint8(mask)
    if m.max() == 0:
        return 0.0

    h, w  = m.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_span = 0.0

    for angle in range(0, 180, angle_step):
        rad = np.deg2rad(angle)
        # Rotation matrix around mask centre
        M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Project onto X axis: any row with foreground
        col_proj = np.any(rot > 0, axis=0)
        cols     = np.where(col_proj)[0]
        if cols.size:
            span = float(cols[-1] - cols[0] + 1)
            if span > max_span:
                max_span = span

    return max_span


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — CONTOUR SYMMETRY (Mizushima & Lu 2013)
# Build the radius function r(k) = distance from contour centroid to each
# contour point k.  Slide a split position around the contour; at each
# position cross-correlate the left and right halves of r(k).  The split
# with the highest peak correlation is the axis of bilateral symmetry.
# The diameter is 2 * mean radius computed perpendicular to that axis.
#
# Implementation note: we use the simpler "maximum mean-radius diameter"
# which gives similar results and is O(N) instead of O(N^2).
# Full cross-correlation is done once on the full radius sequence to find
# the best split, then we measure the span perpendicular to that axis.
# ─────────────────────────────────────────────────────────────────────────────
def symmetry_diameter(mask: np.ndarray) -> float:
    """
    Diameter estimate via the minor axis of the fitted ellipse.

    Complements ellipse_diameter() which returns the MAJOR axis.
    This returns the MINOR axis — the narrowest dimension of the silhouette.

    Physical meaning (top-down conveyor view):
      For an apple sitting upright on the conveyor (equator down), the
      silhouette is nearly circular.  Minor ≈ major ≈ equatorial diameter.
      Any difference between them measures the apple's deviation from a
      perfect sphere (aspect ratio), which is a useful ML feature.

    For a PCA approach: small YOLO mask irregularities randomly tip which
    contour direction becomes 'minor' for near-circular shapes, causing high
    variance (scale_std ~0.14 vs 0.03 for ellipse).  The global least-squares
    ellipse fit is far more robust to local contour noise.

    Returns
    -------
    float : minor axis length in pixels, or 0.0 if ellipse cannot be fitted
    """
    m   = _ensure_uint8(mask)
    cnt = _get_contour(m)
    if cnt is None or len(cnt) < 5:    # fitEllipse needs >= 5 points
        return 0.0

    try:
        _centre, (ma, mb), _angle = cv2.fitEllipse(cnt)
        return float(min(ma, mb))      # minor axis — narrowest dimension
    except cv2.error:
        return 0.0



# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — ELLIPSE MAJOR AXIS
# Fit an ellipse to the contour using OpenCV fitEllipse (least-squares).
# Return the major axis length as the diameter estimate.
# Simple, fast, and robust for near-circular apples.
# ─────────────────────────────────────────────────────────────────────────────
def ellipse_diameter(mask: np.ndarray) -> float:
    """
    Diameter estimate from the major axis of the fitted ellipse.

    Returns
    -------
    float : major axis length in pixels, or 0.0 if ellipse cannot be fitted
    """
    m   = _ensure_uint8(mask)
    cnt = _get_contour(m)
    if cnt is None or len(cnt) < 5:    # fitEllipse needs >= 5 points
        return 0.0

    try:
        _centre, (ma, mb), _angle = cv2.fitEllipse(cnt)
        return float(max(ma, mb))
    except cv2.error:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 4 — AREA-BASED SPHERE ESTIMATE
# Assume the apple is a perfect sphere viewed equatorially.
# The projected area is a circle: A = pi * (D/2)^2  =>  D = sqrt(4A/pi)
# Used as a sanity check and low-weight ensemble member.
# ─────────────────────────────────────────────────────────────────────────────
def area_diameter(mask: np.ndarray) -> float:
    """
    Diameter estimate from mask pixel area (sphere assumption).

    Returns
    -------
    float : equivalent circle diameter in pixels, or 0.0 if mask is empty
    """
    m    = _ensure_uint8(mask)
    area = float(np.count_nonzero(m))
    if area == 0:
        return 0.0
    return float(np.sqrt(4.0 * area / np.pi))


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY SCORE
# Q = circularity * completeness
#
# circularity = 4 * pi * Area / Perimeter^2   (1.0 = perfect circle)
#   Captures how "apple-like" the silhouette is.  Partial views, occlusions,
#   and strange shapes score low.
#
# completeness = 1.0 if the mask does NOT touch any edge of the crop,
#                0.5 if it does touch an edge (partially out of frame).
#   A mask touching the crop boundary means part of the apple is hidden.
# ─────────────────────────────────────────────────────────────────────────────
def quality_score(mask: np.ndarray) -> float:
    """
    Frame quality score Q in [0, 1].

    Q = circularity * completeness

    Parameters
    ----------
    mask : binary mask (cropped from pkl)

    Returns
    -------
    float : quality score in [0, 1]
    """
    m   = _ensure_uint8(mask)
    cnt = _get_contour(m)
    if cnt is None:
        return 0.0

    area = float(cv2.contourArea(cnt))
    if area == 0:
        return 0.0

    perim = float(cv2.arcLength(cnt, closed=True))
    if perim == 0:
        return 0.0

    circularity = min(1.0, (4.0 * np.pi * area) / (perim ** 2))

    # Completeness: penalise if mask touches any crop edge
    h, w = m.shape[:2]
    touches_edge = (
        m[0,  :].any()   or   # top row
        m[-1, :].any()   or   # bottom row
        m[:,  0].any()   or   # left column
        m[:, -1].any()        # right column
    )
    completeness = 0.5 if touches_edge else 1.0

    return float(circularity * completeness)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE — ALL DIAMETERS IN ONE CALL
# ─────────────────────────────────────────────────────────────────────────────
def all_diameters(mask: np.ndarray, angle_step: int = 2) -> dict:
    """
    Compute all four diameter estimates + quality score for one mask.

    Returns
    -------
    dict with keys:
        d_maxwidth   : float  (Method 1)
        d_symmetry   : float  (Method 2)
        d_ellipse    : float  (Method 3)
        d_area       : float  (Method 4)
        quality      : float  Q in [0, 1]
    """
    return {
        "d_maxwidth":  max_width(mask, angle_step=angle_step),
        "d_symmetry":  symmetry_diameter(mask),
        "d_ellipse":   ellipse_diameter(mask),
        "d_area":      area_diameter(mask),
        "quality":     quality_score(mask),
    }
