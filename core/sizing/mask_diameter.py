"""
core/sizing/mask_diameter.py  -  Step 2: Per-Frame Diameter Estimators
=======================================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

PURPOSE
-------
Given a binary uint8 mask (cropped, from the pkl "consensus_mask" field),
compute four independent diameter estimates in pixels plus a frame quality score.

FUNCTIONS
---------
  max_width(mask)          -> float   Method 1 - rotating projection
  symmetry_diameter(mask)  -> float   Method 2 - contour symmetry (Mizushima & Lu 2013)
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

from core.log import get_logger
logger = get_logger(__name__)


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
# METHOD 1 - MAX WIDTH PROJECTION
# Rotate the mask through 180° in steps. For each angle, project the
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

    h, w   = m.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_span = 0.0

    for angle in range(0, 180, angle_step):
        M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        col_proj = np.any(rot > 0, axis=0)
        cols     = np.where(col_proj)[0]
        if cols.size:
            span = float(cols[-1] - cols[0] + 1)
            if span > max_span:
                max_span = span

    return max_span


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 - CONTOUR SYMMETRY (Mizushima & Lu 2013)
#
# For each direction theta in [0, 180°):
#   From the mask centroid, project contour points onto direction theta.
#   d1 = max forward reach (boundary in direction theta)
#   d2 = max backward reach (boundary in direction theta+180°)
#   score(theta) = (1 - |d1-d2| / (d1+d2+eps)) * (d1+d2)
#                   ^-- symmetry factor (1=perfect) ^-- total width
#
# The direction maximising score is the equatorial axis:
#   most symmetric AND widest = physical definition of equatorial diameter.
#
# This formula matches Lu et al. (ASABE 2025) Eq. 5 for symmetry scoring,
# applied here without stem detection.
# ─────────────────────────────────────────────────────────────────────────────
def symmetry_diameter(mask: np.ndarray, n_angles: int = 90) -> float:
    """
    Diameter estimate via bilateral symmetry scoring (Mizushima & Lu 2013).

    Parameters
    ----------
    mask     : binary uint8 or bool array (cropped mask from pkl)
    n_angles : number of directions tested (90 = every 2 degrees)

    Returns
    -------
    float : symmetry-weighted equatorial diameter in pixels, or 0.0
    """
    m   = _ensure_uint8(mask)
    cnt = _get_contour(m)
    if cnt is None or len(cnt) < 5:
        return 0.0

    # Centroid from image moments
    mom = cv2.moments(m)
    if mom["m00"] == 0:
        return 0.0
    cx = mom["m10"] / mom["m00"]
    cy = mom["m01"] / mom["m00"]

    # Centre-relative contour coordinates
    pts = cnt.reshape(-1, 2).astype(np.float32)
    rx  = pts[:, 0] - cx
    ry  = pts[:, 1] - cy

    best_score = -1.0
    best_diam  =  0.0

    for i in range(n_angles):
        theta  = np.deg2rad(i * 180.0 / n_angles)
        proj   = rx * np.cos(theta) + ry * np.sin(theta)
        d1     = float(proj.max())    # boundary forward
        d2     = float(-proj.min())   # boundary backward (positive)
        total  = d1 + d2
        if total < 1.0:
            continue
        symmetry = 1.0 - abs(d1 - d2) / (total + 1e-6)
        score    = symmetry * total   # Lu et al. 2025, Eq. 5 form
        if score > best_score:
            best_score = score
            best_diam  = total

    return best_diam


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 - ELLIPSE MAJOR AXIS
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
    if cnt is None or len(cnt) < 5:
        return 0.0

    try:
        _centre, (ma, mb), _angle = cv2.fitEllipse(cnt)
        return float(max(ma, mb))
    except cv2.error:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3b - ELLIPSE MINOR AXIS (bonus feature for ML, not a diameter)
# The minor axis captures the apple's narrowest visible dimension.
# Used as a feature alongside the major axis; the ratio gives aspect ratio.
# ─────────────────────────────────────────────────────────────────────────────
def ellipse_minor_axis(mask: np.ndarray) -> float:
    """Minor axis of the fitted ellipse (useful as ML feature, not diameter)."""
    m   = _ensure_uint8(mask)
    cnt = _get_contour(m)
    if cnt is None or len(cnt) < 5:
        return 0.0
    try:
        _centre, (ma, mb), _angle = cv2.fitEllipse(cnt)
        return float(min(ma, mb))
    except cv2.error:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 4 - AREA-BASED SPHERE ESTIMATE
# Assume the apple is a perfect sphere viewed equatorially.
# A = pi * (D/2)^2  =>  D = sqrt(4A/pi)
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
# completeness = 1.0 if mask does NOT touch any crop edge, else 0.5
# ─────────────────────────────────────────────────────────────────────────────
def quality_score(mask: np.ndarray) -> float:
    """
    Frame quality score Q in [0, 1].

    Q = circularity * completeness
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

    h, w = m.shape[:2]
    touches_edge = (
        m[0,  :].any()  or
        m[-1, :].any()  or
        m[:,  0].any()  or
        m[:, -1].any()
    )
    completeness = 0.5 if touches_edge else 1.0

    return float(circularity * completeness)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE - ALL DIAMETERS IN ONE CALL
# ─────────────────────────────────────────────────────────────────────────────
def all_diameters(mask: np.ndarray, angle_step: int = 2) -> dict:
    """
    Compute all four diameter estimates + quality score for one mask.

    Returns
    -------
    dict with keys:
        d_maxwidth   : float  Method 1 - max projection width
        d_symmetry   : float  Method 2 - bilateral symmetry score (Mizushima & Lu 2013)
        d_ellipse    : float  Method 3 - ellipse major axis
        d_area       : float  Method 4 - area sphere estimate
        d_minor      : float  ellipse minor axis (ML feature)
        quality      : float  Q in [0, 1]
    """
    return {
        "d_maxwidth": max_width(mask, angle_step=angle_step),
        "d_symmetry": symmetry_diameter(mask),
        "d_ellipse":  ellipse_diameter(mask),
        "d_area":     area_diameter(mask),
        "d_minor":    ellipse_minor_axis(mask),
        "quality":    quality_score(mask),
    }
