"""
gui/workers/size_calibration.py
================================
Position-aware apple diameter calibration module.

Based on: Mizushima & Lu (2013), ASABE Transactions 56(3):813-827.

How it works
------------
The camera is mounted above the conveyor centre lane. An apple directly
below the lens appears larger in pixels than an apple on a side lane,
even if both are physically the same size. This is basic perspective
geometry.

We model the pixel count of a known-diameter object at image-x position x:

    N_px(x)  =  a*x^2  +  b*x  +  c           [quadratic fit]

From which the scale function (mm per pixel) at position x is:

    r(x)  =  d_mm  /  N_px(x)      [mm per pixel: true diameter / pixel diameter]

Because apples sit at different heights above the belt depending on their
size (a 90 mm apple's centre is 45 mm above the belt; a 60 mm apple's
centre is 30 mm above), we keep TWO calibration curves:
  - small group  (apples below median GT diameter)
  - large group  (apples above median GT diameter)

For a given apple we interpolate r(x) between the two curves based on the
apple's own pixel size relative to the two calibration groups.

Pixel-only mode
---------------
If calibration coefficients are not yet available (coefficients = null in
config.yaml), SizeCalibrator.is_ready() returns False and all r() calls
return None. The rest of the system still works; size_mm will simply be
logged as None until calibration is run.

Lens (confirmed): JAI 0824-C3  f=8 mm  pixel_pitch=3.45 um  TV distortion=-0.86 %
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class SizeCalibrator:
    """
    Position- and size-aware scale function for apple diameter estimation.

    Parameters
    ----------
    small_d_mm : float
        Representative diameter of the small calibration group (mm).
    small_coeffs : list[float] | None
        Quadratic coefficients [a, b, c] fitted for the small group.
        N_px(x) = a*x^2 + b*x + c  where x is centroid column in pixels.
    large_d_mm : float
        Representative diameter of the large calibration group (mm).
    large_coeffs : list[float] | None
        Quadratic coefficients [a, b, c] fitted for the large group.
    img_width : int
        Image width in pixels (used for sanity checks only).
    min_diameter_px : int
        Detections smaller than this are ignored (likely partial/noise).
    max_diameter_px : int
        Detections larger than this are clamped (unlikely but defensive).
    """

    def __init__(
        self,
        small_d_mm:      float,
        small_coeffs:    Optional[list],
        large_d_mm:      float,
        large_coeffs:    Optional[list],
        img_width:       int   = 2048,
        min_diameter_px: int   = 40,
        max_diameter_px: int   = 500,
    ) -> None:
        self._small_d  = float(small_d_mm)
        self._large_d  = float(large_d_mm)
        self._small_c  = np.array(small_coeffs, dtype=float) if small_coeffs else None
        self._large_c  = np.array(large_coeffs, dtype=float) if large_coeffs else None
        self._img_w    = img_width
        self._min_px   = min_diameter_px
        self._max_px   = max_diameter_px

        if self._small_c is not None and len(self._small_c) != 3:
            raise ValueError("small_coeffs must be a list of 3 floats [a, b, c]")
        if self._large_c is not None and len(self._large_c) != 3:
            raise ValueError("large_coeffs must be a list of 3 floats [a, b, c]")

        if self.is_ready():
            log.info(
                "SizeCalibrator ready: small=%.1f mm, large=%.1f mm",
                self._small_d, self._large_d,
            )
        else:
            log.info(
                "SizeCalibrator: coefficients not yet set - running in pixel-only mode"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Return True if calibration coefficients are loaded and usable."""
        return self._small_c is not None and self._large_c is not None

    def r(self, cx: float, D_px: float) -> Optional[float]:
        """
        Return scale factor r in mm/pixel at centroid column cx for an
        apple whose bounding-box min-side is D_px pixels.

        Returns None if:
        - calibration is not ready (coefficients missing)
        - D_px is outside sanity bounds [min_diameter_px, max_diameter_px]
        - the fitted curve evaluates to a non-positive value at cx
          (would indicate a bad calibration fit)
        """
        if not self.is_ready():
            return None

        if not (self._min_px <= D_px <= self._max_px):
            return None

        r_small = self._r_from_curve(cx, self._small_c, self._small_d)
        r_large = self._r_from_curve(cx, self._large_c, self._large_d)

        if r_small is None or r_large is None:
            return None

        # Interpolate between small and large calibration curves.
        # The two representative diameters give us two "expected" pixel
        # counts at this position; we interpolate based on where D_px falls.
        n_px_small = self._eval_curve(cx, self._small_c)
        n_px_large = self._eval_curve(cx, self._large_c)

        if n_px_small is None or n_px_large is None:
            return None

        if abs(n_px_large - n_px_small) < 1e-6:
            # Curves are identical - just use either
            return float(r_small)

        # Clamp t to [0, 1] so we don't extrapolate wildly
        t = (D_px - n_px_small) / (n_px_large - n_px_small)
        t = max(0.0, min(1.0, t))

        r_interp = r_small + t * (r_large - r_small)
        return float(r_interp)

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "SizeCalibrator":
        """
        Instantiate from the size_estimation section of config.yaml.

        Expected config structure:
            size_estimation:
              enabled: true
              min_diameter_px: 40
              max_diameter_px: 500
              calibration:
                small_ball_mm: 63.5
                small_coeffs: null          # or [a, b, c]
                large_ball_mm: 76.2
                large_coeffs: null          # or [a, b, c]
        """
        cal = cfg.get("calibration", {})
        return cls(
            small_d_mm      = cal.get("small_ball_mm", 63.5),
            small_coeffs    = cal.get("small_coeffs"),
            large_d_mm      = cal.get("large_ball_mm", 76.2),
            large_coeffs    = cal.get("large_coeffs"),
            img_width       = cfg.get("img_width", 2048),
            min_diameter_px = cfg.get("min_diameter_px", 40),
            max_diameter_px = cfg.get("max_diameter_px", 500),
        )

    @staticmethod
    def fit_from_data(
        x_positions:    list[float],
        pixel_counts:   list[float],
        true_diameter_mm: float,
    ) -> tuple[float, float, float]:
        """
        Fit a quadratic N_px(x) = a*x^2 + b*x + c to calibration data.

        Parameters
        ----------
        x_positions : list[float]
            Centroid column of each detection (pixels).
        pixel_counts : list[float]
            Bounding-box min-side (pixels) for each detection.
        true_diameter_mm : float
            Known true diameter of the calibration object (mm).

        Returns
        -------
        (a, b, c) : tuple[float, float, float]
            Quadratic coefficients. Pass these as small_coeffs / large_coeffs
            in config.yaml.
        """
        xs = np.array(x_positions, dtype=float)
        ys = np.array(pixel_counts, dtype=float)

        if len(xs) < 3:
            raise ValueError(
                f"Need at least 3 data points to fit a quadratic, got {len(xs)}"
            )

        coeffs = np.polyfit(xs, ys, deg=2)   # returns [a, b, c]
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

        # Sanity: c (value at x=0) should be positive and in plausible range
        if c <= 0:
            log.warning(
                "Quadratic fit gives c=%.3f (< 0) at x=0 for d=%.1f mm. "
                "Check calibration data quality.",
                c, true_diameter_mm,
            )

        log.info(
            "Quadratic fit for %.1f mm object: a=%.6f  b=%.4f  c=%.2f",
            true_diameter_mm, a, b, c,
        )
        return a, b, c

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _eval_curve(x: float, coeffs: np.ndarray) -> Optional[float]:
        """Evaluate quadratic at x. Returns None if result is non-positive."""
        a, b, c = coeffs
        val = a * x * x + b * x + c
        return float(val) if val > 0 else None

    def _r_from_curve(
        self, x: float, coeffs: np.ndarray, d_mm: float
    ) -> Optional[float]:
        """
        Compute r = d_mm / N_px(x)   [mm per pixel].

        N_px(x) is the fitted pixel diameter of a d_mm-wide object at column x.
        Simple ratio: real-world diameter / apparent pixel diameter = mm/pixel.
        Returns None if N_px(x) <= 0 (bad fit).
        """
        n_px = self._eval_curve(x, coeffs)
        if n_px is None:
            return None
        return float(d_mm / n_px)
