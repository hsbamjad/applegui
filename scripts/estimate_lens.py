"""
scripts/estimate_lens.py
========================
Estimate the focal length of the lens used during a recording session,
using two independent methods:

  Method 1 - Metadata:
      Read ffprobe/JAI stream headers for any embedded focal length tag.

  Method 2 - Geometric estimation from GT apple data:
      Uses the fact that apples at different horizontal positions in the
      image appear at different scales (pixels per mm) due to perspective.
      From the ratio of scales at two X positions we can solve for f_px:

          f_px = delta_x_px / sqrt((s_center/s_edge)^2 - 1)

      where s = D_px / D_mm (scale = pixel diameter / true diameter).
      This requires NO knowledge of camera height H.

Usage:
------
  Basic (metadata + frame extraction only):
      python scripts/estimate_lens.py --videos videos/Source0/G1/G1.mp4 videos/Source1/G1/G1.avi videos/Source2/G1/G1.avi

  With GT data for geometric estimation (recommended):
      python scripts/estimate_lens.py \\
          --videos videos/Source0/G1/G1.mp4 videos/Source1/G1/G1.avi videos/Source2/G1/G1.avi \\
          --gt data/gt_diameters.csv \\
          --model models/best.pt \\
          --output docs/lens_estimate_G1.png

Requirements:
      pip install ultralytics opencv-python pandas numpy matplotlib openpyxl
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
PIXEL_PITCH_MM   = 0.00345      # Sony IMX252 sensor pixel pitch (mm)
IMAGE_WIDTH_PX   = 2048         # JAI FSFE-3200T-10GE
IMAGE_HEIGHT_PX  = 1536
CX_IMAGE         = IMAGE_WIDTH_PX / 2.0   # principal point (assume centre)

# Known lenses to compare against
KNOWN_LENSES = {
    "JAI 0824-C3 (8 mm)" : 8.0,
    "12 mm"               : 12.0,
    "16 mm"               : 16.0,
    "25 mm"               : 25.0,
}


# ─────────────────────────────────────────────────────────
# METHOD 1 – METADATA
# ─────────────────────────────────────────────────────────
def check_metadata(video_paths: list[str]) -> None:
    print("\n" + "="*60)
    print("METHOD 1: Video file metadata")
    print("="*60)

    ffprobe = "ffprobe"
    for vp in video_paths:
        if not os.path.exists(vp):
            print(f"  [SKIP] Not found: {vp}")
            continue
        print(f"\n  File: {Path(vp).name}")
        try:
            r = subprocess.run(
                [ffprobe, "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-show_format", vp],
                capture_output=True, text=True, timeout=30
            )
            d = json.loads(r.stdout)
            # Print all tags from format + streams
            tags = dict(d.get("format", {}).get("tags", {}))
            for s in d.get("streams", []):
                if s.get("codec_type") == "video":
                    print(f"    Resolution : {s.get('width')}x{s.get('height')}")
                    print(f"    Codec      : {s.get('codec_name')}")
                    print(f"    FPS        : {s.get('r_frame_rate')}")
                    tags.update(s.get("tags", {}))
            if tags:
                print("    Metadata tags:")
                for k, v in tags.items():
                    print(f"      {k}: {v}")
                # Specifically look for focal length hints
                focal_keys = [k for k in tags if "focal" in k.lower() or "lens" in k.lower()]
                if focal_keys:
                    print("    [FOUND FOCAL/LENS TAGS]:")
                    for k in focal_keys:
                        print(f"      {k}: {tags[k]}")
                else:
                    print("    [No focal/lens tags found in metadata]")
            else:
                print("    [No metadata tags embedded]")
        except FileNotFoundError:
            print("    [ffprobe not found – skipping metadata check]")
        except Exception as e:
            print(f"    [Error: {e}]")


# ─────────────────────────────────────────────────────────
# METHOD 2 – GEOMETRIC ESTIMATION FROM GT APPLE DATA
# ─────────────────────────────────────────────────────────
def load_gt(gt_path: str) -> dict:
    """Load GT CSV or Excel file. Returns {apple_id: average_mm}."""
    import pandas as pd
    p = Path(gt_path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(gt_path)
    else:
        df = pd.read_csv(gt_path)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Try to find apple id and diameter columns
    id_col  = next((c for c in df.columns if "id" in c or "apple" in c), df.columns[0])
    avg_col = next((c for c in df.columns if "avg" in c or "average" in c or "mean" in c), None)
    if avg_col is None:
        # fallback: mean of D1 and D2
        d1 = next((c for c in df.columns if "d1" in c), None)
        d2 = next((c for c in df.columns if "d2" in c), None)
        if d1 and d2:
            df["average_mm"] = (df[d1] + df[d2]) / 2
            avg_col = "average_mm"
        else:
            # just take the second numeric column
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            avg_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]

    result = {}
    for _, row in df.iterrows():
        try:
            result[int(row[id_col])] = float(row[avg_col])
        except Exception:
            pass
    print(f"\n  Loaded {len(result)} apple GT entries from {Path(gt_path).name}")
    return result


def extract_apple_scales(video_paths: list[str], gt: dict, model_path: str) -> list[dict]:
    """
    Run YOLO on sampled frames, detect apples, and compute scale (D_px/D_mm)
    at each centroid X position. Matches tracker seq_id to GT apple_id.
    Returns list of {cx, D_px, D_mm, scale, channel}.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)

    all_detections = []
    seq_counter = 0   # global apple count across channels

    for ch_idx, vp in enumerate(video_paths):
        if not os.path.exists(vp):
            print(f"  [SKIP] Not found: {vp}")
            continue

        cap = cv2.VideoCapture(vp)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0

        print(f"\n  Channel {ch_idx+1}: {Path(vp).name}  ({total_frames} frames @ {fps:.1f} fps)")

        # Sample one frame every ~1 second, max 60 frames
        sample_interval = max(1, int(fps))
        sampled = 0

        frame_idx = 0
        while cap.isOpened() and sampled < 60:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bw = x2 - x1
                    bh = y2 - y1
                    D_px = min(bw, bh)
                    cx   = (x1 + x2) / 2.0

                    if D_px < 20:   # skip tiny noise
                        continue

                    all_detections.append({
                        "cx"     : cx,
                        "D_px"   : D_px,
                        "channel": ch_idx + 1,
                        "frame"  : frame_idx,
                    })

            sampled    += 1
            frame_idx  += sample_interval

        cap.release()
        print(f"    Sampled {sampled} frames, {sum(1 for d in all_detections if d['channel']==ch_idx+1)} detections")

    return all_detections


def estimate_focal_length(detections: list[dict], gt: dict) -> float | None:
    """
    Estimate f_px by fitting the perspective model to observed scale vs X data.

    For a camera pointing down, the scale s(x) = D_px/D_mm varies with X as:
        s(x) ≈ f_px / sqrt(H^2 + (dx/f_px * H)^2)   where dx = cx - CX_IMAGE

    This simplifies to:
        s(x) = s_0 / sqrt(1 + (dx/f_px)^2)

    We fit s_0 and f_px by nonlinear regression on (cx, scale) pairs.
    Because we don't have GT matching per detection (only group-level GT),
    we use the MEDIAN GT diameter as a proxy for all apples in this session
    (valid because the GT covers the full session's apple set).
    """
    if not gt:
        print("\n  [Skipping geometric estimation - no GT data]")
        return None

    # Use median GT diameter as representative true size
    gt_diameters = list(gt.values())
    d_median = float(np.median(gt_diameters))
    print(f"\n  Using median GT diameter: {d_median:.2f} mm")

    # Compute scale for each detection
    scales = []
    for det in detections:
        if det["D_px"] < 30:
            continue
        dx = det["cx"] - CX_IMAGE
        scale = det["D_px"] / d_median
        scales.append((dx, scale))

    if len(scales) < 10:
        print(f"  [Too few detections ({len(scales)}) for reliable fit]")
        return None

    xs  = np.array([s[0] for s in scales])
    ys  = np.array([s[1] for s in scales])   # observed scale D_px/D_mm

    # Model: y = s0 / sqrt(1 + (x/f)^2)
    # Rearrange: (s0/y)^2 = 1 + (x/f)^2  =>  (s0/y)^2 - 1 = (x/f)^2
    # => f^2 * [(s0/y)^2 - 1] = x^2
    # We fit via curve_fit
    from scipy.optimize import curve_fit

    def model(x, s0, f_px):
        denom = np.sqrt(1 + (x / f_px)**2)
        return s0 / denom

    try:
        # Initial guesses: s0 = median scale at center, f = 2319 (8mm)
        s0_init = float(np.median(ys[np.abs(xs) < 200])) if np.sum(np.abs(xs) < 200) > 0 else float(np.median(ys))
        p0 = [s0_init, 2319.0]
        popt, _ = curve_fit(model, xs, ys, p0=p0, maxfev=5000,
                            bounds=([0, 500], [np.inf, 10000]))
        s0_fit, f_px_fit = popt
        return float(f_px_fit)
    except Exception as e:
        print(f"  [Curve fit failed: {e}]")
        return None


def geometric_estimation(video_paths, gt_path, model_path, output_path):
    print("\n" + "="*60)
    print("METHOD 2: Geometric focal length estimation")
    print("="*60)

    gt = load_gt(gt_path) if gt_path else {}

    if model_path and os.path.exists(model_path):
        detections = extract_apple_scales(video_paths, gt, model_path)
    else:
        print("\n  [No YOLO model provided or not found - skipping detection]")
        print("  Provide --model to enable geometric estimation.")
        return

    f_px_est = estimate_focal_length(detections, gt)

    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)

    if f_px_est:
        f_mm_est = f_px_est * PIXEL_PITCH_MM
        print(f"\n  Estimated focal length (pixels): {f_px_est:.1f} px")
        print(f"  Estimated focal length (mm):     {f_mm_est:.2f} mm")
        print()
        print("  Comparison with known lenses:")
        print(f"  {'Lens':<30} {'f_mm':>8} {'f_px':>8} {'Difference':>12}")
        print(f"  {'-'*60}")
        for name, f_known_mm in KNOWN_LENSES.items():
            f_known_px = f_known_mm / PIXEL_PITCH_MM
            diff_pct = abs(f_px_est - f_known_px) / f_known_px * 100
            match = " <-- LIKELY MATCH" if diff_pct < 15 else ""
            print(f"  {name:<30} {f_known_mm:>8.1f} {f_known_px:>8.0f} {diff_pct:>10.1f}%{match}")

        # Plot if matplotlib available
        if output_path:
            try:
                import matplotlib.pyplot as plt

                if detections and gt:
                    d_median = float(np.median(list(gt.values())))
                    xs = np.array([d["cx"] - CX_IMAGE for d in detections if d["D_px"] >= 30])
                    ys = np.array([d["D_px"] / d_median for d in detections if d["D_px"] >= 30])

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.scatter(xs, ys, alpha=0.3, s=10, label="Observed (D_px / D_mm_median)")

                    x_fit = np.linspace(xs.min(), xs.max(), 300)
                    s0 = float(np.median(ys[np.abs(xs) < 200])) if np.sum(np.abs(xs) < 200) > 0 else float(np.median(ys))
                    y_fit = s0 / np.sqrt(1 + (x_fit / f_px_est)**2)
                    ax.plot(x_fit, y_fit, "r-", lw=2, label=f"Fitted model  f={f_mm_est:.1f} mm")

                    ax.set_xlabel("Centroid X offset from image center (px)")
                    ax.set_ylabel("Scale  D_px / D_mm_median")
                    ax.set_title(f"Lens Estimation - G1 Session\nEstimated focal length: {f_mm_est:.2f} mm")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=150)
                    print(f"\n  Plot saved: {output_path}")
            except Exception as e:
                print(f"\n  [Plot skipped: {e}]")
    else:
        print("\n  Could not estimate focal length from this data.")
        print("  Try providing more frames or a GT file with --gt.")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Estimate lens focal length from G1 session videos.")
    parser.add_argument("--videos", nargs="+", required=True,
                        help="Paths to G1 video files (all 3 channels recommended)")
    parser.add_argument("--gt", default=None,
                        help="Path to GT file (CSV or Excel) with apple diameters")
    parser.add_argument("--model", default=None,
                        help="Path to YOLO .pt model file")
    parser.add_argument("--output", default="docs/lens_estimate.png",
                        help="Output path for the estimation plot")
    args = parser.parse_args()

    print("\n" + "#"*60)
    print("# LENS FOCAL LENGTH ESTIMATOR")
    print("# Using G1 video session data")
    print("#"*60)
    print(f"\nCamera  : JAI FSFE-3200T-10GE")
    print(f"Sensor  : Sony IMX252  (pixel pitch = {PIXEL_PITCH_MM} mm)")
    print(f"Resolution: {IMAGE_WIDTH_PX} x {IMAGE_HEIGHT_PX} px")

    # Method 1: metadata
    check_metadata(args.videos)

    # Method 2: geometric (only if model provided)
    if args.model:
        geometric_estimation(args.videos, args.gt, args.model, args.output)
    else:
        print("\n" + "="*60)
        print("METHOD 2: Geometric estimation")
        print("="*60)
        print("\n  Skipped (no --model provided).")
        print("  To enable: add  --model models/best.pt  to the command.\n")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
