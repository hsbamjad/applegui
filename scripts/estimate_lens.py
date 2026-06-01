"""
scripts/estimate_lens.py
========================
Estimate the focal length of the lens used during a recording session,
using two independent methods:

  Method 1 - Metadata:
      Read ffprobe/JAI stream headers for any embedded focal length tag.
      Install ffprobe: https://ffmpeg.org/download.html (add to PATH)

  Method 2 - Geometric estimation from GT apple data:
      Uses the fact that apples at different horizontal positions in the
      image appear at different scales (pixels per mm) due to perspective.
      From the ratio of scales at two X positions we can solve for f_px:

          f_px = delta_x_px / sqrt((s_center/s_edge)^2 - 1)

      where s = D_px / D_mm (scale = pixel diameter / true diameter).
      This requires NO knowledge of camera height H.

Usage:
------
  Metadata only (no YOLO needed, ~5 seconds):
      python scripts/estimate_lens.py --videos videos/Source0/G1/G1.mp4 videos/Source1/G1/G1.avi videos/Source2/G1/G1.avi

  With GT data for geometric estimation (recommended):
      python scripts/estimate_lens.py \
          --videos videos/Source0/G1/G1.mp4 videos/Source1/G1/G1.avi videos/Source2/G1/G1.avi \
          --gt data/gt.xlsx \
          --model models/best.pt \
          --output docs/lens_estimate_G1.png

Requirements:
      pip install ultralytics opencv-python pandas numpy matplotlib openpyxl scipy
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
PIXEL_PITCH_MM  = 0.00345   # Sony IMX252 pixel pitch (mm)
IMAGE_WIDTH_PX  = 2048      # JAI FSFE-3200T-10GE
IMAGE_HEIGHT_PX = 1536
CX_IMAGE        = IMAGE_WIDTH_PX / 2.0

KNOWN_LENSES = {
    "JAI 0824-C3 (8 mm)": 8.0,
    "12 mm":              12.0,
    "16 mm":              16.0,
    "25 mm":              25.0,
}


# -------------------------------------------------------
# METHOD 1 - METADATA
# -------------------------------------------------------
def check_metadata(video_paths):
    print("\n" + "="*60)
    print("METHOD 1: Video file metadata")
    print("="*60)

    for vp in video_paths:
        if not os.path.exists(vp):
            print(f"  [SKIP] Not found: {vp}")
            continue
        print(f"\n  File: {Path(vp).name}")
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-show_format", vp],
                capture_output=True, text=True, timeout=30
            )
            d = json.loads(r.stdout)
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
                focal_keys = [k for k in tags if "focal" in k.lower() or "lens" in k.lower()]
                if focal_keys:
                    print("    *** FOCAL/LENS TAGS FOUND ***")
                    for k in focal_keys:
                        print(f"      {k}: {tags[k]}")
                else:
                    print("    [No focal/lens tags found]")
            else:
                print("    [No metadata tags embedded]")
        except FileNotFoundError:
            print("    [ffprobe not found]")
            print("    Install: https://ffmpeg.org/download.html  then add to PATH")
        except Exception as e:
            print(f"    [Error: {e}]")


# -------------------------------------------------------
# METHOD 2 - GEOMETRIC ESTIMATION
# -------------------------------------------------------
def load_gt(gt_path):
    """
    Load GT CSV or Excel. Returns {apple_id: average_mm}.

    Handles the G1 Excel format:
      Col A: Group label (G1) - merged, NaN in most rows
      Col B: Apple number (1, 2, 3 ...)
      Col C: D1 (mm)
      Col D: D2 (mm)
      Col E: Surface Class
      Col F: Average diameter (mm)
    """
    import pandas as pd

    print(f"\n  Reading GT file: {gt_path}")
    p = Path(gt_path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(gt_path, header=0)
    else:
        df = pd.read_csv(gt_path, header=0)

    # Debug: show raw structure
    print(f"  Raw columns : {list(df.columns)}")
    print(f"  Shape       : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  First 3 rows:")
    print(df.head(3).to_string())
    print()

    # Normalise column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Drop fully-empty rows (merged cell artifacts from Excel)
    df = df.dropna(how="all")

    # ---------- Find the AVERAGE DIAMETER column ----------
    avg_col = next(
        (c for c in df.columns
         if any(k in c for k in ("avg", "average", "mean", "diam"))),
        None
    )
    if avg_col is None:
        # Try to compute from D1 + D2
        d1c = next((c for c in df.columns if c.startswith("d1")), None)
        d2c = next((c for c in df.columns if c.startswith("d2")), None)
        if d1c and d2c:
            df["_avg"] = (pd.to_numeric(df[d1c], errors="coerce") +
                          pd.to_numeric(df[d2c], errors="coerce")) / 2
            avg_col = "_avg"
            print(f"  Computed average from columns '{d1c}' + '{d2c}'")
        else:
            # Last resort: rightmost numeric column
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if num_cols:
                avg_col = num_cols[-1]
                print(f"  Warning: using '{avg_col}' as diameter column (last numeric col)")
            else:
                print("  ERROR: cannot find a diameter column. Check GT file format.")
                return {}

    # ---------- Find the APPLE ID column ----------
    id_col = next(
        (c for c in df.columns
         if any(k in c for k in ("apple_id", "_id", "apple", "no.", "num", "seq", "id"))),
        None
    )
    if id_col is None:
        # Pick the first column that looks like sequential positive integers 1..N
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) >= 3 and vals.min() >= 1 and vals.max() <= 2000:
                id_col = c
                break
    if id_col is None:
        df["_apple_id"] = range(1, len(df) + 1)
        id_col = "_apple_id"
        print("  Warning: no ID column found, using row index as apple_id")

    print(f"  Using ID column       : '{id_col}'")
    print(f"  Using diameter column : '{avg_col}'")

    result = {}
    for _, row in df.iterrows():
        try:
            apple_id = int(float(row[id_col]))
            diameter = float(row[avg_col])
            if apple_id > 0 and 20.0 < diameter < 200.0:
                result[apple_id] = diameter
        except Exception:
            pass

    print(f"  Loaded {len(result)} valid apple GT entries")
    if result:
        vals = list(result.values())
        print(f"  Diameter range : {min(vals):.1f} - {max(vals):.1f} mm  "
              f"(median {float(np.median(vals)):.1f} mm)")
    return result


def extract_detections(video_paths, model_path):
    """Run YOLO on sampled frames. Returns list of {cx, D_px, channel}."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    detections = []

    for ch_idx, vp in enumerate(video_paths):
        if not os.path.exists(vp):
            print(f"  [SKIP] Not found: {vp}")
            continue
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"\n  Channel {ch_idx+1}: {Path(vp).name}  "
              f"({total} frames @ {fps:.1f} fps)")

        step    = max(1, int(fps))   # 1 frame per second
        sampled = 0
        n_before = len(detections)

        idx = 0
        while cap.isOpened() and sampled < 60:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            for r in model(frame, verbose=False):
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    D_px = min(x2 - x1, y2 - y1)
                    cx   = (x1 + x2) / 2.0
                    if D_px >= 20:
                        detections.append({"cx": cx, "D_px": D_px, "ch": ch_idx+1})
            sampled += 1
            idx     += step

        cap.release()
        print(f"    Sampled {sampled} frames, "
              f"{len(detections) - n_before} detections")

    return detections


def estimate_focal_length(detections, gt):
    """
    Fit perspective model:  scale(dx) = s0 / sqrt(1 + (dx/f_px)^2)
    where scale = D_px / D_mm_median  and  dx = cx - image_center.
    Returns estimated f_px, or None on failure.
    """
    from scipy.optimize import curve_fit

    if not gt:
        print("\n  [No GT data - cannot run geometric estimation]")
        return None

    d_median = float(np.median(list(gt.values())))
    print(f"\n  Using median GT diameter as proxy: {d_median:.2f} mm")

    pts = [(d["cx"] - CX_IMAGE, d["D_px"] / d_median)
           for d in detections if d["D_px"] >= 30]

    if len(pts) < 20:
        print(f"  [Too few usable detections: {len(pts)}. Need >= 20.]")
        return None

    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])

    def model(x, s0, f_px):
        return s0 / np.sqrt(1.0 + (x / f_px)**2)

    s0_init = float(np.median(ys[np.abs(xs) < 200])) if np.sum(np.abs(xs) < 200) > 0 \
              else float(np.median(ys))
    try:
        popt, _ = curve_fit(model, xs, ys,
                            p0=[s0_init, 2319.0],
                            bounds=([0, 500], [np.inf, 10000]),
                            maxfev=5000)
        return float(popt[1])
    except Exception as e:
        print(f"  [Curve fit failed: {e}]")
        return None


def geometric_estimation(video_paths, gt_path, model_path, output_path):
    print("\n" + "="*60)
    print("METHOD 2: Geometric focal length estimation")
    print("="*60)

    gt = load_gt(gt_path) if gt_path else {}

    if not (model_path and os.path.exists(model_path)):
        print("\n  [YOLO model not found - skipping detection]")
        return

    detections = extract_detections(video_paths, model_path)
    f_px_est   = estimate_focal_length(detections, gt)

    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)

    if f_px_est:
        f_mm_est = f_px_est * PIXEL_PITCH_MM
        print(f"\n  Estimated f (pixels) : {f_px_est:.1f} px")
        print(f"  Estimated f (mm)     : {f_mm_est:.2f} mm")
        print()
        print(f"  {'Lens':<30} {'f_mm':>8} {'f_px':>8} {'Error':>10}")
        print(f"  {'-'*58}")
        for name, f_known_mm in KNOWN_LENSES.items():
            f_known_px = f_known_mm / PIXEL_PITCH_MM
            diff_pct   = abs(f_px_est - f_known_px) / f_known_px * 100
            tag        = "  <-- LIKELY MATCH" if diff_pct < 15 else ""
            print(f"  {name:<30} {f_known_mm:>8.1f} {f_known_px:>8.0f} "
                  f"{diff_pct:>8.1f}%{tag}")

        # Save plot
        if output_path and gt:
            try:
                import matplotlib.pyplot as plt

                d_median = float(np.median(list(gt.values())))
                xs = np.array([d["cx"] - CX_IMAGE
                               for d in detections if d["D_px"] >= 30])
                ys = np.array([d["D_px"] / d_median
                               for d in detections if d["D_px"] >= 30])

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(xs, ys, alpha=0.25, s=8,
                           label="Observed  D_px / D_mm_median")

                x_fit = np.linspace(xs.min(), xs.max(), 300)
                s0 = float(np.median(ys[np.abs(xs) < 200])) \
                     if np.sum(np.abs(xs) < 200) > 0 else float(np.median(ys))
                y_fit = s0 / np.sqrt(1 + (x_fit / f_px_est)**2)
                ax.plot(x_fit, y_fit, "r-", lw=2,
                        label=f"Fitted model  f = {f_mm_est:.1f} mm")

                ax.set_xlabel("Centroid X offset from image center (px)")
                ax.set_ylabel("Scale  D_px / D_mm_median")
                ax.set_title(f"Lens Estimation - G1 Session\n"
                             f"Estimated focal length: {f_mm_est:.2f} mm")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path, dpi=150)
                print(f"\n  Plot saved: {output_path}")
            except Exception as e:
                print(f"\n  [Plot skipped: {e}]")
    else:
        print("\n  Could not estimate focal length.")
        print("  Check that the GT file loaded correctly (see debug output above).")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Estimate lens focal length from G1 session videos.")
    parser.add_argument("--videos", nargs="+", required=True,
                        help="Paths to video files (all 3 channels)")
    parser.add_argument("--gt", default=None,
                        help="GT Excel/CSV file with apple diameters")
    parser.add_argument("--model", default=None,
                        help="YOLO .pt model path")
    parser.add_argument("--output", default="docs/lens_estimate.png",
                        help="Output plot path")
    args = parser.parse_args()

    print("\n" + "#"*60)
    print("# LENS FOCAL LENGTH ESTIMATOR")
    print("#"*60)
    print(f"\nCamera    : JAI FSFE-3200T-10GE")
    print(f"Sensor    : Sony IMX252  (pitch = {PIXEL_PITCH_MM} mm)")
    print(f"Resolution: {IMAGE_WIDTH_PX} x {IMAGE_HEIGHT_PX} px")

    check_metadata(args.videos)

    if args.model:
        geometric_estimation(args.videos, args.gt, args.model, args.output)
    else:
        print("\n" + "="*60)
        print("METHOD 2: Geometric estimation  [SKIPPED - no --model given]")
        print("="*60)
        print("\n  Re-run with --model models/best.pt to enable.\n")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
