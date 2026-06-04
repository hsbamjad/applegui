"""
scripts/compare_methods_G1.py
=============================================================
Compare all 4 diameter methods on G1 using consensus masks.

Methods tested:
  1. Max Width Projection
  2. Contour Symmetry (Mizushima & Lu 2013)
  3. Ellipse Major Axis  (our current approach)
  4. Area Sphere Estimate

For each apple: run all_diameters(consensus_mask), convert px->mm
using a least-squares scale fit, report MAE/RMSE per method.

Usage:
    python scripts/compare_methods_G1.py
"""

import sys, os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.sizing.mask_diameter import all_diameters

PKL_PATH = r"S:\MSU_Research\ASABE AIM26\apple_gui\data\frame_features\G1.pkl"
OUT_DIR  = r"S:\MSU_Research\ASABE AIM26\apple_gui\data\method_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

DARK  = "#0d1117"
PANEL = "#161b22"
TEXT  = "#e6edf3"
MUTED = "#8b949e"
COLORS = {
    "d_maxwidth": "#58a6ff",
    "d_symmetry": "#3fb950",
    "d_ellipse":  "#d29922",
    "d_area":     "#f78166",
    "ensemble":   "#bc8cff",
}
METHOD_LABELS = {
    "d_maxwidth": "M1  Max Width Projection",
    "d_symmetry": "M2  Contour Symmetry (Mizushima & Lu 2013)",
    "d_ellipse":  "M3  Ellipse Major Axis  [current]",
    "d_area":     "M4  Area Sphere Estimate",
    "ensemble":   "M5  Simple Mean Ensemble",
}

# ── Load pkl ──────────────────────────────────────────────────────────────────
print(f"Loading {PKL_PATH} ...")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

apples = sorted(data["apples"], key=lambda a: a["apple_idx"])
print(f"  {len(apples)} apples loaded")

# ── Compute all diameters per apple ───────────────────────────────────────────
records = []
skipped = 0

for a in apples:
    gt = a.get("gt_mm")
    cm = a.get("consensus_mask")
    if gt is None or cm is None:
        skipped += 1
        continue

    # Run all 4 methods on the consensus mask
    d = all_diameters(cm, angle_step=2)   # ~1° resolution for max_width

    records.append({
        "apple_idx":  a["apple_idx"] + 1,
        "lane":       a["lane"],
        "pos":        a["pos_in_lane"],
        "gt_mm":      gt,
        "Q":          a["best_quality"],
        "d_maxwidth": d["d_maxwidth"],
        "d_symmetry": d["d_symmetry"],
        "d_ellipse":  d["d_ellipse"],
        "d_area":     d["d_area"],
    })

if skipped:
    print(f"  Warning: {skipped} apples skipped (missing mask or GT)")

methods = ["d_maxwidth", "d_symmetry", "d_ellipse", "d_area"]
gt_arr  = np.array([r["gt_mm"] for r in records])
n       = len(records)

# ── Fit scale per method (least-squares: sum(D_px * gt) / sum(D_px^2)) ────────
print("\n" + "="*72)
print(f"{'Method':<42} {'Scale':>8} {'MAE':>7} {'RMSE':>8} {'MaxErr':>8} {'R²':>6}")
print("="*72)

results = {}
for method in methods:
    d_px   = np.array([r[method] for r in records])
    valid  = d_px > 10
    if valid.sum() < 3:
        print(f"  {METHOD_LABELS[method]:<42}  INSUFFICIENT DATA")
        continue

    scale     = float(np.sum(d_px[valid] * gt_arr[valid]) / np.sum(d_px[valid] ** 2))
    d_mm      = d_px * scale
    errs      = d_mm[valid] - gt_arr[valid]
    mae       = float(np.mean(np.abs(errs)))
    rmse      = float(np.sqrt(np.mean(errs ** 2)))
    max_err   = float(np.max(np.abs(errs)))
    ss_res    = float(np.sum(errs ** 2))
    ss_tot    = float(np.sum((gt_arr[valid] - gt_arr[valid].mean()) ** 2))
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    results[method] = {"scale": scale, "d_mm": d_mm, "errs": errs[...],
                       "mae": mae, "rmse": rmse, "max_err": max_err, "r2": r2}

    star = " ◄ BEST" if mae == min(
        v["mae"] for v in results.values() if "mae" in v
    ) else ""
    print(f"  {METHOD_LABELS[method]:<42}  {scale:6.3f}  {mae:5.2f}mm  "
          f"{rmse:6.2f}mm  {max_err:6.2f}mm  {r2:5.3f}{star}")

# Simple ensemble: mean of all methods in mm
ens_mm_all = []
for r in records:
    vals_mm = []
    for method in methods:
        if method in results and results[method]["d_mm"] is not None:
            idx = [i for i, rec in enumerate(records) if rec == r][0]
            vals_mm.append(results[method]["d_mm"][idx])
    ens_mm_all.append(np.mean(vals_mm) if vals_mm else 0.0)

ens_mm  = np.array(ens_mm_all)
ens_err = ens_mm - gt_arr
ens_mae = float(np.mean(np.abs(ens_err)))
ens_rmse= float(np.sqrt(np.mean(ens_err ** 2)))
ens_max = float(np.max(np.abs(ens_err)))
ss_res  = float(np.sum(ens_err ** 2))
ss_tot  = float(np.sum((gt_arr - gt_arr.mean()) ** 2))
ens_r2  = 1 - ss_res / ss_tot

results["ensemble"] = {"d_mm": ens_mm, "errs": ens_err,
                       "mae": ens_mae, "rmse": ens_rmse,
                       "max_err": ens_max, "r2": ens_r2}

print(f"  {'--- Simple Mean Ensemble ---':<42}  {'N/A':>6}  {ens_mae:5.2f}mm  "
      f"{ens_rmse:6.2f}mm  {ens_max:6.2f}mm  {ens_r2:5.3f}")
print("="*72)

# ── Per-apple detail table ─────────────────────────────────────────────────────
print(f"\n{'#':>3}  {'GT':>6}  {'MaxW':>7}  {'Sym':>7}  {'Ell':>7}  {'Area':>7}  {'Ens':>7}")
print("-" * 58)
for i, r in enumerate(records):
    vals = {m: f"{results[m]['d_mm'][i]:6.1f}" for m in methods if m in results}
    ens  = f"{results['ensemble']['d_mm'][i]:6.1f}"
    print(f"#{r['apple_idx']:2d}  {r['gt_mm']:6.1f}  "
          f"{vals.get('d_maxwidth','   N/A')}  "
          f"{vals.get('d_symmetry','   N/A')}  "
          f"{vals.get('d_ellipse', '   N/A')}  "
          f"{vals.get('d_area',    '   N/A')}  "
          f"{ens}")

# ── FIGURE 1: MAE comparison bar chart ────────────────────────────────────────
all_method_keys  = list(methods) + ["ensemble"]
all_method_maes  = [results[m]["mae"]  for m in all_method_keys if m in results]
all_method_rmses = [results[m]["rmse"] for m in all_method_keys if m in results]
all_method_names = [METHOD_LABELS[m]   for m in all_method_keys if m in results]
bar_colors       = [COLORS[m]          for m in all_method_keys if m in results]

fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK)
fig.patch.set_facecolor(DARK)

for ax, vals, title in zip(
    axes,
    [all_method_maes, all_method_rmses],
    ["MAE (mm) — lower is better", "RMSE (mm) — lower is better"]
):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)

    bars = ax.bar(range(len(vals)), vals, color=bar_colors,
                  width=0.6, edgecolor=DARK, zorder=2)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(all_method_names, rotation=18, ha="right",
                       fontsize=9, color=TEXT)
    ax.set_ylabel("mm", fontsize=11, color=TEXT)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(2.0, color=MUTED, lw=1, ls="--", alpha=0.5, label="2mm target")
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor="#30363d", labelcolor=MUTED)
    ax.grid(True, axis="y", alpha=0.12, color="#30363d")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{val:.2f}", ha="center", fontsize=10,
                color=TEXT, fontweight="bold")

fig.suptitle("G1 — All 4 Diameter Methods Comparison", fontsize=15,
             fontweight="bold", color=TEXT, y=1.01)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "G1_method_comparison_bars.png")
plt.savefig(out1, dpi=140, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"\nSaved: {out1}")

# ── FIGURE 2: Per-apple error lines ───────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(20, 8), facecolor=DARK)
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
ax.tick_params(colors=MUTED, labelsize=9)

x = np.arange(n)
for method in all_method_keys:
    if method not in results:
        continue
    errs = results[method]["errs"]
    ax.plot(x, errs, "o-", color=COLORS[method], lw=1.8, ms=6,
            label=f"{METHOD_LABELS[method]}  MAE={results[method]['mae']:.2f}mm",
            markeredgecolor=DARK, alpha=0.9)

ax.axhline(0,   color="#30363d", lw=1)
ax.axhline(+2,  color=MUTED, lw=0.8, ls="--", alpha=0.5)
ax.axhline(-2,  color=MUTED, lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"#{r['apple_idx']}\nL{r['lane']}P{r['pos']}"
                    for r in records], fontsize=8, color=TEXT)
ax.set_ylabel("Error: Estimated − GT (mm)", fontsize=11, color=TEXT)
ax.set_title("Per-Apple Error by Method — G1 Session", fontsize=13,
             fontweight="bold", color=TEXT)
ax.legend(fontsize=9.5, facecolor=PANEL, edgecolor="#30363d",
          labelcolor=TEXT, loc="upper right", ncol=2)
ax.grid(True, axis="y", alpha=0.12, color="#30363d")
ax.set_ylim(-7, 7)

out2 = os.path.join(OUT_DIR, "G1_method_comparison_errors.png")
plt.tight_layout()
plt.savefig(out2, dpi=140, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"Saved: {out2}")
print("\nDone.")
