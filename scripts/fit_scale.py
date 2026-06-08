"""
scripts/fit_scale.py  --  Step 4: Geometric Evaluation & Scale Fitting
======================================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

Loads one or more session pkls, runs view_fusion to get per-apple feature
vectors, fits a scale factor per method, and reports accuracy.

Usage:
    # Single session test
    python scripts/fit_scale.py --pkls data/frame_features/G1.pkl

    # Multi-session (fits scale on all, same result as LOO-CV without held-out)
    python scripts/fit_scale.py --pkls data/frame_features/G1.pkl data/frame_features/G2.pkl ...
"""

import sys, os, pickle, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from core.log import get_logger, configure_root

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.sizing.view_fusion import fuse_session, feature_matrix

logger = get_logger(__name__)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Geometric scale fitting and evaluation")
parser.add_argument("--pkls", nargs="+", required=True, help="Session pkl files")
parser.add_argument("--out",  default=None,  help="Output directory for plots")
args = parser.parse_args()

OUT_DIR = args.out or str(Path(args.pkls[0]).parent.parent / "method_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

DARK  = "#0d1117"; PANEL = "#161b22"; TEXT = "#e6edf3"; MUTED = "#8b949e"
COLORS = {
    "d_maxw_wmean": "#58a6ff",  "d_maxw_peak": "#1f6feb",
    "d_sym_wmean":  "#3fb950",  "d_sym_peak":  "#238636",
    "d_ell_wmean":  "#d29922",  "d_ell_peak":  "#9e6a03",
    "d_area_wmean": "#f78166",  "d_area_peak": "#da3633",
    "ell_a":        "#bc8cff",  "ell_b":       "#8957e5",
}
METHOD_LABELS = {
    "d_maxw_wmean": "M1 MaxWidth  Q-mean",
    "d_maxw_peak":  "M1 MaxWidth  Peak",
    "d_sym_wmean":  "M2 Symmetry  Q-mean",
    "d_sym_peak":   "M2 Symmetry  Peak",
    "d_ell_wmean":  "M3 Ellipse   Q-mean",
    "d_ell_peak":   "M3 Ellipse   Peak",
    "d_area_wmean": "M4 Area      Q-mean",
    "d_area_peak":  "M4 Area      Peak",
    "ell_a":        "Consensus ell_a",
    "ell_b":        "Consensus ell_b",
}

configure_root()

# ── Load and fuse ─────────────────────────────────────────────────────────────
all_fused = []
for pkl_path in args.pkls:
    logger.info(f"Loading {Path(pkl_path).name} ...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    fused = fuse_session(data)
    all_fused.extend(fused)
    logger.info(f"  {len(fused)} apples fused")

logger.info(f"Total: {len(all_fused)} apples")

# ── Build feature matrix ──────────────────────────────────────────────────────
EVAL_METHODS = [
    "d_maxw_wmean", "d_maxw_peak",
    "d_sym_wmean",  "d_sym_peak",
    "d_ell_wmean",  "d_ell_peak",
    "d_area_wmean", "d_area_peak",
    "ell_a",        "ell_b",
]
X, y, metas, cols = feature_matrix(all_fused, feature_cols=EVAL_METHODS)
valid_mask = np.isfinite(y) & (y > 0)
X_v = X[valid_mask]; y_v = y[valid_mask]
n = len(y_v)

# ── Fit scale & evaluate per method ──────────────────────────────────────────
logger.info("="*72)
logger.info(f"{'Method':<26} {'Scale':>8} {'MAE':>7} {'RMSE':>8} {'MaxErr':>8} {'R2':>6}")
logger.info("="*72)

results = {}
for ci, method in enumerate(EVAL_METHODS):
    d_px = X_v[:, ci]
    ok = d_px > 5
    if ok.sum() < 3:
        logger.warning(f"  {METHOD_LABELS[method]:<26}  INSUFFICIENT DATA")
        continue
    scale    = float(np.sum(d_px[ok] * y_v[ok]) / np.sum(d_px[ok] ** 2))
    d_mm     = d_px * scale
    errs     = d_mm[ok] - y_v[ok]
    mae      = float(np.mean(np.abs(errs)))
    rmse     = float(np.sqrt(np.mean(errs**2)))
    max_err  = float(np.max(np.abs(errs)))
    ss_res   = float(np.sum(errs**2))
    ss_tot   = float(np.sum((y_v[ok] - y_v[ok].mean())**2))
    r2       = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

    results[method] = dict(scale=scale, d_mm=d_mm, errs=errs,
                           mae=mae, rmse=rmse, max_err=max_err, r2=r2)
    best_marker = ""
    if results and mae == min(v["mae"] for v in results.values()):
        best_marker = " [BEST]"
    logger.info(f"  {METHOD_LABELS[method]:<26}  {scale:6.4f}  {mae:5.2f}mm"
          f"  {rmse:6.2f}mm  {max_err:6.2f}mm  {r2:5.3f}{best_marker}")

logger.info("="*72)

# ── Per-apple detail ──────────────────────────────────────────────────────────
logger.info(f"{'#':>3}  {'GT':>6}  " +
      "  ".join(f"{METHOD_LABELS[m][:8]:>8}" for m in EVAL_METHODS if m in results))
logger.info("-"*(8 + 10*len(results)))
for i, meta in enumerate(metas):
    if not valid_mask[i]: continue
    row = f"#{meta['apple_idx']+1:2d}  {y[i]:6.1f}  "
    for m in EVAL_METHODS:
        if m not in results: continue
        ci = EVAL_METHODS.index(m)
        row += f"{results[m]['d_mm'][i]:8.1f}  "
    logger.info(row)

# ── Plot 1: MAE bar chart ─────────────────────────────────────────────────────
keys = [m for m in EVAL_METHODS if m in results]
maes  = [results[m]["mae"]  for m in keys]
rmses = [results[m]["rmse"] for m in keys]
colors= [COLORS.get(m, "#888") for m in keys]
labels= [METHOD_LABELS[m]       for m in keys]

fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor=DARK)
fig.patch.set_facecolor(DARK)
sessions_str = "+".join(Path(p).stem for p in args.pkls)

for ax, vals, title in zip(axes, [maes, rmses],
                            ["MAE (mm) -- lower is better",
                             "RMSE (mm) -- lower is better"]):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED, labelsize=9)
    bars = ax.bar(range(len(vals)), vals, color=colors,
                  width=0.65, edgecolor=DARK, zorder=2)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9, color=TEXT)
    ax.set_ylabel("mm", fontsize=11, color=TEXT)
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT)
    ax.axhline(2.0, color=MUTED, lw=1, ls="--", alpha=0.5, label="2mm")
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor="#30363d", labelcolor=MUTED)
    ax.grid(True, axis="y", alpha=0.12, color="#30363d")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.2f}", ha="center", fontsize=9, color=TEXT, fontweight="bold")

fig.suptitle(f"Per-Method Scale Fit -- {sessions_str}   (N={n} apples)",
             fontsize=14, fontweight="bold", color=TEXT, y=1.01)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, f"{sessions_str}_fit_scale_bars.png")
plt.savefig(out1, dpi=140, bbox_inches="tight", facecolor=DARK)
plt.close()
logger.info(f"Saved: {out1}")

# ── Plot 2: Per-apple error lines ─────────────────────────────────────────────
valid_idxs = [i for i in range(len(metas)) if valid_mask[i]]
x = np.arange(len(valid_idxs))

# Pick the 5 most informative methods to avoid clutter
plot_methods = ["d_area_wmean", "d_sym_wmean", "d_ell_wmean", "d_maxw_wmean", "ell_a"]

fig2, ax = plt.subplots(figsize=(20, 8), facecolor=DARK)
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
ax.tick_params(colors=MUTED)

for m in plot_methods:
    if m not in results: continue
    errs = results[m]["errs"]
    ax.plot(x, errs, "o-", color=COLORS.get(m,"#888"), lw=1.8, ms=5,
            label=f"{METHOD_LABELS[m]}  MAE={results[m]['mae']:.2f}mm",
            markeredgecolor=DARK, alpha=0.9)

ax.axhline(0, color="#30363d", lw=1)
ax.axhline(+2, color=MUTED, lw=0.8, ls="--", alpha=0.5)
ax.axhline(-2, color=MUTED, lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"#{metas[i]['apple_idx']+1}\nL{metas[i]['lane']}P{metas[i]['pos']}"
                    for i in valid_idxs], fontsize=8, color=TEXT)
ax.set_ylabel("Error: Estimated - GT (mm)", fontsize=11, color=TEXT)
ax.set_title(f"Per-Apple Error -- {sessions_str}   (view_fusion fused features)",
             fontsize=13, fontweight="bold", color=TEXT)
ax.legend(fontsize=10, facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, ncol=2)
ax.grid(True, axis="y", alpha=0.12, color="#30363d")
ax.set_ylim(-7, 7)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, f"{sessions_str}_fit_scale_errors.png")
plt.savefig(out2, dpi=140, bbox_inches="tight", facecolor=DARK)
plt.close()
logger.info(f"Saved: {out2}")
logger.info("Done.")
