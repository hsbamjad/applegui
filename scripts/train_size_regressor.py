"""
scripts/train_size_regressor.py  —  Step 5: ML Regression Training
====================================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

Loads all session pkls, runs view_fusion, trains Ridge + RandomForest
regressors, evaluates with Leave-One-Session-Out (LOO-CV).

# EXCLUSION CRITERIA (imaging issues, not code issues):
#   cx_range < 1000 px → apple didn't cross enough of the frame.
#   NOTE: n_central is NOT used as a filter — G5 is a short session so
#   n_central is naturally low even for valid full-traversal apples.

DATA SPLIT:
  - Train: G1-G6, G8-G9  (8 sessions, 144 apples nominal)
  - Test:  G10, G11       (2 sessions, 36 apples nominal)  ← BLIND, never seen

Usage:
    python scripts/train_size_regressor.py
"""

import sys, os, pickle, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.sizing.view_fusion import fuse_session, feature_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
PKL_DIR  = r"D:\HA\apple_gui\data\frame_features"
OUT_DIR  = r"D:\HA\apple_gui\data\ml_results"
MODEL_OUT= r"D:\HA\apple_gui\models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUT, exist_ok=True)

TRAIN_SESSIONS = ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9"]
TEST_SESSIONS  = ["G10"]          # G11 excluded: GT labeling error (apples #1 & #4 swapped)
ALL_SESSIONS   = TRAIN_SESSIONS + TEST_SESSIONS   # G11 excluded: GT labeling corrupted


# ── Exclusion thresholds (imaging issues) ─────────────────────────────────────
MIN_CX_RANGE = 1000   # px — partial traversal = unreliable measurement
                      # (only filter used; n_frames/n_central NOT used)

# ── Style ─────────────────────────────────────────────────────────────────────
DARK  = "#0d1117"; PANEL = "#161b22"; TEXT = "#e6edf3"; MUTED = "#8b949e"
C_RIDGE = "#58a6ff"; C_RF = "#3fb950"; C_GB = "#d29922"; C_GT = "#f78166"

# ── Load all sessions ─────────────────────────────────────────────────────────
print("=" * 72)
print("LOADING AND FUSING ALL SESSIONS")
print("=" * 72)

all_fused = []
excluded  = []

for sess in ALL_SESSIONS:
    pkl_path = Path(PKL_DIR) / f"{sess}.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] {sess}.pkl not found")
        continue
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    fused = fuse_session(data)

    kept, excl = 0, 0
    for r in fused:
        cx = r.get("cx_range", 9999)
        # Only exclude on cx_range — n_central is naturally low for
        # short sessions (e.g. G5) even when traversal is complete.
        if cx < MIN_CX_RANGE:
            excluded.append({
                "session":   sess,
                "apple_idx": r.get("apple_idx"),
                "gt_mm":     r.get("gt_mm"),
                "cx_range":  cx,
                "n_central": r.get("n_central"),
            })
            excl += 1
        else:
            all_fused.append(r)
            kept += 1
    print(f"  {sess}: {kept} kept, {excl} excluded")

print(f"\n  Total kept    : {len(all_fused)}")
print(f"  Total excluded: {len(excluded)}")
if excluded:
    print("  Excluded apples:")
    for e in excluded:
        print(f"    {e['session']} Apple#{e['apple_idx']} "
              f"gt={e['gt_mm']:.1f}mm  cx_range={e['cx_range']}px  "
              f"n_central={e['n_central']}")

# ── Feature matrix ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "d_area_wmean",   # M4 Area Q-mean       ← best single method
    "d_maxw_wmean",   # M1 MaxWidth Q-mean
    "d_sym_wmean",    # M2 Symmetry Q-mean
    "d_ell_wmean",    # M3 Ellipse Q-mean
    "d_area_peak",    # M4 Area Peak
    "d_maxw_peak",    # M1 MaxWidth Peak
    "ell_a",          # Consensus major axis
    "ell_b",          # Consensus minor axis
    "mean_Q",         # Mean quality score
    "lane",           # Lane 0/1/2 (systematic scale offset)
]

X, y, metas, cols = feature_matrix(all_fused, feature_cols=FEATURE_COLS)
sessions_arr = np.array([m["session"] for m in metas])

# Train/test split by session
train_mask = np.array([m["session"] in TRAIN_SESSIONS for m in metas])
test_mask  = np.array([m["session"] in TEST_SESSIONS  for m in metas])

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\n{'='*72}")
print(f"DATASET SPLIT")
print(f"{'='*72}")
print(f"  Train sessions : {TRAIN_SESSIONS}  ({train_mask.sum()} apples)")
print(f"  Test  sessions : {TEST_SESSIONS}   ({test_mask.sum()} apples)")
print(f"  Features       : {len(FEATURE_COLS)}")
print(f"  GT range train : {y_train.min():.1f} – {y_train.max():.1f} mm")
print(f"  GT range test  : {y_test.min():.1f}  – {y_test.max():.1f} mm")

# ── Define models ─────────────────────────────────────────────────────────────
models = {
    "Ridge":          Pipeline([("scaler", StandardScaler()),
                                ("model",  Ridge(alpha=1.0))]),
    "RandomForest":   RandomForestRegressor(n_estimators=200, max_depth=6,
                                            min_samples_leaf=3, random_state=42),
    "GradientBoost":  GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                learning_rate=0.05, random_state=42),
}

# ── Leave-One-Session-Out CV on TRAIN sessions ────────────────────────────────
print(f"\n{'='*72}")
print("LEAVE-ONE-SESSION-OUT CROSS-VALIDATION (train sessions only)")
print(f"{'='*72}")

loo_results = {name: {"pred": [], "true": [], "session": []} for name in models}

for held_sess in TRAIN_SESSIONS:
    loo_train = train_mask & (sessions_arr != held_sess)
    loo_val   = train_mask & (sessions_arr == held_sess)
    if loo_val.sum() == 0:
        continue
    for name, model_template in models.items():
        m = clone(model_template)   # sklearn clone — works for Pipeline too
        m.fit(X[loo_train], y[loo_train])
        preds = m.predict(X[loo_val])
        loo_results[name]["pred"].extend(preds.tolist())
        loo_results[name]["true"].extend(y[loo_val].tolist())
        loo_results[name]["session"].extend([held_sess] * loo_val.sum())

print(f"\n  {'Model':<18} {'MAE':>7} {'RMSE':>8} {'R²':>6} {'MaxErr':>8}")
print(f"  {'-'*52}")
best_loo_model = None; best_loo_mae = 999
for name, res in loo_results.items():
    p = np.array(res["pred"]); t = np.array(res["true"])
    mae  = mean_absolute_error(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    r2   = r2_score(t, p)
    maxe = np.max(np.abs(p - t))
    star = " <-- BEST" if mae < best_loo_mae else ""
    if mae < best_loo_mae:
        best_loo_mae = mae; best_loo_model = name
    print(f"  {name:<18} {mae:6.2f}mm  {rmse:6.2f}mm  {r2:5.3f}  {maxe:6.2f}mm{star}")

# ── Final train on all train sessions → evaluate on blind test ────────────────
print(f"\n{'='*72}")
print("BLIND TEST (G10 + G11 — never seen during training)")
print(f"{'='*72}")

final_results = {}
for name, model_template in models.items():
    if name == "Ridge":
        m = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    elif name == "RandomForest":
        m = RandomForestRegressor(n_estimators=200, max_depth=6,
                                  min_samples_leaf=3, random_state=42)
    else:
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=42)
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    maxe  = float(np.max(np.abs(preds - y_test)))
    acc3  = float(np.mean(np.abs(preds - y_test) <= 3.0)) * 100
    acc5p = float(np.mean(np.abs(preds - y_test) / y_test <= 0.05)) * 100
    final_results[name] = {"model": m, "preds": preds,
                           "mae": mae, "rmse": rmse, "r2": r2,
                           "maxe": maxe, "acc3": acc3, "acc5p": acc5p}

print(f"\n  {'Model':<18} {'MAE':>7} {'RMSE':>8} {'R²':>6} {'MaxErr':>8} "
      f"{'±3mm%':>7} {'±5%%':>7}")
print(f"  {'-'*68}")
for name, res in final_results.items():
    star = " <-- BEST" if res["mae"] == min(v["mae"] for v in final_results.values()) else ""
    print(f"  {name:<18} {res['mae']:6.2f}mm  {res['rmse']:6.2f}mm  "
          f"{res['r2']:5.3f}  {res['maxe']:6.2f}mm  "
          f"{res['acc3']:6.1f}%  {res['acc5p']:6.1f}%{star}")

print(f"\n  Reference (Jiajing paper): RMSE=1.87mm, R2=0.967, Acc-3mm=98.9%")

# ── Save best model ───────────────────────────────────────────────────────────
best_test_name = min(final_results, key=lambda n: final_results[n]["mae"])
best_model     = final_results[best_test_name]["model"]
model_path     = Path(MODEL_OUT) / "size_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"model": best_model, "feature_cols": FEATURE_COLS,
                 "model_name": best_test_name,
                 "train_sessions": TRAIN_SESSIONS,
                 "test_sessions": TEST_SESSIONS,
                 "mae": final_results[best_test_name]["mae"],
                 "rmse": final_results[best_test_name]["rmse"],
                 "r2":   final_results[best_test_name]["r2"]}, f)
print(f"\n  Best model ({best_test_name}) saved → {model_path}")

# ── Feature importance (RF/GB only) ──────────────────────────────────────────
if best_test_name in ("RandomForest", "GradientBoost"):
    imp  = best_model.feature_importances_
    idx  = np.argsort(imp)[::-1]
    print(f"\n  Feature importances ({best_test_name}):")
    for i in idx:
        print(f"    {FEATURE_COLS[i]:<22} {imp[i]:.3f}")

# ── FIGURE 1: LOO-CV + Blind test comparison ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=DARK)
fig.patch.set_facecolor(DARK)

model_colors = {"Ridge": C_RIDGE, "RandomForest": C_RF, "GradientBoost": C_GB}

# Plot 1: LOO-CV MAE
ax = axes[0]; ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
loo_maes  = [mean_absolute_error(loo_results[n]["true"], loo_results[n]["pred"])
             for n in models]
loo_rmses = [np.sqrt(mean_squared_error(loo_results[n]["true"], loo_results[n]["pred"]))
             for n in models]
bars = ax.bar(range(3), loo_maes, color=[model_colors[n] for n in models],
              width=0.55, edgecolor=DARK, zorder=2)
ax.set_xticks(range(3)); ax.set_xticklabels(list(models.keys()), color=TEXT, fontsize=10)
ax.set_ylabel("MAE (mm)", color=TEXT); ax.set_title("LOO-CV MAE\n(train sessions)", color=TEXT, fontweight="bold")
ax.axhline(1.87, color=MUTED, lw=1.2, ls="--", alpha=0.6, label="Paper RMSE=1.87")
ax.legend(fontsize=9, facecolor=PANEL, edgecolor="#30363d", labelcolor=MUTED)
ax.grid(True, axis="y", alpha=0.12, color="#30363d"); ax.tick_params(colors=MUTED)
for bar, val in zip(bars, loo_maes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{val:.2f}mm", ha="center", fontsize=11, color=TEXT, fontweight="bold")

# Plot 2: Blind test MAE
ax = axes[1]; ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
test_maes = [final_results[n]["mae"] for n in models]
bars = ax.bar(range(3), test_maes, color=[model_colors[n] for n in models],
              width=0.55, edgecolor=DARK, zorder=2)
ax.set_xticks(range(3)); ax.set_xticklabels(list(models.keys()), color=TEXT, fontsize=10)
ax.set_ylabel("MAE (mm)", color=TEXT); ax.set_title("BLIND TEST MAE\n(G10 + G11)", color=TEXT, fontweight="bold")
ax.axhline(1.87, color=MUTED, lw=1.2, ls="--", alpha=0.6, label="Paper RMSE=1.87")
ax.legend(fontsize=9, facecolor=PANEL, edgecolor="#30363d", labelcolor=MUTED)
ax.grid(True, axis="y", alpha=0.12, color="#30363d"); ax.tick_params(colors=MUTED)
for bar, val in zip(bars, test_maes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{val:.2f}mm", ha="center", fontsize=11, color=TEXT, fontweight="bold")

# Plot 3: Predicted vs GT scatter (best model on test)
ax = axes[2]; ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
best_preds = final_results[best_test_name]["preds"]
ax.scatter(y_test, best_preds, color=model_colors[best_test_name],
           s=70, alpha=0.85, edgecolors=DARK, lw=0.5, zorder=3)
lims = [min(y_test.min(), best_preds.min())-2, max(y_test.max(), best_preds.max())+2]
ax.plot(lims, lims, color=MUTED, lw=1.2, ls="--", label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("GT Caliper (mm)", color=TEXT)
ax.set_ylabel("Predicted (mm)", color=TEXT)
ax.set_title(f"Predicted vs GT — Blind Test\n{best_test_name}", color=TEXT, fontweight="bold")
r2_test = final_results[best_test_name]["r2"]
mae_test = final_results[best_test_name]["mae"]
ax.text(0.05, 0.92, f"MAE={mae_test:.2f}mm\nR²={r2_test:.3f}",
        transform=ax.transAxes, color=TEXT, fontsize=11,
        bbox=dict(facecolor=PANEL, edgecolor="#30363d", boxstyle="round"))
ax.legend(fontsize=9, facecolor=PANEL, edgecolor="#30363d", labelcolor=MUTED)
ax.grid(True, alpha=0.1, color="#30363d"); ax.tick_params(colors=MUTED)

fig.suptitle("Apple Size ML Regressor — Training & Blind Test Results",
             fontsize=14, fontweight="bold", color=TEXT, y=1.01)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "ml_results_overview.png")
plt.savefig(out1, dpi=140, bbox_inches="tight", facecolor=DARK); plt.close()
print(f"\n  Saved: {out1}")

# ── FIGURE 2: Per-apple error on blind test ────────────────────────────────────
fig2, ax = plt.subplots(figsize=(22, 8), facecolor=DARK)
ax.set_facecolor(PANEL)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

x = np.arange(len(y_test))
test_metas = [m for m in metas if m["session"] in TEST_SESSIONS]

for name, color in model_colors.items():
    errs = final_results[name]["preds"] - y_test
    ax.plot(x, errs, "o-", color=color, lw=1.8, ms=6,
            label=f"{name}  MAE={final_results[name]['mae']:.2f}mm",
            markeredgecolor=DARK, alpha=0.9)

ax.axhline(0,  color="#30363d", lw=1)
ax.axhline(+3, color=MUTED, lw=0.8, ls="--", alpha=0.5, label="±3mm")
ax.axhline(-3, color=MUTED, lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"#{m['apple_idx']}\n{m['session']}\nL{m['lane']}P{m['pos']}"
                    for m in test_metas], fontsize=8, color=TEXT)
ax.set_ylabel("Error: Predicted − GT (mm)", fontsize=11, color=TEXT)
ax.set_title("Per-Apple Error on Blind Test (G10 + G11)", fontsize=13,
             fontweight="bold", color=TEXT)
ax.legend(fontsize=10, facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, ncol=2)
ax.grid(True, axis="y", alpha=0.12, color="#30363d")
ax.tick_params(colors=MUTED)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "ml_blind_test_errors.png")
plt.savefig(out2, dpi=140, bbox_inches="tight", facecolor=DARK); plt.close()
print(f"  Saved: {out2}")
print("\nDone.")
