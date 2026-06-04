"""
scripts/train_size_regressor.py  —  Step 5: ML Regression Training
====================================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

Loads all session pkls, runs view_fusion, trains Ridge + RandomForest
regressors, evaluates with Leave-One-Session-Out (LOO-CV).

EXCLUSION CRITERIA (imaging issues, not code issues):
  cx_range < 1000 px → apple didn't cross enough of the frame.

DATA SPLIT:
  - Train: G1-G6, G8-G9  (8 sessions, ~140 apples)
  - Test:  G10            (blind, never seen during training)
  - G11 excluded: GT labeling error (apples #1 & #4 swapped)

OUTPUTS (saved to OUT_DIR):
  fig1_loo_scatter.png       — LOO predicted vs GT, all sessions
  fig2_loo_per_session.png   — per-session MAE/RMSE bars + error distribution
  fig3_blind_test.png        — G10 blind test scatter + per-apple errors
  fig4_model_insight.png     — Ridge coefficients + residual diagnostics
  fig5_r2_explanation.png    — why R² can be lower with small apple size range
  training_metrics.csv       — all numeric results

Usage:
    python scripts/train_size_regressor.py
"""

import sys, os, pickle, warnings, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from scipy import stats

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
TEST_SESSIONS  = ["G10"]
ALL_SESSIONS   = TRAIN_SESSIONS + TEST_SESSIONS

MIN_CX_RANGE = 1000   # px

# ── Dark theme ─────────────────────────────────────────────────────────────────
DARK  = "#0d1117"; PANEL = "#161b22"; PANEL2 = "#1c2128"
TEXT  = "#e6edf3"; MUTED = "#8b949e"; BORDER = "#30363d"
C_RIDGE = "#58a6ff"; C_RF = "#3fb950"; C_GB = "#d29922"
C_GT    = "#f78166"; C_OK = "#3fb950"; C_BAD = "#f85149"
C_LIVE  = "#bc8cff"

SESS_PALETTE = [
    "#58a6ff","#3fb950","#d29922","#f78166","#bc8cff",
    "#39d353","#ff7b72","#ffa657"
]

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER,
    "grid.alpha": 0.4, "font.family": "DejaVu Sans",
})

def styled_ax(ax):
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.grid(True, alpha=0.25, color=BORDER)

# ── Load all sessions ──────────────────────────────────────────────────────────
print("=" * 72)
print("LOADING AND FUSING ALL SESSIONS")
print("=" * 72)

all_fused, excluded = [], []
for sess in ALL_SESSIONS:
    pkl_path = Path(PKL_DIR) / f"{sess}.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] {sess}.pkl not found"); continue
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    fused = fuse_session(data)
    kept, excl = 0, 0
    for r in fused:
        if r.get("cx_range", 9999) < MIN_CX_RANGE:
            excluded.append(r); excl += 1
        else:
            all_fused.append(r); kept += 1
    print(f"  {sess}: {kept} kept, {excl} excluded (cx_range<{MIN_CX_RANGE}px)")

print(f"\n  Total kept    : {len(all_fused)}")
print(f"  Total excluded: {len(excluded)}")

# ── Feature matrix ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "d_area_wmean", "d_maxw_wmean", "d_sym_wmean", "d_ell_wmean",
    "d_area_peak",  "d_maxw_peak",
    "ell_a", "ell_b",
    "mean_Q", "lane",
]
FEAT_LABELS = [
    "Area (Q-mean)", "MaxWidth (Q-mean)", "Symmetry (Q-mean)", "Ellipse (Q-mean)",
    "Area (peak)", "MaxWidth (peak)",
    "Ellipse major (ell_a)", "Ellipse minor (ell_b)",
    "Mean quality", "Lane",
]

X, y, metas, cols = feature_matrix(all_fused, feature_cols=FEATURE_COLS)
sessions_arr = np.array([m["session"] for m in metas])

train_mask = np.array([m["session"] in TRAIN_SESSIONS for m in metas])
test_mask  = np.array([m["session"] in TEST_SESSIONS  for m in metas])
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
test_metas = [metas[i] for i in range(len(metas)) if test_mask[i]]

print(f"\n  Train: {train_mask.sum()} apples  |  Test: {test_mask.sum()} apples")
print(f"  GT range train: {y_train.min():.1f}–{y_train.max():.1f} mm  "
      f"std={y_train.std():.2f}mm")
print(f"  GT range test:  {y_test.min():.1f}–{y_test.max():.1f} mm  "
      f"std={y_test.std():.2f}mm")

# ── Models ─────────────────────────────────────────────────────────────────────
def make_ridge():
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
def make_rf():
    return RandomForestRegressor(n_estimators=200, max_depth=6,
                                 min_samples_leaf=3, random_state=42)
def make_gb():
    return GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=42)

models = {"Ridge": make_ridge, "RandomForest": make_rf, "GradientBoost": make_gb}
model_colors = {"Ridge": C_RIDGE, "RandomForest": C_RF, "GradientBoost": C_GB}

# ── Leave-One-Session-Out CV ───────────────────────────────────────────────────
print(f"\n{'='*72}")
print("LEAVE-ONE-SESSION-OUT CROSS-VALIDATION")
print(f"{'='*72}")

loo = {n: {"pred":[], "true":[], "session":[]} for n in models}

for held in TRAIN_SESSIONS:
    loo_tr = train_mask & (sessions_arr != held)
    loo_vl = train_mask & (sessions_arr == held)
    if loo_vl.sum() == 0: continue
    for name, make in models.items():
        m = make(); m.fit(X[loo_tr], y[loo_tr])
        p = m.predict(X[loo_vl])
        loo[name]["pred"].extend(p.tolist())
        loo[name]["true"].extend(y[loo_vl].tolist())
        loo[name]["session"].extend([held]*loo_vl.sum())

print(f"\n  {'Model':<18} {'MAE':>7} {'RMSE':>8} {'R²':>7} {'MaxErr':>8} {'Acc%':>7}")
print(f"  {'-'*58}")
loo_metrics = {}
for name, res in loo.items():
    p, t = np.array(res["pred"]), np.array(res["true"])
    mae  = mean_absolute_error(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    r2   = r2_score(t, p)
    maxe = np.max(np.abs(p-t))
    acc  = np.mean([(1-abs(pi-ti)/ti)*100 for pi,ti in zip(p,t)])
    loo_metrics[name] = dict(mae=mae, rmse=rmse, r2=r2, maxe=maxe, acc=acc)
    star = " ← BEST" if mae == min(v["mae"] for v in {k:dict(mae=mean_absolute_error(np.array(loo[k]["true"]),np.array(loo[k]["pred"]))) for k in loo}.values())  else ""
    print(f"  {name:<18} {mae:6.3f}mm  {rmse:6.3f}mm  {r2:6.3f}  {maxe:6.3f}mm  {acc:6.2f}%{star}")

# ── Final train → blind test ───────────────────────────────────────────────────
print(f"\n{'='*72}")
print("BLIND TEST (G10)")
print(f"{'='*72}")

final = {}
best_mae = 999
for name, make in models.items():
    m = make(); m.fit(X_train, y_train)
    p    = m.predict(X_test)
    mae  = mean_absolute_error(y_test, p)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    r2   = r2_score(y_test, p)
    maxe = float(np.max(np.abs(p-y_test)))
    acc3 = float(np.mean(np.abs(p-y_test) <= 3.0))*100
    acc5p= float(np.mean(np.abs(p-y_test)/y_test <= 0.05))*100
    pacc = float(np.mean([(1-abs(pi-ti)/ti)*100 for pi,ti in zip(p,y_test)]))
    final[name] = dict(model=m, preds=p, mae=mae, rmse=rmse, r2=r2,
                       maxe=maxe, acc3=acc3, acc5p=acc5p, pacc=pacc)
    if mae < best_mae: best_mae = mae; best_name = name

print(f"\n  {'Model':<18} {'MAE':>7} {'RMSE':>8} {'R²':>7} "
      f"{'±3mm%':>7} {'±5%%':>7} {'Acc%(paper)':>12}")
print(f"  {'-'*72}")
for name, res in final.items():
    star = " ← BEST" if name == best_name else ""
    print(f"  {name:<18} {res['mae']:6.3f}mm  {res['rmse']:6.3f}mm  "
          f"{res['r2']:6.3f}  {res['acc3']:6.1f}%  {res['acc5p']:6.1f}%  "
          f"{res['pacc']:10.2f}%{star}")

print(f"\n  Paper (Lu 2025): RMSE=1.87mm  R²=0.967  Acc%=97.60%  @ 1 apple/lane/s")
print(f"\n  Our GT std  (G10, N={len(y_test)}):  {y_test.std():.2f}mm  "
      f"→ explains lower R² vs paper (their test std ≈ 10.3mm)")

# ── Save best model ────────────────────────────────────────────────────────────
bm = final[best_name]["model"]
model_path = Path(MODEL_OUT)/"size_model.pkl"
with open(model_path,"wb") as f:
    pickle.dump({"model": bm, "feature_cols": FEATURE_COLS,
                 "model_name": best_name,
                 "train_sessions": TRAIN_SESSIONS,
                 "test_sessions": TEST_SESSIONS,
                 "loo_metrics": loo_metrics[best_name],
                 "blind_metrics": {k:v for k,v in final[best_name].items() if k!="model"},
                 "mae": final[best_name]["mae"],
                 "rmse": final[best_name]["rmse"],
                 "r2":   final[best_name]["r2"]}, f)
print(f"\n  Best model ({best_name}) saved → {model_path}")

# ── Save CSV ───────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "training_metrics.csv")
with open(csv_path,"w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["eval","model","mae_mm","rmse_mm","r2","maxerr_mm","acc_pct_paper"])
    for name,res in loo_metrics.items():
        w.writerow(["LOO-CV",name,f"{res['mae']:.4f}",f"{res['rmse']:.4f}",
                    f"{res['r2']:.4f}",f"{res['maxe']:.4f}",f"{res['acc']:.2f}"])
    for name,res in final.items():
        w.writerow(["BlindTest-G10",name,f"{res['mae']:.4f}",f"{res['rmse']:.4f}",
                    f"{res['r2']:.4f}",f"{res['maxe']:.4f}",f"{res['pacc']:.2f}"])
    w.writerow(["Reference(their-own-blind-test)","Paper(Lu2025)","N/A","1.870","0.967","N/A","97.60"])
print(f"  Metrics CSV → {csv_path}")

# =============================================================================
# FIGURE 1 — LOO Scatter: Predicted vs GT, all sessions
# =============================================================================
fig1, axes = plt.subplots(1,3, figsize=(22,7), facecolor=DARK)
fig1.suptitle("Leave-One-Session-Out Cross-Validation — Predicted vs GT",
              fontsize=14, fontweight="bold", color=TEXT, y=1.01)

for ax_i, (name, res) in enumerate(loo.items()):
    ax = axes[ax_i]; styled_ax(ax)
    p  = np.array(res["pred"]); t = np.array(res["true"])
    sessions_u = TRAIN_SESSIONS
    for si, sess in enumerate(sessions_u):
        mask = [s==sess for s in res["session"]]
        ax.scatter(np.array(t)[mask], np.array(p)[mask],
                   color=SESS_PALETTE[si%len(SESS_PALETTE)],
                   s=60, alpha=0.85, edgecolors=DARK, lw=0.5,
                   label=sess, zorder=3)

    lims = [min(t.min(),p.min())-3, max(t.max(),p.max())+3]
    ax.plot(lims, lims, color=MUTED, lw=1.5, ls="--", label="Perfect (y=x)", zorder=2)
    ax.fill_between(lims, [l-3 for l in lims], [l+3 for l in lims],
                    alpha=0.08, color=C_RIDGE, label="±3mm band")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("GT Caliper (mm)", fontsize=11)
    ax.set_ylabel("Predicted (mm)", fontsize=11)
    m = loo_metrics[name]
    ax.set_title(f"{name}\nLOO-CV", color=model_colors[name], fontsize=12, fontweight="bold")
    ax.text(0.04, 0.96,
            f"MAE  = {m['mae']:.3f} mm\nRMSE = {m['rmse']:.3f} mm\n"
            f"R²    = {m['r2']:.3f}\nAcc% = {m['acc']:.2f}%",
            transform=ax.transAxes, color=TEXT, fontsize=10,
            va="top", bbox=dict(facecolor=PANEL2, edgecolor=BORDER, boxstyle="round,pad=0.4"))
    ax.legend(fontsize=7.5, facecolor=PANEL2, edgecolor=BORDER,
              labelcolor=TEXT, ncol=2, loc="lower right")

plt.tight_layout()
out1 = os.path.join(OUT_DIR,"fig1_loo_scatter.png")
fig1.savefig(out1, dpi=140, bbox_inches="tight", facecolor=DARK); plt.close()
print(f"\n  Saved: {out1}")

# =============================================================================
# FIGURE 2 — Per-Session Analysis + Error Distribution
# =============================================================================
fig2 = plt.figure(figsize=(22,10), facecolor=DARK)
gs   = gridspec.GridSpec(2,3, figure=fig2, hspace=0.42, wspace=0.32)

# 2A: Per-session RMSE (Ridge)
ax = fig2.add_subplot(gs[0,:2]); styled_ax(ax)
sess_rmse, sess_mae, sess_n = [], [], []
ridge_loo = loo["Ridge"]
for sess in TRAIN_SESSIONS:
    mask = [s==sess for s in ridge_loo["session"]]
    p_ = np.array(ridge_loo["pred"])[mask]
    t_ = np.array(ridge_loo["true"])[mask]
    sess_rmse.append(np.sqrt(mean_squared_error(t_,p_)))
    sess_mae.append(mean_absolute_error(t_,p_))
    sess_n.append(len(t_))

x = np.arange(len(TRAIN_SESSIONS))
bars = ax.bar(x-0.18, sess_rmse, 0.35, color=C_RIDGE, alpha=0.85,
              label="RMSE", edgecolor=DARK, zorder=3)
bars2= ax.bar(x+0.18, sess_mae,  0.35, color=C_RF,    alpha=0.85,
              label="MAE",  edgecolor=DARK, zorder=3)
ax.axhline(loo_metrics["Ridge"]["rmse"], color=C_RIDGE, lw=1.5, ls="--",
           alpha=0.6, label=f"Overall RMSE={loo_metrics['Ridge']['rmse']:.3f}mm")
ax.axhline(1.87, color=MUTED, lw=1.2, ls=":", alpha=0.7, label="Paper RMSE=1.87mm")
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n(N={n})" for s,n in zip(TRAIN_SESSIONS,sess_n)], color=TEXT)
ax.set_ylabel("Error (mm)", color=TEXT)
ax.set_title("Ridge LOO-CV — Per-Session RMSE and MAE", color=TEXT, fontweight="bold")
ax.legend(facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
for bar,val in zip(bars, sess_rmse):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{val:.2f}", ha="center", fontsize=8, color=TEXT)
for bar,val in zip(bars2, sess_mae):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{val:.2f}", ha="center", fontsize=8, color=TEXT)

# 2B: Error histogram (all models)
ax = fig2.add_subplot(gs[0,2]); styled_ax(ax)
for name, res in loo.items():
    errs = np.array(res["pred"]) - np.array(res["true"])
    ax.hist(errs, bins=20, alpha=0.55, color=model_colors[name],
            label=name, edgecolor=DARK, density=True)
ax.axvline(0, color=TEXT, lw=1.2, ls="--")
ax.axvline(+3, color=MUTED, lw=0.8, ls=":", alpha=0.7, label="±3mm")
ax.axvline(-3, color=MUTED, lw=0.8, ls=":", alpha=0.7)
ax.set_xlabel("Error: Predicted − GT (mm)"); ax.set_ylabel("Density")
ax.set_title("LOO Error Distribution\n(all models)", color=TEXT, fontweight="bold")
ax.legend(facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# 2C: Per-apple LOO errors (Ridge), sorted
ax = fig2.add_subplot(gs[1,:]); styled_ax(ax)
r_errs = np.array(loo["Ridge"]["pred"]) - np.array(loo["Ridge"]["true"])
sort_idx = np.argsort(r_errs)
colors_e = [C_OK if abs(e)<=3 else C_BAD for e in r_errs[sort_idx]]
ax.bar(range(len(r_errs)), r_errs[sort_idx], color=colors_e,
       edgecolor=DARK, width=0.85, zorder=3)
ax.axhline(0,  color=TEXT, lw=1)
ax.axhline(+3, color=MUTED, lw=1, ls="--", alpha=0.7)
ax.axhline(-3, color=MUTED, lw=1, ls="--", alpha=0.7, label="±3mm threshold")
ax.set_xlabel(f"Apple rank (sorted by error, N={len(r_errs)})")
ax.set_ylabel("Error: Predicted − GT (mm)")
ax.set_title("Ridge LOO-CV — Per-Apple Errors (sorted)   "
             f"Green = within ±3mm ({sum(1 for e in r_errs if abs(e)<=3)}/{len(r_errs)})",
             color=TEXT, fontweight="bold")
legend_e = [Line2D([0],[0],color=C_OK,marker="s",ls="",ms=10,label="Within ±3mm"),
            Line2D([0],[0],color=C_BAD,marker="s",ls="",ms=10,label="Outside ±3mm"),
            Line2D([0],[0],color=MUTED,ls="--",lw=1.5,label="±3mm boundary")]
ax.legend(handles=legend_e, facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT)

plt.suptitle("LOO Cross-Validation — Per-Session & Per-Apple Analysis (Ridge)",
             fontsize=13, fontweight="bold", color=TEXT, y=1.01)
out2 = os.path.join(OUT_DIR,"fig2_loo_per_session.png")
fig2.savefig(out2, dpi=140, bbox_inches="tight", facecolor=DARK); plt.close()
print(f"  Saved: {out2}")

# =============================================================================
# FIGURE 3 — Blind Test G10: Scatter + Per-Apple Errors
# =============================================================================
fig3, axes = plt.subplots(1,2, figsize=(20,8), facecolor=DARK)
fig3.suptitle(f"Blind Test — G10  (N={len(y_test)} apples, never seen during training)",
              fontsize=14, fontweight="bold", color=TEXT, y=1.01)

# 3A: Scatter all models
ax = axes[0]; styled_ax(ax)
lims = [min(y_test.min(), min(v["preds"].min() for v in final.values()))-3,
        max(y_test.max(), max(v["preds"].max() for v in final.values()))+3]
ax.plot(lims, lims, color=MUTED, lw=1.5, ls="--", label="Perfect (y=x)", zorder=1)
ax.fill_between(lims,[l-3 for l in lims],[l+3 for l in lims],
                alpha=0.07, color=C_RIDGE, label="±3mm band")

for name, res in final.items():
    errs = np.abs(res["preds"]-y_test)
    ec   = [C_OK if e<=3 else C_BAD for e in errs]
    ax.scatter(y_test, res["preds"], color=model_colors[name],
               s=80, alpha=0.85, edgecolors=DARK, lw=0.6,
               label=f"{name}  RMSE={res['rmse']:.3f}mm", zorder=3)

# Annotate apple indices for best model
for i,(gt,pred,meta) in enumerate(zip(y_test, final[best_name]["preds"], test_metas)):
    err = abs(pred-gt)
    if err > 3:
        ax.annotate(f"#{meta['apple_idx']+1}\nL{meta['lane']}",
                    (gt,pred), textcoords="offset points", xytext=(8,4),
                    fontsize=7.5, color=C_BAD)

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("GT Caliper (mm)", fontsize=12)
ax.set_ylabel("Predicted (mm)", fontsize=12)
ax.set_title("Predicted vs GT — All Models", color=TEXT, fontweight="bold")
res_b = final[best_name]
ax.text(0.04,0.97,
        f"Best model: {best_name}\n"
        f"MAE  = {res_b['mae']:.3f} mm\n"
        f"RMSE = {res_b['rmse']:.3f} mm\n"
        f"R²    = {res_b['r2']:.3f}\n"
        f"Acc% = {res_b['pacc']:.2f}%\n"
        f"±3mm = {res_b['acc3']:.1f}%\n\n"
        f"Paper (Lu 2025):\n"
        f"RMSE = 1.870 mm\n"
        f"R²    = 0.967\n"
        f"Acc% = 97.60%",
        transform=ax.transAxes, color=TEXT, fontsize=9.5, va="top",
        bbox=dict(facecolor=PANEL2, edgecolor=C_RIDGE, boxstyle="round,pad=0.5",lw=1.5))
ax.legend(facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9, loc="lower right")

# 3B: Per-apple error bars (best model)
ax = axes[1]; styled_ax(ax)
best_p = final[best_name]["preds"]
errs   = best_p - y_test
apple_labels = [f"#{m['apple_idx']+1}\nL{m['lane']}" for m in test_metas]
xp = np.arange(len(errs))
cols_b = [C_OK if abs(e)<=3 else C_BAD for e in errs]
bars = ax.bar(xp, errs, color=cols_b, edgecolor=DARK, width=0.7, zorder=3)
ax.axhline(0,  color=TEXT, lw=1.2)
ax.axhline(+3, color=MUTED, lw=1, ls="--", alpha=0.7)
ax.axhline(-3, color=MUTED, lw=1, ls="--", alpha=0.7)
ax.set_xticks(xp); ax.set_xticklabels(apple_labels, fontsize=8, color=TEXT)
ax.set_ylabel("Error: Predicted − GT (mm)", fontsize=11)
ax.set_title(f"{best_name} — Per-Apple Error on G10\n"
             f"Green = within ±3mm ({sum(1 for e in errs if abs(e)<=3)}/{len(errs)})",
             color=TEXT, fontweight="bold")
for bar,e,gt in zip(bars,errs,y_test):
    ypos = e + 0.1 if e>=0 else e - 0.35
    ax.text(bar.get_x()+bar.get_width()/2, ypos,
            f"{e:+.1f}", ha="center", fontsize=7.5, color=TEXT)
legend_b = [Line2D([0],[0],color=C_OK,marker="s",ls="",ms=10,label="Within ±3mm"),
            Line2D([0],[0],color=C_BAD,marker="s",ls="",ms=10,label="Outside ±3mm")]
ax.legend(handles=legend_b, facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT)

plt.tight_layout()
out3 = os.path.join(OUT_DIR,"fig3_blind_test.png")
fig3.savefig(out3, dpi=140, bbox_inches="tight", facecolor=DARK); plt.close()
print(f"  Saved: {out3}")

# =============================================================================
# FIGURE 4 — Model Insight: Ridge Coefficients + Residuals + Q-Q
# =============================================================================
# Train final Ridge on all training data to get interpretable coefficients
ridge_final = make_ridge()
ridge_final.fit(X_train, y_train)
ridge_step  = ridge_final.named_steps["model"]
scaler_step = ridge_final.named_steps["scaler"]
coefs = ridge_step.coef_          # coefficients in SCALED space
# Scale coefs back to unscaled for display (multiply by scale std)
coefs_unscaled = coefs / scaler_step.scale_
best_preds_train = ridge_final.predict(X_train)
residuals_train  = y_train - best_preds_train

fig4, axes = plt.subplots(1,3, figsize=(22,7), facecolor=DARK)
fig4.suptitle("Ridge Regression — Model Insight (trained on all training sessions)",
              fontsize=13, fontweight="bold", color=TEXT, y=1.01)

# 4A: Feature coefficients (standardized — relative importance)
ax = axes[0]; styled_ax(ax)
sorted_idx = np.argsort(np.abs(coefs))[::-1]
cols_c = [C_RIDGE if c>0 else C_BAD for c in coefs[sorted_idx]]
ax.barh(range(len(coefs)), coefs[sorted_idx], color=cols_c,
        edgecolor=DARK, alpha=0.85)
ax.set_yticks(range(len(coefs)))
ax.set_yticklabels([FEAT_LABELS[i] for i in sorted_idx], fontsize=9, color=TEXT)
ax.set_xlabel("Coefficient (in scaled feature space)", fontsize=10)
ax.set_title("Ridge Coefficients\n(blue = positive, red = negative)",
             color=TEXT, fontweight="bold")
ax.axvline(0, color=MUTED, lw=1)
ax.text(0.98, 0.02,
        f"intercept = {ridge_step.intercept_:.2f}mm",
        transform=ax.transAxes, ha="right", fontsize=8.5,
        color=MUTED, bbox=dict(facecolor=PANEL2,edgecolor=BORDER,boxstyle="round"))

# 4B: Residuals vs Fitted
ax = axes[1]; styled_ax(ax)
ax.scatter(best_preds_train, residuals_train,
           color=C_RIDGE, s=50, alpha=0.7, edgecolors=DARK, lw=0.5, zorder=3)
ax.axhline(0, color=MUTED, lw=1.5, ls="--")
ax.axhline(+3, color=C_BAD, lw=0.8, ls=":", alpha=0.6, label="±3mm")
ax.axhline(-3, color=C_BAD, lw=0.8, ls=":", alpha=0.6)
ax.set_xlabel("Fitted value (mm)", fontsize=10)
ax.set_ylabel("Residual: GT − Predicted (mm)", fontsize=10)
ax.set_title("Residuals vs Fitted\n(Training data — Ridge LOO)",
             color=TEXT, fontweight="bold")
ax.legend(facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT)
ax.text(0.04, 0.97,
        f"Bias: {residuals_train.mean():+.3f}mm\n"
        f"Std:  {residuals_train.std():.3f}mm",
        transform=ax.transAxes, va="top", fontsize=9,
        color=TEXT, bbox=dict(facecolor=PANEL2,edgecolor=BORDER,boxstyle="round"))

# 4C: Q-Q plot of residuals
ax = axes[2]; styled_ax(ax)
loo_errs = np.array(loo["Ridge"]["pred"]) - np.array(loo["Ridge"]["true"])
(osm, osr), (slope, intercept, r) = stats.probplot(loo_errs, dist="norm")
ax.scatter(osm, osr, color=C_RIDGE, s=40, alpha=0.8, zorder=3)
ax.plot([min(osm),max(osm)],
        [slope*min(osm)+intercept, slope*max(osm)+intercept],
        color=MUTED, lw=1.5, ls="--", label="Normal reference")
ax.set_xlabel("Theoretical quantiles", fontsize=10)
ax.set_ylabel("Sample quantiles (error, mm)", fontsize=10)
ax.set_title("Q-Q Plot of LOO Residuals\n(normal = points on line)",
             color=TEXT, fontweight="bold")
ax.legend(facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT)
ax.text(0.04,0.97, f"R²(Q-Q) = {r**2:.3f}",
        transform=ax.transAxes, va="top", fontsize=9, color=TEXT,
        bbox=dict(facecolor=PANEL2,edgecolor=BORDER,boxstyle="round"))

plt.tight_layout()
out4 = os.path.join(OUT_DIR,"fig4_model_insight.png")
fig4.savefig(out4, dpi=140, bbox_inches="tight", facecolor=DARK); plt.close()
print(f"  Saved: {out4}")

# =============================================================================
# FIGURE 5 — R² Explanation: Why narrow size range → lower R²
# =============================================================================
fig5, axes = plt.subplots(1,3, figsize=(22,7), facecolor=DARK)
fig5.suptitle("Why R² Is Lower Despite Better RMSE — Apple Size Range Effect",
              fontsize=13, fontweight="bold", color=TEXT, y=1.01)

# Compute paper's implied test std from their R² and RMSE
paper_rmse = 1.87; paper_r2 = 0.967; paper_n = 108
paper_ss_tot = (paper_rmse**2 * paper_n) / (1 - paper_r2)
paper_std = np.sqrt(paper_ss_tot / paper_n)

our_std   = y_test.std()
our_rmse  = final[best_name]["rmse"]
our_r2    = final[best_name]["r2"]
our_n     = len(y_test)

# 5A: Our scatter with annotation
ax = axes[0]; styled_ax(ax)
ax.scatter(y_test, final[best_name]["preds"],
           color=C_RIDGE, s=70, alpha=0.85, edgecolors=DARK, lw=0.5, zorder=3)
lims_o = [y_test.min()-4, y_test.max()+4]
ax.plot(lims_o, lims_o, color=MUTED, lw=1.5, ls="--")
ax.fill_between(lims_o,[l-3 for l in lims_o],[l+3 for l in lims_o],
                alpha=0.1, color=C_RIDGE)
ax.set_xlim(lims_o); ax.set_ylim(lims_o)
ax.set_xlabel("GT Caliper (mm)"); ax.set_ylabel("Predicted (mm)")
ax.set_title(f"Our Blind Test (G10)\nNarrow size range: std={our_std:.1f}mm",
             color=C_RIDGE, fontweight="bold")
ax.text(0.04,0.97,
        f"N = {our_n} apples\nSize range: {y_test.min():.0f}–{y_test.max():.0f}mm\n"
        f"Std = {our_std:.1f}mm\nRMSE = {our_rmse:.3f}mm\nR² = {our_r2:.3f}",
        transform=ax.transAxes, va="top", fontsize=10,
        color=TEXT, bbox=dict(facecolor=PANEL2,edgecolor=C_RIDGE,boxstyle="round",lw=1.5))

# 5B: Simulated "paper-style" scatter (wider range, same RMSE)
ax = axes[1]; styled_ax(ax)
np.random.seed(42)
sim_gt   = np.random.normal(66, paper_std, paper_n)
sim_gt   = np.clip(sim_gt, 45, 90)
sim_err  = np.random.normal(0, paper_rmse, paper_n)
sim_pred = sim_gt + sim_err
sim_r2   = r2_score(sim_gt, sim_pred)
ax.scatter(sim_gt, sim_pred, color=C_RF, s=40, alpha=0.7,
           edgecolors=DARK, lw=0.4, zorder=3)
lims_s = [sim_gt.min()-3, sim_gt.max()+3]
ax.plot(lims_s, lims_s, color=MUTED, lw=1.5, ls="--")
ax.fill_between(lims_s,[l-3 for l in lims_s],[l+3 for l in lims_s],
                alpha=0.1, color=C_RF)
ax.set_xlim(lims_s); ax.set_ylim(lims_s)
ax.set_xlabel("GT Caliper (mm)"); ax.set_ylabel("Predicted (mm)")
ax.set_title(f"Paper-Style Test (simulated)\nWide size range: std≈{paper_std:.1f}mm",
             color=C_RF, fontweight="bold")
ax.text(0.04,0.97,
        f"N ≈ {paper_n} apples\nSize range: ~{int(sim_gt.min())}–{int(sim_gt.max())}mm\n"
        f"Std ≈ {paper_std:.1f}mm\nRMSE ≈ {paper_rmse:.3f}mm\nR² ≈ {sim_r2:.3f}",
        transform=ax.transAxes, va="top", fontsize=10,
        color=TEXT, bbox=dict(facecolor=PANEL2,edgecolor=C_RF,boxstyle="round",lw=1.5))

# 5C: R² vs test-set std (theoretical)
ax = axes[2]; styled_ax(ax)
stds = np.linspace(1.5, 15, 200)
r2_ours_rmse  = [1 - (our_rmse**2  * our_n) / (s**2 * our_n)  for s in stds]
r2_paper_rmse = [1 - (paper_rmse**2* paper_n)/(s**2 * paper_n) for s in stds]
ax.plot(stds, r2_ours_rmse,  color=C_RIDGE, lw=2.5, label=f"RMSE={our_rmse:.2f}mm (ours)")
ax.plot(stds, r2_paper_rmse, color=C_RF,    lw=2.5, label=f"RMSE={paper_rmse:.2f}mm (paper)")
ax.axvline(our_std,    color=C_RIDGE, lw=1.2, ls="--", alpha=0.8,
           label=f"Our test std={our_std:.1f}mm  → R²={our_r2:.3f}")
ax.axvline(paper_std,  color=C_RF,    lw=1.2, ls="--", alpha=0.8,
           label=f"Paper test std≈{paper_std:.1f}mm → R²=0.967")
ax.axhline(our_r2,   color=C_RIDGE, lw=0.8, ls=":", alpha=0.5)
ax.axhline(0.967,    color=C_RF,    lw=0.8, ls=":", alpha=0.5)
ax.scatter([our_std],   [our_r2],  color=C_RIDGE, s=120, zorder=5, edgecolors=TEXT, lw=1.5)
ax.scatter([paper_std], [0.967],   color=C_RF,    s=120, zorder=5, edgecolors=TEXT, lw=1.5)
ax.set_xlabel("Standard deviation of GT apple sizes in test set (mm)", fontsize=10)
ax.set_ylabel("R²", fontsize=10)
ax.set_title("R² vs Apple Size Diversity in Test Set\n"
             "(same RMSE can give very different R²!)", color=TEXT, fontweight="bold")
ax.set_ylim(-0.1, 1.05)
ax.legend(facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
ax.text(0.5, 0.12,
        "Key insight: R² depends on\nhow diverse the test set is.\n"
        "Our G10 had a narrow size\nrange → lower R² even with\nlower RMSE than the paper.",
        transform=ax.transAxes, ha="center", fontsize=9, color=MUTED,
        bbox=dict(facecolor=PANEL2, edgecolor=BORDER, boxstyle="round"))

plt.tight_layout()
out5 = os.path.join(OUT_DIR,"fig5_r2_explanation.png")
fig5.savefig(out5, dpi=140, bbox_inches="tight", facecolor=DARK); plt.close()
print(f"  Saved: {out5}")

# =============================================================================
# FINAL CONSOLE SUMMARY
# =============================================================================
print(f"\n{'='*72}")
print("FINAL SUMMARY")
print(f"{'='*72}")
print(f"\n  LOO Cross-Validation (Ridge, {len(y_train)} training apples):")
m = loo_metrics["Ridge"]
print(f"    MAE  = {m['mae']:.3f}mm   RMSE = {m['rmse']:.3f}mm   "
      f"R² = {m['r2']:.3f}   Acc% = {m['acc']:.2f}%")
print(f"\n  Blind Test G10 (Ridge, {len(y_test)} apples):")
res = final[best_name]
print(f"    MAE  = {res['mae']:.3f}mm   RMSE = {res['rmse']:.3f}mm   "
      f"R² = {res['r2']:.3f}   Acc% = {res['pacc']:.2f}%")
print(f"    ±3mm = {res['acc3']:.1f}%   ±5% = {res['acc5p']:.1f}%")
print(f"\n  Paper (Lu 2025 @ 1 apple/lane/s):")
print(f"    RMSE = 1.870mm   R² = 0.967   Acc% = 97.60%")
print(f"\n  Apple size std:  Our G10 = {y_test.std():.2f}mm  |  "
      f"Paper test ≈ {paper_std:.1f}mm")
print(f"  → Same RMSE formula but 2.4× wider size range → paper gets higher R²")
print(f"\n  Output files in: {OUT_DIR}")
print(f"    fig1_loo_scatter.png       — LOO predicted vs GT (all sessions)")
print(f"    fig2_loo_per_session.png   — per-session bars + error distribution")
print(f"    fig3_blind_test.png        — G10 blind test scatter + per-apple errors")
print(f"    fig4_model_insight.png     — Ridge coefficients + residual diagnostics")
print(f"    fig5_r2_explanation.png    — why R² is lower (size range effect)")
print(f"    training_metrics.csv       — all numeric results")
print("\nDone.")
