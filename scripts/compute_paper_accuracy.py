"""
scripts/compute_paper_accuracy.py
==================================
Compute our sizing accuracy using the SAME formula as the reference paper:

    accuracy_i = (1 - |predicted_i - gt_i| / gt_i) * 100%
    mean_accuracy = mean(accuracy_i across all apples)   [Lu et al. 2025 / Xu et al. 2024b]

Uses EXACTLY the same feature extraction (view_fusion.fuse_session) and
the same FEATURE_COLS as train_size_regressor.py -- no approximations.

Also computes LIVE estimate (pixel x scale) for comparison.

Run:
    python scripts/compute_paper_accuracy.py
"""

import sys, os, pickle
import numpy as np
from pathlib import Path

from core.log import get_logger, configure_root

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Import the SAME fusion code used in training ───────────────────────────────
from core.sizing.view_fusion import fuse_session, feature_matrix

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PKL_DIR    = r"D:\HA\apple_gui\data\frame_features"
MODEL_PATH = r"D:\HA\apple_gui\models\size_model.pkl"
SCALE      = 0.377   # mm/px for LIVE estimate

SESSIONS_TRAIN = ["G1","G2","G3","G4","G5","G6","G8","G9"]
SESSION_BLIND  = "G10"
ALL_SESSIONS   = SESSIONS_TRAIN + [SESSION_BLIND]

MIN_CX_RANGE   = 1000  # same exclusion as training

configure_root()

# ── Load model + feature_cols (saved together in pkl) ─────────────────────────
if not Path(MODEL_PATH).exists():
    logger.error(f"Model not found: {MODEL_PATH}"); sys.exit(1)

with open(MODEL_PATH,"rb") as f: saved = pickle.load(f)
model        = saved["model"]
FEATURE_COLS = saved["feature_cols"]          # exact same cols used in training
model_name   = saved.get("model_name","?")
logger.info(f"Model: {model_name}")
logger.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
logger.info(f"Saved RMSE: {saved.get('rmse','?'):.3f}mm  MAE: {saved.get('mae','?'):.3f}mm")

# ── LIVE estimate: quality-weighted mean of d_area * scale (all frames) ────────
def live_estimate(apple):
    frames = apple.get("frames",[])
    if not frames: return None
    Q = np.array([f.get("quality",0.5) for f in frames])
    D = np.array([f.get("d_area",0.0)  for f in frames]) * SCALE
    return float(np.dot(Q,D)/(Q.sum()+1e-9))

# ── Paper accuracy formula ─────────────────────────────────────────────────────
def paper_acc(pred, gt):
    """Per-apple: (1 - |pred-gt|/gt)*100. Returns array + mean."""
    a = np.array([(1-abs(p-g)/g)*100 for p,g in zip(pred,gt)])
    return a, float(a.mean())

# ── Process all sessions ───────────────────────────────────────────────────────
logger.info("="*72)
logger.info(f"{'Session':<12} {'N':>4}  {'ML Acc%':>8}  {'ML RMSE':>9}  "
      f"{'LIVE Acc%':>10}  {'LIVE RMSE':>10}")
logger.info("="*72)

all_ml_gt=[]; all_ml_p=[]; all_lv_gt=[]; all_lv_p=[]
g10_detail = []

for sess in ALL_SESSIONS:
    pkl_path = Path(PKL_DIR)/f"{sess}.pkl"
    if not pkl_path.exists():
        logger.warning(f"  SKIP {sess} -- pkl not found"); continue

    with open(pkl_path,"rb") as f: data = pickle.load(f)

    # ── Fuse using EXACT same code as training ─────────────────────────────────
    fused = fuse_session(data)

    # Apply same exclusion: cx_range < 1000px
    fused = [r for r in fused if r.get("cx_range",9999) >= MIN_CX_RANGE]

    # Build feature matrix using SAME cols as training
    X, y, metas, _ = feature_matrix(fused, feature_cols=FEATURE_COLS)

    # Drop rows with no GT
    valid = ~np.isnan(y)
    X_v   = X[valid]; y_v = y[valid]
    fused_v = [fused[i] for i in range(len(fused)) if valid[i]]

    if len(y_v) == 0:
        logger.warning(f"  SKIP {sess} -- no GT"); continue

    # ML predictions
    ml_preds = model.predict(X_v)

    # LIVE estimates (from raw apple dicts via fused apple_idx)
    apples_by_idx = {a["apple_idx"]: a for a in data["apples"]}
    lv_preds = []
    lv_gt    = []
    for r in fused_v:
        ai  = r.get("apple_idx")
        gt  = r.get("gt_mm")
        lv  = live_estimate(apples_by_idx[ai]) if ai in apples_by_idx else None
        if lv and gt:
            lv_preds.append(lv); lv_gt.append(gt)

    # Metrics
    ml_acc_arr, ml_acc = paper_acc(ml_preds, y_v)
    ml_rmse = float(np.sqrt(np.mean((ml_preds - y_v)**2)))

    lv_acc_arr, lv_acc = paper_acc(lv_preds, lv_gt) if lv_preds else (np.array([]), 0.0)
    lv_rmse = float(np.sqrt(np.mean([(p-g)**2 for p,g in zip(lv_preds,lv_gt)]))) if lv_preds else 0.0

    blind_tag = " <- BLIND" if sess==SESSION_BLIND else ""
    logger.info(f"{sess+blind_tag:<20} {len(y_v):>4}  {ml_acc:>8.2f}%  {ml_rmse:>8.3f}mm  "
          f"{lv_acc:>10.2f}%  {lv_rmse:>10.3f}mm")

    if sess != SESSION_BLIND:
        all_ml_gt.extend(y_v.tolist()); all_ml_p.extend(ml_preds.tolist())
        all_lv_gt.extend(lv_gt);        all_lv_p.extend(lv_preds)
    else:
        # Store per-apple detail for G10
        for i,(r,pred) in enumerate(zip(fused_v, ml_preds)):
            ai  = r.get("apple_idx")
            gt  = r.get("gt_mm")
            lv  = live_estimate(apples_by_idx[ai]) if ai in apples_by_idx else None
            g10_detail.append({
                "apple": ai+1, "lane": r.get("lane"),
                "gt": gt, "ml": float(pred), "live": lv,
            })

logger.info("-"*72)
if all_ml_gt:
    _, oa_ml = paper_acc(all_ml_p, all_ml_gt)
    or_ml    = float(np.sqrt(np.mean([(p-g)**2 for p,g in zip(all_ml_p,all_ml_gt)])))
    _, oa_lv = paper_acc(all_lv_p, all_lv_gt)
    or_lv    = float(np.sqrt(np.mean([(p-g)**2 for p,g in zip(all_lv_p,all_lv_gt)])))
    n = len(all_ml_gt)
    logger.info(f"{'ALL TRAIN (N='+str(n)+')':<20} {n:>4}  {oa_ml:>8.2f}%  {or_ml:>8.3f}mm  "
          f"{oa_lv:>10.2f}%  {or_lv:>10.3f}mm")

# ── Detailed G10 table ─────────────────────────────────────────────────────────
logger.info("")
logger.info("="*72)
logger.info("DETAILED G10 BLIND TEST -- paper accuracy formula")
logger.info("="*72)
logger.info(f"  {'#':>3}  {'L':>2}  {'GT':>6}  {'ML':>8}  {'ML acc%':>8}  "
      f"{'LIVE':>8}  {'LIVE acc%':>9}")
logger.info(f"  {'-'*64}")

g10_ml_acc=[]; g10_lv_acc=[]
for r in sorted(g10_detail, key=lambda x: x["apple"]):
    gt=r["gt"]; ml=r["ml"]; lv=r["live"]
    ma = (1-abs(ml-gt)/gt)*100 if ml else None
    la = (1-abs(lv-gt)/gt)*100 if lv else None
    g10_ml_acc.append(ma); g10_lv_acc.append(la)
    flag = " [FAIL]" if (ma and ma<95) else ""
    logger.info(f"  #{r['apple']:2d}  L{r['lane']}  {gt:>6.1f}"
          f"  {ml:>8.2f}  {(str(round(ma,2))+'%') if ma else '--':>9}{flag}"
          f"  {(lv if lv else 0):>8.2f}  {(str(round(la,2))+'%') if la else '--':>9}")

logger.info(f"  {'-'*64}")
ml_a_clean = [a for a in g10_ml_acc if a]
lv_a_clean = [a for a in g10_lv_acc if a]
logger.info(f"  Mean ML   accuracy (paper formula): {np.mean(ml_a_clean):.2f}%")
logger.info(f"  Mean LIVE accuracy (paper formula): {np.mean(lv_a_clean):.2f}%")
logger.info(f"  Paper (Lu 2025) reported:           97.60%  @ 1 apple/lane/s")
logger.info("")
ml_rmse_g10 = float(np.sqrt(np.mean([(r["ml"]-r["gt"])**2 for r in g10_detail if r["ml"]])))
lv_rmse_g10 = float(np.sqrt(np.mean([(r["live"]-r["gt"])**2 for r in g10_detail if r["live"]])))
logger.info(f"  ML   RMSE G10: {ml_rmse_g10:.3f}mm   (paper: 1.87mm)")
logger.info(f"  LIVE RMSE G10: {lv_rmse_g10:.3f}mm")
logger.info("")
w3_ml = sum(1 for r in g10_detail if r["ml"]   and abs(r["ml"]  -r["gt"])<=3)
w3_lv = sum(1 for r in g10_detail if r["live"] and abs(r["live"]-r["gt"])<=3)
w5p_ml= sum(1 for r in g10_detail if r["ml"]   and abs(r["ml"]  -r["gt"])/r["gt"]<=0.05)
w5p_lv= sum(1 for r in g10_detail if r["live"] and abs(r["live"]-r["gt"])/r["gt"]<=0.05)
n = len(g10_detail)
logger.info(f"  Within +-3mm:      ML={w3_ml}/{n}  LIVE={w3_lv}/{n}")
logger.info(f"  Within +-5% (rel): ML={w5p_ml}/{n}  LIVE={w5p_lv}/{n}")
logger.info("Done.")
