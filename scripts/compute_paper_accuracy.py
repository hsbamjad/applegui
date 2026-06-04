"""
Compute our sizing accuracy using the SAME formula as the reference paper:

    accuracy_i = (1 - |predicted_i - gt_i| / gt_i) * 100%
    mean_accuracy = mean(accuracy_i across all apples)

Also computes it for:
  - All training sessions (LOOCV per-session blind predictions)
  - G10 blind test specifically
  - LIVE estimate (pixel x scale) for comparison

Run from repo root:
    python scripts/compute_paper_accuracy.py
"""

import pickle, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
PKL_DIR   = r"D:\HA\apple_gui\data\frame_features"
MODEL_PATH= r"D:\HA\apple_gui\models\size_model.pkl"
SCALE     = 0.377        # mm/px for LIVE estimate
CENTRAL_FRAC = 0.60
ALPHA_TRIM   = 0.10

SESSIONS_TRAIN = ["G1","G2","G3","G4","G5","G6","G8","G9"]
SESSION_BLIND  = "G10"

# ── Feature builder (same as train_size_regressor.py) ─────────────────────────
def build_features(apple):
    frames = apple.get("frames", [])
    if not frames: return None
    cx = [f["cx_px"] for f in frames]
    lo = min(cx) + (max(cx)-min(cx))*(0.5-CENTRAL_FRAC/2)
    hi = min(cx) + (max(cx)-min(cx))*(0.5+CENTRAL_FRAC/2)
    central = [f for f in frames if lo<=f["cx_px"]<=hi] or frames
    if len(central) < 5: central = frames

    Q  = np.array([f.get("quality",0.5) for f in central])
    DA = np.array([f.get("d_area", 0.0) for f in central])
    DM = np.array([f.get("d_maxw", 0.0) for f in central])
    DS = np.array([f.get("d_sym",  0.0) for f in central])
    DE = np.array([f.get("d_ell",  0.0) for f in central])

    def wm(v): return float(np.dot(Q,v)/(Q.sum()+1e-9))
    def pk(v):
        lo,hi = np.percentile(v,ALPHA_TRIM*100), np.percentile(v,(1-ALPHA_TRIM)*100)
        k = v[(v>=lo)&(v<=hi)]; return float(k.mean()) if len(k) else float(v.mean())

    best = max(central, key=lambda f: f.get("quality",0))
    bb   = best.get("bbox",[0,0,10,10])
    bw,bh= bb[2]-bb[0], bb[3]-bb[1]

    return np.array([[
        wm(DA), wm(DM), wm(DS), wm(DE),
        pk(DA), pk(DM),
        max(bw,bh)/2.0, min(bw,bh)/2.0,
        float(Q.mean()),
        float(apple.get("lane",0)),
    ]])

def live_estimate(apple):
    """Quality-weighted mean of d_area * scale, all frames."""
    frames = apple.get("frames",[])
    if not frames: return None
    Q = np.array([f.get("quality",0.5) for f in frames])
    D = np.array([f.get("d_area",0.0)  for f in frames]) * SCALE
    return float(np.dot(Q,D)/(Q.sum()+1e-9))

# ── Paper accuracy formula ─────────────────────────────────────────────────────
def paper_accuracy(predicted, gt):
    """Returns per-apple accuracy and mean, exactly as Xu et al. 2024b / Lu 2025."""
    acc = [(1 - abs(p-g)/g)*100 for p,g in zip(predicted,gt) if g and g>0]
    return np.array(acc), float(np.mean(acc))

# ── Load model ────────────────────────────────────────────────────────────────
if not Path(MODEL_PATH).exists():
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(1)

with open(MODEL_PATH,"rb") as f: saved = pickle.load(f)
model = saved.get("model") or saved
print(f"Model loaded: {MODEL_PATH}\n")

# ── Helper: process one session ───────────────────────────────────────────────
def process_session(session):
    pkl = Path(PKL_DIR) / f"{session}.pkl"
    if not pkl.exists():
        print(f"  SKIP {session}: pkl not found")
        return None

    with open(pkl,"rb") as f: data = pickle.load(f)
    apples = data["apples"]

    rows = []
    for apple in apples:
        gt = apple.get("gt_mm")
        if not gt or gt <= 0: continue

        feat = build_features(apple)
        ml   = float(model.predict(feat)[0]) if feat is not None else None
        lv   = live_estimate(apple)

        rows.append({
            "session": session,
            "apple":   apple["apple_idx"]+1,
            "lane":    apple["lane"],
            "gt":      gt,
            "ml":      ml,
            "live":    lv,
        })
    return rows

# ── Run all sessions ──────────────────────────────────────────────────────────
print("="*68)
print(f"{'Session':<10} {'N':>4}  {'ML Acc%':>8}  {'ML RMSE':>8}  {'LIVE Acc%':>10}  {'LIVE RMSE':>10}")
print("="*68)

all_ml_gt, all_ml_pred = [], []
all_lv_gt, all_lv_pred = [], []
g10_results = None

for sess in SESSIONS_TRAIN + [SESSION_BLIND]:
    rows = process_session(sess)
    if not rows: continue

    gt_vals   = [r["gt"] for r in rows if r["ml"] is not None]
    ml_vals   = [r["ml"] for r in rows if r["ml"] is not None]
    lv_vals   = [r["live"] for r in rows if r["live"] is not None]
    lv_gt     = [r["gt"]  for r in rows if r["live"] is not None]

    ml_acc_arr, ml_acc   = paper_accuracy(ml_vals, gt_vals)
    lv_acc_arr, lv_acc   = paper_accuracy(lv_vals, lv_gt)
    ml_rmse  = float(np.sqrt(np.mean([(p-g)**2 for p,g in zip(ml_vals,gt_vals)])))
    lv_rmse  = float(np.sqrt(np.mean([(p-g)**2 for p,g in zip(lv_vals,lv_gt)])))

    tag = " ← BLIND" if sess == SESSION_BLIND else ""
    print(f"{sess+tag:<18} {len(gt_vals):>4}  {ml_acc:>8.2f}%  {ml_rmse:>8.3f}mm"
          f"  {lv_acc:>10.2f}%  {lv_rmse:>10.3f}mm")

    if sess != SESSION_BLIND:
        all_ml_gt.extend(gt_vals); all_ml_pred.extend(ml_vals)
        all_lv_gt.extend(lv_gt);   all_lv_pred.extend(lv_vals)
    else:
        g10_results = rows

print("-"*68)

# Overall training
if all_ml_gt:
    _, oa_ml = paper_accuracy(all_ml_pred, all_ml_gt)
    _, oa_lv = paper_accuracy(all_lv_pred, all_lv_gt)
    or_ml    = float(np.sqrt(np.mean([(p-g)**2 for p,g in zip(all_ml_pred,all_ml_gt)])))
    or_lv    = float(np.sqrt(np.mean([(p-g)**2 for p,g in zip(all_lv_pred,all_lv_gt)])))
    print(f"{'ALL TRAIN (N='+str(len(all_ml_gt))+')':<18} {len(all_ml_gt):>4}  "
          f"{oa_ml:>8.2f}%  {or_ml:>8.3f}mm  {oa_lv:>10.2f}%  {or_lv:>10.3f}mm")

print()
print("="*68)
print("DETAILED G10 BLIND TEST  (paper formula)")
print("="*68)
if g10_results:
    print(f"  {'#':>3}  {'L':>2}  {'GT':>6}  {'ML pred':>8}  {'ML acc%':>8}  "
          f"{'LIVE pred':>10}  {'LIVE acc%':>10}")
    print(f"  {'-'*62}")
    ml_accs, lv_accs = [], []
    for r in sorted(g10_results, key=lambda x: x["apple"]):
        ml  = r["ml"];  lv = r["live"]; gt = r["gt"]
        ma  = (1-abs(ml-gt)/gt)*100 if ml else None
        la  = (1-abs(lv-gt)/gt)*100 if lv else None
        ml_s= f"{ml:8.2f}" if ml else "      --"
        lv_s= f"{lv:10.2f}" if lv else "        --"
        ma_s= f"{ma:8.2f}%" if ma else "      --"
        la_s= f"{la:10.2f}%" if la else "        --"
        flag= " ✗" if (ma and ma < 95) else ""
        print(f"  #{r['apple']:2d}  L{r['lane']}  {gt:>6.1f}  {ml_s}  {ma_s}{flag}  {lv_s}  {la_s}")
        if ma: ml_accs.append(ma)
        if la: lv_accs.append(la)

    print(f"  {'-'*62}")
    print(f"  Mean ML  accuracy (paper formula): {np.mean(ml_accs):.2f}%")
    print(f"  Mean LIVE accuracy (paper formula): {np.mean(lv_accs):.2f}%")
    print(f"  Paper reported:                    97.60%  (at 1 apple/lane/s)")
    print()
    print(f"  Within ±3mm  ML:   {sum(1 for r in g10_results if r['ml'] and abs(r['ml']-r['gt'])<=3)}/{len(g10_results)}")
    print(f"  Within ±3mm  LIVE: {sum(1 for r in g10_results if r['live'] and abs(r['live']-r['gt'])<=3)}/{len(g10_results)}")

print("\nDone.")
