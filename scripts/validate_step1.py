"""
scripts/validate_step1.py  —  Step 1 Quality Audit
====================================================
Loads all session pkls and validates:
  1. Apple count (must be exactly 18 per session)
  2. Lane balance (must be 6 per lane)
  3. GT assignment (non-zero, reasonable range 50-85mm)
  4. cx_range (flags apples with < 1000px travel = partial traversal)
  5. n_frames (flags apples with < 400 frames)
  6. Q score (flags apples with Q < 0.85)
"""

import sys, os, pickle
import numpy as np
from pathlib import Path

from core.log import get_logger, configure_root
configure_root()
logger = get_logger(__name__)

PKL_DIR  = r"S:\MSU_Research\ASABE AIM26\apple_gui\data\frame_features"
SESSIONS = ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11"]

MIN_CX_RANGE = 1000
MIN_FRAMES   = 400
MIN_Q        = 0.85
GT_MIN       = 48.0
GT_MAX       = 90.0

warnings = []

logger.info("=" * 100)
logger.info(f"{'STEP 1 VALIDATION':^100}")
logger.info("=" * 100)

all_rows = []

for sess in SESSIONS:
    pkl_path = Path(PKL_DIR) / f"{sess}.pkl"
    if not pkl_path.exists():
        logger.warning(f"[MISSING] {pkl_path}")
        warnings.append(f"{sess}: PKL FILE MISSING")
        continue

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    apples = sorted(data["apples"], key=lambda a: a["apple_idx"])
    n      = len(apples)

    lane_counts = {0: 0, 1: 0, 2: 0}
    for a in apples:
        lane_counts[a["lane"]] = lane_counts.get(a["lane"], 0) + 1

    logger.info(f"{'─'*100}")
    logger.info(f"  {sess}  |  {n} apples  |  "
                f"Lane0={lane_counts.get(0,0)}  Lane1={lane_counts.get(1,0)}  Lane2={lane_counts.get(2,0)}")
    logger.info(f"{'─'*100}")
    logger.info(f"  {'#':>3}  {'Lane':>4}  {'Pos':>3}  {'GT mm':>7}  {'Frames':>7}  "
                f"{'cx_range':>9}  {'Q':>6}  {'ell_a':>7}  {'ell_b':>7}  Status")
    logger.info(f"  {'-'*95}")

    sess_warnings = []

    for a in apples:
        idx      = a["apple_idx"] + 1
        lane     = a["lane"]
        pos      = a["pos_in_lane"]
        gt       = a.get("gt_mm")
        n_frames = len(a.get("frames", []))
        q        = a.get("best_quality", 0)
        cp       = a.get("consensus_params")
        ell_a    = cp["axis_a"] if cp else 0.0
        ell_b    = cp["axis_b"] if cp else 0.0

        # cx_range: stored directly or compute from frames
        if "cx_range" in a:
            cx_range = int(a["cx_range"])
        else:
            cxs = [f["cx_px"] for f in a.get("frames", [])]
            cx_range = int(max(cxs) - min(cxs)) if cxs else 0

        flags = []
        if gt is None:
            flags.append("NO_GT")
        elif gt < GT_MIN or gt > GT_MAX:
            flags.append(f"GT_OOB({gt:.1f}mm)")
        if cx_range < MIN_CX_RANGE:
            flags.append(f"LOW_CX({cx_range}px)")
        if n_frames < MIN_FRAMES:
            flags.append(f"FEW_FRAMES({n_frames})")
        if q < MIN_Q:
            flags.append(f"LOW_Q({q:.3f})")

        status = "  ".join(flags) if flags else "OK"
        marker = "  <==WARNING" if flags else ""
        gt_str = f"{gt:7.1f}" if gt is not None else "    N/A"

        logger.info(f"  #{idx:2d}  lane={lane}  pos={pos}  {gt_str}mm  "
                    f"{n_frames:7d}  {cx_range:9d}px  {q:6.3f}  "
                    f"{ell_a:7.1f}  {ell_b:7.1f}  {status}{marker}")

        for flag in flags:
            w = f"{sess} Apple#{idx} (L{lane}P{pos}): {flag}"
            sess_warnings.append(w)
            warnings.append(w)

        all_rows.append({
            "session": sess, "apple_idx": idx, "lane": lane, "pos": pos,
            "gt_mm": gt, "n_frames": n_frames, "cx_range": cx_range,
            "Q": q, "ell_a": ell_a, "flags": flags
        })

    if n != 18:
        w = f"{sess}: WRONG APPLE COUNT = {n} (expected 18)"
        warnings.append(w)
        logger.warning(f"WARNING: {w}")
    if any(v != 6 for v in lane_counts.values()):
        w = f"{sess}: UNBALANCED LANES = {lane_counts}"
        warnings.append(w)
        logger.warning(f"WARNING: {w}")

    gts = [a.get("gt_mm") for a in apples if a.get("gt_mm") is not None]
    if gts:
        logger.info(f"  GT range: {min(gts):.1f} to {max(gts):.1f} mm  "
                    f"|  mean={np.mean(gts):.1f}mm  std={np.std(gts):.1f}mm")

    if sess_warnings:
        logger.info(f"  Warnings this session: {len(sess_warnings)}")
    else:
        logger.info(f"  All 18 apples: OK")

# ── Cross-session summary ─────────────────────────────────────────────────────
logger.info(f"\n{'='*100}")
logger.info(f"{'CROSS-SESSION SUMMARY':^100}")
logger.info(f"{'='*100}")

gts_all = [r["gt_mm"] for r in all_rows if r["gt_mm"] is not None]
qall    = [r["Q"]        for r in all_rows]
cxall   = [r["cx_range"] for r in all_rows]
ffall   = [r["n_frames"] for r in all_rows]

logger.info(f"  Sessions processed : {len(SESSIONS)}")
logger.info(f"  Total apples       : {len(all_rows)}")
logger.info(f"  GT range           : {min(gts_all):.1f} to {max(gts_all):.1f} mm")
logger.info(f"  GT mean +/- std    : {np.mean(gts_all):.1f} +/- {np.std(gts_all):.1f} mm")
logger.info(f"  Q mean +/- std     : {np.mean(qall):.3f} +/- {np.std(qall):.3f}")
logger.info(f"  cx_range mean      : {np.mean(cxall):.0f} px  (min={min(cxall)}px)")
logger.info(f"  frames mean        : {np.mean(ffall):.0f}  (min={min(ffall)})")

logger.info(f"{'─'*100}")
logger.info(f"  WARNINGS ({len(warnings)} total):")
logger.info(f"{'─'*100}")
if warnings:
    for w in warnings:
        logger.warning(f"WARNING: {w}")
else:
    logger.info("  None -- all sessions clean")

# ── GT table ─────────────────────────────────────────────────────────────────
logger.info(f"{'─'*100}")
logger.info(f"  GT VALUES PER APPLE PER SESSION (mm):")
logger.info(f"{'─'*100}")
logger.info(f"  {'Apple':>7}  " + "  ".join(f"{s:>7}" for s in SESSIONS))
logger.info(f"  {'-'*92}")
for apple_no in range(1, 19):
    row_vals = []
    for sess in SESSIONS:
        match = [r for r in all_rows
                 if r["session"] == sess and r["apple_idx"] == apple_no]
        val = f"{match[0]['gt_mm']:7.1f}" if (match and match[0]["gt_mm"] is not None) else "    N/A"
        row_vals.append(val)
    logger.info(f"  #{apple_no:6d}  " + "  ".join(row_vals))

logger.info(f"{'='*100}")
logger.info("Validation complete.")
