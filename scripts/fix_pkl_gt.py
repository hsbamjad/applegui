"""
fix_pkl_gt.py  —  Post-process pkl files to correct GT assignment
==================================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

PURPOSE
-------
The initial extract_frames.py run stored incorrect lane labels and gt_mm
values due to a buggy column-detection algorithm.

This script reloads each session pkl, recomputes:
  - mean_cy per apple (from stored frame cy_px values)
  - correct lane_id (0=top, 1=mid, 2=bot) from y-centroid
  - correct GT match via interleaved lane ordering

Then overwrites the pkl files in-place with corrected gt_mm, lane, pos_in_lane.
No YOLO re-inference needed.

USAGE
-----
python scripts/fix_pkl_gt.py ^
    --pkl_dir  "D:\\HA\\apple_gui\\data\\frame_features" ^
    --gt_path  "D:\\HA\\apple_gui\\data\\gt.xlsx" ^
    --img_h    1536
"""

import argparse
import pickle
import numpy as np
import openpyxl
from pathlib import Path

APPLES_PER_SESSION = 18
LANES              = 3


# ─────────────────────────────────────────────────────────────────────────────
# GT LOADER  (same as extract_frames.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_gt(gt_path: str, session: str) -> list:
    wb = openpyxl.load_workbook(gt_path, data_only=True)
    ws = wb.active
    values = []
    in_session = False
    for row in ws.iter_rows(min_row=2, values_only=True):
        col_a, col_b, d1, d2, *_ = row
        if col_a is not None:
            if str(col_a).strip().upper() == session.upper():
                in_session = True
            else:
                if in_session:
                    break
                in_session = False
        if not in_session:
            continue
        try:
            gt_mm = (float(d1) + float(d2)) / 2.0
        except (TypeError, ValueError):
            gt_mm = None
        values.append(gt_mm)
    if not values:
        return [None] * APPLES_PER_SESSION
    values += [None] * max(0, APPLES_PER_SESSION - len(values))
    return values[:APPLES_PER_SESSION]


# ─────────────────────────────────────────────────────────────────────────────
# REASSIGN GT using y-centroid lane detection
# ─────────────────────────────────────────────────────────────────────────────
def reassign_gt(apples: list, gt_list: list, img_h: int) -> list:
    """
    Recompute lane assignment from cy_px and reassign gt_mm.
    Input `apples` is the raw list from the pkl (any lane labeling, may be wrong).
    Returns a new list with corrected lane, pos_in_lane, apple_idx, gt_mm.
    """
    lane_h = img_h / LANES

    # Compute mean_cy and lane_id for each apple from its stored frame data
    enriched = []
    for a in apples:
        cy_values = [f["cy_px"] for f in a["frames"] if "cy_px" in f]
        mean_cy = float(np.mean(cy_values)) if cy_values else img_h / 2
        lane_id = min(LANES - 1, int(mean_cy / lane_h))
        enriched.append({
            "original": a,
            "mean_cy": mean_cy,
            "lane_id": lane_id,
            "exit_frame_idx": a.get("exit_frame_idx", 0) or 0,
        })

    # Bucket into lanes, sort within each lane by exit frame
    lanes = [[] for _ in range(LANES)]
    for e in enriched:
        lanes[e["lane_id"]].append(e)
    for lane in lanes:
        lane.sort(key=lambda e: e["exit_frame_idx"])

    # Interleave: lane0[0], lane1[0], lane2[0], lane0[1], lane1[1], lane2[1]...
    max_per_lane = max(len(lane) for lane in lanes) if any(lanes) else 0
    corrected = []
    apple_idx = 0
    for pos in range(max_per_lane):
        for lane_id in range(LANES):
            if pos >= len(lanes[lane_id]):
                continue
            e = lanes[lane_id][pos]
            gt_mm = gt_list[apple_idx] if apple_idx < len(gt_list) else None

            # Build corrected apple dict (keep all original frame data)
            new_apple = dict(e["original"])   # copy all original fields
            new_apple["apple_idx"]      = apple_idx
            new_apple["gt_mm"]          = gt_mm
            new_apple["lane"]           = lane_id
            new_apple["pos_in_lane"]    = pos
            new_apple["mean_cy_px"]     = e["mean_cy"]    # store for reference
            corrected.append(new_apple)
            apple_idx += 1

    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fix GT assignment in session pkl files (no YOLO re-inference)"
    )
    parser.add_argument("--pkl_dir", required=True,
                        help="Directory containing G1.pkl ... G11.pkl")
    parser.add_argument("--gt_path", required=True,
                        help="Path to gt.xlsx")
    parser.add_argument("--img_h",   type=int, default=1536,
                        help="Image height in pixels (default 1536)")
    parser.add_argument("--sessions", nargs="+",
                        default=[f"G{i}" for i in range(1, 12)],
                        help="Sessions to fix e.g. G1 G2 G10")
    args = parser.parse_args()

    pkl_dir = Path(args.pkl_dir)

    for session in args.sessions:
        pkl_path = pkl_dir / f"{session}.pkl"
        if not pkl_path.exists():
            print(f"  ✗ {pkl_path.name} not found — skipping")
            continue

        print(f"\nFixing {session}...")

        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        gt_list = load_gt(args.gt_path, session)
        old_apples = payload["apples"]
        new_apples = reassign_gt(old_apples, gt_list, args.img_h)

        # Print corrected assignment
        for a in new_apples:
            gt_str = f"{a['gt_mm']:.1f}mm" if a["gt_mm"] is not None else "None"
            print(f"  Apple {a['apple_idx']+1:2d}  "
                  f"lane={a['lane']} pos={a['pos_in_lane']}  "
                  f"frames={len(a['frames']):3d}  gt={gt_str}")

        # Overwrite pkl with corrected apples
        payload["apples"] = new_apples
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)

        print(f"  ✔ {pkl_path.name} corrected  "
              f"({len(old_apples)} → {len(new_apples)} apples)")

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
