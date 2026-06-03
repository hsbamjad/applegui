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
    Assign GT using entry-time (pass-line) ordering.

    Each apple's pass_time = frame_idx of its FIRST detected frame
    (approx. when it crosses the 35% entry threshold in extract_frames.py).

    Apples placed in the same physical column enter the FOV almost
    simultaneously → they form tight clusters in pass_time space.
    WITHIN_COL_GAP separates within-column spread from between-column gap.

    Algorithm:
      1. Compute pass_time for each apple
      2. Sort all apples by pass_time
      3. Group into column events (gap <= WITHIN_COL_GAP frames = same column)
      4. Within each column, identify which lanes (0/1/2) are present
      5. Missing lane in a column → consume GT slot, no apple assigned
      6. GT index = col * 3 + lane_id
    """
    LANES            = 3
    WITHIN_COL_GAP   = 60    # frames — apples in same column cross entry line within this
    lane_h = img_h / LANES

    class Apple:
        def __init__(self, d):
            self._data    = d
            # Entry time: first frame record's frame_idx
            self.pass_time = d["frames"][0]["frame_idx"] if d["frames"] else 0
            cy_vals        = [f["cy_px"] for f in d["frames"] if "cy_px" in f]
            self.mean_cy   = float(np.mean(cy_vals)) if cy_vals else img_h / 2
            self.lane_id   = min(LANES - 1, int(self.mean_cy / lane_h))

    apples_obj = [Apple(a) for a in apples]
    apples_obj.sort(key=lambda a: a.pass_time)

    if not apples_obj:
        return []

    # Group into column events by entry-time proximity
    col_events = []
    current_col = [apples_obj[0]]
    for apple in apples_obj[1:]:
        gap = apple.pass_time - current_col[-1].pass_time
        if gap <= WITHIN_COL_GAP:
            current_col.append(apple)
        else:
            col_events.append(current_col)
            current_col = [apple]
    col_events.append(current_col)

    print(f"  Entry-time columns detected: {len(col_events)}")
    for ci, col in enumerate(col_events):
        lanes_str = ",".join(str(a.lane_id) for a in sorted(col, key=lambda a: a.lane_id))
        t0 = col[0].pass_time
        print(f"    Col {ci}: lanes=[{lanes_str}]  pass_time={t0}")

    # Assign GT: for each column event, check each lane 0→1→2
    result = []
    apple_idx = 0
    for col_idx, col in enumerate(col_events):
        lanes_in_col = {}
        for a in col:
            # If two apples land in same lane (shouldn't happen), keep first-entry one
            if a.lane_id not in lanes_in_col:
                lanes_in_col[a.lane_id] = a

        for lane_id in range(LANES):
            if apple_idx >= len(gt_list) + LANES:  # safety stop
                break
            if lane_id not in lanes_in_col:
                apple_idx += 1   # consume GT slot — apple missing in this lane/column
            else:
                gt_mm = gt_list[apple_idx] if apple_idx < len(gt_list) else None
                a = lanes_in_col[lane_id]
                new_apple = dict(a._data)
                new_apple["apple_idx"]   = apple_idx
                new_apple["gt_mm"]       = gt_mm
                new_apple["lane"]        = lane_id
                new_apple["pos_in_lane"] = col_idx
                new_apple["mean_cy_px"]  = a.mean_cy
                new_apple["pass_time"]   = a.pass_time
                result.append(new_apple)
                apple_idx += 1

    return result




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
