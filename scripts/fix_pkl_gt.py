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
    Recompute lane + GT assignment from stored frame data.
    Uses robust gap detection — handles missing apples at START, MIDDLE, END.
    """
    LANES = 3
    GAP_MULTIPLIER = 1.6
    lane_h = img_h / LANES

    # Build stub objects with exit_frame_idx, mean_cy, lane_id, and original dict
    class Stub:
        def __init__(self, apple_dict):
            self.exit_frame_idx = apple_dict.get("exit_frame_idx", 0) or 0
            cy_vals = [f["cy_px"] for f in apple_dict["frames"] if "cy_px" in f]
            self.mean_cy = float(np.mean(cy_vals)) if cy_vals else img_h / 2
            self.lane_id = min(LANES - 1, int(self.mean_cy / lane_h))
            self._data = apple_dict

    stubs = [Stub(a) for a in apples]

    # Bucket into lanes, sort by exit time
    lanes = [[] for _ in range(LANES)]
    for s in stubs:
        lanes[s.lane_id].append(s)
    for lane in lanes:
        lane.sort(key=lambda s: s.exit_frame_idx)

    # Compute typical inter-apple gap (cross-lane median)
    all_gaps = []
    for lane in lanes:
        exits = [s.exit_frame_idx for s in lane]
        for i in range(1, len(exits)):
            all_gaps.append(exits[i] - exits[i - 1])
    typical_gap = float(np.median(all_gaps)) if all_gaps else 300.0
    print(f"  Typical inter-apple gap: {typical_gap:.0f} frames")

    # Expand each lane — insert None for mid/end missing apples
    def expand_lane(tracks):
        if not tracks:
            return []
        slots = [tracks[0]]
        for i in range(1, len(tracks)):
            gap = tracks[i].exit_frame_idx - tracks[i - 1].exit_frame_idx
            n_missing = max(0, round(gap / typical_gap) - 1)
            slots.extend([None] * n_missing)
            slots.append(tracks[i])
        return slots

    expanded = [expand_lane(lane) for lane in lanes]

    # Detect leading missing apples (compare first exits across lanes)
    first_exits = [lane[0].exit_frame_idx for lane in lanes if lane]
    if first_exits:
        earliest = min(first_exits)
        for lane_id, lane in enumerate(lanes):
            if not lane:
                continue
            first_t = lane[0].exit_frame_idx
            n_leading = max(0, round((first_t - earliest) / typical_gap))
            if n_leading > 0:
                expanded[lane_id] = [None] * n_leading + expanded[lane_id]
                print(f"  Lane {lane_id}: {n_leading} missing apple(s) at start")

    # Pad all lanes to same length
    max_len = max((len(e) for e in expanded), default=0)
    for e in expanded:
        while len(e) < max_len:
            e.append(None)

    # Print per-lane summary
    for lid, slots in enumerate(expanded):
        n_ok   = sum(1 for s in slots if s is not None)
        n_miss = sum(1 for s in slots if s is None)
        print(f"  Lane {lid}: {n_ok} present, {n_miss} missing  (slots={len(slots)})")

    # Interleave and assign GT
    corrected = []
    apple_idx = 0
    for pos in range(max_len):
        for lane_id in range(LANES):
            slot = expanded[lane_id][pos] if pos < len(expanded[lane_id]) else None
            if slot is None:
                apple_idx += 1    # consume GT slot for missing apple
            else:
                gt_mm = gt_list[apple_idx] if apple_idx < len(gt_list) else None
                new_apple = dict(slot._data)
                new_apple["apple_idx"]      = apple_idx
                new_apple["gt_mm"]          = gt_mm
                new_apple["lane"]           = lane_id
                new_apple["pos_in_lane"]    = pos
                new_apple["mean_cy_px"]     = slot.mean_cy
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
