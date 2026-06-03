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
# REASSIGN GT  —  Entry-gate ordering
# ─────────────────────────────────────────────────────────────────────────────
def reassign_gt(apples: list, gt_list: list, img_h: int) -> list:
    """
    Recompute lane + GT assignment from stored frame data.

    Strategy: ENTRY-GATE assignment
    --------------------------------
    The apple's GT number is determined the moment it first enters the
    camera view (frames[0]["frame_idx"]).  Each lane is sorted by that
    entry time.  Lanes are then interleaved positionally to recover the
    physical numbering on the GT sheet:

        pos 0: lane0, lane1, lane2   →  GT indices 0, 1, 2
        pos 1: lane0, lane1, lane2   →  GT indices 3, 4, 5
        …

    This is robust against apples that slow down, speed up, or fall off
    the conveyor after they have already been seen at the entry gate.
    No gap detection or inter-apple timing heuristics are needed.
    """
    LANES              = 3
    APPLES_PER_SESSION = 18
    EXPECTED_PER_LANE  = APPLES_PER_SESSION // LANES   # 6
    lane_h = img_h / LANES

    # ── Build stub objects ────────────────────────────────────────────────────
    class Stub:
        def __init__(self, apple_dict):
            cy_vals = [f["cy_px"] for f in apple_dict["frames"] if "cy_px" in f]
            self.mean_cy       = float(np.mean(cy_vals)) if cy_vals else img_h / 2
            self.lane_id       = min(LANES - 1, int(self.mean_cy / lane_h))
            # Entry time = first frame this apple was ever seen
            self.entry_frame   = apple_dict["frames"][0]["frame_idx"] \
                                 if apple_dict["frames"] else 0
            self.exit_frame_idx = apple_dict.get("exit_frame_idx", 0) or 0
            self._data         = apple_dict

    stubs = [Stub(a) for a in apples]

    # ── Bucket into lanes, sort by ENTRY time (not exit) ─────────────────────
    lanes = [[] for _ in range(LANES)]
    for s in stubs:
        lanes[s.lane_id].append(s)
    for lane in lanes:
        lane.sort(key=lambda s: s.entry_frame)

    # ── Print entry-gate order for each lane ─────────────────────────────────
    print("  Entry-gate ordering (sorted by first-seen frame):")
    for lid, lane in enumerate(lanes):
        entries = [s.entry_frame for s in lane]
        print(f"    Lane {lid}: {len(lane)} apple(s)  entry frames={entries}")

    # ── Build positional slots (no gap detection — just sequential) ───────────
    # Each lane occupies exactly EXPECTED_PER_LANE slots.
    # Apples present fill from pos 0; any remaining slots are None (missing).
    expanded = []
    for lane in lanes:
        slots = list(lane)                        # already in entry order
        while len(slots) < EXPECTED_PER_LANE:
            slots.append(None)                    # trailing slot = apple missing
        expanded.append(slots[:EXPECTED_PER_LANE])

    # ── Print per-lane summary ────────────────────────────────────────────────
    for lid, slots in enumerate(expanded):
        n_ok   = sum(1 for s in slots if s is not None)
        n_miss = sum(1 for s in slots if s is None)
        print(f"  Lane {lid}: {n_ok} present, {n_miss} missing  (slots={len(slots)})")

    # ── Interleave and assign GT ──────────────────────────────────────────────
    corrected = []
    apple_idx = 0
    for pos in range(EXPECTED_PER_LANE):
        for lane_id in range(LANES):
            slot = expanded[lane_id][pos]
            if slot is None:
                apple_idx += 1          # consume GT slot — apple absent here
            else:
                gt_mm = gt_list[apple_idx] if apple_idx < len(gt_list) else None
                new_apple = dict(slot._data)
                new_apple["apple_idx"]   = apple_idx
                new_apple["gt_mm"]       = gt_mm
                new_apple["lane"]        = lane_id
                new_apple["pos_in_lane"] = pos
                new_apple["mean_cy_px"]  = slot.mean_cy
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
