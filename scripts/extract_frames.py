"""
extract_frames.py  —  Step 1: Frame Feature Extractor
=======================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

PURPOSE
-------
Run the YOLO segmentation model on all G1-G11 sessions frame by frame.
For each committed apple track, collect:
  - Binary mask per frame (from YOLO segmentation)
  - Bounding box per frame
  - Frame index and frame number
  - Center x position

Match each committed apple to the GT caliper measurement by arrival order
(first apple to exit = apple 1 in GT, etc.).

Output: one pickle file per session saved to OUTPUT_DIR.

USAGE (on shuttle PC)
---------------------
conda run -n applegui python scripts/extract_frames.py \
    --data_root "D:/HA/apple_gui/images" \
    --model_path "D:/HA/apple_gui/models/best.pt" \
    --gt_path   "D:/HA/apple_gui/data/gt.xlsx" \
    --output_dir "D:/HA/apple_gui/data/frame_features" \
    --sessions G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 G11

    If running from this dev machine sample data:
    --data_root "D:/Haseeb/pic"

OUTPUT FORMAT (pickle)
----------------------
sessions/{session_name}.pkl  contains:
{
  "session": "G1",
  "apples": [
      {
          "apple_idx":  0,          # 0-based, order of exit from frame
          "gt_mm":      73.2,       # caliper measurement from GT (None if unmatched)
          "frames": [
              {
                  "frame_no":   1042,         # original BMP frame number
                  "frame_idx":  0,            # 0-based index within session
                  "cx_px":      812,          # centroid x in full-res image
                  "cy_px":      400,          # centroid y in full-res image
                  "bbox":       [x1,y1,x2,y2],# in full-res pixels
                  "crop_rect":  [mx1,my1,mx2,my2], # bbox + MASK_PAD in full-res pixels
                  "mask":       np.ndarray,   # binary uint8, cropped to crop_rect size
                                              # NOT full-frame — use crop_rect to locate
                  "conf":       0.87,         # YOLO detection confidence
                  "class_id":   1,            # 0=Cull 1=Fresh 2=Processing
              },
              ...
          ]
      },
      ...
  ]
}
"""

import argparse
import os
import re
import pickle
import numpy as np
import cv2
import openpyxl
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PATHS
#   Dev machine  : D:\Haseeb\pic  /  D:\Haseeb\Ground Truth.xlsx
#   Shuttle PC   : D:\HA\apple_gui\images  /  D:\HA\apple_gui\data\gt.xlsx
#   Override with --data_root, --gt_path, --model_path, --output_dir flags.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT  = r"D:\Haseeb\pic"
DEFAULT_MODEL_PATH = r"D:\HA\apple_gui\models\best.pt"   # no model on dev machine
DEFAULT_GT_PATH    = r"D:\Haseeb\Ground Truth.xlsx"
DEFAULT_OUTPUT_DIR = r"D:\Haseeb\frame_features"

APPLES_PER_SESSION = 18   # Each session has 18 apples
CONF_THRESHOLD     = 0.40
IOU_THRESHOLD      = 0.45
INPUT_MODE         = "RB-nir1"  # Channel composite: R+B from Source0, NIR1 from Source1
MASK_PAD           = 20         # Pixels of padding around bbox when cropping mask
                                 # Crop mask stored instead of full-frame mask (~60x smaller)

# Tracker settings (mirror config.yaml)
ENTRY_FRAC         = 0.35   # Entry-gate line: apples registered here (left 35% of frame)
EXIT_FRAC          = 0.85   # Apple finalized when centroid crosses this x-fraction
MIN_FRAMES         = 5      # Minimum frames before a track is registered at entry gate
MAX_LOST           = 10     # Frames before track is dropped
MAX_RECOVER_DIST   = 80     # Pixels — max movement between frames for same track
STARTUP_FRAMES     = 400    # First N frames: track apples anywhere in frame
                             # (catches apples already on conveyor when recording starts)
MIN_CX_TRAVEL      = 200    # Min pixels a track's centroid must travel (left→right)
                             # across all its frames.  Stationary background detections
                             # (smudges, reflections) score 0 and are discarded.


# ─────────────────────────────────────────────────────────────────────────────
# NATURAL SORT
# ─────────────────────────────────────────────────────────────────────────────
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE BUILDER
# Builds 3-channel image for YOLO input from 3 source BMPs.
# Mode RB-nir1 → [R channel, B channel, NIR1 channel]
# ─────────────────────────────────────────────────────────────────────────────
def build_composite(src0_path: str, src1_path: str, mode: str = "RB-nir1") -> np.ndarray:
    """Return H×W×3 uint8 composite suitable for YOLO."""
    img0 = cv2.imread(src0_path)          # BGR from Source0 (color)
    img1 = cv2.imread(src1_path)          # read Source1 as-is (may be mono or BGR)

    if img0 is None:
        raise FileNotFoundError(f"Cannot read Source0 image: {src0_path}")
    if img1 is None:
        raise FileNotFoundError(f"Cannot read Source1 image: {src1_path}")

    # Ensure img0 is 3-channel
    if img0.ndim == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)

    # Ensure img1 is single-channel grayscale
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    h, w = img0.shape[:2]

    # Safety: resize img1 if dimensions don't match (rare frame corruption)
    if img1.shape != (h, w):
        img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LINEAR)

    if mode == "RB-nir1":
        # R from Source0, B from Source0, NIR1 from Source1
        r = img0[:, :, 2]   # OpenCV BGR → index 2 is R
        b = img0[:, :, 0]   # index 0 is B
        composite = np.stack([r, b, img1], axis=2)
    elif mode == "RGB":
        composite = img0
    else:
        raise ValueError(f"Unknown input_mode: {mode}")

    return composite


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE IoU TRACKER
# Lightweight tracker that assigns stable IDs to apples across frames.
# Does not use Kalman — uses centroid distance + IoU matching.
# ─────────────────────────────────────────────────────────────────────────────
def iou(a, b):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return inter / (ua + ub - inter)



class Track:
    def __init__(self, tid: int, frame_record: dict, img_w: int,
                 is_startup: bool = False):
        self.tid             = tid
        self.lost            = 0
        self.age             = 1
        self.is_startup      = is_startup  # True if created during STARTUP_FRAMES
        self.img_w           = img_w
        self.frames          = [frame_record]
        self.last_bbox       = frame_record["bbox"]
        self.last_cx         = frame_record["cx_px"]
        # ── Two-phase lifecycle ──────────────────────────────────────────────
        # Phase 1: REGISTERED at entry gate after MIN_FRAMES → gets GT slot
        #          even if the apple later falls off or is lost mid-conveyor
        # Phase 2: COMMITTED at exit → frame collection finalized
        self.registered      = False  # Phase 1 complete
        self.committed       = False  # Phase 2 complete (exit reached)
        self.exit_frame_idx  = None   # set at exit OR end-of-session flush

    def update(self, frame_record: dict):
        self.frames.append(frame_record)
        self.last_bbox = frame_record["bbox"]
        self.last_cx   = frame_record["cx_px"]
        self.lost = 0
        self.age += 1

    @property
    def mean_cy(self) -> float:
        """Average y-centroid across all tracked frames — determines lane."""
        return float(np.mean([f["cy_px"] for f in self.frames]))

    @property
    def cx_frac(self):
        return self.last_cx / self.img_w

    def should_register(self) -> bool:
        """Phase 1: register at entry gate after MIN_FRAMES seen."""
        return not self.registered and self.age >= MIN_FRAMES

    def should_commit(self) -> bool:
        """Phase 2: finalize frame collection when apple reaches exit zone."""
        return (self.registered
                and not self.committed
                and self.cx_frac >= EXIT_FRAC)


class SimpleTracker:
    def __init__(self, img_w: int, img_h: int):
        self.img_w      = img_w
        self.img_h      = img_h
        self.tracks     = []   # currently active tracks
        self.registered = []   # tracks registered at entry gate (in registration order)
        self.committed  = []   # alias: same objects, finalized at exit
        self._next_id   = 0

    def update(self, detections: list, frame_idx: int) -> list:
        """
        detections: list of dicts with keys bbox, cx_px, cy_px, mask, conf, class_id
        frame_idx:  current frame index (used to record entry/exit timing)
        Returns list of newly registered tracks this frame.
        """
        # ── Age out lost tracks ───────────────────────────────────────────────
        for t in self.tracks:
            t.lost += 1
        # Finalise registered tracks that are about to be dropped.
        # An apple that fell off the conveyor mid-way is registered at the
        # entry gate (Phase 1 complete) but never reaches EXIT_FRAC (Phase 2).
        # Stamp exit_frame_idx now so the pkl records the last-seen frame
        # instead of None.  last-seen ≈ frame_idx - t.lost (lost counts back).
        for t in self.tracks:
            if t.lost > MAX_LOST and t.registered and t.exit_frame_idx is None:
                t.exit_frame_idx = max(0, frame_idx - t.lost)
        self.tracks = [t for t in self.tracks if t.lost <= MAX_LOST]

        # ── Match detections to tracks by IoU ─────────────────────────────────
        matched_track_ids = set()
        matched_det_ids   = set()

        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    iou_matrix[i, j] = iou(t.last_bbox, d["bbox"])

            # Greedy match (highest IoU first, threshold 0.3)
            flat = [(iou_matrix[i, j], i, j)
                    for i in range(len(self.tracks))
                    for j in range(len(detections))]
            flat.sort(key=lambda x: -x[0])
            for score, i, j in flat:
                if score < 0.3:
                    break
                if i in matched_track_ids or j in matched_det_ids:
                    continue
                cx_det = detections[j]["cx_px"]
                if abs(cx_det - self.tracks[i].last_cx) > MAX_RECOVER_DIST:
                    continue
                self.tracks[i].update(detections[j])
                self.tracks[i].lost = 0
                matched_track_ids.add(i)
                matched_det_ids.add(j)

        # ── Create new tracks for unmatched detections ────────────────────────
        # During STARTUP_FRAMES: accept apples anywhere in frame
        #   (catches apples already on conveyor at session start)
        # After startup: only accept apples entering from the entry-gate zone
        in_startup = frame_idx < STARTUP_FRAMES
        for j, d in enumerate(detections):
            if j in matched_det_ids:
                continue
            cx_frac = d["cx_px"] / self.img_w
            if in_startup or cx_frac < ENTRY_FRAC:
                t = Track(self._next_id, d, self.img_w, is_startup=in_startup)
                self._next_id += 1
                self.tracks.append(t)

        # ── Phase 1: Register tracks at entry gate ────────────────────────────
        # An apple is registered (gets its GT slot) as soon as it has been
        # seen for MIN_FRAMES.  After registration it stays active so we
        # keep collecting frame data until it exits or is lost.
        newly_registered = []
        for t in self.tracks:
            if t.should_register():
                t.registered = True
                self.registered.append(t)
                newly_registered.append(t)

        # ── Phase 2: Finalize tracks that reached the exit zone ───────────────
        for t in self.tracks:
            if t.should_commit():
                t.committed = True
                t.exit_frame_idx = frame_idx

        return newly_registered


# ─────────────────────────────────────────────────────────────────────────────
# GT LOADER
# GT Excel structure (confirmed):
#   Col A: Session label (G1..G11) — only in first row of each session, else None
#   Col B: Apple index within session (1..18)
#   Col C: D1 — first caliper measurement (mm)
#   Col D: D2 — second caliper measurement (mm)
#   Col E: Surface Class (grade label)
#   Col F: Average formula  →  we compute (D1+D2)/2 directly
# ─────────────────────────────────────────────────────────────────────────────
def load_gt(gt_path: str, session: str) -> list:
    """
    Load GT caliper measurements for a session.
    Returns list of APPLES_PER_SESSION floats (D1+D2)/2, in apple order.
    Uses openpyxl with data_only=True to read computed values.
    """
    wb = openpyxl.load_workbook(gt_path, data_only=True)
    ws = wb.active

    values = []
    in_session = False

    for row in ws.iter_rows(min_row=2, values_only=True):  # skip header
        col_a, col_b, d1, d2, *_ = row

        # Session label appears only in the first row of each session
        if col_a is not None:
            if str(col_a).strip().upper() == session.upper():
                in_session = True
            else:
                if in_session:
                    break   # moved to next session — done
                in_session = False

        if not in_session:
            continue

        # Compute average of the two measurements
        try:
            gt_mm = (float(d1) + float(d2)) / 2.0
        except (TypeError, ValueError):
            gt_mm = None
        values.append(gt_mm)

    if not values:
        print(f"  ⚠  Could not find GT rows for {session} — GT will be None")
        return [None] * APPLES_PER_SESSION

    if len(values) < APPLES_PER_SESSION:
        print(f"  ⚠  GT for {session} has {len(values)} rows, expected {APPLES_PER_SESSION}")
        values += [None] * (APPLES_PER_SESSION - len(values))

    return values[:APPLES_PER_SESSION]


# ─────────────────────────────────────────────────────────────────────────────
# GT ASSIGNMENT  —  Entry-gate ordering
#
# Lane 0 (top)    : mean_cy in [0,         img_h/3)
# Lane 1 (middle) : mean_cy in [img_h/3,   2*img_h/3)
# Lane 2 (bottom) : mean_cy in [2*img_h/3, img_h)
#
# Algorithm:
#   1. Assign each apple to a lane from its mean y-centroid
#   2. Sort each lane by ENTRY time (frames[0]["frame_idx"]) — the moment
#      the apple first appeared in the entry zone
#   3. Interleave lanes positionally; trailing None slots = missing apples
#
# Sorting by entry time (not exit time) makes the assignment robust against
# apples that slow down, speed up, or fall off the conveyor after entry.
# No gap detection is needed.
#
# GT index = pos * LANES + lane_id   (matches physical numbering scheme)
# ─────────────────────────────────────────────────────────────────────────────
LANES             = 3     # physical conveyor lanes
EXPECTED_PER_LANE = APPLES_PER_SESSION // LANES   # 6


def assign_gt_by_column(committed_tracks: list, gt_list: list,
                        img_h: int = 1536) -> list:
    """
    Assign GT values using entry-gate ordering.

    Each apple's GT number is locked in the moment it enters the camera view.
    Within each lane, apples are ordered by their FIRST seen frame index
    (frames[0]["frame_idx"]).  Lanes are then interleaved positionally.

    This handles correctly:
      - Apples that slow down or speed up after the entry gate
      - Apples that fall off the conveyor (they are still counted because
        they were already assigned their slot at entry)
      - No gap-detection heuristics needed
    """
    if not committed_tracks:
        return []

    lane_h = img_h / LANES

    # Step 1: assign lane from y-centroid
    for t in committed_tracks:
        t.lane_id = min(LANES - 1, int(t.mean_cy / lane_h))

    # Step 2: bucket into lanes
    lanes_all = [[] for _ in range(LANES)]
    for t in committed_tracks:
        lanes_all[t.lane_id].append(t)

    # Step 2b: select top EXPECTED_PER_LANE tracks per lane by frame count
    # -----------------------------------------------------------------------
    # First, drop stationary tracks (background smudges / YOLO false positives
    # that persist for the whole session but never move across the belt).
    # A real apple must travel at least MIN_CX_TRAVEL pixels left→right.
    # Then rank remaining candidates by frame count and keep top N.
    # After selection, sort by entry time to recover GT arrival order.
    lanes = []
    for lid, all_tracks in enumerate(lanes_all):
        # Motion filter: keep only tracks that actually moved across the belt
        mobile = []
        static = []
        for t in all_tracks:
            cx_vals  = [f["cx_px"] for f in t.frames if "cx_px" in f]
            cx_range = (max(cx_vals) - min(cx_vals)) if cx_vals else 0
            if cx_range >= MIN_CX_TRAVEL:
                mobile.append(t)
            else:
                static.append(t)
        if static:
            print(f"    Lane {lid}: dropped {len(static)} stationary track(s) "
                  f"(cx travel < {MIN_CX_TRAVEL}px)")

        best = sorted(mobile,
                      key=lambda t: len(t.frames), reverse=True)[:EXPECTED_PER_LANE]
        best.sort(key=lambda t: t.frames[0]["frame_idx"] if t.frames else 0)
        if len(mobile) > len(best):
            best_ids    = {id(t) for t in best}
            discarded   = [t for t in mobile if id(t) not in best_ids]
            max_discard = max((len(t.frames) for t in discarded), default=0)
            min_kept    = min((len(t.frames) for t in best), default=0)
            print(f"    Lane {lid}: {len(mobile)} mobile candidates → "
                  f"kept top {len(best)} by frame count "
                  f"(min kept: {min_kept} frames, "
                  f"max discarded: {max_discard} frames)")
        lanes.append(best)

    # Print entry-gate order for diagnostics
    print("  Entry-gate ordering (sorted by first-seen frame):")
    for lid, lane in enumerate(lanes):
        entries = [t.frames[0]["frame_idx"] if t.frames else 0 for t in lane]
        print(f"    Lane {lid}: {len(lane)} apple(s)  entry frames={entries}")

    # Step 3: build positional slots — no gap detection, just sequential
    # Each lane has exactly EXPECTED_PER_LANE slots.
    # Apples present fill from pos 0; trailing None = apple not seen in that slot.
    expanded = []
    for lane in lanes:
        slots = list(lane)                         # already in entry order
        while len(slots) < EXPECTED_PER_LANE:
            slots.append(None)                     # trailing missing apple
        expanded.append(slots[:EXPECTED_PER_LANE])

    # Print per-lane summary
    for lid, slots in enumerate(expanded):
        n_ok   = sum(1 for s in slots if s is not None)
        n_miss = sum(1 for s in slots if s is None)
        print(f"  Lane {lid}: {n_ok} present, {n_miss} missing  (slots={len(slots)})")

    # Step 4: interleave and assign GT
    apples = []
    apple_idx = 0
    for pos in range(EXPECTED_PER_LANE):
        for lane_id in range(LANES):
            slot = expanded[lane_id][pos]
            if slot is None:
                apple_idx += 1    # consume GT slot — apple was missing here
            else:
                gt_mm = gt_list[apple_idx] if apple_idx < len(gt_list) else None
                apples.append({
                    "apple_idx":      apple_idx,
                    "gt_mm":          gt_mm,
                    "lane":           lane_id,
                    "pos_in_lane":    pos,
                    "exit_frame_idx": slot.exit_frame_idx,
                    "frames":         slot.frames,
                })
                apple_idx += 1

    return apples


def process_session(session: str, data_root: str, model: YOLO,
                    gt_path: str, output_dir: str):
    """Run YOLO + tracker on all frames of one session, save pickle."""

    src0_dir = Path(data_root) / "Source0" / session
    src1_dir = Path(data_root) / "Source1" / session

    if not src0_dir.exists():
        print(f"  ✗ Source0 folder not found: {src0_dir}")
        return

    # List Source0 frames sorted naturally
    src0_files = sorted(
        [f for f in src0_dir.iterdir() if f.suffix.lower() == ".bmp"],
        key=lambda f: natural_key(f.name)
    )
    if not src0_files:
        print(f"  ✗ No BMP files in {src0_dir}")
        return

    print(f"\n{'─'*60}")
    print(f"  Session {session}  |  {len(src0_files)} frames  |  src0: {src0_dir.name}")

    # Detect image size from first frame
    probe = cv2.imread(str(src0_files[0]))
    if probe is None:
        print(f"  ✗ Cannot read first frame: {src0_files[0]}")
        return
    img_h, img_w = probe.shape[:2]
    print(f"  Image size: {img_w}×{img_h}")
    del probe

    tracker = SimpleTracker(img_w, img_h)

    for frame_idx, src0_path in enumerate(src0_files):
        # Frame number from filename e.g. "G1_1042.bmp" → 1042
        stem = src0_path.stem   # "G1_1042"
        frame_no = int(stem.split("_")[-1])

        # Corresponding Source1 frame
        src1_path = src1_dir / src0_path.name
        if not src1_path.exists():
            continue

        # Build composite for YOLO
        try:
            composite = build_composite(str(src0_path), str(src1_path), INPUT_MODE)
        except Exception as e:
            print(f"  ⚠ Frame {frame_no}: {e}")
            continue

        # Run YOLO inference (no built-in tracking — we do our own)
        results = model(
            source=composite,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            save=False,
        )

        # Parse detections
        detections = []
        for r in results:
            boxes  = r.boxes
            masks  = getattr(r, "masks", None)
            mask_data = masks.data.cpu().numpy() if masks is not None else []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                conf    = float(box.conf.cpu().numpy().item())
                cls_id  = int(box.cls.cpu().numpy().item())

                # Get binary mask at YOLO output resolution, resize to img size
                if masks is not None and i < len(mask_data):
                    m = (mask_data[i] > 0.5).astype(np.uint8)
                    if m.shape[:2] != (img_h, img_w):
                        m = cv2.resize(m, (img_w, img_h),
                                       interpolation=cv2.INTER_NEAREST)
                else:
                    # No mask — create circular approximation from bbox
                    m = np.zeros((img_h, img_w), dtype=np.uint8)
                    r_px = min((x2 - x1), (y2 - y1)) // 2
                    cv2.circle(m, (cx, cy), r_px, 1, -1)

                # Crop mask to bbox + padding (avoids storing 3MB full-frame masks)
                mx1 = max(0, x1 - MASK_PAD)
                my1 = max(0, y1 - MASK_PAD)
                mx2 = min(img_w, x2 + MASK_PAD)
                my2 = min(img_h, y2 + MASK_PAD)
                m_crop = m[my1:my2, mx1:mx2]   # typically ~200-400px square

                detections.append({
                    "frame_no":  frame_no,
                    "frame_idx": frame_idx,
                    "bbox":      [x1, y1, x2, y2],
                    "crop_rect": [mx1, my1, mx2, my2],
                    "cx_px":     cx,
                    "cy_px":     cy,
                    "mask":      m_crop,    # cropped binary mask, NOT full-frame
                    "conf":      conf,
                    "class_id":  cls_id,
                })

        # Update tracker (pass frame_idx for exit time recording)
        newly_committed = tracker.update(detections, frame_idx)

    # ── End-of-session flush: mop up tracks still active at last frame ────────
    # Include any registered track that never exited (apple still on belt
    # when session ended, or fell off before reaching EXIT_FRAC).
    last_frame = len(src0_files) - 1
    for t in tracker.tracks:
        if t.registered and t.exit_frame_idx is None:
            t.exit_frame_idx = last_frame

    # Also register any track that was still accumulating frames at session end
    # (handles startup-tracked apples that never crossed EXIT_FRAC)
    for t in tracker.tracks:
        if not t.registered and t.age >= MIN_FRAMES:
            t.registered = True
            t.exit_frame_idx = last_frame
            tracker.registered.append(t)

    print(f"  Gate-registered tracks: {len(tracker.registered)}  "
          f"(expected {APPLES_PER_SESSION})")

    # ── Assign GT using entry-gate ordering (registered tracks)
    gt_list = load_gt(gt_path, session) if gt_path else [None] * APPLES_PER_SESSION
    committed_apples = assign_gt_by_column(tracker.registered, gt_list, img_h)

    # Print summary
    for a in committed_apples:
        gt_mm  = a["gt_mm"]
        gt_str = f"{gt_mm:.1f}mm" if gt_mm is not None else "None"
        print(f"    Apple {a['apple_idx']+1:2d}  "
              f"lane={a['lane']} pos={a['pos_in_lane']}  "
              f"frames={len(a['frames']):3d}  gt={gt_str}")

    print(f"  Total matched: {len(committed_apples)}  (expected {APPLES_PER_SESSION})")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{session}.pkl"
    payload = {
        "session":  session,
        "img_w":    img_w,
        "img_h":    img_h,
        "apples":   committed_apples,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  ✔ Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract per-apple per-frame features from G1-G11 sessions"
    )
    parser.add_argument("--data_root",  default=DEFAULT_DATA_ROOT,
                        help="Root containing Source0/Source1/Source2 folders")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                        help="Path to YOLO best.pt")
    parser.add_argument("--gt_path",    default=DEFAULT_GT_PATH,
                        help="Path to gt.xlsx")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save .pkl files")
    parser.add_argument("--sessions",   nargs="+",
                        default=[f"G{i}" for i in range(1, 12)],
                        help="Sessions to process e.g. G1 G2 G10")
    parser.add_argument("--device",     default="cuda",
                        help="Inference device: cuda or cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model = YOLO(args.model_path)
    model.to(args.device)
    print(f"Model loaded. Device: {args.device}")
    print(f"Sessions to process: {args.sessions}")

    for session in args.sessions:
        process_session(
            session      = session,
            data_root    = args.data_root,
            model        = model,
            gt_path      = args.gt_path,
            output_dir   = args.output_dir,
        )

    print("\n✅ All sessions complete.")


if __name__ == "__main__":
    main()
