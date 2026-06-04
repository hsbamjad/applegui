"""
scripts/visualize_session.py  —  Apple Size Estimation Visualization
=====================================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

Reads a session .pkl (from extract_frames.py) and the ORIGINAL BMP frames,
then renders an annotated video showing:
  • Coloured apple mask (one colour per apple)
  • Apple number, lane, and position label
  • Running diameter estimate that updates each frame (all 4 methods)
  • Quality score bar (green bar = how trustworthy this frame is)
  • GT ground truth caliper value
  • Defect class (Fresh / Processing / Cull) from YOLO
  • Final summary overlay when apple exits

Usage:
    python scripts/visualize_session.py \
        --session G1 \
        --data_root "D:\\HA\\apple_gui\\images" \
        --pkl     "D:\\HA\\apple_gui\\data\\frame_features\\G1.pkl" \
        --out     "D:\\HA\\apple_gui\\data\\viz"

Output:
    D:\\HA\\apple_gui\\data\\viz\\G1_visualization.mp4
"""

import sys, os, pickle, argparse, re
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session",   default="G1")
    p.add_argument("--data_root", default=r"D:\HA\apple_gui\images",
                   help="Folder containing Source0/<session>/ BMP frames")
    p.add_argument("--pkl",       default=None,
                   help="Path to session .pkl (auto-detected if omitted)")
    p.add_argument("--pkl_dir",   default=r"D:\HA\apple_gui\data\frame_features")
    p.add_argument("--out",       default=r"D:\HA\apple_gui\data\viz")
    p.add_argument("--fps",       type=float, default=60.0,
                   help="Output video FPS (default 60, matches frames-to-video.py)")
    p.add_argument("--lossless",  action="store_true",
                   help="Write MJPEG .avi instead of mp4v .mp4 (better for NIR, no compression)")
    p.add_argument("--scale",     type=float, default=0.5,
                   help="Downscale factor for output (0.5 = half resolution)")
    p.add_argument("--max_frames",type=int,   default=0,
                   help="Stop after N frames (0 = full session)")
    return p.parse_args()

# ── Constants ─────────────────────────────────────────────────────────────────
SCALE_MM_PX = 0.377          # mm per pixel (approximate, used for display only)

# One vivid colour per apple (BGR)
APPLE_COLORS = [
    (255, 100,  50),  # blue-ish
    ( 50, 230,  50),  # green
    ( 50,  50, 255),  # red
    (255, 200,   0),  # cyan
    (200,   0, 255),  # magenta
    (  0, 200, 255),  # yellow
    (180, 255,   0),  # lime-cyan
    (255,   0, 180),  # pink
    (  0, 255, 200),  # yellow-green
    (100, 100, 255),  # salmon
    (255, 100, 200),  # lavender
    (  0, 180, 255),  # gold
    (200, 255, 100),  # teal
    (255,  50, 100),  # violet
    ( 50, 255, 255),  # hot yellow
    (200,  50, 255),  # orange-red
    (255, 200, 150),  # sky
    (150, 255, 100),  # mint
]

LANE_NAMES = {0: "Top", 1: "Mid", 2: "Bot"}
CLS_NAMES  = {0: "Fresh", 1: "Proc.", 2: "Cull"}
CLS_COLORS = {0: (50, 220, 50), 1: (50, 200, 255), 2: (50, 50, 255)}

# ── Helpers ───────────────────────────────────────────────────────────────────
def natural_key(s):
    # Match frames-to-video.py: strip all non-digits, sort numerically
    nums = re.sub(r'\D', '', s)
    return int(nums) if nums else 0

def put_text_bg(img, text, org, font_scale=0.55, thickness=1,
                color=(255,255,255), bg=(0,0,0,160)):
    """Draw text with a semi-transparent background box."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 3
    # Background rectangle
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+baseline+pad),
                  bg[:3], -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_quality_bar(img, x, y, quality, width=60, height=8):
    """Draw a horizontal quality bar (green = high quality)."""
    cv2.rectangle(img, (x, y), (x+width, y+height), (60,60,60), -1)
    fill = int(width * max(0, min(1, quality)))
    g = int(50 + 200 * quality)
    r = int(200 * (1-quality))
    cv2.rectangle(img, (x, y), (x+fill, y+height), (0, g, r), -1)
    cv2.rectangle(img, (x, y), (x+width, y+height), (120,120,120), 1)

def overlay_mask(frame, mask_uint8, color_bgr, alpha=0.45):
    """Blend a binary mask onto the frame."""
    colored = np.zeros_like(frame)
    colored[mask_uint8 > 0] = color_bgr
    cv2.addWeighted(colored, alpha, frame, 1.0, 0, frame)

# ── Running estimate accumulator ──────────────────────────────────────────────
class AppleEstimator:
    """Accumulates per-frame measurements and computes running quality-weighted mean."""
    def __init__(self, apple):
        self.apple       = apple
        self.q_vals      = []   # quality scores seen so far
        self.d_area_vals = []
        self.d_maxw_vals = []
        self.d_sym_vals  = []
        self.d_ell_vals  = []
        self.cls_counts  = {0:0, 1:0, 2:0}
        self.n_seen      = 0

    def update(self, frame_meta):
        q  = frame_meta.get("quality", 0)
        self.q_vals.append(q)
        self.d_area_vals.append(frame_meta.get("d_area", 0) * SCALE_MM_PX)
        self.d_maxw_vals.append(frame_meta.get("d_maxw", 0) * SCALE_MM_PX)
        self.d_sym_vals .append(frame_meta.get("d_sym",  0) * SCALE_MM_PX)
        self.d_ell_vals .append(frame_meta.get("d_ell",  0) * SCALE_MM_PX)
        cls = frame_meta.get("cls_id", 0)
        self.cls_counts[cls] = self.cls_counts.get(cls, 0) + 1
        self.n_seen += 1

    def wmean(self, vals):
        if not vals: return 0.0
        w = np.array(self.q_vals[:len(vals)])
        v = np.array(vals)
        return float(np.dot(w, v) / (w.sum() + 1e-9))

    @property
    def est_mm(self):
        return self.wmean(self.d_area_vals)

    @property
    def best_cls(self):
        return max(self.cls_counts, key=self.cls_counts.get)

    @property
    def mean_q(self):
        return float(np.mean(self.q_vals)) if self.q_vals else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Load pkl ───────────────────────────────────────────────────────────────
    pkl_path = args.pkl or str(Path(args.pkl_dir) / f"{args.session}.pkl")
    print(f"Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    apples   = data["apples"]
    img_w    = data.get("img_w", 2048)
    img_h    = data.get("img_h", 1536)
    session  = data.get("session", args.session)

    print(f"  Session: {session}  |  {len(apples)} apples  |  {img_w}×{img_h}")

    # ── Build frame-number → list of (apple, frame_meta) lookup ───────────────
    frame_map = {}   # frame_no → [(apple_dict, frame_meta_dict), ...]
    for apple in apples:
        for fm in apple.get("frames", []):
            fn = fm["frame_no"]
            frame_map.setdefault(fn, []).append((apple, fm))

    # ── Estimator per apple ────────────────────────────────────────────────────
    estimators = {a["apple_idx"]: AppleEstimator(a) for a in apples}

    # ── Find BMP frames ────────────────────────────────────────────────────────
    src0_dir = Path(args.data_root) / "Source0" / session
    if not src0_dir.exists():
        # Fallback: try direct session folder
        src0_dir = Path(args.data_root) / session
    if not src0_dir.exists():
        print(f"ERROR: Cannot find BMP frames at {src0_dir}")
        sys.exit(1)

    bmp_files = sorted(
        [f for f in src0_dir.iterdir() if f.suffix.lower() in (".bmp", ".png", ".jpg")],
        key=lambda p: natural_key(p.name)
    )
    if not bmp_files:
        print(f"ERROR: No image files in {src0_dir}")
        sys.exit(1)

    total_frames = len(bmp_files) if not args.max_frames else min(len(bmp_files), args.max_frames)
    print(f"  BMP frames: {len(bmp_files)}  →  rendering {total_frames}")

    # ── Build frame_no lookup from BMP filenames ───────────────────────────────
    # Frame numbers extracted from filenames like "frame_001234.bmp" or "001234.bmp"
    def extract_frame_no(p):
        nums = re.findall(r"\d+", p.stem)
        return int(nums[-1]) if nums else 0

    bmp_by_no = {extract_frame_no(p): p for p in bmp_files}
    all_frame_nos = sorted(bmp_by_no.keys())[:total_frames]

    # ── Output video writer ────────────────────────────────────────────────────
    out_w = int(img_w  * args.scale)
    out_h = int(img_h  * args.scale)
    if args.lossless:
        out_path = str(Path(args.out) / f"{session}_visualization.avi")
        fourcc   = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        out_path = str(Path(args.out) / f"{session}_visualization.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (out_w, out_h))
    print(f"  Output: {out_path}  ({out_w}×{out_h} @ {args.fps}fps)"
          f"  {'MJPEG lossless' if args.lossless else 'mp4v'}")

    # ── Render loop ────────────────────────────────────────────────────────────
    print(f"\nRendering frames...")
    for fi, frame_no in enumerate(all_frame_nos):
        if fi % 500 == 0:
            print(f"  frame {fi}/{total_frames} ...")

        bmp_path = bmp_by_no[frame_no]
        frame_bgr = cv2.imread(str(bmp_path))
        if frame_bgr is None:
            # Skip unreadable frame
            blank = np.zeros((img_h, img_w, 3), np.uint8)
            frame_bgr = blank

        # ── For each active apple in this frame ────────────────────────────────
        active_apples = frame_map.get(frame_no, [])

        # Update estimators first
        for apple, fm in active_apples:
            idx = apple["apple_idx"]
            estimators[idx].update(fm)

        # Draw masks and labels
        for apple, fm in active_apples:
            idx    = apple["apple_idx"]
            color  = APPLE_COLORS[idx % len(APPLE_COLORS)]
            est    = estimators[idx]

            # ── Draw consensus mask at current bbox position ─────────────────
            # We don't store per-frame masks (too large) but we have bbox
            bbox = fm.get("bbox")  # [x1,y1,x2,y2]
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cx = fm.get("cx_px", (x1+x2)//2)
                cy = fm.get("cy_px", (y1+y2)//2)

                # Draw bbox rectangle
                cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), color, 2)

                # Overlay consensus mask positioned at current bbox
                cons_mask = apple.get("consensus_mask")
                cons_rect = apple.get("consensus_rect")
                if cons_mask is not None and cons_rect is not None:
                    # Scale consensus mask to current bbox size
                    bw, bh = x2-x1, y2-y1
                    if bw > 4 and bh > 4:
                        m_resized = cv2.resize(cons_mask.astype(np.uint8)*255,
                                               (bw, bh),
                                               interpolation=cv2.INTER_NEAREST)
                        # Clip to frame bounds
                        ix1 = max(0, x1); ix2 = min(img_w, x2)
                        iy1 = max(0, y1); iy2 = min(img_h, y2)
                        mx1 = ix1 - x1;   mx2 = mx1 + (ix2 - ix1)
                        my1 = iy1 - y1;   my2 = my1 + (iy2 - iy1)
                        if mx2 > mx1 and my2 > my1:
                            region  = frame_bgr[iy1:iy2, ix1:ix2]
                            mask_r  = m_resized[my1:my2, mx1:mx2]
                            colored = np.zeros_like(region)
                            colored[mask_r > 128] = color
                            cv2.addWeighted(colored, 0.40, region, 1.0, 0, region)
                            frame_bgr[iy1:iy2, ix1:ix2] = region

                # ── Text label ─────────────────────────────────────────────
                lane_name = LANE_NAMES.get(apple["lane"], "?")
                pos       = apple.get("pos_in_lane", apple.get("pos", "?"))
                gt_mm     = apple.get("gt_mm")
                gt_str    = f"GT:{gt_mm:.1f}" if gt_mm else "GT:--"

                # Running estimate
                est_mm  = est.est_mm
                est_str = f"{est_mm:.1f}mm" if est.n_seen > 5 else "..."

                # Error vs GT
                if gt_mm and est.n_seen > 5:
                    err = est_mm - gt_mm
                    err_str = f"({err:+.1f})"
                    err_col = (50,200,50) if abs(err) <= 3 else (50,50,255)
                else:
                    err_str = ""
                    err_col = (200,200,200)

                # Quality bar + cls
                q    = fm.get("quality", 0)
                cls  = fm.get("cls_id", 0)
                cls_str = CLS_NAMES.get(cls, "?")
                cls_col = CLS_COLORS.get(cls, (200,200,200))

                lx = x1
                ly = max(20, y1 - 50)

                # Apple ID header
                put_text_bg(frame_bgr,
                            f"#{apple['apple_idx']+1} L{apple['lane']}P{pos}",
                            (lx, ly), font_scale=0.60, color=color)

                # Estimate + error
                put_text_bg(frame_bgr,
                            f"{est_str} {err_str}",
                            (lx, ly+22), font_scale=0.58, color=err_col)

                # GT
                put_text_bg(frame_bgr, gt_str,
                            (lx, ly+42), font_scale=0.50, color=(180,220,255))

                # Quality bar
                draw_quality_bar(frame_bgr, lx, ly+52, q, width=70, height=7)
                put_text_bg(frame_bgr, f"Q:{q:.2f}",
                            (lx+74, ly+59), font_scale=0.38, color=(200,200,200))

        # ── Header bar ────────────────────────────────────────────────────────
        cv2.rectangle(frame_bgr, (0,0), (img_w, 36), (20,20,20), -1)
        put_text_bg(frame_bgr,
                    f"Session: {session}  |  Frame {frame_no}  ({fi+1}/{total_frames})"
                    f"  |  Active apples: {len(active_apples)}",
                    (10, 24), font_scale=0.65, color=(240,240,240), bg=(20,20,20,0))

        # ── Downscale and write ────────────────────────────────────────────────
        if args.scale != 1.0:
            frame_out = cv2.resize(frame_bgr, (out_w, out_h))
        else:
            frame_out = frame_bgr

        writer.write(frame_out)

    writer.release()

    # ── Print final summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL ESTIMATES  —  {session}")
    print(f"{'='*70}")
    print(f"  {'#':>3}  {'Lane':>5}  {'Pos':>4}  {'GT mm':>7}  "
          f"{'Pred mm':>8}  {'Error':>7}  {'Q':>6}  {'N_frames':>8}")
    print(f"  {'-'*65}")
    for apple in sorted(apples, key=lambda a: a["apple_idx"]):
        idx = apple["apple_idx"]
        est = estimators[idx]
        gt  = apple.get("gt_mm")
        pred = est.est_mm
        err  = (pred - gt) if gt else None
        err_str = f"{err:+.2f}" if err is not None else "  --  "
        flag = " ✓" if (err is not None and abs(err) <= 3) else (" ✗" if err is not None else "")
        print(f"  #{idx+1:2d}  L{apple['lane']}P{apple.get('pos_in_lane', apple.get('pos','?'))}  "
              f"{'':>3}  "
              f"{(str(round(gt,1)) if gt else '--'):>7}  "
              f"{pred:8.2f}  {err_str:>7}  "
              f"{est.mean_q:6.3f}  {est.n_seen:8d}{flag}")

    print(f"\nVideo saved → {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
