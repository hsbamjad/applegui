"""
scripts/visualize_session.py  —  Apple Size Estimation  |  Tech-HUD Visualization
===================================================================================
Michigan State University | Apple GUI | feature/apple-size-ml branch

Professional animated visualization showing HOW the sizing pipeline works:

  ANIMATIONS PER APPLE:
  • Pulsing corner-bracket reticle (targeting lock-on style)
  • Semi-transparent mask with scanline texture (shows segmented region)
  • Fitted ellipse drawn over the apple (shows the ellipse fitting step)
  • Diameter measurement line that rotates (shows max-width scan)
  • Crosshair through apple centroid
  • Convergence sparkline — tiny live graph showing estimate stabilising
  • Circular progress arc — how many frames seen vs total

  FRAME-LEVEL EFFECTS:
  • Vertical scan sweep line moving across frame (lab scanner aesthetic)
  • Lane boundary markers
  • Fixed right-side HUD panel with session stats

  DATA SHOWN PER APPLE:
  • LIVE  — running quality-weighted area×scale estimate (updates every frame)
  • ML    — trained Ridge model prediction (computed from all frames upfront)
  • GT    — caliper ground truth
  • Error — colour-coded (green ≤3mm, orange ≤5mm, red >5mm)

Usage (shuttle):
    python scripts/visualize_session.py ^
        --session G10 ^
        --data_root "G:\\Haseeb\\pic" ^
        --pkl "D:\\HA\\apple_gui\\data\\frame_features\\G10.pkl" ^
        --model "D:\\HA\\apple_gui\\models\\size_model.pkl" ^
        --out "D:\\HA\\apple_gui\\data\\viz"

Quick test — first 600 frames only:
    ... --max_frames 600
"""

import sys, os, pickle, argparse, re, math
import numpy as np
import cv2
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── optional ML import ────────────────────────────────────────────────────────
try:
    from core.sizing.view_fusion import fuse_apple_views
    HAS_FUSION = True
except ImportError:
    HAS_FUSION = False

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session",    default="G1")
    p.add_argument("--data_root",  default=r"D:\HA\apple_gui\images")
    p.add_argument("--pkl",        default=None)
    p.add_argument("--pkl_dir",    default=r"D:\HA\apple_gui\data\frame_features")
    p.add_argument("--model",      default=r"D:\HA\apple_gui\models\size_model.pkl")
    p.add_argument("--out",        default=r"D:\HA\apple_gui\data\viz")
    p.add_argument("--fps",        type=float, default=60.0)
    p.add_argument("--lossless",   action="store_true")
    p.add_argument("--scale",      type=float, default=0.5,
                   help="Output downscale (0.5 = half res, faster render)")
    p.add_argument("--max_frames", type=int,   default=0)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SCALE_MM_PX   = 0.377
SPARKLINE_LEN = 100      # frames of history in convergence graph
ALPHA_TRIM    = 0.10
CENTRAL_FRAC  = 0.60

# Neon palette (BGR)  — one per apple (18 apples max)
PALETTE = [
    (  0, 255, 180),  # neon green
    (255, 140,   0),  # neon orange
    (255,   0, 180),  # neon pink
    (  0, 200, 255),  # neon yellow
    (180,   0, 255),  # neon violet
    (  0, 255, 255),  # neon cyan
    (255, 255,   0),  # electric blue
    (  0, 100, 255),  # hot red
    (100, 255,   0),  # lime
    (255,  80, 200),  # lavender
    (  0, 255, 100),  # spring green
    (200, 255,   0),  # teal-yellow
    (255,   0,  80),  # cobalt
    ( 50, 255, 200),  # mint
    (255, 200,   0),  # sky blue
    (180, 255, 180),  # pale green
    (255, 120, 255),  # orchid
    (120, 200, 255),  # peach
]

BG_DARK     = (12,  14,  20)    # almost black (BGR)
BG_PANEL    = (18,  22,  32)    # side panel background
GRID_COLOR  = (30,  38,  55)    # lane grid lines
SCAN_COLOR  = ( 0, 255,  80)    # scanning line colour
TEXT_MAIN   = (220, 230, 245)
TEXT_DIM    = (100, 120, 150)
ACCENT      = (  0, 255, 160)   # global accent (neon green)

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
def put_text(img, text, org, scale=0.48, color=TEXT_MAIN, thick=1, shadow=True):
    font = cv2.FONT_HERSHEY_DUPLEX
    x, y = int(org[0]), int(org[1])
    if shadow:
        cv2.putText(img, text, (x+1,y+1), font, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), font, scale, color, thick, cv2.LINE_AA)

def put_text_bg(img, text, org, scale=0.48, color=TEXT_MAIN, thick=1,
                bg=BG_DARK, pad=4):
    font = cv2.FONT_HERSHEY_DUPLEX
    x, y = int(org[0]), int(org[1])
    (tw,th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), bg, -1)
    cv2.putText(img, text, (x,y), font, scale, color, thick, cv2.LINE_AA)

def corner_bracket(img, x1, y1, x2, y2, color, thick=2, arm=22, pulse=1.0):
    """Draw corner bracket reticle instead of plain rectangle."""
    c = tuple(min(255, int(v*pulse)) for v in color)
    a = arm
    # top-left
    cv2.line(img,(x1,y1),(x1+a,y1),c,thick,cv2.LINE_AA)
    cv2.line(img,(x1,y1),(x1,y1+a),c,thick,cv2.LINE_AA)
    # top-right
    cv2.line(img,(x2,y1),(x2-a,y1),c,thick,cv2.LINE_AA)
    cv2.line(img,(x2,y1),(x2,y1+a),c,thick,cv2.LINE_AA)
    # bottom-left
    cv2.line(img,(x1,y2),(x1+a,y2),c,thick,cv2.LINE_AA)
    cv2.line(img,(x1,y2),(x1,y2-a),c,thick,cv2.LINE_AA)
    # bottom-right
    cv2.line(img,(x2,y2),(x2-a,y2),c,thick,cv2.LINE_AA)
    cv2.line(img,(x2,y2),(x2,y2-a),c,thick,cv2.LINE_AA)

def crosshair(img, cx, cy, color, size=14, gap=4, thick=1):
    """Draw crosshair with gap at center."""
    cv2.line(img,(cx-size,cy),(cx-gap,cy),color,thick,cv2.LINE_AA)
    cv2.line(img,(cx+gap, cy),(cx+size,cy),color,thick,cv2.LINE_AA)
    cv2.line(img,(cx,cy-size),(cx,cy-gap),color,thick,cv2.LINE_AA)
    cv2.line(img,(cx,cy+gap),(cx,cy+size),color,thick,cv2.LINE_AA)

def draw_scanline_mask(img, mask_uint8, x1, y1, color, alpha=0.35, stride=4):
    """Fill mask with semi-transparent color + horizontal scanline texture."""
    roi = img[y1:y1+mask_uint8.shape[0], x1:x1+mask_uint8.shape[1]]
    if roi.shape[:2] != mask_uint8.shape[:2]:
        return
    fill = np.zeros_like(roi)
    m    = mask_uint8 > 128
    fill[m] = color
    # Scanline effect: dim every stride-th row
    fill[::stride, :] = (fill[::stride, :] * 0.25).astype(np.uint8)
    cv2.addWeighted(fill, alpha, roi, 1.0, 0, roi)
    img[y1:y1+roi.shape[0], x1:x1+roi.shape[1]] = roi

def draw_sparkline(img, values, x, y, w, h, color, bg=BG_PANEL):
    """Mini convergence graph — shows estimate stabilising over time."""
    cv2.rectangle(img, (x-1,y-1), (x+w+1,y+h+1), (50,60,80), 1)
    cv2.rectangle(img, (x,y), (x+w,y+h), bg, -1)
    if len(values) < 2:
        return
    arr  = np.array(list(values), dtype=float)
    mn   = arr.min()
    mx   = arr.max()
    rng  = mx - mn
    if rng < 0.5: rng = 0.5   # minimum range to show
    mid  = (mn+mx)/2
    mn, mx = mid-rng/2, mid+rng/2
    pts = []
    for i, v in enumerate(arr):
        px = x + int(i * (w-1) / max(len(arr)-1,1))
        py = y + h - 1 - int((v-mn)/(mx-mn+1e-9) * (h-2))
        py = max(y, min(y+h-1, py))
        pts.append((px, py))
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], color, 1, cv2.LINE_AA)
    # Label
    put_text(img, f"{arr[-1]:.1f}", (x+w+3, y+h//2+5),
             scale=0.30, color=color, shadow=False)

def draw_arc_progress(img, cx, cy, r, fraction, color, thick=2):
    """Circular arc showing progress through total frames."""
    angle = int(360 * fraction)
    cv2.ellipse(img, (cx,cy), (r,r), -90, 0, angle,
                color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (cx,cy), (r,r), -90, angle, 360,
                (40,50,65), 1, cv2.LINE_AA)

def draw_diameter_line(img, cx, cy, d_px, angle_deg, color, thick=2):
    """Draw a diameter measurement line at given angle through center."""
    a  = math.radians(angle_deg)
    r  = d_px / 2
    x1 = int(cx - r*math.cos(a))
    y1 = int(cy - r*math.sin(a))
    x2 = int(cx + r*math.cos(a))
    y2 = int(cy + r*math.sin(a))
    cv2.line(img, (x1,y1), (x2,y2), color, thick, cv2.LINE_AA)
    # endpoint dots
    cv2.circle(img,(x1,y1),3,color,-1,cv2.LINE_AA)
    cv2.circle(img,(x2,y2),3,color,-1,cv2.LINE_AA)

def scan_sweep_line(img, frame_idx, img_w, img_h):
    """Vertical green scan line that sweeps right across frame."""
    period = img_w * 2          # full sweep period in frames
    t      = frame_idx % period
    x      = t if t < img_w else img_w*2 - t
    if 0 <= x < img_w:
        alpha_strip = np.zeros((img_h, 3), dtype=np.float32)
        for dx, alpha in [(-2,0.04),(-1,0.12),(0,0.55),(1,0.12),(2,0.04)]:
            xx = x + dx
            if 0 <= xx < img_w:
                img[:, xx] = np.clip(
                    img[:, xx].astype(np.float32) +
                    np.array(SCAN_COLOR, dtype=np.float32) * alpha,
                    0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# ML FEATURE BUILDER  (mirrors view_fusion logic)
# ─────────────────────────────────────────────────────────────────────────────
def build_ml_features(apple, img_w):
    frames = apple.get("frames", [])
    if not frames:
        return None
    cx_vals = [f["cx_px"] for f in frames]
    lo_cx   = min(cx_vals) + (max(cx_vals)-min(cx_vals))*(0.5-CENTRAL_FRAC/2)
    hi_cx   = min(cx_vals) + (max(cx_vals)-min(cx_vals))*(0.5+CENTRAL_FRAC/2)
    central = [f for f in frames if lo_cx <= f["cx_px"] <= hi_cx] or frames
    if len(central) < 5:
        central = frames

    Q  = np.array([f.get("quality",0.5) for f in central])
    DA = np.array([f.get("d_area", 0.0) for f in central])
    DM = np.array([f.get("d_maxw", 0.0) for f in central])
    DS = np.array([f.get("d_sym",  0.0) for f in central])
    DE = np.array([f.get("d_ell",  0.0) for f in central])

    def wm(v): return float(np.dot(Q,v)/(Q.sum()+1e-9))
    def pk(v):
        lo,hi = np.percentile(v,ALPHA_TRIM*100), np.percentile(v,(1-ALPHA_TRIM)*100)
        k = v[(v>=lo)&(v<=hi)]
        return float(k.mean()) if len(k) else float(v.mean())

    best = max(central, key=lambda f: f.get("quality",0))
    bb   = best.get("bbox",[0,0,10,10])
    bw,bh = bb[2]-bb[0], bb[3]-bb[1]

    return np.array([[
        wm(DA), wm(DM), wm(DS), wm(DE),
        pk(DA), pk(DM),
        max(bw,bh)/2.0, min(bw,bh)/2.0,
        float(Q.mean()),
        float(apple.get("lane",0)),
    ]])

# ─────────────────────────────────────────────────────────────────────────────
# LIVE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
class LiveEst:
    def __init__(self):
        self.q_sum = 0.0; self.qd_sum = 0.0; self.n = 0
        self.history = deque(maxlen=SPARKLINE_LEN)

    def update(self, fm):
        q = fm.get("quality",0.5)
        d = fm.get("d_area", 0.0) * SCALE_MM_PX
        self.q_sum += q; self.qd_sum += q*d; self.n += 1
        if self.n % 3 == 0:     # record every 3rd frame (smoother graph)
            self.history.append(self.qd_sum/(self.q_sum+1e-9))

    @property
    def mm(self):
        return self.qd_sum/(self.q_sum+1e-9) if self.q_sum>0 else None

# ─────────────────────────────────────────────────────────────────────────────
# SIDE HUD PANEL
# ─────────────────────────────────────────────────────────────────────────────
def draw_side_panel(img, session, n_apples, frame_no, total_frames,
                    ml_preds, live_ests, apples, panel_w=280):
    h, w = img.shape[:2]
    px   = w - panel_w

    # background
    overlay = img.copy()
    cv2.rectangle(overlay, (px,0), (w,h), BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv2.line(img,(px,0),(px,h),ACCENT,1,cv2.LINE_AA)

    y = 30
    put_text(img, "APPLE SIZING", (px+10,y), scale=0.58,
             color=ACCENT, thick=1)
    y += 22
    put_text(img, "MSU Vision Lab", (px+10,y), scale=0.36,
             color=TEXT_DIM)
    y += 28
    cv2.line(img,(px+10,y),(w-10,y),(50,65,90),1)
    y += 18

    put_text(img, f"Session  {session}", (px+10,y), scale=0.44, color=TEXT_MAIN)
    y += 22
    put_text(img, f"Apples   {n_apples}", (px+10,y), scale=0.44, color=TEXT_MAIN)
    y += 22
    pct = frame_no/max(total_frames,1)*100
    put_text(img, f"Progress {pct:.1f}%", (px+10,y), scale=0.44, color=TEXT_MAIN)
    y += 28
    cv2.line(img,(px+10,y),(w-10,y),(50,65,90),1)
    y += 18

    # Per-apple table
    put_text(img, "#   ML      GT    err", (px+10,y), scale=0.36, color=TEXT_DIM)
    y += 18
    for apple in sorted(apples, key=lambda a:a["apple_idx"]):
        idx = apple["apple_idx"]
        gt  = apple.get("gt_mm")
        ml  = ml_preds.get(idx)
        col = PALETTE[idx%len(PALETTE)]
        ml_s = f"{ml:.1f}" if ml else " -- "
        gt_s = f"{gt:.1f}" if gt else " -- "
        if ml and gt:
            err = ml - gt
            ec  = (50,220,50) if abs(err)<=3 else (30,165,255) if abs(err)<=5 else (50,50,255)
            er_s = f"{err:+.1f}"
        else:
            ec,er_s = TEXT_DIM,"  --"
        row = f"#{idx+1:<2} {ml_s:>6}  {gt_s:>6}  {er_s}"
        put_text(img, row, (px+10,y), scale=0.37, color=col)
        y += 17
        if y > h-20: break

    # Legend
    y = h - 80
    cv2.line(img,(px+10,y),(w-10,y),(50,65,90),1)
    y += 14
    put_text(img, "LIVE = area\u00d70.377 (running)", (px+10,y), scale=0.33, color=(150,200,150))
    y += 16
    put_text(img, "ML   = Ridge model (trained)", (px+10,y), scale=0.33, color=(100,180,255))
    y += 16
    put_text(img, "\u2588\u2588 err \u22643mm  \u2588\u2588 \u22645mm  \u2588\u2588 >5mm", (px+10,y),
             scale=0.33, color=TEXT_DIM)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Load pkl ───────────────────────────────────────────────────────────────
    pkl_path = args.pkl or str(Path(args.pkl_dir)/f"{args.session}.pkl")
    print(f"Loading pkl: {pkl_path}")
    with open(pkl_path,"rb") as f: data = pickle.load(f)

    apples  = data["apples"]
    img_w   = data.get("img_w",2048)
    img_h   = data.get("img_h",1536)
    session = data.get("session",args.session)
    print(f"  {session}: {len(apples)} apples  {img_w}x{img_h}")

    # ── Load ML model ──────────────────────────────────────────────────────────
    ml_model = None
    if Path(args.model).exists():
        with open(args.model,"rb") as f: saved = pickle.load(f)
        ml_model = saved.get("model") or saved
        print(f"  ML model: {args.model}")
    else:
        print(f"  [WARN] ML model not found — showing LIVE only")

    # ── Pre-compute ML predictions ─────────────────────────────────────────────
    ml_pred = {}
    if ml_model:
        for apple in apples:
            idx  = apple["apple_idx"]
            feat = build_ml_features(apple, img_w)
            if feat is not None:
                try:    ml_pred[idx] = float(ml_model.predict(feat)[0])
                except: ml_pred[idx] = None
            else:   ml_pred[idx] = None

    # ── Frame map ─────────────────────────────────────────────────────────────
    frame_map = {}
    for apple in apples:
        for fm in apple.get("frames",[]):
            frame_map.setdefault(fm["frame_no"],[]).append((apple,fm))

    # ── Total frames per apple (for progress arc) ──────────────────────────────
    total_frames_per_apple = {a["apple_idx"]: a.get("n_frames",1) for a in apples}

    # ── Live estimators ────────────────────────────────────────────────────────
    live = {a["apple_idx"]: LiveEst() for a in apples}

    # ── Lane boundaries (for lane grid) ───────────────────────────────────────
    if len(apples) > 0:
        cy_vals = {}
        for apple in apples:
            lane = apple["lane"]
            fms  = apple.get("frames",[])
            if fms:
                cy_vals.setdefault(lane,[]).append(
                    np.mean([f["cy_px"] for f in fms]))
        lane_y = {k: int(np.mean(v)) for k,v in cy_vals.items()}
    else:
        lane_y = {}

    # ── BMP frames ────────────────────────────────────────────────────────────
    src0_dir = Path(args.data_root)/"Source0"/session
    if not src0_dir.exists(): src0_dir = Path(args.data_root)/session
    if not src0_dir.exists():
        print(f"ERROR: {src0_dir}"); sys.exit(1)

    bmp_files = sorted(
        [f for f in src0_dir.iterdir()
         if f.suffix.lower() in (".bmp",".png",".jpg")],
        key=lambda p: int(re.sub(r'\D','',p.name) or '0'))

    def fn(p): nums=re.findall(r"\d+",p.stem); return int(nums[-1]) if nums else 0
    bmp_by_no     = {fn(p):p for p in bmp_files}
    total_frames  = min(len(bmp_files),args.max_frames) if args.max_frames else len(bmp_files)
    all_frame_nos = sorted(bmp_by_no.keys())[:total_frames]
    print(f"  Rendering {total_frames} frames...")

    # ── Video writer ──────────────────────────────────────────────────────────
    PANEL_W = 260
    out_w   = int(img_w*args.scale)
    out_h   = int(img_h*args.scale)

    if args.lossless:
        out_path = str(Path(args.out)/f"{session}_visualization.avi")
        fourcc   = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        out_path = str(Path(args.out)/f"{session}_visualization.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (out_w, out_h))
    print(f"  Output: {out_path}  ({out_w}x{out_h} @ {args.fps}fps)\n")

    # ── Render loop ────────────────────────────────────────────────────────────
    for fi, frame_no in enumerate(all_frame_nos):
        if fi % 500 == 0: print(f"  {fi}/{total_frames}")

        # Load frame
        bgr = cv2.imread(str(bmp_by_no[frame_no]))
        if bgr is None: bgr = np.zeros((img_h,img_w,3),np.uint8)

        # Darken slightly for HUD contrast
        bgr = (bgr.astype(np.float32) * 0.78).astype(np.uint8)

        active = frame_map.get(frame_no,[])

        # ── Lane guide lines ──────────────────────────────────────────────────
        for ln, ly in lane_y.items():
            cv2.line(bgr,(0,ly),(img_w-PANEL_W,ly),GRID_COLOR,1,cv2.LINE_AA)
            put_text(bgr, f"L{ln}", (5, ly-5), scale=0.30,
                     color=GRID_COLOR, shadow=False)

        # ── Scan sweep ────────────────────────────────────────────────────────
        scan_sweep_line(bgr, fi, img_w - PANEL_W, img_h)

        # ── Update live estimators first ──────────────────────────────────────
        for apple,fm in active:
            live[apple["apple_idx"]].update(fm)

        # ── Per-apple overlays ────────────────────────────────────────────────
        for apple,fm in active:
            idx   = apple["apple_idx"]
            color = PALETTE[idx % len(PALETTE)]
            bbox  = fm.get("bbox")
            if bbox is None: continue

            x1,y1,x2,y2 = [int(v) for v in bbox]
            cx = fm.get("cx_px",(x1+x2)//2)
            cy = fm.get("cy_px",(y1+y2)//2)
            bw,bh = x2-x1, y2-y1

            # ── 1. Scanline mask ──────────────────────────────────────────────
            cons_mask = apple.get("consensus_mask")
            if cons_mask is not None and bw>4 and bh>4:
                m = cv2.resize(cons_mask.astype(np.uint8)*255,(bw,bh),
                               interpolation=cv2.INTER_NEAREST)
                ix1,ix2 = max(0,x1), min(img_w,x2)
                iy1,iy2 = max(0,y1), min(img_h,y2)
                mx1,my1 = ix1-x1, iy1-y1
                mx2,my2 = mx1+(ix2-ix1), my1+(iy2-iy1)
                if mx2>mx1 and my2>my1:
                    draw_scanline_mask(bgr,
                                       m[my1:my2,mx1:mx2],
                                       ix1, iy1, color,
                                       alpha=0.40, stride=5)
                    # Contour outline (bright)
                    mc = m[my1:my2,mx1:mx2]
                    cnts,_ = cv2.findContours(mc, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    shifted = [c + np.array([ix1,iy1]) for c in cnts]
                    cv2.polylines(bgr, shifted, True, color, 1, cv2.LINE_AA)

            # ── 2. Fitted ellipse ─────────────────────────────────────────────
            ell_a = max(bw,bh)//2
            ell_b = min(bw,bh)//2
            if ell_a>0 and ell_b>0:
                cv2.ellipse(bgr,(cx,cy),(ell_a,ell_b),0,0,360,
                            tuple(int(v*0.55) for v in color),1,cv2.LINE_AA)

            # ── 3. Rotating diameter line (animation) ─────────────────────────
            d_maxw = fm.get("d_maxw", float(bw)) * 1.0   # pixels
            angle  = (fi * 1.5) % 180                    # slow rotation
            draw_diameter_line(bgr, cx, cy, d_maxw, angle, color, thick=1)

            # ── 4. Crosshair ──────────────────────────────────────────────────
            crosshair(bgr, cx, cy, color, size=16, gap=5)

            # ── 5. Pulsing corner bracket reticle ─────────────────────────────
            pulse = 0.65 + 0.35*math.sin(fi*0.18 + idx*1.1)
            arm   = max(18, int(min(bw,bh)*0.22))
            corner_bracket(bgr, x1,y1,x2,y2, color,
                           thick=2, arm=arm, pulse=pulse)

            # ── 6. Progress arc (frames seen / total) ─────────────────────────
            n_total = total_frames_per_apple.get(idx, 1)
            frac    = min(1.0, live[idx].n / max(n_total,1))
            draw_arc_progress(bgr, cx, cy, min(ell_a+10,60),
                              frac, color, thick=2)

            # ── 7. Info HUD above apple ───────────────────────────────────────
            gt_mm   = apple.get("gt_mm")
            pos     = apple.get("pos_in_lane", apple.get("pos","?"))
            live_mm = live[idx].mm
            ml_mm   = ml_pred.get(idx)
            q       = fm.get("quality",0)

            lx  = x1
            ly  = max(78, y1 - 82)
            lspc = 19

            # Apple ID
            put_text_bg(bgr, f"#{idx+1}  L{apple['lane']} P{pos}",
                        (lx, ly), scale=0.50, color=color,
                        bg=(10,10,20), pad=3)
            ly += lspc

            # LIVE estimate
            if live_mm:
                le  = live_mm-gt_mm if gt_mm else None
                lec = (50,220,50) if le and abs(le)<=3 else \
                      (30,165,255) if le and abs(le)<=5 else (50,50,255)
                ls  = f"LIVE {live_mm:5.1f}mm"
                ls += f" ({le:+.1f})" if le is not None else ""
            else:
                lec, ls = TEXT_DIM, "LIVE  ---"
            put_text_bg(bgr, ls, (lx,ly), scale=0.44,
                        color=lec, bg=(10,25,10), pad=3)
            ly += lspc

            # ML prediction
            if ml_mm:
                me  = ml_mm-gt_mm if gt_mm else None
                mec = (50,220,50) if me and abs(me)<=3 else \
                      (30,165,255) if me and abs(me)<=5 else (50,50,255)
                ms  = f"ML   {ml_mm:5.1f}mm"
                ms += f" ({me:+.1f})" if me is not None else ""
            else:
                mec, ms = TEXT_DIM, "ML    ---"
            put_text_bg(bgr, ms, (lx,ly), scale=0.44,
                        color=mec, bg=(10,10,30), pad=3)
            ly += lspc

            # GT
            gt_s = f"GT   {gt_mm:5.1f}mm" if gt_mm else "GT    ---"
            put_text_bg(bgr, gt_s, (lx,ly), scale=0.42,
                        color=(140,200,255), bg=(10,10,20), pad=3)
            ly += lspc

            # Quality bar
            bar_w = min(bw, 80)
            cv2.rectangle(bgr,(lx,ly),(lx+bar_w,ly+5),(30,40,50),-1)
            fill = int(bar_w*q)
            gc   = int(50+200*q)
            rc   = int(200*(1-q))
            cv2.rectangle(bgr,(lx,ly),(lx+fill,ly+5),(0,gc,rc),-1)
            put_text(bgr,f"Q{q:.2f}",(lx+bar_w+4,ly+6),scale=0.32,
                     color=TEXT_DIM, shadow=False)
            ly += 12

            # ── 8. Convergence sparkline ───────────────────────────────────────
            hist = live[idx].history
            if len(hist) > 4:
                spy = ly + 2
                spx = lx
                draw_sparkline(bgr, hist, spx, spy, w=min(bw,80), h=28,
                               color=color, bg=(12,15,22))
                put_text(bgr, "converge", (spx, spy-3),
                         scale=0.28, color=TEXT_DIM, shadow=False)

        # ── Header bar ────────────────────────────────────────────────────────
        cv2.rectangle(bgr,(0,0),(img_w,40),(10,12,18),-1)
        cv2.line(bgr,(0,40),(img_w,40),ACCENT,1)
        put_text(bgr,
                 f"  {session}   frame {frame_no:05d}  [{fi+1}/{total_frames}]"
                 f"   active: {len(active)}   "
                 f"LIVE = area\u00d7scale (live)  |  ML = Ridge model (trained)",
                 (4,27), scale=0.50, color=TEXT_MAIN)

        # ── Side HUD panel ────────────────────────────────────────────────────
        draw_side_panel(bgr, session, len(apples), fi+1, total_frames,
                        ml_pred, live, apples, panel_w=PANEL_W)

        # ── Downscale + write ─────────────────────────────────────────────────
        out_frame = cv2.resize(bgr,(out_w,out_h)) if args.scale!=1.0 else bgr
        writer.write(out_frame)

    writer.release()

    # ── Final console summary ─────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"SUMMARY — {session}")
    print(f"{'='*68}")
    print(f"  {'#':>3} {'L':>2} {'P':>2}  {'GT':>6}  {'LIVE':>7} {'LE':>6}  "
          f"{'ML':>7} {'ME':>6}")
    print(f"  {'-'*60}")
    for apple in sorted(apples,key=lambda a:a["apple_idx"]):
        idx = apple["apple_idx"]
        gt  = apple.get("gt_mm")
        lv  = live[idx].mm
        ml  = ml_pred.get(idx)
        le  = f"{lv-gt:+.2f}" if (lv and gt) else "  -- "
        me  = f"{ml-gt:+.2f}" if (ml and gt) else "  -- "
        lf  = "✓" if (lv and gt and abs(lv-gt)<=3) else "✗" if (lv and gt) else " "
        mf  = "✓" if (ml and gt and abs(ml-gt)<=3) else "✗" if (ml and gt) else " "
        ln  = apple["lane"]
        ps  = apple.get("pos_in_lane",apple.get("pos","?"))
        print(f"  #{idx+1:2d} L{ln} P{ps}  "
              f"{str(round(gt,1)) if gt else '--':>6}  "
              f"{(lv or 0):7.2f}{lf} {le:>6}  "
              f"{(ml or 0):7.2f}{mf} {me:>6}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
