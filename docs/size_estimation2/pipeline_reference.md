# Apple GUI - Full Pipeline Reference
**Michigan State University | ASABE AIM 2026 | `feature/apple-size-ml` branch**

---

## At a Glance

```
Video files (G:\Haseeb\pic\)
        │
        ▼  Step 1
  extract_frames.py  ──────────────────►  G1.pkl … G10.pkl
        │                                 (D:\HA\apple_gui\data\frame_features\)
        │  uses:
        │   core/sizing/mask_diameter.py  (4 diameter methods per frame)
        │   core/camera/camera_interface.py
        │
        ▼  Step 2 (inside extract_frames)
  Per-frame features stored in pkl
  (d_area, d_maxw, d_sym, d_ell, quality, cx_px, bbox, consensus_params…)
        │
        ▼  Step 3 (inside training script)
  core/sizing/view_fusion.py  ──────────►  10 features per apple
        │
        ▼  Step 4
  train_size_regressor.py  ─────────────►  models/size_model.pkl
                                           data/ml_results/fig1…fig5, CSV
        │
        ▼  Evaluation
  compute_paper_accuracy.py  ───────────►  paper-formula accuracy per session
        │
        ▼  Visualization
  visualize_session.py  ────────────────►  data/viz/*.mp4  (HUD video)
```

---

## Repo Layout

```
apple_gui/
│
├── scripts/                    ← ALL runnable scripts (run these)
│   ├── extract_frames.py       ← STEP 1: video → pkl
│   ├── train_size_regressor.py ← STEP 4: training + 5 figures + CSV
│   ├── compute_paper_accuracy.py ← EVALUATION: paper-formula accuracy
│   ├── visualize_session.py    ← VISUALIZATION: HUD video output
│   ├── fit_scale.py            ← utility: diagnose scale factor per session
│   ├── validate_step1.py       ← utility: sanity check pkl outputs
│   ├── compare_methods_G1.py   ← utility: compare 4 diameter methods on G1
│   ├── camera_live_view.py     ← camera: live feed from JAI camera
│   └── camera_probe_jai.py     ← camera: probe JAI camera settings
│
├── core/                       ← library code (imported by scripts)
│   ├── sizing/
│   │   ├── mask_diameter.py    ← 4 per-frame diameter methods (M1-M4)
│   │   └── view_fusion.py      ← fuse frames → 10 features per apple
│   └── camera/
│       └── camera_interface.py ← JAI camera driver (eBUS SDK wrapper)
│
├── models/
│   ├── best.pt                 ← YOLOv11 segmentation model (52MB)
│   └── size_model.pkl          ← trained Ridge regressor (tiny, 1.7KB)
│
├── data/
│   ├── gt.xlsx                 ← ground truth caliper measurements
│   ├── frame_features/         ← G1.pkl … G10.pkl (output of extract_frames)
│   └── ml_results/             ← fig1-fig5, training_metrics.csv
│
└── config/                     ← YAML config files (camera, paths, thresholds)
```

---

## Step-by-Step Execution

### STEP 0 - One-time setup
```bash
conda activate applegui
python scripts/verify_env.py        # check all packages installed
```

---

### STEP 1 - Extract frame features from video
**Script:** `scripts/extract_frames.py`

**What it does:**
- Reads video file (MP4/AVI) frame by frame
- Runs YOLOv11 (`models/best.pt`) on each frame → gets apple masks
- Runs ByteTrack tracker → assigns each detection to an apple ID
- For each apple in each frame, calls `core/sizing/mask_diameter.py`:
  - M1 MaxWidth, M2 Symmetry, M3 Ellipse, M4 Area → d_maxw, d_sym, d_ell, d_area
- Filters ghost detections (must travel ≥80px horizontally)
- Computes consensus ellipse from central 60% of traversal frames
- Saves everything to a `.pkl` file

**Run:**
```bash
python scripts/extract_frames.py \
  --session G10 \
  --video "G:\Haseeb\pic\G10.MP4" \
  --gt    "D:\HA\apple_gui\data\gt.xlsx" \
  --out   "D:\HA\apple_gui\data\frame_features"
```

**Input:**  Video file + `gt.xlsx`
**Output:** `data/frame_features/G10.pkl`

---

### STEP 2 - (Inside Step 1) Per-frame diameter
**Module:** `core/sizing/mask_diameter.py`

Called automatically by `extract_frames.py`. Not run directly.

| Method | Key function | Output field |
|--------|-------------|--------------|
| M1 MaxWidth | rotate mask, find widest | `d_maxw` |
| M2 Symmetry | Mizushima-Lu symmetry axis | `d_sym` |
| M3 Ellipse | `cv2.fitEllipse` major axis | `d_ell` |
| M4 Area | `√(4×area/π)` | `d_area` |

All values in **pixels** at this stage. No mm conversion yet.

---

### STEP 3 - (Inside training) Frame fusion
**Module:** `core/sizing/view_fusion.py`

Called automatically by `train_size_regressor.py`. Can also be imported directly.

**What it does:**
- Filters to central 60% of each apple's traversal (skip edge frames)
- For each of 4 methods: quality-weighted mean + alpha-trimmed peak
- Adds consensus ellipse axes (`ell_a`, `ell_b`) from `consensus_params`
- Adds `mean_Q` (average quality) and `lane` (0/1/2)
- Output: **10 features per apple**

```python
FEATURE_COLS = [
    "d_area_wmean", "d_maxw_wmean", "d_sym_wmean", "d_ell_wmean",
    "d_area_peak",  "d_maxw_peak",
    "ell_a", "ell_b",
    "mean_Q", "lane"
]
```

---

### STEP 4 - Train ML model
**Script:** `scripts/train_size_regressor.py`

**What it does:**
- Loads all session pkls (G1-G10)
- Runs `view_fusion` → 10 features per apple
- Excludes apples with `cx_range < 1000px` (partial traversal)
- Trains Ridge, RandomForest, GradientBoost
- Leave-One-Session-Out CV on training sessions (G1-G9)
- Blind test on G10
- Saves best model + 5 figures + CSV

**Run:**
```bash
python scripts/train_size_regressor.py
```

**Input:**  `data/frame_features/G*.pkl`  +  GT inside each pkl
**Output:**
```
models/size_model.pkl
data/ml_results/
  ├── fig1_loo_scatter.png         ← training scatter all sessions
  ├── fig2_loo_per_session.png     ← per-session bars + per-apple errors
  ├── fig3_blind_test.png          ← G10 blind test scatter + error bars
  ├── fig4_model_insight.png       ← Ridge coefficients + residuals + Q-Q
  ├── fig5_r2_explanation.png      ← why R² differs from paper
  └── training_metrics.csv
```

**Sessions used:**

| Sessions | Role |
|---|---|
| G1, G2, G3, G4, G5, G6, G8, G9 | Training (140 apples) |
| G10 | Blind test (17 apples) |
| G7 |  Skipped - incomplete video |
| G11 |  Excluded - GT labeling error (apples #1/#4 swapped) |

---

### EVALUATION - Paper-formula accuracy
**Script:** `scripts/compute_paper_accuracy.py`

**What it does:**
- Uses saved `size_model.pkl` (no retraining)
- Runs same `fuse_session` as training
- Computes per-apple accuracy = `(1 - |pred-gt|/gt) × 100%`  ← exact Lu 2025 formula
- Compares LIVE estimate vs ML vs paper

**Run:**
```bash
python scripts/compute_paper_accuracy.py
```

**Input:**  `models/size_model.pkl`  +  `data/frame_features/G*.pkl`
**Output:**  Prints table to console (no files saved)

---

### VISUALIZATION - HUD video
**Script:** `scripts/visualize_session.py`

**What it does:**
- Replays original video frames
- Overlays real consensus ellipse (from pkl)
- Shows LIVE estimate vs ML prediction per apple
- Animated progress arc, lane indicator, size HUD

**Run:**
```bash
python scripts/visualize_session.py \
  --session  G1 \
  --data_root "G:\Haseeb\pic" \
  --pkl      "D:\HA\apple_gui\data\frame_features\G1.pkl" \
  --model    "D:\HA\apple_gui\models\size_model.pkl" \
  --out      "D:\HA\apple_gui\data\viz" \
  --max_frames 800
```

**Input:**  Video + pkl + size_model.pkl
**Output:** `data/viz/G1_viz.mp4`

---

## Where Data Lives (Shuttle PC)

| What | Path |
|------|------|
| Raw videos | `G:\Haseeb\pic\G1.MP4` … `G10.MP4` |
| Ground truth | `D:\HA\apple_gui\data\gt.xlsx` |
| Session pkls | `D:\HA\apple_gui\data\frame_features\G1.pkl` … `G10.pkl` |
| YOLO model | `D:\HA\apple_gui\models\best.pt` |
| Size model | `D:\HA\apple_gui\models\size_model.pkl` |
| Training figures | `D:\HA\apple_gui\data\ml_results\` |
| Viz videos | `D:\HA\apple_gui\data\viz\` |
| Code repo | `D:\HA\apple_gui\` |

---

## Key Results

| Metric | LOO-CV (training) | Blind G10 (test) | Paper (Lu 2025) |
|--------|-------------------|------------------|-----------------|
| MAE | 0.905mm | 1.009mm | - |
| RMSE | 1.117mm | **1.703mm** | 1.870mm |
| R² | 0.918 | 0.839 | 0.967* |
| Accuracy (paper formula) | 98.65% | **98.44%** | 97.60% |
| Within ±3mm | 137/140 (97.9%) | 15/17 (88.2%) | - |

*Paper's higher R² is because their test set had std≈10.3mm vs our 4.2mm - see fig5.

---

## What Is Still Needed

| Task | Priority | Details |
|------|----------|---------|
| **Fix G11 GT labels** | High | Apples #1 and #4 were measured in wrong order in Excel. Re-verify with Dr. Lu's group, then retrain → adds 16 more training apples |
| **8mm lens** | High | 2 failing apples in G10 are both Lane 2 → lens edge distortion. 8mm lens has much less barrel distortion |
| **Collect more sessions** | Medium | More data = better generalization. Target: 15+ sessions |
| **Real-time integration** | Future | Connect `size_model.pkl` into `camera_live_view.py` for live belt sizing |

---

*Last updated: June 2026 | Pipeline fully operational on shuttle PC | `feature/apple-size-ml` branch*
