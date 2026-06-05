# Final Implementation Plan: Apple Size Estimation
**Branch:** `feature/apple-size-estimation` (from `phase1/inference-tracking`)

---

## Background & What Changed Since v1

The original plan assumed a simple global `mm_per_px` constant. After studying Mizushima & Lu (2013) - co-authored by Dr. Lu - and confirming the actual lens hardware, the approach has been upgraded:

| Item | Original Plan | Revised Final Plan |
|---|---|---|
| Scale model | Global constant `mm_per_px` | **Position-dependent r(x, N)** - empirical quadratic per X position |
| Correction basis | Geometric `1/cos(theta)` only | **Empirical calibration balls** (primary) + geometric as cross-check |
| Apple height effect | Not handled | **Interpolated** between two ball curves (small + large) |
| Lens | Assumed 16 mm Edmund Optics | **Confirmed: JAI 0824-C3, 8 mm, f/2.4-16, -0.86% distortion** |
| f_px | 4,638 px | **2,319 px** (8 mm / 0.00345 mm pixel pitch) |

**Reference:** Mizushima & Lu (2013), Transactions of ASABE 56(3): 813-827. Their RMSE = **1.79 mm overall** - our benchmark to match or beat.

---

## Confirmed Hardware

| Component | Spec |
|---|---|
| Camera | JAI FSFE-3200T-10GE, Sony IMX252, 2048 × 1536 px |
| Lens | **JAI 0824-C3** - 8 mm focal length, f/2.4-f/16, C-mount, prism-optimized |
| Pixel pitch | 3.45 µm = 0.00345 mm (Sony IMX252) |
| Focal length in pixels | f_px = 8 / 0.00345 = **2,319 px** |
| TV distortion | **-0.86%** (negligible) |
| Conveyor | 3-lane screw conveyor, camera above center lane (Lane 2) |

---

## Architecture Overview

```
Camera frame
    │
    ▼
YOLO tracking (existing) → bounding box (x1,y1,x2,y2) + centroid (cx, cy)
    │
    ▼
[SizeEstimator module]
    ├── D_px_raw = min(box_w, box_h)          per frame
    ├── r(cx, D_px_raw) from calibration       scale function lookup
    └── D_mm_frame = D_px_raw × r              per frame mm estimate
    │
    ▼
Aggregation across all tracked frames → D_mm_peak (+ D_mm_median)
    │
    ├──► GradeRecord (at commit)
    ├──► CSV log columns
    ├──► Live video overlay ("ø 74 mm")
    └──► Offline LR validation script
```

---

## Proposed Changes

---

### 1. NEW: `gui/workers/size_calibration.py`

> [NEW] [size_calibration.py](file:///s:/MSU_Research/ASABE%20AIM26/apple_gui/gui/workers/size_calibration.py)

This is the **core new module**. Implements the Mizushima & Lu empirical variable-scale method.

```python
"""
gui/workers/size_calibration.py
================================
Empirical per-position scale calibration for apple diameter estimation.

Based on: Mizushima & Lu (2013), ASABE Transactions 56(3):813-827.
Method:   Roll two calibration balls of known diameter across conveyor.
          Fit quadratic N_px(x) = a*x^2 + b*x + c for each ball.
          Derive r(x) = (d/2) / sqrt(N_px(x))  [mm/pixel]
          Interpolate between curves for apples between ball sizes.
"""

class SizeCalibrator:
    """
    Holds the empirical scale function r(x, N_px) derived from calibration.

    Parameters loaded from config.yaml → size_estimation.calibration.
    """

    def __init__(self, small_ball_mm, small_coeffs,
                       large_ball_mm, large_coeffs,
                       img_width=2048,
                       min_diameter_px=40, max_diameter_px=500):
        # small/large: known diameters + quadratic coefficients [a, b, c]

    def r(self, cx: int, D_px: float) -> float:
        """
        Returns mm/pixel scale at centroid x=cx for an apple
        whose bounding box min-side is D_px pixels.
        Interpolates between small and large ball curves.
        Returns None if D_px is outside sanity clamps.
        """

    @classmethod
    def from_config(cls, cfg: dict) -> "SizeCalibrator":
        """Load calibration from config.yaml size_estimation block."""

    @classmethod
    def fit_from_data(cls, x_positions, pixel_counts, true_diameter_mm) -> tuple:
        """
        Fit quadratic to calibration ball data.
        Returns (a, b, c) coefficients.
        Call this from the calibration helper script.
        """
```

**Why a separate module:** keeps `tracker.py` clean; the calibrator can be swapped,
re-fit, or disabled without touching tracking logic.

---

### 2. MODIFY: `gui/workers/tracker.py`

> [MODIFY] [tracker.py](file:///s:/MSU_Research/ASABE%20AIM26/apple_gui/gui/workers/tracker.py)

#### 2a. `GradeRecord` dataclass - add size fields

```python
@dataclass
class GradeRecord:
    seq_id:      int
    lane:        int
    class_id:    int
    class_name:  str
    confidence:  float
    frames_seen: int
    # --- NEW ---
    size_px:     float        = 0.0   # peak bounding-box min-side (pixels)
    size_mm:     float | None = None  # converted mm (None if not calibrated)
    cf_factor:   float | None = None  # correction factor at commit position
```

#### 2b. `AppleTracker.__init__` - accept calibrator

```python
def __init__(self, ..., size_calibrator=None):
    self._sizer = size_calibrator   # SizeCalibrator | None
```

#### 2c. `_new_history()` - add size accumulator

```python
"box_sizes_px":   [],   # list of min(bw,bh) per frame
"box_sizes_mm":   [],   # list of D_mm per frame ([] if no calibrator)
```

#### 2d. Per-frame update loop - accumulate sizes

```python
bw = x2 - x1
bh = y2 - y1
D_px = min(bw, bh)
hist["box_sizes_px"].append(D_px)

if self._sizer:
    r = self._sizer.r(cx, D_px)
    if r is not None:
        hist["box_sizes_mm"].append(D_px * r)
```

#### 2e. Grade commit - compute final size

```python
# Size estimation at commit
sizes_px = hist["box_sizes_px"]
size_px  = max(sizes_px) if sizes_px else 0.0

size_mm, cf = None, None
if self._sizer and hist["box_sizes_mm"]:
    size_mm = max(hist["box_sizes_mm"])
    r_at_commit = self._sizer.r(cx, size_px)
    cf = r_at_commit

rec = GradeRecord(
    ...,
    size_px   = size_px,
    size_mm   = size_mm,
    cf_factor = cf,
)
```

#### 2f. `active` dict - expose current size for live overlay

```python
active.append({
    ...,
    "size_px":  max(hist["box_sizes_px"]) if hist["box_sizes_px"] else 0,
    "size_mm":  max(hist["box_sizes_mm"]) if hist["box_sizes_mm"] else None,
})
```

---

### 3. MODIFY: `config/config.yaml`

> [MODIFY] [config.yaml](file:///s:/MSU_Research/ASABE%20AIM26/apple_gui/config/config.yaml)

Replace the old `mm_per_px: null` stub with the full calibration block:

```yaml
# ══════════════════════════════════════════════════════════════
# SIZE ESTIMATION
# Empirical variable-scale method (Mizushima & Lu, 2013)
# Lens: JAI 0824-C3  f=8mm  pixel_pitch=3.45um  distortion=-0.86%
# ══════════════════════════════════════════════════════════════
size_estimation:
  enabled: true
  strategy: "peak"            # "peak" | "median" | "both"
  min_diameter_px: 40         # Reject boxes smaller than this (noise/partial)
  max_diameter_px: 500        # Sanity clamp

  calibration:
    # Run scripts/calibrate_size.py to populate these coefficients.
    # Until calibrated, size_mm will be null (px-only mode).
    small_ball_mm: 63.5       # Known diameter of small calibration ball (mm)
    small_coeffs: null        # [a, b, c] from quadratic fit - fill after calibration

    large_ball_mm: 76.2       # Known diameter of large calibration ball (mm)
    large_coeffs: null        # [a, b, c] from quadratic fit - fill after calibration

  # Geometric cross-check (theoretical, for paper validation)
  geometric:
    enabled: false            # Set true once H and X_lane are measured
    camera_height_mm: null    # H: measure tape from lens to belt surface
    lane_offset_mm: null      # X_lane: center of lane 1/3 from center lane
    focal_length_mm: 8.0      # Confirmed: JAI 0824-C3
    pixel_pitch_mm: 0.00345   # Sony IMX252
```

---

### 4. MODIFY: `gui/workers/inference_worker.py`

> [MODIFY] [inference_worker.py](file:///s:/MSU_Research/ASABE%20AIM26/apple_gui/gui/workers/inference_worker.py)

Wire the `SizeCalibrator` into the `RealInferenceWorker` → `AppleTracker` chain:

```python
from gui.workers.size_calibration import SizeCalibrator

# In RealInferenceWorker.__init__ or wherever AppleTracker is constructed:
size_cfg = config.get("size_estimation", {})
calibrator = None
if size_cfg.get("enabled") and size_cfg.get("calibration", {}).get("small_coeffs"):
    calibrator = SizeCalibrator.from_config(size_cfg)

tracker = AppleTracker(..., size_calibrator=calibrator)
```

---

### 5. NEW: `scripts/calibrate_size.py`

> [NEW] [calibrate_size.py](file:///s:/MSU_Research/ASABE%20AIM26/apple_gui/scripts/calibrate_size.py)

Supports **two calibration methods** - always try the Primary first:

---

#### PRIMARY METHOD: Existing apple videos + GT caliper data *(no new recordings needed - use this)*

```
python scripts/calibrate_size.py --method primary
                                 --video-ch1 videos/Source0/G1/G1.mp4
                                 --video-ch2 videos/Source1/G1/G1.avi
                                 --video-ch3 videos/Source2/G1/G1.avi
                                 --gt data/gt_diameters.csv
                                 --model models/best.pt
                                 --output config/config.yaml
```

**How it works:**
1. Runs YOLO tracking on the existing apple videos
2. Per tracked apple: extract centroid X (cx) and peak bounding-box min-side (D_px_peak)
3. Match tracker sequence IDs to GT caliper diameters from `gt_diameters.csv`
4. Split matched apples into small/large groups by GT diameter (split at median size)
5. Fit quadratic `N_px(x) = a*x^2 + b*x + c` for each group
6. Derive scale function r(x) = (d/2) / sqrt(N_px(x)) for each group
7. Write [a, b, c] coefficients and group diameters into `config.yaml`
8. Output fit quality plot (R²) + scatter of corrected vs GT diameter

**GT CSV format (convert from Excel - already have this for G1 session):**
```csv
apple_id, D1_mm, D2_mm, average_mm, surface_class
1, 57.27, 60.45, 58.86, 3
2, 61.11, 64.0,  62.555, 3
3, 66.06, 68.89, 67.475, 3
...
```
- `apple_id`: sequential number (1, 2, 3... as labeled in video)
- `D1_mm`, `D2_mm`: two caliper measurements
- `average_mm`: use this as the true GT diameter for fitting
- `surface_class`: GT grade - **1=Fresh, 2=Processing, 3=Cull** → convert to YOLO class_id via `class_id = surface_class - 1`

**Apple numbering scheme in video (confirmed from video annotation):**
```
Conveyor moves left → right. Looking at frame at t=0:46s:

  Lane 1 (top):    ... [4] [1] →  exit
  Lane 2 (middle): ... [5] [2] →  exit
  Lane 3 (bottom): ... [6] [3] →  exit

Apples numbered in columns of 3 (one per lane), right column = lower numbers.
Exit order: 1,2,3 roughly together, then 4,5,6, then 7,8,9 ...
```

**Matching tracker sequence IDs to GT apple numbers:**

The tracker assigns `seq_id` (1, 2, 3...) as apples pass through the counting gate.
Since apples enter the frame as columns-of-3, and the tracker counts them as they cross
the exit band, the mapping is roughly:
- Tracker seq_id 1,2,3  → GT apple_id 1,2,3
- Tracker seq_id 4,5,6  → GT apple_id 4,5,6
- etc.

However, exact ordering within a column-of-3 depends on which lane reaches the exit gate
first. The script will log the (seq_id, lane, cx, cy) at commit time - cross-check against
the video to verify ordering for the first ~6 apples, then assume the pattern holds.

> [!NOTE]
> If ordering within a group-of-3 is inconsistent, the script can offer an interactive
> matching mode: show each tracked apple's centroid/lane and let you confirm its GT ID.
> For the paper, even 30-40 well-matched apples give a robust calibration curve.

> [!NOTE]
> Real apples rotate on the screw conveyor, so their bounding box varies per frame.
> Using peak D_px per apple approximates (but may slightly underestimate) equatorial D.
> Expect more scatter in the fit than the Backup method (balls). Still valid if quadratic R² > 0.95.
> The downstream LR model (validate_size.py) corrects any remaining systematic bias.

---

#### BACKUP METHOD: Physical calibration balls *(only if Primary R² < 0.95)*

This is the original Mizushima & Lu (2013) approach. Only use this if the Primary method gives a poor quadratic fit.

```
python scripts/calibrate_size.py --method backup
                                 --video-small path/to/small_ball.mp4
                                 --ball-small-mm 63.5
                                 --video-large path/to/large_ball.mp4
                                 --ball-large-mm 76.2
                                 --model models/best.pt
                                 --output config/config.yaml
```

1. Get 2 balls: ~63.5 mm + ~76.2 mm (billiard balls work perfectly)
2. Roll each ball slowly across the full conveyor width while recording
3. Run this script - expect quadratic R² > 0.97 (balls are perfect spheres, no rotation noise)
4. Coefficients auto-written to `config.yaml`

**Use Backup when:** Primary quadratic R² < 0.95, or GT data cannot be matched per apple.

---

### 6. NEW: `scripts/validate_size.py`

> [NEW] [validate_size.py](file:///s:/MSU_Research/ASABE%20AIM26/apple_gui/scripts/validate_size.py)

Offline LR model validation for the paper. Loads logged CSV + GT caliper measurements.

```
Usage:
  python scripts/validate_size.py --csv data/session_log.csv
                                  --gt data/gt_diameters.csv
                                  --output docs/validation_results.png

Outputs:
  - R², RMSE, MAE, Bias
  - Predicted vs GT scatter plot (P vs D, matching chalkboard sketch)
  - Regression line: D_GT = a * D_mm_predicted + b
  - Target: R² > 0.99
```

---

### 7. MODIFY: CSV logger columns

> [MODIFY] `core/` or wherever CSV logging is implemented

Add to `csv_columns` in `config.yaml` (already has `diameter_px`):

```yaml
csv_columns:
  - timestamp
  - apple_id
  - lane
  - grade
  - confidence
  - size_px              # raw peak bounding-box min-side (pixels)
  - size_mm              # converted mm after empirical scale correction
  - cf_factor            # scale factor r(x) applied (for debugging)
  - conveyor_speed_aps
  - outlet_fired
```

---

### 8. Live GUI Overlay

> [MODIFY] wherever bounding box labels are drawn in the GUI

Add diameter annotation below the grade label on the video feed:

```
[#5] Fresh 91%
ø 83 px  |  ø 74 mm
```

- Show `px` always (available immediately)
- Show `mm` only if calibration coefficients are loaded
- Use small font, same color as grade label, no clutter

---

## Verification Plan

### Phase 1 - Code correctness (simulation mode)
- [ ] `pytest tests/` - no regressions in tracking or grading
- [ ] Unit test: synthetic boxes at known X positions → assert `size_px` correct
- [ ] Unit test: `SizeCalibrator.r(x, N)` with hand-computed coefficients → assert mm output
- [ ] Run GUI in simulation mode → each apple shows `size_px > 0` in overlay and CSV

### Phase 2 - Calibration

**PRIMARY METHOD (existing videos + GT - no new lab session needed):**
- [ ] Confirm GT CSV format: `apple_id, D1_mm, D2_mm, average_mm, surface_class`
- [ ] Run `calibrate_size.py --method primary` on existing 3-channel G1 videos
- [ ] Check quadratic fit R² > 0.95 for both small/large size groups
- [ ] Check fit scatter plot - smooth curve with moderate spread is expected
- [ ] Coefficients auto-written to `config.yaml`
- [ ] Run GUI in simulation with those videos → check `size_mm` values in 50-100 mm range

**BACKUP METHOD (physical balls - only if Primary R² < 0.95):**
- [ ] Get 2 balls: ~63.5 mm + ~76.2 mm (billiard balls)
- [ ] Roll each ball across full conveyor width while recording video
- [ ] Run `calibrate_size.py --method backup` → verify R² > 0.97
- [ ] Coefficients auto-written to `config.yaml`
- [ ] Also measure H (camera height) and X_lane (lane offset) for geometric cross-check

### Phase 3 - Validation against GT (paper)
- [ ] Run `validate_size.py` on logged CSV + GT diameters
- [ ] Check R² > 0.99, RMSE < 1.79 mm (beat Mizushima benchmark)
- [ ] Generate P vs D scatter plot for paper (matches Dr. Lu's chalkboard sketch)

---

## Open Questions (Resolved)

| Question | Answer |
|---|---|
| Lens focal length? | **8 mm (JAI 0824-C3)** - confirmed from physical lens |
| Aggregation strategy? | **Peak** across frames (matches USDA equatorial D definition) |
| Calibration method? | **Mode B (existing G1 video + GT Excel data)** - try first; Mode A (balls) as fallback |
| f_px? | **2,319 px** (8 mm / 0.00345 mm) |
| GT format? | **Confirmed** - Excel: apple_id, D1, D2, Average, Surface Class; use Average column |
| GT matching? | Tracker seq_id matches GT apple_id column-of-3 at a time; verify first 6 manually |
| Surface Class meaning? | **Confirmed:** GT Excel 1=Fresh, 2=Processing, 3=Cull → YOLO class_id = surface_class - 1 |
| Camera height H? | TBD - measure in lab (needed for geometric cross-check only) |
| Lane offset X_lane? | TBD - measure in lab (needed for geometric cross-check only) |
