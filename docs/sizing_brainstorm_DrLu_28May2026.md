# Apple Size Estimation - Research & Engineering Brainstorm
**Michigan State University · 28 May 2026**
**Meeting with Dr. Lu - Notes, Analysis & Plan**

---

## 1. Meeting Notes (Dr. Lu, 28 May 2026)

### What was on the chalkboard (interpreted)

```
   P (predicted)
   │       /
   │      /  ← linear trend, R > .99 target
   │     /
   │    /
   └────────── M (measured / GT)

   [Apple ①]            [Apple ②]
   Directly below cam   Side lane, offset from nadir
   Circle → diameter D  Distorted → ellipse-like projection
   Scale uniform here   Scale compressed here (same D, fewer px)
```

The diagram captures two key ideas:
1. **LR model**: plot predicted pixel-diameter vs GT measured diameter → want R² > 0.99.
2. **Geometric distortion**: the same apple looks different in pixels depending on whether it sits
   under the camera (center lane) or off to the side (outer lanes).

### Dr. Lu's three points, verbatim

| # | Point | Key takeaway |
|---|---|---|
| 1 | "How may we calculate the size?" | Method discussion: bounding-box pixel width vs GT |
| 2 | "We have GTs where size in D is given - we can build an LR model and need R²" | We have physical ground-truth diameters; fit pixel to mm regression and validate |
| 3 | "A previous student did sizing but did NOT account for camera-position correction - apples on side lanes look different than center because camera is mounted right above center row" | Critical gap. We must derive and implement a per-lane (or per-pixel-position) perspective correction factor |

---

## 2. What We Have Right Now

### 2.1 Pipeline
- **JAI FSFE-3200T-10GE** - 3-sensor prism camera, 2048 × 1536 px, 16 mm Edmund Optics VIS-NIR lens
- **YOLO tracking** (`tracker.py`) → bounding box `(x1, y1, x2, y2)` per apple per frame, already committed
- **`GradeRecord`** dataclass + CSV schema already has a reserved `diameter_px` column
- **3-lane screw conveyor**, camera mounted **above the center lane** (lane 2)

### 2.2 Ground Truth (GT) Data
- Physical apple diameters measured with calipers: size given as **D (mm)**
- This is our Y (truth) for the regression model

### 2.3 What the tracker already gives us
```python
active[i] = {
    "track_id":  tid,
    "seq_id":    seq_id,
    "class_id":  disp_cls,
    "box":       (x1, y1, x2, y2),   # ← bounding box pixels
    "center":    (cx, cy),            # ← pixel center of apple
    "lane":      lane,                # ← 1 / 2 / 3
    ...
}
```
The bounding box is there. We just haven't used it for sizing yet.

---

## 3. What We Need to Build

### Summary map

```
Raw YOLO box (px)
      │
      ▼
[Step A] Per-frame diameter in pixels
         min(box_w, box_h) → D_px_raw
      │
      ▼
[Step B] Perspective correction factor CF(cx, cy)
         correct for camera position above center lane
         D_px_corrected = D_px_raw × CF
      │
      ▼
[Step C] Convert to mm
         D_mm = D_px_corrected × mm_per_px_center
      │
      ▼
[Step D] Aggregation across frames
         peak / median across all frames apple was tracked
      │
      ▼
[Step E] LR model: D_mm_predicted → D_GT regression
         compute R², slope, intercept
      │
      ▼
[Step F] Log to CSV + display in GUI overlay
```

---

## 4. Step-by-Step Technical Design

---

### STEP A - Per-Frame Pixel Diameter

**Why `min(box_w, box_h)`?**

The apple is roughly spherical. The YOLO bounding box is axis-aligned. When an apple is near
circular in the image, `min(w, h)` approximates the equatorial diameter and is least affected by
perspective tilt (which elongates one axis). Peak across frames (Step D) then catches the widest
equatorial view.

```python
box_w = x2 - x1
box_h = y2 - y1
D_px_raw = min(box_w, box_h)
```

**Sanity clamps** (configurable):
```yaml
size_estimation:
  min_diameter_px: 40    # < 40 px likely noise / partial detection
  max_diameter_px: 500   # > 500 px is a sensor artifact
```

---

### STEP B - Perspective Correction Factor (THE CRITICAL NEW PART)

#### The physical setup

```
                 Camera (height H above conveyor)
                        │
             ┌──────────┼──────────┐
             │          │          │
           Lane 1    Lane 2     Lane 3
          (offset   (center,  (offset
          -X_lane)  X=0)      +X_lane)
```

The camera optical axis points **straight down** to lane 2 (center).
Lanes 1 and 3 are at physical horizontal offset ±X_lane mm from center.

#### Why does this matter?

A standard perspective (pinhole) camera does NOT have uniform spatial scale across the image.
The mm-per-pixel ratio changes depending on how far the image point is from the optical axis.

For a camera at height H, focal length f, pixel pitch p:
- **At image center (lane 2):**
  - Distance from camera to apple surface ≈ H
  - Local scale S_center = H × p / f  [mm / px]

- **At lateral offset X_lane (lanes 1 & 3):**
  - The ray to that apple makes angle θ = arctan(X_lane / H) with the optical axis
  - True distance from camera: r = H / cos(θ) = √(H² + X_lane²)
  - Local scale S_side = r × p / f = √(H² + X_lane²) × p / f

The **same apple** of true diameter D_mm will span:
- In lane 2: D_px_center = D_mm / S_center  (more pixels, closer / more magnified)
- In lane 1/3: D_px_side = D_mm / S_side    (fewer pixels, farther / less magnified)

Ratio: D_px_center / D_px_side = S_side / S_center = √(H² + X_lane²) / H = 1/cos(θ)

**This means side-lane apples appear SMALLER in pixels than center-lane apples of the same size.**

#### The Correction Factor

To bring all lanes to the **equivalent center-lane pixel scale**:

```
CF(lane) = √(H² + X_lane²) / H   =   1 / cos(θ_lane)

D_px_corrected = D_px_raw × CF(lane)
```

Where:
- Lane 2 (center): CF = 1.0 (no correction needed)
- Lanes 1 & 3 (side): CF > 1.0 (scale up to match center)

> **Example:**
> H = 600 mm, X_lane = 120 mm (lane spacing 120 mm from center)
> θ = arctan(120/600) = 11.3°
> CF = 1/cos(11.3°) = 1.020 → 2% correction per side lane
>
> If H = 400 mm, X_lane = 180 mm: CF = 1/cos(24.2°) = 1.097 → ~10% correction

#### More precise: pixel-position-based correction

We don't have to restrict this to per-lane. Since we know the **pixel center (cx, cy)** of every
apple, and we know the camera intrinsics, we can compute a **per-apple** correction factor:

```python
# cx_center = image width / 2 (optical axis in pixels)
# cy_center = image height / 2
# f_px = focal length in pixels = (f_mm / p_mm)

dx_px = cx - cx_center   # pixel offset from optical axis in X
dy_px = cy - cy_center   # pixel offset in Y (conveyor direction, less relevant)

# Angle from optical axis
theta_x = arctan(dx_px / f_px)   # horizontal angle
theta_y = arctan(dy_px / f_px)   # vertical angle
theta   = arctan(sqrt(dx_px**2 + dy_px**2) / f_px)  # total angle

CF = 1.0 / cos(theta)
D_px_corrected = D_px_raw * CF
```

This is **fully general** - works for any apple position, not just lane centers.

#### What we need to know to implement this

| Parameter | Symbol | How to get it |
|---|---|---|
| Camera height above conveyor | H (mm) | **Measure physically** (tape measure) |
| Horizontal lane offset from center | X_lane (mm) | **Measure physically** (lane spacing) |
| Focal length | f (mm) | **16 mm** (Edmund Optics lens - confirmed) |
| Pixel pitch | p (mm) | **3.45 µm** = 0.00345 mm (Sony IMX252) |
| Focal length in pixels | f_px = f/p | 16 / 0.00345 ≈ **4,638 px** |
| Image center | (cx0, cy0) | 2048/2, 1536/2 = **(1024, 768)** |

**The only unknowns are H and X_lane - both physical measurements we take in the lab.**

---

### STEP C - Converting Pixels to mm

Once corrected to the equivalent center-lane scale:
```
D_mm = D_px_corrected × S_center
     = D_px_corrected × (H × p / f)
     = D_px_corrected × (H_mm × 0.00345 / 16)
```

This collapses to a single scalar once H is measured.

Alternatively, compute `mm_per_px` from GT calibration directly (Step E gives us this as the slope
of the regression line - no need to measure H explicitly if the LR is good enough).

---

### STEP D - Aggregation Strategy

The apple rotates on the screw conveyor so its bounding box dimension changes frame by frame.
Options:

| Strategy | Formula | Pros | Cons |
|---|---|---|---|
| **Peak** | max(D_px per frame) | Best represents equatorial D; partial occlusion only causes underestimates | One noisy frame can inflate |
| **Median** | median(D_px across frames) | Robust to outliers | May underestimate if apple often tilted |
| **Trimmed mean** | mean of middle 60% of frames | Best of both | Slightly more complex |

**Recommendation: Peak for initial study (matches GT caliper equatorial D), then cross-check
with median to assess rotation-induced variance.**

Track per-frame sizes in `hist["box_sizes"]` list, compute at grade commit time.

---

### STEP E - LR Model and R² Validation

#### What we build

A simple **Ordinary Least Squares (OLS)** linear regression:

```
D_mm_predicted = slope × D_GT_mm + intercept
```

Or equivalently (since we predict from pixel data):
```
D_GT_mm = a × D_px_corrected + b   ← this is what we fit
```

Then use it to predict GT from pixel measurements.

#### Metrics to report (for the paper)

| Metric | Formula | Target |
|---|---|---|
| **R²** | 1 - SS_res/SS_tot | > 0.99 (Dr. Lu's target) |
| **RMSE** | √(mean((pred - GT)²)) | < 2 mm |
| **MAE** | mean(\|pred - GT\|) | < 1.5 mm |
| **Slope** | Should be ≈ 1.0 if calibrated | |
| **Bias** | Mean(pred - GT) | Should be ≈ 0 |

#### Where to compute this

- **Offline analysis script** (not in real-time GUI): load the logged CSV with `diameter_px`
  and GT values, apply correction, fit LR, plot, report R².
- **Possible addition:** a `scripts/size_validation.py` script that loads a CSV + GT file and
  outputs the regression stats + plot.

---

### STEP F - Integration into GUI and CSV

#### CSV logging additions
The config already has `diameter_px`. We add:
```yaml
csv_columns:
  - diameter_px           # raw pixel min-side (uncorrected)
  - diameter_px_corrected # after CF correction
  - diameter_mm           # converted to mm (if H known)
  - cf_factor             # correction factor applied (useful for debugging)
```

#### Live overlay
Add diameter annotation next to grade label on the video feed:
```
[#3] Fresh 94%
ø 87 px → 72 mm
```

---

## 5. Unknowns & What We Need to Measure

| Item | Status | Who / When |
|---|---|---|
| Camera height H above conveyor (mm) | ❌ Not yet measured | Haseeb — next lab visit |
| Lane 1 / Lane 3 physical offset from center lane (mm) | ❌ Not yet measured | Haseeb — next lab visit |
| GT apple diameters dataset (CSV) | ✅ Exists | Already available from Dr. Lu's lab |
| Confirm f = 16 mm lens (Edmund Optics) | ✅ Confirmed from hardware docs | |
| Sony IMX252 pixel pitch = 3.45 µm | ✅ Confirmed from datasheet | |

---

## 6. Comparison: Our Approach vs. Previous Student's Approach

| Aspect | Previous Student | Our Approach |
|---|---|---|
| Pixel diameter source | Bounding box (similar) | Bounding box |
| Aggregation | Unknown | Peak (+ median cross-check) |
| Lane correction | ❌ Not applied | ✅ CF = 1/cos(θ_lane) per apple |
| Per-pixel correction | ❌ No | ✅ Per-apple (cx, cy) based |
| LR + R² validation | Unknown | ✅ Required, target R² > 0.99 |
| Calibration | Simple mm_per_px global | ✅ Position-aware scale |
| Paper contribution | Sizing only | Sizing + geometric correction method |

The correction factor is our **primary methodological contribution** over prior work.

---

## 7. Implementation Roadmap

```
Phase A — Measure hardware (Lab visit)
  ├── Measure H (camera height above conveyor, mm)
  ├── Measure X_lane (lane spacing, center-to-side, mm)
  └── Compute: f_px = 16/0.00345 ≈ 4638 px  [already known]

Phase B — Code: tracker.py
  ├── Add box_sizes[] accumulator to history
  ├── Compute D_px_raw = min(box_w, box_h) each frame
  ├── Compute CF(cx, cy) per apple
  ├── At grade commit: peak_px, corrected_px, mm (if H known)
  └── Add size fields to GradeRecord + active dict

Phase C — Code: config.yaml
  └── Add size_estimation block (H, X_lanes, strategy, min/max clamp)

Phase D — Code: GUI overlay
  └── Show ø XX px / YY mm next to grade label on video feed

Phase E — Code: CSV logger
  └── Add diameter_px / corrected / mm / cf columns to log

Phase F — Analysis script
  ├── scripts/size_validation.py
  ├── Load CSV + GT file
  ├── Fit OLS: D_GT = a × D_px_corrected + b
  ├── Compute R², RMSE, MAE, bias
  └── Generate plot: predicted vs GT (P vs D scatter, R > .99 line)
```

---

## 8. Open Questions (to resolve with Dr. Lu)

1. **Do you want lane-based CF or per-apple pixel-position CF?**
   Per-pixel is more accurate but requires knowing camera intrinsics precisely. Lane-based is
   simpler and sufficient for 3-lane operation. We can do both and compare.

2. **GT data format?** Is it a CSV with (apple_id, D_mm) or labeled by session? Need to know
   how to join it with our tracker output (by apple sequence number or by batch).

3. **Is the LR model meant to be fixed post-calibration** (just validate it once) or
   **adaptive** (update as more GT samples come in)?

4. **What is the approximate camera height H?** Even a rough estimate helps us assess how
   significant the correction will be (small H + wide lanes = large correction needed).

5. **Do we report diameter of the whole apple or just the cross-sectional equatorial diameter?**
   From a top-down camera, we're measuring the **maximum cross-section** (equatorial D), which is
   the same as what calipers measure at the widest point - so they should match well.

---

## 9. Key Equations Reference Card

```
# 1. Raw pixel diameter (per frame)
D_px_raw = min(x2 - x1, y2 - y1)

# 2. Perspective correction factor (per apple, most general)
f_px    = f_mm / p_mm            # focal length in pixels: 16 / 0.00345 ≈ 4638
dx      = cx - (img_width  / 2)  # pixel offset from optical axis
dy      = cy - (img_height / 2)
theta   = arctan(sqrt(dx² + dy²) / f_px)
CF      = 1.0 / cos(theta)

# Simplified per-lane version:
theta_lane = arctan(X_lane_mm / H_mm)
CF_lane    = 1.0 / cos(theta_lane)

# 3. Corrected pixel diameter
D_px_corrected = D_px_raw × CF

# 4. Convert to mm
S_center = H_mm × p_mm / f_mm   # mm per pixel at center
D_mm     = D_px_corrected × S_center

# 5. Aggregation (peak over tracked frames)
D_px_final = max(hist["box_sizes"])

# 6. LR validation
D_GT_mm = a × D_px_corrected_final + b
R² = 1 - SS_res / SS_tot
```

---

*Document created: 28 May 2026 - Haseeb Bajwa, MSU*
*Branch: feature/apple-size-estimation*
