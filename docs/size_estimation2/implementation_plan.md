# Apple Size Estimation - Geometry and ML Approach
## (No stem/calyx model, no belt height required)

---

## What We Have Right Now

- Videos: G1 to G11 sessions, 6mm lens, 3 channels (R, B, NIR1 composite)
- GT data: `data/gt.xlsx` with caliper measurements for all 198 apples
- YOLO tracker: already running, produces per-frame masks and tracks
- Branch: `feature/apple-size-ml` (clean, starts from inference-tracking)

## What We Do NOT Need

- Stem/calyx trained model (comes later from Jiajing dataset)
- Belt height (unknown, replaced by GT-fitted scale factor)
- Any new hardware or new recordings

---

## Methods (No Stem Required)

### Method 1 - Max Width Projection
For each frame, project the mask onto every angle 0 to 180 degrees.
Measure the width at each angle. The maximum width is the diameter estimate
for that frame. The equatorial diameter is always the widest cross-section
physically, so this is grounded in geometry.

### Method 2 - Contour Symmetry (Mizushima and Lu 2013)
Build the radius function d(k) = distance from centroid to contour point k.
Cross-correlate the left and right halves of d(k) at every split position x.
The peak of the cross-correlation gives the axis of bilateral symmetry.
Diameter perpendicular to that axis is the equatorial diameter estimate.
No stem detection needed. Runs at approximately 0.5ms per frame.

### Method 3 - Ellipse Major Axis
Fit an ellipse to the contour points using OpenCV fitEllipse.
Use the major axis length as the diameter. Simple and always available.
Works well for round apples. Less accurate for elongated varieties when
the apple is not in an equatorial view.

### Method 4 - Area-based Sphere Estimate
D = sqrt(4 * Area / pi), where Area is the mask pixel count.
This assumes the apple is a sphere and the view is equatorial.
Used as a sanity check and low-weight ensemble member.

### Method 5 - ML Feature Regression (primary final estimate)
Extract a feature vector per apple from all frame-level estimates:
- Quality-weighted mean of each method
- Maximum value of each method across frames
- Frame count, mean quality score, spread of estimates
Train a Ridge or Random Forest regression on G1-G9 with GT labels.
Scale is learned implicitly from the training data.
No belt height needed. Validate on G10-G11 blind.

---

## Three-Stage Pipeline

```
STAGE 1 - Per Frame (runs on every frame of a tracked apple)

  Input: Binary mask from YOLO tracker

  Compute:
    D_maxwidth    = max projection width across 180 angles
    D_symmetry    = diameter from contour cross-correlation symmetry
    D_ellipse     = major axis of fitted ellipse
    D_area        = sqrt(4 * pixel_area / pi)
    Q             = frame quality score

  Quality score Q components:
    circularity   = 4 * pi * Area / Perimeter^2   (1.0 = perfect circle)
    completeness  = 1 if mask does not touch frame edge, else 0.5
    Q             = circularity * completeness

STAGE 2 - Cross-Frame Fusion (one result per apple)

  For each method:
    D_mean_m    = sum(Q_i * D_m_i) / sum(Q_i)   quality-weighted mean
    D_peak_m    = max(D_m_i)                      peak across all frames

  Per method combined estimate:
    D_m = alpha * D_mean_m + (1 - alpha) * D_peak_m
    (alpha auto-tuned on G1-G9 GT, expected around 0.7)

STAGE 3 - Final Size Estimate

  Option A (Geometric): best single method * scale_factor
    scale_factor = fit from G1-G9 GT in one linear regression pass

  Option B (ML): Ridge or RF regression
    Features: [D_mean_maxwidth, D_peak_maxwidth, D_mean_sym, D_peak_sym,
               D_mean_ellipse, D_peak_ellipse, D_mean_area, mean_Q,
               frame_count, spread_of_estimates]
    Target: D_mm from GT caliper
    Train: G1-G9    Validate: G10-G11
```

---

## Data Split

| Split | Sessions | Apples | Purpose |
|---|---|---|---|
| Train + tune | G1 to G9 | 162 | Fit scale, tune alpha, train ML |
| Test (blind) | G10, G11 | 36 | Final R2 and RMSE report |

---

## Implementation Steps (in order)

### Step 1 - Frame Extractor
`scripts/extract_frames.py`
- Run YOLO tracker on each session video (G1 to G11)
- For each committed apple track: save (frame_index, mask, cx, cy, area)
- Match committed apples to GT by order (seq_id 1..18 per session)
- Output: pickle or CSV of per-frame data, one file per session

### Step 2 - Per-Frame Diameter Module
`core/sizing/mask_diameter.py`
- Functions: `max_width(mask)`, `symmetry_diameter(mask)`, 
  `ellipse_diameter(mask)`, `area_diameter(mask)`, `quality_score(mask)`
- Each function takes a binary numpy mask, returns diameter in pixels
- Fully unit-testable with synthetic data

### Step 3 - Cross-Frame Fusion Module
`core/sizing/view_fusion.py`
- Takes list of (D_per_method, Q) across frames
- Returns quality-weighted mean, peak, and combined estimate per method
- Alpha auto-tuned by grid search on G1-G9

### Step 4 - Scale Fitting and Geometric Evaluation
`scripts/fit_scale.py`
- Loads frame data for G1-G9
- Runs fusion, gets D_px per apple
- Fits scale = sum(D_px * D_gt) / sum(D_px^2)
- Evaluates each method: R2, RMSE, accuracy vs GT
- Picks best geometric method

### Step 5 - ML Regression
`scripts/train_size_regressor.py`
- Builds feature matrix from G1-G9 frame data
- Trains Ridge and RandomForest regressors
- Cross-validates within G1-G9
- Saves best model

### Step 6 - Final Evaluation (blind test)
`scripts/evaluate_size.py`
- Runs best geometric method and ML model on G10-G11
- Reports R2, RMSE, MAE, accuracy within 3mm
- Saves P-vs-D scatter plot
- These are the paper-ready numbers

### Step 7 - GUI Integration (after Step 6 passes)
- Wire best model into `tracker.py` and `main_window.py`
- Live overlay: `o 65mm` instead of `o 164px`
- CSV logger updated

---

## Later (not now)

- Stem/calyx model from Jiajing dataset (when data arrives, check camera compatibility)
- Belt height determination (if someone measures it)
- 8mm lens production validation (Phase D)

---

## Target Metrics

| Metric | Target | Reference |
|---|---|---|
| R2 | > 0.97 | Jiajing: 0.967 |
| RMSE | < 1.7 mm | Jiajing: 1.87 mm |
| Accuracy within 3mm | > 98% | 2013 paper: 98.9% |

---

## Files to Create

```
core/
  sizing/
    mask_diameter.py     Step 2
    view_fusion.py       Step 3

scripts/
  extract_frames.py      Step 1
  fit_scale.py           Step 4
  train_size_regressor.py  Step 5
  evaluate_size.py       Step 6
```

---

## Start: Step 1
First we build the frame extractor. This runs the tracker on all G1-G11
videos and produces the per-frame mask data we need for every other step.
