# Paper Study: Mizushima & Lu (2013)
**"A Low-Cost Color Vision System for Automatic Estimation of Apple Fruit Orientation and Maximum Equatorial Diameter"**
*Transactions of the ASABE, Vol. 56(3): 813-827*
*Authors: A. Mizushima, R. Lu (Dr. Lu!)*

---

## PART 1 - What This Paper Is About (Plain English)

Dr. Lu co-authored this paper back in 2013. The big problem they were solving:

> "We have a camera mounted close to a conveyor. Apples pass under it. We want to measure the real-world diameter of each apple in millimeters. But the camera makes things look distorted — apples at the edge of the image look smaller than apples in the center, even if they are the same size. How do we fix this?"

Their goal was an **in-field presorting system** — light, cheap, fast enough to sort apples at 4-6 per second right there in the orchard, before storage.

---

## PART 2 - Their Setup (Hardware)

| Parameter | Their System |
|---|---|
| Camera | Low-cost CCD color camera (IEEE 1394) |
| Resolution | 640 x 480 pixels |
| Focal length | **4.3 mm** (very wide angle - this is the root of the distortion problem) |
| Camera height | **38 cm** above conveyor |
| Conveyor | 2-lane bi-cone roller (apples rotate as they travel) |
| Frame rate | 15 frames/s |
| Images per apple | ~15 per apple as it passes through |

Key difference from our setup: they used a **very wide-angle 4.3 mm lens at only 38 cm height**. This caused serious distortion. Our setup uses a **16 mm lens** (much narrower angle), so our distortion will be significantly less severe.

---

## PART 3 - The Core Problem They Solved (The Distortion/Scale Issue)

This is the part Dr. Lu pointed you to. Read carefully.

### Why a single mm/px value does not work

Imagine you have a camera looking straight down. A ball at the center of the image is directly below the lens. A ball at the edge of the image is farther away from the lens (diagonally). Because of this:

- **Center ball**: closer to lens, appears bigger in pixels
- **Edge ball**: farther from lens, appears smaller in pixels
- **Same physical size, different pixel counts**

With a simple "0.49 mm/pixel" global constant, your size estimate changes depending on WHERE the apple is in the image, not what size it actually is.

They proved this quantitatively: using a global constant, a ball at the edge of the image had **3.5% area error** (RMSE = 112 mm²). Their fix reduced this to **0.5% error** (RMSE = 15.9 mm²).

### They also tried the "standard" fix first - and it FAILED

The standard fix for camera distortion is:
1. Calibrate the camera using a checkerboard (Zhang 2000 method)
2. Correct radial distortion
3. Transform image to "world coordinates" using Direct Linear Transformation (DLT)

This worked for flat objects. But apples are **3D spheres**, not flat. The DLT assumed the calibration plane is at a fixed height (the belt surface). But:
- A small apple sits lower (its center is lower off the belt)
- A large apple sits higher (its center is higher off the belt)

So even after full calibration+DLT correction, smaller apples were **underestimated** and larger apples were **overestimated** because their centers were at different heights than assumed.

### Their actual solution: "Variable Pixels Per Unit Dimension"

Instead of correcting the image, they **measured the distortion empirically** using two known calibration balls:
- Small ball: 63.5 mm diameter
- Large ball: 76.2 mm diameter

They rolled these balls across the full width of the conveyor and measured how many pixels each ball occupied at different X positions (left, center, right). They then fit a **quadratic equation** to model how pixel count changes with X position:

```
pixels_observed(x) = a*x^2 + b*x + c
```

From this, they derived a **position-dependent scale function r(x)**:

```
r(x) = (d / 2) / sqrt(a*x^2 + b*x + c)    [mm/pixel]
```

Where d is the known true diameter of the calibration ball, and x is the pixel position of the apple's centroid.

This gives them a **different mm/px value for every X position** - it accounts for both radial lens distortion AND perspective scale change in one shot, without needing to know the focal length or camera height at all. They just roll two balls and measure.

**Then, for a real apple:**
1. Detect the apple, find centroid x position
2. Look up r(x) from the empirical function (interpolate between small/large ball curves if apple size is between them)
3. Multiply measured pixel diameter by r(x) to get real-world mm

---

## PART 4 - The Orientation Part (Bonus Feature We Don't Need Yet)

Once they had accurate size in mm, they also wanted to know the **stem/calyx orientation** of each apple (needed for USDA-compliant maximum equatorial diameter). They used:

1. **Moment algorithm** - principal axes from pixel mass distribution
2. **Stem detection** - spikes in the contour radius function = stem shape
3. **Symmetry detection** - cross-correlation of left/right halves of contour

Their orientation accuracy was 87.6% for Delicious apples within ±20° (good for elongated varieties, worse for round ones like Empire).

**We don't need this for now** - we're using YOLO bounding boxes and our conveyor geometry is different (screw conveyor, not bi-cone rollers).

---

## PART 5 - Their Results

| Variety | RMSE (mm) | Within ±5 mm |
|---|---|---|
| Delicious | 1.60 | 99% |
| Golden Delicious | 1.54 | 99% |
| Empire | 2.34 | 98% |
| Jonagold | 1.65 | 100% |
| **Overall** | **1.79** | **98.9%** |

They beat a mechanical sizer: **4.3% classification error vs 15.1% for mechanical**.

---

## PART 6 - Comparison: Their Approach vs. Our Planned Approach

| Aspect | Mizushima & Lu 2013 | Our Planned Approach |
|---|---|---|
| **Camera** | Low-cost CCD, 4.3 mm wide-angle | JAI multispectral, 16 mm narrow-angle |
| **Distortion severity** | HIGH (wide angle, close range) | LOWER (narrow angle) |
| **Scale correction method** | Empirical: roll calibration balls, fit quadratic curve per X position | Geometric: physics-based 1/cos(theta) from camera height H + lane offset X_lane |
| **Needs physical measurements?** | NO (calibration is self-contained from the balls) | YES (need to measure H and X_lane in lab) |
| **Works for any apple size?** | Yes (interpolates between two ball curves) | Yes (correction depends only on position, not size) |
| **Accounts for radial lens distortion?** | YES (absorbed into empirical fit) | PARTIALLY (assumes no radial distortion; our 16 mm lens likely OK) |
| **Accounts for apple height (3D sphere)?** | YES (interpolates by apple pixel count = proxy for size) | NOT YET (we correct for lateral position, not for apple height variation) |
| **Ground truth validation** | Benchtop imaging system (rotate apple 360° + camera) | Physical calipers + LR model, R² target > 0.99 |
| **Orientation detection?** | YES (stem, symmetry, moment algorithms) | NOT planned (screw conveyor handles this differently) |

---

## PART 7 - What Is Best from Each Approach & What We Should Borrow

### What they did better

1. **The empirical calibration idea is brilliant and robust.**
   Instead of relying on measuring H and f precisely (which have measurement error), they just roll two balls of known size across the belt and let the data tell them the scale function. This absorbs lens distortion, perspective distortion, AND apple height effects all in one shot.

2. **Two calibration balls (one small, one large) to handle different apple heights.**
   This is the part our approach currently misses. A 60 mm apple and a 90 mm apple sit at different heights above the belt, meaning their centers are at different distances from the camera. Same position in image, but different actual scale. Their interpolation between two ball curves accounts for this.

3. **RMSE of 1.79 mm overall** - this is a strong benchmark for us to aim at or beat.

### What we are doing that is better or different

1. **Our camera is far superior.** 16 mm lens vs 4.3 mm - our perspective distortion is much smaller to begin with. Their correction was critical for them; ours is a refinement.

2. **We are using YOLO tracking** - we get robust multi-frame tracking automatically. They had to build their own region-of-interest tracking from scratch.

3. **We have multispectral data** - future scope for internal quality + size combined.

4. **We target R² > 0.99 with a formal LR model** - they did not frame it this way. We are more rigorous in the validation methodology (something Dr. Lu is clearly adding based on your meeting).

### What we should borrow (concrete ideas)

| Idea | How to adapt for our system |
|---|---|
| **Empirical calibration with known balls** | Roll a ball of known diameter (e.g., 70 mm billiard ball or calibration sphere) across all 3 lanes. Record bounding box pixel width at each X position. Fit a position-dependent scale correction. This is more reliable than computing from H and f alone. |
| **Two balls for size interpolation** | Use one small (~60 mm) and one large (~90 mm) ball to build two scale curves, then interpolate for the actual apple pixel count - just like they did. |
| **Quadratic fit of pixels vs X position** | Their quadratic model `pixels = a*x^2 + b*x + c` is simple, fast, and works well. We can implement the same. |
| **Keep the geometric model too** | Our 1/cos(theta) correction is theoretically motivated and good for documentation and paper. We can use the empirical method for accuracy and the geometric formula as a sanity check / cross-validation. |

---

## PART 8 - Revised Plan for Our Sizing Feature

Based on this paper, here is the updated thinking:

### Step 1 - Lab calibration session (physical)
- Get 2 calibration balls: one ~60-65 mm diameter, one ~75-80 mm diameter (billiard balls work, or buy calibration spheres)
- Roll each ball across each lane (or across the full width of the frame) while recording video
- Extract YOLO bounding boxes for each ball frame
- Record: pixel X position of centroid, bounding box min(w,h) in pixels
- Fit quadratic: `N_pixels(x) = a*x^2 + b*x + c` for each ball

### Step 2 - Build scale function r(x, N)
Following Equation 1 from the paper:
```
r(x) = (d/2) / sqrt(a*x^2 + b*x + c)     [mm/pixel]
```
- For N > large_ball curve: use large ball r(x)
- For N < small_ball curve: use small ball r(x)
- For N between: interpolate proportionally

### Step 3 - Apply to tracked apples
```python
x_centroid = (x1 + x2) / 2
D_px_raw = min(x2 - x1, y2 - y1)
r = get_scale(x_centroid, D_px_raw)   # mm/pixel, position and size aware
D_mm = D_px_raw * r
```

### Step 4 - Aggregate across frames (peak), validate with GT LR model

This replaces our simpler 1/cos(theta) model with something empirically grounded and more accurate - especially for accounting for apple height variation.

We still keep the geometric model in the doc for theoretical justification.

---

## PART 9 - Key Numbers to Remember

| Number | Source | Meaning |
|---|---|---|
| 1.79 mm | Paper (RMSE overall) | Their accuracy - our target to match or beat |
| 4.3 mm | Their focal length | Why they had bad distortion (ours is 16 mm - 4x less distortion) |
| 38 cm | Their camera height | (ours TBD, likely similar or higher) |
| 15 images/apple | Their frame count | We should get similar or more with YOLO tracking |
| 63.5 mm / 76.2 mm | Their calibration ball sizes | Good reference for choosing our own calibration balls |

---

*Study notes based on: Mizushima, A. and Lu, R. (2013). Transactions of the ASABE 56(3): 813-827.*
*Relevant to: feature/apple-size-estimation branch*
