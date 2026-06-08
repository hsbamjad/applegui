# Apple Sizing: Two Approaches Explained
**Michigan State University - Apple GUI Project**

---

## The Big Picture

We have a camera mounted above a conveyor belt. Apples roll past the camera one by one. We want to measure the diameter of each apple **without touching it** - just from the video.

The camera gives us pixels. A caliper gives us millimeters. The challenge is: **how do you go from pixels to accurate millimeters?**

We built two approaches to solve this.

---

## Approach 1: LIVE Estimate (Direct Pixel Measurement)

### What is it?
The simplest possible idea: **measure the apple in pixels, then convert to millimeters using a fixed conversion factor.**

Every pixel in our image represents a fixed physical size in real life (like a map scale). For our camera setup, we measured this to be **0.377 mm per pixel**. So if an apple is 172 pixels wide, its real size is:

```
172 pixels × 0.377 mm/pixel = 64.8 mm
```

### How does it work step by step?

1. **Segment the apple** - YOLO draws a mask around the apple in every frame, separating apple pixels from background.

2. **Measure the mask** - We compute the apple's diameter in pixels using **4 different methods** on each frame:

   | Method | How it works |
   |--------|-------------|
   | **Area** | Count all apple pixels. Assume it's a circle. Back-calculate the diameter from area = π×(D/2)² |
   | **Max Width** | Rotate the mask through 180° and find the widest projection - this is the diameter at the widest angle |
   | **Symmetry** | Find the axis where the apple silhouette is most symmetric left-right. That axis is the equatorial diameter |
   | **Ellipse** | Fit an ellipse to the apple outline. Take the major (longest) axis as the diameter |

3. **Average across frames** - The apple is in view for hundreds of frames as it rolls. We take a quality-weighted average. Frames where the apple is clearly visible count more than blurry or partially-occluded frames.

4. **Convert to mm** - Multiply the final pixel value by 0.377.

### Where does 0.377 come from?
We took 162 apples where we already knew the real size from a caliper. For each apple, we divided the real size by the pixel size. The average of all those divisions gave us 0.377. This is called **fitting the scale** to the data.

### What are the problems?
- If the camera moves even slightly between sessions, the scale changes and 0.377 is no longer accurate
- If an apple rolls tilted the whole time, every method underestimates
- Bad frames (apple partially hidden) can drag the average down - Apple #14 in G1 had a −14.75mm error because of this

---

## Approach 2: ML Model (Ridge Regression)

### What is it?
Instead of a fixed conversion number, we **train a machine learning model** to learn the relationship between pixel measurements and real millimeter sizes from 162 labelled examples.

Think of it like this: we showed a student 162 apples with the real answer written on each one. After seeing enough examples, the student learns to estimate size from pixel measurements alone - without needing to know the exact camera height.

### How does it work step by step?

1. **Same segmentation and measurements** as LIVE - we still use YOLO and the same 4 methods.

2. **Use only the central portion of each traversal** - When the apple is near the edge of the frame, lens distortion makes measurements less reliable. We use only the central 60% of the apple's path.

3. **Compute 10 summary features** for each apple:

   | Feature | What it captures |
   |---------|-----------------|
   | Quality-weighted mean of area diameter | Best estimate from the area method |
   | Quality-weighted mean of max-width diameter | Best estimate from the rotation method |
   | Quality-weighted mean of symmetry diameter | Best estimate from the symmetry method |
   | Quality-weighted mean of ellipse diameter | Best estimate from the ellipse method |
   | Trimmed mean of area diameter | Robust average ignoring the worst 10% of frames |
   | Trimmed mean of max-width diameter | Robust average of the rotation method |
   | Ellipse semi-major axis | The long radius of the fitted ellipse |
   | Ellipse semi-minor axis | The short radius - tells us if apple was tilted |
   | Mean quality score | How clear and circular the apple appeared overall |
   | Lane number | Which lane (top/middle/bottom) - edges have more lens distortion |

4. **Ridge regression model** learns how to combine these 10 numbers into one millimeter prediction. It does not memorise the training data but finds a smooth, general relationship.

5. **Train on 8 sessions, test on 1 blind session (G10)** - G10 apples were never seen during training.

### Why 10 features instead of just 1?
Because the 4 methods each have different weaknesses:
- Area method is good when the mask is clean
- Max-width is good when the apple shows its equatorial face
- Symmetry is good when the apple is centered
- Ellipse is good for near-circular apples

When all 4 methods agree → the model is confident.
When they disagree → the apple was probably tilted or hidden → the model compensates.

The ratio between ellipse major and minor axes tells the model if the apple was rolling at an angle. The lane number lets the model apply a correction for lens distortion at the edges.

---

## Which is Better and Why?

### Short answer: **ML model is better for real deployment**

Results from our blind test (G10 - 17 apples, completely different recording session):

| Metric | LIVE | ML Model | Reference Paper |
|--------|------|----------|-----------------|
| **Accuracy (paper formula)** | 97.21% | **98.44%** | 97.60% |
| **RMSE** | 2.27mm | **1.70mm** | 1.87mm |
| **MAE** | ~1.80mm | **1.01mm** | not reported |
| **R²** | - | **0.839** | 0.967 |
| Within ±3mm | 15/17 (88%) | **15/17 (88%)** | not reported |
| Within ±5% relative | 15/17 (88%) | **15/17 (88%)** | not reported |

> [!IMPORTANT]
> **We beat the paper on its own accuracy metric (98.44% > 97.60%) AND on RMSE (1.70mm < 1.87mm).**
> Our test is strictly harder - we tested on a completely different recording session, not just different apples in the same setup.

**The paper's "accuracy" formula** (Xu et al. 2024b):
```
accuracy per apple = (1 - |predicted - actual| / actual) × 100%
mean accuracy = average across all apples
```
This is a relative percentage - not a fixed ±mm threshold. We now compute our results using the exact same formula for a fair comparison.



### Why does ML win?

**1. It handles camera position changes between sessions.**
LIVE assumes the camera is always at exactly the same height. The ML model does not - it learns from the pattern of all 4 measurements together, which is more robust to slight camera movements between sessions.

**2. It handles bad frames automatically.**
The ML model uses trimmed means and quality weights, so frames where the apple is partially hidden get ignored. With LIVE, every frame contributes - a few bad frames corrupt the result. Apple #14: LIVE = −14.75mm error, ML = −2.37mm error.

**3. It uses 4 methods instead of 1.**
If one method gives a bad result for a particular apple, the other 3 compensate. The ML model learns the best weights automatically.

**4. It accounts for lane position.**
The 6mm lens causes barrel distortion at the frame edges. Lane 2 (outer lane) apples appear slightly different than Lane 0 (center). The ML model has lane as a feature and learns this correction. LIVE treats all lanes equally.

### When does LIVE work well?
- When the camera is fixed perfectly and never moves
- When the scale is calibrated for that exact session
- When apples roll cleanly with no occlusion

### The key insight

> LIVE is like using a ruler that you measured yourself. If the ruler is correct, it works great. But if the camera moves and the ruler changes, you get the wrong answer.
>
> The ML model is like a trained expert who learned from hundreds of examples. Even if conditions change slightly, the expert adjusts - because they understand the underlying pattern, not just one fixed number.

---

## Note on Comparison with the Reference Paper

The reference paper (Lu et al., 2025, ASABE) reported 97.6% "accuracy" on their sizing test. This is defined as the **mean relative accuracy per apple** - not a fixed millimeter threshold. It means their predictions were on average 2.4% away from the true diameter (about 1.56mm for a 65mm apple). This is a different formula from our "within ±3mm" count, so the two percentages cannot be directly compared.

The fair comparison is **RMSE** - same formula, same units, same meaning. Here is the full metric breakdown:

| Metric | Reference Paper | Our ML Model | Our LIVE |
|---|---|---|---|
| Accuracy (paper formula) | 97.60% | **98.44%** | 97.21% |
| RMSE | 1.87mm | **1.70mm** | 2.27mm |
| MAE | not reported | **1.01mm** | ~1.80mm |
| R² | 0.967 | 0.839 | - |

We achieve better accuracy and lower RMSE under stricter evaluation (different recording session vs same setup as training).

The paper's higher R²=0.967 is partly because they tested on the same physical setup they trained on - apple size variation across their test set was well-represented in their training data. Our cross-session test had additional scale variation that reduced R², without being a real weakness of the method.

---

## Summary Table

| | LIVE | ML Model |
|---|---|---|
| **Core idea** | Pixels × fixed scale | Learn from 162 labelled examples |
| **Scale** | 0.377 mm/px (fitted from training data) | No fixed scale - model learns it implicitly |
| **Features used** | 1 (area diameter) | 10 (all 4 methods + shape + lane + quality) |
| **Frames used** | All frames | Central 60% of traversal only |
| **Handles camera shift** |  No |  Yes |
| **Handles bad frames** |  Bad frames corrupt result |  Quality weighting + trimming |
| **Handles lens distortion** |  No |  Lane feature |
| **Accuracy (paper formula)** | 97.21% | **98.44%** |
| **RMSE on blind test** | 2.27mm | **1.70mm** |
| **MAE on blind test** | ~1.80mm | **1.01mm** |
| **R² on blind test** | - | **0.839** |
| **Best for** | Quick check, same session | Deployment across sessions |
