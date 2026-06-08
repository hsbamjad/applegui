# Apple Sizing in the Live GUI
**Michigan State University - Apple GUI Project**

---

## Overview

The offline pipeline (described in `sizing_approaches.md`) produces the best possible sizing results but it runs as a batch job - you give it a folder of video recordings, it processes everything overnight, and outputs a report. That is great for research but useless for a sorting machine that needs to make a decision **right now**, while the apple is still on the belt.

This document explains how we brought the same ML sizing pipeline into the live GUI so it works in real time - with the same Ridge regression model, the same features, and results that are within ~1-2 mm of the offline numbers.

---

## The Core Challenge: Speed vs Accuracy

The offline pipeline does not care how long it takes. It can spend 7 ms measuring one apple mask because it is running on pre-recorded video with no time pressure.

The live GUI runs at 30 frames per second. That means the entire system - camera grab, YOLO inference, tracking, display, sizing - must fit inside **33 ms per frame**. There can be 10-15 apples on screen at the same time. There is simply no time to do the full 36-rotation diameter measurement on every apple in every frame.

The solution we built is a **two-phase architecture** that keeps the expensive work completely off the main loop.

---

## How It Works: Two-Phase Architecture

### Phase 1 - Fast work on the main thread (every frame, ~0.1 ms per apple)

Every time YOLO produces a result, the accumulator runs quickly for each tracked apple:

1. **Get the apple outline** from YOLO's segmentation output. YOLO returns the apple boundary as a list of (x, y) points in the original image coordinate space (2048 Ã- 1536 pixels for our JAI camera).

2. **Compute the convex hull.** The raw YOLO polygon sometimes has small dents or jagged edges from the segmentation. Taking the convex hull smooths this out - it finds the tightest convex shape that wraps around all the boundary points. This is exactly what the offline pipeline does too.

3. **Fit an ellipse** to the hull vertices. This gives us the major axis (longest diameter estimate) and minor axis (shortest diameter estimate) directly in pixels, at full camera resolution. These become the `d_ell` and `d_minor` features.

4. **Compute a quality score** for this frame. Quality = (minor axis / major axis) Ã- completeness flag. A perfectly round apple has ratio = 1.0. A tilted apple has a lower ratio. If the apple is partially cut off at the frame edge, completeness = 0.5. This tells us how reliable this particular frame's measurement is - frames with higher quality scores contribute more to the final average.

5. **Render a small crop image** of the hull, padded by 4 pixels on each side. This crop is typically around 150-250 pixels wide - just the apple region, nothing else. This crop is what goes to Phase 2.

6. **Submit the crop to a background thread pool.** This is the key design decision. The crop is handed off to two background worker threads and the main loop moves on immediately. No waiting.

**Total cost on main thread: about 0.1 ms per apple.** At 15 apples that is 1.5 ms - negligible.

---

### Phase 2 - Accurate work in background (concurrent, ~3.4 ms per apple)

While the main thread continues rendering frames, two background worker threads process the submitted crops:

#### What is max_width and why does it need rotations?

Imagine you want to measure the diameter of a potato using only a ruler that can only measure horizontally. If you hold the ruler at 0Â°, you get the horizontal width. If the potato is wider at a different angle - say 45Â° - you would miss it. To find the true widest diameter you need to rotate the ruler and measure at many different angles.

`max_width()` does exactly this digitally. It rotates the apple mask image by a few degrees, measures the widest horizontal span of the white pixels, then rotates a bit more, measures again, and so on. The largest value found across all rotation angles is the true maximum diameter of that mask.

- **2 rotations** (old, broken): measured only at 0Â° and 90Â°. Fast but misses the true maximum if the apple is oriented between those angles.
- **18 rotations** (current default): measures every 10Â°. Finds the true maximum for any orientation. Takes about 3.4 ms per apple in the background.
- **36 rotations** (offline pipeline): measures every 5Â°. Even finer. Takes 7 ms per apple. Gives the same answer as 18 rotations for near-circular apples.

The background thread also runs the **symmetry diameter** method, which looks for the axis where the apple silhouette is most symmetric left-right - a good independent estimate of the equatorial diameter. And it computes the **area diameter** - treating the apple as a sphere and back-calculating diameter from the pixel count.

---

## Accumulating Measurements Over Time

An apple is visible for typically 100-400 frames as it travels across the camera view. The accumulator collects a measurement dictionary for every frame: the four diameter estimates, the quality score, and the x-position of the apple center.

When the apple finally exits (detected by the tracker crossing the exit zone), `commit()` is called.

---

## What Happens at Commit Time

At this point the apple has been tracked for hundreds of frames and all the background futures have long since completed. `commit()` does the following:

1. **Resolve all background futures.** Each future contains the d_maxw, d_sym, and d_area values for one frame. Resolving them is instant - the work is already done.

2. **Central frame filter.** Not all frames are equally useful. When the apple first enters the frame or is about to leave, it may be partially obscured or at the edge of the lens where distortion is worse. The accumulator keeps only the frames from the central 60% of the apple's horizontal travel path (20% to 80% of the total distance traversed). This is the same filter the offline pipeline uses.

3. **Quality-weighted means.** For each of the four diameter methods, we compute a weighted average across the central frames. Frames with higher quality scores (more circular, fully inside the frame) count more. This is called `d_ell_wmean`, `d_maxw_wmean`, `d_sym_wmean`, `d_area_wmean`.

4. **Trimmed peak.** We also compute the average of the top 10% of frames for max-width and area - this captures the apple's best-viewed moment, filtering out frames where it was partially hidden.

5. **Consensus ellipse axes.** The quality-weighted mean of the major axis values across all central frames gives `ell_a`. Similarly for `ell_b`. These capture the apple's typical shape when fully visible.

6. **Build the 10-feature vector** in the exact order the Ridge model was trained on:

   | Feature | What it is |
   |---|---|
   | `d_area_wmean` | Quality-weighted mean of area diameter (px) |
   | `d_maxw_wmean` | Quality-weighted mean of max-width diameter (px) |
   | `d_sym_wmean` | Quality-weighted mean of symmetry diameter (px) |
   | `d_ell_wmean` | Quality-weighted mean of ellipse diameter (px) |
   | `d_area_peak` | Top-10% trimmed mean of area diameter (px) |
   | `d_maxw_peak` | Top-10% trimmed mean of max-width diameter (px) |
   | `ell_a` | Consensus major axis - average shape (px) |
   | `ell_b` | Consensus minor axis - captures tilt (px) |
   | `mean_Q` | Mean quality score across central frames |
   | `lane` | Lane number 0, 1, or 2 (for lens distortion correction) |

7. **Run the Ridge model.** The scikit-learn Pipeline (StandardScaler + Ridge regression) takes the 10-feature vector and outputs a predicted diameter in millimeters. A sanity clamp of 40-120 mm catches any numerical failures.

8. **Return the size.** The result is passed back to the GUI and displayed in the Results panel.

---

## Why Pixel Values Are the Right Input (Not mm)

The features fed to the model are all in **pixels**, not millimeters. This is intentional. The Ridge model was trained on pixel-space features extracted from the same 2048 Ã- 1536 camera. The model internally learned the pixel-to-mm conversion as part of its regression coefficients, calibrated to our specific camera height and lens.

This means the live GUI must produce features in the same pixel space as the training data. This was the root cause of the 40 mm bug we hit early on - the live code was reading mask data at YOLO's internal inference resolution (640 Ã- 640) instead of the original camera resolution (2048 Ã- 1536). All diameter values were 3Ã- too small, the model predicted ~20 mm, and the clamp floored everything to 40 mm. The fix was to use `masks.xy` - YOLO's polygon coordinates in the original image space - instead of the downsampled raster mask tensor.

---

## Accuracy vs the Offline Pipeline

| | Offline pipeline | Live GUI |
|---|---|---|
| **Source** | Lossless BMP images | MP4 video (H.264) |
| **Rotations** | 36 (every 5Â°) | 18 (every 10Â°, background thread) |
| **Frames** | Full traversal across all BMPs | Tracker entry â exit zone |
| **Expected gap** | Reference (MAE ~1 mm) | ~1-3 mm additional error |

The remaining gap has two causes:

**MP4 compression.** The offline pipeline was built from raw BMP image files - completely lossless. The live GUI processes the same G1 video as an MP4 file. H.264 encoding introduces block artifacts at apple edges that make the polygon boundary slightly noisier. This adds roughly 1-2 mm of error and cannot be fixed without using a different video source.

**Track boundaries.** The tracker uses entry and exit zones to decide when an apple starts and stops. The offline pipeline sees the apple across its entire traversal. Slightly different frame sets contribute to the weighted averages.

**Rotation count.** For near-circular apples the 18-rotation and 36-rotation results are identical. The difference only shows up for partially occluded or tilted apples.

---

## Configuration

The angle resolution is tunable in `config/config.yaml` without changing any code:

```yaml
sizing:
  enabled: true
  model_path: "models/size_model.pkl"
  min_frames: 4
  bg_angle_step: 10   # 18 rotations - change to 5 for full 36-rotation offline quality
```

The background thread handles whatever rotation count you set, so there is no FPS impact at any setting. The default of 10 (18 rotations) is the best balance for practical use.

---

## Where the Code Lives

| File | What it does |
|---|---|
| `core/sizing/accumulator.py` | Main sizing logic - update(), commit(), thread pool |
| `core/sizing/mask_diameter.py` | Low-level diameter functions: max_width, symmetry_diameter, area_diameter |
| `core/sizing/view_fusion.py` | Fuses frame measurements into the 10-feature vector (fuse_apple) |
| `models/size_model.pkl` | Trained Ridge pipeline (StandardScaler + Ridge, saved by train_size_regressor.py) |
| `gui/main_window.py` | Wires accumulator into the inference loop - calls update() each frame, commit() on apple exit |
| `gui/panels/stats_panel.py` | Displays the size in the Recent Results panel |
| `config/config.yaml` | Controls enabled, model path, min_frames, bg_angle_step |

---

## Summary

The live GUI sizing works by running a faithful real-time replica of the offline pipeline, split across two layers:

- **Fast layer (main thread):** polygon â convex hull â ellipse fit â quality score â submit crop to background. Takes 0.1 ms per apple.
- **Slow layer (background threads):** max_width with 18 rotations + symmetry + area. Takes 3.4 ms per apple concurrently, never blocking the display.
- **At commit time:** resolve futures, filter central frames, quality-weight, build 10 features, run Ridge model, output mm.

The result is a system that gives real-time sizing with offline-quality features at no FPS cost.
