# JAI FS-3200T — Frame Preprocessing Pipeline

**Project:** Apple Sorting GUI — MSU ASABE AIM26  
**Applies to:** `core/camera/camera_interface.py` · `scripts/camera_live_view.py`  
**Last updated:** 2026-05-18

---

## Overview

The JAI FS-3200T simultaneously streams 3 independent GEV sources.
Each frame triplet goes through two separate preprocessing paths:

```
Raw hardware frame
       │
       ├─► DISPLAY PATH  (what you see on screen)
       │       └── Demosaic → EMA-Normalize → Resize → Qt display
       │
       └─► INFERENCE PATH  (what the AI model sees)  [Phase 6 — not yet implemented]
               └── Demosaic → Fixed-range normalize → Full-res crop/pad
```

> [!IMPORTANT]
> Everything described in this document currently applies to the **display path only**.
> The inference path is not yet connected. When it is, several display-only steps
> will be removed or replaced.

---

## CH1 — Color Channel (~660 nm, BayerRG8)

### Step 1: Bayer Demosaicing

| | |
|---|---|
| **WHAT** | Converts a single-channel Bayer mosaic image into a full 3-channel BGR image |
| **HOW** | `cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)` — bilinear interpolation |
| **WHY** | The physical sensor has only ONE photodetector per pixel, each covered by a colored filter (R, G, or B in alternating pattern). A raw Bayer image looks like a gray checkerboard, not a color image. Demosaicing reconstructs the missing two color values at each pixel from its neighbors. |
| **Mandatory?** | **YES — always.** Cannot display or feed color data to a model without it. |
| **Inference effect** | Required before any color-based model. The model expects a 3-channel BGR/RGB image. |
| **Temporary?** | **No. Permanent.** This step is fundamental hardware compensation. |

**Before / After:**
```
Raw:   R G R G G B G B R G R G ...  (1 value per pixel, mosaic pattern)
After: B G R  B G R  B G R  ...     (3 values per pixel, normal color image)
```

---

### Step 2: Resize 2048×1536 → 640×480

| | |
|---|---|
| **WHAT** | Downscales the full-resolution color frame |
| **HOW** | `cv2.resize(..., interpolation=cv2.INTER_AREA)` — pixel area averaging |
| **WHY** | Passing a 2048×1536 3-channel image (9.4 MB) across the Qt thread boundary and rendering it in a QLabel is extremely slow. Qt's SmoothTransformation scales it anyway, but in the main thread. Doing it in OpenCV (background thread, SIMD-optimized) is ~10× faster. |
| **Mandatory?** | **Display only. NOT mandatory for inference.** |
| **Inference effect** | The model must receive **full-resolution** frames. Resizing would lose spatial detail needed to detect bruising or defects. This step is REMOVED in the inference path. |
| **Temporary?** | **Yes — display only.** Will not be applied to inference frames. |

---

## CH2 (NIR1 ~800 nm) and CH3 (NIR2 ~900 nm) — Mono8

### Step 1: EMA-Stabilized Gain Normalization

| | |
|---|---|
| **WHAT** | Stretches the low raw pixel values (typically 0–50) to full display range (0–255) using a slowly-adapting gain |
| **HOW** | Exponential Moving Average of the per-frame max value, then linear scaling: `display = clip(raw × (255 / ema_max), 0, 255)` |
| **WHY** | **Two separate problems solved by this one step:** |

**Problem A — NIR channels are physically dark in simultaneous mode:**

The JAI FS-3200T in MultiSource (simultaneous 3-stream) mode uses a shorter effective exposure per pixel than in single-stream mode. NIR sensors also have lower quantum efficiency at 800–900 nm than in the visible range. Combined, the raw pixel values sit in the range 0–50 (out of 255). Without normalization, both NIR panels look nearly black to a human observer.

**Problem B — Per-frame NORM_MINMAX causes brightness flickering:**

`cv2.NORM_MINMAX` sets gain based on the current frame's min and max:
```
gain = 255 / (frame_max - frame_min)
```
If a bright object moves into the frame: `frame_max` jumps → gain drops → whole image dims suddenly.  
If the object moves out: `frame_max` drops → gain spikes → whole image brightens suddenly.  
This produces visible frame-to-frame flicker even in static scenes (sensor noise causes tiny max fluctuations).

**EMA solution:**
```python
ema_max = 0.95 × ema_max_prev + 0.05 × current_frame_max
gain    = 255 / ema_max
```
The gain now changes by at most 5% per frame. A sudden bright pixel takes ~20 frames to fully influence the gain. The image brightness is perceptually stable.

| | |
|---|---|
| **Mandatory (display)?** | **YES** — without it, NIR panels are nearly invisible |
| **Mandatory (inference)?** | **Partially.** The model needs normalized NIR, but NOT with EMA. See below. |
| **Inference effect** | ⚠️ EMA normalization is **wrong for inference**. The EMA gain is scene-dependent and changes over time. Two frames with the same apple will get different normalization if a bright object entered the scene between them. The AI model requires **consistent, reproducible** normalization. For inference, use **fixed-range normalization**: `pixel / 50.0` (dividing by the known physical max for simultaneous mode), or **per-frame NORM_MINMAX** (acceptable if the model was trained with the same method). |
| **Temporary?** | **EMA is display-only and will be removed from the inference path.** The display path keeps EMA. The inference path uses fixed or per-frame normalization. |

---

### Step 2: Resize 2048×1536 → 640×480

Same reasoning as CH1 Step 2 above.

| | |
|---|---|
| **Mandatory?** | **Display only.** |
| **Inference effect** | Removed in inference path — model needs full resolution. |
| **Temporary?** | **Yes — display only.** |

---

## Summary Table

| Step | CH | Display path | Inference path | Permanent? |
|---|---|---|---|---|
| Bayer demosaic | CH1 only | ✅ Yes | ✅ Yes | ✅ Yes — always |
| EMA gain normalization | CH2, CH3 | ✅ Yes | ❌ Removed | Display only |
| Fixed-range normalization | CH2, CH3 | ❌ Not used | ✅ Yes (planned) | Inference only |
| Resize 2048×1536 → 640×480 | All | ✅ Yes | ❌ Removed | Display only |

---

## What Happens in Each Path at Inference Time (Phase 6)

```
JAICamera.grab_raw()  ← NEW method — returns unprocessed FrameTriplet
        │
        ├── ch1 (BayerRG8) → Bayer demosaic → full-res BGR
        │                                          │
        │                                    crop/letterbox
        │                                          │
        │                                    YOLOv8 input
        │
        ├── ch2 (Mono8)  → fixed normalize (÷50) → full-res
        │                                              │
        │                                      concatenate with ch1
        │
        └── ch3 (Mono8)  → fixed normalize (÷50) → full-res
                                                       │
                                               concatenate with ch1
```

The inference worker will call a separate `grab_raw()` method (not yet implemented)
that skips all display-only steps and returns full-resolution, minimally-processed arrays.

---

## EMA Alpha Guide

The `_EMA_ALPHA = 0.05` value can be tuned:

| Alpha | Behavior |
|---|---|
| 0.01 | Very slow — takes ~100 frames to adapt. Maximally stable but slow to respond to scene changes |
| **0.05** | **Current setting** — ~20 frames to adapt. Good balance |
| 0.10 | Faster — ~10 frames. Slight flicker on scene changes |
| 0.30 | Fast — ~3 frames. Visible flicker on object movement |
| 1.00 | Equivalent to per-frame NORM_MINMAX — maximum flicker |

> [!TIP]
> If the display feels too slow to adapt when scene brightness changes dramatically
> (e.g., turning lights on/off), increase to 0.10.
> If there is still residual flicker, decrease to 0.03.
