# JAI FS-3200T — Frame Preprocessing & Visualization Pipeline Handbook

**Project:** Apple Sorting GUI — MSU ASABE AIM26  
**Applies to:** `core/camera/camera_interface.py` · `gui/widgets/image_display.py` · `scripts/camera_live_view.py` · `scripts/camera_probe_jai.py`  
**Last updated:** 2026-05-18

---

## 1. System Overview: Display Path vs. Inference Path

The JAI FS-3200T simultaneously streams three independent hardware-synchronized GigE Vision (GEV) sensors. Depending on the destination, these frames follow two completely separate processing pipelines:

```
                            Raw Hardware Frame Triplet
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[ DISPLAY PATH (What you see) ]                          [ INFERENCE PATH (What the AI sees) ]
  - Goal: Premium visual comfort, rich colors,              - Goal: Maximum diagnostic accuracy,
    stable brightness, zero lag, and no grey glare.           consistent pixel values, and high resolution.
  - Preprocessing: early 10x downsample, Bayer             - Preprocessing: Full-resolution raw grab,
    demosaic, EMA Min-Max norm, and PySide6 .copy().          fixed scale division, and letterbox padding.
```

---

## 2. Script Inventory & Preprocessing Roles

| Script / Component File | Role in Preprocessing | Preprocessing Details Implemented |
| :--- | :--- | :--- |
| **[camera_probe_jai.py](file:///S:/MSU_Research/ASABE%20AIM26/apple_gui/scripts/camera_probe_jai.py)** | Offline Diagnostic / Calibration | Saves **raw** full-resolution frames (Bayer/Mono8 unchanged) alongside display-normalized PNGs (`NORM_MINMAX`). |
| **[camera_live_view.py](file:///S:/MSU_Research/ASABE%20AIM26/apple_gui/scripts/camera_live_view.py)** | Real-Time Diagnostic Visualizer | Conversions at full resolution (Bayer demosaic, per-frame `NORM_MINMAX` stretch) and downscales using standard OpenCV resize. |
| **[camera_interface.py](file:///S:/MSU_Research/ASABE%20AIM26/apple_gui/core/camera/camera_interface.py)** | Core Production Backend (GUI) | Highly optimized: **early 10x downsampling**, full-resolution max highlight tracking, and **EMA-Stabilized Min-Max Normalization** (C++ `convertScaleAbs`). |
| **[image_display.py](file:///S:/MSU_Research/ASABE%20AIM26/apple_gui/gui/widgets/image_display.py)** | Production UI Rendering (GUI) | Performs **memory-safe QImage deep-copying (`.copy()`)** and high-fidelity **bilinear downscaling (`SmoothTransformation`)** in PySide6. |

---

## 3. Step-by-Step Preprocessing: CH1 — Color Channel (BayerRG8)

### Step 1.1: Bayer Demosaicing

*   **WHAT**: Converts a raw single-channel Bayer checkerboard mosaic into a standard 3-channel BGR color image.
*   **WHY**: The physical sensor has a single photodetector layer covered by an alternating Red, Green, and Blue filter grid. The raw frame looks like a gray checkerboard. Demosaicing mathematically interpolates the missing two color channels per pixel from spatial neighbors.
*   **HOW**: `ch1 = cv2.cvtColor(raw0, cv2.COLOR_BayerBG2BGR)` — using SIMD-optimized bilinear interpolation.
*   **WHERE**: Done in `camera_interface.py` (background `JAI-grab` thread), `camera_live_view.py`, and `camera_probe_jai.py`.
*   **MANDATORY?**: **YES — always.** It is impossible to feed standard color models (like YOLO) or display a normal color image to an operator without this hardware compensation step.
*   **Inference Effect**: Essential. The AI model expects a standard 3-channel color image (BGR or RGB) to extract features.
*   **Temporary or Permanent?**: **PERMANENT.** This step will remain identical in both the display and inference paths.

---

### Step 1.2: Downsampling (2048×1536 → 640×480)

*   **WHAT**: Downscales the full-resolution color image to display dimensions.
*   **WHY**: Passing a massive $2048 \times 1536$ color image (9.4 MB) across the PySide6 thread boundary 30 times a second creates severe lag. Downsampling in the background thread reduces the payload by **90.2%** (down to ~920 KB), keeping the GUI light and smooth.
*   **HOW**: `ch1 = cv2.resize(ch1, (640, 480), interpolation=cv2.INTER_LINEAR)` — fast bilinear area interpolation.
*   **WHERE**: Done in `camera_interface.py` (background thread) and `camera_live_view.py`.
*   **MANDATORY?**: **Display only. NOT mandatory for inference.**
*   **Inference Effect**: Resizing would discard the microscopic spatial details (such as tiny skin breaks, punctures, or early insect stings) necessary for accurate apple grading. **This downsampling step will be completely removed from the AI inference path.**
*   **Temporary or Permanent?**: **TEMPORARY (Display-only).** 
*   *How we get the full resolution later for inference*: We will call a separate raw grab method (`grab_raw()`) that retrieves the full-resolution $2048 \times 1536$ array directly from the eBUS pipeline buffers and passes it directly to the model's preprocessing stage (described in Section 6).

---

## 4. Step-by-Step Preprocessing: CH2 & CH3 — NIR Channels (Mono8)

### Step 2.1: Early Downsampling (10x Performance Boost)

*   **WHAT**: Resizes the raw NIR frames from $2048 \times 1536$ down to $640 \times 480$ *before* applying contrast adjustments.
*   **WHY**: Raw mathematical array calculations (EMA limits, scaling, and offset subtraction) are CPU-intensive. Resizing first cuts the raw pixel count from 3.1 million to just 307,200. This delivers a **10x speed boost** to the background thread.
*   **HOW**: `raw_small = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_LINEAR)`.
*   **WHERE**: Done in `camera_interface.py` (background `JAI-grab` thread).
*   **MANDATORY?**: **Display only. NOT mandatory for inference.**
*   **Inference Effect**: Discarding spatial detail before feeding a model would ruin defect detection. **This early downsampling is completely bypassed in the inference path.**
*   **Temporary or Permanent?**: **TEMPORARY (Display-only).**

---

### Step 2.2: Full-Resolution Highlight Max Tracking

*   **WHAT**: Scans the full-resolution $2048 \times 1536$ raw image to locate the absolute min/max peak values.
*   **WHY**: If we downsample the frame *before* finding the maximums, bilinear interpolation will blur out hot pixels and tiny specular reflections. Scanning the full-res array first (<0.3ms) ensures that our normalization limits remain perfectly true to physical highlight changes.
*   **HOW**: `cur_min, cur_max = float(raw.min()), float(raw.max())`.
*   **WHERE**: Done in `camera_interface.py` (background thread) and `camera_probe_jai.py`.
*   **MANDATORY?**: **YES.** It is the only way to protect the pre-processor from underestimating bright highlights.
*   **Inference Effect**: Keeps normalization scaling factors perfectly aligned with full-resolution sensor reality.
*   **Temporary or Permanent?**: **PERMANENT.**

---

### Step 2.3: EMA-Stabilized Min-Max Normalization

*   **WHAT**: Dynamically subtracts the sensor's physical black-level offset (pedestal) and stretches the dark raw pixels (typically 0-50) to the full visual range (0-255) using a temporally smoothed average.
*   **WHY**: 
    *   **Problem A (Grey Glare)**: Max-only scaling does not remove the sensor's black offset (pedestal). This offset gets amplified, turning the black conveyor background into a flat, light-grey glare. We must subtract the minimum value.
    *   **Problem B (Flickering)**: Traditional per-frame min-max normalization (`NORM_MINMAX`) recalculates gain instantly. When a bright apple enters, the gain drops, dimming the entire screen. When it leaves, the gain spikes, creating erratic frame-to-frame flickering.
*   **HOW**: We track both the pedestal (min) and highlight (max) using a slow Exponential Moving Average (EMA, $\alpha = 0.05$). We then calculate the stable range `diff = Max_EMA - Min_EMA` and scale factor, and apply them in a **single, highly optimized C++ operation**:
    ```python
    scale = 255.0 / max(diff, 1.0)
    offset = -Min_EMA * scale
    ch_norm = cv2.convertScaleAbs(raw_small, alpha=scale, beta=offset)
    ```
*   **WHERE**: Done in `camera_interface.py` (background thread).
*   **MANDATORY?**: **Display only. NOT mandatory for inference.**
*   **Inference Effect**: ⚠️ **EMA normalization is unacceptable for AI models.** Because EMA changes dynamically depending on what recently passed on the conveyor belt, two identical apples will yield completely different pixel values if a bright reflection occurred between them. The AI model requires **consistent, reproducible, and deterministic** pixel values. 
    For inference, we will use **Fixed-Range Normalization** (e.g., `pixel / 50.0`, dividing by the known physical limits of the simultaneous multisource setup) or a strict per-frame `NORM_MINMAX` (only if the model was trained with it).
*   **Temporary or Permanent?**: **TEMPORARY (Display-only).**

---

## 5. Step-by-Step Preprocessing: PySide6 GUI Display (`image_display.py`)

### Step 3.1: Memory-Safe Deep Copying (`.copy()`)

*   **WHAT**: Clones the compiled QImage pixel buffer into a persistent, dedicated memory block in Qt space.
*   **WHY**: PySide6's `QImage(numpy_array.data, ...)` wraps a lightweight pointer directly to the NumPy array. Because the background grab thread overwrites and garbage-collects this NumPy array immediately, **Qt is forced to read stale, partially corrupted heap memory**. This causes the colors in the GUI to become dull, washed out, and yellowish-gray. `.copy()` guarantees 100% stable, uncorrupted, rich BGR color representation.
*   **HOW**: `qt_img = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()`
*   **WHERE**: Done in `gui/widgets/image_display.py:L193`.
*   **MANDATORY?**: **YES — always** when displaying numpy buffers in PySide6 to prevent memory leaks and color distortion.
*   **Inference Effect**: None. This is a visual-display safety guard.
*   **Temporary or Permanent?**: **PERMANENT.**

---

### Step 3.2: Bilinear Viewport Scaling (`SmoothTransformation`)

*   **WHAT**: Downscales the display pixmap on the fly to match the dynamic size of the QLabel display cards.
*   **WHY**: Nearest-neighbor scaling (`FastTransformation`) throws away pixels, creating jagged edges and making the text on the watch face blocky and unreadable. Bilinear scaling renders smooth edges and high-fidelity textures.
*   **HOW**: `pixmap.scaled(disp_size, KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)`
*   **WHERE**: Done in `gui/widgets/image_display.py:L203`.
*   **MANDATORY?**: **YES** for premium visual quality.
*   **Inference Effect**: None.
*   **Temporary or Permanent?**: **PERMANENT.**

---

## 6. The Phase 6 Inference Path Architecture

### Is Full-Resolution Necessary for Inference?
**YES, absolutely.** High-speed grading models must detect tiny surface defects (e.g. skin punctures, early decay, and minor bruising) that only occupy a few pixels at full resolution ($2048 \times 1536$). Downsampling the images first to $640 \times 480$ would discard **90.2% of the spatial details**, rendering the AI model ineffective.

### How We Retrieve the Full-Resolution Stream Later for Inference:
We will introduce a separate, non-blocking `grab_raw()` method in `JAICamera`. The inference thread will poll this method independently. It will retrieve the raw, untouched high-resolution buffers directly from the GEV stream, apply **Fixed-Range Normalization** (consistent pixel scaling), and feed them straight to YOLO:

```
JAICamera.grab_raw()  (Pulls untouched 2048×1536 buffers)
        │
        ├── CH1 (BayerRG8) ──► Bayer Demosaic ──► Full-Res BGR (9.4 MB) ──► Crop/Pad ──► YOLOv8
        │
        ├── CH2 (Mono8) ────► Fixed Normalization (÷ 50.0) ──► Full-Res NIR1 (3.1 MB) ──► Concatenate
        │
        └── CH3 (Mono8) ────► Fixed Normalization (÷ 50.0) ──► Full-Res NIR2 (3.1 MB) ──► Concatenate
```

---

## 7. Summary Preprocessing Matrix

| Preprocessing Step | `camera_probe_jai.py` | `camera_live_view.py` | `camera_interface.py` (GUI) | Mandatory? | Inference Path Effect | Permanent? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bayer Demosaicing** | ✅ Yes (Full Res) | ✅ Yes (Full Res) | ✅ Yes (Full Res) | **YES** | REQUIRED for BGR color models. | **PERMANENT** |
| **Early Downsampling** | ❌ Raw (Full Res) | ❌ Raw (Full Res) | ✅ Yes ($640 \times 480$) | **Display Only** | **BYPASSED** (AI needs 2048×1536 detail). | **TEMPORARY** |
| **Max Tracking (Full-Res)** | ✅ Yes | ❌ No | ✅ Yes | **YES** | Protects the scaler from blurring. | **PERMANENT** |
| **Per-Frame Min-Max Norm** | ✅ Yes (normalized save) | ✅ Yes (`NORM_MINMAX`) | ❌ No | **No** | Bypassed. (Causes visual flicker). | **TEMPORARY** |
| **EMA Min-Max Norm** | ❌ No | ❌ No | ✅ Yes (C++ `convertScaleAbs`) | **Display Only** | **REMOVED** (Inference needs consistent scales). | **TEMPORARY** |
| **QImage Deep Copy** | ❌ No | ❌ No | ✅ Yes (`.copy()`) | **YES (GUI)** | None. (Visual color-safe guard). | **PERMANENT** |
| **Bilinear GUI Scaling** | ❌ No | ❌ No | ✅ Yes (Smooth Scale) | **YES (GUI)** | None. (Eliminates jagged jaggies). | **PERMANENT** |
