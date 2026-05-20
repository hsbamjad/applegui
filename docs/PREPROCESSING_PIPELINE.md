# JAI FS-3200T Multispectral Camera Preprocessing & Hardware Controls Pipeline Handbook

**Project:** Apple Sorting GUI — MSU ASABE AIM26  
**Applies to:** `core/camera/camera_interface.py` · `gui/widgets/image_display.py` · `gui/workers/camera_worker.py` · `gui/panels/camera_panel.py`  
**Last Updated:** May 20, 2026

---

## 1. System Overview: End-to-End Processing Architecture

The JAI FS-3200T multispectral camera contains three independent CMOS sensors in a single physical body. It streams three hardware-synchronized GigE Vision (GEV) channels over a single 10 GigE link. 

In our upgraded architecture, **frame acquisition is completely decoupled from visual representation**. The core camera backend operates strictly on raw, full-resolution buffers, while the frontend handles memory-safe rendering, scaling, and interactive ROI previews.

```mermaid
graph TD
    %% Physical Sensors
    subgraph Camera Hardware (JAI FS-3200T-10GE)
        S0[Source0: Color Sensor - BayerRG8]
        S1[Source1: NIR1 Sensor - Mono8]
        S2[Source2: NIR2 Sensor - Mono8]
    end

    %% GenICam Controls
    subgraph Hardware Calibration / Control Layer
        EX[Exposure & Gain Ramping]
        BL[Black Level Pedestal Subtraction]
        WB[One-Push Auto White Balance]
        ROI[GenICam Step-Aligned ROI]
    end

    S0 --> EX
    S1 --> BL
    S2 --> BL

    %% eBUS Streams
    subgraph eBUS Python SDK Driver
        D0[Stream Channel 0]
        D1[Stream Channel 1]
        D2[Stream Channel 2]
    end

    EX --> D0
    BL --> D1
    BL --> D2

    %% Grab Loop & Sync
    subgraph core/camera/camera_interface.py (JAI-grab Thread)
        GL[Background Grab Loop]
        VS[Block ID Sync Validation]
        BDR[Bayer Demosaicing COLOR_BayerBG2BGR]
        FT[FrameTriplet Object 2048x1536]
    end

    D0 & D1 & D2 --> GL
    GL --> VS
    VS -- "LAGGING SOURCE: Advance Frame Buffer" --> GL
    VS -- "SYNCED: GEV Block IDs Match" --> BDR
    BDR --> FT

    %% GUI Representation
    subgraph gui/widgets/image_display.py (PySide6 UI Thread)
        MC[MultiChannelDisplay]
        CP[ChannelPanel - CH1 / CH2 / CH3]
        MC_CP[Memory-Safe QImage Deep Copy .copy]
        ROIO[Interactive Cyan ROI Overlay]
        BLS[Bilinear smooth viewport scaling]
        DISP[High-Fidelity UI Rendering]
    end

    FT -->|QThread sig_frame Emit| MC
    MC --> CP
    CP --> MC_CP
    MC_CP --> ROIO
    ROIO --> BLS
    BLS --> DISP
```

---

## 2. Multi-Spectral Frame Triplet Format

Every successful acquisition outputs a `FrameTriplet` object consisting of three full-resolution NumPy arrays and precise timing indicators:

| Channel | Physical Sensor | Target Spectrum | Data Shape | Pixel Format | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CH1** | Color (Bayer RGB) | Visible (~400–670 nm) | `(1536, 2048, 3)` | `uint8` | Standard color image (Bayer demosaiced to BGR) |
| **CH2** | NIR 1 (Monochrome) | Near-Infrared (~800 nm) | `(1536, 2048)` | `uint8` | Reveals internal fruit structure & bruises |
| **CH3** | NIR 2 (Monochrome) | Near-Infrared (~900 nm) | `(1536, 2048)` | `uint8` | Captures different water absorption/reflectance bands |

---

## 3. Real-Time Hardware Frame Synchronization

Because eBUS retrieves buffers in separate threads for each physical channel, network jitter or firmware interruptions (e.g. during an ROI update) can cause channels to go out of phase.

### GigE Vision Block ID Sync Validation
In `camera_interface.py`, the background `JAI-grab` thread implements active re-synchronization based on the hardware GEV Block ID:
1. Every buffer popped from `PvPipeline` contains a monotonic `BlockID`.
2. The grab loop verifies that `BlockID_CH1 == BlockID_CH2 == BlockID_CH3`.
3. **Lagging Channel Recovery**: If a sync mismatch is detected (e.g., `[142, 140, 142]`), the loop identifies the maximum Block ID (`142`) and actively calls `.grab()` on the lagging channel (CH2) in a tight recovery loop to flush outdated buffers until it catches up.
4. This active correction recovers perfect phase synchronization within **1 to 3 frames** without interrupting the stream or requiring a device disconnect.

---

## 4. Hardware Calibration & Camera Controls

All adjustments are applied directly to the camera's analog and digital electronics inside the FPGA, ensuring maximum signal-to-noise ratio before analog-to-digital conversion.

### 4.1. Exposure Time (Per-Source Independent Ramping)
*   **What**: Integrates photons on each sensor independently.
*   **Safe-Ramping Protocol**: Ramping exposure times abruptly across three channels can trigger device driver timeouts and eBUS pipeline sync failures. We implement a **synchronized incremental ramping algorithm**:
    *   Exposures are stepped by a maximum of `4000 µs` per iteration with a `30 ms` inter-step delay.
    *   Ramp lengths are padded so that all channels conclude their ramping steps in perfect unison, preventing screen flashing and driver locks.
*   **Firmware Clamp**: Capped globally by FPS: $\text{ExposureTime (µs)} \le \frac{1,000,000}{\text{FPS}}$.

### 4.2. Frame Rate (FPS)
*   **What**: Adjusts the global clock of the sensor. 
*   **Dynamic Exposure Clamping**: Setting a high FPS (e.g., 30 to 60) automatically reduces the maximum integration period. Our system captures the `sig_fps_changed` signal, applies the value to `AcquisitionFrameRate` globally, queries the new firmware-clamped exposure limits, and immediately syncs the UI spinboxes to avoid out-of-bounds requests.

### 4.3. Per-Channel Gain
*   **What**: Logarithmic analog and digital amplification inside the sensor (0.0 to 16.0 dB).
*   **White-Balance Protection**: Gain writes are restricted strictly to GenICam `*All` selectors (`DigitalAll`, `AnalogAll`, or `All` depending on firmware structure). Sub-channel selectors (e.g., Red/Green/Blue) are explicitly bypassed in gain sweeps to prevent destroying the active white balance on CH1.

### 4.4. One-Push Auto White Balance & Revert
*   **What**: Calibrates color neutrality under the specific LEDs.
*   **Auto-White Balance Sequence**:
    1. Scopes parameters to `Source0` (Color CH1) using `PvGenStateStack`.
    2. Caches current Red, Green, and Blue ratios into `self._saved_wb` as a recovery point.
    3. Writes `BalanceWhiteAuto = Once` to launch the camera's internal FPGA calibration algorithm.
    4. Polls `BalanceWhiteAuto` every `50 ms` until it returns to `Off` (calibration complete).
    5. Reads the newly calculated digital red and blue gains and updates the UI readout.
*   **Revert Feature**: If calibration is performed against an invalid reference (e.g. a colored background instead of a grey card), the operator can click **↺ Revert** to restore the cached pre-calibration ratios directly to the GenICam registers.

### 4.5. Sensor-Level Black Level Subtraction
*   **What**: Subtracts the physical dark current pedestal (thermal noise and electronics bias) directly at the ADC stage.
*   **Why**: Under normal operation, the NIR channels have a fake floor of `~7 DN` (NIR1) and `~6 DN` (NIR2) even in complete darkness. This dark pedestal shifts all reflectance measurements, leading to grey glare on the conveyor belt and distorted AI inference values.
*   **How**: Scopes to `BlackLevelSelector = All` and adjusts the `BlackLevel` float parameter (0.0 to 64.0 DN) independently for each source. This reclaims the full 8-bit dynamic range, making true darkness evaluate to `0 DN`.

---

## 5. Region of Interest (ROI) Controls

Applying an ROI limits the sensor's active pixel area, reducing data rates and allowing high-speed crop matching. Because GenICam parameters are locked during active acquisition, we implement a safe, sequential stop-write-start routine.

> [!CAUTION]
> Writing ROI parameters in an incorrect sequence or failing to flush streams before/after will trigger driver exceptions, GenICam boundary violations, and immediate block ID desynchronization.

### The GenICam Safe Write Sequence
1.  **Stop Acquisition**: Calls `AcquisitionStop` on all three streams to release hardware resource locks.
2.  **Drain Pre-ROI Buffers**: Flushes up to 16 stale buffers remaining in each `PvPipeline` stream to prevent old full-size frames from mixing with new cropped frames.
3.  **SFNC Write-Order Execution**:
    *   *Constraint*: $\text{OffsetX} + \text{Width} \le \text{MaxWidth (2048)}$ and $\text{OffsetY} + \text{Height} \le \text{MaxHeight (1536)}$.
    *   *Order*:
        1. Set `OffsetX = 0`, `OffsetY = 0` (expands physical boundaries).
        2. Set `Width = MaxWidth`, `Height = MaxHeight` (resets any previous crops).
        3. Write target **Width** and **Height** *first* (shrinks the bounding box).
        4. Write target **OffsetX** and **OffsetY** *second* (moves the bounding box).
4.  **Restart Stream & Post-Drain**: Calls `AcquisitionStart` sequentially. Because sources start at slightly different millisecond offsets, we apply a brief `50 ms` settling delay and discard the first 5 frames from each stream to guarantee the background loop begins with all channels completely in phase.

---

## 6. Real-Time Viewport Rendering & Display Pipeline

To achieve premium UI performance (stable 30+ FPS) without taxing the CPU, all heavy math and downsampling operations have been completely removed from the camera worker thread.

### Step 1: Grayscale and Color Standardization
All incoming frames are processed at full resolution. If a monochrome frame (`ndim == 2`) is received, it is converted to RGB via `cv2.COLOR_GRAY2RGB` to ensure absolute display compatibility. Color channels are converted from standard BGR to RGB.

### Step 2: Memory-Safe Deep Copy (`.copy()`)
PySide6's `QImage` wrapper maps a light pointer directly to the underlying NumPy array. Since the JAI grab thread overwrites and recycles frame buffers rapidly, failing to duplicate this memory causes Qt to read partially corrupted heap blocks. This produces dull, yellowish-gray tones and visual horizontal tearing. Calling **`.copy()`** is **mandatory** to duplicate the pixel buffer into a persistent, dedicated memory block in Qt space, guaranteeing 100% stable, rich BGR color representation.

### Step 3: Bilinear Viewport Scaling (`SmoothTransformation`)
*   **Problem**: High-speed nearest-neighbor scaling (`FastTransformation`) creates severe aliasing, making text or fine defects appear jagged and blocky.
*   **Solution**: The `ChannelPanel` compares the frame's dimensions to the current display card `QLabel` size. If they differ, it applies PySide6's highly optimized **`SmoothTransformation`** (bilinear interpolation) to render smooth edges and high-fidelity surface details.

### Step 4: Aspect-Ratio Aware Cyan ROI Overlay
While an operator is dragging spinboxes, the UI draws a live preview overlay:
*   A dashed **cyan border** `#06b6d4` with solid corner tick marks outlines the targeted region.
*   Everything *outside* the targeted ROI is dimmed with a **translucent dark mask** (`rgba(0, 0, 0, 120)`).
*   **Cropped Space Mapping**: If a hardware ROI is already active (e.g. streaming a $1648 \times 1536$ sub-frame), the coordinates of the preview ROI are mapped dynamically relative to the active ROI's origin, accounting for the current aspect ratio and letterboxing. A floating pill label reads `Width × Height @ (OffsetX, OffsetY)`.

---

## 7. Decoupled AI Inference Path (Future Architecture)

For production apple grading, the AI model requires **untouched, full-resolution raw pixels** to detect microscopic defects like punctures or early decay. The dynamic, display-oriented EMA normalization is bypassed entirely:

```
                  JAICamera.grab() [Pulls synchronized full-res 2048x1536 raw buffers]
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
CH1: Visible RGB            CH2: NIR1 Mono8             CH3: NIR2 Mono8
       │                           │                           │
Bayer Demosaic (BGR)        Fixed Normalization         Fixed Normalization
 (cv2.COLOR_BayerBG2BGR)     (Divide by 255.0)           (Divide by 255.0)
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   ▼
                   Concatenate Channels (C, H, W)
                                   │
                   Crop & Pad (640x640 letterbox)
                                   │
                      YOLOv8 Spectral Grading
```

*   **Fixed-Range Normalization**: Unlike visual normalization (which stretches pixels per-frame and makes identical apples evaluate to different values if lighting shifts), the AI path divides the raw intensity directly by the physical limits of the simultaneous multisource setup, preserving absolute, repeatable radiometric intensity.
