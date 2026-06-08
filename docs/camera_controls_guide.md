# Camera Controls - Tier 1 Reference Guide
### JAI FS-3200T Multispectral System · Apple Sorting GUI

> Everything in Sections 1 through 6 of this document happens at the **hardware/firmware level** inside the camera.
> No software math, no pixel manipulation after the fact. What you set is what the sensor does.
> Section 7 details the **host PC software pipeline** that handles the frames after they leave the camera.

---

## The Big Picture

Your JAI FS-3200T camera has **three sensors in one body**:

| Channel | Sensor | What it sees |
|---|---|---|
| **CH1** | Color (Bayer RGB) | Visible light - what your eyes see |
| **CH2** | NIR 1 (Monochrome) | Near-infrared - invisible to eyes, reveals internal fruit structure |
| **CH3** | NIR 2 (Monochrome) | Near-infrared - different wavelength band |

Each sensor has its own analog readout circuit, amplifier, and ADC (Analog-to-Digital Converter). The controls below tune each of those stages - **before the image is ever digitized and sent to your PC**.

---

## 1. Exposure Time

### What it is in plain English
**How long the camera's "shutter" stays open per frame.**

Think of a bucket under rain. Exposure time = how long you hold the bucket out. Longer = more water (light) collected = brighter image.

### The analogy
- Short exposure (1,000 µs) → quick snapshot → freezes fast motion, but dark
- Long exposure (20,000 µs) → slow snapshot → bright, but fast objects blur

### What it does to the image
| Exposure ↑ | Exposure ↓ |
|---|---|
| Brighter pixels | Darker pixels |
| More motion blur on fast apples | Frozen, sharp fruit |
| Risk of saturating (blowing out) to 255 | Risk of too dark to see detail |

### How it's applied
The firmware register is `ExposureTime` (in microseconds, µs). We scope to each source independently using `SourceSelector`, then write the value. The hardware then controls the physical integration window of each photodiode array.

**Exposure Time Ramping Helper**: If we change the exposure time too abruptly across all channels, it can trigger device driver timeouts and sync losses. To prevent this, our backend writes exposure changes in small, synchronized steps of maximum `4000 µs` with a brief `30 ms` delay between steps, ensuring the screen brightness transitions smoothly without driver lag or sync glitches.

**Hard constraint:** `ExposureTime ≤ 1,000,000 ÷ FPS`
At 30 FPS → max exposure = 33,333 µs. If you set FPS first and then a higher exposure, the firmware silently clamps it.

### Practical use
- Start at **5,000-10,000 µs** for a lit conveyor
- Increase if image is too dark; decrease if it's washed out
- CH1 (Color) and CH2/CH3 (NIR) may need very different values - NIR LEDs and visible LEDs have different intensities

---

## 2. Frame Rate (FPS)

### What it is in plain English
**How many complete images the camera captures per second.**

Like a film reel - 30 FPS means the camera takes 30 full snapshots every second. Your apple moves between each snapshot.

### What it does
- **Higher FPS** → more images per apple → more chances to catch it perfectly → **shorter max exposure** (tradeoff)
- **Lower FPS** → fewer images → longer allowed exposure → brighter possible images

### How it's applied
The firmware register is `AcquisitionFrameRate`. When you change it, the camera recomputes the maximum allowed exposure time automatically. Our code reads back the clamped exposure after every FPS change and updates the UI spinboxes to reflect what the firmware actually accepted.

### Practical use
- **30 FPS** is a good default for a conveyor
- If apples are blurry at 30 FPS, try increasing FPS (and compensating with gain)
- FPS affects all 3 channels simultaneously - it's a global clock

---

## 3. Gain

### What it is in plain English
**An electronic amplifier that multiplies the sensor signal before converting it to a number.**

The classic analogy: a microphone amplifier. If the sound (light) is quiet, you turn up the gain to make it louder (brighter). But if you turn it up too much, you also amplify the hiss (noise).

### The physics
Each photodiode generates electrons when hit by photons. Gain amplifies the electrical signal from those electrons **before the ADC converts it to a digital number**. This is analog amplification - it happens in the real world before any math.

Unit: **dB (decibels)** - a logarithmic scale.
- 0 dB = no amplification (1×)
- 6 dB ≈ 2× amplification
- 12 dB ≈ 4× amplification
- Max on JAI FS-3200T = **16 dB**

### What it does to the image
| Gain ↑ | Gain ↓ |
|---|---|
| Brighter image | Darker image |
| More visible noise (grainy look) | Cleaner, smoother image |
| Can mask fine color/structure differences | Preserves true dynamic range |

### Key difference from Exposure
| | Exposure | Gain |
|---|---|---|
| Works by | Collecting more photons | Amplifying existing signal |
| Noise effect | Less noise (more real signal) | More noise (amplifies noise too) |
| Motion blur | Yes, at long exposures | None |
| **Preference** | Use first | Use only when exposure can't go higher |

### How it's applied
Firmware register: `GainSelector=AnalogAll` (or `DigitalAll`) → `Gain` (float, dB). We scope each source independently.

To protect your White Balance calibration on the color channel, our gain-changing routine **never** writes to the Red, Green, or Blue sub-channels. It only applies gain to the master channel selectors (`DigitalAll` or `AnalogAll`), ensuring all colors are boosted uniformly without introducing color drift.

### Practical use
- Keep gain as low as possible - **noise is the enemy of AI inference**
- If you need brightness, try more exposure first, then add gain
- CH2/CH3 NIR channels at low illumination may need 4-8 dB
- Color CH1 for fruit color inspection: keep ≤ 6 dB to preserve color fidelity

---

## 4. White Balance (Auto WB + Revert)

### What it is in plain English
**Telling the camera what "white" looks like under your specific lights, so colors appear accurate.**

Your inspection LEDs have a color. Fluorescent lights look slightly green. Incandescent looks orange. Your conveyor LEDs look whatever they look like. Without calibration, the camera inherits that tint into every image - a red apple might look orange, a green apple might look yellow.

White Balance removes that tint at the hardware level by adjusting how much Red, Green, and Blue the sensor amplifies relative to each other.

### The analogy
Imagine your eyes adjust when you walk from daylight into a room lit by yellow incandescent bulbs. After a few seconds, white paper still looks white to your brain - because your visual system recalibrates. White Balance does the same thing for the camera, except it does it with math (multiply R, G, B channels by different factors).

### The numbers
White Balance is expressed as three multipliers: **R ratio, G ratio, B ratio.**
- G (green) is always the reference = 1.0
- R and B are adjusted relative to green
- Example from your test: `R=0.54, G=1.0, B=1.65`
  → Red is dimmed by 46%, Blue is boosted by 65%
  → This removes blue-heavy tint from your LEDs

### One-Push Auto WB - what actually happens
1. You point the camera at a **neutral grey or white reference** (piece of white paper, a white calibration tile)
2. Click ** Auto WB** in the UI
3. The firmware sets `BalanceWhiteAuto = Once`
4. The camera measures the R, G, B output from that neutral target
5. It computes: "how much should I multiply each channel so they all equal neutral?"
6. It writes those multipliers to `GainSelector=DigitalRed/DigitalBlue` registers
7. The flag reverts to `Off` - calibration is locked in
8. Our code polls that flag every 50ms, reads the new ratios, and displays them

> **This is 100% hardware.** The firmware does the measurement and the multiplication inside the camera's FPGA. Your PC never touches the raw pixels - only the final calibrated output comes down the GigE cable.

### Revert button
Before triggering AWB, we save the current R/G/B ratios internally. If the result looks wrong (e.g., bad reference target), click **↺ Revert** to restore the previous values - written directly back to the firmware registers.

### Why it only applies to CH1 (Color)
CH2 and CH3 are monochrome NIR sensors. They have no Red/Green/Blue - just a single intensity value per pixel. White Balance is a color concept and physically doesn't exist on monochrome sensors.

### Practical use for apple sorting
- Do this **once per inspection session**, before starting
- Use a **white calibration tile** or matte white paper on the conveyor
- Redo if you change LED power, add/remove lights, or move the system
- Without WB: "red" and "green" apple classification may drift with LED temperature

---

## 5. Black Level

### What it is in plain English
**Setting the camera's definition of "zero light" to actually be zero.**

Even with no light at all - lens cap on, pitch black - a camera sensor reports a small positive number instead of zero. This is called the **dark pedestal**. It comes from:
- **Thermal noise** - heat causes electrons to randomly appear in the photodiode, even without photons
- **Amplifier bias** - the readout electronics have a built-in offset to keep the signal in the positive voltage range

### The analogy
Imagine a kitchen scale that reads "7 grams" when nothing is on it. Every weight you measure is off by 7. You'd calibrate it by pressing "tare" (zero) to subtract that offset. Black Level is the camera's tare.

### What it does to the image
| Black Level = 0 (default) | Black Level = 7 |
|---|---|
| True dark → 7 DN (fake signal) | True dark → 0 DN (correct) |
| 249 usable steps out of 256 | 256 usable steps out of 256 |
| Biased features for AI inference | Truthful pixel values |

Setting `BlackLevel=7` tells the ADC: "subtract 7 from every pixel output before sending it." True darkness becomes 0. The full 8-bit range is reclaimed.

### How it's applied
Firmware registers: `BlackLevelSelector=All` → `BlackLevel` (float, DN). Applied independently per source - NIR channels often have different pedestals than the Color sensor because they use different amplifier circuits.

### How to measure the right value
1. **Block all light** from each channel (lens cap, or cover the conveyor and turn off LEDs)
2. Look at the **minimum pixel value** in a dark frame
3. That number is your pedestal - set BlackLevel to it

From your NIR logs: `CH2 ≈ 7 DN`, `CH3 ≈ 6 DN` → start with those.

### Why it matters for apple sorting
NIR reflectance ratios are the core signal your AI uses for internal quality. If NIR pixel values have a fake 7 DN floor:
- Dark regions appear brighter than they are
- Ratio calculations (band 1 ÷ band 2) are biased
- The AI model was potentially trained on "un-calibrated" data - adding black level correction later could shift the distribution

> **For research-grade data:** always apply black level calibration. It ensures the pixels you log today match the pixels from a session three weeks ago.

---

## 6. Region of Interest (ROI)

### What it is in plain English
**Focusing the camera's eye on a specific sub-rectangle of the scene, physically cutting out irrelevant areas.**

Imagine looking at a stage through a cardboard tube. You physically restrict your field of view to only see the lead actor, completely blocking out the empty stage wings.

### The analogy
Instead of using software to crop an image *after* it arrives on your computer, a hardware ROI tells the physical sensor array: *"Do not even read or transmit the pixels outside this crop box."*

### What it does to the image
*   **Visual Dimension Crop**: The image size drops physically (e.g., from full-size $2048 \times 1536$ down to a conveyor-hugging $1648 \times 1536$ crop).
*   **Network & CPU Load Relief**: Because the camera only transmits the cropped region, it sends **far fewer bytes over the network wire**. This lowers bandwidth usage and drastically reduces your PC's CPU processing loads.
*   **AI Focus**: Prevents the AI model from looking at structural brackets, cables, or background distractions outside the conveyor lane.

### How it's applied
Firmware registers: `Width`, `Height`, `OffsetX`, `OffsetY` (integers). These parameters are **step-aligned** (typically in steps of 16 pixels horizontally and 8 pixels vertically) by the physical hardware layout of the sensor's readout lines.

---

> [!CAUTION]
> **The GenICam Parameter Lock & Sync Constraints**
>
> Because modifying the active pixel grid alters the sensor's electronic readout pathways, the camera **strictly locks** these parameters during active streaming. We have implemented highly specialized hardware sequence guards to handle this.

### Safe Write Order (The SFNC Protocol)
The camera's firmware enforces a strict spatial boundary constraint: $\text{OffsetX} + \text{Width} \le \text{MaxWidth}$ and $\text{OffsetY} + \text{Height} \le \text{MaxHeight}$.

If you write these in the wrong order (e.g. trying to push the `OffsetX` window to the right before shrinking the `Width` box), the camera will throw a boundary violation exception and reject the command.

Our backend uses a bulletproof write order:
1.  **Stop streaming** on all channels (`AcquisitionStop`).
2.  **Reset Offsets to 0** (`OffsetX=0, OffsetY=0`) so the width bounds are completely open.
3.  **Expand Width and Height to maximum** to clear any old crop settings.
4.  **Write target Width and Height** first (shrinking the integration box safely).
5.  **Write target OffsetX and OffsetY** second (sliding the shrunken box to its correct location).
6.  **Restart streaming** on all channels (`AcquisitionStart`).

### Buffer Draining (Preventing Synchronization Scrambles)
When acquisition is stopped and restarted, up to 16 old full-sized frames remain stuck in the network driver's pipeline buffers. When the streams restart sequentially, these old frames mix with the new cropped frames. This causes a **Block ID Mismatch**, scrambling your display and crashing the pipeline.

To fix this, our software performs a double-flush action:
1.  **Pre-Write Drain**: We pull and release all stale buffers immediately after calling `AcquisitionStop`.
2.  **Post-Restart Drain**: After calling `AcquisitionStart`, we introduce a `50 ms` settling delay and discard the first 5 frames from each channel. This ensures all three sources begin active grabbing completely in phase, at the exact same GEV Block ID.

---

## 7. The Host PC Software Pipeline (What happens after the camera?)

### The Execution Boundary: Hardware vs. Software

When developing a research-grade vision system, it is vital to know exactly **where** your image is being processed.

If an operation happens **inside the camera (Hardware/FPGA level)**, it physically alters how light is measured and digitized.
If an operation happens **on your computer (Software/PC level)**, it is post-processing. While post-processing is highly useful for visualization and display, **we must never dynamically alter the raw pixel numbers used by your AI model or logged in your research data.**

Here is exactly where the boundary lies in our conveyor system:

| Phase | Operation | Where it executes | Does it alter raw saved pixel data? |
|---|---|---|---|
| **1. Sensor Capture** | Photon collection & Exposure timing | **Camera (Hardware)** | **Yes** (determines physical signal level) |
| **2. Analog Boost** | Analog Gain amplification (dB) | **Camera (Hardware)** | **Yes** (amplifies analog voltages) |
| **3. Digitization** | Black Level offset subtraction (pedestal) | **Camera (Hardware)** | **Yes** (sets the baseline zero value) |
| **4. Color Calibration** | White Balance multipliers (R/G/B ratios) | **Camera (Hardware)** | **Yes** (balances color channels on FPGA) |
| **5. Spatial Crop** | Sensor physical Region of Interest (ROI) | **Camera (Hardware)** | **Yes** (limits which photodiodes read out) |
| **6. Color Interpolation**| Bayer Demosaicing (Raw BayerRG8 → BGR) | **PC CPU (Software)** | **Yes** (reconstructs 3-channel color image) |
| **7. NIR Standardization** | Mono8 → 3-Channel BGR Expansion | **PC CPU (Software)** | **No** (only standardizes dimensions for display) |
| **8. Memory Isolation** | Deep copy allocation (`.copy()`) | **PC RAM (Software)** | **No** (pure memory safety measure) |
| **9. Display Scaling** | Viewport Bilinear `SmoothTransformation` | **PC GPU (Software)** | **No** (only affects the visual screen size) |
| **10. UI Annotation** | Interactive Guideline & Box Overlays | **PC GUI (Software)** | **No** (only drawn on glass overlay) |

---

### The 5 Software-Side Operations Explained

After the JAI camera finishes its physical sensor-level magic, it bundles the raw pixels and pushes them over the GigE Network cable. When these packets reach your host PC, our PySide6 software GUI executes **five critical operations** to handle, isolate, and display the images safely and beautifully.

#### 1. Bayer Demosaicing (Color Reconstruction)
*   **The Plain-English Analogy**: A "paint-by-numbers" canvas where each square has only one color spot: Red, Green, or Blue. You can't see the scene clearly if you only look at individual dots. Demosaicing is the digital artist who looks at the neighboring spots and smoothly fills in the gaps, revealing a rich, full-color picture.
*   **What it does**: The color sensor (CH1) has a Bayer filter grid. Each physical pixel only records a single color channel (e.g., Red or Green or Blue in a `BayerRG8` grid). Our software on the PC takes this single-channel raw stream and runs a demosaicing algorithm. This interpolates the missing colors for every single pixel, turning the raw grid into a standard, fully colored 3-channel BGR image.
*   **Why it's in software**: The camera hardware *could* do this internally, but doing it on the PC CPU saves valuable camera FPGA bandwidth, allowing the camera to stream raw frames faster and with lower lag.

#### 2. Channel Standardization (NIR Mono8 to 3-Channel BGR)
*   **The Plain-English Analogy**: A puzzle board that only accepts 3D blocks. Two of your pieces are flat gray cardboard squares (1-channel monochrome images), and the third is a thick wooden color block (3-channel BGR color image). Instead of breaking the puzzle board, you duplicate the gray cardboard squares three times to glue them together into thick 3D blocks so they fit the slots perfectly.
*   **What it does**: PySide6 and computer graphics frameworks (like OpenGL/GPU pipelines) expect all display elements to use the same memory structure-typically 3-channel (Red, Green, Blue) format. Because CH2 and CH3 are monochrome Near-Infrared (NIR) sensors, their raw streams are single-channel `Mono8` (just a grayscale brightness value). Our software duplicates this single grayscale channel three times to create a 3-channel BGR structure.
*   **Why it's in software**: This is purely a compatibility layer for the PySide6 display widgets. By standardizing all three channels to a 3-channel layout, our display code can run at lightning speed, treating color and NIR channels identically. **Crucially, this does not alter or corrupt the underlying monochrome data**-the raw 1-channel grayscale bytes are still saved completely untouched in the background research logs.

#### 3. Memory-Safe Deep Copying (`.copy()`)
*   **The Plain-English Analogy**: Passing around a fragile glass vase. If one person is painting the vase while another is trying to take a photo of it, it will slip, drop, and shatter. Instead of passing the actual vase, you take a quick snapshot, make a flawless replica of it (`.copy()`), and hand the replica to the display team. They can look at it as long as they want while you continue work on the next vase.
*   **What it does**: The eBUS driver writes incoming camera frames directly into shared C++ memory buffers (`PvBuffer`). Python references this memory using pointer wrappers to achieve zero-copy speed. However, PySide6's GUI thread runs asynchronously from the camera's acquisition thread. If the camera thread overwrites the memory buffer *while* the GUI thread is in the middle of drawing it to the screen, Python will crash with a segmentation fault (`Access Violation`). To prevent this, our software forces a deep `.copy()` of the image data in RAM before passing it to the display widgets.
*   **Why it's in software**: This is the ultimate shield for software stability. It ensures that the high-speed camera acquisition thread and the graphical user interface never step on each other's toes, completely eliminating memory corruption and app crashes during heavy runs.

#### 4. Bilinear Viewport Resizing (`SmoothTransformation`)
*   **The Plain-English Analogy**: Looking at a high-definition photograph through a magnifying glass versus cutting the photograph with scissors. The photo stays exactly $2048 \times 1536$ pixels in your album, but you zoom in or scale it down so it fits nicely inside your pocket viewer without losing its smooth edges.
*   **What it does**: The JAI camera captures and streams massive, high-detail frames (up to $2048 \times 1536$ per channel) to ensure the AI can spot tiny skin blemishes on apples. If we tried to show this full resolution on a standard computer monitor, the three frames would overflow the screen entirely. Our software uses PySide6's bilinear `SmoothTransformation` to dynamically scale the display viewports down on the fly, rendering a clean, smooth, non-pixelated preview in the GUI window.
*   **Why it's in software**: This scaling happens strictly inside the visual viewport widget (`image_display.py`). It is a temporary optical illusion for the user's eyes. The actual high-resolution, unscaled raw pixel data received from the camera remains fully intact, completely un-manipulated, and is written directly to the disk for research logging and AI model training.

#### 5. Interactive Viewport HUD Overlays
*   **The Plain-English Analogy**: Drawing a box with a dry-erase marker on a clear protective glass sheet placed over your TV screen. You aren't modifying the actual TV show; you're just laying a transparent grid over it to help you target where the actors are standing.
*   **What it does**: To help you position the conveyor and align the apples, the GUI draws real-time guidelines, coordinate numbers, and dynamic crop rectangles (HUD overlays) directly on the screen.
*   **Why it's in software**: Just like the viewport resizing, this drawing happens entirely in the presentation layer. The red target box or ROI guideline is drawn on top of the PySide6 canvas. The actual saved frame has absolutely no drawings, lines, or burned-in text. Your AI model gets pristine, clean, raw apple pixels, while you get an informative, interactive dashboard helper.

---

## Summary Table

| Control | Stage | What changes | CH1 Color | CH2 NIR1 | CH3 NIR2 |
|---|---|---|---|---|---|
| **Exposure** | Camera Hardware | Brightness via physical photon integration window |  |  |  |
| **FPS** | Camera Hardware | Master clock timing & maximum exposure budget |  (global) |  (global) |  (global) |
| **Gain** | Camera Hardware | Analog amplification multiplier, adds noise |  |  |  |
| **White Balance** | Camera Hardware | Digital multiplier ratios on FPGA chip |  only |  N/A |  N/A |
| **Black Level** | Camera Hardware | Dark pedestal offset subtraction at ADC |  |  |  |
| **Region of Interest** | Camera Hardware | Physical crop boundaries and sensor readout size |  |  |  |

---

## Recommended Calibration & Setup Order

```
1. Set FPS first            →   defines maximum exposure budget
2. Set ROI next             →   defines active sensor dimensions & limits
3. Set Exposure             →   get the ballpark brightness right
4. Set Gain                 →   fine-tune brightness with minimum noise
5. Black Level Calibration  →   calibrate zero point (dark frame, lens cap/dark lane)
6. White Balance            →   calibrate color neutrality (white reference under active lights)
7. GUI Viewport Adjustment  →   software scaling, guides, and HUD visual overlays
```

This order matters because each physical hardware step impacts the electronic and mathematical constraints of the subsequent stage.
