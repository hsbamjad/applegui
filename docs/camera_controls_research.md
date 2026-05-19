# Camera Control Parameters — Research for Apple GUI

> **Context:** JAI FS-3200T-10GE multispectral camera (3-source: Color BayerRG8 + NIR1 ~800nm + NIR2 ~900nm).
> Currently implemented: `ExposureTime` + `AcquisitionFrameRate`.
> Question: What else should we add?

---

## Overview: Two Layers of Controls

It is important to distinguish between **two fundamentally different kinds** of controls:

| Layer | Where it runs | Affects raw data? | GenICam? |
|---|---|---|---|
| **Hardware / Sensor** | Inside the camera firmware | YES — changes pixel ADC values | YES |
| **Software / Display** | In Python/OpenCV after grab | NO — only affects what you *see* | NO |

For scientific imaging (grading, spectral analysis), **hardware parameters are always preferred** because they affect the actual photons recorded. Software adjustments are cosmetic.

---

## TIER 1 — Implement First (High Priority, Hardware-native)

### 1. Gain (`DigitalAll` / `AnalogAll`)
| Attribute | Value |
|---|---|
| **GenICam name** | `GainSelector` → select `DigitalAll`, then set `Gain` |
| **eBUS API** | `nm.GetFloat("Gain").SetValue(db_value)` |
| **Unit** | dB (decibels) |
| **Range on FS-3200T** | 0 dB – 24 dB (Master); ±15 dB per channel |
| **Auto mode** | `GainAuto` → `Off` / `Continuous` / `Once` |
| **GUI widget** | `QDoubleSpinBox` (step 0.5 dB) + "Apply Gain" button |
| **Default** | 0 dB |

**Why it matters:**
- Gain is the **electronic amplification** of the signal after the sensor integrates light.
- For apple grading: NIR channels (~800 nm, ~900 nm) often have weak signal in daylight setups. Increasing gain boosts NIR visibility **without changing exposure** (which would also affect color channel brightness).
- Tradeoff: higher gain → more noise (shot noise). Practical limit is ~12 dB before image gets grainy.
- This is the most important control after exposure and FPS.

**Your professor is right** — this is the next control to add. It directly compensates for varying fruit surface reflectance and ambient lighting.

---

### 2. Black Level (`BlackLevel`)
| Attribute | Value |
|---|---|
| **GenICam name** | `BlackLevelSelector` → `All`, then `BlackLevel` |
| **eBUS API** | `nm.GetFloat("BlackLevel").SetValue(value)` |
| **Unit** | DN (Digital Number, 0–255 for 8-bit) |
| **Range** | ~0–64 (camera-dependent) |
| **GUI widget** | `QSpinBox` (step 1) + "Apply" button |
| **Default** | Typically 0 |

**Why it matters:**
- Black level is the **digital pedestal** — the baseline pixel value that represents "no light."
- The FS-3200T NIR sensors sometimes have a nonzero pedestal due to dark current, especially when sensor temperature rises during long sessions.
- Adjusting black level is equivalent to shifting the histogram's zero point. It removes the gray "fog" you sometimes see in NIR images.
- **Note:** Our current EMA normalization compensates for pedestal drift in software, but hardware black level correction is cleaner and more reproducible.

---

### 3. White Balance (Color Channel Only — CH1)
| Attribute | Value |
|---|---|
| **GenICam name** | `BalanceWhiteAuto` (`Off`/`Continuous`/`Once`) |
| **Manual GenICam** | `BalanceRatioSelector` → `Red`/`Blue`, then `BalanceRatio` |
| **eBUS API** | `nm.GetEnum("BalanceWhiteAuto").SetValue("Once")` |
| **Unit** | Ratio (dimensionless, ~0.5–4.0) |
| **GUI widget** | `QPushButton("Auto WB Once")` + separate R/B spinboxes for manual |
| **Presets** | 3200K (tungsten), 5000K (daylight), 6500K (cloudy), 7500K (shade) |

**Why it matters:**
- Applies **only to CH1 (color sensor)** — NIR channels are monochrome and unaffected.
- In an agricultural inspection setting, the conveyor lighting (typically LED arrays) has a specific color temperature that shifts throughout a day as LEDs warm up.
- Without correct white balance, apples that are actually red may appear slightly orange or pink, affecting color-based grading decisions.
- A "One-Push" Auto WB button is the easiest UX — operator clicks it once when setting up.

---

## TIER 2 — Implement Second (Medium Priority)

### 4. Gamma (`Gamma`)
| Attribute | Value |
|---|---|
| **GenICam name** | `Gamma` (some cameras: `GammaEnable` first) |
| **eBUS API** | `nm.GetFloat("Gamma").SetValue(value)` |
| **Unit** | Dimensionless (power curve exponent) |
| **Range** | 0.45 – 2.2 (typical); default 1.0 = linear |
| **GUI widget** | `QDoubleSpinBox` (step 0.05, range 0.45–2.2) |

**Why it matters:**
- Gamma applies a **power curve** to pixel values: `output = input^gamma`.
- Gamma < 1.0 (e.g., 0.7) brightens midtones — useful in NIR where signal is weak but you don't want to blow out highlights.
- Gamma = 1.0 → linear, which is what scientific imaging usually requires for quantitative spectral analysis.
- For display purposes, gamma = 0.5–0.8 can make NIR images look more natural to the human eye.
- **Important:** If you need reproducible spectral indices (like NDVI, which uses NIR), keep gamma = 1.0 and only use gain/exposure.

---

### 5. Pixel Format / Bit Depth
| Attribute | Value |
|---|---|
| **GenICam name** | `PixelFormat` (Enum) |
| **Options on FS-3200T** | `Mono8`, `Mono10`, `Mono12`, `BayerRG8`, `BayerRG10`, `BayerRG12` |
| **GUI widget** | `QComboBox` with available options |

**Why it matters:**
- Currently locked to Mono8 (8-bit) and BayerRG8. Switching to **Mono12 (12-bit)** for NIR channels gives **16× more dynamic range** — critical if apple surfaces vary from very dark (matte) to very bright (waxy/reflective).
- 12-bit data requires adjusting the display pipeline (our EMA normalization) since values go 0–4095, not 0–255.
- Tradeoff: 12-bit doubles bandwidth → may require reducing FPS.

---

### 6. ROI (Region of Interest) — Width / Height / OffsetX / OffsetY
| Attribute | Value |
|---|---|
| **GenICam names** | `Width`, `Height`, `OffsetX`, `OffsetY` |
| **eBUS API** | `nm.GetInteger("Width").SetValue(1024)` etc. |
| **GUI widget** | 4× `QSpinBox` in a sub-section; or presets ComboBox |
| **Presets** | Full (2048×1536), 3-lane (2048×512), Center crop, etc. |

**Why it matters:**
- Full resolution (2048×1536) at 107 FPS requires significant bandwidth.
- If the conveyor only occupies the central ~1536×768 pixels, cutting the ROI **doubles the achievable FPS** or halves the bandwidth.
- This is a standard machine vision optimization. For 3-lane apple sorting, you may only need the center rows.

---

## TIER 3 — Software Controls (Display Pipeline, Low Effort)

These don't touch the camera hardware — they're applied in `_process_raws()` via OpenCV. **No GenICam API calls needed.**

### 7. Brightness / Contrast (Display Only)
| Control | Implementation | Notes |
|---|---|---|
| **Brightness** | `cv2.convertScaleAbs(img, beta=offset)` | Shift pixel values +/- |
| **Contrast** | `cv2.convertScaleAbs(img, alpha=scale)` | Multiply pixel values |

This is what your professor likely means by "contrast." It's already partially done in our EMA normalization for NIR channels. Making it user-controllable is a small addition.

### 8. Sharpening (Display Only)
| Control | Implementation | Notes |
|---|---|---|
| **Sharpening** | Unsharp mask: `cv2.addWeighted(img, 1+k, blurred, -k, 0)` | k = 0–1 |

Useful for making defects on apple surface more visible in the live display.

### 9. NIR EMA Speed (`_EMA_ALPHA`)
Currently hardcoded at 0.05. Exposing this as a slider would let the operator control how fast the NIR auto-normalization adapts:
- **Low alpha (0.02):** Slow/stable — good for consistent lighting
- **High alpha (0.20):** Fast — good when lighting conditions change

---

## What NOT to Add (for now)

| Parameter | Reason to skip |
|---|---|
| `TriggerMode` | Already handled — free-run mode is correct for conveyor |
| `TestPattern` | Engineering only — not useful for operators |
| `TransportLayerControl` | Very advanced, should not be operator-accessible |
| `LineSelector` / GPIO | Relevant only when you integrate hardware triggers |

---

## Recommended Implementation Order

```
Phase 1 (Next sprint):
  ✅ Already done: ExposureTime, AcquisitionFrameRate
  🔲 Add: Gain (hardware) — most impactful
  🔲 Add: White Balance "One-Push" button (color channel)

Phase 2:
  🔲 Add: Black Level (hardware cleanup)
  🔲 Add: Brightness / Contrast sliders (display only, easy)
  🔲 Add: NIR EMA Alpha slider (already in code, just expose it)

Phase 3 (Advanced):
  🔲 Add: Gamma
  🔲 Add: Pixel Format / Bit Depth switcher
  🔲 Add: ROI presets
```

---

## eBUS API Quick Reference

```python
nm = device.GetParameters()

# Gain
nm.GetEnum("GainAuto").SetValue("Off")
nm.GetEnum("GainSelector").SetValue("DigitalAll")
nm.GetFloat("Gain").SetValue(6.0)         # 6 dB

# Black Level
nm.GetEnum("BlackLevelSelector").SetValue("All")
nm.GetFloat("BlackLevel").SetValue(8.0)

# White Balance (one-push)
nm.GetEnum("BalanceWhiteAuto").SetValue("Once")
# Wait for camera to complete (~500ms), then set to "Off"

# Gamma
nm.GetBoolean("GammaEnable").SetValue(True)   # if needed
nm.GetFloat("Gamma").SetValue(1.0)

# ROI
nm.GetInteger("Width").SetValue(2048)
nm.GetInteger("Height").SetValue(768)
nm.GetInteger("OffsetX").SetValue(0)
nm.GetInteger("OffsetY").SetValue(384)
```

> [!WARNING]
> All hardware parameter changes (Gain, Black Level, White Balance) should trigger `_nir_ema_reset_pending = True` just like `set_exposure()` does, so the NIR display adapts immediately.

> [!NOTE]
> Not all parameters on the FS-3200T are visible at "Beginner" visibility level in eBUS. Set `DeviceUserDefinedName` visibility to **Guru** when probing. In production GUI, use the specific Get* calls above — they bypass visibility levels.
