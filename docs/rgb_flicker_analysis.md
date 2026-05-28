# RGB Channel Flicker at High FPS - Root Cause Analysis

**Date:** 2026-05-20
**Camera:** JAI FS-3200T (3-source multispectral)
**Observed by:** Hardware testing during ROI/FPS calibration

---

## Symptom

When FPS is increased beyond a certain threshold, the **CH1 (RGB/Color)** channel
begins flickering - rapid brightness oscillation (high → low → high) visible in the
live feed. CH2 (NIR1 ~800 nm) and CH3 (NIR2 ~900 nm) are **unaffected**.

---

## Root Cause: Light Source Flicker Aliasing

The light source inside the sorting chamber is **not truly constant**. Fluorescent
tubes and most LED panels are powered by AC mains, causing the light output to pulse
at **twice the grid frequency**:

| Grid frequency | Light pulse frequency |
|---|---|
| 60 Hz (US) | **120 Hz** |
| 50 Hz (EU) | **100 Hz** |

This pulsing is invisible to the human eye (too fast) but the camera captures it.

### The Aliasing Mechanism

Each camera frame integrates light over its exposure window. If the camera FPS does
**not divide evenly** into the light pulse frequency, consecutive frames catch the
light at different phases of its cycle - one frame at peak brightness, the next at
trough. This appears as brightness oscillation.

```
Light intensity (120 Hz pulse):
  ▲
  │  ╭─╮   ╭─╮   ╭─╮   ╭─╮
  │╭─╯ ╰─╮╭╯ ╰─╮╭╯ ╰─╮╭╯ ╰─╮
  └──────────────────────────▶ time

Frame timing examples (US grid, 120 Hz light):

  30 FPS → 120 / 30 = 4.00 cycles/frame → integer → STABLE
  40 FPS → 120 / 40 = 3.00 cycles/frame → integer → STABLE
  60 FPS → 120 / 60 = 2.00 cycles/frame → integer → STABLE

  45 FPS → 120 / 45 = 2.67 cycles/frame → NOT integer → FLICKERING
  50 FPS → 120 / 50 = 2.40 cycles/frame → NOT integer → FLICKERING
  48 FPS → 120 / 48 = 2.50 cycles/frame → NOT integer → FLICKERING
```

### Why Only RGB - Not NIR?

The NIR channels (CH2 ~800 nm, CH3 ~900 nm) use **narrowband optical filters**.
Standard fluorescent tubes and most LED panels emit negligible energy above 750 nm.
The NIR channels are effectively isolated from visible-spectrum flicker.

The color channel (CH1) captures the full visible spectrum (400-700 nm) where the
light pulse energy is concentrated - making it the only channel affected.

---

## Safe FPS Values (US 60 Hz grid, 120 Hz light)

Use FPS values that are **integer divisors of 120**:

| FPS | 120 / FPS | Status |
|-----|-----------|--------|
| 20  | 6.00      | Safe |
| 24  | 5.00      | Safe |
| 30  | 4.00      | Safe |
| 40  | 3.00      | Safe |
| 60  | 2.00      | Safe |
| 120 | 1.00      | Safe |
| 25  | 4.80      | Flicker |
| 45  | 2.67      | Flicker |
| 48  | 2.50      | Flicker |
| 50  | 2.40      | Flicker |

> **For EU 50 Hz grid (100 Hz light):** safe values are divisors of 100 - 20, 25,
> 50, 100 FPS.

---

## Exposure-Based Fix (Camera-Side)

Even at non-integer FPS, flickering can be **eliminated by setting the exposure time
to an exact multiple of one light cycle**. The frame integrates a whole number of
complete flicker cycles regardless of phase - the total light captured is always the
same.

| Grid | Light cycle period | Safe exposure multiples |
|---|---|---|
| 60 Hz (US) | 1/120 s = **8,333 µs** | 8333, 16667, 25000 µs … |
| 50 Hz (EU) | 1/100 s = **10,000 µs** | 10000, 20000, 30000 µs … |

**This is the recommended fix** - it allows any FPS to be used without flicker.

### Example (US lab, targeting 45 FPS):
```
Light cycle = 8333 µs
Max exposure at 45 FPS = 1,000,000 / 45 = 22,222 µs
Largest safe multiple ≤ 22,222 µs: 2 × 8333 = 16,666 µs

→ Set Exposure to 16,666 µs (or 16,667 µs rounded)
→ Flickering eliminated at 45 FPS
```

---

## Long-Term Fix (Lighting-Side)

The permanent solution is to replace the chamber light source with one that does
**not flicker at line frequency**:

| Option | Flicker | Notes |
|---|---|---|
| Standard fluorescent tube | 100/120 Hz | Avoid |
| Standard LED strip (AC-driven) | 100/120 Hz | Avoid |
| **Halogen / incandescent** | Negligible | Thermal inertia smooths flicker |
| **DC-powered LED** (constant current driver) | None | Best choice |
| **Machine vision strobe** (triggered) | None | Synced to camera trigger |
| High-frequency ballast fluorescent (>=20 kHz) | None | Acceptable |

For a production sorting line, a **DC-powered LED ring/panel** or **triggered strobe**
synchronized to the camera's trigger signal is the gold standard.

---

## Implementation Notes (Future)

If a strobe-sync solution is adopted:
- The JAI FS-3200T supports **hardware trigger input** (GPIO Line 0)
- Set `TriggerMode = On`, `TriggerSource = Line0`
- Pulse the strobe on the camera's **ExposureActive** output signal
- This ensures the light is only on during the exact exposure window

See `docs/camera_controls_research.md` for GenICam parameter reference.

---

## Summary

| Factor | Detail |
|---|---|
| **Root cause** | FPS not a divisor of light pulse frequency (120 Hz US) |
| **Affected channel** | CH1 (RGB) only - NIR immune due to narrowband filters |
| **Quick fix** | Set exposure = multiple of 8,333 µs (US) / 10,000 µs (EU) |
| **Permanent fix** | DC-powered LED or camera-triggered strobe |
| **Safe FPS (US)** | 20, 24, 30, 40, 60, 120 |
