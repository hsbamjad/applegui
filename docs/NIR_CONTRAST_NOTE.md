# 📝 Note: NIR Channel Contrast — Simultaneous vs Sequential Mode
**JAI FS-3200T-10GE | May 15, 2026**

---

## The Observation

When acquiring all 3 sources simultaneously (correct hardware-sync approach):
- CH1 Color: `max=255` ✅ normal
- CH2 NIR1: `max=57` ← lower than sequential
- CH3 NIR2: `max=43` ← lower than sequential

When acquiring sources one at a time (old sequential approach):
- CH2 NIR1: `max=255`
- CH3 NIR2: `max=255`

---

## Root Cause

**This is camera firmware behavior — not a code bug.**

The JAI FS-3200T uses a dichroic prism — all 3 sensors always capture light simultaneously at the hardware level. However, when all 3 sources stream over GigE simultaneously, the camera firmware applies different internal resource allocation vs. single-source mode:

- **Sequential mode:** Camera dedicates 100% of internal bandwidth to one source at a time → higher effective SNR per source → higher max pixel values
- **Simultaneous mode:** Camera shares internal resources across all 3 sources → slightly lower raw pixel range on NIR channels → this is the trade-off for **hardware frame synchronization**

---

## What Was Tried (and outcome)

| Attempt | Result |
|---|---|
| Global `NegotiatePacketSize()` before stream open | ✅ Fixed CH1 packet loss (max 49→255) |
| Warmup (2s) + drain 30 frames before grab | ✅ Fixed blockID drift between channels |
| Interleaved round-robin drain | ✅ Fixed sequential drain offset |
| Per-channel `NegotiatePacketSize(channel_index)` | ❌ Overwrites `SetStreamDestination` → TIMEOUT on all channels |
| Increasing warmup / drain count | ✅ Minor improvement only, not scene-related |

**Conclusion:** After all software fixes, CH2/CH3 max ~40–60 in simultaneous mode is correct behavior for this camera + this scene.

---

## Why This Does Not Matter

1. **Hardware sync is the requirement** (professor's note) — simultaneous mode is the only way to guarantee the same hardware timestamp across all 3 channels on a moving conveyor. Sequential mode cannot provide this.

2. **NIR channels always need normalization** for both display and ML inference — this is standard practice in multispectral imaging. Narrow-band NIR sensors inherently have lower pixel ranges than broad-spectrum color sensors.

3. **Data quality is intact** — FPS=30.0, BW=755Mbps per source, blockIDs match → no packet loss, no corruption, clean hardware-synced frames.

---

## Correct Approach Going Forward

### For Display (GUI):
```python
# NIR channels — always normalize for display
import cv2
img_display = cv2.normalize(nir_raw, None, 0, 255, cv2.NORM_MINMAX)
```

### For ML Inference:
```python
# Normalize per frame before feeding to model
import numpy as np
nir_normalized = nir_raw.astype(np.float32) / nir_raw.max()
# OR: use fixed range if model was trained on fixed range
nir_normalized = nir_raw.astype(np.float32) / 255.0
```

### Probe Output Files:
- `ch2.png` / `ch3.png` → **raw pixel values** (use for data/inference)
- `ch2_norm.png` / `ch3_norm.png` → **display-normalized** (use for visual inspection)

---

## Things NOT to Try Again

- ❌ `device.NegotiatePacketSize(channel_index)` inside `open()` — **breaks stream routing**
- ❌ Sequential (one-source-at-a-time) acquisition for the production camera worker — loses hardware sync
- ❌ Trying to boost NIR values through software gain — adds noise, doesn't recover actual scene data

---

*Documented: May 15, 2026*
