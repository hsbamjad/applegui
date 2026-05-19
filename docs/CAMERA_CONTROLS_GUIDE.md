# Camera Controls Guide
### Exposure Time & Frame Rate — Plain English

---

## 1. Exposure Time

**What it is:**  
Think of the camera sensor like a bucket collecting light. Exposure time is how long you leave the bucket open per photo.

- **Long exposure** → more light collected → brighter image → but fast objects look blurry
- **Short exposure** → less light → darker image → fast objects look sharp (frozen)

**In our system:**  
Apples move on a conveyor. Short exposure freezes their motion for clean images.

**In the GUI (left panel → Camera card):**
- The `Exposure` spinbox shows the current value in **µs (microseconds)**
- 5,000 µs = 5 milliseconds — a typical starting point
- Click **Apply Exposure** to send the new value to the camera hardware

**Rule:** Exposure cannot exceed one full frame period.  
→ At 60 FPS, each frame lasts 16,667 µs. Exposure must stay below that.

---

## 2. Frame Rate (FPS)

**What it is:**  
FPS = Frames Per Second. How many photos the camera takes every second.

- **Higher FPS** → more photos per second → shorter max exposure → sharper motion
- **Lower FPS** → fewer photos → longer max exposure allowed → brighter image

**In the GUI:**
- The `Frame Rate` spinbox sets camera FPS (1–107 FPS)
- Click **Apply FPS** to send it to the camera hardware
- The **exposure spinbox max updates automatically** as you change FPS (before Apply)

---

## 3. The FPS → Exposure Relationship (Not Random)

```
Max Exposure (µs) = 1,000,000 ÷ FPS

  10 FPS  →  max 100,000 µs
  30 FPS  →  max  33,333 µs   ← default for 1 apple/s sorting
  60 FPS  →  max  16,667 µs
 107 FPS  →  max   9,345 µs   ← camera hardware maximum
```

When you raise FPS, max exposure drops. If your current exposure exceeds the new limit,
the camera firmware clamps it silently — the GUI spinbox then auto-updates to the real value.

---

## 4. Two Different FPS Numbers

The most confusing part. There are **two completely separate FPS values**:

| | Camera FPS | Display FPS |
|---|---|---|
| **What it is** | How fast the sensor captures | How fast the screen updates |
| **Where you see it** | Status bar (bottom) | Channel headers CH1/CH2/CH3 |
| **You control it?** | Yes — Apply FPS button | No — limited by processing speed |
| **Typical range** | 1–107 FPS | 30–90 FPS |

### Status Bar (bottom of window)
Updates every second. Shows **real hardware camera FPS** measured directly in the grab thread.
```
Cam: 107 FPS (sensor)  │  Display target: 107 FPS  │  Max exposure: 9,346 µs
```
**This is ground truth.** The camera sensor IS doing 107 FPS even if the screen shows 90.

### Channel Headers (CH1, CH2, CH3)
Show how many frames per second the **screen is actually rendering**.  
Always equal to or less than camera FPS.

---

## 5. Why Display FPS < Camera FPS

**Reason 1 — Image processing takes time (main bottleneck)**

Before each frame can appear on screen, the software does:
1. Decode raw Bayer sensor data → color image (~6ms)
2. Resize 3 channels: 2048×1536 → 640×480 (~2ms each)
3. EMA normalization for NIR channels (~2ms)

**Total ≈ 10–11ms per frame → max ~90 FPS possible on this machine.**

```
Set  30 FPS → period = 33ms, processing = 11ms → 22ms spare  → display gets 30 ✓
Set  60 FPS → period = 17ms, processing = 11ms →  6ms spare  → display gets ~58 ✓
Set  90 FPS → period = 11ms, processing = 11ms →  0ms spare  → display gets ~88 (at limit)
Set 107 FPS → period =  9ms, processing = 11ms → impossible  → display caps at ~90
```

The gap between set and shown FPS grows because you're approaching the processing ceiling.

**Reason 2 — Windows timer resolution (fixed)**  
Windows `time.sleep()` used to snap to 15.625ms ticks, locking FPS to exactly 64 or 32.
Fixed by calling `timeBeginPeriod(1)` — sets timer precision to 1ms. Now you get smooth values.

---

## 6. Does This Matter for Sorting?

**No.** The sensor IS capturing at your set FPS. At 107 FPS, each apple is photographed
with 9,345 µs exposure and frozen sharply. Display showing 90 FPS means 17 frames/sec
are dropped at the *display layer only* — the inference pipeline still gets all frames.

**Practical sweet spot: 60 FPS**
- Display: ~58 FPS (smooth, no drops)
- Exposure up to 16,667 µs (good NIR signal)
- Motion well-frozen at 1–2 apples/s conveyor

---

## 7. How It Is Coded

```
GUI (camera_panel.py)
  Spinboxes: _spn_exposure, _spn_fps
  FPS spinbox change → exposure max clamps in real-time (before Apply)
  Apply buttons → emit sig_exposure_changed / sig_fps_changed
        ↓
main_window.py
  _on_exposure_changed() → cam_worker.set_exposure()
  _on_fps_changed()      → cam_worker.set_fps()
  _on_cam_fps()          → status bar update (every 1s)
  _on_exposure_readback()→ syncs spinbox after firmware clamp
        ↓
CameraWorker (QThread) — camera_worker.py
  timeBeginPeriod(1)     → 1ms Windows timer (no more 64/32 snapping)
  set_exposure()         → delegates to CameraInterface
  set_fps()              → delegates + reads back actual exposure
  run() loop             → min_interval computed dynamically each iteration
                           so FPS changes take effect immediately
        ↓
CameraInterface → JAICamera (eBUS SDK)
  set_exposure() → GetFloat("ExposureTime").SetValue()
  set_fps()      → GetFloat("AcquisitionFrameRate").SetValue()
  get_exposure() → reads back actual value post-clamp
  _grab_loop()   → daemon thread, tracks grab_fps in real time
```

---

*Branch: `feature/camera-live-controls`*  
*Last updated: May 2026*
