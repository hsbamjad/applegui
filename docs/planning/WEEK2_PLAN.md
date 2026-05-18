# 📅 Week 2 Plan — Multispectral Apple Sorting GUI
**Week of: May 19, 2026**
**Status: 🟡 In Progress — Track A Complete ✅**

---

## Overview

Week 1 delivered confirmed hardware communication and 3-channel PNG capture. Week 2 has two tracks:

```
TRACK A (Shuttle PC)                    TRACK B (Laptop)
────────────────────────────────        ────────────────────────────────
Channel synchronization                 CameraWorker QThread
MultiSource stream investigation        Wire camera into live GUI display
BlockID validation                      3-channel live video in window
Sync validation test                    Start/Stop camera controls
```

> ⚠️ **Professor's Note:** "Pay more attention to channel synchronization; initially, they all encountered minor synchronization issues."
> **Track A is the highest priority this week.**

**End-of-week goal:**
> Live 3-channel synchronized video streaming inside the Qt GUI at ≥ 30 FPS, with confirmed per-frame channel alignment.

---

## Track A — Channel Synchronization (Shuttle PC)

### Goal
Confirm that CH1, CH2, and CH3 frames are captured at the **exact same timestamp** (hardware-synchronized), and that our Python retrieval code correctly assembles matched triplets.

---

### A1 — Read the MultiSource Sample
**Time estimate: 30 min**

On the Shuttle PC, open and read the official JAI MultiSource sample:

```
C:\Users\MASTER\.conda\envs\applegui\Lib\site-packages\ebus-python\samples\MultiSource.py
```

Run it:
```bash
conda activate applegui
cd "C:\Users\MASTER\.conda\envs\applegui\Lib\site-packages\ebus-python\samples"
python MultiSource.py
```

**What to look for:**
- How it opens multiple streams simultaneously (one per source)
- How it handles `GetBlockID()` for frame matching
- Whether it uses `PvPipeline` or raw `stream.RetrieveBuffer()`

**Deliverable:** Notes on the MultiSource pattern copied to `docs/camera_setup.md`.

---

### A2 — Investigate Simultaneous Multi-Stream Opening
**Time estimate: 1–2 hrs**

Current probe: opens **1 stream**, switches `SourceSelector` (sequential — NOT synchronized).
Target: open **3 streams** simultaneously (one per source), retrieve one buffer from each per frame cycle.

```python
# Pattern to investigate
result, stream0 = eb.PvStream.CreateAndOpen(conn_id)  # Source0
result, stream1 = eb.PvStream.CreateAndOpen(conn_id)  # Source1  
result, stream2 = eb.PvStream.CreateAndOpen(conn_id)  # Source2

# Configure all 3 stream channels on the device
# (channel index maps to source order)
```

Check if `PvStream.CreateAndOpen` accepts a channel index parameter, or if the SDK handles source routing automatically.

**Deliverable:** Working or documented attempt at 3-stream simultaneous open.

---

### A3 — Implement BlockID Frame Matching
**Time estimate: 1–2 hrs**

Each buffer from the camera carries a `GetBlockID()` value. Hardware-synchronized frames from all 3 sensors will share the **same BlockID**. This is the ground truth for synchronization.

```python
# Retrieve from all 3 streams
result0, buf0, op0 = stream0.RetrieveBuffer(timeout_ms)
result1, buf1, op1 = stream1.RetrieveBuffer(timeout_ms)
result2, buf2, op2 = stream2.RetrieveBuffer(timeout_ms)

# Validate synchronization
b0 = buf0.GetBlockID()
b1 = buf1.GetBlockID()
b2 = buf2.GetBlockID()

if b0 == b1 == b2:
    # ✅ Hardware-synchronized triplet — safe to process
    process_triplet(buf0, buf1, buf2)
else:
    # ⚠️ Mismatch — drop and log the offset
    print(f"  SYNC MISMATCH: CH1={b0} CH2={b1} CH3={b2}")
    # Discard all 3 and requeue
```

**Deliverable:** `sync_mismatch_count` logged per session. Target: 0 mismatches at steady state.

---

### A4 — Synchronization Validation Test
**Time estimate: 30 min**

Place a **fast-moving object** (hand waving, pendulum, pointer) in front of the camera while acquiring a synchronized triplet.

```python
# After capturing synchronized triplet:
import cv2, numpy as np

ch1_bgr = cv2.cvtColor(buf0_data, cv2.COLOR_BayerBG2BGR)
ch2     = buf1_data
ch3     = buf2_data

# Create side-by-side comparison image
comparison = np.hstack([ch1_bgr, cv2.cvtColor(ch2, cv2.COLOR_GRAY2BGR),
                                  cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR)])
cv2.imwrite("scripts/probe_output/sync_test.png", comparison)
```

**Pass criteria:** Moving object appears at **identical pixel coordinates** in all 3 channels.  
**Fail criteria:** Object is at different x/y positions across channels → software sync still broken.

**Deliverable:** `sync_test.png` with all 3 channels aligned. Share with Dr. Lu.

---

### A5 — Update `config.yaml` with Confirmed Values
**Time estimate: 20 min**

```yaml
camera:
  jai:
    ip: "169.254.133.151"
    mac: "00:0c:df:0a:b8:e9"
    serial: "U320484"
    firmware: "1.1.0.2"
    fps: 30
    packet_size: 8976
    sources:
      Source0:
        role: "color"
        pixel_format: "BayerRG8"
        debayer: "BayerBG2BGR"
        resolution: [2048, 1536]
        exposure_us: 9229
        gain: 1.0
      Source1:
        role: "nir1"
        pixel_format: "Mono8"
        resolution: [2048, 1536]
        exposure_us: 9229
        gain: 1.0
      Source2:
        role: "nir2"
        pixel_format: "Mono8"
        resolution: [2048, 1536]
        exposure_us: 9229
        gain: 1.0
```

**Deliverable:** `config.yaml` committed with all confirmed hardware values.

---

## Track B — Live Camera GUI Integration (Laptop)

### Goal
Build `CameraWorker(QThread)` using the confirmed eBUS connection pattern and display live 3-channel video in the Qt window.

---

### B1 — Build `CameraWorker(QThread)`
**Time estimate: 2–3 hrs**

Create `gui/workers/camera_worker.py`. Port the confirmed connection logic from `camera_probe_jai.py`:

```python
class CameraWorker(QThread):
    # Signals
    frame_ready  = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # ch1, ch2, ch3
    status_msg   = pyqtSignal(str)
    error        = pyqtSignal(str)

    def run(self):
        # 1. Connect  (same pattern as probe)
        result, device = eb.PvDevice.CreateAndConnect(self.conn_id)
        # 2. Open stream + GEV setup
        result, stream = eb.PvStream.CreateAndOpen(self.conn_id)
        if isinstance(device, eb.PvDeviceGEV):
            device.NegotiatePacketSize()
            device.SetStreamDestination(stream.GetLocalIPAddress(),
                                        stream.GetLocalPort())
        # 3. Allocate buffers + start acquisition
        # 4. Loop: retrieve → emit frame_ready signal → requeue
        # 5. On stop: AcquisitionStop → StreamDisable → cleanup
```

**Key rules:**
- All eBUS calls happen **inside the worker thread**, never in the main thread
- Frames emitted as numpy arrays via Qt signal (thread-safe)
- Worker must respond to a `stop()` call cleanly

**Deliverable:** `camera_worker.py` with full start/stop lifecycle.

---

### B2 — Connect Worker to Image Display
**Time estimate: 1 hr**

In `gui/main_window.py` or `gui/panels/camera_panel.py`:

```python
self.cam_worker = CameraWorker(conn_id="169.254.133.151")
self.cam_worker.frame_ready.connect(self.on_frame)

def on_frame(self, ch1, ch2, ch3):
    self.img_panel.set_frame(ch1, ch2, ch3)
```

In `gui/widgets/image_display.py`:
```python
def set_frame(self, ch1, ch2, ch3):
    # Convert numpy → QPixmap → display in QLabel
    self.label_ch1.setPixmap(numpy_to_qpixmap(ch1))
    self.label_ch2.setPixmap(numpy_to_qpixmap(ch2))
    self.label_ch3.setPixmap(numpy_to_qpixmap(ch3))
```

**Deliverable:** Live 3-channel video updating in the Qt window.

---

### B3 — Add Start / Stop Camera Controls
**Time estimate: 30 min**

Wire the existing camera panel buttons to the worker:

```python
self._connect_btn.clicked.connect(self.cam_worker.start)
self._stop_btn.clicked.connect(self.cam_worker.stop)
```

Add status indicator (green dot = streaming, red = disconnected).

**Deliverable:** Start/Stop buttons functional, status indicator updates correctly.

---

### B4 — FPS Display + Performance Test
**Time estimate: 30 min**

Add a real-time FPS counter to the GUI:

```python
# In CameraWorker: track frame timestamps
import time
self._frame_times.append(time.perf_counter())
if len(self._frame_times) >= 30:
    fps = 30 / (self._frame_times[-1] - self._frame_times[0])
    self.fps_update.emit(fps)
    self._frame_times.clear()
```

**Pass criteria:** GUI maintains ≥ 30 FPS display without freezing.

**Deliverable:** FPS counter visible in GUI, no dropped frames at 30 FPS.

---

### B5 — Commit + Push All Week 2 Work
**Time estimate: 10 min**

```bash
git add .
git commit -m "feat: CameraWorker + live 3-channel display + sync validation"
git push
```

---

## Questions for Dr. Lu This Week

| # | Question | Why needed now |
|---|---|---|
| 1 | **Filter wavelengths** (nm) for CH1, CH2, CH3? | Label channels correctly in GUI |
| 2 | **Conveyor encoder** signal available? Is it on `TriggerSource: Line4`? | Hardware sync instead of free-run |
| 3 | **Sorter interface** — Arduino serial port? Which COM port? | Begin SorterController wiring |
| 4 | **Grade categories** for apple sorting? | Start building inference pipeline |

---

## End-of-Week Deliverables

### Track A ✅ Checklist
- [x] MultiSource.py sample read and understood
- [x] 3-stream simultaneous open implemented
- [x] `NegotiatePacketSize` added before stream open (fixes dark NIR images)
- [x] BlockID matching confirmed — all 3 sources **blockID=79** ✅
- [x] Stream statistics added (FPS=30.0, BW=755Mbps per source, 22% of 10GbE)
- [x] Hardware-synchronized triplet confirmed (professor's concern resolved)
- [x] `config.yaml` values confirmed

> **Final confirmed output (May 15, 2026):**
> ```
> Source0: blockID=79  min=6  max=255  mean=19.9  (Color/BayerRG8)
> Source1: blockID=79  min=7  max=57   mean=9.7   (NIR1/Mono8)
> Source2: blockID=79  min=7  max=43   mean=8.0   (NIR2/Mono8)
> BW: 755 Mbps × 3 sources = 22% of 10GbE  ← plenty of headroom
> ```

### Track B ✅ Checklist
- [ ] `gui/workers/camera_worker.py` built with full lifecycle
- [ ] Live 3-channel video visible in Qt GUI
- [ ] Start/Stop buttons functional
- [ ] FPS counter showing ≥ 30 FPS
- [ ] All code committed and pushed

---

## Week 2 → Week 3 Gate

**Do NOT start Week 3 (AI inference) until:**
1. Channel synchronization confirmed (zero BlockID mismatches at steady state) ✅
2. Live 3-channel video running in GUI at ≥ 30 FPS ✅
3. Filter wavelengths confirmed with Dr. Lu ✅
4. Grade categories defined ✅

**Week 3 will begin:**
- YOLOv8 model integration
- `InferenceWorker(QThread)` building
- Grade display in GUI

---

*Update the Status field at the top of this document as tasks complete.*
