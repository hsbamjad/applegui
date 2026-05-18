# Week 3 Plan — Multispectral Apple Sorting GUI
**Week of: May 26, 2026**
**Status: 🟡 In Progress — Camera Pipeline Fully Finalized ✅**

---

## Context: Where ### What Week 2 + Early Week 3 Delivered
```
Camera Pipeline (COMPLETE ✅ — 30 FPS confirmed)
─────────────────────────────────────────────────────────────
  JAI FS-3200T  →  eBUS SDK  →  JAICamera (background grab thread)
       │                              │
  3× hardware-sync                   ▼
  streams (BlockID              FrameTriplet (ch1, ch2, ch3)
   validated)                        │
                                     ▼
                              CameraWorker (QThread)
                              frame_idx dedup — true 30 FPS
                                     │
                                     ▼
                              GUI: 3-channel live display
                              EMA min+max normalized NIR, 30 FPS ✅
```

**Final camera pipeline optimizations (completed May 18):**
- Background `JAI-grab` daemon thread decouples acquisition from Qt display
- EMA tracks **both min AND max** of NIR raw values at full-res (2048×1536) before
  downsampling — subtracts the dark pedestal offset, eliminates grey glare
- `cv2.convertScaleAbs(alpha=scale, beta=offset)` replaces float32 numpy ops → faster
- `INTER_LINEAR` for all resizes (faster than INTER_AREA at these sizes)
- `frame_idx` dedup in CameraWorker ensures only genuinely new frames are counted
- Result: **stable 30 FPS** with clean, flicker-free NIR channels

### What Is Still Mock / Stub
| Component | Current State | Week 3 Goal |
|---|---|---|
| `InferenceWorker` | MockInferenceWorker — random grades | Real YOLOv8 on actual frames |
| `SorterController` | Simulation mode — log only | Real Arduino serial commands |
| `DataLogger` | Disabled / stub | Async CSV + image capture |
| Preprocessing for inference | Not yet built | Full-res, fixed-range, no EMA |

---

## Week 3 Tracks

```
TRACK A (Inference)               TRACK B (Sorter)              TRACK C (Logging)
──────────────────────────────    ──────────────────────────    ──────────────────────────
Wire real frames into             Connect SorterController       Implement DataLogger
InferenceWorker                   to real Arduino hardware        (CSV + TIFF, async)
Build inference preprocessing     Implement timing offset         Wire to InferenceWorker
Load/run YOLOv8 model             Test pneumatic actuators        Enable from GUI toggle
Display real grades in GUI        Verify sorting accuracy
```

**Priority:** A → B → C (A is highest — nothing else matters without real inference)

---

## Track A — AI Inference Integration

### Goal
Replace the `MockInferenceWorker` with a real inference pipeline that receives synchronized FrameTriplets from the camera and outputs apple grades using a YOLOv8 model.

---

### A1 — Build the Inference Preprocessing Pipeline
**Time estimate: 2–3 hrs**

> [!IMPORTANT]
> The display pipeline (already complete) and the inference pipeline are **separate paths**.
> They handle NIR normalization differently:
>
> | | Display (done ✅) | Inference (to build) |
> |---|---|---|
> | NIR normalization | EMA min+max (stable, scene-adaptive) | Fixed-range ÷50 (reproducible) |
> | Resize | 640×480 (INTER_LINEAR) | 640×640 (INTER_LINEAR, YOLOv8 input) |
> | CH1 resize timing | After demosaic | After demosaic |
> | Purpose | Human viewing | Model input |

**What the display pipeline does (current `_process_raws` in `camera_interface.py`):**
```python
# Full-res min/max captured first (preserves hot pixels)
cur_min, cur_max = float(raw.min()), float(raw.max())

# EMA smooths the gain over ~20 frames
ema_min = 0.95*ema_min + 0.05*cur_min
ema_max = 0.95*ema_max + 0.05*cur_max

# Fast C-level scale+offset in one pass
scale  = 255.0 / max(ema_max - ema_min, 1.0)
offset = -ema_min * scale
ch = cv2.convertScaleAbs(raw_640x480, alpha=scale, beta=offset)
```
This is for display only. The model must NOT receive EMA-normalized frames.

**What the inference path needs (`core/inference/preprocessing.py`):**
```python
# core/inference/preprocessing.py

import numpy as np
import cv2

INFER_W = 640   # YOLOv8 standard input width
INFER_H = 640   # YOLOv8 standard input height

# NIR physical max in simultaneous mode (confirmed from probe output)
# Source1: max=57  Source2: max=43 — use 50 as conservative clip point
NIR_PHYSICAL_MAX = 50.0

def preprocess_for_inference(triplet: FrameTriplet) -> np.ndarray:
    """
    Convert raw FrameTriplet to model input tensor.
    Uses FIXED normalization — reproducible across frames and sessions.
    NO EMA, NO scene-adaptive gain.

    Returns:
        np.ndarray shape (1, 5, INFER_H, INFER_W) float32
        Channels: [R, G, B, NIR1_norm, NIR2_norm]
    """
    # CH1: already demosaiced BGR in JAICamera — resize to model input
    ch1 = cv2.resize(triplet.ch1, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
    ch1 = ch1.astype(np.float32) / 255.0   # [0, 1]

    # CH2/CH3: fixed physical range normalization
    # Same formula every frame — identical apples always get identical values
    ch2 = np.clip(triplet.ch2.astype(np.float32) / NIR_PHYSICAL_MAX, 0.0, 1.0)
    ch3 = np.clip(triplet.ch3.astype(np.float32) / NIR_PHYSICAL_MAX, 0.0, 1.0)
    ch2 = cv2.resize(ch2, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
    ch3 = cv2.resize(ch3, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)

    # Stack into 5-channel tensor: [R, G, B, NIR1, NIR2]
    r, g, b = ch1[:, :, 2], ch1[:, :, 1], ch1[:, :, 0]
    tensor = np.stack([r, g, b, ch2, ch3], axis=0)  # (5, H, W)
    return tensor[np.newaxis, ...]                    # (1, 5, H, W)
```

> [!NOTE]
> `NIR_PHYSICAL_MAX = 50.0` confirmed from probe output:
> Source1: min=7 max=57 mean=9.7 and Source2: min=7 max=43 mean=8.0
> If the model was trained with per-frame NORM_MINMAX instead, use that — confirm with Dr. Lu.

**Deliverable:** `core/inference/preprocessing.py` with unit test confirming output shape and value range.

---

### A2 — Build ModelManager
**Time estimate: 2 hrs**

```python
# core/inference/model_manager.py

from pathlib import Path
import logging
from ultralytics import YOLO

log = logging.getLogger(__name__)

class ModelManager:
    """Loads and manages YOLOv8 models from the models/ folder."""

    def __init__(self, models_dir: str = "models/"):
        self._models_dir = Path(models_dir)
        self._model = None
        self._model_name = None

    def available_models(self) -> list[str]:
        return [p.name for p in self._models_dir.glob("*.pt")]

    def load(self, model_name: str) -> bool:
        path = self._models_dir / model_name
        if not path.exists():
            log.error("Model not found: %s", path)
            return False
        try:
            self._model = YOLO(str(path))
            self._model_name = model_name
            log.info("Loaded model: %s", model_name)
            return True
        except Exception as e:
            log.error("Failed to load model %s: %s", model_name, e)
            return False

    def predict(self, tensor: np.ndarray) -> tuple[str, float]:
        """
        Run inference on preprocessed tensor.
        Returns: (grade_label, confidence) e.g. ("Fresh", 0.94)
        """
        if self._model is None:
            return "No Model", 0.0
        results = self._model(tensor)
        # Parse results based on model type (classifier vs detector)
        # Adapt after confirming model format with Dr. Lu
        ...
        return grade, confidence
```

**Key questions to confirm with Dr. Lu before writing predict():**
- Is the model a **classifier** (image → class) or **detector** (image → bounding boxes)?
- Input channels: RGB only, or 5-channel (RGB + NIR1 + NIR2)?
- Grade classes: `Fresh` / `Processing` / `Cull` or different labels?

**Deliverable:** `core/inference/model_manager.py` with `available_models()`, `load()`, `predict()`.

---

### A3 — Replace MockInferenceWorker with Real InferenceWorker
**Time estimate: 2 hrs**

```python
# gui/workers/inference_worker.py (replace mock version)

class InferenceWorker(QThread):
    sig_grade   = pyqtSignal(str, float, int)   # grade, confidence, lane
    sig_status  = pyqtSignal(str, bool)

    def __init__(self, model_manager: ModelManager, config: dict):
        super().__init__()
        self._mm      = model_manager
        self._cfg     = config
        self._queue   = Queue(maxsize=2)   # small — drop stale frames
        self._running = False

    def submit_frame(self, triplet: FrameTriplet, lane: int) -> None:
        try:
            self._queue.put_nowait((triplet, lane))
        except Full:
            pass   # inference can't keep up — drop frame

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                triplet, lane = self._queue.get(timeout=0.1)
            except Empty:
                continue
            tensor = preprocess_for_inference(triplet)
            grade, conf = self._mm.predict(tensor)
            self.sig_grade.emit(grade, conf, lane)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
```

**Deliverable:** Real InferenceWorker wired to CameraWorker, grades displayed in GUI.

---

### A4 — Wire Grade Output to GUI + Downstream Workers
**Time estimate: 1 hr**

```python
# In main_window.py
self._infer_w.sig_grade.connect(self._on_grade)

def _on_grade(self, grade: str, confidence: float, lane: int) -> None:
    self._grade_summary.increment(grade)
    self._recent_list.add_result(grade, confidence, lane)
    self._sorter_w.submit_grade(grade, lane)      # Track B
    self._logger_w.log_result(grade, confidence, lane)  # Track C
```

**Deliverable:** Real grades flowing to UI, sorter, and logger simultaneously.

---

### A5 — Model Hot-Swap UI
**Time estimate: 1 hr**

Wire the existing `Load Model` button in the left panel:

```python
self._model_combo.currentTextChanged.connect(self._on_model_selected)

def _on_model_selected(self, model_name: str) -> None:
    ok = self._model_manager.load(model_name)
    self._load_status.setText("Loaded" if ok else "Failed")
```

**Deliverable:** User can switch model files from the GUI without restarting.

---

## Track B — Sorter Hardware Integration

### Goal
Move `SorterController` from simulation mode to real Arduino serial commands with timed actuation based on conveyor speed and camera-to-gate distance.

---

### B1 — Confirm Hardware Interface (Meeting with Dr. Lu)
**Time estimate: 30 min**

Before any code, confirm:

| Question | Why Needed |
|---|---|
| Which COM port is the Arduino on? | config.yaml sorter.serial.port |
| Arduino command protocol? | What byte/string triggers each lane's actuator? |
| Actuator dwell time? | How long to hold the signal (ms)? |
| Camera-to-gate distance? | Timing offset calculation |
| Conveyor speed (m/s)? | Timing offset calculation |

**Timing offset formula:**
```
delay_ms = (camera_to_gate_m / conveyor_speed_m_s) × 1000
```
Example: gate 0.5m away, conveyor at 0.3 m/s → wait 1,667ms after grading.

---

### B2 — Build SorterWorker with Timing Logic
**Time estimate: 2–3 hrs**

```python
# gui/workers/sorter_worker.py

class SorterWorker(QThread):
    sig_sort_fired = pyqtSignal(str, int)   # grade, lane

    def __init__(self, controller, config: dict):
        super().__init__()
        self._ctrl    = controller
        self._cfg     = config
        self._queue   = Queue()
        self._running = False

    def submit_grade(self, grade: str, lane: int) -> None:
        speed   = self._cfg["conveyor"]["speed_m_s"]
        dist    = self._cfg["conveyor"]["camera_to_gate_m"]
        delay   = dist / max(speed, 0.01)
        fire_at = time.time() + delay
        self._queue.put((grade, lane, fire_at))

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                grade, lane, fire_at = self._queue.get(timeout=0.05)
            except Exception:
                continue
            wait = fire_at - time.time()
            if wait > 0:
                time.sleep(wait)
            self._ctrl.sort(grade, lane)
            self.sig_sort_fired.emit(grade, lane)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
```

**Deliverable:** `SorterWorker` with timed dispatch queue committed.

---

### B3 — Enable Real Serial Mode in SorterController
**Time estimate: 1–2 hrs**

```python
# core/control/sorter_controller.py (real serial)
import serial

def connect(self) -> bool:
    if self._mode == "simulation":
        log.info("Sorter: simulation mode")
        return True
    try:
        self._serial = serial.Serial(
            port=self._cfg["port"],
            baudrate=self._cfg["baudrate"],
            timeout=self._cfg["timeout_s"]
        )
        log.info("Sorter: connected to %s", self._cfg["port"])
        return True
    except serial.SerialException as e:
        log.error("Sorter serial failed: %s", e)
        return False

def sort(self, grade: str, lane: int) -> None:
    outlet = self._grade_outlet_map.get(grade, "C")
    cmd = f"{lane}{outlet}\n"   # e.g. "1A\n" = Lane 1, Fresh
    if self._mode == "serial" and self._serial:
        self._serial.write(cmd.encode())
    log.info("SORT: lane=%d grade=%s outlet=%s", lane, grade, outlet)
```

**Deliverable:** `SorterController` working in both `simulation` and `serial` modes.

---

### B4 — Physical Actuator Test (Shuttle PC)
**Time estimate: 1 hr**

```bash
python scripts/test_sorter.py --lane 1 --grade Fresh
python scripts/test_sorter.py --lane 2 --grade Processing
python scripts/test_sorter.py --lane 3 --grade Cull
```

**Deliverable:** All 3 actuators firing correctly. `scripts/test_sorter.py` committed.

---

## Track C — Data Logging

### Goal
Async CSV grading log and optional TIFF image capture. Must never block camera or inference threads.

---

### C1 — Async CSV Logger
**Time estimate: 2 hrs**

```python
# core/logging/data_logger.py

class DataLogger:
    def __init__(self, output_dir: str = "data/logs/"):
        self._dir   = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._queue = Queue()
        self._running = False

    def start(self, session_name: str) -> None:
        self._csv_path = self._dir / f"{session_name}.csv"
        self._running = True
        threading.Thread(target=self._write_loop, daemon=True).start()

    def log_result(self, grade: str, confidence: float, lane: int,
                   frame_idx: int, timestamp: float) -> None:
        """Non-blocking — queued, background thread writes."""
        self._queue.put({
            "timestamp": timestamp,
            "frame_idx": frame_idx,
            "lane": lane,
            "grade": grade,
            "confidence": round(confidence, 4),
        })

    def _write_loop(self) -> None:
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp","frame_idx","lane","grade","confidence"
            ])
            writer.writeheader()
            while self._running or not self._queue.empty():
                try:
                    row = self._queue.get(timeout=0.1)
                    writer.writerow(row)
                    f.flush()
                except Empty:
                    continue
```

**Deliverable:** `core/logging/data_logger.py` committed with async write.

---

### C2 — Wire Logger to GUI Toggle
**Time estimate: 30 min**

The `Enable Logging` checkbox already exists. Wire it:
```python
self._enable_logging_cb.toggled.connect(self._on_logging_toggled)

def _on_logging_toggled(self, enabled: bool) -> None:
    if enabled:
        session = time.strftime("session_%Y%m%d_%H%M%S")
        self._logger.start(session)
    else:
        self._logger.stop()
```

**Deliverable:** Logging starts/stops from GUI. CSV saved to `data/logs/`.

---

### C3 — TIFF Image Capture on Defect (Stretch Goal)
**Time estimate: 1–2 hrs**

Save full-resolution 5-channel TIFF when grade = "Cull":
```python
def log_image(self, triplet: FrameTriplet, grade: str, frame_idx: int) -> None:
    if grade == "Cull":
        import tifffile
        stack = np.stack([
            triplet.ch1[:,:,2],   # R
            triplet.ch1[:,:,1],   # G
            triplet.ch1[:,:,0],   # B
            triplet.ch2,           # NIR1
            triplet.ch3            # NIR2
        ], axis=0)
        tifffile.imwrite(str(img_dir / f"frame_{frame_idx:06d}_cull.tiff"), stack)
```

> [!NOTE]
> Full-res TIFF = 15.7 MB per image. Only save on defects, never every frame.

---

## End-of-Week Deliverables

### Track A Checklist
- [ ] `core/inference/preprocessing.py` — 5-channel tensor builder, unit tested
- [ ] `core/inference/model_manager.py` — load/predict interface
- [ ] `gui/workers/inference_worker.py` — real inference on live frames
- [ ] Model dropdown loads actual `.pt` files and hot-swaps
- [ ] Real grade labels + confidence visible in GUI

### Track B Checklist
- [ ] `gui/workers/sorter_worker.py` — timed dispatch queue
- [ ] `SorterController` real serial mode on Shuttle PC
- [ ] `scripts/test_sorter.py` committed
- [ ] Physical actuator test passed (all 3 lanes)
- [ ] Timing offset configured in `config.yaml`

### Track C Checklist
- [ ] `core/logging/data_logger.py` async CSV logger
- [ ] `Enable Logging` checkbox functional
- [ ] CSV: timestamp, frame_idx, lane, grade, confidence
- [ ] TIFF capture on Cull (stretch goal)

---

## Week 3 → Week 4 Gate

**Do NOT start Week 4 (system validation + demo prep) until:**
1. Real model running on live camera frames, correct grades showing ✅
2. Sorter actuators firing on timed commands ✅
3. Data logger writing CSV per session ✅
4. End-to-end: Camera → Inference → Sort → Log running simultaneously ✅

**Week 4 covers:**
- Full end-to-end system test at conveyor speed
- Latency measurement (target: camera → sort command < 150ms)
- 1-hour endurance test in lab
- GUI polish + ASABE demo preparation
- README and final documentation

---

## Open Questions Needed from Dr. Lu

| # | Question | Blocks |
|---|---|---|
| 1 | Exact grade classes the model outputs? (`Fresh`/`Processing`/`Cull`?) | A2, A3 |
| 2 | Model type: classifier or detector? (YOLO bbox vs classification head) | A2 |
| 3 | Input channels: RGB-only or 5-channel (RGB + NIR1 + NIR2)? | A1 |
| 4 | Where is the model file (`.pt`)? Google Drive link? | A2, A3 |
| 5 | Arduino COM port on Shuttle PC? | B3 |
| 6 | Arduino command protocol? (byte per lane? ASCII? JSON?) | B3 |
| 7 | Camera-to-gate distance on physical setup (meters)? | B2 |
| 8 | Conveyor speed at test conditions (m/s)? | B2 |

---

## Risk Registry

| Risk | Likelihood | Mitigation |
|---|---|---|
| Model file not yet trained | Medium | Use placeholder ResNet classifier on RGB to test pipeline |
| Model expects RGB-only (not 5-ch) | Medium | Build adapter to pass ch1 only; NIR comes later |
| Arduino COM port unknown | Low | Use `serial.tools.list_ports` script to discover |
| Inference > 100ms on CPU | Medium | Export to ONNX; run on every 3rd frame |
| Physical sorter not accessible | Low | Keep Track B in simulation; gate to Week 4 |

---

*Update Status at top when tasks complete. Created: 2026-05-18*
