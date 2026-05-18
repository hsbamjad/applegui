# 📅 Week 3 Plan — YOLOv8 Multispectral Inference & Sorter Automation

**Michigan State University | ASABE AIM26**  
**Week of: May 20, 2026**  
**Status: 🔵 Initializing — Track A, B, & C Defined**

---

## 🎯 High-Level Objectives

Week 2 successfully achieved 3-channel hardware-synchronized acquisition at 30 FPS with stabilized display normalization (EMA).
Week 3 focuses on **intelligence and execution**: running multispectral deep learning models in real time and synchronizing pneumatic sorting gates with physical conveyor movement.

```
       JAI Camera (30 FPS)
              │
              ▼
   Synchronized FrameTriplet
              │
              ├───────────────────────────────┐
              ▼                               ▼
     [Track A: Inference]             [Track B: GUI Panel]
     • Fixed-Range Normalization       • Real-time Bounding Boxes
     • Custom 5-Channel YOLOv8         • Grade Statistics Updates
     • Thread-safe Bounding Boxes      • PyQtGraph live charts
              │
              ▼
    Grading Decision (Fresh / Processing / Cull)
              │
              ▼
    [Track C: Arduino Automation]
    • Millisecond Conveyor Delay Sync
    • Serial Protocol (COM3)
    • Pneumatic Actuator Solenoids
```

---

## 🛠️ Track A — YOLOv8 Multispectral Inference Pipeline

### Goal
Implement a robust `InferenceWorker(QThread)` that performs real-time multispectral object detection and grading on synchronized `FrameTriplet` streams.

---

### A1 — Define Multispectral Model Input Strategy
**Time estimate: 1.5 hrs**

Since the JAI camera captures 5 spectral bands (RGB from CH1, NIR1 from CH2, NIR2 from CH3), we must choose how to feed this data to YOLOv8:

```
                  CH1: Color BGR ────► [R, G, B]
                                          │
                  CH2: NIR1 ~800nm ──► [NIR1]
                                          │
                  CH3: NIR2 ~900nm ──► [NIR2]
```

We will build the inference worker to support **two configurable input modes** in `config.yaml`:
1. **Stacked RGB-NIR Mode (Standard YOLO):**
   * Feed `[R, G, average(NIR1, NIR2)]` or `[R, NIR1, NIR2]` directly to a standard 3-channel YOLOv8 model.
   * *Benefit:* Allows running standard pre-trained models without custom PyTorch layer modifications.
2. **True 5-Channel Mode (Custom YOLO):**
   * Stack channels into a `[5, H, W]` tensor: `[R, G, B, NIR1, NIR2]`.
   * Modify the model's first convolutional layer (`model.model[0].conv`) to accept 5 input channels instead of 3.

---

### A2 — Fixed-Range Normalization for ML
**Time estimate: 1 hr**

Display-only EMA normalization is highly variable (non-reproducible). The inference pipeline must use **strict, mathematical normalization** to prevent degrading model accuracy.

```python
# PREPROCESSING FOR INFERENCE
# 1. Keep full resolution 2048 x 1536 (no resizing!)
# 2. Demosaic Color channel
ch1_bgr = cv2.cvtColor(raw0, cv2.COLOR_BayerBG2BGR)

# 3. Fixed-range normalization for low-intensity NIR (values max around 50)
ch2_norm = (raw1.astype(np.float32) / 50.0).clip(0.0, 1.0)
ch3_norm = (raw2.astype(np.float32) / 50.0).clip(0.0, 1.0)

# 4. Stack channels to create network input
# E.g. [5, 1536, 2048] tensor
```

---

### A3 — Build the `InferenceWorker(QThread)`
**Time estimate: 3 hrs**

Create `gui/workers/inference_worker.py`. It should:
* Queue incoming raw frames from `JAICamera` to prevent frame drops.
* Run inference asynchronously on the GPU (if CUDA is available) or CPU.
* Output detected bounding boxes, class indices, confidence scores, and lane IDs.
* Emit `sig_inference_ready(results: dict)` to the main GUI.

---

## 🖥️ Track B — Real-Time GUI Visualization & Analytics

### Goal
Render bounding boxes over the live video panels and feed sorting metrics to the dashboard charts.

```
+-------------------------------------------------------------+
| [CH1 Color]              [CH2 NIR1]             [CH3 NIR2]  |
|  +-------+                +-------+              +-------+  |
|  | Apple | [Fresh 96%]    | Apple |              | Apple |  |
|  +-------+                +-------+              +-------+  |
+-------------------------------------------------------------+
```

---

### B1 — Draw Overlaid Bounding Boxes
**Time estimate: 2 hrs**

In `gui/widgets/image_display.py`, implement class-specific color overlay:
* **Fresh:** Green bounding boxes (`#10B981`)
* **Processing:** Yellow bounding boxes (`#F59E0B`)
* **Cull:** Red bounding boxes (`#EF4444`)

Ensure that labels and confidence scores are cleanly visible at 640×480 display scaling.

---

### B2 — PyQtGraph Live Analytics Integration
**Time estimate: 2 hrs**

Wire the inference results to update the main dashboard:
* **Grade Summary Panel:** Real-time counters incrementing based on classification results.
* **Throughput Chart:** A running line chart showing processed apples per minute.
* **Distribution Chart:** A dynamic bar chart showing the ratio of Fresh vs. Processing vs. Cull.

---

## 🔌 Track C — Sorter Serial Control & Conveyor Sync

### Goal
Connect the serial dispatcher to the physical sorting hardware and synchronize pneumatic solenoid timing to matching conveyor speeds.

```
       Camera Field of View
       [   Apple detected   ]  ◄── t = 0 ms
              │
              │  Conveyor movement (distance = 0.50 meters)
              ▼
         Sorting Gate
       [ Solenoid fires! ]     ◄── t = Capture time + Delay (e.g. 500 ms)
```

---

### C1 — Establish Arduino Serial Interface
**Time estimate: 1.5 hrs**

Implement `core/control/sorter_controller.py` using Python's `pyserial`:
* Read Arduino parameters from `config.yaml` (`COM3`, 9600 Baud).
* Implement safe command dispatching (sending single-byte commands like `1A`, `2B` to fire solenoids).
* Implement asynchronous writing so serial delays never block the main loop.

---

### C2 — Microsecond Sorter Actuation Timing
**Time estimate: 2 hrs**

To guarantee that the physical paddle hits the correct apple as it travels down the conveyor, we must calculate the exact millisecond delay:

$$\text{Delay (seconds)} = \frac{\text{Distance: Camera to Gate (meters)}}{\text{Conveyor Speed (m/s)}} - \text{System Latencies}$$

```python
# core/control/timing_sync.py
class SorterSynchronizer:
    def __init__(self, camera_to_gate_m: float, system_latency_s: float = 0.05):
        self.distance = camera_to_gate_m
        self.latency = system_latency_s

    def calculate_delay_ms(self, conveyor_speed_m_s: float) -> int:
        if conveyor_speed_m_s <= 0:
            return 0
        travel_time = self.distance / conveyor_speed_m_s
        net_delay = travel_time - self.latency
        return max(0, int(net_delay * 1000))
```

* Task: Implement a QTimer queue in `SorterController` to schedule physical serial writes at exactly $t + \text{delay\_ms}$.

---

## 📈 Week 3 Deliverables Checklist

### Track A (Inference Pipeline)
- [ ] Configurable input mode (3-channel stacked vs 5-channel native) in `config.yaml`
- [ ] Strict fixed-range normalization for low-intensity NIR channels
- [ ] Functional `InferenceWorker` thread with CUDA/GPU acceleration support
- [ ] Bounding box coordinates mapped correctly from 2048x1536 to 640x480 display size

### Track B (GUI Panel & Visualization)
- [ ] Colored bounding boxes overlaid on live display panels
- [ ] Grade Summary dashboard counters updating dynamically
- [ ] PyQtGraph running charts displaying sorting stats and throughput

### Track C (Arduino & Synchronization)
- [ ] Serial communication established with Arduino on `COM3`
- [ ] Safe-fallback protection (default to Outlet C/Cull on serial loss)
- [ ] Travel-time delay calculator synced to conveyor speed
- [ ] Multi-threaded timed execution of solenoid actuators

---

## ❓ Technical Questions for Dr. Lu

1. **YOLO Model Training Specification:** What input shape was the trained model designed for? Are the channels ordered `[RGB, NIR1, NIR2]` or did you train a standard 3-channel model using specific bands?
2. **Physical System Latency:** Has the actuation delay of the pneumatic solenoid and air cylinders already been calibrated? (e.g. typical values range from 30ms to 80ms).
3. **Arduino Status Code:** Does the Arduino return acknowledgement bytes (e.g., `OK`) after firing a solenoid to monitor hardware health?
