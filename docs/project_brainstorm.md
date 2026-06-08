# Multispectral Apple Sorting GUI - Project Brainstorm & Kickoff Guide
**Michigan State University | ASABE AIM26 | May 2026**

---

## 1. Big Picture: What Are We Building?

We are building a **modular, open-source Python GUI application** that:

1. **Connects to the JAI multispectral camera** and streams 3-channel images in real time.
2. **Runs AI/ML models** on those images to grade/sort apples (detect defects, bruises, color, internal damage).
3. **Controls sorting hardware** (actuators/diverters on the conveyor) based on grading output.
4. **Monitors and visualizes** everything (live images, statistics, conveyor speed, system health).
5. **Logs data** (images + grading results + sorting outcomes) for research reproducibility.

The GUI is the **central nervous system** tying camera → AI → physical sorting hardware into one cohesive, operator-friendly application.

---

## 2. The Camera - What It Is & How It Works

### 2.1 What You're Looking At (From Your Lab Photos)

| Photo | What's Visible |
|---|---|
| Front (lens) | **Edmund Optics 16mm f/1.6 VIS-NIR lens** - red ring = NIR-capable, C-mount |
| Side (JAI green panel) | JAI logo, heat sink on top - this is the camera body |
| Rear panel | **10GE port** (RJ45 for 10 Gigabit Ethernet), **DC IN / TRIG** connector (power + hardware trigger in), **AUX** slot |

### 2.2 Model: JAI FSFE-3200T-10GE ("Fusion Series")

This is a **3-sensor prism-based area scan camera**. Here's what that means:

```
Light → Lens (C-mount) → Dichroic Prism → splits light into 3 wavelengths
                                            ↓           ↓           ↓
                                       Sensor 1     Sensor 2     Sensor 3
                                      (e.g. RGB)   (e.g. NIR1)  (e.g. NIR2)
```

- **Dichroic prism** = a glass block with special coatings that splits light by wavelength precisely.
- All 3 sensors are **physically co-located and hardware-synchronized** - they capture the EXACT same frame at the EXACT same moment. This is the major advantage over using 3 separate cameras.
- Each sensor is a **Sony IMX252 (3.2 MP, 2048×1536)** - global shutter, Pregius CMOS chip.

### 2.3 Key Specs

| Parameter | Value |
|---|---|
| Sensors | 3 × Sony IMX252 (Pregius, global shutter) |
| Resolution | 3.2 MP per sensor (2048 × 1536) |
| Frame Rate | Up to 107 FPS |
| Spectral Range | 405-1000 nm (customizable with filter set) |
| Wavebands | 3 custom narrow bands (as narrow as 25 nm) |
| Interface | **10 GigE Vision** (auto-negotiates to 1G) |
| Bit Depth | 8 / 10 / 12-bit selectable |
| Shutter | Global (no motion artifacts on moving apples) |
| Lens Mount | C-mount (Edmund Optics VIS-NIR lens installed) |
| Sync | Hardware - all 3 sensors capture simultaneously |
| Power | Via DC IN / TRIG circular M12 connector on back |

### 2.4 Waveband Configuration (Recommended for Apple Sorting)

| Channel | Wavelength | What It Sees |
|---|---|---|
| **CH1** | ~660 nm (Red/visible) | Surface color, visual defects |
| **CH2** | ~750-800 nm (NIR) | Bruising, internal structure |
| **CH3** | ~850-900 nm (Deep NIR) | Water content, firmness |

> **Note:** The actual filter configuration needs to be verified - check the camera label or datasheet from your advisor.

### 2.5 How the Camera Connects to the PC

```
Camera rear: [10GE RJ45] ─── Cat6A/Cat7 cable ─── [10GigE NIC in PC]
Camera rear: [DC IN/TRIG] ─── Power supply (+ optional hardware trigger)
Camera rear: [AUX] ─── Optional (encoder for conveyor sync, GPIO)
```

- The PC needs a **10 Gigabit Ethernet NIC** (Intel X550-T2 is common in labs).
- Data bandwidth: 3 sensors × 3.2 MP × 107 FPS × 8-bit ≈ 1 GB/s → that's why 10 GigE is needed.

---

## 3. How Camera Talks to Software - The Software Interface Layer

### 3.1 The Protocol Stack

```
Camera Hardware (JAI)
        ↓ GigE Vision protocol (over 10 GigE Ethernet)
GenTL Producer (.cti file - installed with JAI eBUS SDK)
        ↓ GenICam / GenTL standard API
Harvesters (Python library - open source)
        ↓ NumPy arrays
Your Python GUI Code
```

### 3.2 What is Harvesters?

**Harvesters** is an open-source Python library that implements the GenICam GenTL consumer. It's the bridge between the camera driver and your Python code.

```python
from harvesters.core import Harvester
import numpy as np

h = Harvester()
# Point to JAI eBUS SDK GenTL producer (.cti file)
h.add_file('C:/Program Files/JAI/eBUS SDK/bin/PvGenTL.cti')
h.update_device_info_list()

ia = h.create_image_acquirer(0)  # 0 = first camera
ia.start_image_acquisition()

with ia.fetch_buffer() as buffer:
    # For JAI 3-sensor camera: 3 components per buffer
    ch1 = buffer.payload.components[0].data.reshape(1536, 2048)
    ch2 = buffer.payload.components[1].data.reshape(1536, 2048)
    ch3 = buffer.payload.components[2].data.reshape(1536, 2048)

    # Stack into a 3-channel multispectral image
    multispectral_image = np.stack([ch1, ch2, ch3], axis=2)
```

### 3.3 What is JAI eBUS SDK?

The **low-level driver and GenTL producer** provided by JAI (free download from jai.com). It:
- Provides the `.cti` file Harvesters needs.
- Includes **eBUS Player** - a standalone GUI app to test the camera BEFORE writing any code.
- Provides camera configuration tools (pixel format, exposure, gain, trigger settings).

> **First thing you should do with the camera:** Install eBUS SDK and use eBUS Player to verify you can see all 3 channels streaming live.

---

## 4. Software Architecture - The Full Picture

### 4.1 Technology Stack

| Component | Choice | Why |
|---|---|---|
| Language | **Python 3.10+** | Best AI/ML ecosystem, Harvesters support |
| GUI Framework | **PyQt6** (or PySide6) | Qt-based, LGPL, cross-platform, professional |
| Real-time plotting | **PyQtGraph** | GPU-accelerated, integrates natively with Qt |
| Camera interface | **Harvesters + JAI eBUS SDK** | GenICam standard, open-source |
| AI inference | **PyTorch 2.x** | Research-grade, model flexibility, GPU |
| Computer vision | **OpenCV 4.8+** | Image preprocessing, display conversion |
| Hardware control | **pyserial / NI-DAQmx** | Sorter & conveyor communication (TBD) |
| Data logging | **CSV + TIFF images** | Structured, research-reproducible output |
| Config | **YAML (PyYAML)** | Human-readable settings file |

### 4.2 Software Modules

```
apple_gui/
├── main.py                        ← Entry point, launches the app
├── config/
│   └── config.yaml                ← Camera params, model paths, thresholds
├── core/
│   ├── camera/
│   │   ├── camera_interface.py    ← MODULE 1: Camera acquisition
│   │   └── image_buffer.py        ← Thread-safe frame queue
│   ├── inference/
│   │   ├── model_manager.py       ← MODULE 2: Load/switch AI models
│   │   └── inference_engine.py    ← Run inference on frames
│   ├── control/
│   │   ├── sorter_controller.py   ← MODULE 3: Sorting actuator commands
│   │   └── conveyor_controller.py ← Conveyor speed read/control
│   └── logging/
│       └── data_logger.py         ← MODULE 4: Save images + results
├── gui/
│   ├── main_window.py             ← Main Qt Window
│   ├── panels/
│   │   ├── camera_panel.py        ← Camera controls + live view
│   │   ├── model_panel.py         ← Model selection + switching
│   │   ├── control_panel.py       ← Sorter + conveyor controls
│   │   └── stats_panel.py         ← Dashboard + visualization
│   └── widgets/
│       ├── image_display.py       ← Multi-channel image viewer
│       └── chart_widgets.py       ← Grade distribution, throughput
└── utils/
    ├── image_utils.py             ← Preprocessing helpers
    └── thread_workers.py          ← QThread workers for all background tasks
```

### 4.3 Threading Architecture (CRITICAL)

```
Main Thread (Qt Event Loop)
├── Receives signals from workers via Qt Signal/Slot (thread-safe)
├── Updates all UI elements
└── Handles user input events

Thread 1: CameraWorker (QThread)
├── Continuously grabs frames from camera via Harvesters
└── Emits Qt signal with frame triplet → InferenceWorker + DisplayWorker

Thread 2: InferenceWorker (QThread)
├── Reads frames from camera signal
├── Runs AI model (PyTorch) on GPU
└── Emits grade result signal → GUI + SorterWorker

Thread 3: SorterWorker (QThread)
├── Receives grade results
├── Calculates timing offset (conveyor travel time to sorter gate)
└── Sends timed actuator command

Thread 4: LoggerWorker (QThread)
├── Receives frames + results asynchronously
└── Writes to disk (no blocking main thread)
```

---

## 5. Module Deep Dives

### Module 1: Camera Interface
**Responsibilities:**
- Connect/disconnect from JAI camera via Harvesters.
- Configure: exposure, gain, pixel format, FPS, trigger mode.
- Acquire synchronized 3-channel frame triplets.
- Push frames into thread-safe queue.
- Handle errors gracefully (camera disconnect, buffer overflow).

**Key camera settings:**
- `PixelFormat` → `Mono8` for NIR channels, `BayerRG8` for RGB.
- `ExposureTime` → keep short to freeze motion on conveyor.
- `AcquisitionFrameRate` → match to conveyor throughput.
- `TriggerMode` → `Off` for free-run, `On` for encoder-triggered.

### Module 2: Model Manager
**Responsibilities:**
- Scan `models/` folder for available `.pt` / `.onnx` files.
- Load model into GPU memory on demand.
- Provide uniform `predict(image_triplet) → (grade, confidence)` interface.
- Hot-swap models without stopping acquisition.
- Report model metadata (name, input size, classes, accuracy).

### Module 3: Sorter Controller
**Hardware (confirmed from poster):**
- **3 sorting units**, one per conveyor lane, at end of conveyor
- Each unit has **Submodule A** (Fresh → Outlet A) + **Submodule B** (Processing → Outlet B)
- **Default (no signal)** → Outlet C (Cull) - physically safe
- Components: NIR switch (arrival sensor) + Arduino board + 2× solenoid valve + 2× air cylinder + 2× paddle
- **Communication**: PC → USB/Serial → Arduino → Solenoid valves
- **Command protocol**: ASCII string `"<lane><submodule>\n"` (e.g. `"1A\n"` = Lane 1, Fresh)

**Responsibilities:**
- Maintain timing map: apple graded at time T → reaches gate at T + `(camera_to_gate_m / speed) * 1000` ms
- Schedule `serial.write(cmd)` at the computed fire time using a background thread
- Cull apples send NO command (safe default - apple falls to Outlet C)
- Simulation mode: log commands, don't send to hardware
- Track stats: total / Fresh / Processing / Cull / missed

### Module 4: Data Logger
**Responsibilities:**
- Log grading results to CSV (timestamp, grade, confidence, speed).
- Save raw 3-channel images as TIFF (on demand or on defect detect).
- Save session summary (total apples, grade distribution, runtime).
- All writes are asynchronous (never block the main loop).

### Module 5: Visualization Dashboard
**What it shows:**
- **Live image feed:** 3 channels side by side (CH1, CH2, CH3) + optional composite.
- **Grade distribution:** Real-time bar/donut chart (Grade A%, Grade B%, Defect%).
- **Throughput graph:** Apples per minute over time (rolling window).
- **Conveyor speed gauge:** Current m/s reading.
- **Status indicators:** Camera (online/offline), AI (online/offline), Sorter (online/offline), Logger (online/offline).
- **Recent results list:** Last N apples with thumbnail + grade label.

---

## 6. Development Sequence - Phase by Phase

### Phase 0: Hardware Verification (No Code)
1. Install **JAI eBUS SDK** on lab PC → [jai.com/support-software](https://www.jai.com/support-software/jai-software).
2. Install/confirm **10 GigE NIC** in PC.
3. Configure NIC: Jumbo Frames = 9014 bytes, disable Energy Efficient Ethernet.
4. Connect camera via Cat6A cable.
5. Open **eBUS Player** → verify all 3 channels stream live.
6. Note: exact pixel formats, actual filter wavelengths, achievable FPS.

> **Goal:** "Camera works and PC can receive 3-channel images." Answer this before any code.

### Phase 1: Camera Sandbox Script (Shuttle PC - Track B)
1. `conda activate applegui` on the shuttle PC.
2. `pip install harvesters opencv-python numpy` (or via environment.yml).
3. Write minimal script: connect → grab 1 frame → save 3 PNGs → disconnect.
4. Experiment with exposure/FPS settings in code.
5. Benchmark actual achievable FPS and data throughput.
6. Copy real sample frames back to laptop for mock camera calibration.

> **Goal:** Working Python script that grabs 3-channel images reliably AND real sample frames in hand.

### Phase 2: GUI Skeleton (No Camera)
1. `pip install PyQt6 pyqtgraph`
2. Build main window layout: left sidebar + center display + right panel.
3. Add placeholder/dummy widgets.
4. Implement QThread workers with Qt Signal/Slot.
5. Test with fake camera (random NumPy arrays at 30 FPS).

> **Goal:** Running Qt window with correct layout, responsive threading proven.

### Phase 3: Camera Integration in GUI
1. Wrap Phase 1 camera code into `CameraWorker(QThread)`.
2. Connect camera signals to image display widgets.
3. Show live 3-channel images in GUI.
4. Add Start/Stop camera controls.
5. Verify GUI stays responsive at full frame rate.

> **Goal:** Live multispectral video playing in the GUI.

### Phase 4: AI Inference Integration
1. Get a pre-trained model (any PyTorch `.pt` initially - even a dummy one).
2. Implement `ModelManager` - scan folder, load model.
3. Implement `InferenceWorker` - grab frame, run inference, emit result.
4. Display grade result in GUI.
5. Add model selection dropdown + load button.

> **Goal:** GUI shows live images + grade label updating in real time.

### Phase 5: Sorter Control
1. Confirm sorter hardware interface with advisor.
2. Implement `SorterController` with timing offset logic.
3. Test timing: measure camera-to-gate distance, conveyor speed.
4. Implement simulation mode (log commands, don't send to hardware).

> **Goal:** System correctly times and executes sorting commands.

### Phase 6: Dashboard & Logging
1. Add real-time grade distribution chart.
2. Add throughput graph (rolling window).
3. Implement CSV data logger.
4. Add image capture logging.

> **Goal:** Full visualization + logging operational.

### Phase 7: Testing, Validation & Polish
1. Test at multiple conveyor speeds (0.5, 1.0, 1.5 m/s).
2. Measure actual end-to-end latency (target: <150ms).
3. Run 8-hour endurance test (target: >99% uptime).
4. Polish UI, add error handling, write README.
5. Prepare for ASABE AIM26 demo.

---

## 7. Open Questions - Status After Hardware Clarification

| # | Question | Status | Answer |
|---|---|---|---|
| 1 | Actual filter wavelengths? | Confirmed | RG ~660nm / NIR1 ~800nm / NIR2 ~900nm (5-band camera, 3 used) |
| 2 | Lab PC has 10 GigE NIC? | Confirmed | Shuttle PC: Intel Core i9-11900K + RTX 5060 Ti |
| 3 | Sorter hardware interface? | Confirmed | **Arduino via USB/Serial** - PySerial command `"<lane><submodule>\n"` |
| 4 | Pre-trained models ready? | In progress | YOLOv8m-seg trained - need `.pt` file path on Shuttle PC |
| 5 | Grade categories? | Confirmed | **Fresh** / **Processing** / **Cull** - 3 classes |
| 6 | Conveyor encoder available? | Unknown | Poster shows free-run 60 FPS; encoder-trigger optional |
| 7 | Camera-to-gate distance? | Needs measuring | Config default: 0.5m - **must be measured physically** |
| 8 | Deployment OS? | Confirmed | Windows 10/11 (Shuttle PC) |
| 9 | Arduino COM port? | Needs checking | Default config: COM3 - check Device Manager on Shuttle PC |
| 10 | Valve pulse duration? | Needs tuning | Config default: 80ms - adjust until paddle fully extends |
| 11 | Compressed air available? | Assumed | Poster shows air cylinders - lab has compressor |

---

## 8. Risk Registry

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Camera driver incompatibility with Harvesters | Medium | High | Test Phase 0 early; fall back to JAI SDK direct bindings |
| Sorter hardware interface unknown | Medium | High | Build simulation mode; use serial placeholder |
| AI inference too slow (>100ms on CPU) | Medium | High | Use GPU; export to ONNX/TensorRT for speed |
| Network bandwidth saturation at full FPS | Low | High | Run at 30-60 FPS initially; tune NIC settings |
| GUI freezing under high frame rate | Medium | Medium | Strict threading; display every 3rd frame only |
| Conference deadline pressure | High | High | Build MVP (Phases 1-4) first; polish later |

---

## 9. This Week's Plan (Two Parallel Tracks)

### Track A - Laptop (Start Today)
- [ ] **A1.** Create `applegui` conda environment.
- [ ] **A2.** Initialize project folder structure + Git repo.
- [ ] **A3.** Create `environment.yml`, `.gitignore`, `README.md`, `CHANGELOG.md`.
- [ ] **A4.** Build Qt main window skeleton (layout only, no camera code yet).
- [ ] **A5.** First commit + push to GitHub.

### Track B - Shuttle PC (Next Lab Visit, ASAP)
- [ ] **B1.** Download and install **JAI eBUS SDK** from jai.com.
- [ ] **B2.** Confirm 10 GigE NIC is installed and configured (Jumbo Frames, disable EEE).
- [ ] **B3.** Connect camera → open **eBUS Player** → confirm all 3 channels visible.
- [ ] **B4.** Note exact pixel formats, achievable FPS, channel order.
- [ ] **B5.** Confirm **filter wavelengths** with advisor.
- [ ] **B6.** Ask advisor about **sorter hardware interface** (Serial/DAQ/Arduino?).
- [ ] **B7.** `conda activate applegui` → run Phase 1 sandbox script → save real frames as PNG.
- [ ] **B8.** Copy real frame samples to laptop.

### After Both Tracks Complete
- [ ] Build `MockCamera` calibrated to real camera output (exact format, resolution, bit depth).
- [ ] Begin Phase 2: full GUI + threading on laptop with mock camera.
- [ ] Create GitHub repo if not done in A2.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **GenICam** | Generic Interface for Cameras - an industry standard for camera interoperability |
| **GenTL** | Generic Transport Layer - part of GenICam for data transport |
| **GigE Vision** | Camera-over-Ethernet protocol standard (JAI uses this) |
| **Harvesters** | Open-source Python GenTL consumer - your camera API wrapper |
| **eBUS SDK** | JAI's SDK providing the `.cti` GenTL producer file |
| **.cti file** | Compiled Transport Layer driver for your specific camera brand |
| **Dichroic prism** | Optical element splitting light into 3 spectral bands simultaneously |
| **Global shutter** | All pixels expose simultaneously - no motion blur on moving targets |
| **NIR** | Near-Infrared light (700-1000 nm) - invisible to humans, detects internal features |
| **QThread** | Qt's threading class - runs code off the main GUI thread safely |
| **Signal/Slot** | Qt's thread-safe event system - how threads communicate without shared memory |
| **PyQtGraph** | Real-time plotting library built on Qt - for live charts |
| **ONNX** | Open Neural Network Exchange - portable, fast model inference format |
| **DAQ** | Data Acquisition - hardware for reading/writing digital and analog signals |
