# Apple Sorting GUI

Real-time multispectral vision system for in-field apple grading and pneumatic sorting.
Michigan State University

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)](https://pypi.org/project/PyQt6/)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()

---

## What It Is

A modular GUI application integrating a **JAI FSFE-3200T-10GE multispectral camera**, a **YOLOv8m-seg AI grading pipeline**, and a **pneumatic 3-lane sorting system** into one operator-facing interface for real-time in-field apple sorting.

## System Overview

```
[JAI Camera] -> [AI Grading (YOLOv8)] -> [SorterController] -> [Arduino] -> [Solenoid Valves] -> [Paddles]
  3-channel       Fresh/Process/Cull     timed delay            serial      3 lanes x 2 sub      3 outlets
  multispectral   per apple instance     camera->gate (ms)      USB/COM     modules (A and B)     A / B / C
```

**3 Grades, 3 Outlets:**

| Grade | Outlet | Action |
|---|---|---|
| Fresh | A | Submodule A fires - paddle deflects apple |
| Processing | B | Submodule B fires - paddle deflects apple |
| Cull | C | No command - apple falls through (default/safe) |

---

## Running on a New Machine

**Requirements before you start:**

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA driver version 525 or newer (for CUDA support)
- Windows 10 or 11 (64-bit)

**Step 1 - Run the installer (once):**

Double-click `install.bat`.

This will:
- Create the `applegui` conda environment with all dependencies
- Install PyTorch with CUDA 12.8 support
- Install the JAI eBUS SDK from the included wheel file
- Create an "Apple Sorter" shortcut on your Desktop

**Step 2 - Launch the app:**

Double-click `launch.bat` or the Desktop shortcut.

The app starts in Mock Mode by default. No hardware is required to run it.

---

## Developer Setup

```bash
git clone https://github.com/hsbamjad/applegui.git
cd apple_gui
conda env create -f environment.yml
conda activate applegui
python main.py
```

Note: `environment.yml` does not include PyTorch. After creating the env, install it manually:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Features

- Real-time 3-channel multispectral acquisition via JAI FSFE-3200T and Harvesters/GenTL
- YOLOv8m-seg inference with segmentation, tracking, and multi-view grade voting
- Timed pneumatic sorter control via Arduino and PySerial with camera-to-gate delay
- Live dashboard showing grade distribution, throughput, conveyor speed, and system status
- Apple size estimation using a ridge regression model trained on segmentation mask features
- Asynchronous CSV and TIFF logging per apple instance
- Mock mode for full pipeline simulation without any hardware

---

## Hardware

| Component | Specification |
|---|---|
| Camera | JAI FSFE-3200T-10GE (3-sensor, 5-band, 10 GigE) |
| NIC | 10 Gigabit Ethernet (Intel X550-T2 or equivalent) |
| GPU | NVIDIA RTX series (RTX 5060 Ti on Shuttle PC) |
| RAM | 32 GB recommended |
| Conveyor | 3-lane screw conveyor, 1-3 apples per second per lane |
| Sorter | 3 pneumatic units (solenoid valve + air cylinder + paddle) |
| Controller | Arduino per lane, USB/Serial to PC |
| Air supply | Compressed air for pneumatic cylinders |
| OS | Windows 10/11 |

## Sorter Wiring

```
PC (COM3) --USB--> Arduino --> Solenoid Valve A --> Air Cylinder A --> Paddle A --> Outlet A (Fresh)
                           --> Solenoid Valve B --> Air Cylinder B --> Paddle B --> Outlet B (Processing)
                                                                       (default) --> Outlet C (Cull)
```

Arduino command format: `<lane><submodule>\n` -- e.g. `1A\n` = Lane 1, Fresh.

---

## Configuration

Edit `config/config.yaml` to switch between mock and real hardware:

```yaml
camera:
  mode: "mock"        # "mock" | "jai"

sorter:
  mode: "simulation"  # "simulation" | "serial"
  serial:
    port: "COM3"      # Arduino COM port

conveyor:
  camera_to_gate_m: 0.5      # Physical distance from camera to sorter gate (meters)
  speed_apples_per_sec: 1    # 1 | 2 | 3 apples per second per lane
```
