# Apple Sorting GUI

> Real-time multispectral vision system for in-field apple grading and pneumatic sorting — Michigan State University | ASABE AIM26 | 2026

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)](https://pypi.org/project/PyQt6/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()

---

## What It Is

A modular GUI application integrating a **JAI FSFE-3200T-10GE multispectral camera**, a **YOLOv8m-seg AI grading pipeline**, and a **pneumatic 3-lane sorting system** — all into one operator-facing interface for real-time in-field apple sorting.

## System Overview

```
[JAI Camera] → [AI Grading (YOLOv8)] → [SorterController] → [Arduino] → [Solenoid Valves] → [Paddles]
  3-channel       Fresh/Process/Cull    timed delay             serial     3 lanes × 2 sub      3 outlets
  multispectral   per apple instance    camera→gate (ms)        USB/COM    modules (A & B)       A / B / C
```

**3 Grades → 3 Outlets:**

| Grade | Outlet | Action |
|---|---|---|
| Fresh | A | Submodule A fires — paddle deflects apple |
| Processing | B | Submodule B fires — paddle deflects apple |
| Cull | C | No command — apple falls through (default/safe) |

## Deploying to a New Machine

Two steps, then double-click every time after:

**Step 1 — Run once (first-time setup):**
```powershell
# Right-click install.ps1 → "Run with PowerShell"
.\install.ps1
```
This creates the `applegui` conda environment, installs the eBUS SDK if present,
and puts an **Apple Sorter** shortcut on the Desktop.

**Step 2 — Run the app:**
```
Double-click  launch.bat  (or the Desktop shortcut)
```

> Requires: Miniconda/Anaconda + NVIDIA driver ≥ 525.x on the target machine.
> The JAI eBUS SDK is only needed for live camera mode — the app runs in mock mode without it.

---

## Developer Quick Start

```bash
git clone https://github.com/<username>/apple_gui.git
cd apple_gui
conda env create -f environment.yml
conda activate applegui
python main.py
```

Runs in **Mock Mode** by default (no hardware needed). Set `config/config.yaml → camera.mode: "jai"` and `sorter.mode: "serial"` for real hardware.

## Features

- Real-time 3-channel multispectral acquisition — JAI FSFE-3200T via Harvesters/GenTL
- YOLOv8m-seg inference — segmentation, tracking, multi-view weighted grade voting
- Timed pneumatic sorter control — Arduino via PySerial, `camera_to_gate` delay
- Live dashboard — grade distribution, throughput, conveyor speed, system status
- Asynchronous CSV + TIFF logging per apple instance
- Mock mode — full pipeline simulation without any hardware

## Hardware Requirements

| Component | Specification |
|---|---|
| **Camera** | JAI FSFE-3200T-10GE (3-sensor, 5-band, 10 GigE) |
| **NIC** | 10 Gigabit Ethernet (Intel X550-T2 or equivalent) |
| **GPU** | NVIDIA CUDA (RTX 5060 Ti on Shuttle PC) |
| **RAM** | 32 GB recommended |
| **Conveyor** | 3-lane screw conveyor, 1–3 apples/s/lane |
| **Sorter** | 3× pneumatic units (solenoid valve + air cylinder + paddle) |
| **Controller** | Arduino board per lane, USB/Serial to PC |
| **Air supply** | Compressed air for pneumatic cylinders |
| **OS** | Windows 10/11 (Shuttle PC) |

## Sorter Wiring Summary

```
PC (COM3) ──USB──► Arduino ──► Solenoid Valve A ──► Air Cylinder A ──► Paddle A ──► Outlet A (Fresh)
                           └──► Solenoid Valve B ──► Air Cylinder B ──► Paddle B ──► Outlet B (Processing)
                                                                          (default) ──► Outlet C (Cull)
```

Arduino command format: `"<lane><submodule>\n"` — e.g. `"1A\n"` = Lane 1, Fresh.

## Configuration

Edit `config/config.yaml`:

```yaml
camera:
  mode: "mock"      # "mock" | "jai"

sorter:
  mode: "simulation"   # "simulation" | "serial"
  serial:
    port: "COM3"        # Arduino COM port on Shuttle PC

conveyor:
  camera_to_gate_m: 0.5       # Measure physically
  speed_apples_per_sec: 1     # 1 | 2 | 3 tested
```

## License

MIT License — see [LICENSE](LICENSE)
