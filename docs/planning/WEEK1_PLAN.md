# Week 1 Plan - Multispectral Apple Sorting GUI
**Week of: May 14, 2026**
**Status: In Progress**

---

## Overview

This is the project kickoff week. We run **two parallel tracks** that are completely independent of each other. Neither track blocks the other. Both must complete before Week 2 begins.

```
TRACK A (Laptop)                        TRACK B (Shuttle PC)
────────────────────────────────        ────────────────────────────────
Conda env + project structure           Hardware verification
Git initialization                      JAI eBUS SDK installation
Qt main window skeleton                 Camera live test (eBUS Player)
First working app launch                Phase 1 sandbox script
Push to GitHub                          Capture + share real frames
```

**End-of-week goal:**
> A running Qt application on the laptop displaying 3 placeholder image panels, AND confirmed working camera output from the shuttle with real sample frames in hand.

---

## Strategy Rationale

We are NOT doing mock-first blindly. We run hardware discovery (Track B) in parallel with early GUI development (Track A) so that when we build the `MockCamera`, it exactly mirrors the real camera's output format, resolution, and bit depth. This prevents rework later.

See `WORKFLOW.md §2` and `PROJECT_BRAINSTORM.md §9` for full reasoning.

---

## Track A - Laptop Development

### Goal
Have a running PyQt6 application with correct layout skeleton, connected to Git.

### A1 - Create `applegui` Conda Environment
**Time estimate: 20-30 min**

```bash
# Create the environment
conda create -n applegui python=3.10 -y

# Activate it
conda activate applegui

# Install core GUI packages first (enough to start Track A work)
conda install -c conda-forge pyqt pyqtgraph numpy pyyaml -y

# Install AI / vision stack
conda install -c pytorch -c conda-forge pytorch torchvision pytorch-cuda=11.8 opencv -y

# Install dev tools
conda install -c conda-forge pytest black flake8 isort -y

# Install camera library (pip only)
pip install harvesters

# Verify
python -c "import PyQt6, torch, cv2, yaml, numpy; print('All core packages OK')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Deliverable:** `(applegui)` appears in terminal prompt. CUDA shows True (RTX 4050).

---

### A2 - Create `environment.yml`
**Time estimate: 10 min**

Export the working environment so it's reproducible everywhere:
```bash
conda env export --from-history > environment.yml
```

Then manually clean it up (remove OS-specific build strings) - see `WORKFLOW.md §5.3` for the target format.

**Deliverable:** `environment.yml` committed to repo root.

---

### A3 - Initialize Project Structure + Git
**Time estimate: 20-30 min**

Create all folders and skeleton files:
```
apple_gui/
├── main.py
├── environment.yml
├── .gitignore
├── README.md
├── CHANGELOG.md
├── WORKFLOW.md           ← copy from docs
├── PROJECT_BRAINSTORM.md ← copy from docs
├── WEEK1_PLAN.md         ← copy from docs
├── config/
│   ├── config.yaml
│   ├── config_laptop.yaml
│   └── config_shuttle.yaml
├── core/
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   ├── base_camera.py     ← stub
│   │   └── mock_camera.py     ← stub (filled after Track B)
│   ├── inference/
│   │   └── __init__.py
│   ├── control/
│   │   └── __init__.py
│   └── logging/
│       └── __init__.py
├── gui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── panels/
│   │   └── __init__.py
│   └── widgets/
│       └── __init__.py
├── models/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── tests/
│   └── __init__.py
├── docs/
│   ├── camera_setup.md    ← stub
│   ├── shuttle_setup.md   ← stub
│   └── TESTING_LOG.md     ← stub
└── scripts/
    └── camera_test.py     ← stub
```

Initialize Git:
```bash
cd "s:\MSU_Research\ASABE AIM26\apple_gui"
git init
git add .
git commit -m "chore: initialize project structure with all stubs"
```

**Deliverable:** Project folder exists with all directories, Git history started.

---

### A4 - Build Qt Main Window Skeleton
**Time estimate: 2-3 hrs**

Build the visual layout of the application - **no camera, no AI, just the window structure**. This includes:

**Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Apple Sorting GUI                            [status] │ ← Header bar
├──────────────┬────────────────────────────┬─────────────┤
│              │  [CH1]  │  [CH2]  │  [CH3] │             │
│  LEFT        │  ─────  │  ─────  │  ─────  │  RIGHT      │
│  PANEL       │         Camera Feed         │  PANEL      │
│              ├────────────────────────────┤             │
│  Camera      │   Grade Distribution Chart  │  Status     │
│  Controls    ├────────────────────────────┤  Indicators │
│              │   Throughput Graph          │             │
│  Model       │                            │  Last N     │
│  Selection   ├────────────────────────────┤  Results    │
│              │   System Status Bar         │             │
│  Sorter      │                            │  Speed      │
│  Controls    │                            │  Monitor    │
└──────────────┴────────────────────────────┴─────────────┘
```

**What gets implemented this week:**
- `gui/main_window.py` - `QMainWindow` with `QSplitter` 3-column layout
- `gui/panels/camera_panel.py` - Left sidebar with placeholder buttons
- `gui/widgets/image_display.py` - 3 `QLabel` image panels (shows placeholder color blocks)
- `main.py` - Entry point, launches the window

**What is intentionally NOT implemented yet:**
- Real camera connection (waiting for Track B data)
- Threading (next week)
- Charts (next week)
- AI inference (Phase 4)

**Deliverable:** `python main.py` launches a styled Qt window with correct 3-column layout and 3 image placeholder panels.

---

### A5 - First GitHub Push
**Time estimate: 10 min**

```bash
# Create repo on GitHub (github.com → New repository → "apple_gui")
# Then:
git remote add origin https://github.com/<your-username>/apple_gui.git
git branch -M main
git push -u origin main

# Create develop branch
git checkout -b develop
git push origin develop
```

**Deliverable:** Repo live on GitHub with `main` and `develop` branches.

---

## Track B - Shuttle PC Hardware Verification

### Goal
Confirm the JAI camera works end-to-end from physical connection to Python code, and capture real sample frames to share with the laptop.

> **This entire track is done on the SHUTTLE PC in the lab, NOT the laptop.**

---

### B1 - Download & Install JAI eBUS SDK
**Time estimate: 30-45 min**

1. Go to: **https://www.jai.com/support-software/jai-software**
2. Download: **"eBUS SDK for JAI"** (Windows 64-bit, latest version)
3. Run installer with default settings
4. Reboot if prompted

**What gets installed:**
- GenTL producer: `PvGenTL.cti` (exact path depends on install dir - note it down)
- **eBUS Player** - standalone camera viewer app
- JAI SDK headers and libraries

**Deliverable:** eBUS SDK installed, `PvGenTL.cti` file path noted.

---

### B2 - Configure 10 GigE NIC
**Time estimate: 20-30 min**

Open **Device Manager** → **Network Adapters** → find the 10GE NIC → Properties:

| Setting | Value | Why |
|---|---|---|
| Jumbo Frames | 9014 bytes | Reduces packet overhead for large images |
| Energy Efficient Ethernet | Disabled | Prevents latency spikes |
| Receive Buffers | Maximum | Prevents dropped frames |
| Interrupt Moderation | Disabled | Reduces latency |

Also configure static IP on the NIC:
- IP: `192.168.1.1` (or match camera's subnet)
- Subnet: `255.255.255.0`
- Camera will auto-configure itself via GigE Vision discovery

**Deliverable:** NIC configured, camera and PC on same subnet.

---

### B3 - Camera Verification with eBUS Player
**Time estimate: 20-30 min**

1. Ensure camera is connected via Cat6A/Cat7 to the 10GE NIC
2. Ensure camera has power (DC IN/TRIG connector)
3. Open **eBUS Player** (installed with eBUS SDK)
4. Click **"Select/Connect"** → camera should appear in the list
5. Select the camera → click **"Play"**
6. Confirm all 3 channel streams are visible

**Information to record (fill in below):**

| Parameter | Value Observed |
|---|---|
| Camera detected name | ________________ |
| Pixel format CH1 (default) | ________________ |
| Pixel format CH2 (default) | ________________ |
| Pixel format CH3 (default) | ________________ |
| Resolution per channel | ________________ |
| Default FPS | ________________ |
| Max stable FPS observed | ________________ |
| Filter wavelength CH1 | ________________ |
| Filter wavelength CH2 | ________________ |
| Filter wavelength CH3 | ________________ |
| `PvGenTL.cti` full path | ________________ |

**Deliverable:** eBUS Player showing live 3-channel video. Table above filled in.

---

### B4 - Ask Advisor Key Questions

While at the lab, ask your advisor:

| # | Question |
|---|---|
| 1 | What are the filter wavelengths in the camera (if not labeled)? |
| 2 | What is the sorter hardware interface? (Serial port? NI-DAQ? Arduino?) |
| 3 | What are the apple grade categories we will output? |
| 4 | Is there a conveyor encoder for camera triggering? |
| 5 | Are pre-trained models available, or will training happen in parallel? |

**Deliverable:** Answers to at least questions 1, 2, and 3.

---

### B5 - Install Conda + Run Phase 1 Sandbox Script
**Time estimate: 45-60 min**

```bash
# If Anaconda/Miniconda not on shuttle, install Miniconda first
# Then:
conda create -n applegui python=3.10 -y
conda activate applegui
pip install harvesters opencv-python numpy

# Run the sandbox script (we write this on laptop, transfer via Git)
git clone https://github.com/<username>/apple_gui.git
cd apple_gui
git checkout develop
python scripts/camera_test.py
```

The `camera_test.py` script will:
1. Detect camera via Harvesters
2. Print all camera properties (pixel formats, resolution, etc.)
3. Capture 10 frames
4. Save each channel as PNG: `ch1_frame0.png`, `ch2_frame0.png`, `ch3_frame0.png`
5. Print achieved FPS

**Deliverable:** 9 PNG files (3 channels × 3 frames) saved. Copy these to a USB or share folder for the laptop.

---

### B6 - Update TESTING_LOG.md
After the session, log results:
```bash
# On shuttle, after testing:
git add docs/TESTING_LOG.md
git commit -m "docs(testing): add Week 1 Track B camera verification results"
git push origin develop
```

---

## End-of-Week Deliverables

### Track A Checklist
- [ ] `applegui` conda env created and working (CUDA verified)
- [ ] `environment.yml` committed to repo
- [ ] Full project folder structure initialized
- [ ] Git repo initialized, pushed to GitHub
- [ ] `main.py` launches a styled Qt window with 3-column layout
- [ ] 3 image placeholder panels visible in the window
- [ ] Code on `develop` branch

### Track B Checklist
- [ ] JAI eBUS SDK installed on shuttle
- [ ] 10 GigE NIC configured correctly
- [ ] eBUS Player shows all 3 camera channels live
- [ ] B3 information table filled in
- [ ] Advisor questions answered (at least 1, 2, 3)
- [ ] `camera_test.py` ran successfully
- [ ] Sample frame PNGs (at least 3 frames × 3 channels) captured
- [ ] Frames shared to laptop
- [ ] Results logged in `TESTING_LOG.md`

---

## Week 1 → Week 2 Gate

**Do NOT start Week 2 until:**
1. Qt window launches successfully on laptop (Done)
2. Real frame samples are in hand from shuttle (Done)
3. Filter wavelengths are confirmed (Done)
4. Sorter interface is identified (Done)

**Week 2 will begin:**
- Building `MockCamera` calibrated to real camera output
- Adding threading (`QThread` workers) to the Qt window
- Connecting mock camera frames to the image display panels
- Building the grade distribution chart (PyQtGraph)

---

*Update the Status field at the top of this document as tasks complete.*
