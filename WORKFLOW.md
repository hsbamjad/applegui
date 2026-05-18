# Project Workflow Document
## Multispectral Apple Sorting GUI
**Michigan State University | ASABE AIM26 | Version 1.0**
**Last Updated: May 2026**

---

## Table of Contents
1. [Project Identity](#1-project-identity)
2. [Two-System Development Strategy](#2-two-system-development-strategy)
3. [Repository Structure](#3-repository-structure)
4. [Git Workflow](#4-git-workflow)
5. [Environment Setup](#5-environment-setup)
6. [Coding Standards](#6-coding-standards)
7. [Mock vs Real Camera Mode](#7-mock-vs-real-camera-mode)
8. [Testing Strategy](#8-testing-strategy)
9. [Hardware Integration Protocol](#9-hardware-integration-protocol)
10. [Documentation Standards](#10-documentation-standards)
11. [Development Rhythm](#11-development-rhythm)
12. [Release & Demo Checklist](#12-release--demo-checklist)

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project Name** | Multispectral Apple Sorting GUI |
| **Institution** | Michigan State University (MSU) |
| **Conference Target** | ASABE AIM26 |
| **Camera** | JAI FSFE-3200T-10GE (3-sensor, 10 GigE) |
| **Language** | Python 3.10+ |
| **GUI Framework** | PyQt6 |
| **Package Manager** | Conda (Miniconda/Anaconda) |
| **Conda Environment** | `applegui` |
| **License** | MIT (open-source) |
| **Repository** | TBD (GitHub) |
| **Primary Dev Machine** | Laptop (code + documentation) |
| **Integration Machine** | Lab Shuttle PC (camera + hardware) |

---

## 2. Two-System Development Strategy

### Philosophy
All **code development, design, and documentation** happens on the **laptop**. All **hardware integration and physical testing** happens on the **shuttle PC**. Git is the single source of truth that synchronizes both.

### System Roles

```
┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│         LAPTOP (Dev)            │        │       SHUTTLE PC (Lab)          │
│─────────────────────────────────│        │─────────────────────────────────│
│ • Write all source code         │        │ • JAI eBUS SDK installed        │
│ • Design GUI layout             │  Git   │ • 10 GigE NIC + camera          │
│ • Integrate AI models           │◄──────►│ • Sorter / conveyor hardware    │
│ • Write documentation           │  sync  │ • DAQ hardware                  │
│ • Run unit tests (mock mode)    │        │ • Integration & system tests    │
│ • Manage Git, issues, PRs       │        │ • Performance benchmarking      │
└─────────────────────────────────┘        └─────────────────────────────────┘
```

### Key Rule
> **Never commit code that only works on one system.** All code must function on both machines — using `mock` mode on the laptop, `jai` mode on the shuttle. Any machine-specific config lives in `config.yaml` only, never hardcoded.

### Environment Variable for Machine Identity
Each machine has an environment variable set once:
```bash
# On laptop:
APPLE_GUI_MODE=development

# On shuttle PC:
APPLE_GUI_MODE=production
```
The app reads this to select sensible defaults automatically.

---

## 3. Repository Structure

```
apple_gui/
│
├── README.md                      ← Project intro, quick-start, badges
├── WORKFLOW.md                    ← This document
├── CHANGELOG.md                   ← Version history, what changed when
├── environment.yml                ← Conda environment spec (applegui)
├── environment-dev.yml            ← Conda dev environment spec (adds pytest, black, flake8)
├── .gitignore                     ← Excludes conda dirs, __pycache__, *.log, data/
├── .env.example                   ← Template for environment variables
│
├── config/
│   ├── config.yaml                ← Master config (camera mode, paths, thresholds)
│   ├── config_laptop.yaml         ← Laptop overrides (mock mode)
│   └── config_shuttle.yaml        ← Shuttle overrides (real camera)
│
├── core/                          ← Business logic — no GUI imports here
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   ├── base_camera.py         ← Abstract base class (interface contract)
│   │   ├── mock_camera.py         ← Fake camera for dev/testing
│   │   ├── jai_camera.py          ← Real JAI camera via Harvesters
│   │   └── image_buffer.py        ← Thread-safe frame queue
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_manager.py       ← Load, switch, manage AI models
│   │   └── inference_engine.py    ← Run inference, return grades
│   ├── control/
│   │   ├── __init__.py
│   │   ├── base_controller.py     ← Abstract base class
│   │   ├── mock_controller.py     ← Simulated sorter for dev
│   │   ├── sorter_controller.py   ← Real actuator commands
│   │   └── conveyor_controller.py ← Speed monitoring
│   └── logging/
│       ├── __init__.py
│       └── data_logger.py         ← Async image + CSV logging
│
├── gui/                           ← All Qt/UI code lives here
│   ├── __init__.py
│   ├── main_window.py             ← Main QMainWindow
│   ├── workers/
│   │   ├── camera_worker.py       ← QThread: camera acquisition
│   │   ├── inference_worker.py    ← QThread: AI inference
│   │   ├── sorter_worker.py       ← QThread: sorting control
│   │   └── logger_worker.py       ← QThread: async logging
│   ├── panels/
│   │   ├── camera_panel.py        ← Left panel: camera controls
│   │   ├── model_panel.py         ← Left panel: model selection
│   │   ├── control_panel.py       ← Left panel: sorter/conveyor
│   │   └── stats_panel.py         ← Right panel: live stats
│   └── widgets/
│       ├── image_display.py       ← 3-channel image viewer widget
│       ├── status_bar.py          ← System health indicators
│       └── chart_widgets.py       ← Grade chart, throughput graph
│
├── models/                        ← AI model files (.pt, .onnx)
│   └── .gitkeep                   ← Keep folder in Git, not model files
│
├── data/                          ← Runtime output (NOT tracked in Git)
│   ├── logs/                      ← CSV grading logs
│   ├── images/                    ← Captured images
│   └── sessions/                  ← Session summaries
│
├── tests/                         ← All test files
│   ├── __init__.py
│   ├── test_camera.py
│   ├── test_inference.py
│   ├── test_sorter.py
│   └── test_logger.py
│
├── docs/                          ← Extended documentation
│   ├── camera_setup.md            ← How to connect and configure JAI camera
│   ├── shuttle_setup.md           ← Shuttle PC environment setup guide
│   ├── model_format.md            ← How to prepare and add AI models
│   └── hardware_wiring.md         ← Sorter/DAQ wiring diagram + protocol
│
└── scripts/
    ├── setup_laptop.bat           ← One-click laptop conda env setup (creates applegui)
    ├── setup_shuttle.bat          ← One-click shuttle conda env setup
    └── camera_test.py             ← Standalone camera connection test script
```

---

## 4. Git Workflow

### 4.1 Branching Model

```
main ──────────────────────────────────────────────────► (stable, release-ready)
  │
  ├── develop ────────────────────────────────────────► (integration branch)
  │     │
  │     ├── feature/camera-interface
  │     ├── feature/gui-skeleton
  │     ├── feature/model-manager
  │     ├── feature/sorter-control
  │     └── feature/data-logging
  │
  └── hotfix/xxx                                        (urgent bug fixes)
```

| Branch | Purpose | Who Merges |
|---|---|---|
| `main` | Stable, demo-ready code only | After milestone completion |
| `develop` | Active integration branch | After feature complete + tested |
| `feature/*` | Individual feature development | Via PR into `develop` |
| `hotfix/*` | Critical bug fixes | Directly into `main` + `develop` |

### 4.2 Branch Naming Convention

```
feature/camera-interface
feature/gui-main-window
feature/mock-camera
feature/ai-model-manager
feature/sorter-timing-logic
fix/camera-buffer-overflow
docs/camera-setup-guide
test/camera-unit-tests
```

### 4.3 Commit Message Format

Every commit message follows this format:
```
<type>(<scope>): <short description>

[optional body — what and why, not how]
[optional footer — issue references]
```

**Types:**
| Type | When to Use |
|---|---|
| `feat` | New feature added |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `refactor` | Code restructure (no behavior change) |
| `config` | Config file changes |
| `perf` | Performance improvement |
| `chore` | Maintenance (deps update, cleanup) |

**Examples:**
```
feat(camera): add mock camera with synthetic 3-channel frame generation
fix(inference): handle model loading failure gracefully with error dialog
docs(workflow): add shuttle PC setup instructions
test(camera): add unit tests for image buffer thread safety
config(camera): add jai mode settings to config.yaml
```

### 4.4 Daily Git Routine

**On Laptop (end of each dev session):**
```bash
git status                          # See what changed
git add -p                          # Review and stage changes interactively
git commit -m "feat(gui): add live image display widget"
git push origin feature/gui-skeleton
```

**On Shuttle PC (before each lab session):**
```bash
git fetch origin
git pull origin develop             # Get latest code
# Run integration test
python scripts/camera_test.py
```

### 4.5 .gitignore Rules

The following are NEVER committed:
```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Conda — environments are reproduced from environment.yml, not committed
.conda/
conda-meta/

# Runtime data
data/logs/
data/images/
data/sessions/

# AI models (too large — share via separate link)
models/*.pt
models/*.onnx
models/*.pth

# Environment & secrets
.env
*.env

# IDE
.vscode/settings.json
.idea/
*.suo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

---

## 5. Environment Setup

> **Package Manager: Conda**
> We use **Conda** for environment management. The environment is named `applegui` and is identical on both the laptop and shuttle PC. All dependencies are declared in `environment.yml` — this is the single source of truth.

### 5.1 Laptop (Development)

```bash
# 1. Create the conda environment from the spec file
conda env create -f environment.yml

# 2. Activate the environment (do this at the start of EVERY session)
conda activate applegui

# 3. Verify all key packages
python -c "import PyQt6, torch, cv2, yaml; print('Core packages OK')"

# 4. Register the environment as a Jupyter kernel (optional, for notebooks)
python -m ipykernel install --user --name applegui --display-name "Apple GUI"
```

**To update the environment after pulling new `environment.yml` changes:**
```bash
conda env update -f environment.yml --prune
```

**To completely recreate from scratch:**
```bash
conda env remove -n applegui
conda env create -f environment.yml
```

### 5.2 Shuttle PC (Integration)

Same conda setup PLUS hardware-specific steps:
```bash
# 1. Create the conda environment (same command as laptop)
conda env create -f environment.yml
conda activate applegui

# 2. Install JAI eBUS SDK (download from jai.com — see docs/camera_setup.md)
#    This installs the GenTL producer (.cti file) — NOT via conda

# 3. Configure 10 GigE NIC
#    - Jumbo Frames = 9014 bytes
#    - Disable Energy Efficient Ethernet (EEE)
#    - Set Receive Buffers to maximum

# 4. Test camera connection with eBUS Player (standalone JAI tool)

# 5. Set machine identity environment variable
setx APPLE_GUI_MODE production

# 6. Install NI-DAQmx Python package if using NI hardware
pip install nidaqmx          # Not in conda forge, use pip inside conda env

# 7. Verify full stack
python scripts/camera_test.py
```

### 5.3 Conda Environment Spec (environment.yml)

This file lives in the repo root and defines the entire environment:

```yaml
name: applegui
channels:
  - pytorch          # For CUDA-enabled torch builds
  - conda-forge
  - defaults

dependencies:
  - python=3.10

  # GUI
  - pyqt=6.5.*
  - pyqtgraph=0.13.*

  # AI / Vision
  - pytorch=2.1.*
  - torchvision=0.16.*
  - pytorch-cuda=11.8    # Remove this line if no GPU on laptop
  - opencv=4.8.*
  - numpy=1.24.*

  # Config & Data
  - pyyaml=6.0.*
  - h5py=3.9.*

  # Hardware
  - pyserial=3.5.*

  # Dev tools (included in base env for simplicity)
  - pytest=7.4.*
  - pytest-qt=4.2.*
  - black=23.*
  - flake8=6.*
  - isort=5.12.*

  # Camera (pip-only package)
  - pip
  - pip:
    - harvesters>=1.4.0
```

### 5.4 Daily Environment Activation

**Every single dev session starts with:**
```bash
conda activate applegui
```

To confirm you're in the right environment, your terminal prompt should show `(applegui)`. You can also verify:
```bash
conda info --envs       # Lists all envs, active one marked with *
python --version        # Should show Python 3.10.x
```

---

## 6. Coding Standards

### 6.1 General Rules

- **Language:** Python 3.10+ (use type hints everywhere)
- **Line length:** 100 characters max
- **Formatter:** `black` (run before every commit)
- **Linter:** `flake8` (zero warnings policy)
- **Import order:** `isort` (stdlib → third-party → local)

### 6.2 Type Hints (Required)

```python
# Good
def process_frame(frame: np.ndarray, model_id: str) -> tuple[str, float]:
    ...

# Bad
def process_frame(frame, model_id):
    ...
```

### 6.3 Docstrings (Required for all public methods)

```python
def acquire_frame(self) -> np.ndarray | None:
    """
    Acquire a synchronized 3-channel multispectral frame.

    Returns:
        np.ndarray: Shape (H, W, 3) with channels [CH1, CH2, CH3],
                    or None if acquisition failed.

    Raises:
        CameraNotConnectedError: If camera is not initialized.
    """
```

### 6.4 Error Handling

```python
# Always catch specific exceptions, never bare except:
try:
    frame = self.camera.acquire_frame()
except CameraTimeoutError as e:
    logger.warning(f"Frame timeout: {e}")
    return None
except CameraDisconnectedError as e:
    logger.error(f"Camera disconnected: {e}")
    self.camera_error_signal.emit(str(e))
```

### 6.5 No Magic Numbers

```python
# Bad
time.sleep(0.033)

# Good
FRAME_INTERVAL_SEC = 1.0 / TARGET_FPS  # 30 FPS → 33ms
time.sleep(FRAME_INTERVAL_SEC)
```

### 6.6 GUI ↔ Core Separation

> **Rule:** `core/` modules must have **zero imports from `gui/`**. The GUI depends on core, never the reverse. This ensures core logic can be unit-tested without launching a Qt application.

### 6.7 Thread Safety Rules

- Never update a Qt widget from a non-main thread — always use signals.
- Never share mutable state between threads without a `Queue` or `threading.Lock`.
- All `QThread` workers must implement a `stop()` method and clean up resources.

```python
# Correct: emit signal from worker thread, GUI updates in main thread
class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)  # Signal carries data

    def run(self):
        frame = self.camera.acquire_frame()
        self.frame_ready.emit(frame)       # Safe cross-thread communication
```

---

## 7. Mock vs Real Camera Mode

### 7.1 Abstract Base Class Contract

All camera backends implement the same interface:

```python
# core/camera/base_camera.py
from abc import ABC, abstractmethod
import numpy as np

class BaseCamera(ABC):

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def acquire_frame(self) -> np.ndarray | None:
        """Returns shape (H, W, 3) — channels [CH1, CH2, CH3]"""
        ...

    @abstractmethod
    def set_exposure(self, microseconds: int) -> None: ...

    @abstractmethod
    def set_frame_rate(self, fps: float) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...
```

### 7.2 Mock Camera Behavior

`MockCamera` generates synthetic 3-channel images for laptop development:
- **CH1 (visible):** Randomized apple-like texture with optional simulated defect.
- **CH2 (NIR):** Slightly different response pattern.
- **CH3 (deep NIR):** Gradient variation.
- Configurable FPS simulation (default 30 FPS).
- Can replay a folder of real recorded images for realistic testing.

### 7.3 Config-Driven Selection

```yaml
# config.yaml
camera:
  mode: "mock"          # Options: "mock" | "jai"
  mock:
    fps: 30
    resolution: [640, 480]    # Reduced for dev speed
    replay_folder: null       # Path to recorded frames, or null for synthetic
  jai:
    gentl_path: "C:/Program Files/JAI/eBUS SDK/bin/PvGenTL.cti"
    device_index: 0
    fps: 30
    exposure_us: 5000
    pixel_format: "Mono8"
```

```python
# core/camera/__init__.py — Auto-selects based on config
def create_camera(config: dict) -> BaseCamera:
    mode = config["camera"]["mode"]
    if mode == "mock":
        return MockCamera(config["camera"]["mock"])
    elif mode == "jai":
        return JAICamera(config["camera"]["jai"])
    else:
        raise ValueError(f"Unknown camera mode: {mode}")
```

### 7.4 Same for Sorter Controller

```yaml
# config.yaml
sorter:
  mode: "mock"          # Options: "mock" | "serial" | "nidaqmx"
```
Simulation mode logs all commands to console + CSV but sends nothing to hardware.

---

## 8. Testing Strategy

### 8.1 Test Levels

| Level | What | Where Runs | Tool |
|---|---|---|---|
| **Unit tests** | Individual functions, classes | Laptop + CI | `pytest` |
| **Integration tests** | Module interactions (mock camera) | Laptop | `pytest` |
| **Hardware tests** | Real camera + sorter | Shuttle only | `pytest` + manual |
| **System tests** | Full end-to-end at conveyor speeds | Shuttle + lab | Manual |

### 8.2 Running Tests

```bash
# Run all unit tests (laptop safe)
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=core --cov-report=html

# Run only camera tests
pytest tests/test_camera.py -v

# Skip hardware tests on laptop
pytest tests/ -m "not hardware"
```

### 8.3 Test Markers

```python
import pytest

@pytest.mark.hardware
def test_real_camera_connection():
    """Requires physical JAI camera — shuttle only."""
    ...

@pytest.mark.slow
def test_8_hour_endurance():
    """Long-running endurance test."""
    ...
```

### 8.4 What Must Be Tested

| Module | Key Tests |
|---|---|
| `MockCamera` | Frame shape, FPS timing, channel count |
| `ImageBuffer` | Thread safety, overflow handling, blocking behavior |
| `ModelManager` | Model loading, switching, missing file error |
| `InferenceEngine` | Output format, latency measurement |
| `SorterController` | Timing offset calculation, simulation mode logging |
| `DataLogger` | File creation, CSV format, async write |

---

## 9. Hardware Integration Protocol

### 9.1 When to Move to Shuttle

Move code to the shuttle only after **all of these are true:**
- [ ] Feature is complete and passing all unit tests on laptop.
- [ ] Code is committed and pushed to `develop`.
- [ ] Feature has been code-reviewed (self-review minimum).
- [ ] Mock mode works end-to-end.

### 9.2 Shuttle Integration Checklist

For each hardware integration session on the shuttle PC:
```
Before the session:
  [ ] git pull origin develop
  [ ] Verify camera is physically connected
  [ ] Verify eBUS Player shows 3 channels
  [ ] Check config_shuttle.yaml settings

During the session:
  [ ] Run camera_test.py first (standalone, no GUI)
  [ ] Document any errors with full traceback
  [ ] Note actual FPS and latency numbers
  [ ] Photograph any physical setup changes

After the session:
  [ ] Commit any config/code fixes
  [ ] Push to develop
  [ ] Update docs/camera_setup.md if needed
  [ ] Log results in TESTING_LOG.md
```

### 9.3 Known Hardware Configuration (Update as discovered)

| Parameter | Value | Status |
|---|---|---|
| Camera model | JAI FSFE-3200T-10GE | Confirmed |
| Lens | Edmund Optics 16mm f/1.6 VIS-NIR | Confirmed |
| CH1 wavelength | TBD | Need to verify |
| CH2 wavelength | TBD | Need to verify |
| CH3 wavelength | TBD | Need to verify |
| 10 GigE NIC model | TBD | Need to verify |
| Sorter interface | TBD | Need to confirm with advisor |
| Conveyor encoder | TBD | Need to confirm |
| Camera-to-gate distance | TBD | Measure physically |

---

## 10. Documentation Standards

### 10.1 What Gets Documented

| Document | Location | Updated When |
|---|---|---|
| Project brainstorm | `PROJECT_BRAINSTORM.md` | Project planning phase |
| Workflow (this doc) | `WORKFLOW.md` | Any process change |
| README | `README.md` | Every milestone |
| Changelog | `CHANGELOG.md` | Every version / merge to main |
| Camera setup | `docs/camera_setup.md` | After shuttle integration |
| Hardware wiring | `docs/hardware_wiring.md` | After sorter integration |
| Testing log | `docs/TESTING_LOG.md` | After every shuttle session |
| API docs | Inline docstrings | As code is written |

### 10.2 README Structure

The README always contains:
1. **What it is** — 2-sentence description.
2. **Screenshot / Demo GIF** — visual first impression.
3. **Quick Start** — conda setup → run in under 5 steps:
   ```bash
   git clone <repo>
   cd apple_gui
   conda env create -f environment.yml
   conda activate applegui
   python main.py
   ```
4. **Features list**.
5. **Hardware requirements**.
6. **Configuration guide** (mock vs real mode).
7. **License**.

### 10.3 CHANGELOG Format

```markdown
## [0.3.0] - 2026-05-28
### Added
- Real-time grade distribution bar chart (PyQtGraph)
- CSV data logger with async write

### Fixed
- Camera worker thread not stopping cleanly on disconnect

### Changed
- Moved config loading to startup, not per-module
```

### 10.4 Testing Log Format

Every shuttle session is logged in `docs/TESTING_LOG.md`:

```markdown
## Session 2026-05-21 — Camera Integration Test

**Tester:** [Your name]
**System:** Shuttle PC
**Camera:** JAI FSFE-3200T-10GE connected

### Results
| Test | Result | Notes |
|---|---|---|
| eBUS Player 3-channel stream | Pass | All 3 channels visible |
| Harvesters connection | Pass | Device detected index 0 |
| 30 FPS sustained | Pass | Actual: 28.7 FPS avg |
| 107 FPS sustained | Partial | Drops to 95 FPS after 2min |

### Issues Found
1. Buffer overflow at 107 FPS — increase buffer count in config.

### Action Items
- [ ] Increase `num_buffers` from 10 to 32 in config_shuttle.yaml
```

---

## 11. Development Rhythm

### 11.1 Daily Dev Session (Laptop)

```
START
  1. git pull origin develop
  2. Check open issues / TODO comments
  3. Pick ONE feature or fix to work on
  4. Create / switch to feature branch
  5. Write code + tests
  6. Run tests → all pass
  7. git add + commit (descriptive message)
  8. git push
END
```

### 11.2 Weekly Milestone Review

Every week, assess:
- Which phase are we in? (0–7)
- What was completed this week?
- What's blocked? Why?
- What's the priority for next week?
- Any hardware sessions needed on shuttle?

### 11.3 Development Phases (Progress Tracker)

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Hardware verification (eBUS Player, camera test) | Pending |
| **Phase 1** | Camera sandbox script (Python, no GUI) | Pending |
| **Phase 2** | GUI skeleton + threading (mock camera) | Pending |
| **Phase 3** | Camera integration in GUI (real camera on shuttle) | Pending |
| **Phase 4** | AI model integration (inference in GUI) | Pending |
| **Phase 5** | Sorter control (hardware timing logic) | Pending |
| **Phase 6** | Dashboard, charts, and data logging | Pending |
| **Phase 7** | Testing, validation, polish, ASABE demo prep | Pending |

---

## 12. Release & Demo Checklist

### Pre-Demo (Day Before ASABE Presentation)

**Software:**
- [ ] `main` branch is clean and tagged (e.g., `v1.0.0-aim26`)
- [ ] App starts in < 10 seconds on shuttle PC
- [ ] Camera connects automatically on startup
- [ ] All 3 channels display correctly
- [ ] Model loads and inference runs < 100ms
- [ ] Sorting timing tested at demo conveyor speed
- [ ] Data logging verified (CSV + images saving)
- [ ] No known crashes in last 2-hour test run

**Hardware:**
- [ ] Camera cable secured (no accidental unplug)
- [ ] Conveyor speed set to demo configuration
- [ ] Sorter actuators tested manually
- [ ] Backup laptop with recorded demo video (fallback if hardware fails)

**Presentation:**
- [ ] README is complete and professional
- [ ] GitHub repo is public
- [ ] Screenshots/demo GIF in README
- [ ] License file present
- [ ] `conda env create -f environment.yml` verified on clean machine
- [ ] `environment.yml` is up to date and pinned

---

*Document maintained by the development team. Update this document whenever a process changes — do not let it go stale.*
