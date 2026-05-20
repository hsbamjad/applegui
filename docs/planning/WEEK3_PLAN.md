# Week 3 Plan — Multispectral Apple Sorting GUI
**Week of: May 26, 2026**
**Status: 🟢 Updated for Model & Data-First Focus — Track A & B Prioritized ✅**

---

## Context: The Pivot to Model & Data-First Validation

Following our successful completion of the camera pipeline and real-time GUI display, our primary focus has shifted. 

Rather than rushing to wire the physical sorter actuators and serial communication, **we are prioritizing Model and Dataset Validation**. Developing and refining the AI pipeline first is the most mathematically and scientifically sound way forward. The neural network's accuracy, physical channel inputs (RGB vs. 5-channel), and performance profiles will dictate exactly what data the downstream sorting timing logic must handle.

### The Restructured Week 3 Focus:

```
                  OLD PIPELINE (Week 2 Complete ✅)
                  ────────────────────────────────
                    Camera Streams → Buffer Sync → GUI Display
                                                       │
                                                       ▼
                  NEW WEEK 3 FOCUS (Model & Data Sandbox)
                  ──────────────────────────────────────
                    ┌───────────────────────────────────────────────┐
                    │  1. Import & Evaluate Custom YOLOv8 Weights   │
                    │  2. Port Reference Inference & Track Scripts  │
                    │  3. Playback Offline Multispectral Videos     │
                    │  4. Run Datasets through Preprocessing        │
                    └───────────────────────────────────────────────┘
                                                       │
                                                       ▼
                  DEFERRED FUTURE WORK (Sorter Controls)
                  ──────────────────────────────────────
                    Serial Communication → Timing Delays → Actuators
```

---

## Restructured Week 3 Tracks

```
TRACK A (Model & Scripts)         TRACK B (Datasets & Playback)     TRACK C (Data Logging)
──────────────────────────────    ──────────────────────────────    ──────────────────────────
Port inference scripts            Import raw/labeled datasets       Implement Async DataLogger
Load custom YOLOv8 weights        Build offline video sandbox       Write CSV grading logs
Verify input dimensions           Run validation on real images     Capture Cull defect TIFFs
Run test inference in GUI         Compare metrics against paper     Profile inference latency
```

**Priority: Track A & B (Model & Data Sandbox) → Track C (Logging) → Defer Track D (Sorter Controller to Week 4).**

---

## Track A — Model Loading & Inference Script Porting

### Goal
Import and port the research team's reference scripts, evaluate custom YOLOv8 model weights, verify exact network dimensions (RGB vs. 5-channel), and integrate the model loading controls directly into our GUI architecture.

---

### A1 — Port Reference Inference & Tracking Scripts
*   **Time estimate: 2–3 hrs**
*   Port the existing research team's inference scripts, tracking helpers, and apple grading logic into the project under a sandbox directory (`scripts/sandbox/` or `core/inference/`).
*   **Key aspects to identify in the reference code:**
    *   Does the model perform **Instance Segmentation** (YOLOv8-seg), **Object Detection** (YOLOv8-detect), or **Classification**?
    *   How are tracks assigned? Are they using standard BoT-SORT / ByteTrack, or a custom Euclidean distance centroid tracker?
    *   What is the mathematical threshold/metric used to decide the final grade (e.g. cumulative confidence, majority voting across frames)?
*   **Deliverable**: Reference scripts successfully ported, cleaned, and documented in `core/inference/reference/`.

---

### A2 — Build the ModelManager & Verify Input Dimensions
*   **Time estimate: 2 hrs**
*   Create a robust model loading utility inside `core/inference/model_manager.py`.
*   Verify the exact channel dimensions required by the trained model:
    *   **3-Channel Mode**: RGB only. (NIR channels are used for display only, or bypassed during model prediction).
    *   **5-Channel Mode**: Custom input layers mapping `[R, G, B, NIR1_norm, NIR2_norm]`.
*   **Implementation details**:
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
                log.info("Loaded custom weights: %s", model_name)
                return True
            except Exception as e:
                log.error("Failed to load model %s: %s", model_name, e)
                return False
    ```
*   **Deliverable**: Completed `core/inference/model_manager.py` capable of loading `.pt` weights and listing available files.

---

### A3 — Construct Fixed Preprocessing Pipeline
*   **Time estimate: 2 hrs**
*   Build a reproducible image preprocessor (`core/inference/preprocessing.py`) to convert raw multispectral frames into standard tensors.
*   **Critical Constraint**: Unlike the GUI display pipeline which uses adaptive EMA scaling (essential for human viewing), **inference must use absolute physical normalization** so identical apples receive identical pixel ratings regardless of brightness fluctuations.
*   **Deliverable**: Preprocessing module with verification scripts demonstrating successful image conversions.

---

## Track B — Labeled Datasets & Offline Video Playback Sandbox

### Goal
Set up an offline playback sandbox to load real multispectral videos and labeled datasets in the GUI, run live inference on the recorded data, and validate accuracy before running live tests on the conveyor.

---

### B1 — Implement Offline Video Playback Sandbox
*   **Time estimate: 2–3 hrs**
*   Create a virtual camera backend (`core/camera/video_player.py`) that reads pre-recorded multispectral video files (channels 1, 2, and 3) instead of live camera feeds.
*   Allows the user to test the GUI, inference workers, tracking lines, and data logging offline directly on their development laptop.
*   **Deliverable**: "Offline Video Mode" integrated into `config.yaml` and selectable from the GUI connection panel.

---

### B2 — Run Dataset Validation & Accuracy Audits
*   **Time estimate: 2 hrs**
*   Feed raw test images from the labeled dataset (comprising Fresh, Processing, and Cull classes) through our preprocessing and loaded YOLOv8 models.
*   Generate confusion matrices and confidence logs. Compare the GUI's local classification output with the benchmarks documented in the ASABE sorting research paper.
*   **Deliverable**: Validation report detailing model classification accuracy across the imported test dataset.

---

## Track C — Async Data Logging & GUI Reporting

### Goal
Implement asynchronous, non-blocking CSV logs and TIFF defect frame saving. Run profiling audits to measure processing latency and ensure the app maintains high performance.

---

### C1 — Async CSV Data Logger
*   **Time estimate: 1.5 hrs**
*   Build `core/logging/data_logger.py` to write graded results in the background, utilizing a queue to protect CPU cycles.
*   Log columns: `timestamp`, `frame_idx`, `lane`, `apple_id`, `assigned_grade`, `confidence_score`.
*   **Deliverable**: CSV files successfully generated in `data/logs/` triggered by the GUI's "Enable Logging" checkbox.

---

### C2 — High-Resolution Defect TIFF Capture (Stretch Goal)
*   **Time estimate: 1.5 hrs**
*   Whenever a fruit is graded as **Cull** (defect), trigger a background worker to bundle the full-resolution uncompressed color and NIR frames into a single 5-channel TIFF file. 
*   Provides a persistent research archive of physical defects for future model training adjustments.
*   **Deliverable**: High-res TIFF output utility writing to `data/captures/culls/`.

---

### C3 — GUI Performance Profiling
*   **Time estimate: 1 hr**
*   Measure end-to-end processing speeds:
    $$\text{Latency} = \text{Frame Capture} + \text{Demosaic} + \text{Inference} + \text{UI Draw}$$
*   Identify bottlenecks: verify if GPU-accelerated PyTorch (CUDA `cu124`) is successfully utilized and measure the frame retrieval times to guarantee we stay well within our conveyor budget.
*   **Deliverable**: Latency profiling report in log outputs.

---

## Week 3 → Week 4 Gate

**Do NOT proceed to Week 4 (Sorter serial connection and timed hardware gate integration) until:**
1. Custom YOLOv8 weights are successfully loaded and running inference.
2. Ported tracking and grading scripts are verified against offline multispectral videos.
3. Confusion matrices and confidence logs match research benchmarks.
4. Preprocessing pipelines are locked down with physical fixed-range normalization.

---

## Open Clarifications Needed from Research Team / Dr. Lu

| # | Clarification Item | Core Dependencies |
|---|---|---|
| 1 | **Model format**: YOLOv8 seg, detect, or custom classification head? | A1, A2 |
| 2 | **Input channels**: RGB-only or 5-channel (RGB + NIR1 + NIR2)? | A3 |
| 3 | **Model files**: Access to the custom weights (`.pt`) file? | A2 |
| 4 | **Reference scripts**: Access to tracking and inference python files? | A1 |
| 5 | **Validation videos**: Access to multispectral conveyor videos? | B1 |
| 6 | **Labeled datasets**: Access to labeled fruit images for accuracy audits? | B2 |

---

*Updated: May 20, 2026. Prioritized Model, Labeled Datasets, and Offline Video Sandbox for Week 3.*
