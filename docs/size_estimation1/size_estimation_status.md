# Apple Size Estimation - Status and Next Steps

## What is Belt Height?

Belt height is the **vertical distance from the camera lens down to the surface of the conveyor belt**.

The camera looks down at the apples as they travel on the belt. Apples are round, so a bigger apple sits higher up (its center is further from the belt than a small apple's center). The camera sees apples at slightly different distances depending on their size.

**Why it matters for calibration:**
The number of pixels an apple appears to be depends on how far it is from the camera. If the camera is 80 cm above the belt vs. 85 cm above the belt, the same apple will appear at different pixel sizes. The calibration equation bakes in a specific camera-to-belt distance. If that distance changes after calibration, the mm estimates will be off.

**So "belt height is fixed" means:** the camera mount is bolted down, the belt tension is set, nobody moves anything. Once you run calibration at a certain height, that height must stay the same for the system lifetime (or you re-calibrate).

> [!NOTE]
> This is why Dr. Lu said to wait. If hardware is still being set up and the camera position might change, any calibration done now would be wasted.

---

## Hardware Situation (Current)

| Item | Status |
|---|---|
| Camera | JAI FSFE-3200T-10GE (Sony IMX252, 2048x1536, pixel pitch = 3.45 um) |
| Production lens | JAI 0824-C3 - 8 mm focal length |
| G1 recording lens | 6 mm (0628HN/3CMOS) - NOT the production lens |
| Belt height | To be confirmed - hardware not locked yet |
| Calibration coefficients | null - pixel-only mode |

> [!IMPORTANT]
> The system currently runs in pixel-only mode. It shows `o 164px` in the overlay but cannot convert to mm until calibration is run with the confirmed hardware.
> G1 videos cannot be used for calibration - they were recorded with the 6 mm lens, not the 8 mm production lens.

---

## What Was Done

### Phase 1 - Full Pipeline Code (Done)

All code is written, tested, and verified live in simulation at 23.4 FPS.

| Step | File | What it does |
|---|---|---|
| 1 | `gui/workers/size_calibration.py` | `SizeCalibrator` class - position-aware quadratic scale function, pixel-only safe mode |
| 2 | `gui/workers/tracker.py` | `GradeRecord` gets `size_px`, `size_mm`, `cx_peak`; tracker accumulates D_px per frame |
| 3 | `config/config.yaml` | `size_estimation` block with lens specs, calibration coefficients = null |
| 4 | `gui/main_window.py` | Builds `SizeCalibrator` from config on startup, passes it to tracker |
| 5 | `core/logging/grade_logger.py` | CSV logger writes `size_px`, `size_mm`, `size_cf` per apple |
| 6 | `gui/main_window.py` | Live overlay: `#3 Fresh 87% L2  o 164px` on AI Model Input panel |
| 7 | Live test | Verified in simulation - overlay shows `o Xpx` on every apple |

### Phase 2 - Calibration Scripts (Done)

| Script | Purpose |
|---|---|
| `scripts/calibrate_size.py` | Runs YOLO + tracker on a recorded video session, matches apples to GT caliper data by order, fits quadratic `N_px(x) = ax^2 + bx + c`, writes coefficients to `config.yaml` |
| `scripts/validate_size.py` | Loads coefficients from config, runs tracker on a held-out session, reports R2, RMSE, MAE, saves P-vs-D scatter plot |

---

## What To Do Next - Phase 3 (When Hardware is Locked)

> [!IMPORTANT]
> Before running calibration, confirm with Dr. Lu:
> - The 8 mm lens is physically installed
> - The camera is at its final mounting height and will not move
> - Belt tension is set (height will not change)

### Step 1 - Check if any existing sessions used the 8 mm lens

Run the lens estimator on G2, G3, etc. to see if any were recorded with the 8 mm lens:

```bash
python scripts/estimate_lens.py \
    --videos videos/Source0/G2/G2.avi videos/Source1/G2/G2.avi videos/Source2/G2/G2.avi \
    --gt data/gt.xlsx --model models/best.pt --output docs/lens_G2.png
```

Repeat for G3, G4... If a session reports ~8 mm focal length, it can be used for calibration.

### Step 2 - Record new sessions if needed

If no existing sessions match the 8 mm lens, record 2-3 new sessions:
- Run each apple group through the conveyor
- Measure GT diameters with calipers right before or after running
- Label sessions (e.g. G12, G13) and add them to `data/gt.xlsx`

### Step 3 - Run calibration

```bash
python scripts/calibrate_size.py \
    --videos videos/Source0/G12/G12.avi videos/Source1/G12/G12.avi videos/Source2/G12/G12.avi \
    --gt data/gt.xlsx \
    --groups G12 \
    --model models/best.pt \
    --config config/config.yaml \
    --output docs/calibration_G12.png \
    --orientation LR \
    --device cuda
```

This prints R2 and RMSE on the calibration set, saves a scatter plot, and writes coefficients directly into `config/config.yaml`.

> [!TIP]
> Use `--groups G12 G13` to combine multiple sessions for more data points and a better fit.

### Step 4 - Check the calibration plot

Open `docs/calibration_G12.png`:
- Points should follow the quadratic curve
- Fit R2 should be above 0.95

Also open `config/config.yaml` and confirm the calibration section has real numbers, not null.

### Step 5 - Restart the GUI

No code changes needed. Just restart `python main.py`:
- Calibrator loads the new coefficients automatically
- Overlay switches from `o 164px` to `o 65mm`
- CSV logger writes real mm values

### Step 6 - Run validation on a held-out group

Use a different session from the calibration one:

```bash
python scripts/validate_size.py \
    --videos videos/Source0/G13/G13.avi videos/Source1/G13/G13.avi videos/Source2/G13/G13.avi \
    --gt data/gt.xlsx \
    --groups G13 \
    --model models/best.pt \
    --config config/config.yaml \
    --output docs/validation_G13.png
```

**Target accuracy (Mizushima and Lu 2013):**
- R2 > 0.99
- RMSE < 1.79 mm

If both pass, sizing is production-ready and the plot can go in the paper.

---

## Summary

| Phase | Status | Blocker |
|---|---|---|
| Phase 1 - Code | Done | None |
| Phase 2 - Scripts | Done | None |
| Phase 3 - Calibration | Pending | Hardware must be locked (8 mm lens confirmed, belt height fixed) |
| Phase 3 - Validation | Pending | Calibration must pass first |
