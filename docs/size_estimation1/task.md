# Apple Size Estimation - Task List

## Phase 1 - Code (Do Now)

- [x] **Step 1:** Create `gui/workers/size_calibration.py` (SizeCalibrator class)
- [x] **Step 2:** Modify `gui/workers/tracker.py` (add size fields to GradeRecord + accumulation)
- [x] **Step 3:** Modify `config/config.yaml` (add size_estimation block)
- [x] **Step 4:** Modify `gui/workers/inference_worker.py` (wire calibrator to tracker)
- [x] **Step 5:** Update CSV logger columns
- [x] **Step 6:** Add live GUI overlay (diameter annotation on video feed)
- [x] **Step 7:** Run GUI in simulation mode - verify size_px > 0 in overlay and CSV

## Phase 2 - Scripts (Build now, run later)

- [x] **Step 8:** Create `scripts/calibrate_size.py` (Primary + Backup methods)
- [x] **Step 9:** Create `scripts/validate_size.py` (LR model, R2, P vs D plot)

## Phase 3 - Calibration + Validation (When hardware is locked)

- [ ] Run calibration session (Primary method with confirmed lens)
- [ ] Verify quadratic fit R² > 0.95
- [ ] Run validate_size.py - check R² > 0.99, RMSE < 1.79 mm
- [ ] Generate P vs D scatter plot for paper
