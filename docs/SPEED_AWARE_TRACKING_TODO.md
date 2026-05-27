# Speed-Aware Tracking — Pending Work

**Status:** ⏸ Blocked — waiting on confirmed conveyor speed from lead  
**Created:** 2026-05-27  
**Relevant files:** `gui/workers/tracker.py` · `config/config.yaml`

---

## Background

The tracking and grading pipeline uses several **fixed parameters** that were
tuned for an assumed belt speed. If belt speed changes significantly — either
as a one-time change or dynamically during operation — these parameters need
to be revisited.

---

## Parameters That Are NOT Speed-Aware

All of these live in `config/config.yaml` under `inference.tracking` and are
read **once at startup**. They do not adapt if belt speed changes at runtime.

| Parameter | Default | What it controls | Speed impact |
|---|---|---|---|
| `min_frames` | `5` | Minimum frames seen before an apple can be graded | **Critical.** At fast belts, apple may exit before 5 detections → **never graded, never counted.** |
| `max_recover_dist` | `80` px | Max pixel distance to re-link a lost track ID | At fast belts, apple moves more px/frame → ID re-link fails → vote history reset → grade less reliable. |
| `max_lost_frames` | `10` | Frames an apple can disappear and still be recovered | At fast belts, 10 frames = less real time → too tight a recovery window. |
| `count_memory_frames` | `40` | Frames the proximity buffer remembers (anti double-count) | Less critical, but at very high speeds may need widening. |
| `min_count_dist_frac` | `0.12` | Min spatial distance to count as a new apple (12% of frame) | At high speeds, two apples may pass through counting gate in rapid succession — buffer may be too short in time. |

---

## The Critical Risk: `min_frames` vs Inference FPS vs Belt Speed

An apple must be **detected in at least `min_frames` frames** to receive a grade.
The number of frames captured depends on:

```
frames_captured = inference_fps × apple_transit_time_seconds
apple_transit_time = frame_height_meters / belt_speed_m_s
```

**Example scenario at 15 FPS inference:**

| Belt speed | Transit time (est.) | Frames captured | min_frames=5 | Result |
|---|---|---|---|---|
| 0.3 m/s (slow) | ~2.0 s | ~30 | ✅ 30 ≥ 5 | Graded |
| 0.7 m/s (medium) | ~0.9 s | ~13 | ✅ 13 ≥ 5 | Graded |
| 1.2 m/s (fast) | ~0.5 s | ~8 | ✅ 8 ≥ 5 | Graded |
| 2.0 m/s (very fast) | ~0.3 s | ~4 | ❌ 4 < 5 | **Silent miss** |

> ⚠️ At very fast speeds, apples may pass through the frame without accumulating
> enough detections. They will not be graded and will not be counted — silently.

---

## What Currently Uses the Conveyor Speed Slider

The `conveyor_speed` value from the GUI slider is currently only used for:
- `metrics_group.record_grade(speed)` → **apples/min throughput display only**

It does **not** feed into the tracker, inference, or counting logic.

---

## Proposed Fix (implement after speed is confirmed)

### Option A — Static tuning (simplest)
Once the actual belt speed is known, tune `config.yaml` once:

```yaml
inference:
  tracking:
    min_frames: 3          # lower if belt is fast
    max_recover_dist: 120  # raise if belt is fast
    max_lost_frames: 8     # lower if belt is fast
```

This works perfectly if belt speed is **fixed and known**.

### Option B — Dynamic scaling (if speed varies)
Wire the conveyor speed slider into the tracker so `min_frames` and
`max_recover_dist` scale automatically:

```python
# In _on_speed_changed() → recalculate tracker params
min_frames = max(3, int(inference_fps * transit_time * 0.4))
max_recover_dist = int(base_recover_px * (speed / reference_speed))
self._tracker.update_params(min_frames=min_frames, max_recover_dist=max_recover_dist)
```

Requires adding `update_params()` to `AppleTracker` (~10 lines).

### Option C — Minimum frame count from time instead of frame count
Replace the frame count guard with a **time-based guard**:

```python
# Instead of: frames_seen >= min_frames
# Use: time_seen_ms >= min_visible_ms  (e.g. 150ms)
time_seen = (frame_no - first_frame_no) / inference_fps
if time_seen >= min_visible_seconds:
    # eligible for counting
```

This is automatically speed-independent since it operates in real time, not
frame count. Most robust option but requires the tracker to track timestamps.

---

## Information Needed from Lead

- [ ] Confirmed belt speed in m/s (or ft/min)
- [ ] Whether belt speed is fixed or variable during operation
- [ ] Approximate camera field-of-view height in meters (how much of the belt is visible)
- [ ] Target inference throughput (current: ~15 FPS on GPU, ~5 FPS on CPU)

---

## Related Files

- [`tracker.py`](../gui/workers/tracker.py) — `AppleTracker.update()`, `__init__` params
- [`inference_worker.py`](../gui/workers/inference_worker.py) — `RealInferenceWorker`, queue sizing
- [`config.yaml`](../config/config.yaml) — all tunable params under `inference.tracking`
- [`22-5-26 Moveforward.md`](./22-5-26%20Moveforward.md) — main project roadmap
