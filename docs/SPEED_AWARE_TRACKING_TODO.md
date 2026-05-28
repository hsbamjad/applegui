# Speed-Aware Tracking — Status & Notes

**Status:** ✅ CLOSED — all 3 speeds confirmed safe, no code changes needed  
**Updated:** 2026-05-28 — pitch count measured from simulation video  
**Relevant files:** `gui/workers/tracker.py` · `config/config.yaml`

---

## Conveyor Type: Screw Conveyor (NOT a belt)

Confirmed from mechanical engineer (2026-05-28):

> "One apple is sorted with every full rotation. The conveyor is fixed in the
> axis direction."

This is a **rotating helical screw (auger)**. The screw rotates in place —
it does NOT translate. Each apple sits in a cup between helical flights and
advances **exactly one pitch** per full rotation.

```
Axis direction (screw rotates, does NOT slide):

   🍎        🍎        🍎        🍎
  ╱─╲  ╱╲  ╱─╲  ╱╲  ╱─╲  ╱╲  ╱─╲
 ╱   ╲╱  ╲╱   ╲╱  ╲╱   ╲╱  ╲╱   ╲
 ─────────────────────────────────  ← axis (fixed)
      ↑
   one pitch = one apple slot
```

**Operating speeds:** 1, 2, and 3 apples/second
(= 1, 2, 3 rotations/second of the screw)

---

## Key Insight: Apple Motion is Slow and Discrete

Unlike a belt conveyor where the apple slides continuously, on a screw conveyor:

- Between rotations the apple is **nearly stationary**
- It **steps forward one pitch** per rotation
- From the camera's view: apple moves in discrete jumps, not continuous flow

**This is actually better for tracking** than a belt conveyor.

---

## Transit Time Analysis

The apple's time in the camera FOV depends on how many screw pitches the
camera sees at once:

```
transit_time_seconds = pitches_in_camera_FOV / rotations_per_second
frames_in_view       = transit_time × inference_fps
```

**Example: camera sees 6 pitches, inference at 15 FPS:**

| Speed | Transit time | Frames in view | min_frames=5 | Result |
|---|---|---|---|---|
| 1 apple/s | 6.0 sec | ~90 frames | ✅ well above | Graded, rich vote history |
| 2 apple/s | 3.0 sec | ~45 frames | ✅ well above | Graded, rich vote history |
| 3 apple/s | 2.0 sec | ~30 frames | ✅ well above | Graded, rich vote history |

**Even at 3 apples/s, an apple is visible for 30+ frames — `min_frames=5` is trivially satisfied.**

> ✅ **Current code parameters are fine for all three operating speeds.**

---

## Why the Screw Conveyor is Better to Track Than a Belt

| Concern (from previous belt analysis) | Belt conveyor | Screw conveyor |
|---|---|---|
| `min_frames = 5` satisified? | Risk at high speed | ✅ Easy at all speeds |
| `max_recover_dist = 80px` enough? | Risk at high speed | ✅ Apple barely moves per frame |
| ByteTrack ID stability | Drops ID if apple moves fast | ✅ Stable — apple is nearly stationary |
| Vote quality at 3 apple/s | ~4 frames (risky) | ~30 frames (excellent) |

---

## One Thing Still to Confirm

**How many screw pitches does the camera see at once?**

Ask: *"How many apple positions (screw cups/slots) are visible to the camera
at the same time?"*

- If ≥ 3 pitches → confirmed safe at all speeds (as shown in table above)
- If only 1 pitch → transit time is very short, need to verify frame count

---

## Original Belt-Speed Concerns — Now Resolved

The concerns in the original version of this doc (min_frames risk, max_recover_dist
risk) were written assuming a continuous belt conveyor where apples slide through
quickly. They do not apply to a screw conveyor. No code changes are needed
for speed handling.

---

## What Currently Uses the Conveyor Speed Slider

The `conveyor_speed` value from the GUI slider is currently only used for:
- `metrics_group.record_grade(speed)` → **apples/min throughput display only**

It does **not** feed into the tracker, inference, or counting logic.
This is acceptable — no dynamic speed adaptation is needed for a screw conveyor
running at fixed 1/2/3 apples/s.

---

## ✅ Pitch Count — Measured from Video (2026-05-28)

Video frame shows the simulation video (G1). Counted visible apple positions:

- Top lane:    ~7 apples visible
- Middle lane: ~7 apples visible  
- Bottom lane: ~6 apples visible

**Confirmed: ~6–7 screw pitches in camera FOV.**

```
transit_time   = 6 pitches / 3 apples/s = 2.0 seconds   (worst case, fastest speed)
frames_in_view = 2.0 s × 15 FPS        = 30 frames

30 frames >> min_frames=5  →  ✅ confirmed safe at all speeds
```

Also confirmed from the image: conveyor orientation is **LR** (left→right),
consistent with `config.yaml → conveyor.orientation: "LR"` for simulation.

- [x] Run simulation, count visible apples in frame → **~6–7 pitches**
- [x] Verify: `6 pitches / 3 apples/s × 15 FPS = 30 frames ≥ 5` → ✅ confirmed

**No code changes required. This topic is fully resolved.**

---

## Related Files

- [`tracker.py`](../gui/workers/tracker.py) — `AppleTracker.update()`, tracking params
- [`inference_worker.py`](../gui/workers/inference_worker.py) — `RealInferenceWorker`, queue
- [`config.yaml`](../config/config.yaml) — all tunable params under `inference.tracking`
- [`22-5-26 Moveforward.md`](./22-5-26%20Moveforward.md) — main project roadmap
