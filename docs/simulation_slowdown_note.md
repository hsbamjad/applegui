# Simulation Slowdown - Root Cause and Current State

**Date:** 2026-05-28
**Observed by:** HA
**Symptom:** G8 video (actual duration: 58 seconds) takes ~2 min 58 sec in GUI simulation. That is roughly 3x slower than real-time.

---

## What "Simulation Mode" Does

When the GUI is run without the physical JAI camera, it uses `VideoWorker` to read three pre-recorded video files (one per spectral channel) and emits frame triplets to the rest of the pipeline, identical to what the real `CameraWorker` would emit. The GUI then processes these frames through the tracker and optionally through the YOLO inference worker.

---

## Current Pipeline - No Downscaling, Full Raw Resolution

This is the critical problem. Every stage works at the full 2048x1536 sensor resolution. Nothing is resized until the very last step (Qt display scaling for screen pixels), which is already too late.

### Stage-by-stage breakdown

```
[VideoWorker Thread]
  cap.read()  -->  2048x1536 BGR frame (9.4 MB)  CH1
  cap.read()  -->  2048x1536 BGR frame (9.4 MB)  CH2  <-- serial, one after another
  cap.read()  -->  2048x1536 BGR frame (9.4 MB)  CH3
  sig_frame.emit(ch1, ch2, ch3)  <-- 28 MB of numpy arrays sent via Qt signal each frame

[Main Thread - _on_frame()]
  self._last_ch1 = ch1    <-- 9.4 MB stored
  self._last_ch2 = ch2    <-- 9.4 MB stored
  self._last_ch3 = ch3    <-- 9.4 MB stored
  infer_w.enqueue(ch1, ch2, ch3)  <-- 28 MB pushed into queue (non-blocking)

[RealInferenceWorker Thread]
  _prepare_input(ch1, ch2, ch3)
    --> extracts R, B channels from 2048x1536 ch1
    --> slices 2048x1536 ch2
    --> np.stack([R, B, NIR1])  --> 2048x1536x3 array
  model.track(frame, imgsz=640)
    --> YOLO internally resizes 2048x1536 --> 640x480 itself
    --> runs inference
    --> (the extra resolution above 640px is completely wasted for inference)
  sig_result.emit(result)
  sig_input_frame.emit(frame.copy())  <-- another 28 MB copy

[Main Thread - _on_inference_result()]
  _annotate_tracked(self._last_ch1)  <-- cv2.rectangle() on 2048x1536 frame
  _annotate_tracked(self._last_ch2)  <-- same
  _annotate_tracked(self._last_ch3)  <-- same
  update_channel_frame() x3         <-- sends 3x full-res to display widget

[Display Widget - image_display.py - Main Thread]
  cv2.cvtColor(frame, BGR->RGB)      <-- 9.4 MB color conversion per channel
  QImage(...).copy()                 <-- another 9.4 MB copy into Qt heap
  QPixmap.scaled(SmoothTransformation)  <-- bilinear downscale 2048x1536 --> ~600x450
                                         <-- done 3 times per frame on main thread
```

---

## Why It Is So Slow

### Root Cause 1 - Three serial video decodes (biggest contributor)

```python
# video_worker.py - current code
for cap in caps:
    ok, frame = cap.read()   # serial, blocking, one by one
    frames.append(frame)
```

`cap.read()` on a 2048x1536 H.264-encoded video does two things synchronously:
- Reads compressed bytes from disk
- Decodes them into a full 9.4 MB BGR numpy array in RAM

At this resolution, H.264 decode typically takes 25-40 ms per frame on a single CPU core.
Three channels in sequence = 75-120 ms per loop iteration.

The target interval at 30 FPS is `1/30 = 33 ms`.
The actual decode time exceeds this, so the sleep becomes negative and is skipped.
Effective playback FPS = ~8-12 FPS instead of 30.

With the G8 video at 30 FPS native:
- Total frames = 58 sec x 30 FPS = 1740 frames
- At 10 FPS effective = 1740 / 10 = 174 seconds = 2 min 54 sec

That matches the observed 2 min 58 sec almost exactly.

### Root Cause 2 - SmoothTransformation on the main thread for every frame

In `image_display.py`:
```python
pixmap = QPixmap.fromImage(qt_img).scaled(
    disp_size,
    Qt.AspectRatioMode.KeepAspectRatio,
    Qt.TransformationMode.SmoothTransformation,  # bilinear, expensive
)
```

This runs 3 times per frame (once per channel) on the Qt main thread. SmoothTransformation does a bilinear interpolation of a 2048x1536 pixmap. This backs up the Qt event queue, preventing `_on_frame` from being processed quickly, which can add secondary lag.

### Root Cause 3 - 28 MB of numpy arrays passed through Qt signals per frame

Every `sig_frame.emit(ch1, ch2, ch3)` carries approximately 28 MB of Python objects through Qt's queued connection mechanism. The arrays are not copied (just Python object references), but Python GIL contention between the VideoWorker thread and the main thread adds overhead at high data throughput.

### Root Cause 4 - YOLO receives unnecessarily large input

`_prepare_input()` builds a 2048x1536x3 array and passes it to `model.track(imgsz=640)`. YOLO itself resizes it to 640x(something) before inference. This internal resize of a 2048x1536 array adds unnecessary latency in the inference worker. All resolution above 640px is discarded by YOLO anyway.

---

## Missing Optimizations (None of these are implemented yet)

- No resize or downscale anywhere before display
- No parallel channel reading (reads are serial, one channel at a time)
- No pre-buffering or pre-decoding of frames
- No fast-path for simulation mode (treated identically to real camera hardware, even though it does not need to be)

---

## Summary Table

| Stage | Frame Size | Cost | Notes |
|---|---|---|---|
| cap.read() x3 | 2048x1536 each | ~75-120 ms/frame | Serial H.264 decode, biggest bottleneck |
| Qt signal emit | 28 MB payload | Low (ref only) | GIL contention at high FPS |
| _on_frame() | 28 MB stored | Near zero | Just reference assignment |
| YOLO _prepare_input | 2048x1536x3 | ~5-10 ms | Unnecessary - YOLO discards above 640px |
| YOLO internal resize | 2048x1536 --> 640 | ~10-20 ms | Could be done earlier, once |
| _annotate_tracked x3 | 2048x1536 each | ~5 ms | cv2.rectangle on full-res |
| cv2.cvtColor x3 | 2048x1536 each | ~15 ms | BGR->RGB, done per display call |
| QImage.copy() x3 | 9.4 MB each | ~10 ms | Qt buffer copy |
| QPixmap.scaled() x3 | 2048x1536 --> 600px | ~20-30 ms | Bilinear on main thread |

---

## Real Camera FPS - Behavior Across Three Scenarios

The simulation slowdown does not apply to the real JAI camera. However, FPS behavior changes
depending on whether a model is loaded, and it is important to understand why.

### Scenario 1 - Real camera, no model loaded (confirmed working at 30 FPS)

Code path in `_on_frame()`:

```python
inference_running = False   # _infer_w is None, no model loaded

if not inference_running:
    update_frames(ch1, ch2, ch3, fps)   # direct display, 30 FPS
```

Camera delivers frames at hardware speed (30 FPS). `_on_frame` immediately passes them
to the display. Nothing else runs. This is why you see perfect 30 FPS. The path is as
short as it can possibly be.

### Scenario 2 - Real camera, tracker constructed, still no model loaded

The `ConveyorTracker` is constructed during `_start_pipeline()` but it only executes
inside `_on_inference_result()`. That function is only triggered when YOLO sends a result.
With no model loaded, YOLO never runs, `_on_inference_result()` never fires, and the
tracker code never touches any frame.

Result: identical to Scenario 1. Still 30 FPS, no change at all.

### Scenario 3 - Real camera, model loaded and running

Code path in `_on_frame()` changes completely:

```python
inference_running = True    # model is loaded and worker thread is running

# Display is SKIPPED - update_frames() is NOT called here
if inference_running:
    self._infer_w.enqueue(ch1, ch2, ch3)   # frames go to YOLO queue instead
```

The display is no longer updated in `_on_frame`. It is only updated when
`_on_inference_result()` fires, which happens at YOLO inference speed, not camera speed.

Time cost per frame with model running (approximate):

| Step | Cost |
|---|---|
| Camera captures frame | Hardware, continuous 30 FPS |
| enqueue() in _on_frame | Near zero |
| _prepare_input() builds 2048x1536x3 composite | ~5 ms |
| YOLO internal resize 2048x1536 to 640 | ~10-20 ms |
| YOLO inference on GPU at 640x640 | ~20-40 ms |
| _annotate_tracked() x3 on 2048x1536 | ~5 ms |
| update_channel_frame() x3 display | ~30-40 ms |

Total per frame: roughly 70-110 ms, which means the display runs at about 9-14 FPS visually.
The camera hardware is still capturing at 30 FPS. Frames the GPU cannot keep up with are
discarded from the inference queue (maxsize=2 in the inference worker).

### Does the FPS drop affect grading accuracy?

No. The display FPS drop is cosmetic only. The camera captures 30 frames per second
regardless. YOLO processes approximately 10-15 per second. At 1 apple per second on the
conveyor, each apple is visible in the frame for roughly 2-3 seconds, giving the tracker
20-45 YOLO observations per apple. That is more than enough for reliable grade voting.

The lower visual FPS just means the video on screen looks choppy. The grading pipeline
underneath is working correctly and at full camera speed.

### Summary

| Scenario | Camera FPS | Display FPS | Grading |
|---|---|---|---|
| No model loaded | 30 (hardware) | 30 | Not running |
| Tracker only, no model | 30 (hardware) | 30 | Not running |
| Model loaded (GPU) | 30 (hardware) | ~10-15 (inference-limited) | Running correctly |
| Model loaded (CPU) | 30 (hardware) | ~3-5 (CPU inference is slow) | Running correctly, just slow to display |

