# Multi-spectral Apple Sorting Pipeline Architecture

This document outlines the complete AI inference, tracking, counting, and grading pipeline for the Multi-spectral Vision System. 

## 1. Resolution & Performance Flow

A critical design principle of this system is separating the camera capture, AI inference, and GUI display into decoupled pipelines. This ensures smooth UI performance without sacrificing sorting accuracy.

### Resolution Pipeline
- **Camera Capture (Main Thread)**: The JAI camera streams 3 channels (Visible, NIR1, NIR2) at **Full Resolution (2048×1536)** at 60 FPS.
- **YOLO Inference (GPU Thread)**: The full 2048×1536 frame is passed to YOLO. Internally, ultralytics temporarily downscales this to `640×640` to pass through the neural network (standard for YOLO models). The resulting bounding boxes are automatically mapped back to **Full 2048×1536 coordinates**.
- **Tracking & Grading (Main Thread)**: All geometric tracking, distance calculations, and counting logic operate strictly in **Full 2048×1536 coordinates**.
- **UI Annotation (Main Thread)**: To prevent GUI lag, the bounding boxes and text labels are drawn onto a **512px downscaled copy** of the frame. This copy is then sent to the PyQt display widget. Because the widget is physically small on a laptop screen (~500px wide), the visual quality is identical, but drawing on 512px is 16× faster and eliminates stuttering. The UI labels still report the true `2048×1536` resolution.

### Frame Rate & Queueing
The camera pushes frames at 60 FPS. The GPU inference runs at ~21 FPS.
To prevent the system from falling behind, the pipeline uses a decoupled **Drop-Oldest Queue (maxsize=2)**:
1. Camera puts new frames into the queue.
2. If the queue is full, the oldest frame is discarded.
3. The GPU always pulls the most recent frame available. 
This guarantees the tracker is always analyzing live data with zero latency buildup.

---

## 2. Tracking Logic (`AppleTracker`)

The system uses ultralytics' internal ByteTrack to assign initial IDs to detections, but we wrap this in a custom stateful `AppleTracker` to handle lost IDs, orientation, and robust counting.

### Orientation-Aware "Travel Axis"
Instead of hardcoding `x` or `y`, the algorithm maps the apple's 2D `(cx, cy)` position to a 1D scalar called `travel_pos`. `travel_pos` starts at 0 where the apple enters, and increases to `MAX` where it exits.
- **LR**: `travel_pos = cx`
- **RL**: `travel_pos = Width - cx`
- **TB**: `travel_pos = cy`
- **BT**: `travel_pos = Height - cy` (Our physical setup)
All logic (entry zones, exit bands) operates strictly on `travel_pos`. Lanes are assigned by splitting the orthogonal axis into 3 sections.

### Lost-Track Recovery
If ByteTrack loses an ID (e.g., apple rolls or occlusion) and assigns a new ID a few frames later, the `AppleTracker`:
1. Looks in a "lost ID" buffer for recent tracks.
2. If a new detection appears within `max_recover_dist` (80px) of where an old track disappeared, it merges them.
3. The apple's history (votes, frame counts) is perfectly preserved.

---

## 3. The 5-Stage Counting Gate

To prevent double-counting or phantom detections, an apple MUST pass 5 strict guards before it is counted and assigned a sequential ID (e.g., `#3`).

> [!IMPORTANT]  
> If an apple fails these checks, it is shown on screen as a grey `?` box. It is tracked, but it will never be counted or sent to the physical sorter.

1. **Entry Zone Check**: The apple's *very first detection* must have occurred in the first 35% of the frame (`entry_frac = 0.35`). This prevents an ID reassignment halfway down the belt from being counted as a "new" apple.
2. **Narrow Band Check**: The apple must physically cross the exit band at `exit_frac = 0.85` (± 2.5%). It is not an open-ended zone, it is a narrow tripwire.
3. **Min Frames Check**: The apple must have been tracked for at least `min_frames = 5`.
4. **Proximity Check**: When an apple hits the band, the system checks a history buffer (`count_memory_frames = 40`). If another apple was already counted within `min_count_dist_frac` (12% of screen width) of this location recently, it is flagged as a double-count and ignored.
5. **Single-Fire Commit**: Once counted, `seq_id` is assigned and `committed = True`. It can never be counted again.

---

## 4. The Grading Algorithm

Because the apple rotates as it travels down the screw conveyor, its visible defect might only be visible for a few frames. We do not rely on the final frame; we accumulate votes over the apple's entire journey.

### The Voting System
Every frame, the YOLO confidence is added to a running tally for that class:
`votes[class] += confidence * weight`

### The "Cull Bias"
Because finding bad apples is critical, the system is biased toward the `Cull` class:
- **Cull Weight**: Every Cull detection is multiplied by `cull_weight = 1.5`. A 60% Cull detection carries as much weight as a 90% Fresh detection.
- **Sustained Hits**: If YOLO detects Cull with >60% confidence for `hit_threshold = 20` frames, it forces a Cull grade regardless of the total vote ratio.
- **Ratio Threshold**: If Cull accumulates `cull_ratio_threshold = 0.55` (55%) of the total weighted votes, it wins.

### Final Decision
When the apple crosses the counting gate:
1. If the Cull conditions are met, grade = **Cull**.
2. Otherwise, we remove Cull votes, and compare **Fresh vs. Processing**. Whichever has the higher accumulated vote tally wins.
3. The result is locked in, the UI updates, and the Arduino trigger is queued.
