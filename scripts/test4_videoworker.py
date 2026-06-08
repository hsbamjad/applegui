"""
Test 4 -- VideoWorker Emits Frames
Runs VideoWorker standalone (no GUI window) and checks it emits frames correctly.
Run from project root: python scripts\test4_videoworker.py
"""
import sys
import os
import time

# Must run from project root so 'gui' package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import QTimer
from gui.workers.video_worker import VideoWorker

VIDEOS = {
    "ch1": r"D:\HA\apple_gui\videos\Source0\G1\G1.mp4",
    "ch2": r"D:\HA\apple_gui\videos\Source1\G1\G1.avi",
    "ch3": r"D:\HA\apple_gui\videos\Source2\G1\G1.avi",
}

app     = QApplication(sys.argv)
results = {"count": 0, "errors": [], "last_shapes": {}}
t0      = time.perf_counter()

def on_frame(ch1, ch2, ch3, fps):
    n = results["count"] + 1
    results["count"] = n
    results["last_shapes"] = {
        "ch1": ch1.shape if ch1 is not None else None,
        "ch2": ch2.shape if ch2 is not None else None,
        "ch3": ch3.shape if ch3 is not None else None,
    }
    if n % 30 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  Frame {n:4d}  fps={fps:.1f}"
              f"  ch1={results['last_shapes']['ch1']}"
              f"  ch2={results['last_shapes']['ch2']}"
              f"  ch3={results['last_shapes']['ch3']}"
              f"  elapsed={elapsed:.1f}s")

def on_status(msg, is_error):
    tag = "ERROR" if is_error else "INFO"
    print(f"  [{tag}] {msg}")
    if is_error:
        results["errors"].append(msg)

def on_finished():
    elapsed = time.perf_counter() - t0
    print(f"\nVideoWorker finished -- {results['count']} frames in {elapsed:.1f}s")
    app.quit()

# Safety timeout: quit after 15s even if video is very long
QTimer.singleShot(15_000, app.quit)

worker = VideoWorker(
    path_ch1 = VIDEOS["ch1"],
    path_ch2 = VIDEOS["ch2"],
    path_ch3 = VIDEOS["ch3"],
    fps      = 30,
    loop     = False,       # no loop -- just read through once
)
worker.sig_frame.connect(on_frame)
worker.sig_status.connect(on_status)
worker.finished.connect(on_finished)
worker.start()

print("VideoWorker started -- reading up to 15 seconds of video...")
print()
app.exec()

# ── Results ───────────────────────────────────────────────────────────────────
print()
if results["count"] == 0:
    print("Test 4 FAILED -- no frames were emitted")
    sys.exit(1)
elif results["errors"]:
    print(f"Test 4 FAILED -- errors: {results['errors']}")
    sys.exit(1)
else:
    print(f"Last frame shapes: {results['last_shapes']}")
    print("Test 4 PASSED")
