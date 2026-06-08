"""
Test 2 -- Video Files Open Correctly
Checks that OpenCV can read all 3 channel videos and reports shape + FPS.
Run from project root: python scripts\test2_videos.py
"""
import cv2
import sys

VIDEOS = {
    "CH1 Color": r"D:\HA\apple_gui\videos\Source0\G1\G1.mp4",
    "CH2 NIR1":  r"D:\HA\apple_gui\videos\Source1\G1\G1.avi",
    "CH3 NIR2":  r"D:\HA\apple_gui\videos\Source2\G1\G1.avi",
}

all_ok = True

for name, path in VIDEOS.items():
    cap = cv2.VideoCapture(path)
    opened = cap.isOpened()
    if not opened:
        print(f"  FAIL  {name}: could not open  {path}")
        all_ok = False
        cap.release()
        continue

    ok, frame = cap.read()
    fps       = cap.get(cv2.CAP_PROP_FPS)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not ok or frame is None:
        print(f"  FAIL  {name}: opened but could not read first frame  {path}")
        all_ok = False
    else:
        print(f"  OK    {name}: shape={frame.shape}  fps={fps:.1f}  frames={total}")

print()
if all_ok:
    print("Test 2 PASSED")
else:
    print("Test 2 FAILED -- fix the paths or re-run the conversion script")
    sys.exit(1)
