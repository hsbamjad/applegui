"""
Test 3 -- YOLO Model Loads on GPU
Loads the first .pt file found in models/, runs a dummy inference, prints class names.
Run from project root: python scripts\test3_model.py
"""
import sys
import os
import glob

# ── Find a model file ──────────────────────────────────────────────────────────
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
pt_files   = glob.glob(os.path.join(models_dir, "*.pt"))

if not pt_files:
    print(f"FAIL: No .pt files found in {os.path.abspath(models_dir)}")
    print("      Put your model file there and re-run.")
    sys.exit(1)

model_path = pt_files[0]
print(f"Model found: {model_path}")

# ── Check GPU ─────────────────────────────────────────────────────────────────
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    device  = "cuda" if cuda_ok else "cpu"
    print(f"CUDA available: {cuda_ok}  -->  using device: {device}")
    if not cuda_ok:
        print("  Warning: no GPU -- inference will be slow on CPU.")
except ImportError:
    device = "cpu"
    print("torch not found -- defaulting to cpu")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
try:
    from ultralytics import YOLO
    model = YOLO(model_path)
except Exception as e:
    print(f"FAIL: model load error: {e}")
    sys.exit(1)

print(f"Class names: {model.names}")

# ── Dummy inference ───────────────────────────────────────────────────────────
print("Running dummy inference on a blank 640x640 frame...")
try:
    import numpy as np
    dummy   = np.zeros((640, 640, 3), dtype="uint8")
    results = model(dummy, device=device, verbose=False)
    n_boxes = len(results[0].boxes)
    print(f"Inference OK -- detections on blank frame: {n_boxes}  (0 expected)")
except Exception as e:
    print(f"FAIL: inference error: {e}")
    sys.exit(1)

print()
print("Test 3 PASSED")
