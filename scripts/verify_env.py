"""Quick environment verification script for applegui conda env."""
import sys

print("=" * 45)
print("   applegui Environment Verification")
print("=" * 45)

try:
    import PyQt6.QtCore
    print(f"PyQt6:       {PyQt6.QtCore.PYQT_VERSION_STR}")
except ImportError as e:
    print(f"PyQt6:       FAILED — {e}")

try:
    import pyqtgraph
    print(f"PyQtGraph:   {pyqtgraph.__version__}")
except ImportError as e:
    print(f"PyQtGraph:   FAILED — {e}")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "None"
    print(f"PyTorch:     {torch.__version__}")
    print(f"{'CUDA OK' if cuda_ok else 'CUDA FAILED'}:         {cuda_ok} ({gpu_name})")
except ImportError as e:
    print(f"PyTorch:     FAILED — {e}")

try:
    import cv2
    print(f"OpenCV:      {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV:      FAILED — {e}")

try:
    import numpy as np
    print(f"NumPy:       {np.__version__}")
except ImportError as e:
    print(f"NumPy:       FAILED — {e}")

try:
    import yaml
    print(f"PyYAML:      {yaml.__version__}")
except ImportError as e:
    print(f"PyYAML:      FAILED — {e}")

try:
    import harvesters
    print(f"Harvesters:  {harvesters.__version__}")
except ImportError as e:
    print(f"Harvesters:  FAILED — {e}")

try:
    import serial
    print(f"PySerial:    {serial.__version__}")
except ImportError as e:
    print(f"PySerial:    FAILED — {e}")

print("=" * 45)
print(f"   Python: {sys.version.split()[0]}")
print("=" * 45)
