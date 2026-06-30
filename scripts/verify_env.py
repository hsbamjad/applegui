"""Quick environment verification script for applegui conda env."""
import sys
from core.log import get_logger, configure_root
configure_root()
logger = get_logger(__name__)

logger.info("=" * 45)
logger.info("   applegui Environment Verification")
logger.info("=" * 45)

try:
    import PyQt6.QtCore
    logger.info(f"PyQt6:       {PyQt6.QtCore.PYQT_VERSION_STR}")
except ImportError as e:
    logger.error(f"PyQt6:       FAILED - {e}")

try:
    import pyqtgraph
    logger.info(f"PyQtGraph:   {pyqtgraph.__version__}")
except ImportError as e:
    logger.error(f"PyQtGraph:   FAILED - {e}")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "None"
    logger.info(f"PyTorch:     {torch.__version__}")
    logger.info(f"{'CUDA OK' if cuda_ok else 'CUDA FAILED'}:         {cuda_ok} ({gpu_name})")
except ImportError as e:
    logger.error(f"PyTorch:     FAILED - {e}")

try:
    import cv2
    logger.info(f"OpenCV:      {cv2.__version__}")
except ImportError as e:
    logger.error(f"OpenCV:      FAILED - {e}")

try:
    import numpy as np
    logger.info(f"NumPy:       {np.__version__}")
except ImportError as e:
    logger.error(f"NumPy:       FAILED - {e}")

try:
    import yaml
    logger.info(f"PyYAML:      {yaml.__version__}")
except ImportError as e:
    logger.error(f"PyYAML:      FAILED - {e}")

try:
    import harvesters
    logger.info(f"Harvesters:  {harvesters.__version__}")
except ImportError as e:
    logger.error(f"Harvesters:  FAILED - {e}")

try:
    import serial
    logger.info(f"PySerial:    {serial.__version__}")
except ImportError as e:
    logger.error(f"PySerial:    FAILED - {e}")

logger.info("=" * 45)
logger.info(f"   Python: {sys.version.split()[0]}")
logger.info("=" * 45)
