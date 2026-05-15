"""
gui/workers/__init__.py
"""
from gui.workers.camera_worker import CameraWorker
from gui.workers.inference_worker import MockInferenceWorker

__all__ = ["CameraWorker", "MockInferenceWorker"]
