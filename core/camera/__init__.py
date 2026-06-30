"""
core/camera/__init__.py
========================
Camera Interface Module - Public API

Supported backends:
  - JAI FSFE-3200T-10GE via Harvesters (GenICam/GenTL)  [mode: "jai"]
  - Mock camera using synthetic or pre-recorded frames   [mode: "mock"]

The JAI FSFE-3200T-10GE is a 3-sensor prism-based camera:
  - 3 × Sony IMX252 (3.2 MP, 2048×1536, global shutter)
  - Bands: RG (~660nm) | NIR1 (~800nm) | NIR2 (~900nm)
  - Captures all 3 channels simultaneously (hardware-synchronized)
  - Interface: 10 GigE Vision → requires JAI eBUS SDK (.cti file)
  - Frame rate: up to 107 FPS; 60 FPS used for 3 apples/s/lane

Each buffer from Harvesters contains 3 payload components:
  buffer.payload.components[0] → CH1 (RG,   2048×1536, Mono8)
  buffer.payload.components[1] → CH2 (NIR1, 2048×1536, Mono8)
  buffer.payload.components[2] → CH3 (NIR2, 2048×1536, Mono8)
"""

from core.camera.camera_interface import CameraInterface, FrameTriplet

__all__ = ["CameraInterface", "FrameTriplet"]
