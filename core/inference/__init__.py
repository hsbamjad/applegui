"""
core/inference/__init__.py
===========================
AI Inference Module - Public API

Model: YOLOv8m-seg (instance segmentation + tracking)
Trained on: Gala apples (396 hand-picked, diverse defects)
Classes: Fresh | Processing | Cull
Input:  3-channel multispectral image (RG + NIR1 + NIR2 from JAI)
Output: Per-apple instance - grade, confidence, bounding box, mask, diameter_px

Apple tracking across frames:
  - Each apple is assigned a YOLO tracking ID when it enters the frame
  - Grade is evaluated frame-by-frame as the apple moves through inspection area
  - Final grade uses weighted voting: sum(confidence × class_vote) across all frames
  - Vote is finalized when the apple crosses the "grade line" (config: grade_line_x)
  - Final grade + tracking_id is passed to SorterController.schedule()

Dataset:
  Training:   1907 Fresh | 1034 Processing | 335 Cull  instances
  Validation: 249  Fresh |  320 Processing | 108 Cull
  Testing:    2550 Fresh |  644 Processing |  74 Cull
"""

from core.inference.model_manager import ModelManager, GradeResult

__all__ = ["ModelManager", "GradeResult"]
