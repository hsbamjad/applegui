"""
gui/drawing.py
==============
Shared OpenCV drawing helpers for tracked-object overlays.
Used by the inference worker (off GUI thread) and main window.

Call configure(class_names) after loading a model to update the
module-level CLASS_NAMES and CLASS_COLORS for any grade set
(works for both apple and sweet potato modes).
"""

from __future__ import annotations

import cv2
import numpy as np

DRAW_W = 512

# ── Color palette for N classes ────────────────────────────────────────────────

_PALETTE: list[tuple[int, int, int]] = [
    (52,  211, 153),    # 0: emerald green  (Normal / Fresh)
    (251, 191,  36),    # 1: amber          (Moderate defect / Processing)
    (248, 113, 113),    # 2: coral red      (Severe defect / Cull)
    (129, 140, 248),    # 3: indigo
    (251, 146,  60),    # 4: orange
    (34,  211, 238),    # 5: cyan
    (163, 230,  53),    # 6: lime
    (244, 114, 182),    # 7: pink
]

# Module-level mutable lists - updated by configure()
CLASS_COLORS: list[tuple[int, int, int]] = [
    (52, 211, 153),    # Fresh - emerald green
    (251, 191, 36),    # Processing - amber
    (248, 113, 113),   # Cull - red
]

CLASS_NAMES: list[str] = ["Fresh", "Processing", "Cull"]


def configure(class_names: list[str]) -> None:
    """Update module-level CLASS_NAMES and CLASS_COLORS for any set of grades.

    Call this after loading a model to ensure drawing labels and colors
    match the actual class names the model outputs.

    Parameters
    ----------
    class_names : list of class name strings in class-index order.
                  e.g. ["Normal", "Moderate defect", "Severe defect"]
    """
    global CLASS_NAMES, CLASS_COLORS
    CLASS_NAMES  = list(class_names)
    CLASS_COLORS = [_PALETTE[i % len(_PALETTE)] for i in range(len(class_names))]


def annotate_tracked(
    frame: np.ndarray,
    active: list,
    show_label: bool = True,
    box_ref_w: int | None = None,
) -> np.ndarray:
    """Draw bounding boxes on a downscaled copy for fast display.

    box_ref_w: width of the coordinate space used in active[].box (e.g. 2048
               when frame is already a 512px thumb of that source).
    """
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    ref_w = box_ref_w if box_ref_w else w
    scale_f = DRAW_W / ref_w
    draw_h = int(h * (DRAW_W / w)) if w != DRAW_W else h

    if frame.ndim == 2:
        src = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        src = frame

    if w == DRAW_W:
        small = src
    else:
        small = cv2.resize(src, (DRAW_W, draw_h), interpolation=cv2.INTER_LINEAR)

    if not active:
        return small

    fs = 0.55
    box_thick = 2
    txt_thick = 1

    for t in active:
        cls = t["class_id"]
        conf = t["conf"]
        seq = t.get("seq_id")
        lane = t["lane"]
        eligible = t.get("eligible", True)
        x1, y1, x2, y2 = t["box"]

        sx1 = int(x1 * scale_f)
        sy1 = int(y1 * scale_f)
        sx2 = int(x2 * scale_f)
        sy2 = int(y2 * scale_f)

        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        draw_color = color if eligible else (120, 120, 120)

        cv2.rectangle(small, (sx1, sy1), (sx2, sy2), draw_color, box_thick)

        id_part = f"#{seq}" if seq is not None else "?"
        if show_label:
            name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            label = f"{id_part} {name} {conf * 100:.0f}% L{lane}"
        else:
            label = f"{id_part} L{lane}"

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, txt_thick)
        lx = max(0, sx1)
        ly = max(lh + 4, sy1 - 4)

        cv2.rectangle(small, (lx, ly - lh - 3), (lx + lw + 4, ly + 2), draw_color, -1)
        cv2.putText(
            small, label, (lx + 2, ly - 1),
            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), txt_thick, cv2.LINE_AA,
        )

    return small
