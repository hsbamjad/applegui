"""
gui/widgets/image_display.py
============================
Premium multi-channel image display widget.

Design: Thin 3px colored top-border accent per channel.
        Dark image area. Minimal, clean info footer.
        No garish colored headers — subtle and professional.
"""

from __future__ import annotations

import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QPen, QBrush
from PyQt6.QtCore import Qt, pyqtSlot

from gui.styles import CH_COLORS, BG_BASE, BG_CARD, BORDER, TEXT_1, TEXT_2, TEXT_3


CHANNEL_META = [
    {"name": "CH1", "label": "RG",   "band": "~660 nm"},
    {"name": "CH2", "label": "NIR1", "band": "~800 nm"},
    {"name": "CH3", "label": "NIR2", "band": "~900 nm"},
]


class ChannelPanel(QWidget):
    """
    Single spectral channel display card.

    Visual structure:
        ══════════════════  ← 3px colored top accent line
        ┌─────────────────┐
        │ CH1  Visible    │  ← chip overlay (top-left inside image)
        │                 │
        │   IMAGE AREA    │
        │                 │
        │   NO SIGNAL     │  ← dim centered placeholder text
        │                 │
        └─────────────────┘
        640×480 · -- FPS     ← clean footer
    """

    def __init__(self, idx: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._idx    = idx
        self._meta   = CHANNEL_META[idx]
        self._color  = CH_COLORS[idx]
        self._frames = 0
        self._setup()

    def _setup(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top accent bar ─────────────────────────────────────
        accent = QWidget()
        accent.setFixedHeight(3)
        accent.setStyleSheet(f"background-color: {self._color}; border: none;")
        root.addWidget(accent)

        # ── Image area ─────────────────────────────────────────
        self._img_container = QWidget()
        self._img_container.setStyleSheet(
            f"background-color: {BG_BASE}; border: none;"
        )
        img_layout = QVBoxLayout(self._img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)

        # ── Channel identifier row ──────────────────────────────
        # Colored dot (spectrum pill) + channel name + band info
        id_row = QHBoxLayout()
        id_row.setContentsMargins(10, 8, 10, 0)
        id_row.setSpacing(6)

        dot = QLabel("●")
        dot.setStyleSheet(
            f"color: {self._color}; font-size: 9px; background: transparent;"
        )

        ch_name = QLabel(self._meta["name"])
        ch_name.setStyleSheet(
            f"color: {TEXT_1}; font-size: 11px; font-weight: 700; "
            f"background: transparent; letter-spacing: 0.5px;"
        )

        ch_info = QLabel(
            f"{self._meta['label']}  ·  {self._meta['band']}"
        )
        ch_info.setStyleSheet(
            f"color: {TEXT_3}; font-size: 10px; background: transparent;"
        )

        id_row.addWidget(dot)
        id_row.addWidget(ch_name)
        id_row.addSpacing(4)
        id_row.addWidget(ch_info)
        id_row.addStretch()
        img_layout.addLayout(id_row)

        # QLabel for the image pixmap
        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._display.setStyleSheet("background-color: transparent; border: none;")
        img_layout.addWidget(self._display, stretch=1)

        self._img_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._img_container, stretch=1)

        # ── Footer info bar ────────────────────────────────────
        footer = QWidget()
        footer.setFixedHeight(24)
        footer.setStyleSheet(
            f"background-color: {BG_CARD}; "
            f"border-top: 1px solid {BORDER}; border: none;"
        )
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 0, 10, 0)

        self._lbl_res = QLabel("No Signal")
        self._lbl_res.setStyleSheet(f"color: {TEXT_3}; font-size: 10px;")
        self._lbl_fps = QLabel("-- FPS")
        self._lbl_fps.setStyleSheet(f"color: {TEXT_3}; font-size: 10px;")

        footer_layout.addWidget(self._lbl_res)
        footer_layout.addStretch()
        footer_layout.addWidget(self._lbl_fps)
        root.addWidget(footer)

        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        w, h = max(self._display.width(), 300), max(self._display.height(), 200)
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor(BG_BASE))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = w // 2, h // 2

        # Barely-visible crosshair — 4% opacity
        pen = QPen(QColor(self._color + "0A"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(cx, 0, cx, h)
        painter.drawLine(0, cy, w, cy)

        # Outer guide circle — 5% opacity
        pen.setColor(QColor(self._color + "0D"))
        painter.setPen(pen)
        r = min(w, h) // 3
        painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)

        # NO SIGNAL text — 20% opacity, wide letter spacing
        font = QFont("Segoe UI Variable", 10)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 5)
        painter.setFont(font)
        painter.setPen(QColor(self._color + "33"))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "NO SIGNAL")

        painter.end()
        self._display.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._frames == 0:
            self._draw_placeholder()

    @pyqtSlot(object, float)
    def update_frame(self, frame: np.ndarray, fps: float = 0.0) -> None:
        if frame is None:
            return

        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        h, w = frame.shape[:2]

        # 1. Convert grayscale and color to RGB to ensure absolute display compatibility
        if frame.ndim == 2:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            qt_img = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        else:
            return

        # 2. Viewport-Matching Scaling Bypass: Skip expensive QPixmap.scaled() if viewport matches frame dimensions
        disp_size = self._display.size()
        if w == disp_size.width() and h == disp_size.height():
            pixmap = QPixmap.fromImage(qt_img)
        else:
            pixmap = QPixmap.fromImage(qt_img).scaled(
                disp_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

        self._display.setPixmap(pixmap)
        self._frames += 1
        self._lbl_res.setText(f"{w}×{h}")
        self._lbl_fps.setText(f"{fps:.1f} FPS")

    def reset(self) -> None:
        self._frames = 0
        self._lbl_res.setText("No Signal")
        self._lbl_fps.setText("-- FPS")
        self._draw_placeholder()


class MultiChannelDisplay(QWidget):
    """3-channel display container with thin dividers between panels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panels: list[ChannelPanel] = []
        self.setStyleSheet(f"background-color: {BG_BASE};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)   # 1px divider between channels

        for i in range(3):
            panel = ChannelPanel(i, self)
            self._panels.append(panel)
            layout.addWidget(panel)

    def update_frames(
        self,
        ch1: np.ndarray,
        ch2: np.ndarray,
        ch3: np.ndarray,
        fps: float = 0.0,
    ) -> None:
        for panel, frame in zip(self._panels, [ch1, ch2, ch3]):
            panel.update_frame(frame, fps)

    def reset_all(self) -> None:
        for panel in self._panels:
            panel.reset()
