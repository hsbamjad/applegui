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
from PyQt6.QtCore import Qt, QTimer, pyqtSlot

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
        # Last received frame — re-rendered on resize so maximizing fills correctly
        self._last_frame:      np.ndarray | None       = None
        self._last_fps:        float                   = 0.0
        self._last_orig_shape: tuple[int, int] | None  = None
        # Debounce timer: fires once after window resize settles (avoids animation
        # stutter from re-rendering on every intermediate resizeEvent call).
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_resize_settled)
        # ROI preview overlay — sensor-space rectangle (None = no overlay)
        self._roi_preview: tuple[int, int, int, int] | None = None  # (ox, oy, w, h)
        # Active ROI: the sensor-space region the camera is currently streaming.
        # Defaults to full frame. Updated by set_active_roi() after firmware confirms.
        self._active_roi: tuple[int, int, int, int] = (0, 0, 2048, 1536)
        # Actual frame dimensions from the last received frame (may differ from
        # full sensor after ROI is applied).
        self._frame_w: int = 2048
        self._frame_h: int = 1536
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
        elif self._last_frame is not None:
            # Debounce: resize fires ~60x during maximize animation.
            # Start/restart the timer — _on_resize_settled fires once it settles.
            self._resize_timer.start(100)

    def _on_resize_settled(self) -> None:
        """Called once after window resize is stable. Re-renders at final size."""
        if self._last_frame is not None:
            self._render(self._last_frame, self._last_fps, self._last_orig_shape)

    @pyqtSlot(object, float, object)
    def update_frame(self, frame: np.ndarray, fps: float = 0.0, orig_shape: tuple[int, int] | None = None) -> None:
        if frame is None:
            return

        # Cache for re-render on resize
        self._last_frame      = frame
        self._last_fps        = fps
        self._last_orig_shape = orig_shape
        self._frames += 1   # count real new frames only (not resize re-renders)

        self._render(frame, fps, orig_shape)

    def _render(self, frame: np.ndarray, fps: float, orig_shape: tuple[int, int] | None) -> None:
        """Convert frame to pixmap and push to the display label."""
        if frame is None:
            return

        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        h, w = frame.shape[:2]

        # 1. Convert grayscale and color to RGB to ensure absolute display compatibility.
        # Calling .copy() is CRITICAL here to force a deep-copy of the pixel buffer. Without it,
        # PySide6 wraps the temporary numpy array's data pointer, which is garbage collected immediately,
        # resulting in corrupted color rendering, washed-out tones, and visual noise.
        if frame.ndim == 2:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            qt_img = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        elif frame.ndim == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        else:
            return

        # 2. Viewport-Matching Scaling Bypass: Skip expensive QPixmap.scaled() if viewport matches frame dimensions
        # Use high-quality SmoothTransformation (bilinear interpolation) instead of blocky FastTransformation
        # to eliminate aliasing, jaggies, and pixelation on the watch dial and fine arm details!
        disp_size = self._display.size()
        if w == disp_size.width() and h == disp_size.height():
            pixmap = QPixmap.fromImage(qt_img)
        else:
            pixmap = QPixmap.fromImage(qt_img).scaled(
                disp_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Draw ROI overlay on top of the scaled pixmap (before display)
        if self._roi_preview is not None:
            pixmap = self._draw_roi_overlay(pixmap, w, h)

        self._display.setPixmap(pixmap)
        disp_w = orig_shape[0] if orig_shape else w
        disp_h = orig_shape[1] if orig_shape else h
        self._lbl_res.setText(f"{disp_w}×{disp_h}")
        self._lbl_fps.setText(f"{fps:.1f} FPS")
        # Track live frame dimensions for accurate overlay aspect-ratio mapping
        self._frame_w = disp_w
        self._frame_h = disp_h

    def _draw_roi_overlay(self, pixmap: QPixmap, frame_w: int, frame_h: int) -> QPixmap:
        """
        Draw a cyan ROI cut-line rectangle on top of the pixmap.

        Coordinate mapping:
          - self._active_roi = sensor-space region currently streaming
            (e.g. (400, 0, 1648, 1536) after a previous ROI was applied).
          - self._roi_preview = new sensor-space ROI the user is previewing.
          - The displayed frame represents ONLY the active_roi region.
          - We map the preview ROI relative to the active ROI so the border
            appears at the correct visual position in the current frame.
        """
        if self._roi_preview is None:
            return pixmap
        ox, oy, rw, rh = self._roi_preview
        aox, aoy, aw, ah = self._active_roi

        # No overlay if new ROI exactly matches the active ROI
        if (ox, oy, rw, rh) == (aox, aoy, aw, ah):
            return pixmap

        pw = pixmap.width()
        ph = pixmap.height()

        # Use ACTUAL FRAME aspect ratio (not fixed sensor) for letterbox calculation.
        # After ROI is applied, frame_w/h reflect the cropped dimensions.
        frame_aspect = self._frame_w / max(self._frame_h, 1)
        pixmap_aspect = pw / max(ph, 1)

        if frame_aspect > pixmap_aspect:
            img_w = pw
            img_h = int(pw / frame_aspect)
            img_x = 0
            img_y = (ph - img_h) // 2
        else:
            img_h = ph
            img_w = int(ph * frame_aspect)
            img_x = (pw - img_w) // 2
            img_y = 0

        # Map new ROI from SENSOR space to DISPLAY space, using active ROI as origin.
        # The displayed frame covers sensor region [aox, aox+aw] x [aoy, aoy+ah].
        sx = img_w / max(aw, 1)   # pixels per sensor-pixel (x)
        sy = img_h / max(ah, 1)   # pixels per sensor-pixel (y)

        rx    = img_x + int((ox - aox) * sx)
        ry    = img_y + int((oy - aoy) * sy)
        rw_px = int(rw * sx)
        rh_px = int(rh * sy)

        # Clamp to image bounds (preview can extend outside current frame)
        rx    = max(img_x, min(rx, img_x + img_w))
        ry    = max(img_y, min(ry, img_y + img_h))
        rw_px = max(0, min(rw_px, img_x + img_w - rx))
        rh_px = max(0, min(rh_px, img_y + img_h - ry))

        result = pixmap.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Dim everything OUTSIDE the ROI with a dark translucent overlay
        painter.setBrush(QBrush(QColor(0, 0, 0, 120)))
        painter.setPen(Qt.PenStyle.NoPen)
        # Top strip
        if ry > img_y:
            painter.drawRect(img_x, img_y, img_w, ry - img_y)
        # Bottom strip
        bot = ry + rh_px
        if bot < img_y + img_h:
            painter.drawRect(img_x, bot, img_w, (img_y + img_h) - bot)
        # Left strip
        if rx > img_x:
            painter.drawRect(img_x, ry, rx - img_x, rh_px)
        # Right strip
        right = rx + rw_px
        if right < img_x + img_w:
            painter.drawRect(right, ry, (img_x + img_w) - right, rh_px)

        # Dashed cyan border around ROI
        pen = QPen(QColor("#06b6d4"))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rx, ry, rw_px, rh_px)

        # Corner tick marks for clarity
        tick = 12
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setWidth(3)
        painter.setPen(pen)
        for cx, cy, dx, dy in [
            (rx,           ry,           1,  1),
            (rx + rw_px,   ry,          -1,  1),
            (rx,           ry + rh_px,   1, -1),
            (rx + rw_px,   ry + rh_px,  -1, -1),
        ]:
            painter.drawLine(cx, cy, cx + dx * tick, cy)
            painter.drawLine(cx, cy, cx, cy + dy * tick)

        # Label: 'W×H @ (X,Y)' near top-left of ROI
        from PyQt6.QtGui import QFontMetrics
        label_text = f"{rw}×{rh} @ ({ox},{oy})"
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(label_text) + 8
        th = fm.height() + 4
        lx = rx + 4
        ly = ry + 4
        # Background pill
        painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
        pen2 = QPen(QColor("#06b6d4"))
        pen2.setWidth(1)
        painter.setPen(pen2)
        painter.drawRoundedRect(lx - 2, ly - 2, tw, th, 3, 3)
        painter.setPen(QColor("#06b6d4"))
        painter.drawText(lx + 2, ly + fm.ascent(), label_text)

        painter.end()
        return result

    def set_roi_preview(self, ox: int, oy: int, w: int, h: int) -> None:
        """Update the ROI overlay rectangle. Called by main_window on spinbox change."""
        self._roi_preview = (ox, oy, w, h)

    def set_active_roi(self, ox: int, oy: int, w: int, h: int) -> None:
        """
        Update the active ROI — the sensor-space region the camera is currently
        streaming. Called after firmware confirms a new ROI so subsequent overlays
        map against the correct frame content instead of the full 2048x1536 sensor.
        """
        self._active_roi = (ox, oy, w, h)

    def clear_roi_preview(self) -> None:
        """Remove the ROI overlay (applied — no longer previewing)."""
        self._roi_preview = None

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

    def update_channel_frame(self, ch_idx: int, frame: np.ndarray, fps: float = 0.0, orig_shape: tuple[int, int] | None = None) -> None:
        """Update a single channel panel with an annotated frame (e.g. from inference)."""
        if 0 <= ch_idx < len(self._panels):
            self._panels[ch_idx].update_frame(frame, fps, orig_shape)

    def reset_all(self) -> None:
        for panel in self._panels:
            panel.reset()

    def set_roi_preview(self, ox: int, oy: int, w: int, h: int) -> None:
        """Push ROI overlay to all 3 channel panels simultaneously."""
        for panel in self._panels:
            panel.set_roi_preview(ox, oy, w, h)

    def set_active_roi(self, ox: int, oy: int, w: int, h: int) -> None:
        """Update active ROI on all 3 panels after firmware confirms an apply."""
        for panel in self._panels:
            panel.set_active_roi(ox, oy, w, h)

    def clear_roi_preview(self) -> None:
        """Remove ROI overlay from all panels."""
        for panel in self._panels:
            panel.clear_roi_preview()
