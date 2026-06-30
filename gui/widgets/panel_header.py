"""
gui/widgets/panel_header.py
============================
Shared accent bar + title row for center-panel tabs (AI Model Input,
Analytics, Logs).  One visual language: green accent, surface header.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt

from gui.styles import ACCENT, BG_SURFACE, TEXT_1, TEXT_2, TEXT_3, BORDER


class PanelHeaderBar(QWidget):
    """
    3 px accent stripe + 32 px header row::

        ◆  AI MODEL INPUT  ·  subtitle text …………  [right widget]
    """

    def __init__(
        self,
        icon: str,
        title: str,
        subtitle: str = "",
        right: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._right: QWidget | None = None
        self._build(icon, title, subtitle, right)

    def _build(
        self,
        icon: str,
        title: str,
        subtitle: str,
        right: QWidget | None,
    ) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        accent = QWidget()
        accent.setFixedHeight(3)
        accent.setStyleSheet(f"background-color: {ACCENT}; border: none;")
        root.addWidget(accent)

        hdr = QWidget()
        hdr.setFixedHeight(32)
        hdr.setStyleSheet(
            f"background-color: {BG_SURFACE}; border-bottom: 1px solid {BORDER};"
        )
        self._hdr_layout = QHBoxLayout(hdr)
        self._hdr_layout.setContentsMargins(12, 0, 12, 0)
        self._hdr_layout.setSpacing(8)

        self._icon = QLabel(icon)
        self._icon.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; background: transparent;"
        )

        self._title = QLabel(title.upper())
        self._title.setStyleSheet(
            f"color: {TEXT_1}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 1.2px; background: transparent;"
        )

        self._sep = QLabel("·")
        self._sep.setStyleSheet(f"color: {TEXT_3}; background: transparent;")

        self._subtitle = QLabel(subtitle)
        self._subtitle.setStyleSheet(
            f"color: {TEXT_2}; font-size: 10px; background: transparent;"
        )

        self._hdr_layout.addWidget(self._icon)
        self._hdr_layout.addWidget(self._title)
        self._hdr_layout.addWidget(self._sep)
        self._hdr_layout.addWidget(self._subtitle)
        self._hdr_layout.addStretch()

        self.set_subtitle(subtitle)
        self.set_right_widget(right)

        root.addWidget(hdr)

    def subtitle_label(self) -> QLabel:
        return self._subtitle

    def set_subtitle(self, text: str) -> None:
        self._subtitle.setText(text)
        visible = bool(text)
        self._sep.setVisible(visible)
        self._subtitle.setVisible(visible)

    def set_right_widget(self, widget: QWidget | None) -> None:
        if self._right is not None:
            self._hdr_layout.removeWidget(self._right)
            self._right.setParent(None)
            self._right = None
        if widget is not None:
            self._right = widget
            self._hdr_layout.addWidget(
                widget, alignment=Qt.AlignmentFlag.AlignVCenter,
            )
