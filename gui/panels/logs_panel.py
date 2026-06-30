"""
gui/panels/logs_panel.py
========================
System Logs tab - live view of application logging (INFO, WARNING, ERROR, …).
"""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor, QFont, QTextCharFormat, QColor

from gui.handlers.log_handler import LogEmitter, attach_gui_log_handler
from gui.widgets.panel_header import PanelHeaderBar
from gui.styles import (
    BG_BASE, BG_ELEVATED,
    ACCENT, WARNING, DANGER,
    TEXT_1, TEXT_2, TEXT_3, BORDER, BORDER_LT,
)

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG:    TEXT_3,
    logging.INFO:     TEXT_1,
    logging.WARNING:  WARNING,
    logging.ERROR:    DANGER,
    logging.CRITICAL: DANGER,
}

_MAX_LINES = 2_500


class LogsPanel(QWidget):
    """Read-only scrolling log viewer wired to the root Python logger."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._line_count = 0
        self._auto_scroll = True
        self._emitter = LogEmitter(self)
        self._handler = attach_gui_log_handler(self._emitter)
        self._emitter.record_emitted.connect(self._on_record)
        self._build()

    def _build(self) -> None:
        self.setStyleSheet(f"background-color: {BG_BASE};")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedHeight(24)
        self._btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_clear.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_ELEVATED}; color: {TEXT_2};
                border: 1px solid {BORDER}; font-weight: 600; font-size: 10px;
                border-radius: 5px; padding: 0 10px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT}; color: #0E1A10; border-color: {ACCENT};
            }}
        """)
        self._btn_clear.clicked.connect(self.clear)

        self._header = PanelHeaderBar(
            "▤",
            "Logs",
            "Live output from all pipeline modules",
            right=self._btn_clear,
        )
        root.addWidget(self._header)

        self._view = QTextEdit()
        self._view.setReadOnly(True)
        self._view.setAcceptRichText(False)
        self._view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        font = QFont("Consolas")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setPointSize(10)
        self._view.setFont(font)
        self._view.document().setDocumentMargin(12)
        self._view.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_BASE};
                color: {TEXT_1};
                border: none;
                selection-background-color: {ACCENT};
            }}
            QScrollBar:vertical {{
                background: transparent; width: 6px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_LT}; border-radius: 3px; min-height: 24px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{
                background: transparent; height: 6px; margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background: {BORDER_LT}; border-radius: 3px; min-width: 24px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        """)
        root.addWidget(self._view, stretch=1)

        self._append_system_line("Log viewer ready.", logging.INFO)

    def _append_line(self, text: str, level: int) -> None:
        color = QColor(_LEVEL_COLORS.get(level, TEXT_1))
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        if level >= logging.WARNING:
            fmt.setFontWeight(QFont.Weight.Bold)

        cursor = self._view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text + "\n", fmt)
        self._view.setTextCursor(cursor)
        self._line_count += 1

        if self._auto_scroll:
            self._view.moveCursor(QTextCursor.MoveOperation.End)
            hbar = self._view.horizontalScrollBar()
            if hbar is not None:
                hbar.setValue(0)

    def _append_system_line(self, text: str, level: int) -> None:
        self._append_line(text, level)

    def _on_record(self, message: str, levelno: int) -> None:
        self._append_line(message, levelno)

        if self._line_count > _MAX_LINES:
            self._trim_old_lines()

        if self._auto_scroll:
            self._view.moveCursor(QTextCursor.MoveOperation.End)
            hbar = self._view.horizontalScrollBar()
            if hbar is not None:
                hbar.setValue(0)

    def _trim_old_lines(self) -> None:
        """Drop oldest lines when the buffer grows too large."""
        doc = self._view.document()
        while doc.blockCount() > _MAX_LINES and doc.firstBlock().isValid():
            cursor = QTextCursor(doc)
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        self._line_count = doc.blockCount()

    def clear(self) -> None:
        self._view.clear()
        self._line_count = 0
        self._append_system_line("Log cleared.", logging.INFO)
