"""
gui/handlers/log_handler.py
============================
Thread-safe bridge from Python logging → Qt signals for the Logs panel.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal


_LOG_FMT     = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def _normalize_dashes(text: str) -> str:
    """Replace em/en dashes with ASCII hyphen for display."""
    return text.replace("\u2014", "-").replace("\u2013", "-")


class LogEmitter(QObject):
    """Carries formatted log lines to the GUI thread via queued signal."""

    record_emitted = pyqtSignal(str, int)   # message, levelno


class QtLogHandler(logging.Handler):
    """logging.Handler that forwards records to a LogEmitter (any thread)."""

    def __init__(self, emitter: LogEmitter) -> None:
        super().__init__(level=logging.INFO)
        self._emitter = emitter
        self.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = _normalize_dashes(self.format(record))
            self._emitter.record_emitted.emit(msg, record.levelno)
        except Exception:
            self.handleError(record)


def attach_gui_log_handler(emitter: LogEmitter) -> QtLogHandler:
    """Register a Qt handler on the root logger (safe to call once)."""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, QtLogHandler):
            return h
    handler = QtLogHandler(emitter)
    root.addHandler(handler)
    return handler
