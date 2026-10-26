"""
gui/light_tooltip.py
====================
Custom light info popups that replace native QToolTip.

Windows (esp. dark-mode OS themes) often forces native tooltips black and
ignores QSS/palette. Intercept Tip events and show our own framed label.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, QTimer, Qt
from PyQt6.QtGui import QCursor, QHelpEvent
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from gui.styles import BG_CARD, BORDER_LT, TEXT_1


class LightToolTip(QLabel):
    """Framed light info box used instead of native QToolTip."""

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.ToolTip
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("light_tooltip")
        self.setWordWrap(True)
        self.setMaximumWidth(360)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(f"""
            QLabel#light_tooltip {{
                background-color: {BG_CARD};
                color: {TEXT_1};
                border: 1px solid {BORDER_LT};
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 11px;
                font-weight: 400;
            }}
        """)
        self.hide()

    def show_text(self, text: str, global_pos: QPoint) -> None:
        self.setText(text)
        self.adjustSize()
        self.move(global_pos + QPoint(14, 18))
        self.show()
        self.raise_()


class LightToolTipFilter(QObject):
    """App-wide filter: replace native tooltips with LightToolTip."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tip = LightToolTip()
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._tip.hide)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        et = event.type()

        if et == QEvent.Type.ToolTip and isinstance(obj, QWidget):
            text = obj.toolTip()
            if not text:
                self._tip.hide()
                return True
            self._hide_timer.stop()
            pos = QCursor.pos()
            if isinstance(event, QHelpEvent):
                pos = event.globalPos()
            self._tip.show_text(text, pos)
            return True  # suppress native (black) tooltip

        if et in (
            QEvent.Type.Leave,
            QEvent.Type.WindowDeactivate,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.Wheel,
            QEvent.Type.KeyPress,
        ):
            self._tip.hide()

        return False


def install_light_tooltips(app: QApplication) -> LightToolTipFilter:
    """Install the custom light tooltip filter on the QApplication."""
    filt = LightToolTipFilter(app)
    app.installEventFilter(filt)
    return filt
