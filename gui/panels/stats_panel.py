"""
gui/panels/stats_panel.py
=========================
Right statistics panel - live grading dashboard.

Cards:
  System Status - 4 subsystem health pills
  Grade Summary - Fresh / Processing / Cull live counts + mini bar
  Recent Results - scrollable list: #ID  Lane  Grade → Outlet  Conf
  Live Metrics  - conveyor speed, throughput, total, session time
"""

from __future__ import annotations

from datetime import timedelta

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QColor

from gui.styles import (
    BG_SURFACE, BG_CARD, BG_ELEVATED,
    ACCENT, SUCCESS, WARNING, DANGER,
    TEXT_1, TEXT_2, TEXT_3, BORDER,
)

PANEL_WIDTH = 256

GRADE_COLORS = {
    "Fresh":      SUCCESS,
    "Processing": ACCENT,
    "Cull":       DANGER,
}

OUTLET_COLORS = {
    "A": SUCCESS,
    "B": ACCENT,
    "C": DANGER,
}

# Pill badge backgrounds for system status
PILL = {
    "online":  {"bg": "#162E1A", "text": SUCCESS},
    "offline": {"bg": "#2E1310", "text": DANGER},
    "warning": {"bg": "#2E2210", "text": WARNING},
    "idle":    {"bg": "transparent", "text": TEXT_3},
}


def _section_header(text: str) -> QWidget:
    w = QWidget()
    hl = QHBoxLayout(w)
    hl.setContentsMargins(2, 10, 2, 2)
    hl.setSpacing(8)
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {TEXT_3}; font-size: 10px; font-weight: 700; "
        "letter-spacing: 2px; background: transparent;"
    )
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background-color: {BORDER}; max-height: 1px; border: none;")
    line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    hl.addWidget(lbl)
    hl.addWidget(line)
    return w


def _card_style() -> str:
    return (
        f"background-color: {BG_CARD}; border-radius: 8px; "
        f"border: 1px solid {BORDER};"
    )


# ── System Status Card ────────────────────────────────────────────────────────

class SystemStatusCard(QWidget):
    SUBSYSTEMS = ["Camera", "AI Model", "Sorter", "Logger"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_card_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self._dots:   dict[str, QLabel] = {}
        self._labels: dict[str, QLabel] = {}

        for name in self.SUBSYSTEMS:
            row = QHBoxLayout()
            row.setSpacing(7)
            dot = QLabel()
            dot.setFixedSize(8, 8)
            self._dots[name] = dot
            lbl = QLabel(f"{name}: Offline")
            lbl.setWordWrap(False)
            lbl.setStyleSheet(
                f"color: {TEXT_3}; font-size: 11px; font-weight: 500; background: transparent;"
            )
            self._labels[name] = lbl
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            layout.addLayout(row)
            self._apply_dot(name, DANGER)

    def _apply_dot(self, name: str, color: str) -> None:
        self._dots[name].setStyleSheet(
            f"background-color: {color}; border-radius: 4px; border: none;"
        )

    def set_status(self, subsystem: str, state: str, detail: str = "") -> None:
        s = PILL.get(state, PILL["idle"])
        dot_colors = {"online": SUCCESS, "offline": DANGER, "warning": WARNING, "idle": TEXT_2}
        suffix = detail or state.capitalize()

        if subsystem in self._dots:
            self._apply_dot(subsystem, dot_colors.get(state, TEXT_2))
            pb = s["bg"]
            pt = s["text"]
            # idle state: no pill, but use TEXT_2 (lighter) for visibility
            if pb == "transparent":
                pt = TEXT_2
            bdr = f"border: 1px solid {pt}30;" if pb != "transparent" else ""
            rdx = "border-radius: 4px;" if pb != "transparent" else ""
            pad = "padding: 1px 6px;" if pb != "transparent" else ""

            self._labels[subsystem].setText(f"{subsystem}: {suffix}")
            self._labels[subsystem].setStyleSheet(
                f"color: {pt}; font-size: 11px; font-weight: 600; "
                f"background-color: {pb}; {bdr} {rdx} {pad}"
            )


# ── Grade Summary Card ────────────────────────────────────────────────────────

class GradeSummaryCard(QWidget):
    """Live count of Fresh / Processing / Cull with colored labels."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_card_style())
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(0)

        self._counts = {"Fresh": 0, "Processing": 0, "Cull": 0}
        self._widgets: dict[str, QLabel] = {}

        for grade, color in GRADE_COLORS.items():
            col = QVBoxLayout()
            col.setSpacing(2)
            val = QLabel("0")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet(
                f"color: {color}; font-size: 20px; font-weight: 700; background: transparent;"
            )
            lbl = QLabel(grade[:4])
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"color: {TEXT_3}; font-size: 9px; font-weight: 600; "
                f"letter-spacing: 1px; background: transparent;"
            )
            col.addWidget(val)
            col.addWidget(lbl)
            layout.addLayout(col)
            if grade != "Cull":
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet(f"color: {BORDER}; background: {BORDER}; max-width: 1px; border: none;")
                layout.addWidget(sep)
            self._widgets[grade] = val

    def record(self, grade: str) -> None:
        if grade in self._counts:
            self._counts[grade] += 1
            self._widgets[grade].setText(str(self._counts[grade]))

    def reset(self) -> None:
        for g in self._counts:
            self._counts[g] = 0
            self._widgets[g].setText("0")


# ── Recent Results Card ───────────────────────────────────────────────────────

class RecentResultsCard(QWidget):
    MAX_ITEMS = 80

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_card_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        self._list = QListWidget()
        self._list.setFixedHeight(180)
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setStyleSheet(f"""
            QListWidget {{
                background: transparent;
                border: none;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }}
            QListWidget::item {{
                padding: 4px 6px;
                border-bottom: 1px solid {BORDER};
                color: {TEXT_1};
                background: transparent;
            }}
            QListWidget::item:hover {{ background-color: transparent; }}
        """)
        layout.addWidget(self._list)

        placeholder = QListWidgetItem("  Waiting for grades…")
        placeholder.setForeground(QColor(TEXT_3))
        self._list.addItem(placeholder)

    def add_result(
        self,
        apple_id:   int,
        lane:       int,
        grade:      str,
        confidence: float,
        outlet:     str,
        size_mm:    float | None = None,
    ) -> None:
        if self._list.count() == 1 and "Waiting" in (self._list.item(0).text() or ""):
            self._list.clear()

        g_color = GRADE_COLORS.get(grade, TEXT_2)

        # Row format (fits 256px panel at 11px Consolas):
        #   #0011 L1 Cull  →C  94%  72mm
        grade_short = grade[:5]     # Fresh / Proc / Cull
        size_str    = f"{size_mm:.0f}mm" if size_mm is not None else " - "
        text = (
            f"#{apple_id:04d} L{lane} {grade_short:<5}"
            f" \u2192{outlet}  {confidence * 100:.0f}%  {size_str}"
        )

        item = QListWidgetItem(text)
        item.setForeground(QColor(g_color))
        self._list.insertItem(0, item)

        while self._list.count() > self.MAX_ITEMS:
            self._list.takeItem(self._list.count() - 1)

    def clear_results(self) -> None:
        self._list.clear()


# ── Metric Item ───────────────────────────────────────────────────────────────

class MetricItem(QWidget):
    def __init__(self, label: str, default: str, color: str = ACCENT, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {TEXT_3}; font-size: 9px; font-weight: 700; "
            f"letter-spacing: 1px; background: transparent;"
        )
        self._val = QLabel(default)
        self._val.setStyleSheet(
            f"color: {color}; font-size: 16px; font-weight: 700; background: transparent;"
        )
        layout.addWidget(lbl)
        layout.addWidget(self._val)

    def set_value(self, text: str) -> None:
        self._val.setText(text)


class MetricsCard(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_card_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self._speed      = MetricItem("THROUGHPUT",     "-- apple/min",  SUCCESS)
        self._total      = MetricItem("TOTAL GRADED",   "0",             TEXT_1)
        self._infer_fps  = MetricItem("INFER FPS",      "-- FPS",        ACCENT)
        self._view_fps   = MetricItem("VIEW FPS",      "-- FPS",        TEXT_2)
        self._session    = MetricItem("SESSION TIME",   "00:00:00",      TEXT_2)

        for m in [self._speed, self._total, self._infer_fps, self._view_fps, self._session]:
            layout.addWidget(m)
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"background-color: {BORDER}; max-height: 1px; border: none;")
            layout.addWidget(sep)

        self._elapsed = 0
        self._total_n = 0
        self._timer   = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        self._elapsed += 1
        h = self._elapsed // 3600
        m = (self._elapsed % 3600) // 60
        s = self._elapsed % 60
        self._session.set_value(f"{h:02d}:{m:02d}:{s:02d}")

    def start_session(self) -> None:
        self._elapsed = 0
        self._total_n = 0
        self._timer.start(1000)

    def stop_session(self) -> None:
        self._timer.stop()

    def reset(self) -> None:
        self.stop_session()
        self._elapsed = 0
        self._total_n = 0
        self._speed.set_value("-- apple/min")
        self._total.set_value("0")
        self._infer_fps.set_value("-- FPS")
        self._view_fps.set_value("-- FPS")
        self._session.set_value("00:00:00")

    def record_grade(self) -> None:
        """Called each time a new grade is recorded - only increments total count.
        Throughput display is driven by set_throughput() from the real measurement."""
        self._total_n += 1
        self._total.set_value(str(self._total_n))

    def set_throughput(self, apm: float) -> None:
        """Update THROUGHPUT display from the real timestamp-based APM value."""
        self._speed.set_value(f"{apm:.0f} apple/min")

    def set_infer_fps(self, fps: float) -> None:
        """YOLO pipeline throughput (1 s avg) - how fast AI processes frames."""
        self._infer_fps.set_value(f"{fps:.1f} FPS")

    def set_view_fps(self, fps: float) -> None:
        """Model-input panel repaint rate - how smooth the AI video looks."""
        if fps > 0:
            self._view_fps.set_value(f"{fps:.1f} FPS")


# ── Right Stats Panel ─────────────────────────────────────────────────────────

class RightStatsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(PANEL_WIDTH)
        self.setStyleSheet(f"background-color: {BG_SURFACE};")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 12)
        layout.setSpacing(2)

        layout.addWidget(_section_header("System Status"))
        self.status_group  = SystemStatusCard(self)
        layout.addWidget(self.status_group)

        layout.addWidget(_section_header("Grade Summary"))
        self.grade_summary = GradeSummaryCard(self)
        layout.addWidget(self.grade_summary)

        layout.addWidget(_section_header("Recent Results"))
        self.results_group = RecentResultsCard(self)
        layout.addWidget(self.results_group)

        layout.addWidget(_section_header("Live Metrics"))
        self.metrics_group = MetricsCard(self)
        layout.addWidget(self.metrics_group)

        layout.addStretch()
