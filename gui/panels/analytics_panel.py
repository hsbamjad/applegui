"""
gui/panels/analytics_panel.py
==============================
Analytics tab — two live PyQtGraph charts:

  Left  — Grade Distribution (bar chart, Fresh / Processing / Cull)
  Right — Throughput Over Time (line chart, rolling apples/min window)

Public API
----------
  panel.push_grade(grade: str, apples_per_min: int) -> None
      Called by MainWindow every time an apple is committed. Instantly
      updates both charts.

  panel.reset() -> None
      Clears all data — called when the pipeline is stopped.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QVBoxLayout, QWidget, QFrame,
)

from gui.styles import (
    BG_BASE, BG_ELEVATED, BORDER,
    DANGER, SUCCESS, TEXT_2, TEXT_3,
)

# ── PyQtGraph global defaults ─────────────────────────────────────────────────
pg.setConfigOptions(antialias=True, useOpenGL=False)

# ── Colour constants ──────────────────────────────────────────────────────────
_FRESH_COLOR      = SUCCESS          # "#10B981"
_PROC_COLOR       = "#F59E0B"        # amber  (matches camera panel CH1 color)
_CULL_COLOR       = DANGER           # "#EF4444"
_THROUGHPUT_COLOR = "#38BDF8"        # sky-blue — distinct from the grade bars

# Grade → colour map
_GRADE_COLORS = {
    "Fresh":      _FRESH_COLOR,
    "Processing": _PROC_COLOR,
    "Cull":       _CULL_COLOR,
}
_GRADE_ORDER = ["Fresh", "Processing", "Cull"]


def _hex_to_pg(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Convert #RRGGBB string to (R, G, B, A) tuple for PyQtGraph."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


def _pg_color(hex_color: str, alpha: int = 255) -> QColor:
    r, g, b, a = _hex_to_pg(hex_color, alpha)
    return QColor(r, g, b, a)


def _style_plot(plot: pg.PlotWidget, title: str, y_label: str) -> None:
    """Apply consistent dark-theme styling to a PlotWidget."""
    plot.setBackground(BG_BASE)
    plot.setTitle(
        f'<span style="color:{TEXT_2}; font-size:9pt; font-weight:600; '
        f'letter-spacing:1.5px">{title.upper()}</span>',
    )
    plot.showGrid(x=True, y=True, alpha=0.15)

    for axis_name in ("left", "bottom"):
        ax = plot.getAxis(axis_name)
        ax.setPen(pg.mkPen(color=BORDER, width=1))
        ax.setTextPen(pg.mkPen(color=TEXT_3))
        ax.setStyle(tickFont=QFont("Segoe UI", 8))

    plot.setLabel("left", y_label,
                  **{"color": TEXT_2, "font-size": "8pt"})
    plot.getPlotItem().layout.setContentsMargins(8, 6, 8, 6)


# ── Grade Distribution chart ──────────────────────────────────────────────────

class GradeDistributionChart(QWidget):
    """
    Vertical bar chart — one bar per grade class.
    Bars animate (height only — no x-axis shuffle) on every push_grade().
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._counts = {g: 0 for g in _GRADE_ORDER}
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._plot = pg.PlotWidget()
        _style_plot(self._plot, "Grade Distribution", "Count")

        # X-axis: categorical ticks centred at 0, 1, 2
        ax = self._plot.getAxis("bottom")
        ax.setTicks([[(0, "Fresh"), (1, "Processing"), (2, "Cull")]])

        # Disable x-axis auto-range; fix y minimum at 0
        self._plot.setXRange(-0.6, 2.6, padding=0)
        self._plot.setYRange(0, 10, padding=0.05)

        # Create bars — one per grade
        bar_width  = 0.55
        self._bars: list[pg.BarGraphItem] = []
        for i, grade in enumerate(_GRADE_ORDER):
            color   = _GRADE_COLORS[grade]
            fill    = _pg_color(color, 200)
            outline = _pg_color(color, 255)
            bar = pg.BarGraphItem(
                x=[i], height=[0], width=bar_width,
                brush=fill, pen=pg.mkPen(outline, width=1.5),
            )
            self._plot.addItem(bar)
            self._bars.append(bar)

        # Value labels on top of each bar
        self._value_labels: list[pg.TextItem] = []
        for i, grade in enumerate(_GRADE_ORDER):
            color = _GRADE_COLORS[grade]
            lbl = pg.TextItem(
                text="0",
                color=_pg_color(color),
                anchor=(0.5, 1.0),
            )
            lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            lbl.setPos(i, 0)
            self._plot.addItem(lbl)
            self._value_labels.append(lbl)

        # Horizontal reference line at 0
        zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(color=BORDER, width=1, style=Qt.PenStyle.SolidLine),
        )
        self._plot.addItem(zero_line)

        layout.addWidget(self._plot)

    def push_grade(self, grade: str) -> None:
        if grade not in self._counts:
            return
        self._counts[grade] += 1
        self._refresh()

    def reset(self) -> None:
        self._counts = {g: 0 for g in _GRADE_ORDER}
        self._refresh()

    def _refresh(self) -> None:
        max_count = max(max(self._counts.values()), 10)
        self._plot.setYRange(0, max_count * 1.18, padding=0)

        for i, grade in enumerate(_GRADE_ORDER):
            n = self._counts[grade]
            self._bars[i].setOpts(height=[n])
            self._value_labels[i].setText(str(n))
            self._value_labels[i].setPos(i, n)


# ── Throughput Over Time chart ────────────────────────────────────────────────

class ThroughputChart(QWidget):
    """
    Rolling line chart of apples-per-minute, updated on each grade commit.

    The x-axis shows elapsed seconds since the session started.
    A deque of (elapsed_sec, apples_per_min) pairs is maintained with a
    configurable max length (default 120 data points).
    """

    MAX_POINTS = 120

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._t0: float = time.monotonic()
        self._data: Deque[tuple[float, float]] = deque(maxlen=self.MAX_POINTS)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._plot = pg.PlotWidget()
        _style_plot(self._plot, "Throughput Over Time", "Apples / min")
        self._plot.setLabel("bottom", "Elapsed (s)",
                            **{"color": TEXT_2, "font-size": "8pt"})
        self._plot.setXRange(0, 60, padding=0)
        self._plot.setYRange(0, 20, padding=0.05)

        # Gradient fill under the line
        fill_color = _pg_color(_THROUGHPUT_COLOR, 40)
        self._fill = pg.FillBetweenItem(
            curve1=None, curve2=None, brush=fill_color,
        )

        # Main throughput line
        pen = pg.mkPen(
            color=_pg_color(_THROUGHPUT_COLOR),
            width=2.5,
            style=Qt.PenStyle.SolidLine,
        )
        self._curve = self._plot.plot([], [], pen=pen, symbol=None)

        # Shaded fill below the line
        zero_curve = self._plot.plot([], [], pen=None)
        self._fill = pg.FillBetweenItem(self._curve, zero_curve, brush=fill_color)
        self._plot.addItem(self._fill)
        self._zero_curve = zero_curve

        # Moving dot on latest point
        self._dot = self._plot.plot(
            [], [],
            pen=None,
            symbol="o",
            symbolSize=8,
            symbolBrush=_pg_color(_THROUGHPUT_COLOR),
            symbolPen=pg.mkPen(_pg_color(_THROUGHPUT_COLOR, 180), width=2),
        )

        # Current-value text label top-right
        self._val_lbl = pg.TextItem(
            text="",
            color=_pg_color(_THROUGHPUT_COLOR),
            anchor=(1.0, 0.0),
        )
        self._val_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._plot.addItem(self._val_lbl)

        layout.addWidget(self._plot)

    def push_grade(self, apples_per_min: int) -> None:
        elapsed = time.monotonic() - self._t0
        self._data.append((elapsed, float(apples_per_min)))
        self._refresh()

    def reset(self) -> None:
        self._t0 = time.monotonic()
        self._data.clear()
        self._refresh()

    def _refresh(self) -> None:
        if not self._data:
            self._curve.setData([], [])
            self._zero_curve.setData([], [])
            self._dot.setData([], [])
            self._val_lbl.setText("")
            return

        xs = [p[0] for p in self._data]
        ys = [p[1] for p in self._data]

        self._curve.setData(xs, ys)
        self._zero_curve.setData(xs, [0.0] * len(xs))
        self._dot.setData([xs[-1]], [ys[-1]])

        # Adjust x-axis to show a rolling 60-s window (or full if shorter)
        x_end   = max(xs[-1] + 2, 60)
        x_start = max(0.0, x_end - 60)
        self._plot.setXRange(x_start, x_end, padding=0)

        # Adjust y-axis
        y_max = max(max(ys) * 1.25, 20)
        self._plot.setYRange(0, y_max, padding=0)

        # Position value label near top-right of the visible window
        self._val_lbl.setPos(x_end - 0.5, y_max * 0.92)
        self._val_lbl.setText(f"{ys[-1]:.0f} /min")


# ── Legend strip ──────────────────────────────────────────────────────────────

def _legend_strip() -> QWidget:
    """Compact horizontal legend row shown above both charts."""
    w = QWidget()
    w.setStyleSheet(f"background: {BG_ELEVATED}; border-radius: 6px; border: none;")
    hl = QHBoxLayout(w)
    hl.setContentsMargins(12, 6, 12, 6)
    hl.setSpacing(18)

    items = [
        ("Fresh",      _FRESH_COLOR),
        ("Processing", _PROC_COLOR),
        ("Cull",       _CULL_COLOR),
        ("Throughput", _THROUGHPUT_COLOR),
    ]
    for name, color in items:
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {color}; font-size: 10px; background: transparent;")
        lbl = QLabel(name)
        lbl.setStyleSheet(f"color: {TEXT_2}; font-size: 10px; background: transparent;")
        hl.addWidget(dot)
        hl.addWidget(lbl)

    hl.addStretch()

    # Session stats labels (updated externally)
    _sess_lbl = QLabel("Session:  0 graded")
    _sess_lbl.setObjectName("analytics_session_lbl")
    _sess_lbl.setStyleSheet(
        f"color: {TEXT_3}; font-size: 10px; background: transparent; font-style: italic;"
    )
    hl.addWidget(_sess_lbl)
    w._sess_lbl = _sess_lbl  # type: ignore[attr-defined]

    return w


# ── Main panel ────────────────────────────────────────────────────────────────

class AnalyticsPanel(QWidget):
    """
    Analytics tab content — two live charts side by side with a legend strip.

    Usage
    -----
    Call `push_grade(grade, apples_per_min)` on every committed grade event.
    Call `reset()` when the pipeline stops.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._total = 0
        self.setStyleSheet(f"background-color: {BG_BASE};")
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Legend / session strip ─────────────────────────────────────────
        self._legend_w = _legend_strip()
        root.addWidget(self._legend_w)

        # ── Divider ───────────────────────────────────────────────────────
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {BORDER}; border: none;")
        root.addWidget(div)

        # ── Charts row ────────────────────────────────────────────────────
        charts = QHBoxLayout()
        charts.setContentsMargins(0, 0, 0, 0)
        charts.setSpacing(6)

        self.grade_chart      = GradeDistributionChart()
        self.throughput_chart = ThroughputChart()

        charts.addWidget(self.grade_chart,      stretch=1)

        vsep = QFrame()
        vsep.setFrameShape(QFrame.Shape.VLine)
        vsep.setStyleSheet(f"color: {BORDER}; background: {BORDER}; max-width: 1px; border: none;")
        charts.addWidget(vsep)

        charts.addWidget(self.throughput_chart, stretch=1)

        root.addLayout(charts, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def push_grade(self, grade: str, apples_per_min: int) -> None:
        """Update both charts. Call from MainWindow._on_inference_result()."""
        self._total += 1
        self.grade_chart.push_grade(grade)
        self.throughput_chart.push_grade(apples_per_min)
        self._legend_w._sess_lbl.setText(f"Session:  {self._total} graded")

    def reset(self) -> None:
        """Clear all data — call from MainWindow._stop_pipeline()."""
        self._total = 0
        self.grade_chart.reset()
        self.throughput_chart.reset()
        self._legend_w._sess_lbl.setText("Session:  0 graded")
