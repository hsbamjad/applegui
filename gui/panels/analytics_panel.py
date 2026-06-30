"""
gui/panels/analytics_panel.py
==============================
Analytics tab - two live PyQtGraph charts:

  Left:  Grade Distribution - bar chart
         Fresh (green) / Processing (amber) / Cull (red) accumulated counts.

  Right: Throughput Over Time - rolling line chart.
         True apples/min measured from a sliding 60-second event window.
         Every grade event appends time.monotonic(); the 1-Hz sampler counts
         how many timestamps fall within the last 60 s.

Public API (called from MainWindow):
  panel.record_grade(grade: str) -> None   # no fake speed arg needed
  panel.start() / stop() / reset()
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

try:
    import pyqtgraph as pg
    _PG_OK = True
except ImportError:          # pragma: no cover
    pg = None                # type: ignore[assignment]
    _PG_OK = False

from gui.styles import (
    BG_BASE, BG_CARD, BG_ELEVATED, BG_SURFACE,
    ACCENT, SUCCESS, WARNING, DANGER,
    TEXT_1, TEXT_2, TEXT_3, BORDER,
)
from gui.widgets.panel_header import PanelHeaderBar

# ── Throughput rolling window (seconds = samples at 1 Hz) ─────────────────────
_WINDOW_S = 60          # keep 60 data-points (≈ 60 s at 1-sample/s)
_TICK_MS   = 1_000      # sample throughput every 1 second

# ── Colour map used by both charts ────────────────────────────────────────────
GRADE_ORDER  = ["Fresh", "Processing", "Cull"]
GRADE_COLORS = {
    "Fresh":      SUCCESS,   # #10B981
    "Processing": WARNING,   # #F59E0B
    "Cull":       DANGER,    # #EF4444
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_to_qcolor(h: str):
    """Convert '#RRGGBB' → pyqtgraph-compatible QColor."""
    from PyQt6.QtGui import QColor
    return QColor(h)


def _section_header(text: str) -> QWidget:
    w  = QWidget()
    hl = QHBoxLayout(w)
    hl.setContentsMargins(2, 6, 2, 4)
    hl.setSpacing(8)
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {TEXT_3}; font-size: 10px; font-weight: 700; "
        "letter-spacing: 2px; background: transparent;"
    )
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(
        f"background-color: {BORDER}; max-height: 1px; border: none;"
    )
    line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    hl.addWidget(lbl)
    hl.addWidget(line)
    return w


# ── Shared PyQtGraph theme config ──────────────────────────────────────────────

def _configure_pg() -> None:
    """Apply global PyQtGraph dark-theme settings once."""
    if not _PG_OK:
        return
    pg.setConfigOptions(
        antialias        = True,
        background       = BG_CARD,
        foreground       = TEXT_2,
        useOpenGL        = False,   # safer on all GPU configs
        crashWarning     = False,
    )


# ── Grade Distribution chart ──────────────────────────────────────────────────

class GradeBarChart(QWidget):
    """
    Horizontal bar chart (or vertical) showing accumulated Fresh / Processing /
    Cull counts.  Updates immediately on every record_grade() call.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._counts: dict[str, int] = {g: 0 for g in GRADE_ORDER}
        self.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 8px; border: none;"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Thin accent bar at top
        top_bar = QWidget()
        top_bar.setFixedHeight(2)
        top_bar.setStyleSheet(f"background-color: {ACCENT}60; border: none;")
        root.addWidget(top_bar)

        inner = QVBoxLayout()
        inner.setContentsMargins(14, 8, 14, 10)
        inner.setSpacing(4)

        hdr = QLabel("GRADE DISTRIBUTION")
        hdr.setStyleSheet(
            f"color: {ACCENT}; font-size: 9px; font-weight: 700; "
            "letter-spacing: 1.5px; background: transparent;"
        )
        inner.addWidget(hdr)

        if _PG_OK:
            self._build_pg(inner)
        else:
            self._build_fallback(inner)

        root.addLayout(inner)

    # ── PyQtGraph build ───────────────────────────────────────────────────────

    def _build_pg(self, layout: QVBoxLayout) -> None:
        _configure_pg()

        self._plot = pg.PlotWidget()
        self._plot.setBackground(BG_CARD)
        self._plot.setMinimumHeight(120)

        # Remove axes clutter
        self._plot.getAxis("top").hide()
        self._plot.getAxis("right").hide()

        # X axis: 3 bars → ticks at 0, 1, 2
        ax_x = self._plot.getAxis("bottom")
        ax_x.setTicks([list(enumerate(["Fresh", "Proc.", "Cull"]))])
        ax_x.setStyle(tickLength=-5, tickFont=None)
        ax_x.setPen(pg.mkPen(BORDER))
        ax_x.setTextPen(pg.mkPen(TEXT_2))

        ax_y = self._plot.getAxis("left")
        ax_y.setPen(pg.mkPen(BORDER))
        ax_y.setTextPen(pg.mkPen(TEXT_2))

        self._plot.showGrid(x=False, y=True, alpha=0.12)
        self._plot.setXRange(-0.5, 2.5, padding=0)
        self._plot.setYRange(0, 1, padding=0.05)
        self._plot.getPlotItem().setContentsMargins(0, 0, 0, 0)

        # One BarGraphItem per grade (so each can have its own color)
        self._bars: list[pg.BarGraphItem] = []
        for i, grade in enumerate(GRADE_ORDER):
            bar = pg.BarGraphItem(
                x      = [i],
                height = [0],
                width  = 0.6,
                brush  = pg.mkBrush(_hex_to_qcolor(GRADE_COLORS[grade])),
                pen    = pg.mkPen(None),
            )
            self._plot.addItem(bar)
            self._bars.append(bar)

        layout.addWidget(self._plot, stretch=1)

        # Legend row
        legend_row = QHBoxLayout()
        legend_row.setSpacing(16)
        for grade in GRADE_ORDER:
            dot = QLabel("●")
            dot.setStyleSheet(
                f"color: {GRADE_COLORS[grade]}; font-size: 10px; background: transparent;"
            )
            self._counts_labels: dict[str, QLabel] = getattr(
                self, "_counts_labels", {}
            )
            cnt = QLabel("0")
            cnt.setStyleSheet(
                f"color: {GRADE_COLORS[grade]}; font-size: 13px; font-weight: 700; "
                "background: transparent;"
            )
            lbl = QLabel(grade)
            lbl.setStyleSheet(
                f"color: {TEXT_3}; font-size: 9px; letter-spacing: 1px; background: transparent;"
            )
            col = QVBoxLayout()
            col.setSpacing(0)
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(dot)
            row.addWidget(cnt)
            row.addStretch()
            col.addLayout(row)
            col.addWidget(lbl)
            legend_row.addLayout(col)
            self._counts_labels[grade] = cnt  # type: ignore[attr-defined]

        legend_row.addStretch()
        layout.addLayout(legend_row)

    # ── Fallback (no PyQtGraph) ───────────────────────────────────────────────

    def _build_fallback(self, layout: QVBoxLayout) -> None:
        self._counts_labels: dict[str, QLabel] = {}
        for grade in GRADE_ORDER:
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(
                f"color: {GRADE_COLORS[grade]}; font-size: 11px; background: transparent;"
            )
            lbl = QLabel(grade)
            lbl.setStyleSheet(f"color: {TEXT_2}; background: transparent;")
            cnt = QLabel("0")
            cnt.setStyleSheet(
                f"color: {GRADE_COLORS[grade]}; font-size: 16px; font-weight: 700; "
                "background: transparent;"
            )
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(cnt)
            layout.addLayout(row)
            self._counts_labels[grade] = cnt

        warn = QLabel("pyqtgraph not installed - chart unavailable")
        warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        warn.setStyleSheet(f"color: {TEXT_3}; font-size: 10px; background: transparent;")
        layout.addWidget(warn)

    # ── Public API ────────────────────────────────────────────────────────────

    def record_grade(self, grade: str) -> None:
        if grade not in self._counts:
            return
        self._counts[grade] += 1
        self._refresh()

    def reset(self) -> None:
        for g in self._counts:
            self._counts[g] = 0
        self._refresh()

    def _refresh(self) -> None:
        max_c = max(self._counts.values()) or 1
        for i, grade in enumerate(GRADE_ORDER):
            c = self._counts[grade]
            if _PG_OK and hasattr(self, "_bars"):
                self._bars[i].setOpts(height=[c])
            if hasattr(self, "_counts_labels"):
                self._counts_labels[grade].setText(str(c))

        if _PG_OK and hasattr(self, "_plot"):
            self._plot.setYRange(0, max_c * 1.15, padding=0)


# ── Throughput Over Time chart ─────────────────────────────────────────────────

class ThroughputLineChart(QWidget):
    """
    Rolling 60-point line chart of TRUE apples/min throughput.

    Mechanism:
      - Every grade event calls push_grade_event(), which appends
        time.monotonic() to a deque of timestamps.
      - A 1-Hz QTimer fires _sample_tick(), which counts timestamps
        that fall within the last 60 seconds.  That count IS the
        apples/min value - no conveyor-speed spinner involved.
      - The 60-point history deque stores one sample per second,
        giving a full 60-second view of the rate over time.
      - sig_throughput_updated(float) is emitted each tick so other
        widgets (e.g. the right-panel MetricsCard) can stay in sync.
    """

    sig_throughput_updated = pyqtSignal(float)  # emits true APM every second

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Timestamps of grade events - maxlen keeps only last ~5 min
        # of events so old ones are auto-dropped
        self._grade_times: deque[float] = deque(maxlen=3600)
        # Rolling history: one APM sample per second, 60 slots
        self._history: deque[float] = deque([0.0] * _WINDOW_S, maxlen=_WINDOW_S)
        # All-time peak (session)
        self._peak_apm: float = 0.0
        self._running = False

        self.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 8px; border: none;"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Thin accent bar at top
        top_bar = QWidget()
        top_bar.setFixedHeight(2)
        top_bar.setStyleSheet(f"background-color: {ACCENT}60; border: none;")
        root.addWidget(top_bar)

        inner = QVBoxLayout()
        inner.setContentsMargins(14, 8, 14, 10)
        inner.setSpacing(4)

        # Header row
        hdr_row = QHBoxLayout()
        hdr = QLabel("THROUGHPUT OVER TIME")
        hdr.setStyleSheet(
            f"color: {SUCCESS}; font-size: 9px; font-weight: 700; "
            "letter-spacing: 1.5px; background: transparent;"
        )
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()

        self._live_lbl = QLabel("-- apple/min")
        self._live_lbl.setStyleSheet(
            f"color: {SUCCESS}; font-size: 13px; font-weight: 700; background: transparent;"
        )
        hdr_row.addWidget(self._live_lbl)
        inner.addLayout(hdr_row)

        if _PG_OK:
            self._build_pg(inner)
        else:
            self._build_fallback(inner)

        root.addLayout(inner)

        # 1-Hz sampler
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._sample_tick)

    # ── PyQtGraph build ───────────────────────────────────────────────────────

    def _build_pg(self, layout: QVBoxLayout) -> None:
        _configure_pg()

        self._plot = pg.PlotWidget()
        self._plot.setBackground(BG_CARD)
        self._plot.setMinimumHeight(120)

        self._plot.getAxis("top").hide()
        self._plot.getAxis("right").hide()

        ax_x = self._plot.getAxis("bottom")
        ax_x.setPen(pg.mkPen(BORDER))
        ax_x.setTextPen(pg.mkPen(TEXT_2))
        ax_x.setLabel("seconds ago", color=TEXT_3)

        ax_y = self._plot.getAxis("left")
        ax_y.setPen(pg.mkPen(BORDER))
        ax_y.setTextPen(pg.mkPen(TEXT_2))
        ax_y.setLabel("apple/min", color=TEXT_3)

        self._plot.showGrid(x=True, y=True, alpha=0.12)
        self._plot.setXRange(0, _WINDOW_S, padding=0)
        self._plot.setYRange(0, 10, padding=0.05)
        self._plot.getPlotItem().setContentsMargins(0, 0, 0, 0)

        # Fill area under curve
        self._fill = pg.FillBetweenItem(
            curve1 = pg.PlotDataItem(np.zeros(_WINDOW_S)),
            curve2 = pg.PlotDataItem(np.zeros(_WINDOW_S)),
            brush  = pg.mkBrush(SUCCESS + "28"),
        )
        self._plot.addItem(self._fill)

        pen = pg.mkPen(color=_hex_to_qcolor(SUCCESS), width=2)
        self._curve = self._plot.plot(
            np.zeros(_WINDOW_S), pen=pen, antialias=True
        )

        # Peak marker
        self._peak_line = pg.InfiniteLine(
            angle = 0,
            pen   = pg.mkPen(color=ACCENT + "60", width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot.addItem(self._peak_line)

        layout.addWidget(self._plot, stretch=1)

        # Footer stats
        footer = QHBoxLayout()
        footer.setSpacing(20)
        for label_txt, attr in [("PEAK", "_peak_lbl"), ("AVG (60s)", "_avg_lbl")]:
            col = QVBoxLayout()
            col.setSpacing(0)
            val = QLabel("--")
            val.setStyleSheet(
                f"color: {TEXT_1}; font-size: 12px; font-weight: 700; background: transparent;"
            )
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(
                f"color: {TEXT_3}; font-size: 9px; letter-spacing: 1px; background: transparent;"
            )
            col.addWidget(val)
            col.addWidget(lbl)
            footer.addLayout(col)
            setattr(self, attr, val)
        footer.addStretch()
        layout.addLayout(footer)

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _build_fallback(self, layout: QVBoxLayout) -> None:
        warn = QLabel("pyqtgraph not installed - chart unavailable")
        warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        warn.setStyleSheet(f"color: {TEXT_3}; font-size: 10px; background: transparent;")
        layout.addWidget(warn)

    # ── Public API ────────────────────────────────────────────────────────────

    def push_grade_event(self) -> None:
        """
        Call once per committed grade.  Appends time.monotonic() so the
        1-Hz sampler can count real events in the sliding 60-s window.
        """
        self._grade_times.append(time.monotonic())

    def start(self) -> None:
        """Begin 1-Hz sampling; call when pipeline starts."""
        self._running = True
        self._timer.start(_TICK_MS)

    def stop(self) -> None:
        self._running = False
        self._timer.stop()

    def reset(self) -> None:
        self.stop()
        self._grade_times.clear()
        self._history   = deque([0.0] * _WINDOW_S, maxlen=_WINDOW_S)
        self._peak_apm  = 0.0
        self._live_lbl.setText("-- apple/min")
        if _PG_OK and hasattr(self, "_curve"):
            self._curve.setData(np.zeros(_WINDOW_S))
            self._peak_line.setValue(0)
            if hasattr(self, "_peak_lbl"):
                self._peak_lbl.setText("--")
                self._avg_lbl.setText("--")
        self._update_fill()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sample_tick(self) -> None:
        """
        Fires every 1 second.  Counts grade events in the last 60 s to get
        the true rolling apples/min, then appends to the history buffer.
        """
        now    = time.monotonic()
        cutoff = now - _WINDOW_S          # 60 seconds ago

        # Drop timestamps older than 60 s from the left of the deque
        while self._grade_times and self._grade_times[0] < cutoff:
            self._grade_times.popleft()

        # Count remaining = events in last 60 s = apples / min
        apm = float(len(self._grade_times))

        if apm > self._peak_apm:
            self._peak_apm = apm

        self._history.append(apm)
        self._live_lbl.setText(f"{apm:.0f} apple/min")
        self.sig_throughput_updated.emit(apm)   # → right panel MetricsCard
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        if not (_PG_OK and hasattr(self, "_curve")):
            return

        data  = np.array(self._history, dtype=float)
        x     = np.arange(len(data))
        self._curve.setData(x, data)

        peak  = self._peak_apm
        # Average only over non-zero samples to avoid diluting with idle seconds
        nz    = data[data > 0]
        avg   = float(np.mean(nz)) if len(nz) else 0.0
        y_max = max(peak * 1.2, 10.0)

        self._plot.setYRange(0, y_max, padding=0)
        self._peak_line.setValue(peak)

        if hasattr(self, "_peak_lbl"):
            self._peak_lbl.setText(f"{peak:.0f}")
            self._avg_lbl.setText(f"{avg:.0f}")

        self._update_fill(data, x)

    def _update_fill(
        self,
        data: np.ndarray | None = None,
        x:    np.ndarray | None = None,
    ) -> None:
        if not (_PG_OK and hasattr(self, "_fill")):
            return
        if data is None:
            data = np.zeros(_WINDOW_S)
        if x is None:
            x = np.arange(len(data))
        self._fill.setCurves(
            pg.PlotDataItem(x, data),
            pg.PlotDataItem(x, np.zeros_like(data)),
        )


# ── Analytics Panel ────────────────────────────────────────────────────────────

class AnalyticsPanel(QWidget):
    """
    Drop-in replacement for the Phase 6 placeholder in CenterPanel._tabs.
    Contains two side-by-side live charts.

    Usage in MainWindow:
        # On each committed grade (no speed arg - measured internally):
        self._center.analytics_panel.record_grade(grade)

        # On pipeline start/stop:
        self._center.analytics_panel.start()
        self._center.analytics_panel.stop()
        self._center.analytics_panel.reset()
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {BG_BASE};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = PanelHeaderBar(
            "▣",
            "Analytics",
            "Grade distribution · throughput over time",
        )
        root.addWidget(self._header)

        charts = QHBoxLayout()
        charts.setContentsMargins(6, 6, 6, 6)
        charts.setSpacing(6)

        self.grade_chart      = GradeBarChart(self)
        self.throughput_chart = ThroughputLineChart(self)

        charts.addWidget(self.grade_chart,      stretch=1)
        charts.addWidget(self.throughput_chart, stretch=1)
        root.addLayout(charts, stretch=1)

    # ── Forwarded public API ──────────────────────────────────────────────────

    def record_grade(self, grade: str) -> None:
        """
        Record one committed grade event.  Internally stamps time.monotonic()
        so throughput is computed from real event timing - no conveyor-speed
        spinner involved.
        """
        self.grade_chart.record_grade(grade)
        self.throughput_chart.push_grade_event()

    def start(self) -> None:
        """Start the 1-Hz throughput sampler (call when pipeline starts)."""
        self.throughput_chart.start()

    def stop(self) -> None:
        """Pause throughput sampling (call when pipeline stops)."""
        self.throughput_chart.stop()

    def reset(self) -> None:
        """Clear all chart data."""
        self.grade_chart.reset()
        self.throughput_chart.reset()
