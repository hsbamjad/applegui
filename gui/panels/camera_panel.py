"""
gui/panels/camera_panel.py
==========================
Left control sidebar — hardware-aware layout.

Sections:
  CAMERA   — connect/disconnect, exposure, FPS
  CONVEYOR — speed (1/2/3 apples/s/lane), camera-to-gate distance
  AI MODEL — model selector, load
  SORTER   — enable, mode, outlet legend
  LOGGING  — enable, output path
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from gui.styles import (
    BG_SURFACE, BG_CARD, BG_ELEVATED,
    ACCENT, ACCENT_HV, ACCENT_DK, SUCCESS, WARNING, DANGER,
    TEXT_1, TEXT_2, TEXT_3, BORDER,
)

PANEL_WIDTH  = 290    # Wide enough for label + widget without clipping
LABEL_W      = 105    # Fixed label width in field rows
WIDGET_MIN_W = 120    # Minimum widget width in field rows


# ── Helpers ───────────────────────────────────────────────────────────────────

class _SectionHeader(QWidget):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 10, 2, 4)
        layout.setSpacing(8)
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            "color: #A8B4CC; font-size: 10px; font-weight: 700; "
            "letter-spacing: 2px; background: transparent;"
        )
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {BORDER}; max-height: 1px; border: none;")
        line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(lbl)
        layout.addWidget(line)


class _Card(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 8px; border: 1px solid {BORDER};"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(7)

    def add(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)


class _StatusDot(QWidget):
    def __init__(self, label: str = "Disconnected", color: str = DANGER, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600; background: transparent;")
        layout.addWidget(self._dot)
        layout.addWidget(self._lbl)
        layout.addStretch()
        self._set_dot(color)

    def _set_dot(self, color: str) -> None:
        self._dot.setStyleSheet(
            f"background-color: {color}; border-radius: 4px; border: none;"
        )

    def set_state(self, state: str, label: str = "") -> None:
        c = {"online": SUCCESS, "offline": DANGER, "warning": WARNING, "idle": TEXT_3}.get(state, TEXT_3)
        self._set_dot(c)
        self._lbl.setText(label or state.capitalize())
        self._lbl.setStyleSheet(f"color: {c}; font-size: 11px; font-weight: 600; background: transparent;")


def _field(label_text: str, widget: QWidget) -> QWidget:
    """Labeled field row: [Label 105px] [Widget min 120px]."""
    row = QWidget()
    row.setStyleSheet("background: transparent; border: none;")
    hl = QHBoxLayout(row)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(8)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(LABEL_W)
    lbl.setStyleSheet(f"color: {TEXT_2}; font-size: 11px; background: transparent;")
    lbl.setWordWrap(False)
    widget.setMinimumWidth(WIDGET_MIN_W)
    hl.addWidget(lbl)
    hl.addWidget(widget, stretch=1)
    return row


def _btn_primary(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(38)          # Must set in Python — CSS min-height alone doesn't constrain VBoxLayout
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {ACCENT}; color: white; border: none;
            font-weight: 700; font-size: 12px;
            border-radius: 8px;
        }}
        QPushButton:hover   {{ background-color: {ACCENT_HV}; }}
        QPushButton:pressed {{ background-color: {ACCENT_DK}; }}
        QPushButton:disabled {{ background-color: {BG_ELEVATED}; color: {TEXT_3}; }}
    """)
    return btn


def _btn_secondary(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(34)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {BG_ELEVATED}; color: {TEXT_1};
            border: 1px solid {ACCENT}55; font-weight: 600; font-size: 12px;
            border-radius: 7px;
        }}
        QPushButton:hover   {{ background-color: {ACCENT}; color: white; border-color: {ACCENT}; }}
        QPushButton:pressed {{ background-color: {ACCENT_DK}; color: white; }}
    """)
    return btn


def _btn_danger_style() -> str:
    return f"""
        QPushButton {{
            background-color: {DANGER}; color: white; border: none;
            font-weight: 700; font-size: 12px;
            border-radius: 8px;
        }}
        QPushButton:hover   {{ background-color: #F87171; }}
        QPushButton:pressed {{ background-color: #DC2626; }}
    """


def _spinbox(mn: int, mx: int, val: int, step: int) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(mn, mx)
    sb.setValue(val)
    sb.setSingleStep(step)
    sb.setMinimumWidth(WIDGET_MIN_W)
    return sb


def _dspinbox(mn: float, mx: float, val: float, step: float, dec: int, suffix: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(mn, mx)
    sb.setValue(val)
    sb.setSingleStep(step)
    sb.setDecimals(dec)
    if suffix:
        sb.setSuffix(suffix)
    sb.setMinimumWidth(WIDGET_MIN_W)
    return sb


def _sep(card_layout) -> None:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet(f"background-color: {BORDER}; max-height: 1px; border: none;")
    card_layout._layout.addWidget(sep)


# ── Left Control Panel ────────────────────────────────────────────────────────

class LeftControlPanel(QWidget):
    """Left sidebar — all operator controls."""

    sig_connect_camera   = pyqtSignal(bool)
    sig_load_model       = pyqtSignal(str)
    sig_sorter_toggled   = pyqtSignal(bool)
    sig_logging_toggled  = pyqtSignal(bool)
    sig_speed_changed    = pyqtSignal(int)
    sig_exposure_changed = pyqtSignal(int, int, int)  # CH1/CH2/CH3 µs — emitted on Apply
    sig_fps_changed      = pyqtSignal(float)           # FPS — emitted on Apply
    sig_gains_changed    = pyqtSignal(float, float, float)  # CH1/CH2/CH3 dB — emitted on Apply All
    sig_awb_triggered    = pyqtSignal()                # One-Push AWB requested (Color CH1 / Source0)
    sig_wb_revert        = pyqtSignal()                # Revert to pre-AWB ratios requested

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._connected = False
        self.setFixedWidth(PANEL_WIDTH)
        self.setStyleSheet(f"background-color: {BG_SURFACE};")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._build()

    def _build(self) -> None:
        from PyQt6.QtWidgets import QScrollArea

        # ── Scrollable inner container ─────────────────────────────────────────
        # Prevents items from being compressed when window height is small,
        # which would cause VBoxLayout to collapse widgets on top of each other.
        inner = QWidget()
        inner.setStyleSheet(f"background-color: {BG_SURFACE};")
        vlayout = QVBoxLayout(inner)
        vlayout.setContentsMargins(12, 6, 12, 12)
        vlayout.setSpacing(0)

        vlayout.addWidget(_SectionHeader("Camera"))
        vlayout.addWidget(self._camera_card())

        vlayout.addWidget(_SectionHeader("Conveyor"))
        vlayout.addWidget(self._conveyor_card())

        vlayout.addWidget(_SectionHeader("AI Model"))
        vlayout.addWidget(self._model_card())

        vlayout.addWidget(_SectionHeader("Sorter"))
        vlayout.addWidget(self._sorter_card())

        vlayout.addWidget(_SectionHeader("Data Logging"))
        vlayout.addWidget(self._logging_card())

        vlayout.addStretch()

        # ── Scroll area wrapper ────────────────────────────────────────────────
        scroll = QScrollArea(self)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: transparent; width: 4px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #334155; border-radius: 2px; min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)

    # ── Card builders ─────────────────────────────────────────────────────────

    def _camera_card(self) -> QWidget:
        card = _Card()

        self._cam_status = _StatusDot("Disconnected", DANGER)
        card.add(self._cam_status)

        self._btn_connect = _btn_primary("Connect Camera")
        self._btn_connect.setToolTip("Connect to JAI FSFE-3200T-10GE via 10 GigE")
        self._btn_connect.clicked.connect(self._on_connect)
        card.add(self._btn_connect)

        _sep(card)

        # ── Per-channel Exposure (CH1 Color / CH2 NIR1 / CH3 NIR2) ───────────
        # Each source has independent exposure control.
        ch_meta = [
            ("CH1 Color", "#f59e0b"),  # amber
            ("CH2 NIR1",  "#22d3ee"),  # cyan
            ("CH3 NIR2",  "#a78bfa"),  # violet
        ]
        self._spn_exposures: list[QSpinBox] = []
        for ch_label, ch_color in ch_meta:
            spn = _spinbox(100, 100_000, 5_000, 500)
            spn.setSuffix(" µs")
            spn.setToolTip(
                f"Exposure time for {ch_label} only.\n"
                "Short (1000–5000 µs)   → dark but sharp on fast conveyor\n"
                "Medium (5000–15000 µs) → balanced for 1 apple/s\n"
                "Long  (>15000 µs)      → brighter but risk of motion blur\n\n"
                "Maximum is capped by FPS:  max = 1,000,000 / FPS"
            )
            # Colored label row
            row = QWidget()
            row.setStyleSheet("background: transparent; border: none;")
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 2, 0, 2)
            hl.setSpacing(6)
            lbl = QLabel(f"● {ch_label}")
            lbl.setFixedWidth(LABEL_W)
            lbl.setStyleSheet(
                f"color: {ch_color}; font-size: 11px; font-weight: 700;"
                " background: transparent;"
            )
            spn.setMinimumWidth(WIDGET_MIN_W)
            hl.addWidget(lbl)
            hl.addWidget(spn, stretch=1)
            card.add(row)
            self._spn_exposures.append(spn)

        # Apply All Exposures + Reset side-by-side
        self._btn_apply_exposure = _btn_secondary("Apply Exposures")
        self._btn_apply_exposure.setToolTip("Send CH1/CH2/CH3 exposure values to camera independently")
        self._btn_apply_exposure.clicked.connect(self._on_apply_exposure)

        self._btn_reset_exposure = QPushButton("↺  Reset")
        self._btn_reset_exposure.setFixedHeight(34)
        self._btn_reset_exposure.setToolTip(
            "Reset all 3 exposure times to 5,000 µs (safe conveyor default)"
        )
        self._btn_reset_exposure.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_ELEVATED}; color: {TEXT_2};
                border: 1px solid {BORDER}; font-weight: 600; font-size: 11px;
                border-radius: 7px; padding: 0 10px;
            }}
            QPushButton:hover   {{ background-color: {WARNING}22; color: {WARNING};
                                   border-color: {WARNING}55; }}
            QPushButton:pressed {{ background-color: {WARNING}44; }}
        """)
        self._btn_reset_exposure.clicked.connect(self._on_reset_exposure)

        exp_btn_hl = QHBoxLayout()
        exp_btn_hl.setContentsMargins(0, 0, 0, 0)
        exp_btn_hl.setSpacing(6)
        exp_btn_hl.addWidget(self._btn_apply_exposure, stretch=1)
        exp_btn_hl.addWidget(self._btn_reset_exposure)
        card.add_layout(exp_btn_hl)

        _sep(card)

        # Frame rate spinbox + Apply button
        self._spn_fps = _spinbox(1, 107, 30, 1)
        self._spn_fps.setSuffix(" FPS")
        self._spn_fps.setToolTip(
            "Camera acquisition frame rate (1–107 FPS).\n"
            "Higher FPS → smoother video, shorter max exposure.\n"
            "Lower FPS  → more light per frame, better NIR signal.\n\n"
            "Changing FPS auto-updates the exposure maximum.\n"
            "Click 'Apply FPS' to send to camera."
        )
        # Cross-link: when FPS spinbox changes, update exposure max immediately
        self._spn_fps.valueChanged.connect(self._on_fps_spinbox_changed)
        card.add(_field("Frame Rate", self._spn_fps))

        self._btn_apply_fps = _btn_secondary("Apply FPS")
        self._btn_apply_fps.setToolTip("Send new frame rate to camera")
        self._btn_apply_fps.clicked.connect(self._on_apply_fps)
        card.add(self._btn_apply_fps)

        # Fix initial max — valueChanged fires before signal is connected so call manually
        self._on_fps_spinbox_changed(self._spn_fps.value())

        _sep(card)

        # ── Per-channel Gain (CH1 Color / CH2 NIR1 / CH3 NIR2) ───────────────
        # Each source has independent gain control.
        # Color: hardcoded hex matching channel header colors in image_display.py
        ch_meta = [
            ("CH1 Color", "#f59e0b"),  # amber — matches channel 1 header
            ("CH2 NIR1",  "#22d3ee"),  # cyan  — matches channel 2 header
            ("CH3 NIR2",  "#a78bfa"),  # violet — matches channel 3 header
        ]
        self._spn_gains: list[QDoubleSpinBox] = []
        for ch_label, ch_color in ch_meta:
            spn = _dspinbox(1.0, 16.0, 1.0, 0.5, 1)
            spn.setSuffix(" dB")
            spn.setToolTip(
                f"Gain for {ch_label} only.\n"
                "1 dB  = hardware minimum (camera floor)\n"
                "6 dB  = ~2× signal boost\n"
                "12 dB = ~4× boost (recommended max for clean NIR)\n"
                "16 dB = hardware maximum (camera ceiling)"
            )
            # Colored label row
            row = QWidget()
            row.setStyleSheet("background: transparent; border: none;")
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 2, 0, 2)
            hl.setSpacing(6)
            lbl = QLabel(f"● {ch_label}")
            lbl.setFixedWidth(LABEL_W)
            lbl.setStyleSheet(
                f"color: {ch_color}; font-size: 11px; font-weight: 700;"
                " background: transparent;"
            )
            spn.setMinimumWidth(WIDGET_MIN_W)
            hl.addWidget(lbl)
            hl.addWidget(spn, stretch=1)
            card.add(row)
            self._spn_gains.append(spn)

        # Apply All Gains + Reset side-by-side
        self._btn_apply_gain = _btn_secondary("Apply All Gains")
        self._btn_apply_gain.setToolTip(
            "Send CH1/CH2/CH3 gain values to camera independently"
        )
        self._btn_apply_gain.clicked.connect(self._on_apply_gains)

        self._btn_reset_gain = QPushButton("↺  Reset")
        self._btn_reset_gain.setFixedHeight(34)
        self._btn_reset_gain.setToolTip("Reset all 3 channels to hardware minimum (1 dB)")
        self._btn_reset_gain.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_ELEVATED}; color: {TEXT_2};
                border: 1px solid {BORDER}; font-weight: 600; font-size: 11px;
                border-radius: 7px; padding: 0 10px;
            }}
            QPushButton:hover   {{ background-color: {WARNING}22; color: {WARNING};
                                   border-color: {WARNING}55; }}
            QPushButton:pressed {{ background-color: {WARNING}44; }}
        """)
        self._btn_reset_gain.clicked.connect(self._on_reset_gains)

        gain_btn_hl = QHBoxLayout()
        gain_btn_hl.setContentsMargins(0, 0, 0, 0)
        gain_btn_hl.setSpacing(6)
        gain_btn_hl.addWidget(self._btn_apply_gain, stretch=1)
        gain_btn_hl.addWidget(self._btn_reset_gain)
        card.add_layout(gain_btn_hl)

        _sep(card)

        # ── White Balance (Source0 / Color CH1 only) ────────────────────────────
        # WB ratios live on GainSelector=Red/Green/Blue for Source0 (Color CH1) only.
        # NIR channels have no Bayer pattern and no WB concept.
        wb_hdr = QWidget()
        wb_hdr.setStyleSheet("background: transparent; border: none;")
        wb_hdr_hl = QHBoxLayout(wb_hdr)
        wb_hdr_hl.setContentsMargins(0, 2, 0, 2)
        wb_hdr_hl.setSpacing(6)
        wb_ch_lbl = QLabel("● White Balance")
        wb_ch_lbl.setStyleSheet(
            "color: #f59e0b; font-size: 11px; font-weight: 700; background: transparent;"
        )
        wb_hdr_hl.addWidget(wb_ch_lbl)
        wb_hdr_hl.addStretch()
        self._lbl_wb_ratios = QLabel("R: —   G: —   B: —")
        self._lbl_wb_ratios.setStyleSheet(
            f"color: {TEXT_3}; font-size: 10px; background: transparent;"
        )
        wb_hdr_hl.addWidget(self._lbl_wb_ratios)
        card.add(wb_hdr)

        # Auto WB button (amber accent)
        self._btn_awb = QPushButton("⚡  Auto WB")
        self._btn_awb.setFixedHeight(34)
        self._btn_awb.setToolTip(
            "One-Push Auto White Balance on the Color channel (Source0).\n"
            "Point camera at a neutral white/grey reference target, then click.\n"
            "Firmware computes optimal R/G/B ratios under current illumination.\n"
            "Previous ratios are saved automatically for Revert."
        )
        self._btn_awb.setStyleSheet(f"""
            QPushButton {{
                background-color: #f59e0b22; color: #f59e0b;
                border: 1px solid #f59e0b55; font-weight: 700; font-size: 12px;
                border-radius: 7px;
            }}
            QPushButton:hover:enabled   {{ background-color: #f59e0b44; color: #fbbf24;
                                           border-color: #f59e0b99; }}
            QPushButton:pressed:enabled {{ background-color: #f59e0b66; }}
            QPushButton:disabled {{ background-color: {BG_ELEVATED}; color: {TEXT_3}; border-color: {BORDER}; }}
        """)
        self._btn_awb.clicked.connect(self.sig_awb_triggered.emit)

        # Revert WB button (warning/reset style) — disabled until first AWB run
        self._btn_revert_wb = QPushButton("↺  Revert")
        self._btn_revert_wb.setFixedHeight(34)
        self._btn_revert_wb.setEnabled(False)
        self._btn_revert_wb.setToolTip(
            "Restore R/G/B WB ratios to their values before the last Auto WB.\n"
            "Only available after Auto WB has been run at least once."
        )
        self._btn_revert_wb.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_ELEVATED}; color: {TEXT_2};
                border: 1px solid {BORDER}; font-weight: 600; font-size: 11px;
                border-radius: 7px; padding: 0 10px;
            }}
            QPushButton:hover:enabled   {{ background-color: {WARNING}22; color: {WARNING};
                                           border-color: {WARNING}55; }}
            QPushButton:pressed:enabled {{ background-color: {WARNING}44; }}
            QPushButton:disabled {{ background-color: {BG_ELEVATED}; color: {TEXT_3}; border-color: {BORDER}; }}
        """)
        self._btn_revert_wb.clicked.connect(self.sig_wb_revert.emit)

        wb_btn_hl = QHBoxLayout()
        wb_btn_hl.setContentsMargins(0, 4, 0, 0)
        wb_btn_hl.setSpacing(6)
        wb_btn_hl.addWidget(self._btn_awb, stretch=1)
        wb_btn_hl.addWidget(self._btn_revert_wb)
        card.add_layout(wb_btn_hl)

        return card

    # ── Camera card slots ─────────────────────────────────────────────────────

    def _on_fps_spinbox_changed(self, fps: int) -> None:
        """
        Auto-clamp exposure maximum when the FPS spinbox value changes.
        Reflects hardware constraint: max_exposure_us = 1,000,000 / FPS.
        User sees correct range before clicking Apply.
        """
        max_exp = min(100_000, int(1_000_000 / max(fps, 1)))
        for spn in self._spn_exposures:
            spn.setMaximum(max_exp)
            if spn.value() > max_exp:
                spn.setValue(max_exp)

    def _on_apply_exposure(self) -> None:
        """Emit per-channel exposures → main_window._on_exposure_changed."""
        self.sig_exposure_changed.emit(
            self._spn_exposures[0].value(),
            self._spn_exposures[1].value(),
            self._spn_exposures[2].value(),
        )

    def _on_reset_exposure(self) -> None:
        """Reset all 3 exposure times to 5,000 µs (safe conveyor default) and immediately apply."""
        for spn in self._spn_exposures:
            spn.setValue(5_000)
        self._on_apply_exposure()

    def update_exposures(self, ch1: int, ch2: int, ch3: int) -> None:
        """Sync exposure spinboxes with actual firmware values (invoked on readback)."""
        for spn in self._spn_exposures:
            spn.blockSignals(True)
        if len(self._spn_exposures) >= 3:
            if ch1 >= 0: self._spn_exposures[0].setValue(ch1)
            if ch2 >= 0: self._spn_exposures[1].setValue(ch2)
            if ch3 >= 0: self._spn_exposures[2].setValue(ch3)
        for spn in self._spn_exposures:
            spn.blockSignals(False)


    def _on_apply_fps(self) -> None:
        """Emit FPS signal → main_window._on_fps_changed."""
        self.sig_fps_changed.emit(float(self._spn_fps.value()))

    def _on_apply_gains(self) -> None:
        """Emit per-channel gains → main_window._on_gains_changed."""
        self.sig_gains_changed.emit(
            self._spn_gains[0].value(),
            self._spn_gains[1].value(),
            self._spn_gains[2].value(),
        )

    def _on_reset_gains(self) -> None:
        """Reset all 3 channel gains to hardware minimum (1 dB) and immediately apply."""
        for spn in self._spn_gains:
            spn.setValue(1.0)
        self.sig_gains_changed.emit(1.0, 1.0, 1.0)

    def update_gains(self, ch1: float, ch2: float, ch3: float) -> None:
        """Called by main_window after firmware readback to sync spinboxes to truth."""
        values = [ch1, ch2, ch3]
        for spn, val in zip(self._spn_gains, values):
            if val >= 0:
                spn.setValue(val)

    def update_white_balance(self, success: bool, r: float, g: float, b: float) -> None:
        """
        Called by main_window after AWB or manual WB write is confirmed by firmware.
        Updates the ratio readout label and enables the Revert button.
        """
        if success:
            self._lbl_wb_ratios.setText(f"R: {r:.3f}   G: {g:.3f}   B: {b:.3f}")
            self._lbl_wb_ratios.setStyleSheet(
                "color: #f59e0b; font-size: 10px; background: transparent;"
            )
            self._btn_revert_wb.setEnabled(True)
        else:
            self._lbl_wb_ratios.setText("R: —   G: —   B: — (failed)")
            self._lbl_wb_ratios.setStyleSheet(
                f"color: {DANGER}; font-size: 10px; background: transparent;"
            )


    def _conveyor_card(self) -> QWidget:
        card = _Card()

        # Lane count — fixed hardware
        lane_row = QHBoxLayout()
        lane_row.setContentsMargins(0, 0, 0, 0)
        lane_row.setSpacing(8)
        lane_lbl = QLabel("Lanes")
        lane_lbl.setFixedWidth(LABEL_W)
        lane_lbl.setStyleSheet(f"color: {TEXT_2}; font-size: 11px; background: transparent;")
        lane_val = QLabel("3  (hardware fixed)")
        lane_val.setStyleSheet(f"color: {TEXT_1}; font-size: 11px; font-weight: 600; background: transparent;")
        lane_row.addWidget(lane_lbl)
        lane_row.addWidget(lane_val)
        lane_row.addStretch()
        card.add_layout(lane_row)

        # Speed: use a ComboBox — no suffix clipping issues
        self._combo_speed = QComboBox()
        self._combo_speed.addItems(["1  apple / s", "2  apples / s", "3  apples / s"])
        self._combo_speed.setToolTip(
            "Conveyor speed per lane:\n"
            "  1/s → sort accuracy 99.9%\n"
            "  2/s → sort accuracy 99.7%\n"
            "  3/s → sort accuracy 97.4%"
        )
        self._combo_speed.currentIndexChanged.connect(
            lambda i: self.sig_speed_changed.emit(i + 1)
        )
        self._combo_speed.setMinimumWidth(WIDGET_MIN_W)
        card.add(_field("Speed", self._combo_speed))

        # Gate distance
        self._spn_gate = _dspinbox(0.10, 2.00, 0.50, 0.05, 2, " m")
        self._spn_gate.setToolTip(
            "Physical distance from camera center to sorter gate.\n"
            "Measure on the physical setup and update."
        )
        card.add(_field("Camera → Gate", self._spn_gate))
        return card

    def _model_card(self) -> QWidget:
        card = _Card()

        self._model_status = _StatusDot("No model loaded", TEXT_3)
        card.add(self._model_status)

        self._combo_model = QComboBox()
        self._combo_model.setToolTip("YOLOv8m-seg models found in models/")
        self._combo_model.setMinimumWidth(WIDGET_MIN_W)
        card.add(self._combo_model)

        self._btn_load = _btn_secondary("Load Model")
        self._btn_load.setToolTip("Load selected model into GPU memory")
        self._btn_load.clicked.connect(self._on_load_model)
        card.add(self._btn_load)

        self._lbl_model_detail = QLabel("—")
        self._lbl_model_detail.setStyleSheet(
            f"color: {TEXT_3}; font-size: 10px; background: transparent;"
        )
        self._lbl_model_detail.setWordWrap(True)
        card.add(self._lbl_model_detail)
        return card

    def _sorter_card(self) -> QWidget:
        card = _Card()

        self._chk_sorter = QCheckBox("Enable Sorting")
        self._chk_sorter.setToolTip("Activate pneumatic sorting actuators")
        self._chk_sorter.toggled.connect(self.sig_sorter_toggled.emit)
        card.add(self._chk_sorter)

        self._combo_sorter = QComboBox()
        self._combo_sorter.addItems([
            "Simulation  (log only)",
            "Serial — Arduino (COM3)",
        ])
        self._combo_sorter.setToolTip(
            "Simulation: commands logged, no hardware fired\n"
            "Serial: commands sent to Arduino via USB/Serial"
        )
        self._combo_sorter.setMinimumWidth(WIDGET_MIN_W)
        card.add(self._combo_sorter)

        # Outlet legend — one row per grade
        card.add(self._outlet_legend())
        return card

    def _outlet_legend(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        entries = [
            (SUCCESS, "Outlet A", "Fresh"),
            (ACCENT,  "Outlet B", "Processing"),
            (DANGER,  "Outlet C", "Cull  (default)"),
        ]
        for color, outlet, grade in entries:
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel("●")
            dot.setFixedWidth(14)
            dot.setStyleSheet(f"color: {color}; font-size: 9px; background: transparent;")
            ol = QLabel(outlet)
            ol.setFixedWidth(54)
            ol.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 700; background: transparent;")
            gl = QLabel(grade)
            gl.setStyleSheet(f"color: {TEXT_2}; font-size: 11px; background: transparent;")
            row.addWidget(dot)
            row.addWidget(ol)
            row.addWidget(gl)
            row.addStretch()
            layout.addLayout(row)
        return w

    def _logging_card(self) -> QWidget:
        card = _Card()
        self._chk_logging = QCheckBox("Enable Logging")
        self._chk_logging.setToolTip("Log grades to CSV + save 3-channel TIFF images")
        self._chk_logging.toggled.connect(self.sig_logging_toggled.emit)
        card.add(self._chk_logging)
        self._lbl_log_path = QLabel("Output:  data/")
        self._lbl_log_path.setStyleSheet(
            f"color: {TEXT_3}; font-size: 10px; background: transparent;"
        )
        card.add(self._lbl_log_path)
        return card

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_connect(self) -> None:
        self._connected = not self._connected
        self.sig_connect_camera.emit(self._connected)
        self._refresh_btn()

    def _on_load_model(self) -> None:
        name = self._combo_model.currentText()
        if name and "No model" not in name:
            self.sig_load_model.emit(name)

    def _refresh_btn(self) -> None:
        if self._connected:
            self._btn_connect.setText("Disconnect")
            self._btn_connect.setStyleSheet(_btn_danger_style())
        else:
            self._btn_connect.setText("Connect Camera")
            self._btn_connect.setStyleSheet(f"""
                QPushButton {{
                    background-color: {ACCENT}; color: white; border: none;
                    font-weight: 700; font-size: 12px;
                    border-radius: 8px;
                }}
                QPushButton:hover   {{ background-color: {ACCENT_HV}; }}
                QPushButton:pressed {{ background-color: {ACCENT_DK}; }}
            """)
        self._btn_connect.setFixedHeight(38)   # re-enforce after every stylesheet swap

    # ── Public API ────────────────────────────────────────────────────────────

    def set_camera_connected(self, connected: bool) -> None:
        self._connected = connected
        self._cam_status.set_state(
            "online" if connected else "offline",
            "Connected" if connected else "Disconnected",
        )
        self._refresh_btn()

    def set_model_loaded(self, name: str) -> None:
        self._model_status.set_state("online", "Loaded")
        self._lbl_model_detail.setText(f"▶  {name}")
        self._lbl_model_detail.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; background: transparent;"
        )

    def populate_models(self, names: list[str]) -> None:
        self._combo_model.clear()
        if names:
            self._combo_model.addItems(names)
        else:
            self._combo_model.addItem("No models in  models/")

    @property
    def conveyor_speed(self) -> int:
        return self._combo_speed.currentIndex() + 1

    @property
    def gate_distance(self) -> float:
        return self._spn_gate.value()
