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
    QCheckBox, QFrame, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QPoint, QSize,
)

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
    # Single permanent stylesheet — danger state is toggled via the 'danger' dynamic property.
    # Base QPushButton rule = purple (always applies); [danger="true"] overrides to red.
    # Avoids calling setStyleSheet() again on state change (which resets Qt size constraints).
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {ACCENT}; color: white; border: none;
            font-weight: 700; font-size: 12px;
            border-radius: 8px;
        }}
        QPushButton:hover   {{ background-color: {ACCENT_HV}; }}
        QPushButton:pressed {{ background-color: {ACCENT_DK}; }}
        QPushButton:disabled {{ background-color: {BG_ELEVATED}; color: {TEXT_3}; }}
        QPushButton[danger="true"] {{
            background-color: {DANGER};
        }}
        QPushButton[danger="true"]:hover   {{ background-color: #F87171; }}
        QPushButton[danger="true"]:pressed {{ background-color: #DC2626; }}
    """)
    btn.setProperty("danger", "false")
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
            min-height: 38px; max-height: 38px;
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


def _sub_header(card: "_Card", text: str, icon: str = "", color: str = TEXT_2) -> None:
    """
    Adds a compact section heading inside a _Card, e.g. '⏱  Exposure Time'.
    Uses a thin left accent stripe for visual hierarchy.
    """
    w = QWidget()
    w.setStyleSheet("background: transparent; border: none;")
    hl = QHBoxLayout(w)
    hl.setContentsMargins(0, 6, 0, 2)
    hl.setSpacing(6)

    # Thin colored marker
    bar = QFrame()
    bar.setFixedSize(3, 14)
    bar.setStyleSheet(f"background-color: {color}; border-radius: 2px; border: none;")
    hl.addWidget(bar)

    full_text = f"{icon}  {text}" if icon else text
    lbl = QLabel(full_text)
    lbl.setStyleSheet(
        f"color: {color}; font-size: 10px; font-weight: 700; "
        f"letter-spacing: 1.5px; background: transparent;"
    )
    hl.addWidget(lbl)
    hl.addStretch()
    card._layout.addWidget(w)


# ── Camera Controls Floating Window ────────────────────────────────────────────

class CameraControlsWindow(QWidget):
    """
    Frameless floating sub-window — 2-column, no scroll.
    Left column: Exposure · FPS · Gain
    Right column: WB · Black Level · ROI
    Drag by title bar. Close button syncs sidebar toggle.
    """

    sig_hidden = pyqtSignal()   # emitted when window is closed/hidden

    POPUP_WIDTH  = 660
    POPUP_TITLE  = "Camera Controls"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(self.POPUP_WIDTH)
        self._drag_pos: QPoint | None = None
        self._build_shell()

    # ── Shell (title bar + 2-column content) ───────────────────────────

    def _build_shell(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Outer container — the visible rounded panel
        outer = QWidget()
        outer.setObjectName("cam_outer")
        outer.setStyleSheet(f"""
            QWidget#cam_outer {{
                background-color: {BG_SURFACE};
                border: 1px solid {SUCCESS}55;
                border-radius: 12px;
            }}
        """)
        outer_vl = QVBoxLayout(outer)
        outer_vl.setContentsMargins(0, 0, 0, 0)
        outer_vl.setSpacing(0)

        # ── Title bar ───────────────────────────────────────────────
        title_bar = QWidget()
        title_bar.setObjectName("cam_titlebar")
        title_bar.setFixedHeight(44)
        title_bar.setCursor(Qt.CursorShape.SizeAllCursor)
        title_bar.setStyleSheet(f"""
            QWidget#cam_titlebar {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {SUCCESS}55, stop:1 {SUCCESS}22
                );
                border-radius: 12px 12px 0 0;
                border-bottom: 1px solid {SUCCESS}66;
            }}
        """)
        tb_hl = QHBoxLayout(title_bar)
        tb_hl.setContentsMargins(16, 0, 12, 0)
        tb_hl.setSpacing(12)


        ttl = QLabel("Camera Controls")

        ttl.setStyleSheet(
            f"color: {TEXT_1}; font-size: 14px; font-weight: 700; "
            "background: transparent; border: none; letter-spacing: 0.2px;"
        )
        tb_hl.addWidget(ttl)
        tb_hl.addStretch()


        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_ELEVATED}; color: {TEXT_2};
                border: 1px solid {BORDER}; border-radius: 14px;
                font-size: 12px; font-weight: 700; padding: 0;
            }}
            QPushButton:hover {{
                background-color: {DANGER}; color: white; border-color: {DANGER};
            }}
            QPushButton:pressed {{ background-color: #DC2626; }}
        """)
        close_btn.clicked.connect(self.hide)
        tb_hl.addWidget(close_btn)

        title_bar.mousePressEvent   = self._tb_mouse_press
        title_bar.mouseMoveEvent    = self._tb_mouse_move
        title_bar.mouseReleaseEvent = self._tb_mouse_release
        outer_vl.addWidget(title_bar)

        # ── Accent divider ────────────────────────────────────────────
        div = QFrame()
        div.setFixedHeight(2)
        div.setStyleSheet(
            f"background: qlineargradient("
            f"x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {SUCCESS},stop:0.6 {SUCCESS}88,stop:1 transparent);"
            f" border: none;"
        )
        outer_vl.addWidget(div)

        # ── Two-column content (NO scroll) ────────────────────────────
        body = QWidget()
        body.setStyleSheet("background: transparent; border: none;")
        body_hl = QHBoxLayout(body)
        body_hl.setContentsMargins(12, 10, 12, 14)
        body_hl.setSpacing(10)

        # Vertical separator between columns
        def _make_col() -> tuple[QWidget, QVBoxLayout]:
            w = QWidget()
            w.setStyleSheet("background: transparent; border: none;")
            vl = QVBoxLayout(w)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(8)
            return w, vl

        self._left_w,  self._left_col  = _make_col()
        self._right_w, self._right_col = _make_col()

        col_sep = QFrame()
        col_sep.setFixedWidth(1)
        col_sep.setStyleSheet(
            f"background: qlineargradient("
            f"x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 transparent,stop:0.2 {BORDER},"
            f"stop:0.8 {BORDER},stop:1 transparent);"
            f" border: none;"
        )

        body_hl.addWidget(self._left_w,  stretch=1)
        body_hl.addWidget(col_sep)
        body_hl.addWidget(self._right_w, stretch=1)

        outer_vl.addWidget(body)
        root.addWidget(outer)

    # ── Content population (called by LeftControlPanel) ────────────────

    def add_widget(self, widget: QWidget, col: int = 0) -> None:
        """Add widget to left (col=0) or right (col=1) column."""
        if col == 0:
            self._left_col.addWidget(widget)
        else:
            self._right_col.addWidget(widget)

    def finalize(self) -> None:
        """Call after all widgets are added. Sizes window to fit content without clipping."""
        self._left_col.addStretch()
        self._right_col.addStretch()
        # adjustSize() computes from layout sizeHints but may change width;
        # re-apply our fixed width afterward so columns stay at correct proportions.
        self.adjustSize()
        self.setFixedWidth(self.POPUP_WIDTH)
    # ── Show / position ────────────────────────────────────────────

    def show_beside(self, anchor: QWidget) -> None:
        """
        Position the window to the right of `anchor` widget,
        vertically aligned with it, and show it.
        """
        global_pos = anchor.mapToGlobal(QPoint(0, 0))
        x = global_pos.x() + anchor.width() + 6   # 6 px gap
        y = global_pos.y()

        # Keep on screen
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        if x + self.width() > screen.right():
            x = global_pos.x() - self.width() - 6
        y = max(screen.top(), min(y, screen.bottom() - self.height()))

        self.move(x, y)
        self.show()
        self.raise_()

    # ── Drag support ─────────────────────────────────────────────────

    def _tb_mouse_press(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def _tb_mouse_move(self, event) -> None:
        if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def _tb_mouse_release(self, event) -> None:
        self._drag_pos = None

    def hideEvent(self, event) -> None:
        """Emit sig_hidden so sidebar toggle button can sync its checked state."""
        super().hideEvent(event)
        self.sig_hidden.emit()


# ── Left Control Panel ────────────────────────────────────────────────────────

class LeftControlPanel(QWidget):
    """Left sidebar — all operator controls."""

    sig_connect_camera        = pyqtSignal(bool)
    sig_load_model            = pyqtSignal(str)
    sig_sorter_toggled        = pyqtSignal(bool)
    sig_logging_toggled       = pyqtSignal(bool)
    sig_speed_changed         = pyqtSignal(int)
    sig_exposure_changed      = pyqtSignal(int, int, int)    # CH1/CH2/CH3 µs — emitted on Apply
    sig_fps_changed           = pyqtSignal(float)            # FPS — emitted on Apply
    sig_gains_changed         = pyqtSignal(float, float, float)   # CH1/CH2/CH3 dB — emitted on Apply
    sig_awb_triggered         = pyqtSignal()                 # One-Push AWB requested (Color CH1 / Source0)
    sig_wb_revert             = pyqtSignal()                 # Revert to pre-AWB ratios requested
    sig_black_level_changed   = pyqtSignal(float, float, float)  # CH1/CH2/CH3 DN — emitted on Apply
    sig_roi_changed           = pyqtSignal(int, int, int, int)   # OffsetX, OffsetY, Width, Height
    sig_roi_reset             = pyqtSignal()                 # Reset ROI to full frame
    sig_roi_preview           = pyqtSignal(int, int, int, int)   # Live preview as spinboxes change

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._connected = False
        self.setFixedWidth(PANEL_WIDTH)
        self.setStyleSheet(f"background-color: {BG_SURFACE};")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # Build the floating camera controls window first (so cards can be added)
        self._cam_win = CameraControlsWindow(self.window())
        self._build()
        # Populate floating window: left col = Exposure/FPS/Gain, right col = WB/BL/ROI
        cam_card, wb_bl_card = self._camera_cards_split()
        self._cam_win.add_widget(cam_card,   col=0)
        self._cam_win.add_widget(wb_bl_card, col=1)   # right card now contains WB+BL+ROI
        self._cam_win.finalize()
        # Sync sidebar button when popup is closed via X or hide()
        self._cam_win.sig_hidden.connect(
            lambda: self._btn_cam_controls.setChecked(False)
        )

    def _build(self) -> None:
        # ── Scrollable inner container ─────────────────────────────────────────
        # Prevents items from being compressed when window height is small,
        # which would cause VBoxLayout to collapse widgets on top of each other.
        inner = QWidget()
        inner.setStyleSheet(f"background-color: {BG_SURFACE};")
        vlayout = QVBoxLayout(inner)
        vlayout.setContentsMargins(12, 6, 12, 12)
        vlayout.setSpacing(0)

        # ── Camera section ─────────────────────────────────────────────────
        vlayout.addWidget(_SectionHeader("Camera"))

        # Status dot + connect button live directly in the sidebar
        cam_status_card = _Card()
        self._cam_status = _StatusDot("Disconnected", DANGER)
        cam_status_card.add(self._cam_status)
        self._btn_connect = _btn_primary("Connect Camera")
        self._btn_connect.setToolTip("Connect to JAI FSFE-3200T-10GE via 10 GigE")
        self._btn_connect.clicked.connect(self._on_connect)
        cam_status_card.add(self._btn_connect)
        vlayout.addWidget(cam_status_card)
        vlayout.addSpacing(6)

        # Camera Controls popup toggle button — identical geometry to Connect Camera.
        # The Connect Camera button lives inside a _Card (10 px h-margins), so we
        # wrap this button in the same 10 px inset to make them visually identical.
        self._btn_cam_controls = QPushButton("Camera Controls")
        self._btn_cam_controls.setFixedHeight(38)
        self._btn_cam_controls.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_cam_controls.setCheckable(True)
        self._btn_cam_controls.setToolTip(
            "Open Camera Controls panel\n"
            "Exposure · FPS · Gain · White Balance · Black Level · ROI"
        )
        self._btn_cam_controls.setStyleSheet(f"""
            QPushButton {{
                background-color: {SUCCESS};
                color: white;
                border: none;
                font-weight: 700;
                font-size: 12px;
                border-radius: 8px;
            }}
            QPushButton:hover   {{ background-color: {SUCCESS}CC; }}
            QPushButton:checked {{
                background-color: {SUCCESS}AA;
                border: 2px solid {SUCCESS};
                color: white;
            }}
            QPushButton:pressed {{ background-color: {SUCCESS}88; }}
        """)
        self._btn_cam_controls.clicked.connect(self._on_cam_controls_toggle)

        # Add with the same 10px side inset as _Card so it matches Connect Camera width
        _cam_row = QHBoxLayout()
        _cam_row.setContentsMargins(10, 0, 10, 0)
        _cam_row.addWidget(self._btn_cam_controls)
        vlayout.addLayout(_cam_row)


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

    @staticmethod
    def _row_sep(card: "_Card") -> None:
        """Hairline divider between individual control rows."""
        f = QFrame()
        f.setFixedHeight(1)
        f.setStyleSheet(f"background-color: {BORDER}55; border: none; margin: 0 4px;")
        card._layout.addWidget(f)

    @staticmethod
    def _double_sep(card: "_Card") -> None:
        """Two hairlines — used after Apply buttons to mark section end.
        A small gap is added first so the lines don't touch the button border."""
        card._layout.addSpacing(7)   # breathing room below the button
        for _ in range(2):
            f = QFrame()
            f.setFixedHeight(1)
            f.setStyleSheet(f"background-color: {BORDER}55; border: none; margin: 0 4px;")
            card._layout.addWidget(f)

    def _on_cam_controls_toggle(self) -> None:
        """Show or hide the floating Camera Controls window."""
        if self._cam_win.isVisible():
            self._cam_win.hide()
            self._btn_cam_controls.setChecked(False)
        else:
            self._cam_win.show_beside(self)
            self._btn_cam_controls.setChecked(True)


    def _camera_cards_split(self) -> tuple[QWidget, QWidget]:
        """Return (left_card, right_card) for the 2-column popup layout.

        Left  card: Exposure Time · Frame Rate · Sensor Gain
        Right card: White Balance · Black Level
        """
        left  = _Card()
        right = _Card()

        # ── Sub-section: Exposure Time ────────────────────────────────
        _sub_header(left, "EXPOSURE TIME", color="#f59e0b")

        # ── Per-channel Exposure (CH1 Color / CH2 NIR1 / CH3 NIR2) ───────────
        # Each source has independent exposure control.
        ch_meta = [
            ("CH1 Color", "#f59e0b"),  # amber
            ("CH2 NIR1",  "#22d3ee"),  # cyan
            ("CH3 NIR2",  "#a78bfa"),  # violet
        ]
        self._spn_exposures: list[QSpinBox] = []
        for i, (ch_label, ch_color) in enumerate(ch_meta):
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
            left.add(row)
            if i < 2:
                self._row_sep(left)
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
        left.add_layout(exp_btn_hl)
        self._double_sep(left)



        # ── Sub-section: Frame Rate ────────────────────────────────────────
        _sub_header(left, "FRAME RATE", color="#22d3ee")

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
        left.add(_field("Frame Rate", self._spn_fps))

        self._btn_apply_fps = _btn_secondary("Apply FPS")
        self._btn_apply_fps.setToolTip("Send new frame rate to camera")
        self._btn_apply_fps.clicked.connect(self._on_apply_fps)
        left.add(self._btn_apply_fps)
        self._double_sep(left)


        # Fix initial max — valueChanged fires before signal is connected so call manually
        self._on_fps_spinbox_changed(self._spn_fps.value())



        # ── Sub-section: Gain ─────────────────────────────────────────────
        _sub_header(left, "SENSOR GAIN", color="#a78bfa")

        # ── Per-channel Gain (CH1 Color / CH2 NIR1 / CH3 NIR2) ───────────────
        # Each source has independent gain control.
        # Color: hardcoded hex matching channel header colors in image_display.py
        ch_meta = [
            ("CH1 Color", "#f59e0b"),  # amber — matches channel 1 header
            ("CH2 NIR1",  "#22d3ee"),  # cyan  — matches channel 2 header
            ("CH3 NIR2",  "#a78bfa"),  # violet — matches channel 3 header
        ]
        self._spn_gains: list[QDoubleSpinBox] = []
        for i, (ch_label, ch_color) in enumerate(ch_meta):
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
            left.add(row)
            if i < 2:
                self._row_sep(left)
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
        left.add_layout(gain_btn_hl)
        self._double_sep(left)


        # ══════════════════════════════════════════════════════════════
        # RIGHT CARD — White Balance · Black Level
        # ══════════════════════════════════════════════════════════════

        # ── Sub-section: White Balance ────────────────────────────────────
        _sub_header(right, "WHITE BALANCE", color="#f59e0b")

        # ── White Balance (Source0 / Color CH1 only) ────────────────────────────
        # WB ratios live on GainSelector=Red/Green/Blue for Source0 (Color CH1) only.
        # NIR channels have no Bayer pattern and no WB concept.
        self._lbl_wb_ratios = QLabel("R: —   G: —   B: —")
        self._lbl_wb_ratios.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_wb_ratios.setStyleSheet(
            f"color: {TEXT_3}; font-size: 10px; background: {BG_ELEVATED}40; "
            f"border: 1px solid {BORDER}; border-radius: 5px; padding: 3px 6px;"
        )
        right.add(self._lbl_wb_ratios)
        self._row_sep(right)


        # Auto WB button (amber accent)
        self._btn_awb = QPushButton("Auto WB")
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
        right.add_layout(wb_btn_hl)
        self._double_sep(right)



        # ── Sub-section: Black Level ──────────────────────────────────────
        _sub_header(right, "BLACK LEVEL", color="#94a3b8")

        # ── Black Level (per-source hardware pedestal) ─────────────────────
        # GenICam: BlackLevelSelector=All, BlackLevel float (DN)
        # Removing ambient dark-pedestal and thermal noise at hardware level.

        # Channel colors match the rest of the panel (CH1=amber, CH2/CH3=cyan)
        _BL_COLORS  = ["#f59e0b", "#22d3ee", "#a78bfa"]  # amber · cyan · purple — matches Exposure / Gain

        _BL_LABELS  = ["CH1 Color", "CH2 NIR1", "CH3 NIR2"]
        _BL_TIPS    = [
            "Color sensor (Source0) black level pedestal (DN).",
            "NIR1 sensor (Source1) black level pedestal (DN).",
            "NIR2 sensor (Source2) black level pedestal (DN).",
        ]
        self._spn_black_levels: list[QDoubleSpinBox] = []
        for ch_idx in range(3):
            bl_row = QWidget()
            bl_row.setStyleSheet("background: transparent; border: none;")
            bl_rl = QHBoxLayout(bl_row)
            bl_rl.setContentsMargins(0, 1, 0, 1)
            bl_rl.setSpacing(6)

            lbl_col = _BL_COLORS[ch_idx]
            ch_label = QLabel(_BL_LABELS[ch_idx])
            ch_label.setFixedWidth(LABEL_W)
            ch_label.setStyleSheet(
                f"color: {lbl_col}; font-size: 11px; "
                f"font-weight: 600; background: transparent;"
            )

            spn = QDoubleSpinBox()
            spn.setRange(0.0, 64.0)    # JAI FS-3200T: practical dark pedestal range 0–64 DN
            spn.setDecimals(1)
            spn.setSingleStep(1.0)
            spn.setValue(0.0)
            spn.setSuffix(" DN")
            spn.setToolTip(_BL_TIPS[ch_idx])
            spn.setMinimumWidth(WIDGET_MIN_W)
            spn.setStyleSheet(f"""
                QDoubleSpinBox {{
                    background-color: {BG_ELEVATED}; color: {TEXT_1};
                    border: 1px solid {BORDER}; border-radius: 5px; padding: 2px 6px;
                }}
                QDoubleSpinBox:focus {{ border-color: {lbl_col}; }}
            """)
            self._spn_black_levels.append(spn)

            bl_rl.addWidget(ch_label)
            bl_rl.addWidget(spn, stretch=1)
            right.add(bl_row)
            if ch_idx < 2:
                self._row_sep(right)

        # Apply + Reset — use same _btn_secondary style as Exposure/FPS/Gain buttons
        self._btn_apply_bl = _btn_secondary("Apply Black Levels")
        self._btn_apply_bl.setToolTip(
            "Write Black Level values to firmware for all 3 sources."
        )
        self._btn_apply_bl.clicked.connect(self._on_apply_black_levels)

        self._btn_reset_bl = QPushButton("↺  Reset")
        self._btn_reset_bl.setFixedHeight(34)
        self._btn_reset_bl.setToolTip("Reset all Black Levels to 0 DN (sensor default).")
        self._btn_reset_bl.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_ELEVATED}; color: {TEXT_2};
                border: 1px solid {BORDER}; font-weight: 600; font-size: 11px;
                border-radius: 7px; padding: 0 10px;
            }}
            QPushButton:hover   {{ background-color: {WARNING}22; color: {WARNING};
                                   border-color: {WARNING}55; }}
            QPushButton:pressed {{ background-color: {WARNING}44; }}
        """)
        self._btn_reset_bl.clicked.connect(self._on_reset_black_levels)

        bl_btn_hl = QHBoxLayout()
        bl_btn_hl.setContentsMargins(0, 4, 0, 0)
        bl_btn_hl.setSpacing(6)
        bl_btn_hl.addWidget(self._btn_apply_bl, stretch=1)
        bl_btn_hl.addWidget(self._btn_reset_bl)
        right.add_layout(bl_btn_hl)
        self._double_sep(right)

        # ══════════════════════════════════════════════════════════════
        # ROI — inlined into right card so it aligns with SENSOR GAIN
        # ══════════════════════════════════════════════════════════════
        _sub_header(right, "ROI", color="#06b6d4")

        self._lbl_roi = QLabel("Full Frame  —  2048 × 1536 @ (0, 0)")
        self._lbl_roi.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_roi.setStyleSheet(
            "color: #06b6d4; font-size: 10px; font-weight: 600; "
            "background: transparent; padding: 4px 0;"
        )
        right.add(self._lbl_roi)
        self._row_sep(right)

        ROI_PARAMS = [
            ("OffsetX", "px",  0, 2032, 16, "Left crop start. Must be multiple of 16."),
            ("OffsetY", "px",  0, 1528,  8, "Top crop start. Must be multiple of 8."),
            ("Width",   "px", 16, 2048, 16, "Capture width. Must be multiple of 16."),
            ("Height",  "px",  8, 1536,  8, "Capture height. Must be multiple of 8."),
        ]
        self._spn_roi: dict[str, QSpinBox] = {}
        for i, (param, unit, lo, hi, step, tip) in enumerate(ROI_PARAMS):
            row_w = QWidget()
            row_w.setStyleSheet("background: transparent; border: none;")
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 1, 0, 1)
            row_l.setSpacing(6)
            lbl = QLabel(param)
            lbl.setFixedWidth(LABEL_W)
            lbl.setStyleSheet(
                f"color: {TEXT_2}; font-size: 11px; "
                f"font-weight: 600; background: transparent;"
            )
            spn = QSpinBox()
            spn.setRange(lo, hi)
            spn.setSingleStep(step)
            spn.setValue(hi if param in ("Width", "Height") else 0)
            spn.setSuffix(f" {unit}")
            spn.setToolTip(tip)
            spn.setMinimumWidth(WIDGET_MIN_W)
            spn.setStyleSheet(f"""
                QSpinBox {{
                    background-color: {BG_ELEVATED}; color: {TEXT_1};
                    border: 1px solid {BORDER}; border-radius: 5px; padding: 2px 6px;
                }}
                QSpinBox:focus {{ border-color: #06b6d4; }}
            """)
            spn.valueChanged.connect(self._update_roi_label)
            self._spn_roi[param] = spn
            row_l.addWidget(lbl)
            row_l.addWidget(spn, stretch=1)
            right.add(row_w)
            if i < len(ROI_PARAMS) - 1:
                self._row_sep(right)

        self._btn_apply_roi = _btn_secondary("Apply ROI")
        self._btn_apply_roi.setToolTip("Apply ROI to all 3 sensors.")
        self._btn_apply_roi.clicked.connect(self._on_apply_roi)

        self._btn_reset_roi = QPushButton("Full Frame")
        self._btn_reset_roi.setFixedHeight(34)
        self._btn_reset_roi.setToolTip("Reset ROI to full 2048×1536 sensor frame.")
        self._btn_reset_roi.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_ELEVATED}; color: {TEXT_1};
                border: 1px solid {BORDER}; border-radius: 7px;
                font-weight: 600; font-size: 11px;
            }}
            QPushButton:hover:enabled {{
                background-color: {WARNING}33; color: {WARNING};
                border-color: {WARNING}88;
            }}
            QPushButton:pressed:enabled {{ background-color: {WARNING}55; }}
        """)
        self._btn_reset_roi.clicked.connect(self.sig_roi_reset.emit)

        roi_btn_hl = QHBoxLayout()
        roi_btn_hl.setContentsMargins(0, 4, 0, 0)
        roi_btn_hl.setSpacing(6)
        roi_btn_hl.addWidget(self._btn_apply_roi, stretch=1)
        roi_btn_hl.addWidget(self._btn_reset_roi)
        self._row_sep(right)
        right.add_layout(roi_btn_hl)

        return left, right

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

    def _on_apply_black_levels(self) -> None:
        """Emit per-channel black levels → main_window._on_black_level_changed."""
        self.sig_black_level_changed.emit(
            self._spn_black_levels[0].value(),
            self._spn_black_levels[1].value(),
            self._spn_black_levels[2].value(),
        )

    def _on_reset_black_levels(self) -> None:
        """Reset all 3 channel black levels to 0 DN (sensor default) and immediately apply."""
        for spn in self._spn_black_levels:
            spn.setValue(0.0)
        self.sig_black_level_changed.emit(0.0, 0.0, 0.0)

    def update_black_levels(self, ch1: float, ch2: float, ch3: float) -> None:
        """Called by main_window after firmware readback to sync spinboxes to truth."""
        values = [ch1, ch2, ch3]
        for spn, val in zip(self._spn_black_levels, values):
            spn.blockSignals(True)
            spn.setValue(val)
            spn.blockSignals(False)

    # ── ROI card ─────────────────────────────────────────────────────────────

    def _roi_card(self) -> QWidget:
        """
        ROI (Region of Interest) card.
        Controls: OffsetX, OffsetY, Width, Height spinboxes.
        GenICam registers: OffsetX, OffsetY, Width, Height (integer, per-source).
        All 3 sources receive the same ROI (co-registered sensors, same FOV).
        """
        card = _Card()
        card.add(_SectionHeader("ROI — Region of Interest"))

        # Readout label — shows current ROI as 'W × H  @ (X, Y)'
        self._lbl_roi = QLabel("Full Frame  —  2048 × 1536 @ (0, 0)")
        self._lbl_roi.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_roi.setStyleSheet(
            "color: #06b6d4; font-size: 10px; font-weight: 600; "
            "background: transparent; padding: 4px 0;"
        )
        card.add(self._lbl_roi)

        # ── Four spinboxes ──────────────────────────────────────────────
        # Layout: Label (fixed) | Spinbox (stretch)
        # Hardware step sizes on JAI FS-3200T (both confirmed by observation):
        #   OffsetX / Width  → multiples of 16 (200 → 192)
        #   OffsetY / Height → multiples of  8 ( 10 →   8)
        ROI_PARAMS = [
            ("OffsetX", "px",  0, 2032, 16, "Left crop start. Must be multiple of 16 — arrow keys step by 16."),
            ("OffsetY", "px",  0, 1528,  8, "Top crop start. Must be multiple of 8 — arrow keys step by 8."),
            ("Width",   "px", 16, 2048, 16, "Capture width. Must be multiple of 16 — arrow keys step by 16."),
            ("Height",  "px",  8, 1536,  8, "Capture height. Must be multiple of 8 — arrow keys step by 8."),
        ]
        self._spn_roi: dict[str, QSpinBox] = {}

        for param, unit, lo, hi, step, tip in ROI_PARAMS:
            row_w = QWidget()
            row_w.setStyleSheet("background: transparent; border: none;")
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 1, 0, 1)
            row_l.setSpacing(6)

            lbl = QLabel(param)
            lbl.setFixedWidth(LABEL_W)
            lbl.setStyleSheet(
                f"color: {TEXT_2}; font-size: 11px; "
                f"font-weight: 600; background: transparent;"
            )

            spn = QSpinBox()
            spn.setRange(lo, hi)
            spn.setSingleStep(step)
            spn.setValue(hi if param in ("Width", "Height") else 0)
            spn.setSuffix(f" {unit}")
            spn.setToolTip(tip)
            spn.setMinimumWidth(WIDGET_MIN_W)
            spn.setStyleSheet(f"""
                QSpinBox {{
                    background-color: {BG_ELEVATED}; color: {TEXT_1};
                    border: 1px solid {BORDER}; border-radius: 5px; padding: 2px 6px;
                }}
                QSpinBox:focus {{ border-color: #06b6d4; }}
            """)
            # Live preview: update the readout label on any value change
            spn.valueChanged.connect(self._update_roi_label)
            self._spn_roi[param] = spn

            row_l.addWidget(lbl)
            row_l.addWidget(spn, stretch=1)
            card.add(row_w)

        # NOTE: No cross-constraint wiring — static ranges are clear.
        # Firmware clamps OffsetX+Width<=2048 etc. on Apply automatically.

        # ── Buttons: Apply + Full Frame ──────────────────────────────
        self._btn_apply_roi = _btn_secondary("Apply ROI")
        self._btn_apply_roi.setToolTip(
            "Apply ROI to all 3 sensors. Values are step-aligned by firmware."
        )
        self._btn_apply_roi.clicked.connect(self._on_apply_roi)

        self._btn_reset_roi = QPushButton("▦ Full Frame")
        self._btn_reset_roi.setFixedHeight(34)
        self._btn_reset_roi.setToolTip(
            "Reset ROI to full 2048×1536 sensor frame."
        )
        self._btn_reset_roi.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_ELEVATED}; color: {TEXT_1};
                border: 1px solid {BORDER}; border-radius: 7px;
                font-weight: 600; font-size: 11px;
            }}
            QPushButton:hover:enabled {{
                background-color: {WARNING}33; color: {WARNING};
                border-color: {WARNING}88;
            }}
            QPushButton:pressed:enabled {{ background-color: {WARNING}55; }}
        """)
        self._btn_reset_roi.clicked.connect(self.sig_roi_reset.emit)

        btn_hl = QHBoxLayout()
        btn_hl.setContentsMargins(0, 4, 0, 0)
        btn_hl.setSpacing(6)
        btn_hl.addWidget(self._btn_apply_roi, stretch=1)
        btn_hl.addWidget(self._btn_reset_roi)
        card.add_layout(btn_hl)

        return card

    def _update_roi_label(self) -> None:
        """Refresh the live ROI readout label and push preview overlay to display."""
        x = self._spn_roi["OffsetX"].value()
        y = self._spn_roi["OffsetY"].value()
        w = self._spn_roi["Width"].value()
        h = self._spn_roi["Height"].value()
        full = (x == 0 and y == 0 and w == 2048 and h == 1536)
        pct  = round(100 * w * h / (2048 * 1536))
        if full:
            self._lbl_roi.setText("Full Frame  —  2048 × 1536 @ (0, 0)")
        else:
            self._lbl_roi.setText(f"{w} × {h} px @ ({x}, {y})  —  {pct}% of frame")
        # Push live preview overlay to camera display (appears immediately, no Apply needed)
        self.sig_roi_preview.emit(x, y, w, h)

    def _on_apply_roi(self) -> None:
        """Emit ROI values → main_window._on_roi_changed."""
        self.sig_roi_changed.emit(
            self._spn_roi["OffsetX"].value(),
            self._spn_roi["OffsetY"].value(),
            self._spn_roi["Width"].value(),
            self._spn_roi["Height"].value(),
        )

    def update_roi(self, x: int, y: int, w: int, h: int) -> None:
        """
        Called by main_window after firmware readback to sync spinboxes
        to the actual (step-aligned, clamped) ROI values accepted by the camera.
        """
        for key, val in (("OffsetX", x), ("OffsetY", y),
                         ("Width",   w), ("Height",  h)):
            spn = self._spn_roi[key]
            spn.blockSignals(True)
            spn.setValue(val)
            spn.blockSignals(False)
        self._update_roi_label()

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
        """Toggle the connect button between primary and danger styles.

        Uses a dynamic Qt property ('danger') instead of calling setStyleSheet()
        so that the stylesheet — and therefore the button's computed geometry —
        never changes.  Only the paint color is updated.
        """
        if self._connected:
            self._btn_connect.setText("Disconnect")
            self._btn_connect.setProperty("danger", "true")
        else:
            self._btn_connect.setText("Connect Camera")
            self._btn_connect.setProperty("danger", "false")
        # Re-polish so Qt re-evaluates the property selector without touching size
        self._btn_connect.style().unpolish(self._btn_connect)
        self._btn_connect.style().polish(self._btn_connect)
        self._btn_connect.update()

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
