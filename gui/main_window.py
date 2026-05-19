"""
gui/main_window.py
==================
Main application window — fully wired demo pipeline.

On "Connect Camera":
  1. CameraWorker (QThread) starts → pushes mock frames to 3-channel display
  2. MockInferenceWorker (QThread) starts → emits grades at conveyor speed
  3. Results panel, grade summary, and metrics update live
  4. Session timer starts

On "Disconnect":
  1. Both workers stopped
  2. Display reset to NO SIGNAL
  3. Stats preserved until next session

Layout:
  [Header 56px]
  [Left 252px | Center (expand) | Right 230px]
  [Status bar 24px]
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QStatusBar, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPainter, QLinearGradient

from gui.styles import (
    APP_STYLESHEET, BG_BASE, BG_SURFACE, BG_CARD,
    ACCENT, ACCENT_HV, ACCENT_DK, SUCCESS, WARNING, DANGER,
    TEXT_1, TEXT_2, TEXT_3, BORDER, CH_COLORS,
)
from gui.panels.camera_panel import LeftControlPanel
from gui.panels.stats_panel  import RightStatsPanel
from gui.widgets.image_display import MultiChannelDisplay
from gui.workers.camera_worker import CameraWorker
from gui.workers.inference_worker import MockInferenceWorker

log = logging.getLogger(__name__)


def _load_config(path: str = "config/config.yaml") -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


# ── Header ────────────────────────────────────────────────────────────────────

class HeaderBar(QWidget):
    def __init__(self, mode: str = "mock", parent=None) -> None:
        super().__init__(parent)
        self._mode = mode
        self.setFixedHeight(54)
        self._build()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(0)

        # Logo hexagon
        logo = QLabel("⬡")
        logo.setFont(QFont("Segoe UI", 18))
        logo.setStyleSheet(f"color: {ACCENT}; background: transparent; border: none;")
        logo.setFixedWidth(30)

        title = QLabel("Apple Sorting  GUI")
        title.setFont(QFont("Segoe UI Variable", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT_1}; background: transparent; border: none;")

        sub = QLabel("  Multispectral Vision System")
        sub.setStyleSheet(
            f"color: {TEXT_3}; font-size: 11px; background: transparent; border: none;"
        )

        layout.addWidget(logo)
        layout.addSpacing(10)
        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addStretch()

        # Mode badge — subtle pill
        if self._mode == "mock":
            badge_bg, badge_bdr, badge_tc = "#1C2240", "#2E3D68", "#6A7899"
            badge_txt = "MOCK MODE"
        else:
            badge_bg, badge_bdr, badge_tc = "#0D2218", "#1A4832", SUCCESS
            badge_txt = "JAI  LIVE"

        self._badge = QLabel(f"  {badge_txt}  ")
        self._badge.setStyleSheet(f"""
            QLabel {{
                background-color: {badge_bg}; color: {badge_tc};
                border: 1px solid {badge_bdr}; border-radius: 5px;
                font-size: 10px; font-weight: 600;
                letter-spacing: 1.5px; padding: 3px 10px;
            }}
        """)
        layout.addWidget(self._badge)

    def set_mode(self, mode: str) -> None:
        """Update badge at runtime to reflect actual camera mode."""
        if mode == "jai":
            bg, bdr, tc, txt = "#0D2218", "#1A4832", SUCCESS, "JAI  LIVE"
        else:
            bg, bdr, tc, txt = "#1C2240", "#2E3D68", "#6A7899", "MOCK MODE"
        self._badge.setText(f"  {txt}  ")
        self._badge.setStyleSheet(f"""
            QLabel {{
                background-color: {bg}; color: {tc};
                border: 1px solid {bdr}; border-radius: 5px;
                font-size: 10px; font-weight: 600;
                letter-spacing: 1.5px; padding: 3px 10px;
            }}
        """)


    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, QColor("#0F172A"))
        grad.setColorAt(0.5, QColor("#162032"))
        grad.setColorAt(1.0, QColor("#0F172A"))
        painter.fillRect(self.rect(), grad)
        pen = painter.pen()
        pen.setColor(QColor(ACCENT + "50"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        painter.end()


# ── Chart placeholder ─────────────────────────────────────────────────────────

class ChartPlaceholder(QWidget):
    def __init__(self, title: str, body: str, color: str, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 8px; border: none;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        accent = QWidget()
        accent.setFixedHeight(2)
        accent.setStyleSheet(f"background-color: {color}40; border: none;")
        layout.addWidget(accent)

        inner = QVBoxLayout()
        inner.setContentsMargins(14, 8, 14, 8)

        hdr = QLabel(title.upper())
        hdr.setStyleSheet(
            f"color: {color}; font-size: 9px; font-weight: 700; "
            f"letter-spacing: 1.5px; background: transparent;"
        )
        inner.addWidget(hdr)

        body_lbl = QLabel(body)
        body_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body_lbl.setStyleSheet(
            f"color: {TEXT_2}; font-size: 11px; font-weight: 500; background: transparent;"
        )
        inner.addWidget(body_lbl, stretch=1)
        layout.addLayout(inner)


# ── Center panel ──────────────────────────────────────────────────────────────

class CenterPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {BG_BASE};")
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(f"""
            QSplitter::handle:vertical {{
                background-color: {BORDER};
            }}
            QSplitter::handle:vertical:hover {{
                background-color: {ACCENT};
            }}
        """)

        self.channel_display = MultiChannelDisplay(self)
        splitter.addWidget(self.channel_display)

        chart_row = QWidget()
        chart_row.setStyleSheet(f"background-color: {BG_BASE};")
        chart_layout = QHBoxLayout(chart_row)
        chart_layout.setContentsMargins(1, 1, 1, 1)
        chart_layout.setSpacing(1)

        chart_layout.addWidget(ChartPlaceholder(
            "Grade Distribution",
            "PyQtGraph live chart  ·  Phase 6",
            ACCENT,
        ))
        chart_layout.addWidget(ChartPlaceholder(
            "Throughput Over Time",
            "Apples / min rolling window  ·  Phase 6",
            SUCCESS,
        ))

        splitter.addWidget(chart_row)
        splitter.setSizes([700, 300])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        root.addWidget(splitter)


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Apple Sorting GUI — fully wired demo pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self._cfg    = _load_config()
        self._mode   = self._cfg.get("camera", {}).get("mode", "mock")
        self._cam_w:  CameraWorker | None        = None
        self._inf_w:  MockInferenceWorker | None = None
        self._total   = 0

        self._setup_window()
        self._build_ui()
        self._connect_signals()
        self._post_init()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle("Apple Sorting GUI  —  MSU ASABE AIM26")
        self.setMinimumSize(1360, 760)
        self.resize(1560, 920)
        self.setStyleSheet(APP_STYLESHEET)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setStyleSheet(f"background-color: {BG_BASE};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = HeaderBar(mode=self._mode)
        root.addWidget(self._header)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {BORDER}; border: none;")
        root.addWidget(sep)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(1)

        self._left   = LeftControlPanel()
        self._center = CenterPanel()
        self._right  = RightStatsPanel()

        self._splitter.addWidget(self._left)
        self._splitter.addWidget(self._center)
        self._splitter.addWidget(self._right)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(2, False)

        root.addWidget(self._splitter, stretch=1)

        status = QStatusBar()
        status.setStyleSheet(
            f"QStatusBar {{ background: {BG_SURFACE}; color: {TEXT_2}; "
            f"font-size: 11px; border-top: 1px solid {BORDER}; padding: 0 10px; }}"
        )
        self.setStatusBar(status)
        status.showMessage(f"Ready  ·  Mode: {self._mode.upper()}")

    def _connect_signals(self) -> None:
        self._left.sig_connect_camera.connect(self._on_camera_toggle)
        self._left.sig_load_model.connect(self._on_load_model)
        self._left.sig_sorter_toggled.connect(self._on_sorter_toggle)
        self._left.sig_logging_toggled.connect(self._on_logging_toggle)
        self._left.sig_speed_changed.connect(self._on_speed_changed)
        # Camera hardware controls — forwarded to CameraWorker while streaming
        self._left.sig_exposure_changed.connect(self._on_exposure_changed)
        self._left.sig_fps_changed.connect(self._on_fps_changed)
        self._left.sig_gains_changed.connect(self._on_gains_changed)

    def _post_init(self) -> None:
        models_dir = Path(self._cfg.get("inference", {}).get("model_dir", "models/"))
        models = [p.name for p in models_dir.glob("*.pt")] + \
                 [p.name for p in models_dir.glob("*.onnx")]
        self._left.populate_models(models)

        sg = self._right.status_group
        sg.set_status("Camera",   "offline", "Disconnected")
        sg.set_status("AI Model", "idle",    "No model")
        sg.set_status("Sorter",   "idle",    "Simulation")
        sg.set_status("Logger",   "idle",    "Off")

    def closeEvent(self, event) -> None:
        """Guarantee camera disconnect on window close — releases eBUS device lock."""
        log.info("MainWindow closing — stopping pipeline …")
        self._stop_pipeline()
        event.accept()



    # ── Camera toggle ─────────────────────────────────────────────────────────

    @pyqtSlot(bool)
    def _on_camera_toggle(self, connect: bool) -> None:
        if connect:
            self._start_pipeline()
        else:
            self._stop_pipeline()

    def _start_pipeline(self) -> None:
        sg = self._right.status_group
        sg.set_status("Camera", "warning", "Connecting…")
        self.statusBar().showMessage("Starting camera pipeline…")

        # ── Camera worker ──────────────────────────────────────────
        display_fps = self._cfg.get("display", {}).get("fps_limit", 24)
        self._cam_w = CameraWorker(
            config      = self._cfg.get("camera", {}),
            display_fps = display_fps,
        )
        self._cam_w.sig_frame.connect(self._on_frame)
        self._cam_w.sig_status.connect(self._on_cam_status)
        self._cam_w.sig_exposure_readback.connect(self._on_exposure_readback)
        self._cam_w.sig_gains_readback.connect(self._on_gains_readback)
        self._cam_w.sig_cam_fps.connect(self._on_cam_fps)
        self._cam_w.start()

        # ── Inference worker ───────────────────────────────────────
        speed = self._left.conveyor_speed
        self._inf_w = MockInferenceWorker(apples_per_sec=speed)
        self._inf_w.sig_grade.connect(self._on_grade)
        self._inf_w.start()

        # ── UI state ───────────────────────────────────────────────
        self._left.set_camera_connected(True)
        sg.set_status("AI Model", "online", "Mock pipeline")
        sg.set_status("Sorter",   "idle",   "Simulation")
        self._right.metrics_group.start_session()
        self._total = 0
        self.statusBar().showMessage(
            f"Pipeline running  ·  {speed} apple/s/lane × 3 lanes"
        )

    def _stop_pipeline(self) -> None:
        if self._inf_w:
            self._inf_w.stop()
            self._inf_w = None
        if self._cam_w:
            self._cam_w.stop()
            self._cam_w = None

        self._left.set_camera_connected(False)
        self._center.channel_display.reset_all()

        sg = self._right.status_group
        sg.set_status("Camera",   "offline", "Disconnected")
        sg.set_status("AI Model", "idle",    "Stopped")
        sg.set_status("Sorter",   "idle",    "Simulation")

        self._right.metrics_group.stop_session()
        self.statusBar().showMessage("Pipeline stopped.")

    # ── Worker signals ────────────────────────────────────────────────────────

    @pyqtSlot(object, object, object, float)
    def _on_frame(self, ch1, ch2, ch3, fps: float) -> None:
        self._center.channel_display.update_frames(ch1, ch2, ch3, fps)

    @pyqtSlot(str, bool)
    def _on_cam_status(self, msg: str, is_error: bool) -> None:
        state = "offline" if is_error else "online"
        self._right.status_group.set_status("Camera", state, msg)
        self.statusBar().showMessage(msg)

        # Update header badge to reflect actual runtime mode
        if "JAI" in msg.upper() and not is_error:
            self._header.set_mode("jai")
        elif "MOCK" in msg.upper() and not is_error:
            self._header.set_mode("mock")


    @pyqtSlot(int, int, str, float, str)
    def _on_grade(
        self,
        apple_id: int,
        lane: int,
        grade: str,
        confidence: float,
        outlet: str,
    ) -> None:
        # Results list
        self._right.results_group.add_result(apple_id, lane, grade, confidence, outlet)

        # Grade summary counts
        self._right.grade_summary.record(grade)

        # Metrics
        speed = self._left.conveyor_speed
        self._right.metrics_group.record_grade(speed)

        # Status bar
        self.statusBar().showMessage(
            f"#{apple_id:04d}  Lane {lane}  →  {grade}  (Outlet {outlet})  "
            f"{confidence * 100:.1f}%"
        )

    # ── Other controls ────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_load_model(self, name: str) -> None:
        self._right.status_group.set_status("AI Model", "warning", "Loading…")
        self.statusBar().showMessage(f"Loading model: {name}")
        # TODO Phase 4: InferenceWorker with real YOLO model

    @pyqtSlot(bool)
    def _on_sorter_toggle(self, enabled: bool) -> None:
        self._right.status_group.set_status(
            "Sorter",
            "online" if enabled else "idle",
            "Active" if enabled else "Simulation",
        )

    @pyqtSlot(bool)
    def _on_logging_toggle(self, enabled: bool) -> None:
        self._right.status_group.set_status(
            "Logger",
            "online" if enabled else "idle",
            "Recording" if enabled else "Off",
        )

    @pyqtSlot(int)
    def _on_speed_changed(self, speed: int) -> None:
        if self._inf_w:
            self._inf_w.set_speed(speed)
            self.statusBar().showMessage(
                f"Speed updated: {speed} apple/s/lane  "
                f"({speed * 3 * 60} apple/min total)"
            )

    @pyqtSlot(int)
    def _on_exposure_changed(self, exposure_us: int) -> None:
        """Forward exposure change to camera while streaming."""
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running:
            self._cam_w.set_exposure(exposure_us)
            self.statusBar().showMessage(
                f"Exposure set: {exposure_us:,} µs  —  "
                f"max at current FPS: {1_000_000 // max(self._left._spn_fps.value(), 1):,} µs"
            )
        else:
            self.statusBar().showMessage("Exposure: camera not connected — connect first")

    @pyqtSlot(float)
    def _on_fps_changed(self, fps: float) -> None:
        """
        Set camera hardware acquisition FPS. Does NOT change display render rate.
        Firmware will silently clamp ExposureTime if current value exceeds
        1,000,000/fps. CameraWorker reads back actual ExposureTime and emits
        sig_exposure_readback to sync the GUI spinbox.
        """
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running:
            self._cam_w.set_fps(fps)
            max_exp = int(1_000_000 / max(fps, 1))
            self.statusBar().showMessage(
                f"Camera hardware FPS set: {fps:.0f} FPS  —  "
                f"max exposure: {max_exp:,} µs"
            )
        else:
            self.statusBar().showMessage("Frame rate: camera not connected — connect first")

    @pyqtSlot(float, float, float)
    def _on_gains_changed(self, ch1_db: float, ch2_db: float, ch3_db: float) -> None:
        """Forward per-channel gains to all 3 camera sources while streaming."""
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running:
            self._cam_w.set_gains(ch1_db, ch2_db, ch3_db)
            self.statusBar().showMessage(
                f"Gain: applying CH1={ch1_db:.1f} CH2={ch2_db:.1f} CH3={ch3_db:.1f} dB…"
            )
        else:
            self.statusBar().showMessage("Gain: camera not connected — connect first")

    @pyqtSlot(float, float, float)
    def _on_gains_readback(self, ch1: float, ch2: float, ch3: float) -> None:
        """
        Receives actual gains read back from firmware after an Apply.
        Updates all 3 gain spinboxes to show real hardware values.
        """
        self._left.update_gains(ch1, ch2, ch3)
        self.statusBar().showMessage(
            f"Gain confirmed —  "
            f"CH1: {ch1:.1f} dB  |  CH2: {ch2:.1f} dB  |  CH3: {ch3:.1f} dB"
        )
        log.info("Gain readback — CH1=%.1f CH2=%.1f CH3=%.1f dB", ch1, ch2, ch3)

    @pyqtSlot(int)
    def _on_exposure_readback(self, actual_us: int) -> None:
        """
        Receives the actual ExposureTime read back from firmware after a FPS change.
        Updates the exposure spinbox to reflect the real (possibly clamped) value.
        This prevents the GUI from showing a value the camera isn't actually using.
        """
        self._left._spn_exposure.setValue(actual_us)
        log.info("Exposure spinbox synced to firmware value: %d µs", actual_us)

    @pyqtSlot(float)
    def _on_cam_fps(self, cam_fps: float) -> None:
        """Shows real camera FPS vs display target in status bar every second."""
        disp = int(self._cam_w._display_fps) if self._cam_w else 0
        self.statusBar().showMessage(
            f"Cam: {cam_fps:.0f} FPS (sensor)   │   "
            f"Display target: {disp} FPS (actual may be less at high FPS due to processing)   │   "
            f"Max exposure at {cam_fps:.0f} FPS: {int(1_000_000 / max(cam_fps, 1)):,} µs"
        )
