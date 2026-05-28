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

import numpy as np
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
from gui.workers.video_worker  import VideoWorker
from gui.workers.inference_worker import MockInferenceWorker, RealInferenceWorker
from gui.workers.tracker import AppleTracker as ConveyorTracker

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

        title = QLabel("Multispectral Vision System")
        title.setFont(QFont("Segoe UI Variable", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT_1}; background: transparent; border: none;")

        layout.addWidget(logo)
        layout.addSpacing(10)
        layout.addWidget(title)
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
        elif mode == "simulation":
            bg, bdr, tc, txt = "#1F1A00", "#4A3C00", "#FBBF24", "VIDEO  SIM"
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


# ── Chart placeholder (used in Analytics tab) ─────────────────────────────────

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


# ── Model Input Panel ─────────────────────────────────────────────────────────

class ModelInputPanel(QWidget):
    """
    Displays the exact spectral composite (e.g. RB+NIR1) that the YOLO model
    sees at inference time, with tracking boxes and grade labels overlaid.

    This makes it immediately clear to the viewer which spectral channels
    drive the AI decision — scientifically honest and great for demo purposes.
    """

    # Band-combo → human-readable description
    _BAND_LABELS: dict[str, str] = {
        "rb-nir1":     "R · B · NIR1     (660nm Red · 660nm Blue · 800nm NIR)",
        "rg-nir1":     "R · G · NIR1     (660nm Red · 660nm Green · 800nm NIR)",
        "r-nir1-nir2": "R · NIR1 · NIR2  (660nm Red · 800nm NIR · 900nm NIR)",
        "rgb":         "RGB  (full color 660nm)",
        "ch1":         "CH1  (raw color sensor)",
    }

    def __init__(self, input_mode: str = "RB-nir1", parent=None) -> None:
        super().__init__(parent)
        self._mode      = input_mode
        self._has_frame = False
        self.setStyleSheet(f"background-color: {BG_BASE};")
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top accent bar (indigo — AI colour) ──────────────────────
        accent = QWidget()
        accent.setFixedHeight(3)
        accent.setStyleSheet(f"background-color: {ACCENT}; border: none;")
        root.addWidget(accent)

        # ── Header row ───────────────────────────────────────────────
        hdr = QWidget()
        hdr.setFixedHeight(32)
        hdr.setStyleSheet(
            f"background-color: {BG_SURFACE}; "
            f"border-bottom: 1px solid {BORDER}; border: none;"
        )
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(12, 0, 12, 0)
        hdr_layout.setSpacing(8)

        icon = QLabel("◆")
        icon.setStyleSheet(f"color: {ACCENT}; font-size: 10px; background: transparent;")

        title_lbl = QLabel("AI MODEL INPUT")
        title_lbl.setStyleSheet(
            f"color: {TEXT_1}; font-size: 10px; font-weight: 700; "
            f"letter-spacing: 1.2px; background: transparent;"
        )

        sep = QLabel("·")
        sep.setStyleSheet(f"color: {TEXT_3}; background: transparent;")

        band_desc = self._BAND_LABELS.get(self._mode.lower(), self._mode)
        self._band_lbl = QLabel(band_desc)
        self._band_lbl.setStyleSheet(
            f"color: {TEXT_2}; font-size: 10px; background: transparent;"
        )

        hdr_layout.addWidget(icon)
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addWidget(sep)
        hdr_layout.addWidget(self._band_lbl)
        hdr_layout.addStretch()

        self._count_lbl = QLabel("0 graded")
        self._count_lbl.setStyleSheet(
            f"color: {SUCCESS}; font-size: 10px; font-weight: 600; "
            f"background: transparent;"
        )
        self._fps_lbl = QLabel("-- FPS")
        self._fps_lbl.setStyleSheet(
            f"color: {TEXT_3}; font-size: 10px; background: transparent;"
        )
        hdr_layout.addWidget(self._count_lbl)
        hdr_layout.addSpacing(12)
        hdr_layout.addWidget(self._fps_lbl)
        root.addWidget(hdr)

        # ── Image area ───────────────────────────────────────────────
        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._display.setStyleSheet(
            "background-color: transparent; border: none;"
        )
        root.addWidget(self._display, stretch=1)

        self._draw_placeholder()

    # ------------------------------------------------------------------
    def _draw_placeholder(self) -> None:
        from PyQt6.QtGui import QPixmap, QPainter
        w = max(self._display.width(), 400)
        h = max(self._display.height(), 160)
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor(BG_BASE))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Segoe UI Variable", 10)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 5)
        painter.setFont(font)
        painter.setPen(QColor(ACCENT + "33"))
        painter.drawText(
            pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "WAITING FOR MODEL"
        )
        painter.end()
        self._display.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self._has_frame:
            self._draw_placeholder()

    # ------------------------------------------------------------------
    def update_frame(
        self, frame: np.ndarray, fps: float, graded_count: int
    ) -> None:
        """Push an annotated model-input composite to the display."""
        import cv2
        from PyQt6.QtGui import QPixmap, QImage

        if frame is None:
            return

        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        h, w = frame.shape[:2]
        if frame.ndim == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            return

        qt_img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        disp_size = self._display.size()
        pixmap = QPixmap.fromImage(qt_img).scaled(
            disp_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(pixmap)
        self._fps_lbl.setText(f"{fps:.1f} FPS")
        self._count_lbl.setText(f"{graded_count} graded")
        self._has_frame = True

    def reset(self) -> None:
        self._has_frame = False
        self._fps_lbl.setText("-- FPS")
        self._count_lbl.setText("0 graded")
        self._draw_placeholder()

    def set_mode(self, input_mode: str) -> None:
        """Update the band-config label in the header — called when a new model is loaded."""
        self._mode = input_mode
        band_desc = self._BAND_LABELS.get(input_mode.lower(), input_mode)
        self._band_lbl.setText(band_desc)


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

        # Top: raw 3-channel sensor display
        self.channel_display = MultiChannelDisplay(self)
        splitter.addWidget(self.channel_display)

        # Bottom: tab widget — AI Model Input | Analytics
        from PyQt6.QtWidgets import QTabWidget
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)   # removes the pane border for a cleaner look

        # ── Tab 0: AI Model Input ──────────────────────────────────────
        cfg = _load_config()
        input_mode = cfg.get("inference", {}).get("input_mode", "RB-nir1")
        self.model_input_panel = ModelInputPanel(input_mode=input_mode)
        self._tabs.addTab(self.model_input_panel, "◆  AI Model Input")

        # ── Tab 1: Analytics (Phase 6 placeholder) ─────────────────────
        analytics_widget = QWidget()
        analytics_widget.setStyleSheet(f"background-color: {BG_BASE};")
        analytics_layout = QHBoxLayout(analytics_widget)
        analytics_layout.setContentsMargins(1, 1, 1, 1)
        analytics_layout.setSpacing(1)
        analytics_layout.addWidget(ChartPlaceholder(
            "Grade Distribution",
            "PyQtGraph live chart  ·  Phase 6",
            ACCENT,
        ))
        analytics_layout.addWidget(ChartPlaceholder(
            "Throughput Over Time",
            "Apples / min rolling window  ·  Phase 6",
            SUCCESS,
        ))
        self._tabs.addTab(analytics_widget, "⬛  Analytics")

        splitter.addWidget(self._tabs)
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
        self._sim_cfg = self._cfg.get("camera", {}).get("simulation", {})
        self._is_sim  = self._sim_cfg.get("enabled", False)
        self._cam_w:   CameraWorker | VideoWorker | None = None
        self._inf_w:   MockInferenceWorker | None        = None
        self._infer_w: RealInferenceWorker | None        = None
        self._tracker: ConveyorTracker | None            = None
        self._infer_fps: float = 0.0
        self._loading_model_name: str = ""
        self._last_ch1:        np.ndarray | None = None
        self._last_ch2:        np.ndarray | None = None
        self._last_ch3:        np.ndarray | None = None
        self._last_input_frame: np.ndarray | None = None  # spectral composite fed to YOLO
        self._exit_x_frac: float = 0.85   # set again in _start_pipeline from config
        self._total        = 0
        self._total_graded = 0             # running count for model input panel badge
        self._wb_reverting = False   # True while a revert_white_balance() call is in flight

        self._setup_window()
        self._build_ui()
        self._connect_signals()
        self._post_init()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle("Infield Apple Sorting System")
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

        self._lbl_sync = QLabel("Sync: --  ·  IDs: -- / -- / --")
        self._lbl_sync.setStyleSheet("font-family: monospace; font-size: 11px; margin-right: 10px;")
        status.addPermanentWidget(self._lbl_sync)

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
        # White balance controls — Source0 (Color CH1) only
        self._left.sig_awb_triggered.connect(self._on_awb_triggered)
        self._left.sig_wb_revert.connect(self._on_wb_revert)
        # Black Level controls — all 3 sources independently
        self._left.sig_black_level_changed.connect(self._on_black_level_changed)
        # ROI controls — all 3 sources simultaneously
        self._left.sig_roi_changed.connect(self._on_roi_changed)
        self._left.sig_roi_reset.connect(self._on_roi_reset)
        self._left.sig_roi_preview.connect(self._on_roi_preview)

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

        # ── Camera / Video worker ──────────────────────────────────
        display_fps = self._cfg.get("display", {}).get("fps_limit", 24)

        if self._is_sim:
            sim_vids = self._sim_cfg.get("videos", {})
            sim_fps  = self._sim_cfg.get("fps", 30)
            sim_loop = self._sim_cfg.get("loop", True)
            self._cam_w = VideoWorker(
                path_ch1 = sim_vids.get("ch1", ""),
                path_ch2 = sim_vids.get("ch2", ""),
                path_ch3 = sim_vids.get("ch3", ""),
                fps      = sim_fps,
                loop     = sim_loop,
            )
            self._header.set_mode("simulation")
        else:
            self._cam_w = CameraWorker(
                config      = self._cfg.get("camera", {}),
                display_fps = display_fps,
            )
            self._cam_w.sig_exposure_readback.connect(self._on_exposure_readback)
            self._cam_w.sig_gains_readback.connect(self._on_gains_readback)
            self._cam_w.sig_cam_fps.connect(self._on_cam_fps)
            self._cam_w.sig_block_ids.connect(self._on_block_ids)
            self._cam_w.sig_wb_readback.connect(self._on_wb_readback)
            self._cam_w.sig_black_level_readback.connect(self._on_black_level_readback)
            self._cam_w.sig_roi_readback.connect(self._on_roi_readback)

        self._cam_w.sig_frame.connect(self._on_frame)
        self._cam_w.sig_status.connect(self._on_cam_status)
        self._cam_w.start()

        # ── Conveyor tracker ───────────────────────────────────────
        inf_cfg      = self._cfg.get("inference", {})
        inf_tracking = inf_cfg.get("tracking", {})
        conv_cfg     = self._cfg.get("conveyor", {})
        self._exit_x_frac = inf_tracking.get("exit_frac", 0.85)
        self._tracker = ConveyorTracker(
            n_lanes              = conv_cfg.get("lanes", 3),
            orientation          = conv_cfg.get("orientation", "BT"),
            exit_frac            = inf_tracking.get("exit_frac",            0.85),
            band_half_frac       = inf_tracking.get("band_half_frac",       0.025),
            entry_frac           = inf_tracking.get("entry_frac",           0.35),
            min_frames           = inf_tracking.get("min_frames",           5),
            max_lost_frames      = inf_tracking.get("max_lost_frames",      10),
            max_recover_dist     = inf_tracking.get("max_recover_dist",     80),
            min_count_dist_frac  = inf_tracking.get("min_count_dist_frac",  0.12),
            count_memory_frames  = inf_tracking.get("count_memory_frames",  40),
            cull_weight          = inf_tracking.get("cull_weight",          1.5),
            hit_threshold        = inf_tracking.get("hit_threshold",        20),
            cull_ratio_threshold = inf_tracking.get("cull_ratio_threshold", 0.55),
        )


        # ── UI state ───────────────────────────────────────────────
        self._left.set_camera_connected(True)
        sg.set_status("AI Model", "idle",   "Waiting for model")
        sg.set_status("Sorter",   "idle",   "Simulation")
        self._right.metrics_group.start_session()
        self._total = 0
        self.statusBar().showMessage("Camera connected  ·  Load a model to start grading")

    def _stop_pipeline(self) -> None:
        if self._infer_w:
            self._infer_w.stop()
            self._infer_w = None
        if self._tracker:
            self._tracker.reset()
        if self._inf_w:
            self._inf_w.stop()
            self._inf_w = None
        if self._cam_w:
            self._cam_w.stop()
            self._cam_w = None

        self._left.set_camera_connected(False)
        self._center.channel_display.reset_all()
        self._center.model_input_panel.reset()
        self._last_input_frame = None
        self._total_graded = 0

        sg = self._right.status_group
        sg.set_status("Camera",   "offline", "Disconnected")
        sg.set_status("AI Model", "idle",    "Stopped")
        sg.set_status("Sorter",   "idle",    "Simulation")

        self._right.metrics_group.stop_session()
        self.statusBar().showMessage("Pipeline stopped.")

    # ── Worker signals ────────────────────────────────────────────────────────

    @pyqtSlot(object, object, object, float)
    def _on_frame(self, ch1, ch2, ch3, fps: float) -> None:
        # Always store latest frames so inference can annotate them
        self._last_ch1 = ch1
        self._last_ch2 = ch2
        self._last_ch3 = ch3

        inference_running = (
            self._infer_w is not None and self._infer_w.isRunning()
        )

        if not inference_running:
            # No inference — show raw video on all channels
            self._center.channel_display.update_frames(ch1, ch2, ch3, fps)

        # Feed frames to inference worker if running
        if inference_running:
            self._infer_w.enqueue(ch1, ch2, ch3)

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
        if not name or "No model" in name:
            return

        # Stop any existing inference worker
        if self._infer_w is not None:
            self._infer_w.stop()
            self._infer_w = None

        # Stop mock worker — real inference takes over all stats from here
        if self._inf_w is not None:
            self._inf_w.stop()
            self._inf_w = None
            # Clear mock data from stats panel so only real data shows
            self._right.grade_summary.reset()
            self._right.results_group.clear_results()
            self._right.metrics_group.reset()

        # Re-read config from disk every time a model is loaded so that changes
        # to config.yaml (e.g. swapping input_mode) take effect without restarting.
        fresh_cfg  = _load_config()
        inf_cfg    = fresh_cfg.get("inference", {})
        models_dir = Path(inf_cfg.get("model_dir", "models/"))
        model_path = str(models_dir / name)
        input_mode = inf_cfg.get("input_mode", "RB-nir1")

        self._right.status_group.set_status("AI Model", "warning", f"Loading {name}...")
        self._left.set_model_loading(True)   # disable button + combo while GPU loads
        self.statusBar().showMessage(f"Loading model: {name}")
        self._loading_model_name = name      # remember for set_model_loaded callback

        self._total_graded = 0
        self._center.model_input_panel.reset()
        self._center.model_input_panel.set_mode(input_mode)   # update header label live
        self._infer_w = RealInferenceWorker(
            model_path     = model_path,
            conf_threshold = inf_cfg.get("confidence_threshold", 0.5),
            iou_threshold  = inf_cfg.get("iou_threshold", 0.45),
            device         = inf_cfg.get("device", "cuda"),
            input_mode     = input_mode,
        )
        self._infer_w.sig_result.connect(self._on_inference_result)
        self._infer_w.sig_input_frame.connect(self._on_input_frame)
        self._infer_w.sig_fps.connect(self._on_inference_fps)
        self._infer_w.sig_status.connect(self._on_inference_status)
        self._infer_w.start()

    # ── Class colours (BGR): 0=Fresh-green  1=Processing-amber  2=Cull-red ──
    _CLASS_COLORS = [
        (52,  211, 153),   # 0 Fresh      — green
        (251, 191,  36),   # 1 Processing — amber
        (248, 113, 113),   # 2 Cull       — red
    ]
    _CLASS_NAMES = ["Fresh", "Processing", "Cull"]
    _OUTLET_MAP  = {"Fresh": "A", "Processing": "B", "Cull": "C"}

    @pyqtSlot(object)
    def _on_inference_result(self, result) -> None:
        """Vote on tracks, commit grades, annotate CH1 with stored latest frame."""
        if self._tracker is None or self._last_ch1 is None:
            return

        active, graded = self._tracker.update(result, self._last_ch1.shape)

        # ── Commit finished grades to stats panel ─────────────────
        for rec in graded:
            outlet = self._OUTLET_MAP.get(rec.class_name, "?")
            self._right.results_group.add_result(
                rec.seq_id, rec.lane, rec.class_name, rec.confidence, outlet
            )
            self._right.grade_summary.record(rec.class_name)
            self._right.metrics_group.record_grade(self._left.conveyor_speed)
            self.statusBar().showMessage(
                f"#{rec.seq_id}  Lane {rec.lane}  →  {rec.class_name}  "
                f"{rec.confidence * 100:.1f}%  ({rec.frames_seen} frames)"
            )

        # ── Annotate all 3 channels with same boxes ───────────────
        ann_ch1 = self._annotate_tracked(self._last_ch1, active, show_label=False)
        ann_ch2 = self._annotate_tracked(self._last_ch2, active, show_label=False)
        ann_ch3 = self._annotate_tracked(self._last_ch3, active, show_label=False)
        fps = self._infer_fps
        # Pass the original full resolution so the UI label stays correct
        orig_shape = (self._last_ch1.shape[1], self._last_ch1.shape[0])
        self._center.channel_display.update_channel_frame(0, ann_ch1, fps, orig_shape)
        self._center.channel_display.update_channel_frame(1, ann_ch2, fps, orig_shape)
        self._center.channel_display.update_channel_frame(2, ann_ch3, fps, orig_shape)

        # ── Push annotated spectral composite to AI Model Input panel ─
        if self._last_input_frame is not None:
            self._total_graded += len(graded)
            ann_input = self._annotate_tracked(self._last_input_frame, active, show_label=True)
            self._center.model_input_panel.update_frame(
                ann_input, fps, self._total_graded
            )

    @pyqtSlot(object)
    def _on_input_frame(self, frame: np.ndarray) -> None:
        """Cache the spectral composite emitted by RealInferenceWorker."""
        self._last_input_frame = frame

    def _annotate_tracked(
        self, frame: np.ndarray, active: list, show_label: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes on a downscaled copy for speed.

        Args:
            frame:      Source numpy frame (any channel layout).
            active:     List of active track dicts from AppleTracker.
            show_label: If True, draw grade + ID pill above each box.
                        If False, draw the coloured box only — used for
                        the raw CH1/CH2/CH3 panels where grading info is
                        shown exclusively in the AI Model Input panel.
        """
        import cv2

        if frame is None:
            return frame

        if frame.ndim == 2:
            out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            out = frame.copy()

        if not active:
            return out

        h, w = out.shape[:2]

        # ── Downscale for fast drawing ─────────────────────────────────────
        DRAW_W = 512
        scale_f = DRAW_W / w
        draw_h  = int(h * scale_f)
        small   = cv2.resize(out, (DRAW_W, draw_h), interpolation=cv2.INTER_LINEAR)

        # Drawing params for 512px canvas
        fs        = 0.55   # font scale
        box_thick = 2
        txt_thick = 1

        for t in active:
            cls      = t["class_id"]
            conf     = t["conf"]
            seq      = t["seq_id"]
            lane     = t["lane"]
            eligible = t.get("eligible", True)
            x1, y1, x2, y2 = t["box"]

            # Scale box coords to draw canvas
            sx1 = int(x1 * scale_f); sy1 = int(y1 * scale_f)
            sx2 = int(x2 * scale_f); sy2 = int(y2 * scale_f)

            color      = self._CLASS_COLORS[cls % len(self._CLASS_COLORS)]
            name       = self._CLASS_NAMES[cls] if cls < len(self._CLASS_NAMES) else str(cls)
            draw_color = color if eligible else (120, 120, 120)

            # Box
            cv2.rectangle(small, (sx1, sy1), (sx2, sy2), draw_color, box_thick)

            # Label pill ─────────────────────────────────────────────────────
            # show_label=True  (AI Model Input panel): "#3 Fresh 87% L2"
            # show_label=False (raw CH1/CH2/CH3):      "#3 L2"  — ID + lane only
            id_part = f"#{seq}" if seq is not None else "?"
            if show_label:
                name  = self._CLASS_NAMES[cls] if cls < len(self._CLASS_NAMES) else str(cls)
                label = f"{id_part} {name} {conf*100:.0f}% L{lane}"
            else:
                label = f"{id_part} L{lane}"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, txt_thick)
            lx = max(0, sx1)
            ly = max(lh + 4, sy1 - 4)

            # Small filled pill behind text
            cv2.rectangle(small, (lx, ly - lh - 3), (lx + lw + 4, ly + 2), draw_color, -1)
            cv2.putText(small, label, (lx + 2, ly - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), txt_thick, cv2.LINE_AA)

        # Return the fast 512px render directly — NO expensive upscale!
        return small


    @pyqtSlot(float)
    def _on_inference_fps(self, fps: float) -> None:
        self._infer_fps = fps
        self._right.metrics_group.set_infer_fps(fps)
        self.statusBar().showMessage(f"Inference: {fps:.1f} FPS")

    @pyqtSlot(str, bool)
    def _on_inference_status(self, msg: str, is_error: bool) -> None:
        state = "offline" if is_error else "online"
        self._right.status_group.set_status("AI Model", state, msg)
        self.statusBar().showMessage(msg)
        if is_error:
            # Re-enable loader so user can try another model
            self._left.set_model_loading(False)
        elif "Model loaded" in msg:
            # Model is in GPU memory and running — update left panel
            name = getattr(self, "_loading_model_name", "")
            self._left.set_model_loaded(name)
            self._left.set_model_loading(False)

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
        if self._tracker:
            self._tracker.set_conveyor_speed(speed)
        self.statusBar().showMessage(
            f"Speed updated: {speed} apple/s/lane  "
            f"({speed * 3 * 60} apple/min total)"
        )

    @pyqtSlot(int, int, int)
    def _on_exposure_changed(self, ch1_us: int, ch2_us: int, ch3_us: int) -> None:
        """Forward independent per-channel exposure changes to camera while streaming."""
        if self._is_sim:
            return  # no hardware to update in simulation mode
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running:
            self._cam_w.set_exposures(ch1_us, ch2_us, ch3_us)
            self.statusBar().showMessage(
                f"Exposure: applying CH1={ch1_us:,} CH2={ch2_us:,} CH3={ch3_us:,} µs…"
            )
        else:
            self.statusBar().showMessage("Exposure: camera not connected — connect first")

    @pyqtSlot(float)
    def _on_fps_changed(self, fps: float) -> None:
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running and isinstance(self._cam_w, CameraWorker):
            self._cam_w.set_fps(fps)
            max_exp = int(1_000_000 / max(fps, 1))
            self.statusBar().showMessage(
                f"Camera hardware FPS set: {fps:.0f} FPS  "
                f"max exposure: {max_exp:,} µs"
            )
        elif self._is_sim:
            self.statusBar().showMessage("FPS: not applicable in video simulation mode")
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

    @pyqtSlot(int, int, int)
    def _on_exposure_readback(self, ch1: int, ch2: int, ch3: int) -> None:
        """
        Receives the actual ExposureTimes read back from firmware after an Apply or FPS change.
        Updates all 3 exposure spinboxes to reflect the real (possibly clamped) values.
        """
        self._left.update_exposures(ch1, ch2, ch3)
        self.statusBar().showMessage(
            f"Exposure confirmed —  "
            f"CH1: {ch1:,} µs  |  CH2: {ch2:,} µs  |  CH3: {ch3:,} µs"
        )
        log.info("Exposure readback — CH1=%d CH2=%d CH3=%d µs", ch1, ch2, ch3)

    @pyqtSlot(float)
    def _on_cam_fps(self, cam_fps: float) -> None:
        """Shows real camera FPS vs display target in status bar every second."""
        disp = int(self._cam_w._display_fps) if self._cam_w else 0
        self.statusBar().showMessage(
            f"Cam: {cam_fps:.0f} FPS (sensor)   │   "
            f"Display target: {disp} FPS (actual may be less at high FPS due to processing)   │   "
            f"Max exposure at {cam_fps:.0f} FPS: {int(1_000_000 / max(cam_fps, 1)):,} µs"
        )

    @pyqtSlot(bool, int, int, int)
    def _on_block_ids(self, synced: bool, ch1_bid: int, ch2_bid: int, ch3_bid: int) -> None:
        """Updates the permanent sync status label on the right corner of the status bar."""
        if ch1_bid == -1:
            self._lbl_sync.setText("Sync: --  ·  IDs: -- / -- / --")
            self._lbl_sync.setStyleSheet("color: #94A3B8; font-family: monospace; font-size: 11px; margin-right: 10px;")
            return
        
        sync_str = "OK" if synced else "MISMATCH"
        color = "#10B981" if synced else "#EF4444"  # emerald green if OK, bright red if MISMATCH
        self._lbl_sync.setText(f"Sync: {sync_str}  ·  IDs: {ch1_bid} / {ch2_bid} / {ch3_bid}")
        self._lbl_sync.setStyleSheet(f"color: {color}; font-family: monospace; font-size: 11px; font-weight: bold; margin-right: 10px;")

    # ── White Balance slots ────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_awb_triggered(self) -> None:
        """
        Forward One-Push AWB trigger to CameraWorker.
        The worker runs trigger_auto_white_balance() (blocking ~0–3 s in its own
        thread), then emits sig_wb_readback which calls _on_wb_readback here.
        """
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running:
            self.statusBar().showMessage(
                "White Balance: running One-Push calibration on Color channel…"
            )
            self._cam_w.trigger_awb()
        else:
            self.statusBar().showMessage(
                "White Balance: camera not connected — connect first"
            )

    @pyqtSlot()
    def _on_wb_revert(self) -> None:
        """Revert WB to the ratios saved before the last One-Push AWB."""
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running:
            self.statusBar().showMessage(
                "White Balance: reverting to pre-calibration ratios…"
            )
            self._wb_reverting = True   # flag so _on_wb_readback disables Revert button
            self._cam_w.revert_white_balance()
        else:
            self.statusBar().showMessage(
                "White Balance: camera not connected — connect first"
            )

    @pyqtSlot(bool, float, float, float)
    def _on_wb_readback(
        self, success: bool, r: float, g: float, b: float
    ) -> None:
        """
        Called after AWB calibration completes or Revert finishes.
        Updates the WB ratio display and status bar.
        When the readback follows a Revert, the Revert button is disabled again
        because _saved_wb has been cleared in camera_interface — no snapshot remains.
        """
        was_revert = self._wb_reverting
        self._wb_reverting = False   # reset flag for next operation
        self._left.update_white_balance(success, r, g, b, revert_done=was_revert)
        if success:
            self.statusBar().showMessage(
                f"White Balance confirmed — R: {r:.4f}  G: {g:.4f}  B: {b:.4f}"
            )
            log.info("WB readback — R=%.4f G=%.4f B=%.4f", r, g, b)
        else:
            self.statusBar().showMessage(
                "White Balance: calibration failed — check connection and try again"
            )
            log.warning("WB readback: failed")

    # ── Black Level slots ──────────────────────────────────────────────────────

    @pyqtSlot(float, float, float)
    def _on_black_level_changed(
        self, ch1: float, ch2: float, ch3: float
    ) -> None:
        """Forward per-channel black level changes to CameraWorker while streaming."""
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running:
            self._cam_w.set_black_levels(ch1, ch2, ch3)
            self.statusBar().showMessage(
                f"Black Level: applying CH1={ch1:.1f}  CH2={ch2:.1f}  CH3={ch3:.1f} DN…"
            )
        else:
            self.statusBar().showMessage(
                "Black Level: camera not connected — connect first"
            )

    @pyqtSlot(float, float, float)
    def _on_black_level_readback(
        self, ch1: float, ch2: float, ch3: float
    ) -> None:
        """
        Receives actual BlackLevel values read back from firmware after an Apply.
        Updates all 3 spinboxes to reflect the real (possibly clamped) values.
        """
        self._left.update_black_levels(ch1, ch2, ch3)
        self.statusBar().showMessage(
            f"Black Level confirmed — CH1: {ch1:.1f}  CH2: {ch2:.1f}  CH3: {ch3:.1f} DN"
        )
        log.info("Black Level readback — CH1=%.1f CH2=%.1f CH3=%.1f DN", ch1, ch2, ch3)

    # ── ROI slots ──────────────────────────────────────────────────────────────

    @pyqtSlot(int, int, int, int)
    def _on_roi_changed(
        self, offset_x: int, offset_y: int, width: int, height: int
    ) -> None:
        """Forward ROI Apply to CameraWorker while streaming."""
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running:
            self._cam_w.set_roi(offset_x, offset_y, width, height)
            pct = round(100 * width * height / (2048 * 1536))
            self.statusBar().showMessage(
                f"ROI: applying {width}×{height} @ ({offset_x}, {offset_y}) "
                f"— {pct}% of full frame…"
            )
        else:
            self.statusBar().showMessage(
                "ROI: camera not connected — connect first"
            )

    @pyqtSlot()
    def _on_roi_reset(self) -> None:
        """Reset ROI to full 2048×1536 frame via CameraWorker."""
        running = self._cam_w is not None and self._cam_w.isRunning()
        if running:
            self._cam_w.reset_roi()
            self.statusBar().showMessage("ROI: resetting to full frame 2048×1536…")
        else:
            # Update spinboxes to full frame values even if not connected
            self._left.update_roi(0, 0, 2048, 1536)
            self.statusBar().showMessage("ROI reset to full frame (camera not connected)")

    @pyqtSlot(int, int, int, int)
    def _on_roi_readback(
        self, x: int, y: int, w: int, h: int
    ) -> None:
        """
        Receives actual ROI values confirmed by firmware after Apply/Reset.
        Syncs spinboxes to the step-aligned, clamped values the camera accepted.
        """
        self._left.update_roi(x, y, w, h)
        pct = round(100 * w * h / (2048 * 1536))
        full = (x == 0 and y == 0 and w == 2048 and h == 1536)
        if full:
            msg = "ROI confirmed — Full Frame 2048×1536"
        else:
            msg = (f"ROI confirmed — {w}×{h} px @ ({x}, {y})  "
                   f"[{pct}% of frame — {(100-pct)}% data saved]")
        self.statusBar().showMessage(msg)
        log.info("ROI readback — x=%d y=%d w=%d h=%d (%d%% of frame)", x, y, w, h, pct)
        # Update the active ROI reference so future overlay previews map correctly
        # against the new frame content, then clear the preview border.
        self._center.channel_display.set_active_roi(x, y, w, h)
        self._center.channel_display.clear_roi_preview()

    @pyqtSlot(int, int, int, int)
    def _on_roi_preview(
        self, ox: int, oy: int, w: int, h: int
    ) -> None:
        """
        Instantly update the ROI cut-line overlay on the live camera display
        as the user moves the spinboxes — no Apply click required to see the
        effect on the image.
        """
        self._center.channel_display.set_roi_preview(ox, oy, w, h)
