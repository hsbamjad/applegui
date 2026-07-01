"""
gui/main_window.py
==================
Main application window - fully wired demo pipeline.

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

import threading
import time

from core.log import get_logger
from pathlib import Path

import numpy as np
import yaml
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QStatusBar, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QFont, QColor, QPainter, QLinearGradient

from gui.styles import (
    APP_STYLESHEET, BG_BASE, BG_SURFACE, BG_CARD,
    ACCENT, ACCENT_HV, ACCENT_DK, SUCCESS, WARNING, DANGER, INFO,
    TEXT_1, TEXT_2, TEXT_3, BORDER, CH_COLORS,
)
from gui.panels.camera_panel import LeftControlPanel
from gui.panels.stats_panel      import RightStatsPanel
from gui.panels.analytics_panel  import AnalyticsPanel
from gui.panels.logs_panel       import LogsPanel
from gui.widgets.panel_header import PanelHeaderBar
from gui.widgets.image_display import MultiChannelDisplay
from gui.workers.camera_worker import CameraWorker
from gui.workers.video_worker  import VideoWorker
from gui.workers.inference_worker import MockInferenceWorker, RealInferenceWorker
from gui.workers.tracker import AppleTracker as ConveyorTracker
from gui.drawing import annotate_tracked
from core.control import SorterController
from core.logging import GradingRecorder

log = get_logger(__name__)


def _load_config(path=None) -> dict:
    """Load config.yaml - defaults to the canonical location via utils.paths."""
    from utils.paths import CONFIG_PATH
    resolved = Path(path) if path else CONFIG_PATH
    try:
        with open(resolved, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("config.yaml not found at %s", resolved)
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

        # Mode badge - subtle pill
        if self._mode == "mock":
            badge_bg, badge_bdr, badge_tc = "#1A2A1C", "#2C3E2F", "#536550"
            badge_txt = "MOCK MODE"
        else:
            badge_bg, badge_bdr, badge_tc = "#163320", "#2A5535", SUCCESS
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
            bg, bdr, tc, txt = "#163320", "#2A5535", SUCCESS, "JAI  LIVE"
        elif mode == "simulation":
            bg, bdr, tc, txt = "#2A1E08", "#5A3E10", "#D4A843", "VIDEO  SIM"
        else:
            bg, bdr, tc, txt = "#1A2A1C", "#2C3E2F", "#536550", "MOCK MODE"
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
        grad.setColorAt(0.0, QColor("#0A1209"))
        grad.setColorAt(0.5, QColor("#122015"))
        grad.setColorAt(1.0, QColor("#0A1209"))
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
    drive the AI decision - scientifically honest and great for demo purposes.
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
        self._disp_times: list[float] = []
        self._last_display_fps = 0.0
        self.setStyleSheet(f"background-color: {BG_BASE};")
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        band_desc = self._BAND_LABELS.get(self._mode.lower(), self._mode)
        self._count_lbl = QLabel("0 graded")
        self._count_lbl.setStyleSheet(
            f"color: {SUCCESS}; font-size: 10px; font-weight: 600; "
            f"background: transparent;"
        )
        self._header = PanelHeaderBar(
            "◆", "AI Model Input", band_desc, right=self._count_lbl,
        )
        self._band_lbl = self._header.subtitle_label()
        root.addWidget(self._header)

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
        self, frame: np.ndarray, graded_count: int,
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
        self._last_display_fps = self._measure_display_fps()
        self._count_lbl.setText(f"{graded_count} graded")
        self._has_frame = True

    def _measure_display_fps(self) -> float:
        import time

        now = time.perf_counter()
        self._disp_times.append(now)
        cutoff = now - 1.0
        self._disp_times = [t for t in self._disp_times if t >= cutoff]
        if len(self._disp_times) < 2:
            return 0.0
        span = self._disp_times[-1] - self._disp_times[0]
        return (len(self._disp_times) - 1) / span if span > 1e-6 else 0.0

    def reset(self) -> None:
        self._has_frame = False
        self._disp_times.clear()
        self._last_display_fps = 0.0
        self._count_lbl.setText("0 graded")
        self._draw_placeholder()

    def set_mode(self, input_mode: str) -> None:
        """Update the band-config label in the header - called when a new model is loaded."""
        self._mode = input_mode
        band_desc = self._BAND_LABELS.get(input_mode.lower(), input_mode)
        self._header.set_subtitle(band_desc)


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

        # Bottom: tab widget - AI Model Input | Analytics | Logs
        from PyQt6.QtWidgets import QTabWidget
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)   # removes the pane border for a cleaner look

        # ── Tab 0: AI Model Input ──────────────────────────────────────
        cfg = _load_config()
        input_mode = cfg.get("inference", {}).get("input_mode", "RB-nir1")
        self.model_input_panel = ModelInputPanel(input_mode=input_mode)
        self._tabs.addTab(self.model_input_panel, "◆  AI Model Input")

        # ── Tab 1: Analytics - live PyQtGraph charts ──────────────────
        self.analytics_panel = AnalyticsPanel()
        self._tabs.addTab(self.analytics_panel, "▣  Analytics")

        # ── Tab 2: System Logs ─────────────────────────────────────────
        self.logs_panel = LogsPanel()
        self._tabs.addTab(self.logs_panel, "▤  Logs")

        splitter.addWidget(self._tabs)
        splitter.setSizes([700, 300])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        root.addWidget(splitter)


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Apple Sorting GUI - fully wired demo pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self._cfg    = _load_config()
        self._mode   = self._cfg.get("camera", {}).get("mode", "mock")
        self._sim_cfg = self._cfg.get("camera", {}).get("simulation", {})
        self._is_sim  = self._sim_cfg.get("enabled", False)
        self._cam_w:   CameraWorker | VideoWorker | None = None
        self._inf_w:   MockInferenceWorker | None        = None
        self._infer_w: RealInferenceWorker | None        = None
        self._tracker:  ConveyorTracker | None            = None
        self._sorter:   SorterController | None           = None
        self._sorting_enabled: bool                       = False   # gated by Enable Sorting toggle
        self._logging_enabled: bool                        = False   # gated by Save mode (backward compat)
        self._save_mode:       bool                        = False   # Save mode: enables disk logging
        self._detect_mode:     bool                        = False   # Detect mode: enables inference
        self._log_raw:         bool                        = True    # log Raw Frames (default ON)
        self._log_detected:    bool                        = False   # log Detected Frames (full-res + boxes)
        self._custom_save_path: str                        = ""      # operator-chosen base dir ("" = use config)
        self._grading_recorder: GradingRecorder | None   = None
        self._size_acc = None   # AppleSizeAccumulator - created in _start_pipeline
        self._infer_fps: float = 0.0
        self._loading_model_name: str = ""
        self._last_ch1:        np.ndarray | None = None
        self._last_ch2:        np.ndarray | None = None
        self._last_ch3:        np.ndarray | None = None
        self._last_input_frame: np.ndarray | None = None
        self._exit_x_frac: float = 0.85   # set again in _start_pipeline from config
        self._total        = 0
        self._total_graded = 0             # running count for model input panel badge
        self._wb_reverting = False   # True while a revert_white_balance() call is in flight
        self._display_pending = False
        self._graded_ui_pending = False
        self._graded_ui_queue: list = []
        self._size_lock = threading.Lock()
        self._preview_ui_pending = False
        self._cam_fps_reported = 0.0
        self._frame_coalesce_pending = False
        self._pending_frame: tuple | None = None

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(33)   # ~30 Hz channel refresh, off hot path
        self._preview_timer.timeout.connect(self._flush_channel_preview)

        self._setup_window()
        self._build_ui()
        self._connect_signals()
        self._post_init()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle("Infield Apple Sorting System")
        self.setMinimumSize(1640, 760)
        self.resize(1720, 920)
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

        # Left panel is dual-column (~549 px); give center the remaining space.
        self._splitter.setSizes([549, 900, 256])

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
        log.info("Application started  ·  mode=%s", self._mode)

    def _connect_signals(self) -> None:
        self._left.sig_connect_camera.connect(self._on_camera_toggle)
        self._left.sig_load_model.connect(self._on_load_model)
        self._left.sig_unload_model.connect(self._on_unload_model)
        self._left.sig_sorter_toggled.connect(self._on_sorter_toggle)
        self._left.sig_save_mode_changed.connect(self._on_save_mode_changed)
        self._left.sig_detect_mode_changed.connect(self._on_detect_mode_changed)
        self._left.sig_logging_options.connect(self._on_logging_options)
        self._left.sig_save_path_changed.connect(self._on_save_path_changed)
        self._left.sig_save_interval_changed.connect(self._on_save_interval_changed)
        self._left.sig_save_resolution_changed.connect(self._on_save_resolution_changed)
        # Camera hardware controls - forwarded to CameraWorker while streaming
        self._left.sig_exposure_changed.connect(self._on_exposure_changed)
        self._left.sig_fps_changed.connect(self._on_fps_changed)
        self._left.sig_gains_changed.connect(self._on_gains_changed)
        # White balance controls - Source0 (Color CH1) only
        self._left.sig_awb_triggered.connect(self._on_awb_triggered)
        self._left.sig_wb_revert.connect(self._on_wb_revert)
        # Black Level controls - all 3 sources independently
        self._left.sig_black_level_changed.connect(self._on_black_level_changed)
        # ROI controls - all 3 sources simultaneously
        self._left.sig_roi_changed.connect(self._on_roi_changed)
        self._left.sig_roi_reset.connect(self._on_roi_reset)
        self._left.sig_roi_preview.connect(self._on_roi_preview)
        # Analytics → right panel throughput sync
        self._center.analytics_panel.throughput_chart.sig_throughput_updated.connect(
            self._on_throughput_updated
        )
        # GT ID Mode toggle → tracker
        self._right.gt_id_toggle.sig_gt_id_mode_changed.connect(self._on_gt_id_mode_changed)


    def _post_init(self) -> None:
        from utils.paths import APP_ROOT, MODELS_DIR
        raw_dir = self._cfg.get("inference", {}).get("model_dir", "models/")
        # If config gives a relative path, resolve it against APP_ROOT
        models_dir = Path(raw_dir) if Path(raw_dir).is_absolute() else APP_ROOT / raw_dir
        if not models_dir.exists():
            models_dir = MODELS_DIR  # fallback to canonical location
        models = [p.name for p in models_dir.glob("*.pt")] + \
                 [p.name for p in models_dir.glob("*.onnx")]
        self._left.populate_models(models)

        sg = self._right.status_group
        sg.set_status("Camera",   "offline", "Disconnected")
        sg.set_status("AI Model", "idle",    "No model")
        sg.set_status("Sorter",   "idle",    "Simulation")
        sg.set_status("Logger",   "idle",    "Off")

        log_cfg = self._cfg.get("logging", {})
        if log_cfg.get("enabled", False):
            self._save_mode = True
            self._logging_enabled = True
            self._left.set_save_mode(True)   # sync button state without emitting signal
            self._right.status_group.set_status("Logger", "idle", "Armed")
        self._left.set_save_max_dim(int(log_cfg.get("save_max_dim", 0)))

    def closeEvent(self, event) -> None:
        """Guarantee camera disconnect on window close - releases eBUS device lock."""
        log.info("MainWindow closing - stopping pipeline …")
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
        self._preview_timer.start()

        # ── Conveyor tracker ───────────────────────────────────────
        inf_cfg      = self._cfg.get("inference", {})
        inf_tracking = inf_cfg.get("tracking", {})
        conv_cfg     = self._cfg.get("conveyor", {})
        self._exit_x_frac = inf_tracking.get("exit_frac", 0.85)
        self._tracker = ConveyorTracker(
            n_lanes                     = conv_cfg.get("lanes", 3),
            orientation                 = conv_cfg.get("orientation", "BT"),
            exit_frac                   = inf_tracking.get("exit_frac",                    0.85),
            band_half_frac              = inf_tracking.get("band_half_frac",               0.025),
            entry_frac                  = inf_tracking.get("entry_frac",                   0.35),
            min_frames                  = inf_tracking.get("min_frames",                   15),
            max_lost_frames             = inf_tracking.get("max_lost_frames",              10),
            max_recover_dist            = inf_tracking.get("max_recover_dist",             80),
            min_count_dist_frac         = inf_tracking.get("min_count_dist_frac",          0.12),
            count_memory_frames         = inf_tracking.get("count_memory_frames",          40),
            count_merge_frames          = inf_tracking.get("count_merge_frames",            5),
            cull_weight                 = inf_tracking.get("cull_weight",                  1.0),
            hit_threshold               = inf_tracking.get("hit_threshold",                25),
            cull_ratio_threshold        = inf_tracking.get("cull_ratio_threshold",         0.65),
            peak_conf_override          = inf_tracking.get("peak_conf_override",           0.50),
            overwhelming_cull_threshold = inf_tracking.get("overwhelming_cull_threshold",  60),
        )

        # ── Apple size accumulator ─────────────────────────────────
        size_cfg = self._cfg.get("sizing", {})
        if size_cfg.get("enabled", True):
            from core.sizing.accumulator import AppleSizeAccumulator
            from utils.paths import APP_ROOT, MODELS_DIR
            raw_size_path = size_cfg.get("model_path", "models/size_model.pkl")
            size_model_path = (
                Path(raw_size_path) if Path(raw_size_path).is_absolute()
                else APP_ROOT / raw_size_path
            )
            if not size_model_path.exists():
                size_model_path = MODELS_DIR / "size_model.pkl"
            self._size_acc = AppleSizeAccumulator(
                model_path    = str(size_model_path),
                min_frames    = size_cfg.get("min_frames", 4),
                bg_angle_step = size_cfg.get("bg_angle_step", 10),
            )
        else:
            self._size_acc = None

        self._left.set_camera_connected(True)
        
        # If the inference worker is already running, preserve its "online" status
        # and push the newly created tracker and size accumulator to it.
        if self._infer_w is not None and getattr(self._infer_w, "_running", False):
            sg.set_status("AI Model", "online", "Model loaded successfully")
            self._infer_w.set_tracker(self._tracker)
            self._infer_w.set_size_acc(self._size_acc)
            self.statusBar().showMessage("Camera connected  ·  Grading active")
        else:
            sg.set_status("AI Model", "idle",   "Waiting for model")
            self.statusBar().showMessage("Camera connected  ·  Load a model to start grading")

        sg.set_status("Sorter",   "idle",   "Simulation")
        self._right.metrics_group.start_session()
        self._center.analytics_panel.start()
        self._total = 0

        # ── Sorter controller ─────────────────────────────────────
        self._sorter = SorterController(
            sorter_cfg   = self._cfg.get("sorter", {}),
            conveyor_cfg = self._cfg.get("conveyor", {}),
        )
        self._sorter.start()
        log.info("SorterController started (mode=%s)", self._cfg.get("sorter", {}).get("mode", "simulation"))

        if self._save_mode and self._grading_recorder is None:
            self._start_grading_session()

    def _start_grading_session(self) -> None:
        """Create GradingRecorder and a timestamped session folder."""
        from utils.paths import APP_ROOT, SESSIONS_DIR

        log_cfg = self._cfg.get("logging", {})

        # Prefer the operator-selected custom path over config default
        if self._custom_save_path:
            base_dir = Path(self._custom_save_path)
        else:
            raw_out  = log_cfg.get("output_dir", "data/sessions")
            base_dir = Path(raw_out) if Path(raw_out).is_absolute() else APP_ROOT / raw_out
            if not base_dir.exists():
                base_dir = SESSIONS_DIR

        self._grading_recorder = GradingRecorder(
            image_format          = log_cfg.get("image_format", "jpg"),
            jpeg_quality          = int(log_cfg.get("jpeg_quality", 92)),
            save_detected_crops   = self._log_detected,
            crop_padding_frac     = float(log_cfg.get("crop_padding_frac", 0.20)),
            raw_frame_stride      = self._left.get_save_interval(),
            save_max_dim          = self._left.get_save_max_dim(),
            save_raw_full_frames  = self._log_raw,
            max_pending_batches   = int(log_cfg.get("max_pending_batches", 2)),
            max_crops_per_batch   = int(log_cfg.get("max_crops_per_batch", 8)),
            heavy_threshold       = int(log_cfg.get("heavy_threshold", 12)),
        )
        session_dir = self._grading_recorder.start_session(base_dir)
        self._wire_infer_logging()
        try:
            rel = session_dir.relative_to(APP_ROOT)
        except ValueError:
            rel = session_dir
        self._left.set_logging_path(str(rel))
        self._right.status_group.set_status("Logger", "online", "Recording")

    def _stop_grading_session(self) -> None:
        """Flush and tear down the grading recorder."""
        if self._grading_recorder is not None:
            self._grading_recorder.stop_session()
            self._grading_recorder = None
        self._wire_infer_logging()
        log_cfg = self._cfg.get("logging", {})
        raw_out = log_cfg.get("output_dir", "data/sessions")
        self._left.set_logging_path(raw_out)
        if self._save_mode:
            self._right.status_group.set_status("Logger", "idle", "Armed")
        else:
            self._right.status_group.set_status("Logger", "idle", "Off")

    def _stop_pipeline(self) -> None:
        self._preview_timer.stop()
        self._stop_grading_session()
        if self._sorter:
            self._sorter.stop()
            self._sorter = None
        self._sorting_enabled = False
        self._left.set_sorter_enabled(False)   # uncheck toggle on disconnect

        # ── Auto-unload AI model on disconnect ────────────────────────
        if self._infer_w:
            self._infer_w.stop()
            self._infer_w = None
        if self._inf_w:
            self._inf_w.stop()
            self._inf_w = None
        # Notify left panel so the Load button resets to its idle state
        self._left.set_model_loaded("")

        if self._size_acc is not None:
            self._size_acc.clear()
            self._size_acc = None
        if self._tracker:
            self._tracker.reset()
        if self._cam_w:
            self._cam_w.stop()
            self._cam_w = None

        self._left.set_camera_connected(False)
        self._center.channel_display.reset_all()
        self._center.model_input_panel.reset()
        self._last_input_frame = None
        self._total_graded = 0

        # ── Clear right panel results and logs on disconnect ──────────
        self._right.results_group.clear_results()
        self._right.grade_summary.reset()
        self._right.metrics_group.reset()
        self._center.analytics_panel.reset()
        self._center.logs_panel.clear()

        sg = self._right.status_group
        sg.set_status("Camera",   "offline", "Disconnected")
        sg.set_status("AI Model", "idle",    "No model")
        sg.set_status("Sorter",   "idle",    "Simulation")

        self._center.analytics_panel.stop()
        self.statusBar().showMessage("Pipeline stopped.")
        log.info("Pipeline stopped - panel data cleared, model unloaded.")

    # ── Worker signals ────────────────────────────────────────────────────────

    _PREVIEW_W = 512

    @pyqtSlot(object, object, object, float)
    def _on_frame(self, ch1, ch2, ch3, fps: float) -> None:
        """Coalesce bursts - always keep latest frame, never flood infer/GUI."""
        self._pending_frame = (ch1, ch2, ch3, fps)
        if not self._frame_coalesce_pending:
            self._frame_coalesce_pending = True
            QTimer.singleShot(0, self._apply_pending_frame)

    def _apply_pending_frame(self) -> None:
        self._frame_coalesce_pending = False
        pf = self._pending_frame
        if pf is None:
            return

        ch1, ch2, ch3, fps = pf
        self._last_ch1 = ch1
        self._last_ch2 = ch2
        self._last_ch3 = ch3
        self._cam_fps_reported = fps

        # Submit raw full-frames for logging when Save mode + Raw Frames option is active
        if (
            self._save_mode
            and self._log_raw
            and self._grading_recorder is not None
            and self._grading_recorder._active
        ):
            self._grading_recorder.submit_raw_frame(ch1, ch2, ch3)

        if self._infer_w is not None and self._infer_w.isRunning():
            self._infer_w.enqueue(ch1, ch2, ch3)
        else:
            self._center.channel_display.update_frames(ch1, ch2, ch3, fps)

        # A newer frame arrived while we were processing - apply it once more.
        if self._pending_frame is not pf:
            self._frame_coalesce_pending = True
            QTimer.singleShot(0, self._apply_pending_frame)

    def _wire_infer_logging(self) -> None:
        if self._infer_w is None:
            return
        # Attach the recorder to inference when:
        #   - Save mode is ON, AND
        #   - Either Detect mode is ON, OR Detected crops are selected
        #     (crops need inference to run even if user didn't press Detect button)
        crops_needed = self._log_detected
        infer_needed = self._detect_mode or crops_needed
        rec = (
            self._grading_recorder
            if self._save_mode and infer_needed
            else None
        )
        self._infer_w.set_grading_recorder(rec)

    def _flush_channel_preview(self) -> None:
        """Timer-driven channel refresh - never blocks camera or inference enqueue."""
        if self._last_ch1 is None:
            return
        if self._infer_w is not None and self._infer_w.isRunning():
            fps = self._cam_fps_reported
            orig_shape = (self._last_ch1.shape[1], self._last_ch1.shape[0])
            for idx, frame in enumerate((self._last_ch1, self._last_ch2, self._last_ch3)):
                self._center.channel_display.update_channel_frame(
                    idx, self._downscale_preview(frame), fps, orig_shape,
                )

    def _downscale_preview(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if frame is None:
            return frame
        h, w = frame.shape[:2]
        nh = max(1, int(h * (self._PREVIEW_W / w)))
        return cv2.resize(
            frame, (self._PREVIEW_W, nh), interpolation=cv2.INTER_LINEAR,
        )

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
        self._right.metrics_group.record_grade()
        self._center.analytics_panel.record_grade(grade)

        # Status bar
        self.statusBar().showMessage(
            f"#{apple_id:04d}  Lane {lane}  →  {grade}  (Outlet {outlet})  "
            f"{confidence * 100:.1f}%"
        )


    # ── GT ID Mode ────────────────────────────────────────────────────────────

    @pyqtSlot(bool)
    def _on_gt_id_mode_changed(self, enabled: bool) -> None:
        """Forward GT ID mode toggle to the live tracker."""
        if self._tracker is not None:
            self._tracker.set_gt_id_mode(enabled)
        mode_str = "GT interleaved" if enabled else "Normal (global sequential)"
        log.info("Apple ID mode switched to: %s", mode_str)
        self.statusBar().showMessage(
            f"ID mode: {'GT interleaved lane IDs' if enabled else 'Global sequential (#1, #2, #3…)'}"
        )

    # ── Other controls ────────────────────────────────────────────────────────


    @pyqtSlot()
    def _on_unload_model(self) -> None:
        """Unload the AI model and stop the inference worker."""
        if self._infer_w is not None:
            self._infer_w.stop()
            self._infer_w = None
        
        # Reset UI states
        self._left.set_model_loaded("")
        self._right.status_group.set_status("AI Model", "idle", "No model loaded")
        self.statusBar().showMessage("AI model unloaded")
        
        # Re-evaluate grading recorder wiring
        self._wire_infer_logging()
        
        # Unpause video if we are in video sim
        if isinstance(self._cam_w, VideoWorker):
            self._cam_w.resume()

    @pyqtSlot(str)
    def _on_load_model(self, name: str) -> None:
        if not name or "No model" in name:
            return

        # Stop any existing inference worker
        if self._infer_w is not None:
            self._infer_w.stop()
            self._infer_w = None

        # Stop mock worker - real inference takes over all stats from here
        if self._inf_w is not None:
            self._inf_w.stop()
            self._inf_w = None
            # Clear mock data from stats panel so only real data shows
            self._right.grade_summary.reset()
            self._right.results_group.clear_results()
            self._right.metrics_group.reset()
            self._center.analytics_panel.reset()

        # Re-read config from disk every time a model is loaded so that changes
        # to config.yaml (e.g. swapping input_mode) take effect without restarting.
        fresh_cfg  = _load_config()
        inf_cfg    = fresh_cfg.get("inference", {})
        from utils.paths import APP_ROOT, MODELS_DIR
        raw_dir    = inf_cfg.get("model_dir", "models/")
        models_dir = Path(raw_dir) if Path(raw_dir).is_absolute() else APP_ROOT / raw_dir
        if not models_dir.exists():
            models_dir = MODELS_DIR
        model_path = str(models_dir / name)
        input_mode = inf_cfg.get("input_mode", "RB-nir1")

        self._right.status_group.set_status("AI Model", "warning", f"Loading {name}...")
        self._left.set_model_loading(True)   # disable button + combo while GPU loads
        self.statusBar().showMessage(f"Loading model: {name}")
        self._loading_model_name = name      # remember for set_model_loaded callback

        self._total_graded = 0
        self._center.model_input_panel.reset()
        self._center.model_input_panel.set_mode(input_mode)   # update header label live

        # Pause the video so it does not advance while the GPU loads the model.
        # For a real JAI camera this is a no-op (CameraWorker has no pause).
        if isinstance(self._cam_w, VideoWorker):
            self._cam_w.pause()
            log.info("_on_load_model: VideoWorker paused while model loads")

        self._infer_w = RealInferenceWorker(
            model_path     = model_path,
            conf_threshold = self._left.get_confidence_threshold(),
            iou_threshold  = inf_cfg.get("iou_threshold", 0.45),
            device         = inf_cfg.get("device", "cuda"),
            input_mode     = input_mode,
            tracker        = self._tracker,
            size_acc       = self._size_acc,
            size_lock      = self._size_lock,
        )
        self._infer_w.sig_model_ready.connect(self._on_model_ready)
        self._infer_w.sig_preview.connect(self._on_track_preview)
        self._infer_w.sig_graded.connect(self._on_graded_batch)
        self._infer_w.sig_fps.connect(self._on_inference_fps)
        self._infer_w.sig_status.connect(self._on_inference_status)
        self._wire_infer_logging()
        self._infer_w.start()

    # ── Class colours (BGR): 0=Fresh-green  1=Processing-amber  2=Cull-red ──
    _OUTLET_MAP  = {"Fresh": "A", "Processing": "B", "Cull": "C"}

    @pyqtSlot()
    def _on_model_ready(self) -> None:
        """
        Called (in the main thread via Qt signal) once RealInferenceWorker has
        finished loading the model and is about to enter the inference loop.

        At this point the VideoWorker is still paused, so no new frames have
        entered the inference queue since the pause.  We drain any stale frames
        that were already in the queue before the pause, then resume the video
        so it continues from exactly where it stopped.
        """
        if self._infer_w is not None:
            drained = self._infer_w.drain_pending()
            if drained:
                log.debug("_on_model_ready: drained %d stale frame(s)", drained)

        # Resume video from the exact frame it was paused at.
        if isinstance(self._cam_w, VideoWorker):
            self._cam_w.resume()
            log.info("_on_model_ready: VideoWorker resumed")

    @pyqtSlot(object, list)
    def _on_track_preview(self, thumb: np.ndarray, active: list) -> None:
        """Coalesced UI refresh - logging already handled on inference thread."""
        self._last_input_frame = thumb
        self._display_active = active
        if not self._preview_ui_pending:
            self._preview_ui_pending = True
            QTimer.singleShot(0, self._flush_track_preview)

    def _flush_track_preview(self) -> None:
        self._preview_ui_pending = False
        active = getattr(self, "_display_active", None)
        if active is None:
            return
        self._schedule_display_update(active, self._infer_fps)

    @pyqtSlot(list)
    def _on_graded_batch(self, graded: list) -> None:
        """Grade commits - always processed immediately, never coalesced."""
        for rec in graded:
            if self._grading_recorder is not None:
                self._grading_recorder.on_grade_committed(
                    seq_id=rec.seq_id,
                    lane=rec.lane,
                    class_name=rec.class_name,
                    confidence=rec.confidence,
                    track_id=rec.track_id,
                )
            if self._sorter and self._sorting_enabled:
                _MIN_DISPATCH_CONF = 0.28
                is_cull = rec.class_name == "Cull"
                if is_cull or rec.confidence >= _MIN_DISPATCH_CONF:
                    self._sorter.schedule(
                        apple_id=rec.seq_id,
                        lane=rec.lane,
                        grade=rec.class_name,
                        confidence=rec.confidence,
                    )
                else:
                    log.warning(
                        "Grade #%d rejected -- conf=%.2f < %.2f minimum",
                        rec.seq_id, rec.confidence, _MIN_DISPATCH_CONF,
                    )
            self._graded_ui_queue.append(rec)

        if graded:
            self._schedule_graded_ui()

    def _schedule_graded_ui(self) -> None:
        if not self._graded_ui_pending:
            self._graded_ui_pending = True
            QTimer.singleShot(0, self._flush_graded_ui)

    def _flush_graded_ui(self) -> None:
        self._graded_ui_pending = False
        queue = self._graded_ui_queue
        self._graded_ui_queue = []
        for rec in queue:
            outlet = self._OUTLET_MAP.get(rec.class_name, "?")
            size_mm = None
            if self._size_acc is not None:
                with self._size_lock:
                    size_mm = self._size_acc.commit(rec.track_id, rec.lane)
            self._right.results_group.add_result(
                rec.seq_id, rec.lane, rec.class_name, rec.confidence, outlet, size_mm,
            )
            self._right.grade_summary.record(rec.class_name)
            self._right.metrics_group.record_grade()
            self._center.analytics_panel.record_grade(rec.class_name)
            self._total_graded += 1
            self.statusBar().showMessage(
                f"#{rec.seq_id}  Lane {rec.lane}  →  {rec.class_name}  "
                f"{rec.confidence * 100:.1f}%  ({rec.frames_seen} frames)"
            )

    def _schedule_display_update(self, active: list, fps: float) -> None:
        """Coalesce model-input repaints (one per event-loop tick)."""
        self._display_active = active
        self._display_fps = fps
        if not self._display_pending:
            self._display_pending = True
            QTimer.singleShot(0, self._render_deferred_display)

    def _render_deferred_display(self) -> None:
        self._display_pending = False
        active = getattr(self, "_display_active", None)
        if active is None or self._last_input_frame is None:
            return

        box_ref_w = self._last_ch1.shape[1] if self._last_ch1 is not None else None
        ann_input = annotate_tracked(
            self._last_input_frame, active, show_label=True, box_ref_w=box_ref_w,
        )
        self._center.model_input_panel.update_frame(
            ann_input, self._total_graded,
        )
        self._right.metrics_group.set_view_fps(
            self._center.model_input_panel._last_display_fps,
        )

    @pyqtSlot(float)
    def _on_inference_fps(self, fps: float) -> None:
        self._infer_fps = fps
        self._right.metrics_group.set_infer_fps(fps)

    @pyqtSlot(float)
    def _on_throughput_updated(self, apm: float) -> None:
        """Receives the true rolling APM from the analytics chart every second."""
        self._right.metrics_group.set_throughput(apm)

    @pyqtSlot(str, bool)
    def _on_inference_status(self, msg: str, is_error: bool) -> None:
        state = "offline" if is_error else "online"
        self._right.status_group.set_status("AI Model", state, msg)
        self.statusBar().showMessage(msg)
        if is_error:
            # Re-enable loader so user can try another model
            self._left.set_model_loading(False)
        elif "Model loaded" in msg:
            # Model is in GPU memory and running - update left panel
            name = getattr(self, "_loading_model_name", "")
            self._left.set_model_loading(False)
            self._left.set_model_loaded(name)

    @pyqtSlot(bool)
    def _on_sorter_toggle(self, enabled: bool) -> None:
        """Switch SorterController between simulation and live serial mode."""
        self._sorting_enabled = enabled
        if self._sorter:
            self._sorter.set_mode("serial" if enabled else "simulation")
            actual_mode = self._sorter._mode   # may differ from requested if connect failed
        else:
            actual_mode = "simulation"

        if not enabled:
            state, label = "idle", "Simulation"
        elif actual_mode == "serial":
            state, label = "online", "Active (Serial)"
        else:
            # requested serial but fell back - warn the user
            state, label = "warning", "Simulation (no Arduino)"

        self._right.status_group.set_status("Sorter", state, label)

    @pyqtSlot(bool)
    def _on_logging_toggle(self, enabled: bool) -> None:
        """Legacy slot - kept for compat; prefer _on_save_mode_changed."""
        self._on_save_mode_changed(enabled)

    @pyqtSlot(bool)
    def _on_save_mode_changed(self, enabled: bool) -> None:
        """Save mode toggled - controls whether the GradingRecorder is active."""
        self._save_mode    = enabled
        self._logging_enabled = enabled   # backward-compat alias
        if enabled:
            if self._cam_w is not None and self._grading_recorder is None:
                self._start_grading_session()
            elif self._grading_recorder is None:
                self._right.status_group.set_status("Logger", "idle", "Armed")
        else:
            self._stop_grading_session()
            self._right.status_group.set_status("Logger", "idle", "Off")
        self._wire_infer_logging()

    @pyqtSlot(bool)
    def _on_detect_mode_changed(self, enabled: bool) -> None:
        """Detect mode toggled - enables on-screen detections and inference."""
        self._detect_mode = enabled
        self._wire_infer_logging()

    @pyqtSlot(bool, bool)
    def _on_logging_options(self, raw: bool, detected: bool) -> None:
        """
        User changed Data Logging options in the popup.
        Updates recorder flags live - no session restart needed.
        Detected selection automatically enables inference.
        """
        self._log_raw       = raw
        self._log_detected  = detected
        rec = self._grading_recorder
        if rec is not None:
            rec.set_save_options(
                save_raw_full_frames = raw,
                save_detected_crops  = detected,
            )
        # Re-wire: selecting Detected should start inference even
        # if the Detect mode button is not explicitly toggled.
        self._wire_infer_logging()

    def _on_save_path_changed(self, path: str) -> None:
        """
        Operator selected a custom save path in the Data Logging popup.
        Stored for use when the next grading session starts.
        Updates the sidebar tooltip to reflect the pending path.
        """
        self._custom_save_path = path
        if path:
            self._left.set_logging_path(path)
        else:
            log_cfg  = self._cfg.get("logging", {})
            raw_out  = log_cfg.get("output_dir", "data/sessions")
            self._left.set_logging_path(raw_out)

    def _on_save_interval_changed(self, every_n: int) -> None:
        """
        Operator changed the raw-frame save interval in the Data Logging popup.
        The value is read from the panel spinbox at the next session start.
        """
        log.info("Save interval updated: every %d frame(s)", every_n)
        self.statusBar().showMessage(
            f"Save interval: every {every_n} frame(s) - takes effect on next session start"
        )

    def _on_save_resolution_changed(self, max_dim: int) -> None:
        """
        Operator changed the save resolution in the Data Logging popup.
        The value is read from the panel at the next session start.
        """
        if max_dim <= 0:
            log.info("Save resolution updated: default (full resolution)")
            self.statusBar().showMessage(
                "Save resolution: full sensor / native apple crops - next session"
            )
        else:
            log.info("Save resolution updated: max %d px", max_dim)
            self.statusBar().showMessage(
                f"Save resolution: downscale cap {max_dim} px longest side - next session"
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
            self.statusBar().showMessage("Exposure: camera not connected - connect first")

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
            self.statusBar().showMessage("Frame rate: camera not connected - connect first")

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
            self.statusBar().showMessage("Gain: camera not connected - connect first")

    @pyqtSlot(float, float, float)
    def _on_gains_readback(self, ch1: float, ch2: float, ch3: float) -> None:
        """
        Receives actual gains read back from firmware after an Apply.
        Updates all 3 gain spinboxes to show real hardware values.
        """
        self._left.update_gains(ch1, ch2, ch3)
        self.statusBar().showMessage(
            f"Gain confirmed -  "
            f"CH1: {ch1:.1f} dB  |  CH2: {ch2:.1f} dB  |  CH3: {ch3:.1f} dB"
        )
        log.info("Gain readback - CH1=%.1f CH2=%.1f CH3=%.1f dB", ch1, ch2, ch3)

    @pyqtSlot(int, int, int)
    def _on_exposure_readback(self, ch1: int, ch2: int, ch3: int) -> None:
        """
        Receives the actual ExposureTimes read back from firmware after an Apply or FPS change.
        Updates all 3 exposure spinboxes to reflect the real (possibly clamped) values.
        """
        self._left.update_exposures(ch1, ch2, ch3)
        self.statusBar().showMessage(
            f"Exposure confirmed -  "
            f"CH1: {ch1:,} µs  |  CH2: {ch2:,} µs  |  CH3: {ch3:,} µs"
        )
        log.info("Exposure readback - CH1=%d CH2=%d CH3=%d µs", ch1, ch2, ch3)

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
            self._lbl_sync.setStyleSheet(f"color: {TEXT_2}; font-family: monospace; font-size: 11px; margin-right: 10px;")
            return
        
        sync_str = "OK" if synced else "MISMATCH"
        color = SUCCESS if synced else DANGER
        self._lbl_sync.setText(f"Sync: {sync_str}  ·  IDs: {ch1_bid} / {ch2_bid} / {ch3_bid}")
        self._lbl_sync.setStyleSheet(f"color: {color}; font-family: monospace; font-size: 11px; font-weight: bold; margin-right: 10px;")

    # ── White Balance slots ────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_awb_triggered(self) -> None:
        """
        Forward One-Push AWB trigger to CameraWorker.
        The worker runs trigger_auto_white_balance() (blocking ~0-3 s in its own
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
                "White Balance: camera not connected - connect first"
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
                "White Balance: camera not connected - connect first"
            )

    @pyqtSlot(bool, float, float, float)
    def _on_wb_readback(
        self, success: bool, r: float, g: float, b: float
    ) -> None:
        """
        Called after AWB calibration completes or Revert finishes.
        Updates the WB ratio display and status bar.
        When the readback follows a Revert, the Revert button is disabled again
        because _saved_wb has been cleared in camera_interface - no snapshot remains.
        """
        was_revert = self._wb_reverting
        self._wb_reverting = False   # reset flag for next operation
        self._left.update_white_balance(success, r, g, b, revert_done=was_revert)
        if success:
            self.statusBar().showMessage(
                f"White Balance confirmed - R: {r:.4f}  G: {g:.4f}  B: {b:.4f}"
            )
            log.info("WB readback - R=%.4f G=%.4f B=%.4f", r, g, b)
        else:
            self.statusBar().showMessage(
                "White Balance: calibration failed - check connection and try again"
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
                "Black Level: camera not connected - connect first"
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
            f"Black Level confirmed - CH1: {ch1:.1f}  CH2: {ch2:.1f}  CH3: {ch3:.1f} DN"
        )
        log.info("Black Level readback - CH1=%.1f CH2=%.1f CH3=%.1f DN", ch1, ch2, ch3)

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
                f"- {pct}% of full frame…"
            )
        else:
            self.statusBar().showMessage(
                "ROI: camera not connected - connect first"
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
            msg = "ROI confirmed - Full Frame 2048×1536"
        else:
            msg = (f"ROI confirmed - {w}×{h} px @ ({x}, {y})  "
                   f"[{pct}% of frame - {(100-pct)}% data saved]")
        self.statusBar().showMessage(msg)
        log.info("ROI readback - x=%d y=%d w=%d h=%d (%d%% of frame)", x, y, w, h, pct)
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
        as the user moves the spinboxes - no Apply click required to see the
        effect on the image.
        """
        self._center.channel_display.set_roi_preview(ox, oy, w, h)
