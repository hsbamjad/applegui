"""
gui/styles.py - Fresh Cull Horizon Theme
========================================
Light gradient dashboard for precision agriculture / fruit sorting.

Palette inspiration: soft light mist with green↔red grade accents.

  Background:  #E6EBE7  - Soft orchard mist (base; also painted as gradient)
  Surface:     #DCE4DE  - Pale leaf wash
  Card:        #FFFFFF  - Clean white (soft vertical glaze)
  Elevated:    #F0F4F1  - Lifted panel
  Hover:       #CDD8D1  - Soft sage hover

  Accent:      #2F9B55  - Fresh green
  Accent HV:   #3BB566  - Bright fresh
  Accent DK:   #247A42  - Deep fresh
  Accent END:  #C04A3C  - Cull red (gradient end-stop)
  On Accent:   #FFFFFF  - Button label

  Text 1:      #1A2B22  - Near-black green
  Text 2:      #5A6B60  - Sage secondary
  Text 3:      #8A9A8E  - Soft muted

  Border:      #D0D9D2  - Soft edge
  Border LT:   #B4C2B8  - Cool edge

  Success:     #2F9B55  - Fresh green
  Warning:     #C4841A  - Amber alert
  Danger:      #C04A3C  - Cull red
  Info:        #4A8BB0  - Soft steel blue
"""

# ── Fresh Cull Horizon Palette ────────────────────────────────────────────────
BG_BASE     = "#E6EBE7"   # soft orchard mist (darker)
BG_SURFACE  = "#DCE4DE"   # pale leaf wash (darker)
BG_CARD     = "#FFFFFF"   # clean white
BG_ELEVATED = "#F0F4F1"   # lifted panel
BG_HOVER    = "#CDD8D1"   # soft sage hover

ACCENT      = "#2F9B55"   # fresh green - primary action start
ACCENT_HV   = "#3BB566"   # bright fresh - hover start
ACCENT_DK   = "#247A42"   # deep fresh - pressed start
ACCENT_END  = "#C04A3C"   # cull red - gradient end-stop
ON_ACCENT   = "#FFFFFF"   # text on accent fills

SUCCESS     = "#2F9B55"   # fresh green
WARNING     = "#C4841A"   # amber alert
DANGER      = "#C04A3C"   # cull red
INFO        = "#4A8BB0"   # soft steel blue

# Explicit 4-color set for Camera Controls headings / channel labels
GREEN       = "#2F9B55"
RED         = "#C04A3C"
YELLOW      = "#D4A017"
ORANGE      = "#E07020"

# Channel label colors (CH1 color / CH2–CH3 NIR)
# CH1 = red (visible color); CH2/CH3 = dark grey (NIR, less emphasis)
CH_LABEL_COLORS = [RED, "#4A5560", "#4A5560"]

TEXT_1      = "#1A2B22"   # near-black green - primary text
TEXT_2      = "#5A6B60"   # sage - secondary / label
TEXT_3      = "#8A9A8E"   # soft muted - disabled / dim

BORDER      = "#D0D9D2"   # soft edge
BORDER_LT   = "#B4C2B8"   # cool edge

# Channel spectrum accents: amber, fresh green, cull red
CH_COLORS   = ["#C4841A", "#2F9B55", "#C04A3C"]

# Semantic status pill fills (light tinted surfaces)
PILL_ONLINE_BG  = "#E2F4E9"
PILL_OFFLINE_BG = "#F8E8E6"
PILL_WARNING_BG = "#F8F0E0"
BADGE_LIVE_BG   = "#E0F3E8"
BADGE_LIVE_BDR  = "#8FCB9E"
BADGE_SIM_BG    = "#F8F0E0"
BADGE_SIM_BDR   = "#E0C88A"
BADGE_MOCK_BG   = "#EEF1EE"
BADGE_MOCK_BDR  = BORDER

# Header bar gradient - deeper green mist → mid mist → deeper red mist
HEADER_GRAD_0 = "#BFDCC9"
HEADER_GRAD_1 = "#E6EBE7"
HEADER_GRAD_2 = "#E5CEC9"

# QSS gradient fills (stronger green ↔ red / soft glazes)
GRAD_BASE = (
    "qlineargradient(x1:0, y1:0, x2:1, y2:1, "
    "stop:0 #C5DFD0, stop:0.5 #E6EBE7, stop:1 #E6D2CD)"
)
GRAD_SURFACE = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, "
    "stop:0 #D5E3DA, stop:1 #C8D5CD)"
)
GRAD_CARD = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, "
    "stop:0 #FFFFFF, stop:1 #EAF1EC)"
)
GRAD_BTN = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, "
    "stop:0 #F5F8F6, stop:1 #D9E3DC)"
)
GRAD_ACCENT = (
    f"qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    f"stop:0 {ACCENT}, stop:1 {ACCENT_END})"
)
GRAD_ACCENT_HV = (
    "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    "stop:0 #3BB566, stop:1 #D46A5C)"
)
GRAD_ACCENT_DK = (
    "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    "stop:0 #247A42, stop:1 #A03C32)"
)
# Soft pastel green → coral (primary camera CTAs)
GRAD_ACCENT_LIGHT = (
    "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    "stop:0 #B5E0C2, stop:1 #F0C2BB)"
)
GRAD_ACCENT_LIGHT_HV = (
    "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    "stop:0 #A0D6B2, stop:1 #E8B0A8)"
)
GRAD_ACCENT_LIGHT_DK = (
    "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    "stop:0 #8BC9A0, stop:1 #DCA098)"
)
ON_ACCENT_LIGHT = "#1A2B22"   # dark label on light gradient fills
GRAD_DANGER = (
    "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    "stop:0 #C04A3C, stop:1 #D46A5C)"
)
GRAD_TAB = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, "
    "stop:0 #D5E3DA, stop:1 #C8D5CD)"
)

COLORS = {
    "bg_base": BG_BASE, "bg_surface": BG_SURFACE, "bg_card": BG_CARD,
    "bg_elevated": BG_ELEVATED, "bg_hover": BG_HOVER,
    "accent": ACCENT, "accent_hv": ACCENT_HV, "accent_dk": ACCENT_DK,
    "on_accent": ON_ACCENT,
    "success": SUCCESS, "warning": WARNING, "danger": DANGER, "info": INFO,
    "text_1": TEXT_1, "text_2": TEXT_2, "text_3": TEXT_3,
    "border": BORDER, "border_lt": BORDER_LT,
}

APP_STYLESHEET = f"""
/* ══════════════════════════════════════════════════════════════
   Apple Sorting GUI  ·  Fresh Cull Horizon Theme
   Michigan State University  ·  ASABE AIM26  ·  2026
   ══════════════════════════════════════════════════════════════ */

/* ── Base ─────────────────────────────────────────────────────── */
QMainWindow {{ background: {GRAD_BASE}; }}
QWidget {{
    color: {TEXT_1};
    font-family: 'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif;
    font-size: 12px;
}}
/* Transparent fill only where needed — never on QToolTip (Windows paints it black). */
QLabel, QCheckBox {{
    background-color: transparent;
}}

/* ── Group Boxes ──────────────────────────────────────────────── */
QGroupBox {{
    background: {GRAD_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 18px;
    padding: 14px 10px 10px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 1px 8px;
    background-color: {BG_CARD};
    color: {TEXT_3};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
}}

/* ── Buttons ──────────────────────────────────────────────────── */
QPushButton {{
    background: {GRAD_BTN};
    color: {TEXT_2};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 7px 16px;
    font-size: 12px;
    font-weight: 500;
    min-height: 32px;
    outline: none;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {BORDER_LT};
    color: {TEXT_1};
}}
        QPushButton:pressed {{
    background-color: {ACCENT_DK};
    border-color: {ACCENT_DK};
    color: white;
}}
QPushButton:disabled {{
    background-color: {BG_CARD};
    color: {TEXT_3};
    border-color: {BORDER};
}}

/* ── Primary CTA: solid fresh green ──────────────────────────── */
QPushButton#btn_primary {{
    background-color: {ACCENT};
    color: {ON_ACCENT};
    border: none;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.3px;
    min-height: 38px;
    border-radius: 8px;
}}
QPushButton#btn_primary:hover {{
    background-color: {ACCENT_HV};
    color: {ON_ACCENT};
}}
QPushButton#btn_primary:pressed {{
    background-color: {ACCENT_DK};
    color: {ON_ACCENT};
}}
QPushButton#btn_primary:disabled {{
    background-color: {BG_ELEVATED};
    color: {TEXT_3};
}}

/* ── Danger: cull red ─────────────────────────────────────────── */
QPushButton#btn_danger {{
    background-color: {DANGER};
    color: white;
    border: none;
    font-weight: 700;
    font-size: 12px;
    min-height: 38px;
    border-radius: 8px;
}}
QPushButton#btn_danger:hover {{
    background-color: #D46A5C;
    color: white;
}}
QPushButton#btn_danger:pressed {{
    background-color: #A03C32;
    color: white;
}}

/* ── ComboBox ─────────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_ELEVATED};
    color: {TEXT_1};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 0 12px;
    min-height: 34px;
    font-size: 12px;
    font-weight: 400;
    selection-background-color: {ACCENT_DK};
}}
QComboBox:hover {{
    border-color: {BORDER_LT};
    color: {TEXT_1};
}}
QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: right center;
    width: 30px;
    border: none;
    border-left: 1px solid {BORDER};
    background-color: {BG_HOVER};
    border-radius: 0 7px 7px 0;
}}
QComboBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_2};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER_LT};
    border-radius: 8px;
    color: {TEXT_1};
    selection-background-color: {ACCENT};
    selection-color: {ON_ACCENT};
    padding: 4px;
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    min-height: 30px;
    padding: 4px 14px;
    border-radius: 4px;
    color: {TEXT_1};
}}
QComboBox QAbstractItemView::item:selected {{
    background-color: {ACCENT};
    color: {ON_ACCENT};
}}

/* ── SpinBox ──────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {BG_ELEVATED};
    color: {TEXT_1};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 0 8px;
    min-height: 34px;
    font-size: 12px;
    selection-background-color: {ACCENT};
}}
QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {BORDER_LT}; }}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {BG_HOVER};
    border: none;
    border-left: 1px solid {BORDER};
    width: 22px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {ACCENT};
}}
QSpinBox::up-button   {{ border-radius: 0 7px 0 0; }}
QSpinBox::down-button {{ border-radius: 0 0 7px 0; }}
QDoubleSpinBox::up-button   {{ border-radius: 0 7px 0 0; }}
QDoubleSpinBox::down-button {{ border-radius: 0 0 7px 0; }}

/* ── CheckBox ─────────────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT_1};
    font-size: 12px;
    spacing: 10px;
}}
QCheckBox::indicator {{
    width: 17px; height: 17px;
    border: 1.5px solid {BORDER_LT};
    border-radius: 4px;
    background-color: {BG_ELEVATED};
}}
QCheckBox::indicator:hover  {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* ── List Widget ──────────────────────────────────────────────── */
QListWidget {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 7px;
    color: {TEXT_1};
    font-size: 11px;
    outline: none;
}}
QListWidget::item {{
    padding: 5px 10px;
    border-bottom: 1px solid {BORDER};
    background-color: transparent;
    color: {TEXT_1};
}}
QListWidget::item:hover {{
    background-color: {BG_HOVER};
}}
QListWidget::item:selected {{
    background-color: {ACCENT}22;
    color: {TEXT_1};
    border-left: 2px solid {ACCENT};
}}

/* ── ScrollBar ────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 6px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_LT};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}

/* ── Splitter ─────────────────────────────────────────────────── */
QSplitter::handle:horizontal {{
    background-color: {BORDER};
    width: 1px;
}}
QSplitter::handle:vertical {{
    background-color: {BORDER};
    height: 1px;
}}
QSplitter::handle:hover {{ background-color: {ACCENT}; }}

/* ── Status Bar ───────────────────────────────────────────────── */
QStatusBar {{
    background: {GRAD_SURFACE};
    color: {TEXT_2};
    border-top: 1px solid {BORDER};
    font-size: 11px;
    padding: 0 10px;
}}
QStatusBar::item {{ border: none; }}

/* ── ToolTip ──────────────────────────────────────────────────── */
/* Styled via TOOLTIP_STYLESHEET on QApplication - do not rely on MainWindow QSS. */
QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT_1};
    border: 1px solid {BORDER_LT};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 400;
}}
"""

# Applied on QApplication only. Keep this free of `QWidget { background: transparent }`
# — that rule makes Windows tooltips render black (QToolTip is a QWidget).
TOOLTIP_STYLESHEET = f"""
QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT_1};
    border: 1px solid {BORDER_LT};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 400;
}}
"""

# Keep trailing content that was after QToolTip in APP_STYLESHEET
_APP_STYLESHEET_TAIL = f"""
/* ── ScrollArea ───────────────────────────────────────────────── */
QScrollArea {{ border: none; }}
QScrollArea > QWidget > QWidget {{ background-color: transparent; }}

/* ── Label base ───────────────────────────────────────────────── */
QLabel {{ background: transparent; color: {TEXT_1}; }}

/* ── HLine frame ──────────────────────────────────────────────── */
QFrame[frameShape="4"] {{
    background-color: {BORDER};
    color: {BORDER};
    max-height: 1px;
    border: none;
}}

/* ── Tab Widget ───────────────────────────────────────────────── */
QTabWidget::pane {{
    border: none;
    background: {GRAD_BASE};
}}
QTabWidget::tab-bar {{
    alignment: left;
}}
QTabBar {{
    background: transparent;
    border-bottom: 1px solid {BORDER};
}}
QTabBar::tab {{
    background: {GRAD_TAB};
    color: {TEXT_3};
    border: none;
    border-left: 1px solid {BORDER};
    border-bottom: 2px solid transparent;
    padding: 6px 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    min-width: 120px;
}}
QTabBar::tab:first {{
    border-left: none;
}}
QTabBar::tab:selected {{
    background: {GRAD_CARD};
    color: {TEXT_1};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    background-color: {BG_CARD};
    color: {TEXT_2};
}}
"""

APP_STYLESHEET = APP_STYLESHEET + _APP_STYLESHEET_TAIL
