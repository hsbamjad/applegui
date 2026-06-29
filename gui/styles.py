"""
gui/styles.py — Verdant Grove Theme
=====================================
"Forest & Cream" professional dashboard aesthetic.
Designed for a precision agriculture / fruit sorting context.

Palette inspiration: old-growth forest, worn parchment, copper tools.

  Background:  #0E1A10  — Deep Forest Night
  Surface:     #162019  — Shadowed Grove
  Card:        #1E2E21  — Undergrowth
  Elevated:    #263629  — Moss Bank
  Hover:       #2E4032  — Fern Shadow

  Accent:      #5DA669  — Leaf Green
  Accent HV:   #77C285  — Young Shoot
  Accent DK:   #3F8750  — Deep Canopy

  Text 1:      #EEE8D9  — Warm Cream (primary)
  Text 2:      #A8B89A  — Sage (secondary / label)
  Text 3:      #536550  — Dim Moss (disabled)

  Border:      #2C3E2F  — Dark Bark
  Border LT:   #3D5440  — Lighter Bark

  Success:     #72C472  — Ripe Green
  Warning:     #D4A843  — Harvest Amber
  Danger:      #C85C4A  — Autumn Red
  Info:        #62B6C0  — Morning Mist
"""

# ── Verdant Grove Palette ─────────────────────────────────────────────────────
BG_BASE     = "#0E1A10"   # deep forest night
BG_SURFACE  = "#162019"   # shadowed grove
BG_CARD     = "#1E2E21"   # undergrowth
BG_ELEVATED = "#263629"   # moss bank
BG_HOVER    = "#2E4032"   # fern shadow

ACCENT      = "#5DA669"   # leaf green — primary action
ACCENT_HV   = "#77C285"   # young shoot — hover
ACCENT_DK   = "#3F8750"   # deep canopy — pressed

SUCCESS     = "#72C472"   # ripe green
WARNING     = "#D4A843"   # harvest amber
DANGER      = "#C85C4A"   # autumn red
INFO        = "#62B6C0"   # morning mist

TEXT_1      = "#EEE8D9"   # warm cream — primary text
TEXT_2      = "#A8B89A"   # sage — secondary / label
TEXT_3      = "#536550"   # dim moss — disabled / dim

BORDER      = "#2C3E2F"   # dark bark
BORDER_LT   = "#3D5440"   # lighter bark

# Channel spectrum accent colors — forest family: warm gold, sky mist, copper
CH_COLORS   = ["#D4A843", "#62B6C0", "#C87B4A"]   # amber, mist-blue, copper

COLORS = {
    "bg_base": BG_BASE, "bg_surface": BG_SURFACE, "bg_card": BG_CARD,
    "bg_elevated": BG_ELEVATED, "bg_hover": BG_HOVER,
    "accent": ACCENT, "accent_hv": ACCENT_HV, "accent_dk": ACCENT_DK,
    "success": SUCCESS, "warning": WARNING, "danger": DANGER, "info": INFO,
    "text_1": TEXT_1, "text_2": TEXT_2, "text_3": TEXT_3,
    "border": BORDER, "border_lt": BORDER_LT,
}

APP_STYLESHEET = f"""
/* ══════════════════════════════════════════════════════════════
   Apple Sorting GUI  ·  Verdant Grove Theme
   Michigan State University  ·  ASABE AIM26  ·  2026
   ══════════════════════════════════════════════════════════════ */

/* ── Base ─────────────────────────────────────────────────────── */
QMainWindow {{ background-color: {BG_BASE}; }}
QWidget {{
    background-color: transparent;
    color: {TEXT_1};
    font-family: 'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif;
    font-size: 12px;
}}

/* ── Group Boxes ──────────────────────────────────────────────── */
QGroupBox {{
    background-color: {BG_CARD};
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
    background-color: {BG_ELEVATED};
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

/* ── Primary CTA: solid leaf-green, dark forest text ─────────── */
QPushButton#btn_primary {{
    background-color: {ACCENT};
    color: #0E1A10;
    border: none;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.3px;
    min-height: 38px;
    border-radius: 8px;
}}
QPushButton#btn_primary:hover {{
    background-color: {ACCENT_HV};
    color: #0E1A10;
}}
QPushButton#btn_primary:pressed {{
    background-color: {ACCENT_DK};
    color: {TEXT_1};
}}
QPushButton#btn_primary:disabled {{
    background-color: {BG_ELEVATED};
    color: {TEXT_3};
}}

/* ── Danger: autumn red ───────────────────────────────────────── */
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
    background-color: #D97060;
    color: white;
}}
QPushButton#btn_danger:pressed {{
    background-color: #A0473A;
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
    selection-color: #0E1A10;
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
    color: #0E1A10;
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
    background-color: {BG_SURFACE};
    color: {TEXT_2};
    border-top: 1px solid {BORDER};
    font-size: 11px;
    padding: 0 10px;
}}
QStatusBar::item {{ border: none; }}

/* ── ToolTip ──────────────────────────────────────────────────── */
QToolTip {{
    background-color: {BG_ELEVATED};
    color: {TEXT_1};
    border: 1px solid {ACCENT}80;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 400;
}}

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
    background-color: {BG_BASE};
}}
QTabWidget::tab-bar {{
    alignment: left;
}}
QTabBar {{
    background: transparent;
}}
QTabBar::tab {{
    background-color: {BG_SURFACE};
    color: {TEXT_3};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 6px 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    min-width: 120px;
}}
QTabBar::tab:selected {{
    background-color: {BG_BASE};
    color: {TEXT_1};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    background-color: {BG_CARD};
    color: {TEXT_2};
}}
"""
