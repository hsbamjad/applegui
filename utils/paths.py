"""
utils/paths.py
==============
Central path resolver for the Infield Apple Sorting System.

All application code should import paths from here instead of using
hardcoded relative strings like "models/best.pt" or "config/config.yaml".
This guarantees that paths resolve correctly regardless of the current
working directory when the app is launched.

Usage
-----
    from utils.paths import APP_ROOT, CONFIG_PATH, MODELS_DIR

    cfg_path   = CONFIG_PATH              # config/config.yaml
    model_path = MODELS_DIR / "best.pt"  # models/best.pt
"""

from pathlib import Path

# The project root is the directory that contains this file's parent (utils/).
# This works whether you launch via:
#   - python main.py            (from project root)
#   - launch.bat                (sets CWD to project root before calling python)
#   - double-click AppleSorter.exe  (PyInstaller bundle - uses sys._MEIPASS)
import sys

if getattr(sys, "frozen", False):
    # Running inside a PyInstaller bundle
    APP_ROOT: Path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    # Running from source - walk up from this file to the project root
    APP_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Canonical paths ───────────────────────────────────────────────────────────

CONFIG_PATH: Path = APP_ROOT / "config" / "config.yaml"
"""Absolute path to config/config.yaml"""

CONFIG_SWEETP_PATH: Path = APP_ROOT / "config" / "config_sweetp.yaml"
"""Absolute path to config/config_sweetp.yaml (sweet potato mode)"""

MODELS_DIR: Path = APP_ROOT / "models"
"""Absolute path to the models/ directory"""

DATA_DIR: Path = APP_ROOT / "data"
"""Absolute path to the data/ output directory"""

SESSIONS_DIR: Path = DATA_DIR / "sessions"
"""Absolute path to per-session grading export output"""

LOGS_DIR: Path = APP_ROOT / "logs"
"""Absolute path to the logs/ directory"""
