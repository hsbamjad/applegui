"""
Infield Apple Sorting System — Entry Point
==========================================
Michigan State University | ASABE AIM26 | 2026

Usage:
    conda activate applegui
    python main.py
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

try:
    from gui.main_window import MainWindow
except ImportError:
    logger.error("gui/main_window.py not found.")
    logger.error("This is built in Step A4. Run A4 first, then launch main.py.")
    sys.exit(1)


def main() -> None:
    """Launch the Infield Apple Sorting System."""
    app = QApplication(sys.argv)
    app.setApplicationName("Infield Apple Sorting System")
    app.setOrganizationName("MSU")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
