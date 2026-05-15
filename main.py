"""
Apple Sorting GUI — Entry Point
================================
Michigan State University | ASABE AIM26 | 2026

Usage:
    conda activate applegui
    python main.py
"""

import sys
from PyQt6.QtWidgets import QApplication

try:
    from gui.main_window import MainWindow
except ImportError:
    print("[ERROR] gui/main_window.py not found.")
    print("        This is built in Step A4. Run A4 first, then launch main.py.")
    sys.exit(1)


def main() -> None:
    """Launch the Apple Sorting GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Apple Sorting GUI")
    app.setOrganizationName("MSU")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
