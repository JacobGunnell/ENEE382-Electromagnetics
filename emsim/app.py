"""Application entry point."""

from __future__ import annotations

import sys

from PyQt6 import QtGui, QtWidgets

DARK_QSS = """
QWidget       { background: #14161c; color: #dfe4ee; }
QGroupBox     { border: 1px solid #2b303c; border-radius: 6px;
                margin-top: 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px;
                   color: #9fb0d0; }
QPushButton   { background: #232838; border: 1px solid #333a4d;
                border-radius: 4px; padding: 4px 10px; }
QPushButton:hover   { background: #2c3244; }
QPushButton:pressed { background: #1c2130; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background: #1b1f29; border: 1px solid #333a4d;
                border-radius: 4px; padding: 2px 6px; }
QTableWidget  { background: #171a22; gridline-color: #262b36;
                border: 1px solid #2b303c; }
QHeaderView::section { background: #1e222c; border: 0;
                       border-bottom: 1px solid #2b303c; padding: 3px; }
QSlider::groove:horizontal { height: 4px; background: #2b303c; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px;
                             background: #6f8ad0; }
QScrollArea   { border: 0; }
QStatusBar    { color: #8a93a6; }
"""


def main() -> int:
    fmt = QtGui.QSurfaceFormat()
    fmt.setSamples(4)
    fmt.setDepthBufferSize(24)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("emsim")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)

    from .ui import MainWindow
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
