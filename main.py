from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pyside_app.main_window import MainWindow


GLOBAL_STYLESHEET = """
QWidget {
    color: #0f1b2b;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 15px;
}
QMainWindow, QTabWidget::pane, QTabWidget {
    background: #f0f4f8;
}
QScrollArea {
    background: transparent;
    border: none;
}
QScrollBar:vertical {
    width: 8px;
    background: #e2e8f0;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #94a3b8;
    border-radius: 4px;
    min-height: 30px;
}
QTabWidget::pane {
    border: none;
    background: #f0f4f8;
}
QTabBar::tab {
    background: #e2e8f0;
    color: #0f1b2b;
    padding: 10px 22px;
    min-width: 110px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background: #001f41;
    color: white;
}
QPushButton {
    background: #001f41;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #b60021;
    border-color: #b60021;
}
QLabel {
    color: #0f1b2b;
}
QTextBrowser, QListWidget, QPlainTextEdit {
    font-size: 15px;
}
"""


def configure_desktop_graphics() -> dict[str, str]:
    print("[debug][main] graphics:configure:start", flush=True)
    defaults = {
        "QTWEBENGINE_DISABLE_SANDBOX": "1",
        "QT_OPENGL": "software",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "QT_QUICK_BACKEND": "software",
        "QTWEBENGINE_CHROMIUM_FLAGS": "--disable-gpu --disable-gpu-compositing --disable-features=Vulkan",
    }
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
        print(f"[debug][main] graphics:env {key}={os.environ.get(key)!r}", flush=True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    print("[debug][main] graphics:qt_attr AA_UseSoftwareOpenGL=True", flush=True)
    print("[debug][main] graphics:qt_attr AA_ShareOpenGLContexts=True", flush=True)
    return {key: os.environ[key] for key in defaults}


def main() -> int:
    print("[debug][main] start", flush=True)
    configure_desktop_graphics()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    window = MainWindow()
    window.show()
    print("[debug][main] exec", flush=True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
