from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pyside_app.main_window import MainWindow


GLOBAL_STYLESHEET = """
QWidget {
    color: #0f1b2b;
    font-family: 'Inter', sans-serif;
    font-size: 17px;
    font-weight: 700;
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
QMenu {
    background: #2c313a;
    color: #ffffff;
    border: 1px solid #3e4451;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item {
    color: #ffffff;
    background: transparent;
    padding: 6px 26px;
}
QMenu::item:selected {
    color: #0b1220;
    background: #61afef;
}
QMenu::item:disabled {
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background: #3e4451;
    margin: 4px 8px;
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
    border-bottom: 3px solid #61afef;
}
QAbstractItemView {
    selection-background-color: #61afef;
    selection-color: #0b1220;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 2px solid #61afef;
    border-radius: 4px;
    background: #21252b;
}
QCheckBox::indicator:checked {
    background: #61afef;
    border: 2px solid #f8fafc;
}
QCheckBox::indicator:hover {
    border-color: #9cdcfe;
}
QRadioButton::indicator {
    width: 15px;
    height: 15px;
    border: 2px solid #94a3b8;
    border-radius: 8px;
    background: #21252b;
}
QRadioButton::indicator:checked {
    background: #61afef;
    border: 3px solid #f8fafc;
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
QTextBrowser, QListWidget {
    font-size: 17px;
}
"""


def application_icon_path() -> Path:
    path = Path(__file__).resolve().parent / "assets" / "calculator.png"
    print(f"[debug][main] app_icon:path path={str(path)!r}", flush=True)
    print(f"[debug][main] app_icon:exists exists={path.exists()}", flush=True)
    return path


def build_application_icon() -> QIcon:
    path = application_icon_path()
    icon = QIcon(str(path))
    print(f"[debug][main] app_icon:loaded is_null={icon.isNull()}", flush=True)
    return icon


def apply_application_icon(app: QApplication, window: MainWindow) -> QIcon:
    print("[debug][main] app_icon:apply:start", flush=True)
    icon = build_application_icon()
    app.setWindowIcon(icon)
    print(f"[debug][main] app_icon:apply:app is_null={app.windowIcon().isNull()}", flush=True)
    window.setWindowIcon(icon)
    print(f"[debug][main] app_icon:apply:window is_null={window.windowIcon().isNull()}", flush=True)
    if hasattr(window, "title_bar"):
        window.title_bar.icon_label.setPixmap(icon.pixmap(18, 18))
        pixmap = window.title_bar.icon_label.pixmap()
        print(
            f"[debug][main] app_icon:apply:title_bar pixmap_is_null={pixmap is None or pixmap.isNull()}",
            flush=True,
        )
    print("[debug][main] app_icon:apply:done", flush=True)
    return icon


def configure_desktop_graphics() -> dict[str, str]:
    print("[debug][main] graphics:configure:start", flush=True)
    defaults = {
        "QTWEBENGINE_DISABLE_SANDBOX": "1",
        "QT_OPENGL": "software",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "QT_QUICK_BACKEND": "software",
        "QTWEBENGINE_CHROMIUM_FLAGS": "--enable-webgl --ignore-gpu-blocklist --enable-gpu-rasterization --enable-unsafe-swiftshader --disable-features=Vulkan",
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
    print("[debug][main] stylesheet:applied qmenu_text='#ffffff' qmenu_background='#2c313a'", flush=True)
    window = MainWindow()
    apply_application_icon(app, window)
    window.show()
    print("[debug][main] exec", flush=True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
