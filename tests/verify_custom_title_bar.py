from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from pyside_app.main_window import MainWindow


def main() -> None:
    print("[verify][custom-title-bar] start", flush=True)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    frameless_enabled = bool(window.windowFlags() & Qt.WindowType.FramelessWindowHint)
    print(f"[verify][custom-title-bar] frameless_enabled={frameless_enabled}", flush=True)
    print(f"[verify][custom-title-bar] title={window.title_bar.title_label.text()!r}", flush=True)
    print(f"[verify][custom-title-bar] subtitle={window.title_bar.subtitle_label.text()!r}", flush=True)
    print(
        "[verify][custom-title-bar] buttons="
        f"{[button.objectName() for button in (window.title_bar.minimize_button, window.title_bar.maximize_button, window.title_bar.close_button)]}",
        flush=True,
    )

    if not frameless_enabled:
        raise SystemExit("[verify][custom-title-bar] FAIL: FramelessWindowHint not enabled")
    if window.title_bar.title_label.text() != "calculationNotebook":
        raise SystemExit("[verify][custom-title-bar] FAIL: title label mismatch")
    if window.title_bar.subtitle_label.text() != "Desktop":
        raise SystemExit("[verify][custom-title-bar] FAIL: subtitle label mismatch")
    if any(button.icon().isNull() for button in (window.title_bar.minimize_button, window.title_bar.maximize_button, window.title_bar.close_button)):
        raise SystemExit("[verify][custom-title-bar] FAIL: control button icon missing")

    print("[verify][custom-title-bar] PASS", flush=True)
    app.quit()


if __name__ == "__main__":
    main()
