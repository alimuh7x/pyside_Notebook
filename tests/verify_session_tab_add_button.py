from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton, QTabBar

from pyside_app.main_window import MainWindow


app = QApplication.instance() or QApplication([])
window = MainWindow()
window.add_workspace_session()
QApplication.processEvents()

tab_bar = window.session_tabs.tabBar()
button = window.add_session_button
last_tab = tab_bar.tabRect(tab_bar.count() - 1)
last_close_container = tab_bar.tabButton(tab_bar.count() - 1, QTabBar.ButtonPosition.RightSide)
last_close_button = last_close_container.findChild(QPushButton, "sessionTabCloseButton")

print(
    "[verify][session-tabs] "
    f"tabs={tab_bar.count()} "
    f"last_tab_label={window.session_tabs.tabText(tab_bar.count() - 1)!r} "
    f"last_tab_has_close_icon={not last_close_button.icon().isNull()} "
    f"add_button_has_icon={not button.icon().isNull()} "
    f"last_tab_right={last_tab.right()} "
    f"button_pos=({button.x()}, {button.y()}) "
    f"button_size=({button.width()}x{button.height()}) "
    f"button_parent_is_tab_bar={button.parent() is tab_bar} "
    f"corner_widget_is_none={window.session_tabs.cornerWidget(Qt.Corner.TopRightCorner) is None}",
    flush=True,
)
