from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QStyle, QToolButton, QWidget


class TitleBar(QFrame):
    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print(f"[debug][title-bar] init:start title={title!r} subtitle={subtitle!r}", flush=True)
        self._drag_offset: QPoint | None = None
        self.setObjectName("customTitleBar")
        self.setFixedHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "QFrame#customTitleBar {"
            " background: #1e1e2e;"
            " border-top-left-radius: 8px;"
            " border-top-right-radius: 8px;"
            " border-bottom: 1px solid rgba(255, 255, 255, 0.08);"
            "}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 10, 0)
        layout.setSpacing(10)

        self.icon_label = QLabel(self)
        self.icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_DesktopIcon).pixmap(18, 18))
        self.icon_label.setFixedSize(22, 22)
        self.icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.icon_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.title_label = QLabel(title, self)
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.title_label.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: 700; background: transparent;")
        layout.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.subtitle_label = QLabel(subtitle, self)
        self.subtitle_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.subtitle_label.setStyleSheet(
            "color: #cbd5e1;"
            " font-size: 11px;"
            " font-weight: 600;"
            " padding: 3px 8px;"
            " border-radius: 8px;"
            " background: rgba(148, 163, 184, 0.16);"
        )
        self.subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self.subtitle_label, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)

        self.minimize_button = self._create_control_button(
            "minimizeButton",
            QStyle.StandardPixmap.SP_TitleBarMinButton,
            self._minimize_window,
        )
        self.maximize_button = self._create_control_button(
            "maximizeButton",
            QStyle.StandardPixmap.SP_TitleBarMaxButton,
            self._toggle_maximize_restore,
        )
        self.close_button = self._create_control_button(
            "closeButton",
            QStyle.StandardPixmap.SP_TitleBarCloseButton,
            self._close_window,
        )
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(self.close_button)
        self._update_maximize_icon()

        window = self.window()
        if window is not None:
            window.installEventFilter(self)
        print("[debug][title-bar] init:done", flush=True)

    def _create_control_button(
        self,
        object_name: str,
        standard_icon: QStyle.StandardPixmap,
        handler: object,
    ) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(object_name)
        button.setIcon(self.style().standardIcon(standard_icon))
        button.setAutoRaise(True)
        button.setCursor(Qt.CursorShape.ArrowCursor)
        button.setFixedSize(30, 30)
        button.clicked.connect(handler)
        hover_background = "rgba(148, 163, 184, 0.18)"
        if object_name == "closeButton":
            hover_background = "#b91c1c"
        button.setStyleSheet(
            f"QToolButton#{object_name} {{"
            " background: transparent;"
            " border: none;"
            " border-radius: 8px;"
            " padding: 0px;"
            "}}"
            f"QToolButton#{object_name}:hover {{ background: {hover_background}; }}"
            f"QToolButton#{object_name}:pressed {{ background: {hover_background}; }}"
        )
        return button

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched is self.window() and event.type() == QEvent.Type.WindowStateChange:
            print("[debug][title-bar] event_filter window_state_change", flush=True)
            self._update_maximize_icon()
        return super().eventFilter(watched, event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            print("[debug][title-bar] mouse_double_click", flush=True)
            self._toggle_maximize_restore()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            if window is not None and not window.isMaximized():
                if self._begin_system_move():
                    self._drag_offset = None
                    print("[debug][title-bar] mouse_press using_native_system_move", flush=True)
                else:
                    self._drag_offset = event.globalPosition().toPoint() - window.frameGeometry().topLeft()
                    print(f"[debug][title-bar] mouse_press drag_offset={self._drag_offset}", flush=True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        window = self.window()
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self._drag_offset is not None
            and window is not None
            and not window.isMaximized()
        ):
            target_position = event.globalPosition().toPoint() - self._drag_offset
            print(f"[debug][title-bar] mouse_move target_position={target_position}", flush=True)
            window.move(target_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def _minimize_window(self) -> None:
        print("[debug][title-bar] minimize_window", flush=True)
        window = self.window()
        if window is not None:
            window.showMinimized()

    def _toggle_maximize_restore(self) -> None:
        window = self.window()
        if window is None:
            return
        if window.isMaximized():
            print("[debug][title-bar] restore_window", flush=True)
            window.showNormal()
        else:
            print("[debug][title-bar] maximize_window", flush=True)
            window.showMaximized()
        self._update_maximize_icon()

    def _close_window(self) -> None:
        print("[debug][title-bar] close_window", flush=True)
        window = self.window()
        if window is not None:
            window.close()

    def _window_handle(self) -> object | None:
        window = self.window()
        if window is None:
            return None
        return window.windowHandle()

    def _begin_system_move(self) -> bool:
        handle = self._window_handle()
        if handle is None or not hasattr(handle, "startSystemMove"):
            print("[debug][title-bar] begin_system_move unavailable", flush=True)
            return False
        started = bool(handle.startSystemMove())
        print(f"[debug][title-bar] begin_system_move started={started}", flush=True)
        return started

    def _update_maximize_icon(self) -> None:
        window = self.window()
        if window is None:
            return
        icon = QStyle.StandardPixmap.SP_TitleBarNormalButton if window.isMaximized() else QStyle.StandardPixmap.SP_TitleBarMaxButton
        self.maximize_button.setIcon(self.style().standardIcon(icon))
        print(f"[debug][title-bar] update_maximize_icon maximized={window.isMaximized()}", flush=True)
