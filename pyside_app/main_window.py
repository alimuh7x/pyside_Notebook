from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyside_app.fft_tab import FFTTab
from pyside_app.formula_plot_tab import FormulaPlotTab
from pyside_app.graph_state import NotebookGraphState
from pyside_app.graphs_tab import GraphsTab
from pyside_app.notebook_tab import NotebookTab
from pyside_app.title_bar import TitleBar


_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_PLUS_ICON = _ASSETS_DIR / "plus.png"
_REMOVE_ICON = _ASSETS_DIR / "remove.png"


class WorkspaceSession(QWidget):
    """One independent notebook workspace with its own graph and analysis state."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Build a clean Notebook / Graphs / Formula Plot / FFT session."""
        super().__init__(parent)
        print(f"[debug][workspace-session] init:start title={title!r}", flush=True)
        self.title = title
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        self.graph_state = NotebookGraphState(self)
        print(f"[debug][workspace-session] graph_state:new title={title!r}", flush=True)
        self.workspace_tabs = QTabWidget(self)
        self.workspace_tabs.setStyleSheet(
            "QTabWidget::pane { background:#21252b; border:none; }"
            "QTabBar::tab { background:#21252b; color:#5c6370; padding:7px 18px;"
            " border:none; border-bottom:2px solid transparent; font-size:14px; font-weight:700; }"
            "QTabBar::tab:selected { background:#2c313a; color:#ffffff; border-bottom:3px solid #61afef; }"
            "QTabBar::tab:hover { color:#d7dae0; background:#282c34; }"
        )
        print(f"[debug][workspace-session] workspace_tabs:new title={title!r}", flush=True)
        self.notebook_tab = NotebookTab(self.workspace_tabs, graph_state=self.graph_state)
        print(f"[debug][workspace-session] notebook_tab:new title={title!r}", flush=True)
        self.graphs_tab = GraphsTab(self.graph_state, self.workspace_tabs)
        print(f"[debug][workspace-session] graphs_tab:new title={title!r}", flush=True)
        self.formula_tab = FormulaPlotTab(
            self.workspace_tabs,
            notebook_namespace_provider=lambda: self.notebook_tab.execution_engine.get_namespace(),
        )
        print(f"[debug][workspace-session] formula_tab:new title={title!r}", flush=True)
        self.fft_tab = FFTTab(graph_state=self.graph_state, parent=self.workspace_tabs)
        print(f"[debug][workspace-session] fft_tab:new title={title!r}", flush=True)
        self.workspace_tabs.addTab(self.notebook_tab, "Notebook")
        self.workspace_tabs.addTab(self.graphs_tab, "Graphs")
        self.workspace_tabs.addTab(self.formula_tab, "Formula Plot")
        self.workspace_tabs.addTab(self.fft_tab, "FFT Analysis")
        self.root_layout.addWidget(self.workspace_tabs)
        print(f"[debug][workspace-session] init:done title={title!r} inner_tabs={self.workspace_tabs.count()}", flush=True)


class SessionTabBar(QTabBar):
    """Tab bar with a trailing add button kept next to the open tabs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.add_button = QToolButton(self)
        self.add_button.setToolTip("New clean notebook tab")
        self.add_button.setFixedSize(26, 26)
        self.add_button.setIcon(QIcon(str(_PLUS_ICON)))
        self.add_button.setIconSize(QSize(18, 18))
        self.add_button.setStyleSheet(
            "QToolButton { background:transparent; border:none; border-radius:4px; padding:4px; }"
            "QToolButton:hover { background:#353b45; }"
        )
        self._position_add_button()

    def tabLayoutChange(self) -> None:
        """Reposition the add button whenever tabs are added, removed, or moved."""
        super().tabLayoutChange()
        self._position_add_button()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        """Keep the add button aligned after resizes."""
        super().resizeEvent(event)
        self._position_add_button()

    def _position_add_button(self) -> None:
        """Place the add button directly after the last visible tab."""
        if self.count() == 0:
            x = 4
            y = 4
        else:
            last_tab = self.tabRect(self.count() - 1)
            x = last_tab.right() + 4
            y = last_tab.top() + max(0, (last_tab.height() - self.add_button.height()) // 2)
        self.add_button.move(x, y)
        self.setMinimumWidth(x + self.add_button.width() + 6)


class MainWindow(QMainWindow):
    """Compose the desktop shell and wire the notebook and graphs tabs together."""

    def __init__(self) -> None:
        """Build the frameless main window and shared application state."""
        super().__init__()
        print("[debug][main-window] init:start", flush=True)
        self.setWindowTitle("Calculation Notebook Desktop")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.resize(1440, 960)
        self.window_shell = QWidget(self)
        shell_layout = QVBoxLayout(self.window_shell)
        shell_layout.setContentsMargins(12, 12, 12, 12)
        shell_layout.setSpacing(0)

        self.window_surface = QFrame(self.window_shell)
        self.window_surface.setObjectName("windowSurface")
        self.window_surface.setStyleSheet(
            "QFrame#windowSurface {"
            " background: #21252b;"
            " border: 1px solid rgba(0,0,0,0.6);"
            " border-radius: 8px;"
            "}"
        )
        shadow = QGraphicsDropShadowEffect(self.window_surface)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(Qt.GlobalColor.black)
        self.window_surface.setGraphicsEffect(shadow)

        surface_layout = QVBoxLayout(self.window_surface)
        surface_layout.setContentsMargins(0, 0, 0, 0)
        surface_layout.setSpacing(0)

        self.title_bar = TitleBar("calculationNotebook", "Desktop", self.window_surface)
        surface_layout.addWidget(self.title_bar)

        self.content_widget = QWidget(self.window_surface)
        self.content_widget.setObjectName("windowContent")
        self.content_widget.setStyleSheet(
            "QWidget#windowContent {"
            " background: #21252b;"
            " border-bottom-left-radius: 8px;"
            " border-bottom-right-radius: 8px;"
            "}"
        )
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        self._session_counter = 0
        print("[debug][main-window] session_tabs:init", flush=True)
        self.session_tabs = QTabWidget(self.content_widget)
        self.session_tab_bar = SessionTabBar(self.session_tabs)
        self.session_tabs.setTabBar(self.session_tab_bar)
        self.session_tabs.setTabsClosable(False)
        self.session_tabs.setMovable(True)
        self.session_tabs.setDocumentMode(True)
        self.session_tabs.setStyleSheet(
            "QTabWidget::pane { background:#21252b; border:none; }"
            "QTabBar::tab { background:#2c313a; color:#f1f5f9; padding:8px 14px 9px 12px;"
            " min-width:112px; border:none; border-right:1px solid #21252b;"
            " border-bottom:3px solid transparent; font-size:14px; font-weight:700; }"
            "QTabBar::tab:selected { background:#3e4451; color:#ffffff; border-bottom:3px solid #61afef; }"
            "QTabBar::tab:hover { background:#353b45; color:#ffffff; }"
            "QWidget#sessionTabCloseContainer { background:transparent; border:none; }"
            "QPushButton#sessionTabCloseButton { background:transparent; border:none; padding:1px; }"
            "QPushButton#sessionTabCloseButton:hover { background:#4b5563; border-radius:4px; }"
        )
        self.session_tabs.currentChanged.connect(self._sync_current_session_aliases)
        self.session_tabs.tabCloseRequested.connect(self.close_workspace_session)

        self.add_session_button = self.session_tab_bar.add_button
        self.add_session_button.clicked.connect(lambda _checked=False: self.add_workspace_session())
        self.content_layout.addWidget(self.session_tabs)
        self.add_workspace_session()
        surface_layout.addWidget(self.content_widget, 1)

        shell_layout.addWidget(self.window_surface)
        self.setCentralWidget(self.window_shell)
        print("[debug][main-window] init:done", flush=True)

    def current_session(self) -> WorkspaceSession:
        """Return the active independent workspace session."""
        session = self.session_tabs.currentWidget()
        print(
            f"[debug][main-window] session_tab:current index={self.session_tabs.currentIndex()} "
            f"type={type(session).__name__!r}",
            flush=True,
        )
        if not isinstance(session, WorkspaceSession):
            raise RuntimeError("Current tab is not a workspace session")
        return session

    def add_workspace_session(self) -> WorkspaceSession:
        """Create a new clean top-level workspace tab."""
        self._session_counter += 1
        title = f"Tab {self._session_counter}"
        print(f"[debug][main-window] session_tab:add:start title={title!r}", flush=True)
        session = WorkspaceSession(title, self.session_tabs)
        index = self.session_tabs.addTab(session, title)
        self._install_session_tab_header(index, title)
        self.session_tabs.setCurrentIndex(index)
        self._sync_current_session_aliases(index)
        print(
            f"[debug][main-window] session_tab:add index={index} title={title!r} count={self.session_tabs.count()}",
            flush=True,
        )
        return session

    def _install_session_tab_header(self, index: int, title: str) -> None:
        """Attach the custom tab header used for top-level workspace tabs."""
        print(f"[debug][main-window] session_tab:header index={index} title={title!r}", flush=True)
        close_container = QWidget(self.session_tab_bar)
        close_container.setObjectName("sessionTabCloseContainer")
        close_container.setFixedSize(36, 25)
        close_layout = QHBoxLayout(close_container)
        close_layout.setContentsMargins(0, 0, 14, 3)
        close_layout.setSpacing(0)

        close_button = QPushButton(close_container)
        close_button.setObjectName("sessionTabCloseButton")
        close_button.setFlat(True)
        close_button.setFixedSize(20, 20)
        close_button.setIcon(QIcon(str(_REMOVE_ICON)))
        close_button.setIconSize(QSize(17, 17))
        close_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        close_layout.addWidget(close_button)
        close_button.clicked.connect(lambda _checked=False, b=close_button: self._close_session_button_tab(b))
        self.session_tabs.setTabText(index, title)
        self.session_tab_bar.setTabButton(index, QTabBar.ButtonPosition.RightSide, close_container)

    def _close_session_button_tab(self, button: QPushButton | None) -> None:
        """Close the session associated with a right-side tab close button."""
        if button is None:
            print("[debug][main-window] session_tab:button_close skipped reason='missing_button'", flush=True)
            return
        for index in range(self.session_tab_bar.count()):
            container = self.session_tab_bar.tabButton(index, QTabBar.ButtonPosition.RightSide)
            if container is button.parent():
                print(f"[debug][main-window] session_tab:button_close index={index}", flush=True)
                self.close_workspace_session(index)
                return
        print("[debug][main-window] session_tab:button_close skipped reason='button_not_found'", flush=True)

    def close_workspace_session(self, index: int) -> None:
        """Close one workspace tab while keeping at least one clean session open."""
        print(
            f"[debug][main-window] session_tab:close:start index={index} count={self.session_tabs.count()}",
            flush=True,
        )
        if self.session_tabs.count() <= 1:
            print("[debug][main-window] session_tab:close:blocked reason='last_tab'", flush=True)
            return
        session = self.session_tabs.widget(index)
        title = session.title if isinstance(session, WorkspaceSession) else self.session_tabs.tabText(index)
        self.session_tabs.removeTab(index)
        if session is not None:
            session.deleteLater()
        self._sync_current_session_aliases(self.session_tabs.currentIndex())
        print(
            f"[debug][main-window] session_tab:close:done index={index} title={title!r} count={self.session_tabs.count()}",
            flush=True,
        )

    def _sync_current_session_aliases(self, index: int) -> None:
        """Expose active-session attributes for existing tests and integrations."""
        print(f"[debug][main-window] session_tab:sync_aliases index={index}", flush=True)
        if index < 0 or self.session_tabs.count() == 0:
            print("[debug][main-window] session_tab:sync_aliases skipped", flush=True)
            return
        session = self.current_session()
        self.graph_state = session.graph_state
        self.workspace_tabs = session.workspace_tabs
        self.notebook_tab = session.notebook_tab
        self.graphs_tab = session.graphs_tab
        self.formula_tab = session.formula_tab
        self.fft_tab = session.fft_tab
        print(
            f"[debug][main-window] session_tab:sync_aliases done title={session.title!r} "
            f"inner_tabs={self.workspace_tabs.count()}",
            flush=True,
        )
