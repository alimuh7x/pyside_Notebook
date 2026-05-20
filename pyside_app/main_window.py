from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QMainWindow, QTabWidget, QToolButton, QVBoxLayout, QWidget

from pyside_app.fft_tab import FFTTab
from pyside_app.formula_plot_tab import FormulaPlotTab
from pyside_app.graph_state import NotebookGraphState
from pyside_app.graphs_tab import GraphsTab
from pyside_app.notebook_tab import NotebookTab
from pyside_app.title_bar import TitleBar


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
            "QTabBar::tab:selected { color:#d7dae0; border-bottom:2px solid #61afef; }"
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
        self.session_tabs.setTabsClosable(True)
        self.session_tabs.setMovable(True)
        self.session_tabs.setDocumentMode(True)
        self.session_tabs.setStyleSheet(
            "QTabWidget::pane { background:#21252b; border:none; }"
            "QTabBar::tab { background:#2c313a; color:#d7dae0; padding:8px 18px;"
            " border:none; border-right:1px solid #21252b; font-size:14px; font-weight:700; }"
            "QTabBar::tab:selected { background:#3e4451; color:#ffffff; }"
            "QTabBar::tab:hover { background:#353b45; color:#ffffff; }"
        )
        self.session_tabs.currentChanged.connect(self._sync_current_session_aliases)
        self.session_tabs.tabCloseRequested.connect(self.close_workspace_session)

        self.add_session_button = QToolButton(self.session_tabs)
        self.add_session_button.setText("+")
        self.add_session_button.setToolTip("New clean notebook tab")
        self.add_session_button.clicked.connect(lambda _checked=False: self.add_workspace_session())
        self.add_session_button.setStyleSheet(
            "QToolButton { background:#001f41; color:white; border:none; border-radius:6px;"
            " padding:4px 10px; font-size:16px; font-weight:700; }"
            "QToolButton:hover { background:#b60021; }"
        )
        self.session_tabs.setCornerWidget(self.add_session_button, Qt.Corner.TopRightCorner)
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
        self.session_tabs.setCurrentIndex(index)
        self._sync_current_session_aliases(index)
        print(
            f"[debug][main-window] session_tab:add index={index} title={title!r} count={self.session_tabs.count()}",
            flush=True,
        )
        return session

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
        title = self.session_tabs.tabText(index)
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
