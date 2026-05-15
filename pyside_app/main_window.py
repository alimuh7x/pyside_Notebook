from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from pyside_app.fft_tab import FFTTab
from pyside_app.formula_plot_tab import FormulaPlotTab
from pyside_app.graph_state import NotebookGraphState
from pyside_app.graphs_tab import GraphsTab
from pyside_app.notebook_tab import NotebookTab
from pyside_app.title_bar import TitleBar


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
            " background: #f0f4f8;"
            " border: 1px solid rgba(15, 23, 42, 0.18);"
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
            " background: #f0f4f8;"
            " border-bottom-left-radius: 8px;"
            " border-bottom-right-radius: 8px;"
            "}"
        )
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        self.graph_state = NotebookGraphState(self)
        self.workspace_tabs = QTabWidget(self.content_widget)
        self.notebook_tab = NotebookTab(self.workspace_tabs, graph_state=self.graph_state)
        self.graphs_tab = GraphsTab(self.graph_state, self.workspace_tabs)
        self.formula_tab = FormulaPlotTab(self.workspace_tabs)
        self.fft_tab = FFTTab(graph_state=self.graph_state, parent=self.workspace_tabs)
        self.workspace_tabs.addTab(self.notebook_tab, "Notebook")
        self.workspace_tabs.addTab(self.graphs_tab, "Graphs")
        self.workspace_tabs.addTab(self.formula_tab, "Formula Plot")
        self.workspace_tabs.addTab(self.fft_tab, "FFT Analysis")
        self.content_layout.addWidget(self.workspace_tabs)
        surface_layout.addWidget(self.content_widget, 1)

        shell_layout.addWidget(self.window_surface)
        self.setCentralWidget(self.window_shell)
        print("[debug][main-window] init:done", flush=True)
