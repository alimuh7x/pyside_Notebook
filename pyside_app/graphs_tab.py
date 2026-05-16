from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from pyside_app.graph_state import NotebookGraphState
from pyside_app.notebook_plot_panel import NotebookPlotPanel


class GraphsTab(QWidget):
    """Host the advanced graphs workspace and keep it synced with notebook state."""

    def __init__(self, graph_state: NotebookGraphState, parent: QWidget | None = None) -> None:
        """Create the scrollable graphs tab and connect shared graph-state signals."""
        super().__init__(parent)
        print("[debug][graphs-tab] init:start", flush=True)
        self.graph_state = graph_state

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(self.scroll.horizontalScrollBarPolicy().ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { background:#21252b; border:none; }")
        root.addWidget(self.scroll)

        container = QWidget(self.scroll)
        container.setStyleSheet("background:#21252b;")
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(12, 12, 12, 12)
        container_layout.setSpacing(12)

        self.plot_panel = NotebookPlotPanel(container, layout_mode="advanced")
        container_layout.addWidget(self.plot_panel)
        container_layout.addStretch(1)
        self.scroll.setWidget(container)

        self.graph_state.namespace_changed.connect(self._handle_namespace_changed)
        self.graph_state.latest_plot_changed.connect(self._handle_latest_plot_changed)

        self.plot_panel.set_namespace(self.graph_state.namespace)
        print("[debug][graphs-tab] init:done", flush=True)

    def _handle_namespace_changed(self, namespace: dict) -> None:
        """Refresh the graph builder when the notebook namespace changes."""
        print(f"[debug][graphs-tab] namespace_changed count={len(namespace)}", flush=True)
        self.plot_panel.set_namespace(namespace)

    def _handle_latest_plot_changed(self, title: str, html: str) -> None:
        """React to the latest executed plot update from the notebook tab."""
        print(
            f"[debug][graphs-tab] latest_plot_changed title={title!r} html_length={len(html)}",
            flush=True,
        )
