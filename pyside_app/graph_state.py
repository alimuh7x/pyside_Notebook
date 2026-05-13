from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal


class NotebookGraphState(QObject):
    """Store the shared namespace and latest plot payload for graph consumers."""

    namespace_changed = Signal(object)
    latest_plot_changed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the shared graph-state container."""
        super().__init__(parent)
        self._namespace: dict[str, Any] = {}
        self._latest_plot_title = ""
        self._latest_plot_html = ""
        print("[debug][graph-state] init", flush=True)

    @property
    def namespace(self) -> dict[str, Any]:
        """Return a copy of the most recent notebook namespace snapshot."""
        return dict(self._namespace)

    @property
    def latest_plot_title(self) -> str:
        """Return the title associated with the latest executed plot output."""
        return self._latest_plot_title

    @property
    def latest_plot_html(self) -> str:
        """Return the latest executed plot as embeddable HTML."""
        return self._latest_plot_html

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        """Publish a new namespace snapshot to all graph listeners."""
        self._namespace = dict(namespace)
        print(f"[debug][graph-state] set_namespace count={len(self._namespace)}", flush=True)
        self.namespace_changed.emit(dict(self._namespace))

    def set_latest_plot(self, title: str, html: str) -> None:
        """Publish the newest rendered plot payload to graph listeners."""
        self._latest_plot_title = title
        self._latest_plot_html = html
        print(
            f"[debug][graph-state] set_latest_plot title={title!r} html_length={len(html)}",
            flush=True,
        )
        self.latest_plot_changed.emit(title, html)

    def clear_latest_plot(self) -> None:
        """Reset the latest-plot payload and notify listeners of the clear."""
        self._latest_plot_title = ""
        self._latest_plot_html = ""
        print("[debug][graph-state] clear_latest_plot", flush=True)
        self.latest_plot_changed.emit("", "")
