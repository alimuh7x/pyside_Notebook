from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal


class NotebookGraphState(QObject):
    namespace_changed = Signal(object)
    latest_plot_changed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._namespace: dict[str, Any] = {}
        self._latest_plot_title = ""
        self._latest_plot_html = ""
        print("[debug][graph-state] init", flush=True)

    @property
    def namespace(self) -> dict[str, Any]:
        return dict(self._namespace)

    @property
    def latest_plot_title(self) -> str:
        return self._latest_plot_title

    @property
    def latest_plot_html(self) -> str:
        return self._latest_plot_html

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        self._namespace = dict(namespace)
        print(f"[debug][graph-state] set_namespace count={len(self._namespace)}", flush=True)
        self.namespace_changed.emit(dict(self._namespace))

    def set_latest_plot(self, title: str, html: str) -> None:
        self._latest_plot_title = title
        self._latest_plot_html = html
        print(
            f"[debug][graph-state] set_latest_plot title={title!r} html_length={len(html)}",
            flush=True,
        )
        self.latest_plot_changed.emit(title, html)

    def clear_latest_plot(self) -> None:
        self._latest_plot_title = ""
        self._latest_plot_html = ""
        print("[debug][graph-state] clear_latest_plot", flush=True)
        self.latest_plot_changed.emit("", "")
