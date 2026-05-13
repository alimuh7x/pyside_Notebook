from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import QUrl
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover - environment-specific fallback
    QWebEngineView = None


def _local_mathjax_url() -> str:
    """Return the local MathJax asset URL used by embedded plot HTML."""
    asset_path = Path(__file__).resolve().parent.parent / "assets" / "mathjax" / "tex-svg.js"
    if not asset_path.exists():
        print(f"[debug][plot-view] mathjax:missing path={str(asset_path)!r}", flush=True)
        return ""
    url = QUrl.fromLocalFile(str(asset_path)).toString()
    print(f"[debug][plot-view] mathjax:url url={url!r}", flush=True)
    return url


class PlotView(QWidget):
    """Render Plotly HTML inside Qt using WebEngine or a QTextBrowser fallback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the HTML host widget and temporary file location."""
        super().__init__(parent)
        print("[debug][plot-view] init:start", flush=True)
        self._html_dir = Path(tempfile.gettempdir()) / "calculation_notebook_plotview"
        self._html_dir.mkdir(parents=True, exist_ok=True)
        self._html_path = self._html_dir / f"plot-{uuid.uuid4().hex}.html"
        self.setMinimumHeight(420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        if QWebEngineView is not None:
            self._view: QWidget = QWebEngineView(self)
        else:
            self._view = QTextBrowser(self)
        self._view.setMinimumHeight(420)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._layout.addWidget(self._view)
        print(
            f"[debug][plot-view] init:view view_type={type(self._view).__name__!r} min_height={self.minimumHeight()} html_path={str(self._html_path)!r}",
            flush=True,
        )
        self.destroyed.connect(self._cleanup_temp_file)
        if hasattr(self._view, "loadFinished"):
            self._view.loadFinished.connect(self._on_load_finished)

    def _on_load_finished(self, ok: bool) -> None:
        """Log the result of a WebEngine HTML load."""
        print(f"[debug][plot-view] load_finished ok={ok} path={str(self._html_path)!r}", flush=True)

    def _cleanup_temp_file(self, *args: Any) -> None:
        """Delete the temporary HTML file created for the current plot."""
        print(f"[debug][plot-view] cleanup_temp_file:start path={str(self._html_path)!r}", flush=True)
        self._html_path.unlink(missing_ok=True)
        print(f"[debug][plot-view] cleanup_temp_file:done exists={self._html_path.exists()}", flush=True)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Clean up temporary plot files when the widget closes."""
        print("[debug][plot-view] close_event:start", flush=True)
        self._cleanup_temp_file()
        super().closeEvent(event)
        print("[debug][plot-view] close_event:done", flush=True)

    def set_html(self, html: str) -> None:
        """Wrap and display raw plot HTML inside the embedded browser widget."""
        print(f"[debug][plot-view] set_html length={len(html)}", flush=True)
        mathjax_url = _local_mathjax_url()
        mathjax_loader = (
            "<script>window.MathJax = {tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]}, svg: {fontCache: 'global'}};</script>"
            f"<script src='{mathjax_url}'></script>"
            if mathjax_url
            else ""
        )
        wrapped_html = (
            "<html><head><style>"
            "html, body { margin: 0; padding: 0; overflow: hidden; background: white; width: 100%; height: 100%; }"
            "#plot-root { width: 100%; min-height: 100%; overflow: auto; background: white; display:flex; justify-content:center; align-items:flex-start; }"
            ".js-plotly-plot, .plot-container { max-width:100%; min-height:100%; margin:0 auto; }"
            ".main-svg { overflow: visible !important; }"
            "</style>"
            f"{mathjax_loader}"
            "</head><body><div id='plot-root'>"
            f"{html}"
            "</div>"
            "</body></html>"
        )
        print(
            f"[debug][plot-view] set_html wrapper_scroll enabled=False mathjax_loaded={bool(mathjax_loader)}",
            flush=True,
        )
        if hasattr(self._view, "load") and QWebEngineView is not None and isinstance(self._view, QWebEngineView):
            self._html_path.write_text(wrapped_html, encoding="utf-8")
            file_url = QUrl.fromLocalFile(str(self._html_path))
            print(f"[debug][plot-view] set_html:file_load url={file_url.toString()!r}", flush=True)
            self._view.load(file_url)
        elif hasattr(self._view, "setHtml"):
            self._view.setHtml(wrapped_html)

    def set_figure(self, figure: Any) -> None:
        """Convert a Plotly figure object to HTML and display it."""
        html = figure.to_html(
            include_plotlyjs=True,
            full_html=False,
            config={"responsive": True, "displaylogo": False},
        )
        self.set_html(html)
