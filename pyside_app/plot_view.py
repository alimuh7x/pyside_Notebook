from __future__ import annotations

import base64
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

_WEBGL_CHROMIUM_FLAGS = (
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--enable-gpu-rasterization",
    "--enable-unsafe-swiftshader",
)


def _request_webgl_chromium_flags() -> str:
    """Request WebGL support before Qt WebEngine is imported."""
    current = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    parts = [part for part in current.split() if part not in {"--disable-gpu", "--disable-gpu-compositing"}]
    for flag in _WEBGL_CHROMIUM_FLAGS:
        if flag not in parts:
            parts.append(flag)
    value = " ".join(parts).strip()
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = value
    print(f"[debug][plot-view] webgl:chromium_flags={value!r}", flush=True)
    return value


_request_webgl_chromium_flags()

try:
    from PySide6.QtWebEngineCore import QWebEngineSettings
except Exception:  # pragma: no cover - environment-specific fallback
    QWebEngineSettings = None

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
        self._load_revision = 0
        self.setMinimumHeight(420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Download PNG toolbar
        self._toolbar = QWidget(self)
        toolbar_layout = QHBoxLayout(self._toolbar)
        toolbar_layout.setContentsMargins(0, 2, 4, 2)
        toolbar_layout.setSpacing(0)
        toolbar_layout.addStretch(1)
        self._download_btn = QPushButton("Download PNG", self._toolbar)
        self._download_btn.setStyleSheet(
            "QPushButton {"
            " background:#3e4451; color:#d7dae0; border:1px solid #4a5568;"
            " border-radius:5px; padding:3px 10px; font-size:14px; font-weight:600;"
            "}"
            "QPushButton:hover { background:#4a5568; }"
        )
        self._download_btn.clicked.connect(self._download_png)
        toolbar_layout.addWidget(self._download_btn)
        self._layout.addWidget(self._toolbar)

        if QWebEngineView is not None:
            self._view: QWidget = QWebEngineView(self)
            self._enable_webgl()
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

    def _enable_webgl(self) -> None:
        """Enable WebEngine settings needed by Plotly 3D traces."""
        if QWebEngineSettings is None or QWebEngineView is None or not isinstance(self._view, QWebEngineView):
            print("[debug][plot-view] webgl:settings_unavailable", flush=True)
            return
        settings = self._view.settings()
        attribute_names = (
            "JavascriptEnabled",
            "WebGLEnabled",
            "Accelerated2dCanvasEnabled",
            "LocalContentCanAccessFileUrls",
            "LocalContentCanAccessRemoteUrls",
        )
        for name in attribute_names:
            attr = getattr(QWebEngineSettings.WebAttribute, name, None)
            if attr is None:
                print(f"[debug][plot-view] webgl:attribute_missing name={name!r}", flush=True)
                continue
            settings.setAttribute(attr, True)
            print(f"[debug][plot-view] webgl:attribute_enabled name={name!r}", flush=True)

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

    def _download_png(self) -> None:
        """Download the current plot as a PNG file."""
        print("[debug][plot-view] download_png:start", flush=True)
        if QWebEngineView is None or not isinstance(self._view, QWebEngineView):
            self._save_pixmap_png()
            return
        js = (
            "(function(){"
            "  window.__png_data__ = null;"
            "  var gd = document.querySelector('.js-plotly-plot');"
            "  if (!gd) { window.__png_data__ = '__no_plot__'; return; }"
            "  var w = gd.offsetWidth || 900, h = gd.offsetHeight || 500;"
            "  Plotly.toImage(gd, {format:'png', width:w, height:h, scale:2})"
            "    .then(function(url){ window.__png_data__ = url; })"
            "    .catch(function(){ window.__png_data__ = '__error__'; });"
            "})();"
        )
        self._view.page().runJavaScript(js)
        QTimer.singleShot(1500, self._read_png_data)

    def _read_png_data(self) -> None:
        """Read the PNG data URL stored by JS and trigger save."""
        self._view.page().runJavaScript("window.__png_data__ || ''", self._handle_png_data)

    def _handle_png_data(self, data_url: object) -> None:
        """Decode the base64 PNG data URL and open a save dialog."""
        url = str(data_url) if data_url else ""
        print(f"[debug][plot-view] handle_png_data status={'ok' if url.startswith('data:') else url[:30]!r}", flush=True)
        if not url.startswith("data:image/png;base64,"):
            self._save_pixmap_png()
            return
        b64 = url.split(",", 1)[1]
        try:
            png_bytes = base64.b64decode(b64)
        except Exception:
            self._save_pixmap_png()
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "graph.png", "PNG Image (*.png)")
        if not path:
            return
        Path(path).write_bytes(png_bytes)
        print(f"[debug][plot-view] download_png:saved path={path!r}", flush=True)

    def _save_pixmap_png(self) -> None:
        """Grab the widget as a pixmap and save (fallback when JS is unavailable)."""
        print("[debug][plot-view] download_png:pixmap_fallback", flush=True)
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "graph.png", "PNG Image (*.png)")
        if not path:
            return
        self._view.grab().save(path, "PNG")
        print(f"[debug][plot-view] download_png:pixmap_saved path={path!r}", flush=True)

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
            self._load_revision += 1
            file_url.setQuery(f"v={self._load_revision}")
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
