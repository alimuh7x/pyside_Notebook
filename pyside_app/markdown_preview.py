from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import markdown
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QSizePolicy, QTextBrowser, QVBoxLayout, QWidget

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover - environment-specific fallback
    QWebEngineView = None


def katex_assets_dir() -> Path:
    assets = Path(__file__).resolve().parent.parent / "assets" / "katex"
    print(f"[debug][markdown-preview] katex_assets_dir path={str(assets)!r}", flush=True)
    return assets


def build_markdown_preview_html(source: str) -> str:
    print(f"[debug][markdown-preview] build_html source_len={len(source)}", flush=True)
    body_html = markdown.markdown(source, extensions=["fenced_code", "tables", "toc"])
    assets = katex_assets_dir()
    css_url = QUrl.fromLocalFile(str(assets / "katex.min.css")).toString()
    katex_js_url = QUrl.fromLocalFile(str(assets / "katex.min.js")).toString()
    auto_render_url = QUrl.fromLocalFile(str(assets / "auto-render.min.js")).toString()
    print(f"[debug][markdown-preview] build_html css_url={css_url!r}", flush=True)
    print(f"[debug][markdown-preview] build_html js_url={katex_js_url!r}", flush=True)
    return (
        "<!DOCTYPE html>"
        "<html><head>"
        "<meta charset='utf-8'/>"
        f"<link rel='stylesheet' href='{css_url}'>"
        "<style>"
        "html, body { margin: 0; padding: 0; background: #ffffff; color: #0f1b2b; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 15px; overflow: hidden; }"
        "body { padding: 6px 10px; line-height: 1.45; }"
        "h1, h2, h3, h4 { color: #001f41; margin: 0.35em 0 0.18em; line-height: 1.2; }"
        "h1:first-child, h2:first-child, h3:first-child, h4:first-child, p:first-child { margin-top: 0; }"
        "p, ul, ol, pre, table, blockquote { margin: 0.24em 0; }"
        "ul, ol { padding-left: 1.2em; }"
        "code { background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }"
        "pre { background: #f8fafc; padding: 10px 12px; border: 1px solid #d1dce8; border-radius: 6px; overflow-x: auto; }"
        "pre code { background: transparent; padding: 0; }"
        "table { border-collapse: collapse; width: 100%; }"
        "th, td { border: 1px solid #d1dce8; padding: 6px 8px; text-align: left; }"
        "blockquote { border-left: 4px solid #cbd5e1; padding-left: 10px; color: #475569; }"
        ".katex-display { overflow-x: auto; overflow-y: hidden; padding: 0; margin: 0.18em 0 !important; }"
        ".katex-display > .katex { padding: 0; }"
        ".katex { line-height: 1.15; }"
        "</style>"
        f"<script defer src='{katex_js_url}'></script>"
        f"<script defer src='{auto_render_url}'></script>"
        "</head><body>"
        f"<div id='markdown-root'>{body_html}</div>"
        "<script>"
        "document.addEventListener('DOMContentLoaded', function () {"
        "  if (typeof renderMathInElement === 'function') {"
        "    renderMathInElement(document.getElementById('markdown-root'), {"
        "      delimiters: ["
        "        {left: '$$', right: '$$', display: true},"
        "        {left: '$', right: '$', display: false},"
        "        {left: '\\\\(', right: '\\\\)', display: false},"
        "        {left: '\\\\[', right: '\\\\]', display: true}"
        "      ],"
        "      throwOnError: false"
        "    });"
        "  }"
        "});"
        "</script>"
        "</body></html>"
    )


class MarkdownPreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][markdown-preview] init:start", flush=True)
        self._minimum_preview_height = 32
        self._html_dir = Path(tempfile.gettempdir()) / "calculation_notebook_markdown_preview"
        self._html_dir.mkdir(parents=True, exist_ok=True)
        self._html_path = self._html_dir / f"markdown-{uuid.uuid4().hex}.html"
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        if QWebEngineView is not None:
            self._view: QWidget = QWebEngineView(self)
        else:
            self._view = QTextBrowser(self)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._layout.addWidget(self._view)
        self.setMinimumHeight(self._minimum_preview_height)
        self._view.setMinimumHeight(self._minimum_preview_height)
        self.setFixedHeight(self._minimum_preview_height)
        self._view.setFixedHeight(self._minimum_preview_height)
        print(f"[debug][markdown-preview] init:view view_type={type(self._view).__name__!r}", flush=True)
        if hasattr(self._view, "loadFinished"):
            self._view.loadFinished.connect(self._on_load_finished)

    def _on_load_finished(self, ok: bool) -> None:
        print(f"[debug][markdown-preview] load_finished ok={ok} path={str(self._html_path)!r}", flush=True)
        if not ok or not hasattr(self._view, "page"):
            return
        self._view.page().runJavaScript(
            "Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)",
            self._apply_height,
        )

    def _apply_height(self, value: object) -> None:
        try:
            height = max(self._minimum_preview_height, int(float(value)) + 8)
        except Exception:
            height = self._minimum_preview_height
        print(f"[debug][markdown-preview] apply_height height={height}", flush=True)
        self.setFixedHeight(height)
        self._view.setFixedHeight(height)

    def set_markdown(self, source: str) -> None:
        print(f"[debug][markdown-preview] set_markdown source_len={len(source)}", flush=True)
        html = build_markdown_preview_html(source)
        if hasattr(self._view, "load") and QWebEngineView is not None and isinstance(self._view, QWebEngineView):
            self._html_path.write_text(html, encoding="utf-8")
            file_url = QUrl.fromLocalFile(str(self._html_path))
            print(f"[debug][markdown-preview] set_markdown:file_load url={file_url.toString()!r}", flush=True)
            self._view.load(file_url)
        else:
            self._view.setHtml(html)
            self._apply_height(self._view.document().documentLayout().documentSize().height())
