from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from pyside_app.controls import AutoCloseComboBox
from pyside_app.notebook_plot_panel import QuickGraphPreviewPanel


PRIMARY_BTN = (
    "QPushButton { background:#001f41; color:white; border-radius:6px; padding:2px 10px; "
    "min-height:24px; max-height:24px; font-weight:700; } "
    "QPushButton:hover { background:#0d3567; }"
)
SECONDARY_BTN = (
    "QPushButton { background:#374151; color:white; border-radius:6px; padding:2px 10px; "
    "min-height:24px; max-height:24px; } "
    "QPushButton:hover { background:#1f2937; }"
)
LIGHT_BTN = (
    "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:2px 10px; "
    "min-height:24px; max-height:24px; } "
    "QPushButton:hover { background:#cbd5e1; }"
)


class NotebookToolbar(QWidget):
    """Toolbar-only widget for notebook actions and status display."""

    run_all_requested = Signal()
    restart_requested = Signal()
    autosave_requested = Signal()
    save_requested = Signal()
    save_example_requested = Signal()
    open_requested = Signal()
    functions_requested = Signal()
    markdown_help_requested = Signal()

    def __init__(self, status_style: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][notebook-toolbar] init:start", flush=True)
        root = QHBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(0, 0, 0, 0)

        left_buttons = [
            ("run_all_button", "Run All", self.run_all_requested.emit, PRIMARY_BTN),
            ("restart_button", "Restart Kernel", self.restart_requested.emit, SECONDARY_BTN),
            ("autosave_button", "Autosave", self.autosave_requested.emit, LIGHT_BTN),
            ("save_button", "Save", self.save_requested.emit, LIGHT_BTN),
            ("save_example_button", "Save Example", self.save_example_requested.emit, LIGHT_BTN),
            ("open_button", "Open", self.open_requested.emit, LIGHT_BTN),
        ]
        right_buttons = [
            ("functions_toggle_button", "Functions", self.functions_requested.emit, LIGHT_BTN),
            ("markdown_toggle_button", "Markdown Help", self.markdown_help_requested.emit, LIGHT_BTN),
        ]

        for attr, text, slot, style in left_buttons:
            print(f"[debug][notebook-toolbar] add_left attr={attr}", flush=True)
            button = self._create_toolbar_button(text, slot, style)
            setattr(self, attr, button)
            root.addWidget(button)

        root.addStretch()

        for attr, text, slot, style in right_buttons:
            print(f"[debug][notebook-toolbar] add_right attr={attr}", flush=True)
            button = self._create_toolbar_button(text, slot, style)
            setattr(self, attr, button)
            root.addWidget(button)

        self.status_label = QLabel("Ready", self)
        self.status_label.setStyleSheet(status_style)
        root.addWidget(self.status_label)
        print("[debug][notebook-toolbar] init:done", flush=True)

    def _toolbar_button(self, label: str, handler: object, style: str) -> QPushButton:
        print(f"[debug][notebook-toolbar] toolbar_button label={label!r}", flush=True)
        button = QPushButton(label, self)
        button.setStyleSheet(style + " QPushButton { font-size:12px; font-weight:400; }")
        button.clicked.connect(handler)
        return button

    def _create_toolbar_button(self, text: str, slot: object, style: str) -> QPushButton:
        print(f"[debug][notebook-toolbar] create_toolbar_button text={text!r}", flush=True)
        return self._toolbar_button(text, slot, style)

    def button(self, name: str) -> QPushButton:
        print(f"[debug][notebook-toolbar] button name={name!r}", flush=True)
        return getattr(self, name)


#-------------------------------------------------------------------------------------------------------------------------
# -- Note: SidebarWidgetget
#-------------------------------------------------------------------------------------------------------------------------

class SidebarWidget(QFrame):
    """Sidebar showing only the live variables browser."""

    def __init__(
        self,
        text_style: str,
        heading_style: str,
        namespace_skip: set[str],
        summarize_value: Callable[[object], tuple[str, str]],
        combo_style: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        print("[debug][sidebar-widget] init:start", flush=True)
        self._namespace_skip = namespace_skip
        self._summarize_value = summarize_value
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #d1dce8; }"
            "QLabel { background:transparent; border:none; " + text_style + " }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.examples_bar = self._build_examples_row(combo_style)
        layout.addWidget(self.examples_bar)

        self.variables_panel = self._build_variables_panel(heading_style)
        layout.addWidget(self.variables_panel, 1)
        print("[debug][sidebar-widget] init:done", flush=True)

    def _build_examples_row(self, combo_style: str) -> "ExamplesBar":
        bar = ExamplesBar(combo_style=combo_style, parent=self)
        return bar

    def _build_variables_panel(self, heading_style: str) -> QWidget:
        print("[debug][sidebar-widget] build_variables_panel", flush=True)
        wrapper = QWidget(self)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 6, 0, 0)
        wrapper_layout.setSpacing(8)
        variables_header = QHBoxLayout()
        variables_label = QLabel("Variables", wrapper)
        variables_label.setStyleSheet(heading_style)
        variables_header.addWidget(variables_label)
        variables_header.addStretch(1)
        wrapper_layout.addLayout(variables_header)

        self.variables_browser = QTextBrowser(wrapper)
        self.variables_browser.setMinimumHeight(360)
        self.variables_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.variables_browser.setStyleSheet(
            "QTextBrowser {"
            " border:1px solid #d1dce8;"
            " background:#ffffff;"
            " font-family:'Segoe UI', 'Roboto Condensed', Arial, sans-serif;"
            " font-size:14px;"
            " color:#0f1b2b;"
            "}"
        )
        self.variables_browser.document().setDocumentMargin(8)
        wrapper_layout.addWidget(self.variables_browser, 1)
        self._refresh_variables_panel({})
        return wrapper

    def _refresh_variables_panel(self, namespace: dict[str, object]) -> None:
        print("[debug][sidebar-widget] refresh_variables_panel", flush=True)
        print(f"[debug][sidebar-widget] refresh_variables_panel_namespace count={len(namespace)}", flush=True)
        rows: list[tuple[str, str]] = []
        for name, value in sorted(namespace.items()):
            if name in self._namespace_skip or name.startswith("_"):
                continue
            if isinstance(value, ModuleType) or callable(value):
                continue
            if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
                continue
            summary, _type_name = self._summarize_value(value)
            rows.append((name, summary))
        print(f"[debug][sidebar-widget] refresh_variables_panel rows={len(rows)}", flush=True)
        if not rows:
            self.variables_browser.setHtml("<p style='color:#64748b; font-style:italic;'>Run code to see variables here.</p>")
            return

        table_rows = "".join(
            "<tr>"
            f"<td style='padding:4px 7px; border-bottom:1px solid #e2e8f0; white-space:nowrap; vertical-align:top; line-height:1.1;'>"
            f"<code style='font-family:Consolas, monospace; color:#001f41; font-size:14px; font-weight:700;'>{html.escape(name)}</code></td>"
            f"<td style='padding:4px 7px; border-bottom:1px solid #e2e8f0; font-family:Consolas, monospace; font-size:14px; color:#001f41; font-weight:700; line-height:1.1;'>{html.escape(summary)}</td>"
            "</tr>"
            for name, summary in rows
        )
        self.variables_browser.setHtml(
            "<table style='width:100%; border-collapse:collapse; font-family:Segoe UI, Arial, sans-serif; font-size:14px;'>"
            "<thead>"
            "<tr>"
            "<th style='text-align:left; padding:5px 7px; border-bottom:2px solid #d1dce8; color:#001f41; font-size:14px; font-weight:700; line-height:1.1;'>Variable</th>"
            "<th style='text-align:left; padding:5px 7px; border-bottom:2px solid #d1dce8; color:#001f41; font-size:14px; font-weight:700; line-height:1.1;'>Value</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{table_rows}</tbody></table>"
        )

#-------------------------------------------------------------------------------------------------------------------------
# -- Note: ExamplesBar
#-------------------------------------------------------------------------------------------------------------------------

class ExamplesBar(QWidget):
    """Compact top-right bar: example dropdown + Insert button."""

    insert_example_requested = Signal()

    def __init__(
        self,
        combo_style: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.examples: list[object] = []
        self.setStyleSheet("background:#f0f4f8;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        lbl = QLabel("Examples", self)
        lbl.setStyleSheet("font-weight:700; font-size:12px; color:#001f41; background:transparent;")
        layout.addWidget(lbl)

        self.example_combo = AutoCloseComboBox(self)
        self.example_combo.setStyleSheet(combo_style)
        self.example_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.example_combo, 1)

        self.insert_btn = QPushButton("Insert Example", self)
        self.insert_btn.setStyleSheet(
            "QPushButton { background:#001f41; color:white; border-radius:6px; padding:5px 14px;"
            " font-weight:600; font-size:12px; min-height:26px; }"
            "QPushButton:hover { background:#0d3567; }"
        )
        self.insert_btn.clicked.connect(self.insert_example_requested.emit)
        layout.addWidget(self.insert_btn)

    def set_examples(self, examples: Sequence[object], select_title: str | None = None) -> None:
        self.examples = list(examples)
        self.example_combo.blockSignals(True)
        self.example_combo.clear()
        selected_index = 0
        for i, ex in enumerate(self.examples):
            self.example_combo.addItem(ex.title, ex.id)  # type: ignore[attr-defined]
            if select_title and ex.title == select_title:  # type: ignore[attr-defined]
                selected_index = i
        self.example_combo.blockSignals(False)
        if self.example_combo.count():
            self.example_combo.setCurrentIndex(selected_index)


#-------------------------------------------------------------------------------------------------------------------------
# -- Note: HelpPanelsWidget
#-------------------------------------------------------------------------------------------------------------------------

class HelpPanelsWidget(QWidget):
    """Container for notebook help panels."""

    def __init__(self, functions_html: str, markdown_html: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][help-panels] init:start", flush=True)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self.setMaximumWidth(460)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.functions_panel = QTextBrowser(self)
        self.functions_panel.setHtml(functions_html)
        self.functions_panel.hide()
        self.functions_panel.setMinimumHeight(180)
        self.functions_panel.setStyleSheet("QTextBrowser { font-size:12px; color:#355070; }")
        root.addWidget(self.functions_panel)

        self.markdown_help_panel = QTextBrowser(self)
        self.markdown_help_panel.setHtml(markdown_html)
        self.markdown_help_panel.hide()
        self.markdown_help_panel.setMinimumHeight(180)
        self.markdown_help_panel.setStyleSheet("QTextBrowser { font-size:12px; color:#355070; }")
        root.addWidget(self.markdown_help_panel)
        print("[debug][help-panels] init:done", flush=True)

    def toggle_functions_panel(self) -> None:
        visible = not self.functions_panel.isVisible()
        print(f"[debug][help-panels] toggle_functions visible={visible}", flush=True)
        self.functions_panel.setVisible(visible)

    def toggle_markdown_help(self) -> None:
        visible = not self.markdown_help_panel.isVisible()
        print(f"[debug][help-panels] toggle_markdown_help visible={visible}", flush=True)
        self.markdown_help_panel.setVisible(visible)

#-------------------------------------------------------------------------------------------------------------------------
# -- Note: NotebookColumnsWidget
#-------------------------------------------------------------------------------------------------------------------------

class NotebookColumnsWidget(QWidget):
    """Widget responsible for the notebook cell column layout."""

    add_code_requested = Signal(str)
    add_markdown_requested = Signal(str)
    insert_example_requested = Signal(str)

    def __init__(
        self,
        heading_style: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        print("[debug][columns-widget] init:start", flush=True)
        self.setStyleSheet("background:#ffffff;")
        wrapper_layout = QVBoxLayout(self)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)
        header = QHBoxLayout()
        label = QLabel("Notebook", self)
        label.setStyleSheet(heading_style)
        header.addWidget(label)
        header.addStretch(1)
        header.addWidget(
            self._toolbar_button(
                "Add Code",
                self._request_add_code,
                "QPushButton { background:#001f41; color:white; border-radius:6px; padding:4px 10px; font-weight:600; } "
                "QPushButton:hover { background:#0d3567; }",
            )
        )
        header.addWidget(
            self._toolbar_button(
                "Add Markdown",
                self._request_add_markdown,
                "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:4px 10px; } "
                "QPushButton:hover { background:#cbd5e1; }",
            )
        )
        wrapper_layout.addLayout(header)

        self.container = QWidget(self)
        self.container.setStyleSheet("background:#ffffff;")
        print("[debug][columns-widget] build_column container_style column='left' background='#ffffff'", flush=True)
        self.cells_layout = QVBoxLayout(self.container)
        self.cells_layout.setContentsMargins(0, 0, 0, 0)
        self.cells_layout.setSpacing(8)
        self.cells_layout.addStretch(1)
        wrapper_layout.addWidget(self.container, 1)
        print("[debug][columns-widget] init:done", flush=True)

    def _toolbar_button(self, label: str, handler: object, style: str) -> QPushButton:
        print(f"[debug][columns-widget] toolbar_button label={label!r}", flush=True)
        button = QPushButton(label, self)
        button.setStyleSheet(style + " QPushButton { font-size:12px; font-weight:400; }")
        button.clicked.connect(handler)
        return button

    def _request_add_code(self) -> None:
        print("[debug][columns-widget] request_add_code column='left'", flush=True)
        self.add_code_requested.emit("left")

    def _request_add_markdown(self) -> None:
        print("[debug][columns-widget] request_add_markdown column='left'", flush=True)
        self.add_markdown_requested.emit("left")

    def _request_insert_example(self) -> None:
        print("[debug][columns-widget] request_insert_example column='left'", flush=True)
        self.insert_example_requested.emit("left")

    def rebuild_cells(self, cells: Sequence[QWidget]) -> None:
        print(f"[debug][columns-widget] rebuild_cells count={len(cells)}", flush=True)
        while self.cells_layout.count():
            item = self.cells_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for cell in cells:
            self.cells_layout.addWidget(cell)
        self.cells_layout.addStretch(1)


class GraphPanelWidget(QWidget):
    """Right-column graph container: holds one or more QuickGraphPreviewPanel cards."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][graph-panel-widget] init:start", flush=True)
        self._namespace: dict[str, Any] = {}
        self._current_label = "baseline"
        self._panels: list[QuickGraphPreviewPanel] = []
        self._cards: list[QWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        # ── toolbar: "Add Graph Panel" ────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 6)
        toolbar.addStretch(1)
        add_btn = QPushButton("+ Add Graph Panel", self)
        add_btn.setStyleSheet(
            "QPushButton { background:#eff6ff; color:#1d4ed8; border:1px solid #93c5fd;"
            " border-radius:6px; padding:4px 12px; font-weight:600; font-size:12px; }"
            "QPushButton:hover { background:#dbeafe; }"
        )
        add_btn.clicked.connect(self._add_panel)
        toolbar.addWidget(add_btn)
        root.addLayout(toolbar)

        # ── panels ────────────────────────────────────────────────────────────
        self._panels_layout = QVBoxLayout()
        self._panels_layout.setContentsMargins(0, 0, 0, 0)
        self._panels_layout.setSpacing(10)
        root.addLayout(self._panels_layout)
        root.addStretch(1)

        self._add_panel()
        self.quick_preview_panel = self._panels[0]   # backward compat
        print("[debug][graph-panel-widget] init:done", flush=True)

    # ── internal ──────────────────────────────────────────────────────────────

    def _add_panel(self) -> None:
        n = len(self._panels) + 1
        panel = QuickGraphPreviewPanel(self)
        if self._namespace:
            panel.set_namespace(self._namespace)
            panel.set_current_label(self._current_label)
            panel.load_saved_runs_from_store()

        # ── card wrapper ──────────────────────────────────────────────────────
        card = QFrame(self)
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #d1dce8; border-radius:8px; }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # header
        hdr = QWidget(card)
        hdr.setStyleSheet(
            "QWidget { background:#f0f4f8; border-bottom:1px solid #d1dce8;"
            " border-top-left-radius:8px; border-top-right-radius:8px; border-bottom:none; }"
            "QLabel { background:transparent; border:none; }"
            "QPushButton { background:transparent; border:none; }"
        )
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(10, 4, 6, 4)
        hdr_layout.setSpacing(6)

        title_lbl = QLabel(f"Graph Panel {n}", hdr)
        title_lbl.setStyleSheet("font-weight:700; font-size:12px; color:#001f41; background:transparent; border:none;")
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addStretch(1)

        remove_btn = QPushButton("✕", hdr)
        remove_btn.setFixedSize(20, 20)
        remove_btn.setStyleSheet(
            "QPushButton { color:#64748b; font-size:11px; font-weight:700; background:transparent; border:none; }"
            "QPushButton:hover { color:#dc2626; }"
        )
        remove_btn.clicked.connect(lambda: self._remove_panel(card, panel))
        hdr_layout.addWidget(remove_btn)
        card_layout.addWidget(hdr)

        # panel body
        card_layout.addWidget(panel)

        self._panels_layout.addWidget(card)
        self._panels.append(panel)
        self._cards.append(card)
        self._update_remove_buttons()

    def _remove_panel(self, card: QWidget, panel: "QuickGraphPreviewPanel") -> None:
        if len(self._panels) <= 1:
            return
        idx = self._panels.index(panel)
        self._panels.pop(idx)
        self._cards.pop(idx)
        card.hide()
        card.deleteLater()
        self.quick_preview_panel = self._panels[0]
        self._update_remove_buttons()
        self._renumber_panels()

    def _update_remove_buttons(self) -> None:
        only_one = len(self._cards) <= 1
        for card in self._cards:
            hdr = card.layout().itemAt(0).widget()
            if hdr:
                btn = hdr.findChild(QPushButton)
                if btn:
                    btn.setEnabled(not only_one)

    def _renumber_panels(self) -> None:
        for i, card in enumerate(self._cards):
            hdr = card.layout().itemAt(0).widget()
            if hdr:
                lbl = hdr.findChild(QLabel)
                if lbl:
                    lbl.setText(f"Graph Panel {i + 1}")

    # ── public API ────────────────────────────────────────────────────────────

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        print(f"[debug][graph-panel-widget] set_namespace count={len(namespace)}", flush=True)
        self._namespace = namespace
        self._current_label = "baseline"
        for panel in self._panels:
            panel.set_namespace(namespace)

    def add_run(self, namespace_snapshot: dict[str, Any], label: str) -> None:
        """Record a slider run in all panels for history trace overlay."""
        self._namespace = dict(namespace_snapshot)
        self._current_label = label or "current"
        for panel in self._panels:
            panel.add_run(namespace_snapshot, label)

    def clear_history(self) -> None:
        """Clear run history from all panels. Called on kernel restart."""
        for panel in self._panels:
            panel.clear_history()

    def set_latest_plot(self, title: str, html_value: str) -> None:
        print(
            f"[debug][graph-panel-widget] set_latest_plot title={title!r} html_length={len(html_value)}",
            flush=True,
        )
        self.quick_preview_panel.set_latest_plot(title, html_value)


def _fmt_scalar(v: float) -> str:
    """Format a scalar value compactly."""
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0.0):
        return f"{v:.3g}"
    return f"{v:.4g}"


def summarize_namespace_value(value: object) -> tuple[str, str]:
    """Return a rich summary for one namespace value."""
    if isinstance(value, np.ndarray):
        dtype_name = str(value.dtype)
        shape_str = "×".join(str(d) for d in value.shape) if value.ndim > 0 else "scalar"
        if value.size == 0:
            return f"[empty] shape=({shape_str}) dtype={dtype_name}", "ndarray"
        try:
            flat = value.ravel().astype(float)
            mn, mx, mean = float(flat.min()), float(flat.max()), float(flat.mean())
            stats = f"min={_fmt_scalar(mn)}  max={_fmt_scalar(mx)}  mean={_fmt_scalar(mean)}"
        except Exception:
            stats = ""
        summary = f"({shape_str}) {dtype_name}"
        if stats:
            summary += f"  ·  {stats}"
        return summary, "ndarray"
    if isinstance(value, pd.DataFrame):
        rows, cols = value.shape
        col_names = ", ".join(str(c) for c in value.columns[:6])
        if len(value.columns) > 6:
            col_names += ", …"
        return f"{rows}×{cols}  [{col_names}]", "DataFrame"
    if isinstance(value, pd.Series):
        try:
            stats = f"min={_fmt_scalar(float(value.min()))}  max={_fmt_scalar(float(value.max()))}"
        except Exception:
            stats = ""
        return f"len={len(value)} dtype={value.dtype}" + (f"  ·  {stats}" if stats else ""), "Series"
    if isinstance(value, (list, tuple)):
        if len(value) <= 6 and all(not isinstance(item, (list, tuple, dict)) for item in value):
            return repr(value), type(value).__name__
        return f"{type(value).__name__} len={len(value)}", type(value).__name__
    if isinstance(value, dict):
        keys = list(value.keys())
        if len(keys) <= 4:
            preview = ", ".join(str(key) for key in keys)
            return "{" + preview + "}", "dict"
        return f"dict keys={len(keys)}", "dict"
    if isinstance(value, str):
        return (value if len(value) <= 40 else value[:37] + "..."), "str"
    if isinstance(value, complex):
        return str(value), "complex"
    if isinstance(value, bool):
        return str(value), "bool"
    if isinstance(value, int):
        return str(value), "int"
    if isinstance(value, float):
        return _fmt_scalar(value), "float"
    if isinstance(value, go.Figure):
        n_traces = len(value.data)
        return f"Plotly Figure  {n_traces} trace{'s' if n_traces != 1 else ''}", "plot"
    return type(value).__name__, type(value).__name__
