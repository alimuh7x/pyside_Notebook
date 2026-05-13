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
    """Sidebar widget containing examples and the live variables browser."""

    example_insert_requested = Signal()

    def __init__(
        self,
        examples: Sequence[object],
        combo_style: str,
        text_style: str,
        heading_style: str,
        namespace_skip: set[str],
        summarize_value: Callable[[object], tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        print("[debug][sidebar-widget] init:start", flush=True)
        self.examples = list(examples)
        self._namespace_skip = namespace_skip
        self._summarize_value = summarize_value
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #d1dce8; }"
            "QLabel { background:transparent; border:none; " + text_style + " }"
            "QPushButton { font-size:12px; font-weight:400; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Notebook Sidebar", self)
        title.setStyleSheet(heading_style)
        layout.addWidget(title)

        self.example_preview = QTextBrowser(self)
        self.example_preview.setMinimumHeight(180)
        self.example_preview.setMaximumHeight(220)
        self.example_preview.setStyleSheet("font-size:12px; line-height:1.5; color:#355070;")
        layout.addWidget(self.example_preview)

        examples_label = QLabel("Examples", self)
        examples_label.setStyleSheet(heading_style)
        layout.addWidget(examples_label)

        self.example_combo = AutoCloseComboBox(self)
        self.example_combo.setStyleSheet(combo_style)
        print("[debug][sidebar-widget] example_combo style=graph_panel_combo", flush=True)
        self.example_combo.currentIndexChanged.connect(self._update_example_preview)
        layout.addWidget(self.example_combo)

        self.insert_example_button = self._toolbar_button(
            "Insert Example",
            self.example_insert_requested.emit,
            "QPushButton { background:#001f41; color:white; border-radius:6px; padding:5px 12px; font-weight:600; } "
            "QPushButton:hover { background:#0d3567; }",
        )
        layout.addWidget(self.insert_example_button)

        self.variables_panel = self._build_variables_panel(heading_style)
        layout.addWidget(self.variables_panel, 1)
        layout.addStretch(1)

        self._reload_examples_list()
        print("[debug][sidebar-widget] init:done", flush=True)

    def _toolbar_button(self, label: str, handler: object, style: str) -> QPushButton:
        print(f"[debug][sidebar-widget] toolbar_button label={label!r}", flush=True)
        button = QPushButton(label, self)
        button.setStyleSheet(style + " QPushButton { font-size:12px; font-weight:400; }")
        button.clicked.connect(handler)
        return button

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
            " font-size:12px;"
            " color:#355070;"
            "}"
        )
        self.variables_browser.document().setDocumentMargin(8)
        wrapper_layout.addWidget(self.variables_browser, 1)
        self._refresh_variables_panel({})
        return wrapper

    def _update_example_preview(self) -> None:
        if not hasattr(self, "example_combo"):
            return
        title = self.example_combo.currentText().strip()
        example = next((item for item in self.examples if getattr(item, "title", "") == title), None)
        if example is None:
            return
        print(f"[debug][sidebar-widget] update_example_preview title={title!r}", flush=True)
        tags = ", ".join(example.tags)
        self.example_preview.setHtml(
            f"<h3>{example.title}</h3>"
            f"<p><b>Category:</b> {example.category or 'general'}</p>"
            f"<p>{example.description}</p>"
            f"<p><b>Tags:</b> {tags}</p>"
        )

    def _reload_examples_list(self, select_title: str | None = None) -> None:
        print(f"[debug][sidebar-widget] reload_examples_list select_title={select_title!r}", flush=True)
        self.example_combo.blockSignals(True)
        self.example_combo.clear()
        selected_index = 0
        for index, example in enumerate(self.examples):
            self.example_combo.addItem(example.title, example.id)
            if select_title and example.title == select_title:
                selected_index = index
        self.example_combo.blockSignals(False)
        if self.example_combo.count():
            self.example_combo.setCurrentIndex(selected_index)
        self._update_example_preview()

    def set_examples(self, examples: Sequence[object], select_title: str | None = None) -> None:
        print(f"[debug][sidebar-widget] set_examples count={len(examples)}", flush=True)
        self.examples = list(examples)
        self._reload_examples_list(select_title)

    def _refresh_variables_panel(self, namespace: dict[str, object]) -> None:
        print("[debug][sidebar-widget] refresh_variables_panel", flush=True)
        print(f"[debug][sidebar-widget] refresh_variables_panel_namespace count={len(namespace)}", flush=True)
        rows: list[tuple[str, str, str]] = []
        for name, value in sorted(namespace.items()):
            if name in self._namespace_skip or name.startswith("_"):
                continue
            if isinstance(value, ModuleType) or callable(value):
                continue
            summary, type_name = self._summarize_value(value)
            rows.append((name, summary, type_name))
        print(f"[debug][sidebar-widget] refresh_variables_panel rows={len(rows)}", flush=True)
        if not rows:
            self.variables_browser.setHtml("<p style='color:#64748b; font-style:italic;'>Run code to see variables here.</p>")
            return
        table_rows = "".join(
            "<tr>"
            f"<td style='padding:6px 8px; border-bottom:1px solid #e2e8f0;'><code>{html.escape(name)}</code></td>"
            f"<td style='padding:6px 8px; border-bottom:1px solid #e2e8f0;'>{html.escape(summary)}</td>"
            f"<td style='padding:6px 8px; border-bottom:1px solid #e2e8f0; color:#64748b;'>{html.escape(type_name)}</td>"
            "</tr>"
            for name, summary, type_name in rows
        )
        self.variables_browser.setHtml(
            "<table style='width:100%; border-collapse:collapse;'>"
            "<thead>"
            "<tr>"
            "<th style='text-align:left; padding:6px 8px; border-bottom:2px solid #d1dce8;'>Variable</th>"
            "<th style='text-align:left; padding:6px 8px; border-bottom:2px solid #d1dce8;'>Value</th>"
            "<th style='text-align:left; padding:6px 8px; border-bottom:2px solid #d1dce8;'>Type</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{table_rows}</tbody></table>"
        )

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
        header.addWidget(
            self._toolbar_button(
                "Insert Example",
                self._request_insert_example,
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
    """Thin wrapper around the notebook quick graph preview panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][graph-panel-widget] init:start", flush=True)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.quick_preview_panel = QuickGraphPreviewPanel(self)
        root.addWidget(self.quick_preview_panel)
        print("[debug][graph-panel-widget] init:done", flush=True)

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        print(f"[debug][graph-panel-widget] set_namespace count={len(namespace)}", flush=True)
        self.quick_preview_panel.set_namespace(namespace)

    def set_latest_plot(self, title: str, html_value: str) -> None:
        print(
            f"[debug][graph-panel-widget] set_latest_plot title={title!r} html_length={len(html_value)}",
            flush=True,
        )
        self.quick_preview_panel.set_latest_plot(title, html_value)


def summarize_namespace_value(value: object) -> tuple[str, str]:
    """Return a compact summary for one namespace value."""
    if isinstance(value, np.ndarray):
        shape = value.shape
        summary = f"shape={shape}"
        if value.ndim == 1:
            summary += f" len={value.size}"
        return summary, "ndarray"
    if isinstance(value, pd.DataFrame):
        return f"DataFrame shape={value.shape}", "DataFrame"
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
    if isinstance(value, (int, float, complex, bool)):
        return str(value), type(value).__name__
    if isinstance(value, go.Figure):
        return "Plotly Figure", "plot"
    return type(value).__name__, type(value).__name__
