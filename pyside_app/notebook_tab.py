from __future__ import annotations

import html
from pathlib import Path
from types import ModuleType
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from pyside_app.cell_widgets import NotebookCellWidget, NotebookCodeEditor
from pyside_app.controls import AutoCloseComboBox
from pyside_app.editor_intelligence import NotebookLspClient, build_completion_words
from pyside_app.notebook_examples import (
    get_desktop_notebook_examples,
    get_notebook_examples_dir,
    suggest_example_filename,
)
from pyside_app.functions_reference import build_functions_reference_html
from pyside_app.execution_engine import ExecutionEngine, ExecutionOutput, ExecutionResult
from pyside_app.execution_worker import ExecutionWorker
from pyside_app.graph_state import NotebookGraphState
from pyside_app.notebook_plot_panel import NotebookGraphWorkspace, QuickGraphPreviewPanel
from pyside_app.storage import NotebookDocument, NotebookStorage


class NotebookTab(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        lsp_client: NotebookLspClient | None = None,
        graph_state: NotebookGraphState | None = None,
    ) -> None:
        super().__init__(parent)
        print("[debug][notebook-tab] init:start", flush=True)
        self.execution_engine = ExecutionEngine()
        self.lsp_client = lsp_client or NotebookLspClient(self)
        self.graph_state = graph_state or NotebookGraphState(self)
        self.storage = NotebookStorage()
        self.thread_pool = QThreadPool.globalInstance()
        self.current_path: Path | None = None
        self.autosave_path = Path.home() / ".calculation_notebook_autosave.json"
        self.cells: list[NotebookCellWidget] = []
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(10000)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(lambda: self.autosave_now(self.autosave_path))
        self._done_timer = QTimer(self)
        self._done_timer.setInterval(1200)
        self._done_timer.setSingleShot(True)
        self._done_timer.timeout.connect(self._set_ready_status)
        self._run_all_queue: list[NotebookCellWidget] = []
        self._is_running = False
        self._active_workers: set[ExecutionWorker] = set()
        self._is_loading_document = False
        self._namespace_skip = set(self.execution_engine._base_namespace.keys()) | {"__builtins__"}
        self.examples_dir = get_notebook_examples_dir()
        self.examples = get_desktop_notebook_examples()
        self.lsp_client.completions_ready.connect(self._handle_lsp_completions)
        self.lsp_client.diagnostics_ready.connect(self._handle_lsp_diagnostics)
        self.lsp_client.availability_changed.connect(self._handle_lsp_availability)
        self.lsp_client.start()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.workspace_scroll = QScrollArea(self)
        self.workspace_scroll.setWidgetResizable(True)
        self.workspace_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.workspace_scroll.setStyleSheet("QScrollArea { background:#f0f4f8; border:none; }")
        print("[debug][notebook-tab] init:workspace_scroll enabled=True", flush=True)
        root.addWidget(self.workspace_scroll)

        self.workspace_content = QWidget(self.workspace_scroll)
        self.workspace_content.setStyleSheet("background:#f0f4f8;")
        self.workspace_scroll.setWidget(self.workspace_content)

        workspace_layout = QVBoxLayout(self.workspace_content)
        workspace_layout.setContentsMargins(12, 12, 12, 12)
        workspace_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self.workspace_content)
        workspace_layout.addWidget(self.main_splitter)

        self.sidebar = self._build_sidebar()
        self.main_splitter.addWidget(self.sidebar)

        content = QWidget(self)
        content_layout = QVBoxLayout(content)
        toolbar = QHBoxLayout()
        self.run_all_button = self._toolbar_button(
            "Run All",
            self.run_all_cells,
            "QPushButton { background:#001f41; color:white; border-radius:6px; padding:5px 12px; font-weight:700; } "
            "QPushButton:hover { background:#0d3567; }",
        )
        toolbar.addWidget(self.run_all_button)
        self.restart_button = self._toolbar_button(
            "Restart Kernel",
            self.restart_kernel,
            "QPushButton { background:#374151; color:white; border-radius:6px; padding:5px 12px; } "
            "QPushButton:hover { background:#1f2937; }",
        )
        toolbar.addWidget(self.restart_button)
        self.autosave_button = self._toolbar_button(
            "Autosave",
            lambda: self.autosave_now(self.autosave_path),
            "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:5px 12px; } "
            "QPushButton:hover { background:#cbd5e1; }",
        )
        toolbar.addWidget(self.autosave_button)
        self.save_button = self._toolbar_button(
            "Save",
            self.save_document,
            "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:5px 12px; } "
            "QPushButton:hover { background:#cbd5e1; }",
        )
        toolbar.addWidget(self.save_button)
        self.save_example_button = self._toolbar_button(
            "Save Example",
            self.save_example_document,
            "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:5px 12px; } "
            "QPushButton:hover { background:#cbd5e1; }",
        )
        toolbar.addWidget(self.save_example_button)
        self.open_button = self._toolbar_button(
            "Open",
            self.open_document,
            "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:5px 12px; } "
            "QPushButton:hover { background:#cbd5e1; }",
        )
        toolbar.addWidget(self.open_button)
        toolbar.addStretch(1)
        self.functions_toggle_button = self._toolbar_button(
            "Functions",
            self.toggle_functions_panel,
            "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:5px 12px; } "
            "QPushButton:hover { background:#cbd5e1; }",
        )
        toolbar.addWidget(self.functions_toggle_button)
        self.markdown_toggle_button = self._toolbar_button(
            "Markdown Help",
            self.toggle_markdown_help,
            "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:5px 12px; } "
            "QPushButton:hover { background:#cbd5e1; }",
        )
        toolbar.addWidget(self.markdown_toggle_button)
        self.status_label = QLabel("Ready", self)
        self.status_label.setStyleSheet("color: #15803d; font-weight: 600;")
        toolbar.addWidget(self.status_label)
        content_layout.addLayout(toolbar)

        self.help_panels_container = QWidget(content)
        help_panels_layout = QVBoxLayout(self.help_panels_container)
        help_panels_layout.setContentsMargins(0, 0, 0, 0)
        help_panels_layout.setSpacing(8)
        self.help_panels_container.setMaximumWidth(460)
        self.help_panels_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.functions_panel = QTextBrowser(self.help_panels_container)
        self.functions_panel.setHtml(self._functions_reference_html())
        self.functions_panel.hide()
        self.functions_panel.setMinimumHeight(180)
        help_panels_layout.addWidget(self.functions_panel)

        self.markdown_help_panel = QTextBrowser(self.help_panels_container)
        self.markdown_help_panel.setHtml(self._markdown_help_html())
        self.markdown_help_panel.hide()
        self.markdown_help_panel.setMinimumHeight(180)
        help_panels_layout.addWidget(self.markdown_help_panel)

        help_container_row = QHBoxLayout()
        help_container_row.setContentsMargins(0, 0, 0, 0)
        help_container_row.addStretch(1)
        help_container_row.addWidget(self.help_panels_container)
        content_layout.addLayout(help_container_row)

        self.columns_splitter = QSplitter(Qt.Orientation.Horizontal, content)
        self.left_column_widget, self.left_cells_layout = self._build_column("left", "Notebook")
        self.right_column_widget = self._build_graphs_panel()
        self.columns_splitter.addWidget(self.left_column_widget)
        self.columns_splitter.addWidget(self.right_column_widget)
        content_layout.addWidget(self.columns_splitter, 1)
        self.main_splitter.addWidget(content)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([280, 1160])
        self.columns_splitter.setChildrenCollapsible(False)
        self.columns_splitter.setSizes([840, 420])

        self.add_code_cell("left")
        self._refresh_completion_words()
        print("[debug][notebook-tab] init:done", flush=True)

    def closeEvent(self, event: QCloseEvent) -> None:
        print("[debug][notebook-tab] close_event:start", flush=True)
        shutdown = getattr(self.lsp_client, "shutdown", None)
        if callable(shutdown):
            shutdown()
        super().closeEvent(event)
        print("[debug][notebook-tab] close_event:done", flush=True)

    def _cell_count_label(self) -> str:
        count = len(self.cells)
        label = f"{count} cell{'s' if count != 1 else ''}"
        print(f"[debug][notebook-tab] cell_count_label label={label!r}", flush=True)
        return label

    def _code_cells(self) -> list[NotebookCellWidget]:
        code_cells = [cell for cell in self.cells if cell.cell_type == "code"]
        print(f"[debug][notebook-tab] code_cells count={len(code_cells)}", flush=True)
        return code_cells

    def _document_uri_for_cell(self, cell_id: str) -> str:
        uri = f"file:///desktop-notebook/{cell_id}.py"
        print(f"[debug][notebook-tab] document_uri_for_cell cell_id={cell_id!r} uri={uri!r}", flush=True)
        return uri

    def _completion_words(self) -> list[str]:
        namespace = self.execution_engine.get_namespace()
        print(f"[debug][notebook-tab] completion_words_namespace count={len(namespace)}", flush=True)
        words = build_completion_words(namespace)
        print(f"[debug][notebook-tab] completion_words count={len(words)}", flush=True)
        return words

    def _refresh_completion_words(self) -> None:
        words = self._completion_words()
        for cell in self._code_cells():
            if isinstance(cell.editor, NotebookCodeEditor):
                print(f"[debug][notebook-tab] refresh_completion_words cell_id={cell.cell_id!r}", flush=True)
                cell.editor.set_completion_words(words)

    def _handle_lsp_availability(self, available: bool) -> None:
        print(f"[debug][notebook-tab] lsp_availability available={available}", flush=True)

    def _handle_lsp_completions(self, uri: str, labels: list[str]) -> None:
        print(f"[debug][notebook-tab] handle_lsp_completions uri={uri!r} count={len(labels)}", flush=True)
        for cell in self._code_cells():
            if isinstance(cell.editor, NotebookCodeEditor) and cell.editor.document_uri == uri:
                cell.editor.set_lsp_completions(labels)
                return

    def _handle_lsp_diagnostics(self, uri: str, diagnostics: list[dict[str, object]]) -> None:
        print(f"[debug][notebook-tab] handle_lsp_diagnostics uri={uri!r} count={len(diagnostics)}", flush=True)
        for cell in self._code_cells():
            if isinstance(cell.editor, NotebookCodeEditor) and cell.editor.document_uri == uri:
                cell.editor.apply_diagnostics(diagnostics)
                return

    def _build_sidebar(self) -> QWidget:
        frame = QFrame(self)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("QFrame { background:#ffffff; border:1px solid #d1dce8; }")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        title = QLabel("Notebook Sidebar", frame)
        title.setStyleSheet("color:#001f41; font-weight:700; font-size:22px;")
        layout.addWidget(title)

        self.example_preview = QTextBrowser(frame)
        self.example_preview.setMinimumHeight(180)
        self.example_preview.setMaximumHeight(220)
        self.example_preview.setStyleSheet("font-size:15px; line-height:1.5;")
        layout.addWidget(self.example_preview)

        examples_label = QLabel("Examples", frame)
        examples_label.setStyleSheet("color:#001f41; font-weight:600; font-size:18px;")
        layout.addWidget(examples_label)
        self.example_combo = AutoCloseComboBox(frame)
        self.example_combo.setStyleSheet(
            "QComboBox {"
            " border:1px solid #d1dce8;"
            " background:#ffffff;"
            " font-size:15px;"
            " padding:6px 8px;"
            " min-height:32px;"
            "} "
            "QComboBox QAbstractItemView {"
            " font-size:15px;"
            " selection-background-color:#c7def5;"
            " selection-color:#0f1b2b;"
            " outline:0;"
            "} "
            "QComboBox QAbstractItemView::item:hover {"
            " background:#c7def5;"
            " color:#0f1b2b;"
            "}"
        )
        for example in self.examples:
            self.example_combo.addItem(example.title, example.id)
        self.example_combo.currentIndexChanged.connect(self._update_example_preview)
        layout.addWidget(self.example_combo)
        if self.example_combo.count():
            self.example_combo.setCurrentIndex(0)
        self.insert_example_button = self._toolbar_button(
            "Insert Example",
            lambda: self.insert_example(),
            "QPushButton { background:#001f41; color:white; border-radius:6px; padding:5px 12px; font-weight:600; } "
            "QPushButton:hover { background:#0d3567; }",
        )
        layout.addWidget(self.insert_example_button)

        self.variables_panel = self._build_variables_panel(frame)
        layout.addWidget(self.variables_panel, 1)
        layout.addStretch(1)
        self._update_example_preview()
        return frame

    def _update_example_preview(self) -> None:
        if not hasattr(self, "example_combo"):
            return
        title = self.example_combo.currentText().strip()
        example = next((item for item in self.examples if item.title == title), None)
        if example is None:
            return
        print(f"[debug][notebook-tab] update_example_preview title={title!r}", flush=True)
        tags = ", ".join(example.tags)
        self.example_preview.setHtml(
            f"<h3>{example.title}</h3>"
            f"<p><b>Category:</b> {example.category or 'general'}</p>"
            f"<p>{example.description}</p>"
            f"<p><b>Tags:</b> {tags}</p>"
        )

    def _reload_examples_list(self, select_title: str | None = None) -> None:
        print(f"[debug][notebook-tab] reload_examples_list select_title={select_title!r}", flush=True)
        self.examples = get_desktop_notebook_examples()
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

    def _functions_reference_html(self) -> str:
        print("[debug][notebook-tab] functions_reference delegate", flush=True)
        return build_functions_reference_html()

    def _markdown_help_html(self) -> str:
        print("[debug][notebook-tab] markdown_help_html", flush=True)
        html = (
            "<h3>Markdown Help</h3>"
            "<p><b>Headings:</b> <code># Heading</code>, <code>## Section</code>, <code>### Subsection</code></p>"
            "<p><b>Inline styles:</b> <code>**bold**</code>, <code>*italic*</code>, <code>`code`</code></p>"
            "<p><b>Lists:</b></p>"
            "<pre>- item one\n- item two\n1. first\n2. second</pre>"
            "<p><b>Links:</b> <code>[OpenPhase](https://example.com)</code></p>"
            "<p><b>Code fences:</b></p>"
            "<pre>```python\nvalue = 42\nprint(value)\n```</pre>"
            "<p><b>Tables:</b></p>"
            "<pre>| Name | Value |\n| --- | ---: |\n| A | 10 |</pre>"
        )
        return html

    def toggle_functions_panel(self) -> None:
        visible = not self.functions_panel.isVisible()
        print(f"[debug][notebook-tab] toggle_functions visible={visible}", flush=True)
        self.functions_panel.setVisible(visible)

    def toggle_markdown_help(self) -> None:
        visible = not self.markdown_help_panel.isVisible()
        print(f"[debug][notebook-tab] toggle_markdown_help visible={visible}", flush=True)
        self.markdown_help_panel.setVisible(visible)

    def _build_column(self, column: str, title: str) -> tuple[QWidget, QVBoxLayout]:
        wrapper = QWidget(self)
        wrapper.setStyleSheet("background:#ffffff;")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(8)
        header = QHBoxLayout()
        label = QLabel(title, wrapper)
        label.setStyleSheet("color:#001f41; font-weight:700; font-size:13px;")
        header.addWidget(label)
        header.addStretch(1)
        add_code = self._toolbar_button(
            "Add Code",
            lambda checked=False, target=column: self.add_code_cell(target),
            "QPushButton { background:#001f41; color:white; border-radius:6px; padding:4px 10px; font-weight:600; } "
            "QPushButton:hover { background:#0d3567; }",
        )
        header.addWidget(add_code)
        add_md = self._toolbar_button(
            "Add Markdown",
            lambda checked=False, target=column: self.add_markdown_cell(target),
            "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:4px 10px; } "
            "QPushButton:hover { background:#cbd5e1; }",
        )
        header.addWidget(add_md)
        insert_example = self._toolbar_button(
            "Insert Example",
            lambda checked=False, target=column: self.insert_example(target),
            "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:4px 10px; } "
            "QPushButton:hover { background:#cbd5e1; }",
        )
        header.addWidget(insert_example)
        wrapper_layout.addLayout(header)

        container = QWidget(wrapper)
        container.setStyleSheet("background:#ffffff;")
        print(f"[debug][notebook-tab] build_column container_style column={column!r} background='#ffffff'", flush=True)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addStretch(1)
        wrapper_layout.addWidget(container, 1)

        self.left_container = container
        return wrapper, layout

    def _build_variables_panel(self, parent: QWidget | None = None) -> QWidget:
        wrapper = QWidget(parent or self)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 6, 0, 0)
        wrapper_layout.setSpacing(8)
        variables_header = QHBoxLayout()
        variables_label = QLabel("Variables", wrapper)
        variables_label.setStyleSheet("color:#001f41; font-weight:700; font-size:15px;")
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
            " font-size:14px;"
            "}"
        )
        self.variables_browser.document().setDocumentMargin(8)
        wrapper_layout.addWidget(self.variables_browser, 1)
        self._refresh_variables_panel()
        return wrapper

    def _build_graphs_panel(self) -> QWidget:
        print("[debug][notebook-tab] build_graphs_panel:quick_preview_panel", flush=True)
        self.quick_preview_panel = QuickGraphPreviewPanel(self)
        return self.quick_preview_panel

    def _summarize_namespace_value(self, value: object) -> tuple[str, str]:
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

    def _refresh_variables_panel(self) -> None:
        print("[debug][notebook-tab] refresh_variables_panel", flush=True)
        namespace = self.execution_engine.get_namespace()
        print(f"[debug][notebook-tab] refresh_variables_panel_namespace count={len(namespace)}", flush=True)
        rows: list[tuple[str, str, str]] = []
        for name, value in sorted(namespace.items()):
            if name in self._namespace_skip or name.startswith("_"):
                continue
            if isinstance(value, ModuleType) or callable(value):
                continue
            summary, type_name = self._summarize_namespace_value(value)
            rows.append((name, summary, type_name))
        print(f"[debug][notebook-tab] refresh_variables_panel rows={len(rows)}", flush=True)
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

    def _rebuild_columns(self) -> None:
        print(f"[debug][notebook-tab] rebuild_columns count={len(self.cells)}", flush=True)
        while self.left_cells_layout.count():
            item = self.left_cells_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for cell in self.cells:
            cell.set_column("left")
            cell.move_left_btn.hide()
            cell.move_right_btn.hide()
            self.left_cells_layout.addWidget(cell)
        self.left_cells_layout.addStretch(1)
        self._refresh_variables_panel()
        self._refresh_graphs_panel()

    def _refresh_graphs_panel(self) -> None:
        print("[debug][notebook-tab] refresh_graphs_panel", flush=True)
        namespace = self.execution_engine.get_namespace()
        print(f"[debug][notebook-tab] refresh_graphs_panel_namespace count={len(namespace)}", flush=True)
        self.graph_state.set_namespace(namespace)
        self.quick_preview_panel.set_namespace(namespace)
        title, html = self._latest_plot_output()
        if html:
            self.graph_state.set_latest_plot(title, html)
        else:
            self.graph_state.clear_latest_plot()
        self.quick_preview_panel.set_latest_plot(title, html)

    def _latest_plot_output(self) -> tuple[str, str]:
        print("[debug][notebook-tab] latest_plot_output:start", flush=True)
        latest_title = ""
        latest_html = ""
        for cell in self.cells:
            result = getattr(cell, "last_result", None)
            if result is None:
                continue
            for output in result.outputs:
                if output.kind == "plotly":
                    latest_title = self._graph_title_for_cell(cell)
                    latest_html = output.data.get("html", "")
                    print(
                        f"[debug][notebook-tab] latest_plot_output:plotly title={latest_title!r} html_length={len(latest_html)}",
                        flush=True,
                    )
                elif output.kind == "html" and "data:image" in output.data.get("html", ""):
                    latest_title = self._graph_title_for_cell(cell)
                    latest_html = output.data.get("html", "")
                    print(
                        f"[debug][notebook-tab] latest_plot_output:image title={latest_title!r} html_length={len(latest_html)}",
                        flush=True,
                    )
        print(
            f"[debug][notebook-tab] latest_plot_output:done title={latest_title!r} html_length={len(latest_html)}",
            flush=True,
        )
        return latest_title, latest_html

    def _graph_title_for_cell(self, cell: NotebookCellWidget) -> str:
        source_lines = [line.strip() for line in cell.source().splitlines() if line.strip()]
        title = source_lines[0] if source_lines else getattr(cell, "cell_id", "plot")
        if len(title) > 48:
            title = title[:45] + "..."
        print(f"[debug][notebook-tab] graph_title title={title!r}", flush=True)
        return title

    def _apply_layout_state(self, left_column_width_pct: int) -> None:
        total = max(1, self.columns_splitter.width() or 1000)
        left = max(320, int(total * left_column_width_pct / 100.0))
        right = max(320, total - left)
        print(
            f"[debug][notebook-tab] apply_layout_state pct={left_column_width_pct} total={total} left={left} right={right}",
            flush=True,
        )
        self.columns_splitter.setSizes([left, right])

    def _toolbar_button(self, label: str, handler: object, style: str) -> QPushButton:
        print(f"[debug][notebook-tab] toolbar_button label={label!r}", flush=True)
        button = QPushButton(label, self)
        button.setStyleSheet(style)
        button.clicked.connect(handler)
        return button

    def _set_ready_status(self) -> None:
        print("[debug][notebook-tab] status:ready", flush=True)
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: #15803d; font-weight: 600;")

    def _insert_cell(self, cell: NotebookCellWidget) -> None:
        self.cells.append(cell)
        self._rebuild_columns()
        cell.editor.textChanged.connect(self.schedule_autosave)
        if self._is_loading_document:
            return
        self.status_label.setText(self._cell_count_label())
        self.status_label.setStyleSheet("color: #64748b; font-weight: 600;")

    def _create_cell(self, cell_type: str, source: str = "", column: str = "left") -> NotebookCellWidget:
        temp_cell_id = f"cell-{uuid.uuid4().hex[:8]}"
        document_uri = self._document_uri_for_cell(temp_cell_id) if cell_type == "code" else ""
        words = self._completion_words() if cell_type == "code" else None
        print(
            f"[debug][notebook-tab] create_cell cell_type={cell_type!r} column={column!r} uri={document_uri!r}",
            flush=True,
        )
        cell = NotebookCellWidget(
            cell_type,
            source,
            "left",
            None,
            "",
            words,
            self.run_cell,
            self.delete_cell,
            self.move_cell,
            self.move_cell_to_column,
            self,
        )
        if cell_type == "code" and isinstance(cell.editor, NotebookCodeEditor):
            cell.editor.document_uri = self._document_uri_for_cell(cell.cell_id)
            cell.editor.attach_lsp_client(self.lsp_client, cell.editor.document_uri)
        return cell

    def add_code_cell(self, column: str = "left", source: str = "") -> None:
        print(f"[debug][notebook-tab] add_code_cell column={column!r}", flush=True)
        self._insert_cell(self._create_cell("code", source, column))

    def add_markdown_cell(self, column: str = "left", source: str = "") -> None:
        print(f"[debug][notebook-tab] add_markdown_cell column={column!r}", flush=True)
        self._insert_cell(self._create_cell("markdown", source, column))

    def insert_example(self, column: str = "left") -> None:
        title = self.example_combo.currentText().strip() if hasattr(self, "example_combo") else ""
        example = next((item for item in self.examples if item.title == title), None)
        print(f"[debug][notebook-tab] insert_example title={title!r} column={column!r}", flush=True)
        if example is None:
            return
        if (
            len(self.cells) == 1
            and self.cells[0].cell_type == "code"
            and not self.cells[0].source().strip()
            and self.cells[0].last_result is None
        ):
            self.delete_cell(self.cells[0])
        for payload in example.cells:
            cell_type = payload.get("type", "code")
            source = payload.get("source", "")
            if cell_type == "markdown":
                self.add_markdown_cell(column, source)
            else:
                self.add_code_cell(column, source)
        self.schedule_autosave()

    def delete_cell(self, cell: NotebookCellWidget) -> None:
        print(f"[debug][notebook-tab] delete_cell cell_id={cell.cell_id!r}", flush=True)
        if cell in self.cells:
            self.cells.remove(cell)
        cell.hide()
        cell.deleteLater()
        self._rebuild_columns()
        self.schedule_autosave()
        self.status_label.setText(self._cell_count_label())
        self.status_label.setStyleSheet("color: #64748b; font-weight: 600;")

    def move_cell(self, cell: NotebookCellWidget, direction: int) -> None:
        if cell not in self.cells:
            return
        old_index = self.cells.index(cell)
        new_index = max(0, min(len(self.cells) - 1, old_index + direction))
        if new_index == old_index:
            return
        print(
            f"[debug][notebook-tab] move_cell cell_id={cell.cell_id!r} "
            f"old_index={old_index} new_index={new_index}",
            flush=True,
        )
        self.cells.pop(old_index)
        self.cells.insert(new_index, cell)
        self._rebuild_columns()
        self.schedule_autosave()

    def move_cell_to_column(self, cell: NotebookCellWidget, column: str) -> None:
        if cell not in self.cells or column != "left":
            return

    def run_cell(self, cell: NotebookCellWidget) -> None:
        print(f"[debug][notebook-tab] run_cell cell_id={cell.cell_id!r}", flush=True)
        if self._is_running and not self._run_all_queue:
            self.status_label.setText("Execution already running")
            self.status_label.setStyleSheet("color: #b60021; font-weight: 600;")
            return
        self.status_label.setText("Running cell...")
        self.status_label.setStyleSheet("color: #2563eb; font-weight: 600;")
        self._is_running = True
        worker = ExecutionWorker(self.execution_engine, cell.source())
        self._active_workers.add(worker)
        worker.signals.finished.connect(
            lambda result, target=cell, current_worker=worker: self._handle_worker_finished(current_worker, target, result)
        )
        self.thread_pool.start(worker)

    def _handle_worker_finished(self, worker: ExecutionWorker, cell: NotebookCellWidget, result: ExecutionResult) -> None:
        print(f"[debug][notebook-tab] handle_worker_finished cell_id={cell.cell_id!r}", flush=True)
        self._active_workers.discard(worker)
        self._handle_cell_result(cell, result)

    def _handle_cell_result(self, cell: NotebookCellWidget, result: ExecutionResult) -> None:
        print(f"[debug][notebook-tab] handle_cell_result cell_id={cell.cell_id!r}", flush=True)
        cell.set_result(result)
        self._refresh_completion_words()
        self._refresh_variables_panel()
        self._refresh_graphs_panel()
        self._is_running = False
        if result.error:
            self._run_all_queue.clear()
            self.status_label.setText("Execution error")
            self.status_label.setStyleSheet("color: #b60021; font-weight: 600;")
        elif self._run_all_queue:
            next_cell = self._run_all_queue.pop(0)
            self.run_cell(next_cell)
            return
        else:
            self.status_label.setText("Done")
            self.status_label.setStyleSheet("color: #15803d; font-weight: 600;")
            self._done_timer.start()
        self.schedule_autosave()

    def run_all_cells(self) -> None:
        print(f"[debug][notebook-tab] run_all_cells count={len(self.cells)}", flush=True)
        code_cells = [cell for cell in self.cells if cell.cell_type == "code"]
        if not code_cells:
            self.status_label.setText("No code cells")
            self.status_label.setStyleSheet("color: #64748b; font-weight: 600;")
            return
        self._run_all_queue = code_cells[1:]
        self.run_cell(code_cells[0])

    def restart_kernel(self) -> None:
        print("[debug][notebook-tab] restart_kernel", flush=True)
        self._done_timer.stop()
        self._run_all_queue.clear()
        self._is_running = False
        restarted_now = self.execution_engine.restart()
        for cell in self.cells:
            cell.clear_output()
            if isinstance(cell.editor, NotebookCodeEditor):
                cell.editor.apply_diagnostics([])
                cell.editor.set_lsp_completions([])
        self._refresh_completion_words()
        self._refresh_variables_panel()
        self._refresh_graphs_panel()
        self.status_label.setText("Kernel restarted" if restarted_now else "Kernel restart pending")
        self.status_label.setStyleSheet("color: #15803d; font-weight: 600;" if restarted_now else "color: #2563eb; font-weight: 600;")
        self.schedule_autosave()

    def to_document(self) -> NotebookDocument:
        cells = []
        for cell in self.cells:
            outputs = []
            if cell.last_result is not None:
                for output in cell.last_result.outputs:
                    payload = dict(output.data)
                    outputs.append({"kind": output.kind, "data": {k: v for k, v in payload.items() if k in {"text", "html"}}})
            cells.append(
                {
                    "id": cell.cell_id,
                    "type": cell.cell_type,
                    "column": cell.column,
                    "panel_width": getattr(cell, "panel_width", None),
                    "source": cell.source(),
                    "outputs": outputs,
                }
            )
        return NotebookDocument(
            cells=cells,
            metadata={
                "version": self.storage.version,
                "title": self.current_path.stem.replace("_", " ").strip() if self.current_path is not None else "Desktop Notebook",
            },
            layout={"left_column_width_pct": self._current_left_column_pct()},
        )

    def _current_left_column_pct(self) -> int:
        sizes = self.columns_splitter.sizes()
        if len(sizes) < 2:
            return 50
        total = max(1, sum(sizes))
        pct = int(round((sizes[0] / total) * 100))
        print(f"[debug][notebook-tab] current_left_column_pct pct={pct}", flush=True)
        return pct

    def load_document(self, document: NotebookDocument) -> None:
        print(f"[debug][notebook-tab] load_document count={len(document.cells)}", flush=True)
        self._is_loading_document = True
        for cell in list(self.cells):
            self.delete_cell(cell)
        if not document.cells:
            self._is_loading_document = False
            self.add_code_cell()
            return
        try:
            for payload in document.cells:
                cell = self._create_cell(
                    payload.get("type", "code"),
                    payload.get("source", ""),
                    payload.get("column", "left"),
                )
                cell.panel_width = payload.get("panel_width")
                self._insert_cell(cell)
                if payload.get("outputs"):
                    restored = ExecutionResult(
                        outputs=[ExecutionOutput(kind=output.get("kind", "value"), data=dict(output.get("data") or {})) for output in payload.get("outputs", [])],
                        error=None,
                    )
                    cell.set_result(restored)
        finally:
            self._is_loading_document = False
        self._rebuild_columns()
        self._refresh_completion_words()
        self.status_label.setText(self._cell_count_label())
        self.status_label.setStyleSheet("color: #64748b; font-weight: 600;")
        QTimer.singleShot(0, lambda: self._apply_layout_state(int(document.layout.get("left_column_width_pct", 50))))
        self.schedule_autosave()

    def save_document(self) -> None:
        default_path = self.current_path or (self.examples_dir / "desktop_notebook.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save Notebook", str(default_path), "JSON Files (*.json)")
        if not path:
            return
        print(f"[debug][notebook-tab] save_document path={path!r}", flush=True)
        self.current_path = Path(path)
        document = self.to_document()
        document.metadata["title"] = self.current_path.stem.replace("_", " ").strip() or "Desktop Notebook"
        self.storage.save(self.current_path, document)
        if self.current_path.parent == self.examples_dir:
            self._reload_examples_list(select_title=document.metadata["title"])
        self.status_label.setText(f"Saved {self.current_path.name}")
        self.status_label.setStyleSheet("color: #15803d; font-weight: 600;")

    def save_example_document(self) -> None:
        title_seed = self.current_path.stem if self.current_path is not None else "desktop_notebook"
        default_path = self.examples_dir / suggest_example_filename(title_seed)
        path, _ = QFileDialog.getSaveFileName(self, "Save Notebook Example", str(default_path), "JSON Files (*.json)")
        if not path:
            return
        print(f"[debug][notebook-tab] save_example_document path={path!r}", flush=True)
        self.current_path = Path(path)
        document = self.to_document()
        document.metadata["title"] = self.current_path.stem.replace("_", " ").strip() or "Desktop Notebook"
        document.metadata.setdefault("category", "custom")
        document.metadata.setdefault("description", "Saved from the desktop notebook.")
        document.metadata.setdefault("tags", ["saved", "desktop"])
        self.storage.save(self.current_path, document)
        self._reload_examples_list(select_title=document.metadata["title"])
        self.status_label.setText(f"Saved example {self.current_path.name}")
        self.status_label.setStyleSheet("color: #15803d; font-weight: 600;")

    def open_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Notebook", "", "JSON Files (*.json)")
        if not path:
            return
        print(f"[debug][notebook-tab] open_document path={path!r}", flush=True)
        self.current_path = Path(path)
        self.load_document(self.storage.load(self.current_path))
        self.status_label.setText(f"Opened {self.current_path.name}")
        self.status_label.setStyleSheet("color: #15803d; font-weight: 600;")

    def schedule_autosave(self) -> None:
        self.autosave_timer.start()

    def autosave_now(self, path: str | Path) -> None:
        if self._is_running:
            self.autosave_timer.start()
            return
        target = Path(path)
        self._done_timer.stop()
        print(f"[debug][notebook-tab] autosave_now path={str(target)!r}", flush=True)
        self.storage.save(target, self.to_document())
        self.status_label.setText(f"Autosaved {target.name}")
        self.status_label.setStyleSheet("color: #64748b; font-weight: 600;")
