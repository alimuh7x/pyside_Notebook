from __future__ import annotations

from pathlib import Path
import uuid

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyside_app.cell_widgets import NotebookCellWidget, NotebookCodeEditor
from pyside_app.editor_intelligence import NotebookLspClient, build_completion_words
from pyside_app.notebook_examples import (
    get_desktop_notebook_examples,
    get_notebook_examples_dir,
    suggest_example_filename,
)
from pyside_app.functions_reference import build_functions_reference_html
from pyside_app.execution_engine import ExecutionEngine, ExecutionResult
from pyside_app.execution_worker import ExecutionWorker
from pyside_app.graph_state import NotebookGraphState
from pyside_app.notebook_components import (
    ExamplesBar,
    GraphPanelWidget,
    HelpPanelsWidget,
    NotebookColumnsWidget,
    NotebookToolbar,
    SidebarWidget,
    summarize_namespace_value,
)
from pyside_app.notebook_controllers import (
    DocumentController,
    ExecutionController,
    GraphController,
    NotebookDocumentPort,
    NotebookExecutionPort,
    NotebookUiPort,
)
from pyside_app.notebook_model import NotebookState
from pyside_app.storage import NotebookDocument, NotebookStorage

_NB_TEXT_SS = "color:#d7dae0; font-weight:700; font-size:14px;"
_NB_MUTED_SS = "color:#5c6370; font-weight:700; font-size:14px;"
_NB_HEADING_SS = "color:#e5c07b; font-weight:700; font-size:14px;"
_NB_STATUS_READY_SS = "color:#98c379; font-weight:700; font-size:14px;"
_NB_STATUS_MUTED_SS = "color:#5c6370; font-weight:700; font-size:14px;"
_NB_STATUS_INFO_SS = "color:#61afef; font-weight:700; font-size:14px;"
_NB_STATUS_ERROR_SS = "color:#e06c75; font-weight:700; font-size:14px;"
_NB_GRAPH_COMBO_SS = (
    "QComboBox {"
    " background:#2c313a;"
    " border:1px solid #3e4451;"
    " border-radius:8px;"
    " padding:5px 8px;"
    " font-size:14px;"
    " font-weight:700;"
    " color:#d7dae0;"
    " min-height:32px;"
    "} "
    "QComboBox QAbstractItemView {"
    " background:#2c313a; color:#d7dae0;"
    " selection-background-color:#3e4451;"
    " selection-color:#61afef;"
    " outline:0;"
    "} "
    "QComboBox QAbstractItemView::item:hover {"
    " background:#3e4451;"
    " color:#61afef;"
    "}"
)


def _values_equal(a: object, b: object) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def parameter_change_label(overrides: dict[str, object], previous: dict[str, object]) -> str:
    """Return a compact label containing only parameters changed from previous."""
    changed = {
        key: value
        for key, value in overrides.items()
        if key not in previous or not _values_equal(previous[key], value)
    }
    if not changed:
        print("[debug][parameter-label] no_changed_values label='current'", flush=True)
        return "current"
    label = "  ".join(
        f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in changed.items()
    )
    print(f"[debug][parameter-label] changed={list(changed)} label={label!r}", flush=True)
    return label


class _NotebookUiAdapter(NotebookUiPort):
    def __init__(self, tab: NotebookTab) -> None:
        self.tab = tab

    def set_status(self, text: str, style: str) -> None:
        print(f"[debug][notebook-ui-adapter] set_status text={text!r}", flush=True)
        self.tab.status_label.setText(text)
        self.tab.status_label.setStyleSheet(style)

    def refresh_completion_words(self) -> None:
        print("[debug][notebook-ui-adapter] refresh_completion_words", flush=True)
        self.tab._refresh_completion_words()

    def refresh_variables_panel(self) -> None:
        print("[debug][notebook-ui-adapter] refresh_variables_panel", flush=True)
        self.tab._refresh_variables_panel()

    def schedule_autosave(self) -> None:
        print("[debug][notebook-ui-adapter] schedule_autosave", flush=True)
        self.tab.schedule_autosave()

    def cell_count_label(self) -> str:
        print("[debug][notebook-ui-adapter] cell_count_label", flush=True)
        return self.tab._cell_count_label()


class _NotebookDocumentAdapter(NotebookDocumentPort):
    def __init__(self, tab: NotebookTab) -> None:
        self.tab = tab

    def create_cell(self, cell_type: str, source: str, column: str) -> object:
        print(f"[debug][notebook-document-adapter] create_cell cell_type={cell_type!r}", flush=True)
        return self.tab._create_cell(cell_type, source, column)

    def insert_cell(self, cell: object) -> None:
        print(f"[debug][notebook-document-adapter] insert_cell cell_id={getattr(cell, 'cell_id', None)!r}", flush=True)
        self.tab._insert_cell(cell)

    def delete_cell(self, cell: object) -> None:
        print(f"[debug][notebook-document-adapter] delete_cell cell_id={getattr(cell, 'cell_id', None)!r}", flush=True)
        self.tab.delete_cell(cell)

    def rebuild_columns(self) -> None:
        print("[debug][notebook-document-adapter] rebuild_columns", flush=True)
        self.tab._rebuild_columns()

    def apply_layout_state(self, left_column_width_pct: int) -> None:
        print(f"[debug][notebook-document-adapter] apply_layout_state pct={left_column_width_pct}", flush=True)
        self.tab._apply_layout_state(left_column_width_pct)

    def reload_examples_list(self, select_title: str | None = None) -> None:
        print(f"[debug][notebook-document-adapter] reload_examples_list title={select_title!r}", flush=True)
        self.tab._reload_examples_list(select_title)


class _NotebookExecutionAdapter(NotebookExecutionPort):
    def __init__(self, tab: NotebookTab) -> None:
        self.tab = tab

    def cell_source(self, cell: object) -> str:
        print(f"[debug][notebook-execution-adapter] cell_source cell_id={getattr(cell, 'cell_id', None)!r}", flush=True)
        return cell.source()

    def apply_cell_result(self, cell: object, result: ExecutionResult) -> None:
        print(f"[debug][notebook-execution-adapter] apply_cell_result cell_id={getattr(cell, 'cell_id', None)!r}", flush=True)
        cell.set_result(result)

    def clear_cell_runtime(self, cell: object) -> None:
        print(f"[debug][notebook-execution-adapter] clear_cell_runtime cell_id={getattr(cell, 'cell_id', None)!r}", flush=True)
        self.tab._clear_cell_runtime(cell)


class NotebookTab(QWidget):
    """Own the notebook authoring workflow, execution flow, and notebook-side graphs."""

    def __init__(
        self,
        parent: QWidget | None = None,
        lsp_client: NotebookLspClient | None = None,
        graph_state: NotebookGraphState | None = None,
    ) -> None:
        """Build the notebook workspace, toolbar, sidebar, and execution services."""
        super().__init__(parent)
        print("[debug][notebook-tab] init:start", flush=True)
        self.execution_engine                         = ExecutionEngine()
        self.lsp_client                               = lsp_client or NotebookLspClient(self)
        self.graph_state                              = graph_state or NotebookGraphState(self)
        self.storage                                  = NotebookStorage()
        self.thread_pool                              = QThreadPool.globalInstance()
        self.state                                    = NotebookState()
        self.autosave_path                            = Path.home() / ".calculation_notebook_autosave.json"
        self.autosave_timer                           = QTimer(self)
        self.autosave_timer.setInterval(10000)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self._autosave_timeout)
        self._done_timer                              = QTimer(self)
        self._done_timer.setInterval(1200)
        self._done_timer.setSingleShot(True)
        self._done_timer.timeout.connect(self._set_ready_status)
        self._namespace_skip                          = set(self.execution_engine._base_namespace.keys()) | {"__builtins__"}
        self.examples_dir                             = get_notebook_examples_dir()
        self.examples                                 = get_desktop_notebook_examples()
        self.ui_adapter                               = _NotebookUiAdapter(self)
        self.document_adapter                         = _NotebookDocumentAdapter(self)
        self.execution_adapter                        = _NotebookExecutionAdapter(self)
        self.lsp_client.completions_ready.connect(self._handle_lsp_completions)
        self.lsp_client.diagnostics_ready.connect(self._handle_lsp_diagnostics)
        self.lsp_client.availability_changed.connect(self._handle_lsp_availability)
        self.lsp_client.start()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.workspace_content = QWidget(self)
        self.workspace_content.setStyleSheet("background:#21252b;")
        root.addWidget(self.workspace_content)

        workspace_layout = QVBoxLayout(self.workspace_content)
        workspace_layout.setContentsMargins(12, 12, 12, 12)
        workspace_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self.workspace_content)
        workspace_layout.addWidget(self.main_splitter)

        self.sidebar = self._build_sidebar()
        self.main_splitter.addWidget(self.sidebar)

        content             = QWidget(self)
        content_layout      = QVBoxLayout(content)
        self.toolbar_widget = self._build_toolbar()
        content_layout.addWidget(self.toolbar_widget)

        self.help_panels_container = HelpPanelsWidget(
            self._functions_reference_html(),
            self._markdown_help_html(),
            content,
        )

        self.functions_panel     = self.help_panels_container.functions_panel
        self.markdown_help_panel = self.help_panels_container.markdown_help_panel

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

        self.graph_controller                         = GraphController(self.state, self.graph_state, self.graph_panel_widget)
        self.execution_controller                     = self._build_execution_controller()
        self.document_controller                      = self._build_document_controller()

        self.add_code_cell("left")
        self._refresh_completion_words()
        print("[debug][notebook-tab] init:done", flush=True)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Shut down notebook-side background services before closing the tab."""
        print("[debug][notebook-tab] close_event:start", flush=True)
        shutdown = getattr(self.lsp_client, "shutdown", None)
        if callable(shutdown):
            shutdown()
        super().closeEvent(event)
        print("[debug][notebook-tab] close_event:done", flush=True)

    @property
    def cells(self) -> list[NotebookCellWidget]:
        return self.state.cells

    @property
    def current_path(self) -> Path | None:
        return self.state.current_path

    @current_path.setter
    def current_path(self, value: Path | None) -> None:
        self.state.current_path = value

    @property
    def run_all_button(self):
        return self.toolbar_widget.button("run_all_button")

    @property
    def restart_button(self):
        return self.toolbar_widget.button("restart_button")

    @property
    def autosave_button(self):
        return self.toolbar_widget.button("autosave_button")

    @property
    def save_button(self):
        return self.toolbar_widget.button("save_button")

    @property
    def save_example_button(self):
        return self.toolbar_widget.button("save_example_button")

    @property
    def open_button(self):
        return self.toolbar_widget.button("open_button")

    @property
    def functions_toggle_button(self):
        return self.toolbar_widget.button("functions_toggle_button")

    @property
    def markdown_toggle_button(self):
        return self.toolbar_widget.button("markdown_toggle_button")

    def _autosave_timeout(self) -> None:
        print("[debug][notebook-tab] autosave_timeout", flush=True)
        self.autosave_now(self.autosave_path)

    def _handle_toolbar_autosave(self) -> None:
        print("[debug][notebook-tab] handle_toolbar_autosave", flush=True)
        self.autosave_now(self.autosave_path)

    def _build_execution_controller(self) -> ExecutionController:
        print("[debug][notebook-tab] build_execution_controller", flush=True)
        return ExecutionController(
            state                    = self.state,
            execution_engine=self.execution_engine,
            thread_pool=self.thread_pool,
            done_timer=self._done_timer,
            port                     = self.execution_adapter,
            ui                       = self.ui_adapter,
            graph_controller         = self.graph_controller,
            ready_style              = _NB_STATUS_READY_SS,
            muted_style              = _NB_STATUS_MUTED_SS,
            info_style               = _NB_STATUS_INFO_SS,
            error_style              = _NB_STATUS_ERROR_SS,
        )

    def _build_document_controller(self) -> DocumentController:
        print("[debug][notebook-tab] build_document_controller", flush=True)
        return DocumentController(
            parent                   = self,
            state                    = self.state,
            storage                  = self.storage,
            examples_dir             = self.examples_dir,
            ui                       = self.ui_adapter,
            port                     = self.document_adapter,
            to_document              = self.to_document,
            suggest_example_filename = suggest_example_filename,
            muted_style              = _NB_STATUS_MUTED_SS,
            ready_style              = _NB_STATUS_READY_SS,
        )

    def _set_status(self, text: str, style: str) -> None:
        print(f"[debug][notebook-tab] set_status text={text!r}", flush=True)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)

    def _clear_cell_runtime(self, cell: NotebookCellWidget) -> None:
        print(f"[debug][notebook-tab] clear_cell_runtime cell_id={cell.cell_id!r}", flush=True)
        cell.clear_output()
        if isinstance(cell.editor, NotebookCodeEditor):
            cell.editor.apply_diagnostics([])
            cell.editor.set_lsp_completions([])

    def _cell_count_label(self) -> str:
        """Return a compact status label describing the current cell count."""
        count = len(self.cells)
        label = f"{count} cell{'s' if count != 1 else ''}"
        print(f"[debug][notebook-tab] cell_count_label label={label!r}", flush=True)
        return label

    def _code_cells(self) -> list[NotebookCellWidget]:
        """Return only the code cells from the notebook cell list."""
        code_cells = [cell for cell in self.cells if cell.cell_type == "code"]
        print(f"[debug][notebook-tab] code_cells count={len(code_cells)}", flush=True)
        return code_cells

    def _document_uri_for_cell(self, cell_id: str) -> str:
        """Build a stable pseudo-file URI for LSP-backed code editors."""
        uri = f"file:///desktop-notebook/{cell_id}.py"
        print(f"[debug][notebook-tab] document_uri_for_cell cell_id={cell_id!r} uri={uri!r}", flush=True)
        return uri

    def _completion_words(self) -> list[str]:
        """Build editor completion words from the live execution namespace."""
        namespace = self.execution_engine.get_namespace()
        print(f"[debug][notebook-tab] completion_words_namespace count={len(namespace)}", flush=True)
        words = build_completion_words(namespace)
        print(f"[debug][notebook-tab] completion_words count={len(words)}", flush=True)
        return words

    def _refresh_completion_words(self) -> None:
        """Push the latest completion word list into each code editor."""
        words = self._completion_words()
        for cell in self._code_cells():
            if isinstance(cell.editor, NotebookCodeEditor):
                print(f"[debug][notebook-tab] refresh_completion_words cell_id={cell.cell_id!r}", flush=True)
                cell.editor.set_completion_words(words)

    def _handle_lsp_availability(self, available: bool) -> None:
        """Log LSP availability changes for notebook editors."""
        print(f"[debug][notebook-tab] lsp_availability available={available}", flush=True)

    def _handle_lsp_completions(self, uri: str, labels: list[str]) -> None:
        """Route asynchronous LSP completion results to the matching editor."""
        print(f"[debug][notebook-tab] handle_lsp_completions uri={uri!r} count={len(labels)}", flush=True)
        for cell in self._code_cells():
            if isinstance(cell.editor, NotebookCodeEditor) and cell.editor.document_uri == uri:
                cell.editor.set_lsp_completions(labels)
                return

    def _handle_lsp_diagnostics(self, uri: str, diagnostics: list[dict[str, object]]) -> None:
        """Route diagnostics from the LSP client to the matching code editor."""
        print(f"[debug][notebook-tab] handle_lsp_diagnostics uri={uri!r} count={len(diagnostics)}", flush=True)
        for cell in self._code_cells():
            if isinstance(cell.editor, NotebookCodeEditor) and cell.editor.document_uri == uri:
                cell.editor.apply_diagnostics(diagnostics)
                return

    def _build_sidebar(self) -> QWidget:
        """Build the sidebar with examples row and variables panel."""
        print("[debug][notebook-tab] build_sidebar", flush=True)
        self.sidebar_widget = SidebarWidget(
            text_style=_NB_TEXT_SS,
            heading_style=_NB_HEADING_SS,
            namespace_skip=self._namespace_skip,
            combo_style=_NB_GRAPH_COMBO_SS,
            summarize_value=self._summarize_namespace_value,
            parent=self,
        )
        self.variables_panel   = self.sidebar_widget.variables_panel
        self.variables_browser = self.sidebar_widget.variables_browser
        self.examples_bar      = self.sidebar_widget.examples_bar
        self.examples_bar.set_examples(self.examples)
        self.examples_bar.insert_example_requested.connect(self.insert_example)
        self.example_combo = self.examples_bar.example_combo
        return self.sidebar_widget

    def _reload_examples_list(self, select_title: str | None = None) -> None:
        """Reload example metadata from disk and optionally reselect one example."""
        print(f"[debug][notebook-tab] reload_examples_list select_title={select_title!r}", flush=True)
        self.examples = get_desktop_notebook_examples()
        if hasattr(self, "examples_bar"):
            self.examples_bar.set_examples(self.examples, select_title)

    def _functions_reference_html(self) -> str:
        """Return the HTML used in the notebook functions help panel."""
        print("[debug][notebook-tab] functions_reference delegate", flush=True)
        return build_functions_reference_html()

    def _markdown_help_html(self) -> str:
        """Return the HTML used in the notebook markdown help panel."""
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
        """Show or hide the functions reference browser."""
        print("[debug][notebook-tab] toggle_functions_panel delegate", flush=True)
        self.help_panels_container.toggle_functions_panel()

    def toggle_markdown_help(self) -> None:
        """Show or hide the markdown help browser."""
        print("[debug][notebook-tab] toggle_markdown_help delegate", flush=True)
        self.help_panels_container.toggle_markdown_help()

    def _build_column(self, column: str, title: str) -> tuple[QScrollArea, QVBoxLayout]:
        """Build the main notebook cell column container and its action header."""
        print(f"[debug][notebook-tab] build_column column={column!r} title={title!r}", flush=True)
        self.columns_widget = NotebookColumnsWidget(
            heading_style=_NB_HEADING_SS,
            parent=self,
        )
        self.columns_widget.add_code_requested.connect(self.add_code_cell)
        self.columns_widget.add_markdown_requested.connect(self.add_markdown_cell)
        self.columns_widget.insert_example_requested.connect(self.insert_example)
        self.left_container = self.columns_widget.container

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background:#282c34; border:none; }")
        scroll.setWidget(self.columns_widget)
        return scroll, self.columns_widget.cells_layout

    def _build_variables_panel(self, parent: QWidget | None = None) -> QWidget:
        """Build the sidebar panel that displays live namespace variables."""
        print("[debug][notebook-tab] build_variables_panel delegate", flush=True)
        return self.sidebar_widget.variables_panel

    def _build_graphs_panel(self) -> QWidget:
        """Build the right column: param panel on top, graph scroll below."""
        print("[debug][notebook-tab] build_graphs_panel:graph_panel_widget", flush=True)
        self.graph_panel_widget = GraphPanelWidget(self)
        self.quick_preview_panel = self.graph_panel_widget.quick_preview_panel

        graph_scroll = QScrollArea(self)
        graph_scroll.setWidgetResizable(True)
        graph_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        graph_scroll.setStyleSheet("QScrollArea { background:#21252b; border:none; }")
        graph_scroll.setWidget(self.graph_panel_widget)

        # Wrapper: param section (top) + graph scroll (bottom)
        wrapper = QWidget(self)
        wrapper.setStyleSheet("background:#21252b;")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)

        self._param_section = QWidget(wrapper)
        self._param_section.setStyleSheet(
            "QWidget { background:#2c313a; border-bottom:1px solid #3e4451; }"
            "QLabel { background:transparent; border:none; }"
        )
        ps_layout = QVBoxLayout(self._param_section)
        ps_layout.setContentsMargins(10, 8, 10, 8)
        ps_layout.setSpacing(4)
        ps_hdr = QLabel("Parameters", self._param_section)
        ps_hdr.setStyleSheet("font-weight:700; font-size:14px; color:#e5c07b; background:transparent;")
        ps_layout.addWidget(ps_hdr)
        self._param_body_layout = ps_layout
        self._param_current_widget = None
        self._param_section.hide()
        wrapper_layout.addWidget(self._param_section)
        wrapper_layout.addWidget(graph_scroll, 1)

        return wrapper

    def set_param_widget(self, widget) -> None:
        """Keep the legacy graph-column parameter slot hidden."""
        print("[debug][notebook-tab] set_param_widget legacy_hidden", flush=True)
        if self._param_current_widget is not None:
            self._param_body_layout.removeWidget(self._param_current_widget)
            self._param_current_widget.setParent(None)
        self._param_section.hide()
        self._param_current_widget = None

    def _summarize_namespace_value(self, value: object) -> tuple[str, str]:
        """Convert one namespace value into a compact sidebar summary and type name."""
        return summarize_namespace_value(value)

    def _refresh_variables_panel(self) -> None:
        """Rebuild the variables sidebar from the current execution namespace."""
        print("[debug][notebook-tab] refresh_variables_panel", flush=True)
        namespace = self.execution_engine.get_namespace()
        self.sidebar_widget._refresh_variables_panel(namespace)

    def _rebuild_columns(self) -> None:
        """Re-render notebook cells into the visible column layout."""
        print(f"[debug][notebook-tab] rebuild_columns count={len(self.cells)}", flush=True)
        for cell in self.cells:
            cell.set_column("left")
            cell.move_left_btn.hide()
            cell.move_right_btn.hide()
        self.columns_widget.rebuild_cells(self.cells)
        self._refresh_variables_panel()
        self._refresh_graphs_panel()

    def _refresh_graphs_panel(self) -> None:
        """Push namespace and latest-plot state into graph consumers."""
        self.graph_controller.refresh_graphs_panel(self.execution_engine.get_namespace())

    def _latest_plot_output(self) -> tuple[str, str]:
        """Find the latest graph-like output produced by any notebook cell."""
        return self.graph_controller.latest_plot_output()

    def _graph_title_for_cell(self, cell: NotebookCellWidget) -> str:
        """Build a readable graph title from a cell's source code."""
        return self.graph_controller._graph_title_for_cell(cell)

    def _apply_layout_state(self, left_column_width_pct: int) -> None:
        """Restore the notebook/graph splitter ratio from saved layout state."""
        total = max(1, self.columns_splitter.width() or 1000)
        left = max(320, int(total * left_column_width_pct / 100.0))
        right = max(320, total - left)
        print(
            f"[debug][notebook-tab] apply_layout_state pct={left_column_width_pct} total={total} left={left} right={right}",
            flush=True,
        )
        self.columns_splitter.setSizes([left, right])

    def _build_toolbar(self) -> NotebookToolbar:
        """Build the notebook toolbar with left controls, right controls, and status."""
        print("[debug][notebook-tab] build_toolbar start", flush=True)
        toolbar = NotebookToolbar(status_style=_NB_STATUS_READY_SS, parent=self)
        toolbar.run_all_requested.      connect(self.run_all_cells)
        toolbar.restart_requested.      connect(self.restart_kernel)
        toolbar.autosave_requested.     connect(self._handle_toolbar_autosave)
        toolbar.save_requested.         connect(self.save_document)
        toolbar.save_example_requested. connect(self.save_example_document)
        toolbar.open_requested.         connect(self.open_document)
        toolbar.functions_requested.    connect(self.toggle_functions_panel)
        toolbar.markdown_help_requested.connect(self.toggle_markdown_help)

        self.status_label = toolbar.status_label
        print("[debug][notebook-tab] build_toolbar done", flush=True)
        return toolbar

    def _set_ready_status(self) -> None:
        """Restore the ready status appearance after transient status messages."""
        print("[debug][notebook-tab] status:ready", flush=True)
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet(_NB_STATUS_READY_SS)

    def _insert_cell(self, cell: NotebookCellWidget) -> None:
        """Insert one cell into notebook state and connect autosave tracking."""
        self.cells.append(cell)
        self._rebuild_columns()
        cell.editor.textChanged.connect(self.schedule_autosave)
        if self.state.is_loading_document:
            return
        self.status_label.setText(self._cell_count_label())
        self.status_label.setStyleSheet(_NB_STATUS_MUTED_SS)

    def _create_cell(self, cell_type: str, source: str = "", column: str = "left") -> NotebookCellWidget:
        """Create a configured notebook cell widget for code or markdown."""
        temp_cell_id = f"cell-{uuid.uuid4().hex[:8]}"
        document_uri = self._document_uri_for_cell(temp_cell_id) if cell_type == "code" else ""
        words = self._completion_words() if cell_type == "code" else None
        print(
            f"[debug][notebook-tab] create_cell cell_type={cell_type!r} column={column!r} uri={document_uri!r}",
            flush=True,
        )
        rerun_fn, on_run_success = self._make_rerun_fn() if cell_type == "code" else (None, None)
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
            rerun_fn=rerun_fn,
        )
        if on_run_success is not None:
            cell._on_run_success = on_run_success
        if cell_type == "code" and isinstance(cell.editor, NotebookCodeEditor):
            cell.editor.document_uri = self._document_uri_for_cell(cell.cell_id)
            cell.editor.attach_lsp_client(self.lsp_client, cell.editor.document_uri)
        return cell

    def _make_rerun_fn(self):
        """Build a rerun callback and initial-run hook for parameter exploration."""
        from pyside_app.execution_worker import OverrideWorker
        from pyside_app.execution_engine import detect_cell_parameters
        engine = self.execution_engine
        thread_pool = self.thread_pool
        graph_panel = self.graph_panel_widget
        tab = self

        _active_workers: list = []
        _prev_overrides: dict = {}

        def _ensure_panel(source: str) -> None:
            params = detect_cell_parameters(source)
            print(f"[debug][notebook-tab] parameter_controls:ensure count={len(params)}", flush=True)
            if params:
                tab.sidebar_widget.set_parameter_controls(params, source, rerun_fn)
                tab._refresh_variables_panel()
            else:
                tab.sidebar_widget.clear_parameter_controls()

        def on_run_success(source: str, result) -> None:
            """Called after the main Run Cell completes to create the slider panel."""
            _ensure_panel(source)

        def rerun_fn(source: str, overrides: dict, on_result) -> None:
            nonlocal _prev_overrides
            worker = OverrideWorker(engine, source, overrides)
            _active_workers.append(worker)
            _captured_overrides = dict(overrides)
            _captured_source = source

            if not _prev_overrides:
                _prev_overrides = {k: spec["value"] for k, spec in detect_cell_parameters(source).items()}
            _captured_prev = dict(_prev_overrides)
            _prev_overrides = dict(overrides)

            def _on_result_and_graph(result) -> None:
                on_result(result)
                if result.namespace_snapshot and not result.error:
                    tab._refresh_variables_panel()
                    label = parameter_change_label(_captured_overrides, _captured_prev)
                    from pyside_app import array_store
                    array_store.store_run(result.namespace_snapshot, label)
                    graph_panel.add_run(result.namespace_snapshot, label)
                    _ensure_panel(_captured_source)

                for output in result.outputs:
                    if output.kind == "plotly":
                        graph_panel.set_latest_plot("", output.data.get("html", ""))
                        break

            worker.signals.finished.connect(_on_result_and_graph)
            worker.signals.finished.connect(lambda _r, w=worker: _active_workers.remove(w) if w in _active_workers else None)
            thread_pool.start(worker)

        return rerun_fn, on_run_success

    def add_code_cell(self, column: str = "left", source: str = "") -> None:
        """Append a new code cell to the notebook."""
        print(f"[debug][notebook-tab] add_code_cell column={column!r}", flush=True)
        self._insert_cell(self._create_cell("code", source, column))

    def add_markdown_cell(self, column: str = "left", source: str = "") -> None:
        """Append a new markdown cell to the notebook."""
        print(f"[debug][notebook-tab] add_markdown_cell column={column!r}", flush=True)
        self._insert_cell(self._create_cell("markdown", source, column))

    def insert_example(self, column: str = "left") -> None:
        """Insert all cells from the currently selected example notebook."""
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
        """Remove a cell from the notebook and refresh dependent UI."""
        print(f"[debug][notebook-tab] delete_cell cell_id={cell.cell_id!r}", flush=True)
        if cell in self.cells:
            self.cells.remove(cell)
        cell.hide()
        cell.deleteLater()
        self._rebuild_columns()
        self.schedule_autosave()
        self.status_label.setText(self._cell_count_label())
        self.status_label.setStyleSheet(_NB_STATUS_MUTED_SS)

    def move_cell(self, cell: NotebookCellWidget, direction: int) -> None:
        """Move a cell forward or backward in the notebook order."""
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
        """Handle column-move requests for notebook cells."""
        if cell not in self.cells or column != "left":
            return

    def run_cell(self, cell: NotebookCellWidget) -> None:
        """Execute one code cell asynchronously through an execution worker."""
        self.execution_controller.run_cell(cell)

    def _handle_worker_finished(self, worker: ExecutionWorker, cell: NotebookCellWidget, result: ExecutionResult) -> None:
        """Finalize one background execution worker and process its result."""
        self.execution_controller._handle_worker_finished(worker, cell, result)

    def _handle_cell_result(self, cell: NotebookCellWidget, result: ExecutionResult) -> None:
        """Apply one execution result and refresh all namespace-driven views."""
        self.execution_controller._handle_cell_result(cell, result)

    def run_all_cells(self) -> None:
        """Queue every code cell and execute them from top to bottom."""
        self.execution_controller.run_all_cells()

    def restart_kernel(self) -> None:
        """Reset the execution namespace and clear cell outputs."""
        self.execution_controller.restart_kernel()

    def to_document(self) -> NotebookDocument:
        """Serialize the current notebook UI state into a saveable document."""
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
        """Return the current notebook-column width as a percentage."""
        sizes = self.columns_splitter.sizes()
        if len(sizes) < 2:
            return 50
        total = max(1, sum(sizes))
        pct = int(round((sizes[0] / total) * 100))
        print(f"[debug][notebook-tab] current_left_column_pct pct={pct}", flush=True)
        return pct

    def load_document(self, document: NotebookDocument) -> None:
        """Rebuild the notebook UI from a previously saved document."""
        self.document_controller.load_document(document)

    def save_document(self) -> None:
        """Save the current notebook to a user-selected JSON file."""
        self.document_controller.save_document()

    def save_example_document(self) -> None:
        """Save the current notebook as a reusable example entry."""
        self.document_controller.save_example_document(suggest_example_filename)

    def open_document(self) -> None:
        """Open a notebook JSON file and load it into the current tab."""
        document = self.document_controller.open_document()
        if document is None:
            return
        current_path = self.current_path
        self.load_document(document)
        if current_path is not None:
            self._set_status(f"Opened {current_path.name}", _NB_STATUS_READY_SS)

    def schedule_autosave(self) -> None:
        """Restart the autosave timer after an editing change."""
        self.autosave_timer.start()

    def autosave_now(self, path: str | Path) -> None:
        """Persist an autosave snapshot immediately when execution is idle."""
        if self.state.execution.is_running:
            self.autosave_timer.start()
            return
        self.document_controller.autosave_now(path, self._done_timer)
