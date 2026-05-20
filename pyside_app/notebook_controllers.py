from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QWidget

from pyside_app.execution_engine import ExecutionOutput, ExecutionResult
from pyside_app.execution_worker import ExecutionWorker
from pyside_app.graph_state import NotebookGraphState
from pyside_app.notebook_model import NotebookState
from pyside_app.storage import NotebookDocument, NotebookStorage


class NotebookUiPort(Protocol):
    def set_status(self, text: str, style: str) -> None: ...
    def refresh_completion_words(self) -> None: ...
    def refresh_variables_panel(self) -> None: ...
    def schedule_autosave(self) -> None: ...
    def cell_count_label(self) -> str: ...


class NotebookDocumentPort(Protocol):
    def create_cell(self, cell_type: str, source: str, column: str) -> object: ...
    def insert_cell(self, cell: object) -> None: ...
    def delete_cell(self, cell: object) -> None: ...
    def rebuild_columns(self) -> None: ...
    def apply_layout_state(self, left_column_width_pct: int) -> None: ...
    def reload_examples_list(self, select_title: str | None = None) -> None: ...


class NotebookExecutionPort(Protocol):
    def cell_source(self, cell: object) -> str: ...
    def apply_cell_result(self, cell: object, result: ExecutionResult) -> None: ...
    def clear_cell_runtime(self, cell: object) -> None: ...


class GraphController:
    """Own notebook graph refresh logic."""

    def __init__(self, state: NotebookState, graph_state: NotebookGraphState, graph_panel: object) -> None:
        self.state = state
        self.graph_state = graph_state
        self.graph_panel = graph_panel

    def refresh_graphs_panel(self, namespace: dict[str, object]) -> None:
        print("[debug][graph-controller] refresh_graphs_panel", flush=True)
        print(f"[debug][graph-controller] refresh_graphs_panel_namespace count={len(namespace)}", flush=True)
        self.graph_state.set_namespace(namespace)
        self.graph_panel.set_namespace(namespace)
        title, html = self.latest_plot_output()
        if html:
            self.graph_state.set_latest_plot(title, html)
        else:
            self.graph_state.clear_latest_plot()
        self.graph_panel.set_latest_plot(title, html)

    def latest_plot_output(self) -> tuple[str, str]:
        print("[debug][graph-controller] latest_plot_output:start", flush=True)
        latest_title = ""
        latest_html = ""
        for cell in self.state.cells:
            result = getattr(cell, "last_result", None)
            if result is None:
                continue
            for output in result.outputs:
                if output.kind == "plotly":
                    latest_title = self._graph_title_for_cell(cell)
                    latest_html = output.data.get("html", "")
                    print(
                        f"[debug][graph-controller] latest_plot_output:plotly title={latest_title!r} html_length={len(latest_html)}",
                        flush=True,
                    )
                elif output.kind == "html" and "data:image" in output.data.get("html", ""):
                    latest_title = self._graph_title_for_cell(cell)
                    latest_html = output.data.get("html", "")
                    print(
                        f"[debug][graph-controller] latest_plot_output:image title={latest_title!r} html_length={len(latest_html)}",
                        flush=True,
                    )
        print(
            f"[debug][graph-controller] latest_plot_output:done title={latest_title!r} html_length={len(latest_html)}",
            flush=True,
        )
        return latest_title, latest_html

    def _graph_title_for_cell(self, cell: object) -> str:
        source_lines = [line.strip() for line in cell.source().splitlines() if line.strip()]
        title = source_lines[0] if source_lines else getattr(cell, "cell_id", "plot")
        if len(title) > 48:
            title = title[:45] + "..."
        print(f"[debug][graph-controller] graph_title title={title!r}", flush=True)
        return title


class ExecutionController:
    """Coordinate notebook execution flow without owning widgets."""

    def __init__(
        self,
        state: NotebookState,
        execution_engine: object,
        thread_pool: object,
        done_timer: QTimer,
        port: NotebookExecutionPort,
        ui: NotebookUiPort,
        graph_controller: GraphController,
        ready_style: str,
        muted_style: str,
        info_style: str,
        error_style: str,
    ) -> None:
        self.state = state
        self.execution_engine = execution_engine
        self.thread_pool = thread_pool
        self.done_timer = done_timer
        self.port = port
        self.ui = ui
        self.graph_controller = graph_controller
        self.ready_style = ready_style
        self.muted_style = muted_style
        self.info_style = info_style
        self.error_style = error_style

    def run_cell(self, cell: object) -> None:
        print(f"[debug][execution-controller] run_cell cell_id={getattr(cell, 'cell_id', None)!r}", flush=True)
        if self.state.execution.is_running and not self.state.execution.run_all_queue:
            self.ui.set_status("Execution already running", self.error_style)
            return
        self.ui.set_status("Running cell...", self.info_style)
        self.state.execution.is_running = True
        worker = ExecutionWorker(self.execution_engine, self.port.cell_source(cell))
        self.state.execution.active_workers.add(worker)
        worker.signals.finished.connect(
            lambda result, target=cell, current_worker=worker: self._handle_worker_finished(current_worker, target, result)
        )
        self.thread_pool.start(worker)

    def _handle_worker_finished(self, worker: ExecutionWorker, cell: object, result: ExecutionResult) -> None:
        print(f"[debug][execution-controller] handle_worker_finished cell_id={getattr(cell, 'cell_id', None)!r}", flush=True)
        self.state.execution.active_workers.discard(worker)
        self._handle_cell_result(cell, result)

    def _handle_cell_result(self, cell: object, result: ExecutionResult) -> None:
        print(f"[debug][execution-controller] handle_cell_result cell_id={getattr(cell, 'cell_id', None)!r}", flush=True)
        self.port.apply_cell_result(cell, result)
        self.ui.refresh_completion_words()
        self.ui.refresh_variables_panel()
        namespace = self.execution_engine.get_namespace()
        self.graph_controller.refresh_graphs_panel(namespace)
        if not result.error:
            # Label the baseline with actual param values rather than "baseline"
            from pyside_app.execution_engine import detect_cell_parameters
            from pyside_app import array_store
            source = cell.source() if callable(getattr(cell, "source", None)) else ""
            params = detect_cell_parameters(source)
            if params:
                store_label = "current"
                self.graph_controller.graph_panel.set_current_label(store_label)
            else:
                store_label = "baseline"
            array_store.store_run(namespace, label=store_label)
            print(f"[debug][execution-controller] arrays saved to {array_store.get_store_dir()}", flush=True)
        self.state.execution.is_running = False
        if result.error:
            self.state.execution.run_all_queue.clear()
            self.ui.set_status("Execution error", self.error_style)
        elif self.state.execution.run_all_queue:
            next_cell = self.state.execution.run_all_queue.pop(0)
            self.run_cell(next_cell)
            return
        else:
            self.ui.set_status("Done", self.ready_style)
            self.done_timer.start()
        self.ui.schedule_autosave()

    def run_all_cells(self) -> None:
        print(f"[debug][execution-controller] run_all_cells count={len(self.state.cells)}", flush=True)
        code_cells = [cell for cell in self.state.cells if getattr(cell, "cell_type", None) == "code"]
        if not code_cells:
            self.ui.set_status("No code cells", self.muted_style)
            return
        self.state.execution.run_all_queue = code_cells[1:]
        self.run_cell(code_cells[0])

    def restart_kernel(self) -> None:
        print("[debug][execution-controller] restart_kernel", flush=True)
        self.done_timer.stop()
        self.state.execution.run_all_queue.clear()
        self.state.execution.is_running = False
        restarted_now = self.execution_engine.restart()
        for cell in self.state.cells:
            self.port.clear_cell_runtime(cell)
        self.ui.refresh_completion_words()
        self.ui.refresh_variables_panel()
        self.graph_controller.refresh_graphs_panel(self.execution_engine.get_namespace())
        # Clear graph history and temp array files
        self.graph_controller.graph_panel.clear_history()
        from pyside_app import array_store
        array_store.clear()
        self.ui.set_status("Kernel restarted" if restarted_now else "Kernel restart pending", self.ready_style if restarted_now else self.info_style)
        self.ui.schedule_autosave()


class DocumentController:
    """Handle notebook persistence and document restoration."""

    def __init__(
        self,
        parent: QWidget,
        state: NotebookState,
        storage: NotebookStorage,
        examples_dir: Path,
        ui: NotebookUiPort,
        port: NotebookDocumentPort,
        to_document: Callable[[], NotebookDocument],
        suggest_example_filename: Callable[[str], str],
        ready_style: str,
        muted_style: str,
    ) -> None:
        self.parent = parent
        self.state = state
        self.storage = storage
        self.examples_dir = examples_dir
        self.ui = ui
        self.port = port
        self.to_document = to_document
        self.suggest_example_filename = suggest_example_filename
        self.ready_style = ready_style
        self.muted_style = muted_style

    def load_document(self, document: NotebookDocument) -> None:
        print(f"[debug][document-controller] load_document count={len(document.cells)}", flush=True)
        self.state.is_loading_document = True
        for cell in list(self.state.cells):
            self.port.delete_cell(cell)
        if not document.cells:
            self.state.is_loading_document = False
            self.port.insert_cell(self.port.create_cell("code", "", "left"))
            return
        try:
            for payload in document.cells:
                cell = self.port.create_cell(
                    payload.get("type", "code"),
                    payload.get("source", ""),
                    payload.get("column", "left"),
                )
                setattr(cell, "panel_width", payload.get("panel_width"))
                self.port.insert_cell(cell)
                if payload.get("outputs"):
                    restored = ExecutionResult(
                        outputs=[
                            ExecutionOutput(kind=output.get("kind", "value"), data=dict(output.get("data") or {}))
                            for output in payload.get("outputs", [])
                        ],
                        error=None,
                    )
                    cell.set_result(restored)
        finally:
            self.state.is_loading_document = False
        self.port.rebuild_columns()
        self.ui.refresh_completion_words()
        self.ui.set_status(self.ui.cell_count_label(), self.muted_style)
        QTimer.singleShot(0, lambda: self.port.apply_layout_state(int(document.layout.get("left_column_width_pct", 50))))
        self.ui.schedule_autosave()

    def save_document(self) -> None:
        default_path = self.state.current_path or (self.examples_dir / "desktop_notebook.json")
        path, _ = QFileDialog.getSaveFileName(self.parent, "Save Notebook", str(default_path), "JSON Files (*.json)")
        if not path:
            return
        print(f"[debug][document-controller] save_document path={path!r}", flush=True)
        self.state.current_path = Path(path)
        document = self.to_document()
        assert self.state.current_path is not None
        document.metadata["title"] = self.state.current_path.stem.replace("_", " ").strip() or "Desktop Notebook"
        self.storage.save(self.state.current_path, document)
        if self.state.current_path.parent == self.examples_dir:
            self.port.reload_examples_list(document.metadata["title"])
        self.ui.set_status(f"Saved {self.state.current_path.name}", self.ready_style)

    def save_example_document(self) -> None:
        title_seed = self.state.current_path.stem if self.state.current_path is not None else "desktop_notebook"
        default_path = self.examples_dir / self.suggest_example_filename(title_seed)
        path, _ = QFileDialog.getSaveFileName(self.parent, "Save Notebook Example", str(default_path), "JSON Files (*.json)")
        if not path:
            return
        print(f"[debug][document-controller] save_example_document path={path!r}", flush=True)
        self.state.current_path = Path(path)
        document = self.to_document()
        assert self.state.current_path is not None
        document.metadata["title"] = self.state.current_path.stem.replace("_", " ").strip() or "Desktop Notebook"
        document.metadata.setdefault("category", "custom")
        document.metadata.setdefault("description", "Saved from the desktop notebook.")
        document.metadata.setdefault("tags", ["saved", "desktop"])
        self.storage.save(self.state.current_path, document)
        self.port.reload_examples_list(document.metadata["title"])
        self.ui.set_status(f"Saved example {self.state.current_path.name}", self.ready_style)

    def open_document(self) -> NotebookDocument | None:
        path, _ = QFileDialog.getOpenFileName(self.parent, "Open Notebook", "", "JSON Files (*.json)")
        if not path:
            return None
        print(f"[debug][document-controller] open_document path={path!r}", flush=True)
        self.state.current_path = Path(path)
        return self.storage.load(self.state.current_path)

    def autosave_now(self, path: str | Path, done_timer: QTimer) -> None:
        if self.state.execution.is_running:
            return
        target = Path(path)
        done_timer.stop()
        print(f"[debug][document-controller] autosave_now path={str(target)!r}", flush=True)
        self.storage.save(target, self.to_document())
        self.ui.set_status(f"Autosaved {target.name}", self.muted_style)
