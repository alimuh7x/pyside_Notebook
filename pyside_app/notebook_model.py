from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyside_app.execution_worker import ExecutionWorker


@dataclass
class NotebookExecutionState:
    """Mutable execution state for one notebook tab."""

    is_running: bool = False
    run_all_queue: list[object] = field(default_factory=list)
    active_workers: set[ExecutionWorker] = field(default_factory=set)


@dataclass
class NotebookState:
    """Central notebook state shared by controllers."""

    cells: list[object] = field(default_factory=list)
    current_path: Path | None = None
    is_loading_document: bool = False
    execution: NotebookExecutionState = field(default_factory=NotebookExecutionState)
