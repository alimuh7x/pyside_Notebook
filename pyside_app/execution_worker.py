from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from pyside_app.execution_engine import ExecutionEngine, ExecutionResult


class ExecutionWorkerSignals(QObject):
    """Signal bundle used by background execution workers."""

    finished = Signal(object)


class ExecutionWorker(QRunnable):
    """Run notebook code execution on a background Qt thread-pool worker."""

    def __init__(self, engine: ExecutionEngine, source: str) -> None:
        """Capture the execution engine and source code for one run request."""
        super().__init__()
        self.engine = engine
        self.source = source
        self.signals = ExecutionWorkerSignals()

    @Slot()
    def run(self) -> None:
        """Execute the source code and emit the resulting execution payload."""
        print(f"[debug][execution-worker] run:start source={self.source!r}", flush=True)
        result: ExecutionResult = self.engine.execute(self.source)
        self.signals.finished.emit(result)
        print("[debug][execution-worker] run:done", flush=True)
