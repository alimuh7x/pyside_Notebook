from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from pyside_app.execution_engine import ExecutionEngine, ExecutionResult


class ExecutionWorkerSignals(QObject):
    finished = Signal(object)


class ExecutionWorker(QRunnable):
    def __init__(self, engine: ExecutionEngine, source: str) -> None:
        super().__init__()
        self.engine = engine
        self.source = source
        self.signals = ExecutionWorkerSignals()

    @Slot()
    def run(self) -> None:
        print(f"[debug][execution-worker] run:start source={self.source!r}", flush=True)
        result: ExecutionResult = self.engine.execute(self.source)
        self.signals.finished.emit(result)
        print("[debug][execution-worker] run:done", flush=True)
