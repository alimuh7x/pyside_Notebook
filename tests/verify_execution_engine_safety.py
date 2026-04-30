from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

from PySide6.QtWidgets import QApplication

from pyside_app.execution_engine import ExecutionEngine
from pyside_app.plot_view import PlotView


def main() -> None:
    app = QApplication.instance() or QApplication([])

    engine = ExecutionEngine()
    blocked_open = engine.execute("open('forbidden.txt', 'w')")
    blocked_import = engine.execute("import os\nos.getcwd()")
    engine.execute("alpha = 7")
    snapshot = engine.get_namespace()
    snapshot["alpha"] = -1
    latest = engine.get_namespace()

    print(f"[verify][execution-safety] open_blocked={blocked_open.error is not None}", flush=True)
    print(f"[verify][execution-safety] import_blocked={blocked_import.error is not None}", flush=True)
    print(f"[verify][execution-safety] alpha_snapshot={snapshot['alpha']}", flush=True)
    print(f"[verify][execution-safety] alpha_live={latest['alpha']}", flush=True)

    view = PlotView()
    view.set_html("<div>cleanup check</div>")
    html_path = view._html_path
    print(f"[verify][execution-safety] temp_exists_before_close={html_path.exists()}", flush=True)
    view.close()
    app.processEvents()
    print(f"[verify][execution-safety] temp_exists_after_close={html_path.exists()}", flush=True)


if __name__ == "__main__":
    main()
