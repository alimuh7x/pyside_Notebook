from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyside_app.execution_engine import ExecutionOutput, ExecutionResult
from pyside_app.notebook_plot_panel import NotebookPlotPanel


class _FakeCell:
    def __init__(self, cell_id: str, source: str, result: ExecutionResult | None) -> None:
        self.cell_id = cell_id
        self._source = source
        self.last_result = result

    def source(self) -> str:
        return self._source


app = QApplication.instance() or QApplication([])

panel = NotebookPlotPanel()
namespace = {
    "x": np.linspace(0.0, 10.0, 6),
    "temperature": np.array([300.0, 350.0, 420.0, 510.0, 620.0, 760.0]),
    "stress": np.array([5.0, 12.0, 19.0, 27.0, 38.0, 50.0]),
}
print(f"[verify][notebook-plot-panel] namespace_keys={sorted(namespace.keys())}")
panel.set_namespace(namespace)
panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
panel.y_combo.set_checked_values(["temperature"])
panel.title_edit.setText("Notebook Array Plot")
panel.x_label_edit.setText("time")
panel.y_label_edit.setText("value")
panel.refresh_controller_plot()
figure = panel.current_controller_figure()
print(f"[verify][notebook-plot-panel] controller_traces={len(figure.data)}")
print(f"[verify][notebook-plot-panel] controller_names={[trace.name for trace in figure.data]}")

panel.y_combo.set_checked_values(["temperature", "x"])
panel.refresh_controller_plot()
multi_figure = panel.current_controller_figure()
print(f"[verify][notebook-plot-panel] multi_controller_traces={len(multi_figure.data)}")
print(f"[verify][notebook-plot-panel] multi_controller_names={[trace.name for trace in multi_figure.data]}")

panel.sync_cell_outputs(
    [
        _FakeCell(
            "cell-plot",
            "go.Figure(data=[...])",
            ExecutionResult(
                outputs=[
                    ExecutionOutput(
                        kind="plotly",
                        data={"html": "<div>plot-output</div>", "text": "Plotly Figure"},
                    )
                ]
            ),
        )
    ]
)
print(f"[verify][notebook-plot-panel] output_titles={panel.output_titles()}")
print(f"[verify][notebook-plot-panel] output_count={panel.output_count()}")
