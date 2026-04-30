from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyside_app.notebook_plot_panel import NotebookPlotPanel


def main() -> None:
    app = QApplication.instance() or QApplication([])
    panel = NotebookPlotPanel()
    namespace = {
        "time": np.array([0.0, 1.0, 2.0]),
        "x": np.array([10.0, 20.0, 30.0, 40.0]),
        "history": np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        ),
    }
    print(f"[verify][notebook-plot-evolution] namespace_keys={sorted(namespace.keys())}", flush=True)
    panel.set_namespace(namespace)
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("evolution"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("history"))
    panel.evolution_time_combo.setCurrentIndex(panel.evolution_time_combo.findData("time"))
    panel.evolution_value_combo.setCurrentIndex(panel.evolution_value_combo.findData("x"))
    panel.evolution_step_slider.setValue(1)
    figure = panel.current_controller_figure()
    trace = figure.data[0]
    print(f"[verify][notebook-plot-evolution] mode={panel.mode_combo.currentData()!r}", flush=True)
    print(f"[verify][notebook-plot-evolution] slider={panel.evolution_step_slider.value()} max={panel.evolution_step_slider.maximum()}", flush=True)
    print(f"[verify][notebook-plot-evolution] title={figure.layout.title.text!r}", flush=True)
    print(f"[verify][notebook-plot-evolution] x={list(trace.x)}", flush=True)
    print(f"[verify][notebook-plot-evolution] y={list(trace.y)}", flush=True)
    print(f"[verify][notebook-plot-evolution] status={panel.controller_status.text()!r}", flush=True)
    app.processEvents()


if __name__ == "__main__":
    main()
