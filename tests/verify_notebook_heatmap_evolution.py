from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyside_app.notebook_plot_panel import NotebookPlotPanel
from pyside_app.notebook_plot_panel import build_notebook_heatmap_figure
from pyside_app.notebook_plot_panel import extract_notebook_array_variables_with_3d


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def main() -> None:
    namespace = {
        "time_history": np.linspace(0.0, 0.4, 5),
        "x": np.linspace(0.0, 1.0, 4),
        "y": np.linspace(0.0, 2.0, 3),
        "field": np.arange(12.0).reshape(3, 4),
        "u_history": np.arange(60.0).reshape(5, 3, 4),
    }
    arrays_1d, arrays_2d, arrays_3d = extract_notebook_array_variables_with_3d(namespace)
    print(f"[verify][notebook-heatmap] arrays_1d={sorted(arrays_1d)}", flush=True)
    print(f"[verify][notebook-heatmap] arrays_2d={sorted(arrays_2d)}", flush=True)
    print(f"[verify][notebook-heatmap] arrays_3d={sorted(arrays_3d)}", flush=True)

    static_figure = build_notebook_heatmap_figure(
        arrays_1d, arrays_2d, arrays_3d,
        "field", None, "x", "y", False,
        "field", "x", "y",
    )
    animated_figure = build_notebook_heatmap_figure(
        arrays_1d, arrays_2d, arrays_3d,
        "u_history", "time_history", "x", "y", True,
        "u history", "x", "y",
    )
    print(f"[verify][notebook-heatmap] static_trace={static_figure.data[0].type}", flush=True)
    print(f"[verify][notebook-heatmap] animated_frames={len(animated_figure.frames)}", flush=True)
    print(f"[verify][notebook-heatmap] slider_count={len(animated_figure.layout.sliders)}", flush=True)

    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(namespace)
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("heatmap"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("u_history"))
    panel.evolution_time_combo.setCurrentIndex(panel.evolution_time_combo.findData("time_history"))
    panel.evolution_value_combo.setCurrentIndex(panel.evolution_value_combo.findData("x"))
    panel.evolution_y_combo.setCurrentIndex(panel.evolution_y_combo.findData("y"))
    panel.evolution_animate_check.setChecked(True)
    panel.refresh_controller_plot()
    panel_figure = panel.current_controller_figure()
    print(f"[verify][notebook-heatmap] panel_mode={panel.mode_combo.currentData()!r}", flush=True)
    print(f"[verify][notebook-heatmap] panel_field={panel.evolution_matrix_combo.currentData()!r}", flush=True)
    print(f"[verify][notebook-heatmap] panel_frames={len(panel_figure.frames)}", flush=True)

    if static_figure.data[0].type != "heatmap":
        raise SystemExit("[verify][notebook-heatmap] FAIL: static field is not a heatmap")
    if len(animated_figure.frames) != namespace["u_history"].shape[0]:
        raise SystemExit("[verify][notebook-heatmap] FAIL: animated frame count mismatch")
    if len(animated_figure.layout.sliders) != 1 or len(animated_figure.layout.updatemenus) != 1:
        raise SystemExit("[verify][notebook-heatmap] FAIL: animation controls missing")
    if len(panel_figure.frames) != namespace["u_history"].shape[0]:
        raise SystemExit("[verify][notebook-heatmap] FAIL: panel did not build heatmap animation")
    print("[verify][notebook-heatmap] PASS", flush=True)


if __name__ == "__main__":
    main()
