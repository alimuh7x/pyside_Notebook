from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pyside_app.execution_engine import ExecutionEngine
from pyside_app.notebook_plot_panel import NotebookPlotPanel, build_notebook_evolution_figure


def main() -> None:
    app = QApplication.instance() or QApplication([])
    namespace = {
        "time": np.array([0.0, 0.5, 1.0, 1.5]),
        "x": np.linspace(0.0, 1.0, 5),
        "u_history": np.array(
            [
                [0.0, 0.2, 0.5, 0.2, 0.0],
                [0.0, 0.4, 0.8, 0.4, 0.0],
                [0.0, 0.6, 1.0, 0.6, 0.0],
                [0.0, 0.3, 0.7, 0.3, 0.0],
            ]
        ),
    }
    print(f"[verify][notebook-evolution-animation] input_keys={sorted(namespace.keys())}", flush=True)
    panel = NotebookPlotPanel()
    panel.set_namespace(namespace)
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("evolution"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("u_history"))
    panel.evolution_time_combo.setCurrentIndex(panel.evolution_time_combo.findData("time"))
    panel.evolution_value_combo.setCurrentIndex(panel.evolution_value_combo.findData("x"))
    panel.evolution_animate_check.setChecked(True)
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    static_figure = build_notebook_evolution_figure(
        {"time": namespace["time"], "x": namespace["x"]},
        {"u_history": namespace["u_history"]},
        "u_history",
        "time",
        "x",
        "lines",
        "",
        "",
        "",
    )
    dense_rows = 120
    dense_history = np.arange(dense_rows * 2.0).reshape(dense_rows, 2)
    panel.set_namespace(
        {
            "time": np.linspace(0.0, 12.0, dense_rows),
            "x": np.array([0.0, 1.0]),
            "u_history": dense_history,
        }
    )
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("evolution"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("u_history"))
    panel.evolution_time_combo.setCurrentIndex(panel.evolution_time_combo.findData("time"))
    panel.evolution_value_combo.setCurrentIndex(panel.evolution_value_combo.findData("x"))
    panel.evolution_animate_check.setChecked(True)
    panel.refresh_controller_plot()
    dense_figure = panel.current_controller_figure()
    print(f"[verify][notebook-evolution-animation] frame_count={len(figure.frames)}", flush=True)
    print(f"[verify][notebook-evolution-animation] static_trace_count={len(static_figure.data)}", flush=True)
    print(f"[verify][notebook-evolution-animation] dense_frame_count={len(dense_figure.frames)}", flush=True)
    print(f"[verify][notebook-evolution-animation] first_y={list(figure.data[0].y)}", flush=True)
    print(f"[verify][notebook-evolution-animation] last_y={list(figure.frames[-1].data[0].y)}", flush=True)
    print(f"[verify][notebook-evolution-animation] button_controls={len(figure.layout.updatemenus)}", flush=True)
    print(f"[verify][notebook-evolution-animation] slider_controls={len(figure.layout.sliders)}", flush=True)
    print(f"[verify][notebook-evolution-animation] button_x={figure.layout.updatemenus[0].x}", flush=True)
    print(f"[verify][notebook-evolution-animation] slider_pad_t={figure.layout.sliders[0].pad.t}", flush=True)
    print(f"[verify][notebook-evolution-animation] margin_b={figure.layout.margin.b}", flush=True)
    print(f"[verify][notebook-evolution-animation] status={panel.controller_status.text()!r}", flush=True)
    slider_args = figure.layout.sliders[0].steps[0].args[1]
    print(f"[verify][notebook-evolution-animation] redraw={slider_args['frame']['redraw']}", flush=True)
    if len(figure.frames) != len(namespace["time"]):
        raise SystemExit("[verify][notebook-evolution-animation] FAIL: frame count mismatch")
    play_args = figure.layout.updatemenus[0].buttons[0].args[1]
    if len(figure.layout.updatemenus) != 1 or len(figure.layout.sliders) != 1:
        raise SystemExit("[verify][notebook-evolution-animation] FAIL: animation should expose play and slider controls")
    if slider_args["frame"]["redraw"]:
        raise SystemExit("[verify][notebook-evolution-animation] FAIL: slider animation should avoid full redraws")
    if play_args["frame"]["redraw"] or play_args["transition"]["duration"] != 0:
        raise SystemExit("[verify][notebook-evolution-animation] FAIL: play should use the same lightweight update path as slider")
    if figure.layout.updatemenus[0].x < 0.7 or figure.layout.margin.b < 160:
        raise SystemExit("[verify][notebook-evolution-animation] FAIL: animation controls need more separation from time slider")
    if len(static_figure.data) != 4:
        raise SystemExit("[verify][notebook-evolution-animation] FAIL: static plot should use available rows up to five")
    if len(dense_figure.frames) < 100:
        raise SystemExit("[verify][notebook-evolution-animation] FAIL: dense animation should keep at least 100 frames")

    engine_source = """
import numpy as np
Nx = 24
Nt = 2000
dt = 0.01
x = np.linspace(0.0, 1.0, Nx)
u = np.sin(np.pi * x)
for n in range(Nt):
    u = np.sin(np.pi * x) * np.cos(n * dt)
"""
    engine = ExecutionEngine()
    engine_result = engine.execute(engine_source)
    engine_namespace = engine.get_namespace()
    if engine_result.error:
        raise SystemExit(f"[verify][notebook-evolution-animation] FAIL: engine error {engine_result.error}")
    saved_shape = engine_namespace["u_history"].shape
    saved_time_shape = engine_namespace["time_history"].shape
    print(f"[verify][notebook-evolution-animation] engine_u_history_shape={saved_shape}", flush=True)
    print(f"[verify][notebook-evolution-animation] engine_time_history_shape={saved_time_shape}", flush=True)
    if saved_shape != (100, 24):
        raise SystemExit("[verify][notebook-evolution-animation] FAIL: auto history should save 100 frames for Nt=2000")
    print("[verify][notebook-evolution-animation] PASS", flush=True)
    app.processEvents()


if __name__ == "__main__":
    main()
