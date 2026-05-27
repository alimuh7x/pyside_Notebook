from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyside_app.notebook_plot_panel import NotebookPlotPanel


app = QApplication.instance() or QApplication([])

panel = NotebookPlotPanel()
namespace = {
    "x": np.linspace(-2.0, 2.0, 81),
    "phi": np.linspace(-2.0, 2.0, 81) ** 3 - np.linspace(-2.0, 2.0, 81),
}
print(f"[verify][notebook-graph-post] input_keys={sorted(namespace.keys())}")
panel.set_namespace(namespace)
panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
panel.y_combo.set_checked_values(["phi"])
analysis = panel.main_card._analysis
analysis.vertical_line_check.setChecked(True)
analysis.vertical_line_value.setValue(0.0)
analysis.horizontal_line_check.setChecked(True)
analysis.horizontal_line_value.setValue(0.0)
analysis.first_deriv_check.setChecked(True)
analysis.second_deriv_check.setChecked(True)
analysis.mark_first_zero_check.setChecked(True)
analysis.mark_second_zero_check.setChecked(True)
panel.refresh_controller_plot()

figure = panel.current_controller_figure()
trace_names = [trace.name for trace in figure.data]
shape_count = len(figure.layout.shapes or [])
print(f"[verify][notebook-graph-post] trace_names={trace_names}")
print(f"[verify][notebook-graph-post] shape_count={shape_count}")
print(f"[verify][notebook-graph-post] yaxis2_overlaying={figure.layout.yaxis2.overlaying!r}")
print(f"[verify][notebook-graph-post] yaxis3_overlaying={figure.layout.yaxis3.overlaying!r}")
print(f"[verify][notebook-graph-post] summary={analysis.summary_text()!r}")
print(f"[verify][notebook-graph-post] trace_count={len(figure.data)}")

if shape_count < 2:
    raise SystemExit("[verify][notebook-graph-post] FAIL: expected reference lines")
if figure.layout.yaxis2.overlaying != "y" or figure.layout.yaxis3.overlaying != "y":
    raise SystemExit("[verify][notebook-graph-post] FAIL: expected derivative right axes")
if not any("dy/dx = 0" in (name or "") for name in trace_names):
    raise SystemExit("[verify][notebook-graph-post] FAIL: expected first derivative zero markers")
if not any("d2y/dx2 = 0" in (name or "") for name in trace_names):
    raise SystemExit("[verify][notebook-graph-post] FAIL: expected second derivative zero markers")

print("[verify][notebook-graph-post] PASS")
