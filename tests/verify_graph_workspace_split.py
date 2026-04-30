from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_DISABLE_SANDBOX', '1')

from PySide6.QtWidgets import QApplication

from pyside_app.graph_state import NotebookGraphState
from pyside_app.graphs_tab import GraphsTab
from pyside_app.notebook_tab import NotebookTab

app = QApplication.instance() or QApplication([])
state = NotebookGraphState()
notebook = NotebookTab(graph_state=state)
graphs = GraphsTab(state)

result = notebook.execution_engine.execute('import numpy as np\nimport plotly.graph_objects as go\nx = np.arange(5)\nphi = x**2\nfig = go.Figure()\nfig.add_scatter(x=x, y=phi)\nfig.update_layout(title=r"$\\phi$ preview", xaxis_title=r"$x$", yaxis_title=r"$\\phi$")\nfig')
notebook.cells[0].set_result(result)
notebook._refresh_graphs_panel()

print(f"[verify][graph-workspace] latest_title={state.latest_plot_title!r}")
print(f"[verify][graph-workspace] latest_html_length={len(state.latest_plot_html)}")
print(f"[verify][graph-workspace] namespace_count={len(state.namespace)}")
print(f"[verify][graph-workspace] quick_mode={notebook.quick_preview_panel.mode_combo.currentData()!r}")
print(f"[verify][graph-workspace] quick_x_index={notebook.quick_preview_panel.x_combo.currentData()!r}")
print(f"[verify][graph-workspace] quick_y={notebook.quick_preview_panel.y_combo.checked_values()!r}")
print(f"[verify][graph-workspace] quick_trace_count={len(notebook.quick_preview_panel.current_figure().data)}")
print(f"[verify][graph-workspace] preview_hidden={notebook.quick_preview_panel.plot_view.isHidden()}")
print(f"[verify][graph-workspace] graph_tab_height={graphs.plot_panel.controller_plot.minimumHeight()}")
print(f"[verify][graph-workspace] mathjax_in_preview={'tex-svg.js' in notebook.quick_preview_panel.plot_view._html_path.read_text(encoding='utf-8')}")
print(f"[verify][graph-workspace] outputs_hidden={graphs.plot_panel.outputs_scroll.isHidden()}")
