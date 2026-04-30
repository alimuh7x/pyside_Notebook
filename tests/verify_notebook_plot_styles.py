from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyside_app.notebook_plot_panel import build_notebook_plot_figure


arrays = {
    "x": np.array([0.0, 1.0, 2.0]),
    "phi": np.array([1.0, 3.0, 5.0]),
    "phi_exact": np.array([1.0, 2.0, 4.0]),
}

figure = build_notebook_plot_figure(
    arrays,
    x_var="x",
    y_vars=["phi", "phi_exact"],
    plot_type="lines",
    title="Styled Plot",
    x_title="x",
    y_title="value",
    series_styles={
        "phi": {"plot_type": "lines", "line_style": "dash"},
        "phi_exact": {"plot_type": "markers", "line_style": "solid"},
    },
)

for trace in figure.data:
    dash = getattr(getattr(trace, "line", None), "dash", None)
    mode = getattr(trace, "mode", None)
    print(f"[verify][plot-styles] trace={trace.name!r} mode={mode!r} dash={dash!r}", flush=True)
