from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyside_app.notebook_plot_panel import (
    _format_marker_label,
    _marker_label_positions,
    build_notebook_plot_figure,
)


arrays = {
    "x": np.array([0.0, 100.0, 200.0, 300.0, 400.0]),
    "phi": np.array([0.0038, 0.0040, 0.0, -0.0038, -0.0040]),
}
figure = build_notebook_plot_figure(
    arrays,
    x_var="x",
    y_vars=["phi"],
    plot_type="lines+markers",
    title="",
    x_title="index",
    y_title="Phi",
    style_options={"graph_width": 700, "graph_height": 700},
)

sample_x = [0.0, 100.0, 400.0]
sample_y = [0.0038, 0.0040, -0.0040]
labels = [_format_marker_label(x, y) for x, y in zip(sample_x, sample_y)]
positions = _marker_label_positions(sample_x, sample_y)

print(
    "[verify][graph-marker-labels] "
    f"labels={labels!r} "
    f"positions={positions!r} "
    f"layout_width={figure.layout.width} "
    f"layout_height={figure.layout.height} "
    f"margin_r={figure.layout.margin.r}",
    flush=True,
)
