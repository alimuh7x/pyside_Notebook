from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import plotly.graph_objects as go
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pyside_app.controls import AutoCloseComboBox, CheckableComboBox
from pyside_app.plot_view import PlotView


PLOT_TYPE_OPTIONS = (
    ("Lines", "lines"),
    ("Markers", "markers"),
    ("Lines + Markers", "lines+markers"),
    ("Bar", "bar"),
    ("Histogram", "histogram"),
)

LINE_STYLE_OPTIONS = (
    ("Solid", "solid"),
    ("Dash", "dash"),
    ("Dot", "dot"),
    ("Dash Dot", "dashdot"),
    ("Long Dash", "longdash"),
    ("Long Dash Dot", "longdashdot"),
)

MARKER_STYLE_OPTIONS = (
    ("Circle", "circle"),
    ("Square", "square"),
    ("Diamond", "diamond"),
    ("Cross", "cross"),
    ("X", "x"),
    ("Triangle Up", "triangle-up"),
    ("Triangle Down", "triangle-down"),
)

LINE_COLOR_OPTIONS = (
    ("Auto", "auto"),
    ("Blue", "#1f77b4"),
    ("Red", "#d62728"),
    ("Green", "#2ca02c"),
    ("Black", "#000000"),
    ("Orange", "#ff7f0e"),
    ("Purple", "#9467bd"),
)

FONT_SIZE_OPTIONS = (12, 14, 16, 18, 20, 24, 28, 32)
LINE_WIDTH_OPTIONS = (1, 2, 3, 4, 5, 6, 7,8, 9, 10)
MARKER_SIZE_OPTIONS = (4, 6, 8, 10, 12, 14, 16, 18, 20)
GRAPH_SIZE_OPTIONS = (
    ("Auto", None),
    ("Landscape", "landscape"),
    ("Square", "square"),
    ("Tall", "tall"),
    ("Wide", "wide"),
)


def _default_scientific_style_options() -> dict[str, Any]:
    options = {
        "font_size": 16,
        "line_width": 2,
        "marker_size": 7,
        "show_grid": True,
        "show_box": True,
        "ticks_inside": True,
        "show_minor_ticks": True,
        "graph_aspect": "auto",
        "graph_width": None,
        "graph_height": 520,
    }
    print(f"[debug][notebook-plot-panel] default_style_options options={options!r}", flush=True)
    return options


def _axis_style_options(style_options: dict[str, Any], axis_title: str) -> dict[str, Any]:
    font_size = int(style_options.get("font_size", 16))
    show_grid = bool(style_options.get("show_grid", True))
    show_box = bool(style_options.get("show_box", True))
    ticks_inside = bool(style_options.get("ticks_inside", True))
    show_minor_ticks = bool(style_options.get("show_minor_ticks", True))
    tick_direction = "inside" if ticks_inside else "outside"
    axis_options = {
        "title": {"text": axis_title, "font": {"size": font_size + 2, "color": "#0f1b2b"}},
        "tickfont": {"size": font_size, "color": "#334155"},
        "showgrid": show_grid,
        "gridcolor": "rgba(200,210,220,0.5)",
        "zeroline": False,
        "showline": show_box,
        "mirror": "allticks" if show_box else False,
        "linecolor": "#0f1b2b",
        "linewidth": 2.5,
        "ticks": tick_direction,
        "ticklen": 10,
        "tickwidth": 1.5,
        "tickcolor": "#0f1b2b",
        "minor": {
            "ticks": tick_direction if show_minor_ticks else "",
            "ticklen": 5,
            "tickcolor": "#475569",
            "showgrid": False,
        },
    }
    print(f"[debug][notebook-plot-panel] axis_style_options axis={axis_title!r} options={axis_options!r}", flush=True)
    return axis_options


def _apply_scientific_layout(
    fig: go.Figure,
    *,
    title: str | None,
    xaxis_title: str,
    yaxis_title: str,
    showlegend: bool,
    style_options: dict[str, Any] | None = None,
    barmode: str | None = None,
) -> None:
    resolved = dict(_default_scientific_style_options())
    if style_options:
        resolved.update(style_options)
    font_size = int(resolved.get("font_size", 16))
    graph_width = resolved.get("graph_width")
    graph_height = resolved.get("graph_height")
    print(
        f"[debug][notebook-plot-panel] apply_scientific_layout title={title!r} "
        f"xaxis_title={xaxis_title!r} yaxis_title={yaxis_title!r} "
        f"showlegend={showlegend} style_options={resolved!r} "
        f"graph_width={graph_width!r} graph_height={graph_height!r}",
        flush=True,
    )
    fig.update_layout(
        title={"text": title or None, "font": {"size": font_size + 4, "color": "#0f1b2b"}},
        showlegend=showlegend,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 60, "r": 20, "t": 35, "b": 55},
        font={"size": font_size, "color": "#0f1b2b"},
        legend={"orientation": "v", "x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top", "font": {"size": font_size}},
        barmode=barmode,
        width=graph_width,
        height=graph_height,
    )
    fig.update_xaxes(**_axis_style_options(resolved, xaxis_title))
    fig.update_yaxes(**_axis_style_options(resolved, yaxis_title))


def _coerce_numeric_series(value: Any) -> np.ndarray | None:
    print(f"[debug][notebook-plot-panel] coerce_numeric_series:type type={type(value).__name__!r}", flush=True)
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        print("[debug][notebook-plot-panel] coerce_numeric_series:skip unsupported_container", flush=True)
        return None
    print(f"[debug][notebook-plot-panel] coerce_numeric_series:shape shape={getattr(array, 'shape', None)!r}", flush=True)
    if array.ndim != 1:
        print("[debug][notebook-plot-panel] coerce_numeric_series:skip ndim", flush=True)
        return None
    if not np.issubdtype(array.dtype, np.number):
        print(f"[debug][notebook-plot-panel] coerce_numeric_series:skip dtype={array.dtype!r}", flush=True)
        return None
    result = np.asarray(array, dtype=float)
    print(f"[debug][notebook-plot-panel] coerce_numeric_series:done size={result.size}", flush=True)
    return result


def _coerce_numeric_matrix(value: Any) -> np.ndarray | None:
    print(f"[debug][notebook-plot-panel] coerce_numeric_matrix:type type={type(value).__name__!r}", flush=True)
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        print("[debug][notebook-plot-panel] coerce_numeric_matrix:skip unsupported_container", flush=True)
        return None
    print(f"[debug][notebook-plot-panel] coerce_numeric_matrix:shape shape={getattr(array, 'shape', None)!r}", flush=True)
    if array.ndim != 2:
        print("[debug][notebook-plot-panel] coerce_numeric_matrix:skip ndim", flush=True)
        return None
    if not np.issubdtype(array.dtype, np.number):
        print(f"[debug][notebook-plot-panel] coerce_numeric_matrix:skip dtype={array.dtype!r}", flush=True)
        return None
    result = np.asarray(array, dtype=float)
    print(f"[debug][notebook-plot-panel] coerce_numeric_matrix:done shape={result.shape!r}", flush=True)
    return result


def extract_notebook_array_variables(namespace: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    print(f"[debug][notebook-plot-panel] extract_arrays:start count={len(namespace)}", flush=True)
    arrays_1d: dict[str, np.ndarray] = {}
    arrays_2d: dict[str, np.ndarray] = {}
    for name, value in sorted(namespace.items()):
        print(f"[debug][notebook-plot-panel] extract_arrays:item name={name!r}", flush=True)
        if name.startswith("_"):
            print(f"[debug][notebook-plot-panel] extract_arrays:skip_private name={name!r}", flush=True)
            continue
        if callable(value):
            print(f"[debug][notebook-plot-panel] extract_arrays:skip_callable name={name!r}", flush=True)
            continue
        if isinstance(value, ModuleType):
            print(f"[debug][notebook-plot-panel] extract_arrays:skip_module name={name!r}", flush=True)
            continue
        coerced = _coerce_numeric_series(value)
        if coerced is not None:
            arrays_1d[name] = coerced
            print(
                f"[debug][notebook-plot-panel] extract_arrays:accepted_1d name={name!r} "
                f"shape={coerced.shape!r}",
                flush=True,
            )
            continue
        matrix = _coerce_numeric_matrix(value)
        if matrix is None:
            continue
        arrays_2d[name] = matrix
        print(
            f"[debug][notebook-plot-panel] extract_arrays:accepted_2d name={name!r} "
            f"shape={matrix.shape!r}",
            flush=True,
        )
    print(
        f"[debug][notebook-plot-panel] extract_arrays:done count_1d={len(arrays_1d)} count_2d={len(arrays_2d)}",
        flush=True,
    )
    return arrays_1d, arrays_2d


def build_notebook_plot_figure(
    arrays: dict[str, np.ndarray],
    x_var: str | None,
    y_vars: list[str],
    plot_type: str,
    title: str,
    x_title: str,
    y_title: str,
    series_styles: dict[str, dict[str, str]] | None = None,
    style_options: dict[str, Any] | None = None,
) -> go.Figure:
    print(
        f"[debug][notebook-plot-panel] build_figure:start x_var={x_var!r} "
        f"y_vars={y_vars!r} plot_type={plot_type!r}",
        flush=True,
    )
    fig = go.Figure()
    resolved_style_options = dict(_default_scientific_style_options())
    if style_options:
        resolved_style_options.update(style_options)
    line_width = int(resolved_style_options.get("line_width", 2))
    marker_size = int(resolved_style_options.get("marker_size", 7))
    font_size = int(resolved_style_options.get("font_size", 16))
    print(
        f"[debug][notebook-plot-panel] build_figure:resolved_style_options options={resolved_style_options!r}",
        flush=True,
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    x_data = arrays.get(x_var) if x_var else None
    print(
        f"[debug][notebook-plot-panel] build_figure:x_data "
        f"present={x_data is not None} len={len(x_data) if x_data is not None else 0}",
        flush=True,
    )
    if not y_vars:
        print("[debug][notebook-plot-panel] build_figure:empty_y_selection", flush=True)
        fig.add_annotation(
            text="Select Y variable(s) to build a notebook plot.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": font_size, "color": "#94a3b8"},
        )
    for index, y_name in enumerate(y_vars):
        print(f"[debug][notebook-plot-panel] build_figure:y_loop name={y_name!r}", flush=True)
        y_data = arrays.get(y_name)
        if y_data is None:
            print(f"[debug][notebook-plot-panel] build_figure:skip_missing_y name={y_name!r}", flush=True)
            continue
        style = dict((series_styles or {}).get(y_name) or {})
        trace_plot_type = style.get("plot_type") or plot_type or "lines"
        line_style = style.get("line_style") or "solid"
        marker_style = style.get("marker_style") or "circle"
        selected_line_color = style.get("line_color") or "#000000"
        print(
            f"[debug][notebook-plot-panel] build_figure:style name={y_name!r} "
            f"trace_plot_type={trace_plot_type!r} line_style={line_style!r} "
            f"marker_style={marker_style!r} selected_line_color={selected_line_color!r}",
            flush=True,
        )
        x_plot = x_data if x_data is not None and len(x_data) == len(y_data) else np.arange(len(y_data))
        print(
            f"[debug][notebook-plot-panel] build_figure:trace_lengths x_len={len(x_plot)} y_len={len(y_data)}",
            flush=True,
        )
        auto_color = colors[index % len(colors)]
        resolved_color = auto_color if selected_line_color == "auto" else str(selected_line_color)
        print(
            f"[debug][notebook-plot-panel] build_figure:resolved_color name={y_name!r} "
            f"auto_color={auto_color!r} resolved_color={resolved_color!r}",
            flush=True,
        )
        if trace_plot_type == "histogram":
            fig.add_trace(go.Histogram(x=y_data, name=y_name, marker_color=resolved_color, opacity=0.75))
        elif trace_plot_type == "bar":
            fig.add_trace(go.Bar(x=x_plot, y=y_data, name=y_name, marker_color=resolved_color))
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_data,
                    mode=trace_plot_type or "lines",
                    name=y_name,
                    line={"color": resolved_color, "width": line_width, "dash": line_style},
                    marker={"color": resolved_color, "size": marker_size, "symbol": marker_style},
                )
            )
        print(f"[debug][notebook-plot-panel] build_figure:trace_added name={y_name!r}", flush=True)
    final_x_title = x_title or (x_var or "index")
    final_y_title = y_title or (y_vars[0] if len(y_vars) == 1 else "value")
    _apply_scientific_layout(
        fig,
        title=title or None,
        xaxis_title=final_x_title,
        yaxis_title=final_y_title,
        showlegend=len(y_vars) > 1,
        style_options=resolved_style_options,
        barmode="group" if plot_type == "bar" else None,
    )
    print(f"[debug][notebook-plot-panel] build_figure:done traces={len(fig.data)}", flush=True)
    return fig


def build_notebook_evolution_figure(
    arrays_1d: dict[str, np.ndarray],
    arrays_2d: dict[str, np.ndarray],
    matrix_var: str | None,
    time_var: str | None,
    value_var: str | None,
    step_index: int,
    plot_type: str,
    title: str,
    x_title: str,
    y_title: str,
    style_options: dict[str, Any] | None = None,
) -> go.Figure:
    print(
        f"[debug][notebook-plot-panel] build_evolution:start matrix_var={matrix_var!r} "
        f"time_var={time_var!r} value_var={value_var!r} step_index={step_index} plot_type={plot_type!r}",
        flush=True,
    )
    fig = go.Figure()
    resolved_style_options = dict(_default_scientific_style_options())
    if style_options:
        resolved_style_options.update(style_options)
    line_width = int(resolved_style_options.get("line_width", 2))
    marker_size = int(resolved_style_options.get("marker_size", 7))
    font_size = int(resolved_style_options.get("font_size", 16))
    print(
        f"[debug][notebook-plot-panel] build_evolution:resolved_style_options options={resolved_style_options!r}",
        flush=True,
    )
    matrix = arrays_2d.get(matrix_var) if matrix_var else None
    if matrix is None:
        print("[debug][notebook-plot-panel] build_evolution:no_matrix", flush=True)
        fig.add_annotation(
            text="Select a 2D array to plot its evolution.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": font_size, "color": "#94a3b8"},
        )
        return fig
    rows, cols = matrix.shape
    clamped_step = min(max(0, step_index), max(rows - 1, 0))
    print(
        f"[debug][notebook-plot-panel] build_evolution:matrix_shape rows={rows} cols={cols} clamped_step={clamped_step}",
        flush=True,
    )
    time_axis = arrays_1d.get(time_var) if isinstance(time_var, str) and time_var else None
    if time_axis is None or len(time_axis) != rows:
        time_axis = np.arange(rows, dtype=float)
        print("[debug][notebook-plot-panel] build_evolution:time_axis fallback=index", flush=True)
    else:
        print(f"[debug][notebook-plot-panel] build_evolution:time_axis matched len={len(time_axis)}", flush=True)
    value_axis = arrays_1d.get(value_var) if isinstance(value_var, str) and value_var else None
    if value_axis is None or len(value_axis) != cols:
        value_axis = np.arange(cols, dtype=float)
        print("[debug][notebook-plot-panel] build_evolution:value_axis fallback=index", flush=True)
    else:
        print(f"[debug][notebook-plot-panel] build_evolution:value_axis matched len={len(value_axis)}", flush=True)
    y_data = matrix[clamped_step]
    time_value = time_axis[clamped_step] if len(time_axis) > clamped_step else clamped_step
    trace_name = f"{matrix_var or 'matrix'} @ step {clamped_step}"
    print(
        f"[debug][notebook-plot-panel] build_evolution:trace_data x_len={len(value_axis)} y_len={len(y_data)} time_value={time_value!r}",
        flush=True,
    )
    if plot_type == "bar":
        fig.add_trace(go.Bar(x=value_axis, y=y_data, name=trace_name, marker_color="#1f77b4"))
    else:
        fig.add_trace(
            go.Scatter(
                x=value_axis,
                y=y_data,
                mode=plot_type if plot_type not in {"histogram"} else "lines",
                name=trace_name,
                line={"color": "#1f77b4", "width": line_width},
                marker={"color": "#1f77b4", "size": marker_size},
            )
        )
    _apply_scientific_layout(
        fig,
        title=title or f"{matrix_var or '2D array'} evolution at t={time_value}",
        xaxis_title=x_title or (value_var or "value index"),
        yaxis_title=y_title or (matrix_var or "value"),
        showlegend=False,
        style_options=resolved_style_options,
    )
    print(f"[debug][notebook-plot-panel] build_evolution:done traces={len(fig.data)}", flush=True)
    return fig


class NotebookPlotPanel(QWidget):
    def __init__(self, parent: QWidget | None = None, layout_mode: str = "advanced") -> None:
        super().__init__(parent)
        print(f"[debug][notebook-plot-panel] init:start layout_mode={layout_mode!r}", flush=True)
        self.layout_mode = layout_mode
        self.setStyleSheet("background:#ffffff;")
        self._arrays_1d: dict[str, np.ndarray] = {}
        self._arrays_2d: dict[str, np.ndarray] = {}
        self._current_figure = build_notebook_plot_figure({}, None, [], "lines", "", "", "")
        self._output_widgets: dict[str, tuple[QLabel, QWidget, str]] = {}
        self._output_empty_label: QLabel | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Graphs", self)
        title.setStyleSheet("color:#001f41; font-weight:700; font-size:15px;")
        header.addWidget(title)
        header.addStretch(1)
        root.addLayout(header)

        self.controller_card = QWidget(self)
        self.controller_card.setObjectName("plotBuilderCard")
        self.controller_card.setStyleSheet(
            """
            QWidget#plotBuilderCard {
                background:#ffffff;
                border:1px solid #d1dce8;
            }
            QWidget#plotBuilderCard QLabel {
                background:transparent;
                border:none;
            }
            QWidget#plotBuilderCard QComboBox,
            QWidget#plotBuilderCard QLineEdit,
            QWidget#plotBuilderCard QListWidget {
                background:#ffffff;
                border:1px solid #d1dce8;
                border-radius:6px;
                padding:4px 6px;
            }
            QWidget#plotBuilderCard QListView {
                background:#ffffff;
                border:1px solid #d1dce8;
            }
            QWidget#plotBuilderCard QComboBox QAbstractItemView {
                selection-background-color:#c7def5;
                selection-color:#0f1b2b;
                outline:0;
            }
            QWidget#plotBuilderCard QComboBox QAbstractItemView::item:hover {
                background:#c7def5;
                color:#0f1b2b;
            }
            """
        )
        card_layout = QHBoxLayout(self.controller_card)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(12)

        self.settings_panel = QWidget(self.controller_card)
        self.settings_panel.setStyleSheet("background:#ffffff;")
        self.settings_panel.setMinimumWidth(420)
        self.settings_panel.setMaximumWidth(520)
        settings_layout = QVBoxLayout(self.settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)

        controller_label = QLabel("Notebook Plot Builder", self.settings_panel)
        controller_label.setStyleSheet("color:#001f41; font-weight:700; font-size:13px;")
        settings_layout.addWidget(controller_label)

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(8)
        mode_label = QLabel("Mode", self.controller_card)
        mode_label.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        self.mode_combo = AutoCloseComboBox(self.controller_card)
        self.mode_combo.addItem("Series (1D)", "series")
        self.mode_combo.addItem("Evolution (2D)", "evolution")
        graph_size_label = QLabel("Graph size", self.controller_card)
        graph_size_label.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        self.graph_size_combo = AutoCloseComboBox(self.controller_card)
        for label, value in GRAPH_SIZE_OPTIONS:
            self.graph_size_combo.addItem(label, value)
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.mode_combo, 1)
        mode_row.addWidget(graph_size_label)
        mode_row.addWidget(self.graph_size_combo, 1)
        controls_layout.addLayout(mode_row)

        self.series_controls = QWidget(self.controller_card)
        series_layout = QHBoxLayout(self.series_controls)
        series_layout.setContentsMargins(0, 0, 0, 0)
        series_layout.setSpacing(8)

        x_block = QWidget(self.series_controls)
        x_block_layout = QVBoxLayout(x_block)
        x_block_layout.setContentsMargins(0, 0, 0, 0)
        x_block_layout.setSpacing(4)
        self.x_combo = AutoCloseComboBox(self.controller_card)
        self.x_combo.addItem("Index", "")
        x_label = QLabel("X variable", x_block)
        x_label.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        x_block_layout.addWidget(x_label)
        x_block_layout.addWidget(self.x_combo)
        series_layout.addWidget(x_block, 1)

        plot_type_block = QWidget(self.series_controls)
        plot_type_block_layout = QVBoxLayout(plot_type_block)
        plot_type_block_layout.setContentsMargins(0, 0, 0, 0)
        plot_type_block_layout.setSpacing(4)
        self.plot_type_combo = AutoCloseComboBox(self.controller_card)
        for label, value in PLOT_TYPE_OPTIONS:
            self.plot_type_combo.addItem(label, value)
        plot_type_label = QLabel("Plot type", plot_type_block)
        plot_type_label.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        plot_type_block_layout.addWidget(plot_type_label)
        plot_type_block_layout.addWidget(self.plot_type_combo)
        series_layout.addWidget(plot_type_block, 1)

        y_block = QWidget(self.series_controls)
        y_block_layout = QVBoxLayout(y_block)
        y_block_layout.setContentsMargins(0, 0, 0, 0)
        y_block_layout.setSpacing(4)
        self.y_combo = CheckableComboBox(self.controller_card)
        y_label = QLabel("Y variable(s)", y_block)
        y_label.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        y_block_layout.addWidget(y_label)
        y_block_layout.addWidget(self.y_combo)
        series_layout.addWidget(y_block, 2)
        controls_layout.addWidget(self.series_controls)

        self.series_style_card = QWidget(self.controller_card)
        self.series_style_card.setStyleSheet("background:#f8fbfe; border:1px solid #dbe5ef;")
        self.series_style_layout = QVBoxLayout(self.series_style_card)
        self.series_style_layout.setContentsMargins(8, 8, 8, 8)
        self.series_style_layout.setSpacing(6)
        self.series_style_title = QLabel("Per-Series Styles", self.series_style_card)
        self.series_style_title.setStyleSheet("color:#001f41; font-weight:700; font-size:12px;")
        self.series_style_layout.addWidget(self.series_style_title)
        self.series_style_grid = QGridLayout()
        self.series_style_grid.setContentsMargins(0, 0, 0, 0)
        self.series_style_grid.setHorizontalSpacing(8)
        self.series_style_grid.setVerticalSpacing(6)
        self.series_style_layout.addLayout(self.series_style_grid)
        controls_layout.addWidget(self.series_style_card)
        self._series_style_widgets: dict[str, dict[str, Any]] = {}

        self.evolution_controls = QWidget(self.controller_card)
        evolution_layout = QFormLayout(self.evolution_controls)
        evolution_layout.setContentsMargins(0, 0, 0, 0)
        evolution_layout.setSpacing(8)
        self.evolution_matrix_combo = AutoCloseComboBox(self.controller_card)
        self.evolution_matrix_combo.addItem("Select 2D array", "")
        evolution_layout.addRow("Evolution array", self.evolution_matrix_combo)

        self.evolution_time_combo = AutoCloseComboBox(self.controller_card)
        self.evolution_time_combo.addItem("Row index", "")
        evolution_layout.addRow("Time axis", self.evolution_time_combo)

        self.evolution_value_combo = AutoCloseComboBox(self.controller_card)
        self.evolution_value_combo.addItem("Column index", "")
        evolution_layout.addRow("Value axis", self.evolution_value_combo)

        self.evolution_step_slider = QSlider(Qt.Orientation.Horizontal, self.controller_card)
        self.evolution_step_slider.setRange(0, 0)
        evolution_layout.addRow("Time step", self.evolution_step_slider)

        self.evolution_step_label = QLabel("Step 0 / 0", self.controller_card)
        evolution_layout.addRow("Selected step", self.evolution_step_label)
        controls_layout.addWidget(self.evolution_controls)

        self.style_controls = QWidget(self.controller_card)
        style_controls_layout = QVBoxLayout(self.style_controls)
        style_controls_layout.setContentsMargins(0, 0, 0, 0)
        style_controls_layout.setSpacing(6)

        style_title = QLabel("Plot Style", self.style_controls)
        style_title.setStyleSheet("color:#001f41; font-weight:700; font-size:12px;")
        style_controls_layout.addWidget(style_title)

        style_row_top = QHBoxLayout()
        style_row_top.setContentsMargins(0, 0, 0, 0)
        style_row_top.setSpacing(8)

        self.font_size_combo = AutoCloseComboBox(self.controller_card)
        for value in FONT_SIZE_OPTIONS:
            self.font_size_combo.addItem(str(value), value)
        self.font_size_combo.setCurrentIndex(self.font_size_combo.findData(16))

        self.line_width_combo = AutoCloseComboBox(self.controller_card)
        for value in LINE_WIDTH_OPTIONS:
            self.line_width_combo.addItem(str(value), value)
        self.line_width_combo.setCurrentIndex(self.line_width_combo.findData(2))

        self.marker_size_combo = AutoCloseComboBox(self.controller_card)
        for value in MARKER_SIZE_OPTIONS:
            self.marker_size_combo.addItem(str(value), value)
        marker_index = self.marker_size_combo.findData(8)
        if marker_index < 0:
            marker_index = self.marker_size_combo.findData(7)
        self.marker_size_combo.setCurrentIndex(max(marker_index, 0))

        for label_text, widget in (
            ("Font size", self.font_size_combo),
            ("Line width", self.line_width_combo),
            ("Marker size", self.marker_size_combo),
        ):
            block = QWidget(self.style_controls)
            block_layout = QVBoxLayout(block)
            block_layout.setContentsMargins(0, 0, 0, 0)
            block_layout.setSpacing(4)
            block_label = QLabel(label_text, block)
            block_label.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
            block_layout.addWidget(block_label)
            block_layout.addWidget(widget)
            style_row_top.addWidget(block, 1)
        style_controls_layout.addLayout(style_row_top)

        style_row_bottom = QHBoxLayout()
        style_row_bottom.setContentsMargins(0, 0, 0, 0)
        style_row_bottom.setSpacing(12)

        self.show_grid_check = QCheckBox("Grid", self.style_controls)
        self.show_grid_check.setChecked(True)
        self.show_box_check = QCheckBox("Box", self.style_controls)
        self.show_box_check.setChecked(True)
        self.ticks_inside_check = QCheckBox("Ticks inside", self.style_controls)
        self.ticks_inside_check.setChecked(True)
        self.minor_ticks_check = QCheckBox("Minor ticks", self.style_controls)
        self.minor_ticks_check.setChecked(True)
        for checkbox in (
            self.show_grid_check,
            self.show_box_check,
            self.ticks_inside_check,
            self.minor_ticks_check,
        ):
            checkbox.setStyleSheet(
                """
                QCheckBox {
                    color:#355070;
                    font-weight:600;
                    font-size:12px;
                }
                QCheckBox::indicator {
                    width:14px;
                    height:14px;
                    border:1.5px solid #000000;
                    border-radius:3px;
                    background:#ffffff;
                }
                QCheckBox::indicator:checked {
                    border:1.5px solid #000000;
                    background:#d8b4fe;
                }
                """
            )
            print(
                f"[debug][notebook-plot-panel] checkbox_style_applied text={checkbox.text()!r} border='#000000'",
                flush=True,
            )
            style_row_bottom.addWidget(checkbox)
        style_row_bottom.addStretch(1)
        style_controls_layout.addLayout(style_row_bottom)
        controls_layout.addWidget(self.style_controls)

        shared_labels = QWidget(self.controller_card)
        shared_layout = QHBoxLayout(shared_labels)
        shared_layout.setContentsMargins(0, 0, 0, 0)
        shared_layout.setSpacing(8)
        self.title_edit = QLineEdit(self.controller_card)
        self.x_label_edit = QLineEdit(self.controller_card)
        self.y_label_edit = QLineEdit(self.controller_card)
        for label_text, widget in (
            ("Title", self.title_edit),
            ("X label", self.x_label_edit),
            ("Y label", self.y_label_edit),
        ):
            section_widget = QWidget(shared_labels)
            section = QVBoxLayout(section_widget)
            section.setContentsMargins(0, 0, 0, 0)
            section.setSpacing(4)
            section_label = QLabel(label_text, section_widget)
            section_label.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
            section.addWidget(section_label)
            section.addWidget(widget)
            shared_layout.addWidget(section_widget, 1)
        controls_layout.addWidget(shared_labels)

        settings_layout.addLayout(controls_layout)
        settings_layout.addStretch(1)

        self.preview_panel = QWidget(self.controller_card)
        preview_layout = QVBoxLayout(self.preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)

        self.controller_status = QLabel("Waiting for notebook arrays", self.preview_panel)
        self.controller_status.setStyleSheet("color:#64748b; font-size:12px;")
        preview_layout.addWidget(self.controller_status)

        self.controller_plot = PlotView(self.preview_panel)
        self.controller_plot.setMinimumHeight(520)
        self.controller_plot.setMaximumHeight(16777215)
        preview_layout.addWidget(self.controller_plot, 1)
        card_layout.addWidget(self.settings_panel, 0)
        card_layout.addWidget(self.preview_panel, 1)
        root.addWidget(self.controller_card)

        self.outputs_scroll = QScrollArea(self)
        self.outputs_scroll.setWidgetResizable(True)
        self.outputs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.outputs_scroll.setStyleSheet("QScrollArea { background:#ffffff; border:1px solid #d1dce8; }")
        self.outputs_container = QWidget(self.outputs_scroll)
        self.outputs_container.setStyleSheet("background:#ffffff;")
        print("[debug][notebook-plot-panel] outputs_container_style background='#ffffff'", flush=True)
        self.outputs_layout = QVBoxLayout(self.outputs_container)
        self.outputs_layout.setContentsMargins(0, 0, 0, 0)
        self.outputs_layout.setSpacing(8)
        self.outputs_layout.addStretch(1)
        self.outputs_scroll.setWidget(self.outputs_container)
        self.outputs_scroll.hide()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.mode_combo.currentIndexChanged.connect(self._sync_mode_ui)
        self.x_combo.currentIndexChanged.connect(self.refresh_controller_plot)
        self.y_combo.checkedItemsChanged.connect(self._handle_y_selection_changed)
        self.plot_type_combo.currentIndexChanged.connect(self.refresh_controller_plot)
        self.evolution_matrix_combo.currentIndexChanged.connect(self._on_evolution_matrix_changed)
        self.evolution_time_combo.currentIndexChanged.connect(self.refresh_controller_plot)
        self.evolution_value_combo.currentIndexChanged.connect(self.refresh_controller_plot)
        self.evolution_step_slider.valueChanged.connect(self._on_evolution_step_changed)
        self.title_edit.textChanged.connect(self.refresh_controller_plot)
        self.x_label_edit.textChanged.connect(self.refresh_controller_plot)
        self.y_label_edit.textChanged.connect(self.refresh_controller_plot)
        self.font_size_combo.currentIndexChanged.connect(self.refresh_controller_plot)
        self.line_width_combo.currentIndexChanged.connect(self.refresh_controller_plot)
        self.marker_size_combo.currentIndexChanged.connect(self.refresh_controller_plot)
        self.graph_size_combo.currentIndexChanged.connect(self.refresh_controller_plot)
        self.show_grid_check.toggled.connect(self.refresh_controller_plot)
        self.show_box_check.toggled.connect(self.refresh_controller_plot)
        self.ticks_inside_check.toggled.connect(self.refresh_controller_plot)
        self.minor_ticks_check.toggled.connect(self.refresh_controller_plot)

        self._sync_mode_ui()
        self.refresh_controller_plot()
        print("[debug][notebook-plot-panel] init:done", flush=True)

    def current_controller_figure(self) -> go.Figure:
        print(f"[debug][notebook-plot-panel] current_controller_figure traces={len(self._current_figure.data)}", flush=True)
        return self._current_figure

    def output_count(self) -> int:
        count = len(self._output_widgets)
        print(f"[debug][notebook-plot-panel] output_count count={count}", flush=True)
        return count

    def output_titles(self) -> list[str]:
        titles = [widget_tuple[0].text() for widget_tuple in self._output_widgets.values()]
        print(f"[debug][notebook-plot-panel] output_titles titles={titles!r}", flush=True)
        return titles

    def _selected_y_vars(self) -> list[str]:
        selected = self.y_combo.checked_values()
        print(f"[debug][notebook-plot-panel] selected_y_vars selected={selected!r}", flush=True)
        return [value for value in selected if isinstance(value, str)]

    def _handle_y_selection_changed(self) -> None:
        print("[debug][notebook-plot-panel] handle_y_selection_changed", flush=True)
        self._refresh_series_style_rows()
        self.refresh_controller_plot()

    def _clear_series_style_rows(self) -> None:
        print(f"[debug][notebook-plot-panel] clear_series_style_rows count={len(self._series_style_widgets)}", flush=True)
        while self.series_style_grid.count():
            item = self.series_style_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._series_style_widgets.clear()

    def _refresh_series_style_rows(self) -> None:
        selected_y = self._selected_y_vars()
        previous_values = {
            name: {
                "plot_type": widgets["plot_type"].currentData(),
                "line_style": widgets["line_style"].currentData(),
                "marker_style": widgets["marker_style"].currentData(),
                "line_color": widgets["line_color"].currentData(),
            }
            for name, widgets in self._series_style_widgets.items()
        }
        print(
            f"[debug][notebook-plot-panel] refresh_series_style_rows selected_y={selected_y!r} "
            f"previous_values={previous_values!r}",
            flush=True,
        )
        self._clear_series_style_rows()
        self.series_style_card.setVisible(bool(selected_y))
        if not selected_y:
            return
        headers = ("Y variable", "Plot type", "Line style", "Marker style", "Line color")
        for column, header in enumerate(headers):
            label = QLabel(header, self.series_style_card)
            label.setStyleSheet("color:#355070; font-weight:700; font-size:11px;")
            self.series_style_grid.addWidget(label, 0, column)
        for row, name in enumerate(selected_y, start=1):
            name_label = QLabel(name, self.series_style_card)
            name_label.setStyleSheet("color:#0f1b2b; font-weight:600; font-size:12px;")
            plot_type_combo = AutoCloseComboBox(self.series_style_card)
            for label, value in PLOT_TYPE_OPTIONS:
                plot_type_combo.addItem(label, value)
            line_style_combo = AutoCloseComboBox(self.series_style_card)
            for label, value in LINE_STYLE_OPTIONS:
                line_style_combo.addItem(label, value)
            marker_style_combo = AutoCloseComboBox(self.series_style_card)
            for label, value in MARKER_STYLE_OPTIONS:
                marker_style_combo.addItem(label, value)
            line_color_combo = AutoCloseComboBox(self.series_style_card)
            for label, value in LINE_COLOR_OPTIONS:
                line_color_combo.addItem(label, value)
            plot_type_value = previous_values.get(name, {}).get("plot_type") or (self.plot_type_combo.currentData() or "lines")
            line_style_value = previous_values.get(name, {}).get("line_style") or "solid"
            marker_style_value = previous_values.get(name, {}).get("marker_style") or "circle"
            line_color_value = previous_values.get(name, {}).get("line_color") or "#000000"
            plot_index = plot_type_combo.findData(plot_type_value)
            if plot_index >= 0:
                plot_type_combo.setCurrentIndex(plot_index)
            line_index = line_style_combo.findData(line_style_value)
            if line_index >= 0:
                line_style_combo.setCurrentIndex(line_index)
            marker_index = marker_style_combo.findData(marker_style_value)
            if marker_index >= 0:
                marker_style_combo.setCurrentIndex(marker_index)
            line_color_index = line_color_combo.findData(line_color_value)
            if line_color_index >= 0:
                line_color_combo.setCurrentIndex(line_color_index)
            plot_type_combo.currentIndexChanged.connect(self.refresh_controller_plot)
            line_style_combo.currentIndexChanged.connect(self.refresh_controller_plot)
            marker_style_combo.currentIndexChanged.connect(self.refresh_controller_plot)
            line_color_combo.currentIndexChanged.connect(self.refresh_controller_plot)
            self.series_style_grid.addWidget(name_label, row, 0)
            self.series_style_grid.addWidget(plot_type_combo, row, 1)
            self.series_style_grid.addWidget(line_style_combo, row, 2)
            self.series_style_grid.addWidget(marker_style_combo, row, 3)
            self.series_style_grid.addWidget(line_color_combo, row, 4)
            self._series_style_widgets[name] = {
                "plot_type": plot_type_combo,
                "line_style": line_style_combo,
                "marker_style": marker_style_combo,
                "line_color": line_color_combo,
            }
            print(
                f"[debug][notebook-plot-panel] refresh_series_style_rows:item name={name!r} "
                f"plot_type={plot_type_value!r} line_style={line_style_value!r} "
                f"marker_style={marker_style_value!r} line_color={line_color_value!r}",
                flush=True,
            )

    def _series_style_map(self) -> dict[str, dict[str, str]]:
        style_map: dict[str, dict[str, str]] = {}
        for name, widgets in self._series_style_widgets.items():
            style_map[name] = {
                "plot_type": str(widgets["plot_type"].currentData() or "lines"),
                "line_style": str(widgets["line_style"].currentData() or "solid"),
                "marker_style": str(widgets["marker_style"].currentData() or "circle"),
                "line_color": str(widgets["line_color"].currentData() or "#000000"),
            }
            print(f"[debug][notebook-plot-panel] series_style_map:item name={name!r} style={style_map[name]!r}", flush=True)
        return style_map

    def _scientific_style_options(self) -> dict[str, Any]:
        options = {
            "font_size": int(self.font_size_combo.currentData() or 16),
            "line_width": int(self.line_width_combo.currentData() or 2),
            "marker_size": int(self.marker_size_combo.currentData() or 8),
            "show_grid": self.show_grid_check.isChecked(),
            "show_box": self.show_box_check.isChecked(),
            "ticks_inside": self.ticks_inside_check.isChecked(),
            "show_minor_ticks": self.minor_ticks_check.isChecked(),
        }
        graph_size = self.graph_size_combo.currentData()
        aspect_heights = {
            "auto": 520,
            "landscape": 520,
            "square": 700,
            "tall": 860,
            "wide": 460,
        }
        aspect_name = str(graph_size or "auto")
        options["graph_aspect"] = aspect_name
        options["graph_width"] = None
        options["graph_height"] = int(aspect_heights.get(aspect_name, 520))
        print(f"[debug][notebook-plot-panel] scientific_style_options options={options!r}", flush=True)
        return options

    def _sync_mode_ui(self) -> None:
        mode = self.mode_combo.currentData() or "series"
        print(f"[debug][notebook-plot-panel] sync_mode_ui mode={mode!r}", flush=True)
        is_series = mode == "series"
        self.series_controls.setVisible(is_series)
        self.series_style_card.setVisible(is_series and bool(self._selected_y_vars()))
        self.evolution_controls.setVisible(not is_series)
        self.refresh_controller_plot()

    def _on_evolution_matrix_changed(self) -> None:
        matrix_name = self.evolution_matrix_combo.currentData()
        print(f"[debug][notebook-plot-panel] evolution_matrix_changed matrix_name={matrix_name!r}", flush=True)
        matrix = self._arrays_2d.get(matrix_name) if isinstance(matrix_name, str) and matrix_name else None
        rows = int(matrix.shape[0]) if matrix is not None else 0
        max_index = max(rows - 1, 0)
        self.evolution_step_slider.blockSignals(True)
        self.evolution_step_slider.setRange(0, max_index)
        self.evolution_step_slider.setValue(min(self.evolution_step_slider.value(), max_index))
        self.evolution_step_slider.blockSignals(False)
        self._update_evolution_step_label()
        self.refresh_controller_plot()

    def _on_evolution_step_changed(self, value: int) -> None:
        print(f"[debug][notebook-plot-panel] evolution_step_changed value={value}", flush=True)
        self._update_evolution_step_label()
        self.refresh_controller_plot()

    def _update_evolution_step_label(self) -> None:
        current = self.evolution_step_slider.value()
        total = self.evolution_step_slider.maximum() + 1 if self.evolution_step_slider.maximum() >= 0 else 0
        label = f"Step {current} / {max(total - 1, 0)}"
        print(f"[debug][notebook-plot-panel] evolution_step_label label={label!r}", flush=True)
        self.evolution_step_label.setText(label)

    def _graph_title_for_cell(self, cell: Any) -> str:
        source_lines = [line.strip() for line in cell.source().splitlines() if line.strip()]
        title = source_lines[0] if source_lines else getattr(cell, "cell_id", "plot")
        if len(title) > 48:
            title = title[:45] + "..."
        print(f"[debug][notebook-plot-panel] graph_title title={title!r}", flush=True)
        return title

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        print(f"[debug][notebook-plot-panel] set_namespace:start count={len(namespace)}", flush=True)
        previous_x = self.x_combo.currentData()
        previous_y = set(self._selected_y_vars())
        previous_matrix = self.evolution_matrix_combo.currentData()
        previous_time = self.evolution_time_combo.currentData()
        previous_value = self.evolution_value_combo.currentData()
        print(
            f"[debug][notebook-plot-panel] set_namespace:previous x={previous_x!r} "
            f"y_multi={sorted(previous_y)!r} "
            f"matrix={previous_matrix!r} time={previous_time!r} value={previous_value!r}",
            flush=True,
        )
        self._arrays_1d, self._arrays_2d = extract_notebook_array_variables(namespace)
        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)
        self.evolution_matrix_combo.blockSignals(True)
        self.evolution_time_combo.blockSignals(True)
        self.evolution_value_combo.blockSignals(True)
        self.x_combo.clear()
        self.x_combo.addItem("Index", "")
        self.y_combo.clear()
        self.evolution_matrix_combo.clear()
        self.evolution_matrix_combo.addItem("Select 2D array", "")
        self.evolution_time_combo.clear()
        self.evolution_time_combo.addItem("Row index", "")
        self.evolution_value_combo.clear()
        self.evolution_value_combo.addItem("Column index", "")
        for name in sorted(self._arrays_1d.keys()):
            self.x_combo.addItem(name, name)
            self.evolution_time_combo.addItem(name, name)
            self.evolution_value_combo.addItem(name, name)
            self.y_combo.add_check_item(name, name, checked=name in previous_y)
            print(f"[debug][notebook-plot-panel] set_namespace:item name={name!r}", flush=True)
        for name in sorted(self._arrays_2d.keys()):
            self.evolution_matrix_combo.addItem(name, name)
            print(f"[debug][notebook-plot-panel] set_namespace:matrix_item name={name!r}", flush=True)
        x_index = self.x_combo.findData(previous_x)
        if x_index >= 0:
            self.x_combo.setCurrentIndex(x_index)
        matrix_index = self.evolution_matrix_combo.findData(previous_matrix)
        if matrix_index >= 0:
            self.evolution_matrix_combo.setCurrentIndex(matrix_index)
        time_index = self.evolution_time_combo.findData(previous_time)
        if time_index >= 0:
            self.evolution_time_combo.setCurrentIndex(time_index)
        value_index = self.evolution_value_combo.findData(previous_value)
        if value_index >= 0:
            self.evolution_value_combo.setCurrentIndex(value_index)
        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)
        self.evolution_matrix_combo.blockSignals(False)
        self.evolution_time_combo.blockSignals(False)
        self.evolution_value_combo.blockSignals(False)
        self._on_evolution_matrix_changed()
        self._refresh_series_style_rows()
        status_text = (
            f"{len(self._arrays_1d)} 1D array(s), {len(self._arrays_2d)} 2D array(s) available for plotting"
            if self._arrays_1d or self._arrays_2d
            else "No numeric notebook arrays available yet"
        )
        self.controller_status.setText(status_text)
        print(f"[debug][notebook-plot-panel] set_namespace:status text={status_text!r}", flush=True)
        self.refresh_controller_plot()

    def refresh_controller_plot(self) -> None:
        mode = self.mode_combo.currentData() or "series"
        x_var = self.x_combo.currentData()
        y_vars = self._selected_y_vars()
        current_style_names = sorted(self._series_style_widgets.keys())
        if mode == "series" and current_style_names != sorted(y_vars):
            print(
                f"[debug][notebook-plot-panel] refresh_controller_plot:style_rows_out_of_sync "
                f"style_names={current_style_names!r} y_vars={sorted(y_vars)!r}",
                flush=True,
            )
            self._refresh_series_style_rows()
        plot_type = self.plot_type_combo.currentData() or "lines"
        matrix_var = self.evolution_matrix_combo.currentData()
        time_var = self.evolution_time_combo.currentData()
        value_var = self.evolution_value_combo.currentData()
        step_index = self.evolution_step_slider.value()
        title = self.title_edit.text().strip()
        x_title = self.x_label_edit.text().strip()
        y_title = self.y_label_edit.text().strip()
        series_styles = self._series_style_map()
        style_options = self._scientific_style_options()
        print(
            f"[debug][notebook-plot-panel] refresh_controller_plot:start mode={mode!r} x_var={x_var!r} "
            f"y_vars={y_vars!r} matrix_var={matrix_var!r} step_index={step_index} plot_type={plot_type!r} "
            f"series_styles={series_styles!r} style_options={style_options!r}",
            flush=True,
        )
        if mode == "evolution":
            self._current_figure = build_notebook_evolution_figure(
                self._arrays_1d,
                self._arrays_2d,
                matrix_var if isinstance(matrix_var, str) else None,
                time_var if isinstance(time_var, str) else None,
                value_var if isinstance(value_var, str) else None,
                step_index,
                plot_type if isinstance(plot_type, str) else "lines",
                title,
                x_title,
                y_title,
                style_options,
            )
        else:
            self._current_figure = build_notebook_plot_figure(
                self._arrays_1d,
                x_var if isinstance(x_var, str) else None,
                y_vars,
                plot_type if isinstance(plot_type, str) else "lines",
                title,
                x_title,
                y_title,
                series_styles,
                style_options,
            )
        html = self._current_figure.to_html(
            include_plotlyjs=True,
            full_html=False,
            config={"responsive": True, "displaylogo": False},
        )
        graph_height = int(style_options.get("graph_height", 520))
        self.controller_plot.setMinimumHeight(graph_height)
        print(f"[debug][notebook-plot-panel] refresh_controller_plot:plot_height height={graph_height}", flush=True)
        print(f"[debug][notebook-plot-panel] refresh_controller_plot:html length={len(html)}", flush=True)
        self.controller_plot.set_html(html)
        if mode == "evolution":
            self.controller_status.setText(
                f"Evolution plot updated: matrix={matrix_var or 'none'} step={step_index} type={plot_type}"
            )
        else:
            self.controller_status.setText(
                f"Controller plot updated: x={x_var or 'index'} y={', '.join(y_vars) if y_vars else 'none'} type={plot_type}"
            )
        print("[debug][notebook-plot-panel] refresh_controller_plot:done", flush=True)

    def sync_cell_outputs(self, cells: list[Any]) -> None:
        print(f"[debug][notebook-plot-panel] sync_cell_outputs:start cells={len(cells)}", flush=True)
        desired_order: list[str] = []
        for cell in cells:
            result = getattr(cell, "last_result", None)
            print(
                f"[debug][notebook-plot-panel] sync_cell_outputs:cell cell_id={getattr(cell, 'cell_id', None)!r} "
                f"has_result={result is not None}",
                flush=True,
            )
            if result is None:
                continue
            for output_index, output in enumerate(result.outputs):
                print(
                    f"[debug][notebook-plot-panel] sync_cell_outputs:output index={output_index} kind={output.kind!r}",
                    flush=True,
                )
                if output.kind == "plotly":
                    key = f"{cell.cell_id}:plotly:{output_index}"
                elif output.kind == "html" and "data:image" in output.data.get("html", ""):
                    key = f"{cell.cell_id}:image:{output_index}"
                else:
                    continue
                desired_order.append(key)
                title_text = self._graph_title_for_cell(cell)
                existing = self._output_widgets.get(key)
                html = output.data["html"]
                if existing is None:
                    print(f"[debug][notebook-plot-panel] sync_cell_outputs:create key={key!r}", flush=True)
                    title = QLabel(title_text, self.outputs_container)
                    title.setStyleSheet("color:#001f41; font-weight:600;")
                    plot = PlotView(self.outputs_container)
                    plot.set_html(html)
                    self._output_widgets[key] = (title, plot, html)
                else:
                    title, plot, old_html = existing
                    title.setText(title_text)
                    if old_html != html:
                        print(f"[debug][notebook-plot-panel] sync_cell_outputs:update key={key!r}", flush=True)
                        plot.set_html(html)
                        self._output_widgets[key] = (title, plot, html)
        obsolete_keys = [key for key in self._output_widgets if key not in desired_order]
        print(f"[debug][notebook-plot-panel] sync_cell_outputs:obsolete obsolete={obsolete_keys!r}", flush=True)
        for key in obsolete_keys:
            title, widget, _html = self._output_widgets.pop(key)
            title.hide()
            widget.hide()
            title.deleteLater()
            widget.deleteLater()
            print(f"[debug][notebook-plot-panel] sync_cell_outputs:removed key={key!r}", flush=True)
        while self.outputs_layout.count():
            item = self.outputs_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        if self._output_empty_label is not None:
            self._output_empty_label.deleteLater()
            self._output_empty_label = None
        if not desired_order:
            empty = QLabel("Run a plotting cell to see output graphs here.", self.outputs_container)
            empty.setStyleSheet("color:#64748b; font-style:italic;")
            self.outputs_layout.addWidget(empty)
            self._output_empty_label = empty
            print("[debug][notebook-plot-panel] sync_cell_outputs:empty_label", flush=True)
        else:
            for key in desired_order:
                title, widget, _html = self._output_widgets[key]
                self.outputs_layout.addWidget(title)
                self.outputs_layout.addWidget(widget)
                print(f"[debug][notebook-plot-panel] sync_cell_outputs:attached key={key!r}", flush=True)
        self.outputs_layout.addStretch(1)
        print(f"[debug][notebook-plot-panel] sync_cell_outputs:done outputs={len(desired_order)}", flush=True)


class QuickGraphPreviewPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][quick-graph-preview] init:start", flush=True)
        self._arrays_1d: dict[str, np.ndarray] = {}
        self._arrays_2d: dict[str, np.ndarray] = {}
        self._latest_plot_title = ""
        self._latest_plot_html = ""
        self._current_figure = build_notebook_plot_figure({}, None, [], "lines", "", "", "")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        header = QHBoxLayout()
        label = QLabel("Quick Graph Preview", self)
        label.setStyleSheet("color:#001f41; font-weight:700; font-size:15px;")
        header.addWidget(label)
        header.addStretch(1)
        root.addLayout(header)

        self.status_label = QLabel("Use the quick controls to preview notebook graphs.", self)
        self.status_label.setStyleSheet("color:#64748b; font-size:12px;")
        root.addWidget(self.status_label)

        controls_card = QWidget(self)
        controls_card.setStyleSheet(
            "QWidget { background:#ffffff; border:1px solid #d1dce8; }"
            "QLabel { border:none; background:transparent; color:#355070; font-weight:600; font-size:12px; }"
            "QComboBox, QLineEdit { background:#ffffff; border:1px solid #d1dce8; border-radius:6px; padding:4px 6px; }"
            "QComboBox QAbstractItemView { selection-background-color:#c7def5; selection-color:#0f1b2b; outline:0; }"
            "QComboBox QAbstractItemView::item:hover { background:#c7def5; color:#0f1b2b; }"
        )
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(8)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(8)
        mode_label = QLabel("Mode", controls_card)
        self.mode_combo = AutoCloseComboBox(controls_card)
        self.mode_combo.addItem("Series (1D)", "series")
        self.mode_combo.addItem("Evolution (2D)", "evolution")
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.mode_combo, 1)
        controls_layout.addLayout(mode_row)

        self.series_controls = QWidget(controls_card)
        series_layout = QHBoxLayout(self.series_controls)
        series_layout.setContentsMargins(0, 0, 0, 0)
        series_layout.setSpacing(8)

        x_block = QWidget(self.series_controls)
        x_layout = QVBoxLayout(x_block)
        x_layout.setContentsMargins(0, 0, 0, 0)
        x_layout.setSpacing(4)
        x_layout.addWidget(QLabel("X variable", x_block))
        self.x_combo = AutoCloseComboBox(x_block)
        self.x_combo.addItem("Index", "")
        x_layout.addWidget(self.x_combo)
        series_layout.addWidget(x_block, 1)

        plot_type_block = QWidget(self.series_controls)
        plot_type_layout = QVBoxLayout(plot_type_block)
        plot_type_layout.setContentsMargins(0, 0, 0, 0)
        plot_type_layout.setSpacing(4)
        plot_type_layout.addWidget(QLabel("Plot type", plot_type_block))
        self.plot_type_combo = AutoCloseComboBox(plot_type_block)
        for text, value in PLOT_TYPE_OPTIONS:
            self.plot_type_combo.addItem(text, value)
        plot_type_layout.addWidget(self.plot_type_combo)
        series_layout.addWidget(plot_type_block, 1)

        y_block = QWidget(self.series_controls)
        y_layout = QVBoxLayout(y_block)
        y_layout.setContentsMargins(0, 0, 0, 0)
        y_layout.setSpacing(4)
        y_layout.addWidget(QLabel("Y variable(s)", y_block))
        self.y_combo = CheckableComboBox(y_block)
        y_layout.addWidget(self.y_combo)
        series_layout.addWidget(y_block, 2)
        controls_layout.addWidget(self.series_controls)

        self.evolution_controls = QWidget(controls_card)
        evolution_layout = QGridLayout(self.evolution_controls)
        evolution_layout.setContentsMargins(0, 0, 0, 0)
        evolution_layout.setHorizontalSpacing(8)
        evolution_layout.setVerticalSpacing(6)
        evolution_layout.addWidget(QLabel("Evolution array", self.evolution_controls), 0, 0)
        self.evolution_matrix_combo = AutoCloseComboBox(self.evolution_controls)
        self.evolution_matrix_combo.addItem("Select 2D array", "")
        evolution_layout.addWidget(self.evolution_matrix_combo, 0, 1)
        evolution_layout.addWidget(QLabel("Time axis", self.evolution_controls), 1, 0)
        self.evolution_time_combo = AutoCloseComboBox(self.evolution_controls)
        self.evolution_time_combo.addItem("Row index", "")
        evolution_layout.addWidget(self.evolution_time_combo, 1, 1)
        evolution_layout.addWidget(QLabel("Value axis", self.evolution_controls), 2, 0)
        self.evolution_value_combo = AutoCloseComboBox(self.evolution_controls)
        self.evolution_value_combo.addItem("Column index", "")
        evolution_layout.addWidget(self.evolution_value_combo, 2, 1)
        evolution_layout.addWidget(QLabel("Time step", self.evolution_controls), 3, 0)
        self.evolution_step_slider = QSlider(Qt.Orientation.Horizontal, self.evolution_controls)
        self.evolution_step_slider.setRange(0, 0)
        evolution_layout.addWidget(self.evolution_step_slider, 3, 1)
        self.evolution_step_label = QLabel("Step 0 / 0", self.evolution_controls)
        evolution_layout.addWidget(self.evolution_step_label, 4, 1)
        controls_layout.addWidget(self.evolution_controls)
        root.addWidget(controls_card)

        self.plot_title = QLabel("", self)
        self.plot_title.setStyleSheet("color:#001f41; font-weight:600; font-size:13px;")
        self.plot_title.hide()
        root.addWidget(self.plot_title)

        self.plot_view = PlotView(self)
        self.plot_view.setMinimumHeight(360)
        self.plot_view.setMaximumHeight(360)
        root.addWidget(self.plot_view)

        self.empty_label = QLabel("No plot output yet.", self)
        self.empty_label.setStyleSheet("color:#64748b; font-style:italic;")
        root.addWidget(self.empty_label)
        self.plot_view.hide()

        self.mode_combo.currentIndexChanged.connect(self._sync_mode_ui)
        self.x_combo.currentIndexChanged.connect(self.refresh_preview)
        self.plot_type_combo.currentIndexChanged.connect(self.refresh_preview)
        self.y_combo.checkedItemsChanged.connect(self.refresh_preview)
        self.evolution_matrix_combo.currentIndexChanged.connect(self._on_evolution_matrix_changed)
        self.evolution_time_combo.currentIndexChanged.connect(self.refresh_preview)
        self.evolution_value_combo.currentIndexChanged.connect(self.refresh_preview)
        self.evolution_step_slider.valueChanged.connect(self._on_evolution_step_changed)
        self._sync_mode_ui()
        print("[debug][quick-graph-preview] init:done", flush=True)

    def _quick_style_options(self) -> dict[str, Any]:
        options = {
            "font_size": 14,
            "line_width": 2,
            "marker_size": 8,
            "show_grid": True,
            "show_box": True,
            "ticks_inside": True,
            "show_minor_ticks": True,
            "graph_aspect": "auto",
            "graph_width": None,
            "graph_height": 360,
        }
        print(f"[debug][quick-graph-preview] style_options options={options!r}", flush=True)
        return options

    def _selected_y_vars(self) -> list[str]:
        selected = self.y_combo.checked_values()
        print(f"[debug][quick-graph-preview] selected_y_vars selected={selected!r}", flush=True)
        return selected

    def _sync_mode_ui(self) -> None:
        mode = self.mode_combo.currentData() or "series"
        print(f"[debug][quick-graph-preview] sync_mode_ui mode={mode!r}", flush=True)
        self.series_controls.setVisible(mode == "series")
        self.evolution_controls.setVisible(mode == "evolution")
        self.refresh_preview()

    def _on_evolution_matrix_changed(self) -> None:
        matrix_name = self.evolution_matrix_combo.currentData()
        print(f"[debug][quick-graph-preview] evolution_matrix_changed matrix_name={matrix_name!r}", flush=True)
        matrix = self._arrays_2d.get(matrix_name) if isinstance(matrix_name, str) else None
        rows = int(matrix.shape[0]) if matrix is not None else 0
        max_step = max(rows - 1, 0)
        self.evolution_step_slider.blockSignals(True)
        self.evolution_step_slider.setRange(0, max_step)
        if self.evolution_step_slider.value() > max_step:
            self.evolution_step_slider.setValue(max_step)
        self.evolution_step_slider.blockSignals(False)
        self._update_evolution_step_label()
        self.refresh_preview()

    def _on_evolution_step_changed(self, *_args: object) -> None:
        self._update_evolution_step_label()
        self.refresh_preview()

    def _update_evolution_step_label(self) -> None:
        maximum = self.evolution_step_slider.maximum()
        label = f"Step {self.evolution_step_slider.value()} / {maximum}"
        self.evolution_step_label.setText(label)
        print(f"[debug][quick-graph-preview] evolution_step_label label={label!r}", flush=True)

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        print(f"[debug][quick-graph-preview] set_namespace:start count={len(namespace)}", flush=True)
        previous_x = self.x_combo.currentData()
        previous_y = self._selected_y_vars()
        previous_matrix = self.evolution_matrix_combo.currentData()
        previous_time = self.evolution_time_combo.currentData()
        previous_value = self.evolution_value_combo.currentData()
        print(
            f"[debug][quick-graph-preview] set_namespace:previous x={previous_x!r} y={previous_y!r} "
            f"matrix={previous_matrix!r} time={previous_time!r} value={previous_value!r}",
            flush=True,
        )
        self._arrays_1d, self._arrays_2d = extract_notebook_array_variables(namespace)

        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)
        self.evolution_matrix_combo.blockSignals(True)
        self.evolution_time_combo.blockSignals(True)
        self.evolution_value_combo.blockSignals(True)

        self.x_combo.clear()
        self.x_combo.addItem("Index", "")
        for name in self._arrays_1d:
            self.x_combo.addItem(name, name)
        x_data = previous_x if isinstance(previous_x, str) and previous_x in self._arrays_1d else ""
        self.x_combo.setCurrentIndex(max(0, self.x_combo.findData(x_data)))

        self.y_combo.clear()
        selected_y = [name for name in previous_y if name in self._arrays_1d]
        available_series = list(self._arrays_1d.keys())
        for name in available_series:
            default_checked = name in selected_y
            self.y_combo.add_check_item(name, name, checked=default_checked)
        if not selected_y and available_series:
            default_y = available_series[1] if len(available_series) > 1 and x_data == available_series[0] else available_series[0]
            self.y_combo.set_checked_values([default_y])

        self.evolution_matrix_combo.clear()
        self.evolution_matrix_combo.addItem("Select 2D array", "")
        for name in self._arrays_2d:
            self.evolution_matrix_combo.addItem(name, name)
        matrix_data = previous_matrix if isinstance(previous_matrix, str) and previous_matrix in self._arrays_2d else ""
        matrix_index = self.evolution_matrix_combo.findData(matrix_data)
        if matrix_index < 0:
            matrix_index = 1 if self.evolution_matrix_combo.count() > 1 else 0
        self.evolution_matrix_combo.setCurrentIndex(max(matrix_index, 0))

        self.evolution_time_combo.clear()
        self.evolution_time_combo.addItem("Row index", "")
        self.evolution_value_combo.clear()
        self.evolution_value_combo.addItem("Column index", "")
        for name in self._arrays_1d:
            self.evolution_time_combo.addItem(name, name)
            self.evolution_value_combo.addItem(name, name)
        time_data = previous_time if isinstance(previous_time, str) and previous_time in self._arrays_1d else ""
        value_data = previous_value if isinstance(previous_value, str) and previous_value in self._arrays_1d else ""
        self.evolution_time_combo.setCurrentIndex(max(0, self.evolution_time_combo.findData(time_data)))
        self.evolution_value_combo.setCurrentIndex(max(0, self.evolution_value_combo.findData(value_data)))

        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)
        self.evolution_matrix_combo.blockSignals(False)
        self.evolution_time_combo.blockSignals(False)
        self.evolution_value_combo.blockSignals(False)

        self._on_evolution_matrix_changed()
        self.refresh_preview()

    def set_latest_plot(self, title: str, html: str) -> None:
        print(
            f"[debug][quick-graph-preview] set_latest_plot title={title!r} html_length={len(html)}",
            flush=True,
        )
        self._latest_plot_title = title
        self._latest_plot_html = html
        self.refresh_preview()

    def current_figure(self) -> go.Figure:
        print(f"[debug][quick-graph-preview] current_figure traces={len(self._current_figure.data)}", flush=True)
        return self._current_figure

    def refresh_preview(self) -> None:
        mode = self.mode_combo.currentData() or "series"
        x_var = self.x_combo.currentData()
        y_vars = self._selected_y_vars()
        matrix_var = self.evolution_matrix_combo.currentData()
        time_var = self.evolution_time_combo.currentData()
        value_var = self.evolution_value_combo.currentData()
        plot_type = self.plot_type_combo.currentData() or "lines"
        step_index = self.evolution_step_slider.value()
        style_options = self._quick_style_options()
        print(
            f"[debug][quick-graph-preview] refresh_preview:start mode={mode!r} x_var={x_var!r} "
            f"y_vars={y_vars!r} matrix_var={matrix_var!r} step_index={step_index} plot_type={plot_type!r}",
            flush=True,
        )

        has_series = bool(self._arrays_1d)
        has_evolution = bool(self._arrays_2d)
        use_latest_plot = not has_series and not has_evolution and bool(self._latest_plot_html)
        if use_latest_plot:
            self.plot_title.setVisible(True)
            self.plot_title.setText(self._latest_plot_title or "Latest notebook plot")
            self.plot_view.setVisible(True)
            self.empty_label.setVisible(False)
            self.status_label.setText("Showing latest executed notebook plot.")
            self.plot_view.set_html(self._latest_plot_html)
            print("[debug][quick-graph-preview] refresh_preview:fallback_latest_plot", flush=True)
            return

        if mode == "evolution":
            self._current_figure = build_notebook_evolution_figure(
                self._arrays_1d,
                self._arrays_2d,
                matrix_var if isinstance(matrix_var, str) else None,
                time_var if isinstance(time_var, str) else None,
                value_var if isinstance(value_var, str) else None,
                step_index,
                plot_type if isinstance(plot_type, str) else "lines",
                "",
                "",
                "",
                style_options,
            )
            has_plot = matrix_var in self._arrays_2d
            status = f"Quick preview: evolution {matrix_var or 'none'} step {step_index}"
            title = matrix_var or "Evolution preview"
        else:
            self._current_figure = build_notebook_plot_figure(
                self._arrays_1d,
                x_var if isinstance(x_var, str) else None,
                y_vars,
                plot_type if isinstance(plot_type, str) else "lines",
                "",
                "",
                "",
                {},
                style_options,
            )
            has_plot = bool(y_vars)
            status = f"Quick preview: x={x_var or 'index'} y={', '.join(y_vars) if y_vars else 'none'}"
            title = ", ".join(y_vars) if y_vars else "Quick preview"

        html = self._current_figure.to_html(
            include_plotlyjs=True,
            full_html=False,
            config={"responsive": True, "displaylogo": False},
        )
        self.plot_title.setVisible(has_plot)
        self.plot_title.setText(title)
        self.plot_view.setVisible(True)
        self.empty_label.setVisible(not has_plot)
        self.status_label.setText(status if has_plot else "Run code to populate arrays or create a plot output.")
        self.plot_view.set_html(html)
        print(
            f"[debug][quick-graph-preview] refresh_preview:done has_plot={has_plot} html_length={len(html)}",
            flush=True,
        )
