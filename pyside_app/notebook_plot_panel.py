from __future__ import annotations

import os
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pyside_app.controls import AutoCloseComboBox, CheckableComboBox
from pyside_app.plot_view import PlotView
from utils.fitting import BUILTIN_MODELS, FitError, fit_series


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
LINE_WIDTH_OPTIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
MARKER_SIZE_OPTIONS = (4, 6, 8, 10, 12, 14, 16, 18, 20)
GRAPH_SIZE_OPTIONS = (
    ("800 × 700", (800, 700)),
    ("800 × 600", (800, 600)),
    ("Square", (700, 700)),
    ("700 × 700", (700, 700)),
)


# ── scientific layout helpers ────────────────────────────────────────────────

def _default_scientific_style_options() -> dict[str, Any]:
    return {
        "font_size": 16,
        "line_width": 2,
        "marker_size": 7,
        "show_grid": True,
        "show_box": True,
        "ticks_inside": True,
        "show_minor_ticks": True,
        "graph_width": 800,
        "graph_height": 600,
    }


def _axis_style_options(style_options: dict[str, Any], axis_title: str) -> dict[str, Any]:
    font_size = int(style_options.get("font_size", 16))
    show_grid = bool(style_options.get("show_grid", True))
    show_box = bool(style_options.get("show_box", True))
    ticks_inside = bool(style_options.get("ticks_inside", True))
    show_minor_ticks = bool(style_options.get("show_minor_ticks", True))
    tick_direction = "inside" if ticks_inside else "outside"
    return {
        "title": {"text": axis_title, "font": {"size": font_size + 2, "color": "#0f1b2b"}},
        "automargin": True,
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
    fig.update_layout(
        title={"text": title or None, "font": {"size": font_size + 4, "color": "#0f1b2b"}},
        showlegend=showlegend,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 60, "r": 20, "t": 35, "b": 90},
        font={"size": font_size, "color": "#0f1b2b"},
        legend={
            "orientation": "v", "x": 0.01, "y": 0.99,
            "xanchor": "left", "yanchor": "top",
            "font": {"size": font_size},
        },
        barmode=barmode,
        width=graph_width,
        height=graph_height,
    )
    fig.update_xaxes(**_axis_style_options(resolved, xaxis_title))
    fig.update_yaxes(**_axis_style_options(resolved, yaxis_title))


# ── array extraction ─────────────────────────────────────────────────────────

def _coerce_numeric_series(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        return None
    if array.ndim != 1:
        return None
    if not np.issubdtype(array.dtype, np.number):
        return None
    return np.asarray(array, dtype=float)


def _coerce_numeric_matrix(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        return None
    if array.ndim != 2:
        return None
    if not np.issubdtype(array.dtype, np.number):
        return None
    return np.asarray(array, dtype=float)


def extract_notebook_array_variables(
    namespace: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays_1d: dict[str, np.ndarray] = {}
    arrays_2d: dict[str, np.ndarray] = {}
    for name, value in sorted(namespace.items()):
        if name.startswith("_") or callable(value) or isinstance(value, ModuleType):
            continue
        coerced = _coerce_numeric_series(value)
        if coerced is not None:
            arrays_1d[name] = coerced
            continue
        matrix = _coerce_numeric_matrix(value)
        if matrix is not None:
            arrays_2d[name] = matrix
    return arrays_1d, arrays_2d


# ── figure builders ──────────────────────────────────────────────────────────

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
    fig = go.Figure()
    resolved = dict(_default_scientific_style_options())
    if style_options:
        resolved.update(style_options)
    line_width = int(resolved.get("line_width", 2))
    marker_size = int(resolved.get("marker_size", 7))
    font_size = int(resolved.get("font_size", 16))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    x_data = arrays.get(x_var) if x_var else None
    if not y_vars:
        fig.add_annotation(
            text="Select Y variable(s) to build a notebook plot.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font={"size": font_size, "color": "#94a3b8"},
        )
    for index, y_name in enumerate(y_vars):
        y_data = arrays.get(y_name)
        if y_data is None:
            continue
        style = dict((series_styles or {}).get(y_name) or {})
        trace_plot_type = style.get("plot_type") or plot_type or "lines"
        line_style = style.get("line_style") or "solid"
        marker_style = style.get("marker_style") or "circle"
        selected_line_color = style.get("line_color") or "auto"
        x_plot = (
            x_data if x_data is not None and len(x_data) == len(y_data)
            else np.arange(len(y_data))
        )
        auto_color = colors[index % len(colors)]
        resolved_color = auto_color if selected_line_color == "auto" else str(selected_line_color)
        if trace_plot_type == "histogram":
            fig.add_trace(go.Histogram(x=y_data, name=y_name, marker_color=resolved_color, opacity=0.75))
        elif trace_plot_type == "bar":
            fig.add_trace(go.Bar(x=x_plot, y=y_data, name=y_name, marker_color=resolved_color))
        else:
            fig.add_trace(go.Scatter(
                x=x_plot, y=y_data,
                mode=trace_plot_type or "lines",
                name=y_name,
                line={"color": resolved_color, "width": line_width, "dash": line_style},
                marker={"color": resolved_color, "size": marker_size, "symbol": marker_style},
            ))
    _apply_scientific_layout(
        fig,
        title=title or None,
        xaxis_title=x_title or (x_var or "index"),
        yaxis_title=y_title or (y_vars[0] if len(y_vars) == 1 else "value"),
        showlegend=len(y_vars) > 1,
        style_options=resolved,
        barmode="group" if plot_type == "bar" else None,
    )
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
    fig = go.Figure()
    resolved = dict(_default_scientific_style_options())
    if style_options:
        resolved.update(style_options)
    line_width = int(resolved.get("line_width", 2))
    marker_size = int(resolved.get("marker_size", 7))
    font_size = int(resolved.get("font_size", 16))
    matrix = arrays_2d.get(matrix_var) if matrix_var else None
    if matrix is None:
        fig.add_annotation(
            text="Select a 2D array to plot its evolution.",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font={"size": font_size, "color": "#94a3b8"},
        )
        return fig
    rows, cols = matrix.shape
    clamped_step = min(max(0, step_index), max(rows - 1, 0))
    time_axis = arrays_1d.get(time_var) if isinstance(time_var, str) and time_var else None
    if time_axis is None or len(time_axis) != rows:
        time_axis = np.arange(rows, dtype=float)
    value_axis = arrays_1d.get(value_var) if isinstance(value_var, str) and value_var else None
    if value_axis is None or len(value_axis) != cols:
        value_axis = np.arange(cols, dtype=float)
    y_data = matrix[clamped_step]
    time_value = time_axis[clamped_step] if len(time_axis) > clamped_step else clamped_step
    trace_name = f"{matrix_var or 'matrix'} @ step {clamped_step}"
    if plot_type == "bar":
        fig.add_trace(go.Bar(x=value_axis, y=y_data, name=trace_name, marker_color="#1f77b4"))
    else:
        fig.add_trace(go.Scatter(
            x=value_axis, y=y_data,
            mode=plot_type if plot_type not in {"histogram"} else "lines",
            name=trace_name,
            line={"color": "#1f77b4", "width": line_width},
            marker={"color": "#1f77b4", "size": marker_size},
        ))
    _apply_scientific_layout(
        fig,
        title=title or f"{matrix_var or '2D array'} evolution at t={time_value}",
        xaxis_title=x_title or (value_var or "value index"),
        yaxis_title=y_title or (matrix_var or "value"),
        showlegend=False,
        style_options=resolved,
    )
    return fig


# ── GraphBuilderCard ─────────────────────────────────────────────────────────

class GraphBuilderCard(QWidget):
    """Full graph builder card: identical settings panel (left) + plot (right)."""

    remove_requested: Signal = Signal(object)

    _CARD_SS = """
        QWidget#graphBuilderCard {
            background: #ffffff;
            border: 1px solid #d1dce8;
        }
        QWidget#graphBuilderCard QLabel {
            background: transparent;
            border: none;
        }
        QWidget#graphBuilderCard QComboBox,
        QWidget#graphBuilderCard QLineEdit,
        QWidget#graphBuilderCard QListWidget {
            background: #ffffff;
            border: 1px solid #d1dce8;
            border-radius: 6px;
            padding: 4px 6px;
        }
        QWidget#graphBuilderCard QListView {
            background: #ffffff;
            border: 1px solid #d1dce8;
        }
        QWidget#graphBuilderCard QComboBox QAbstractItemView {
            selection-background-color: #c7def5;
            selection-color: #0f1b2b;
            outline: 0;
        }
        QWidget#graphBuilderCard QComboBox QAbstractItemView::item:hover {
            background: #c7def5;
            color: #0f1b2b;
        }
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        show_remove: bool = False,
        show_analysis: bool = False,
    ) -> None:
        super().__init__(parent)
        print(f"[debug][graph-builder-card] init:start show_remove={show_remove}", flush=True)
        self._arrays_1d: dict[str, np.ndarray] = {}
        self._arrays_2d: dict[str, np.ndarray] = {}
        self._notebook_arrays_1d: dict[str, np.ndarray] = {}
        self._notebook_arrays_2d: dict[str, np.ndarray] = {}
        self._csv_arrays_1d: dict[str, np.ndarray] = {}
        self._data_source: str = "notebook"
        self._current_figure = build_notebook_plot_figure({}, None, [], "lines", "", "", "")
        self._series_style_widgets: dict[str, dict[str, Any]] = {}
        self._latest_plot_title = ""
        self._latest_plot_html = ""
        self._card_title = "Notebook Plot Builder"

        self.setObjectName("graphBuilderCard")
        self.setStyleSheet(self._CARD_SS)

        card_layout = QHBoxLayout(self)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(12)

        # ── LEFT: settings panel ─────────────────────────────────────
        self.settings_panel = QWidget(self)
        self.settings_panel.setStyleSheet("background:#ffffff;")
        self.settings_panel.setMinimumWidth(620 if show_analysis else 380)
        self.settings_panel.setMaximumWidth(760 if show_analysis else 480)
        settings_layout = QVBoxLayout(self.settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)

        hdr_row = QHBoxLayout()
        hdr_label = QLabel("Notebook Plot Builder", self.settings_panel)
        hdr_label.setStyleSheet("color:#001f41; font-weight:700; font-size:13px;")
        self.title_label = hdr_label
        hdr_row.addWidget(hdr_label)
        hdr_row.addStretch(1)
        if show_remove:
            rm_btn = QLabel("✕", self.settings_panel)
            rm_btn.setStyleSheet(
                "color:#94a3b8; font-size:16px; border:none; padding:0 6px; border-radius:4px;"
            )
            rm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            rm_btn.mousePressEvent = lambda _e: self.remove_requested.emit(self)  # type: ignore[assignment]
            self.remove_label = rm_btn
            hdr_row.addWidget(rm_btn)
        else:
            self.remove_label = QLabel("", self.settings_panel)
        settings_layout.addLayout(hdr_row)

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        # ── data source ──────────────────────────────────────────────
        src_widget = QWidget(self)
        src_widget.setStyleSheet(
            "QWidget { background:#f0f4f8; border:1px solid #e2e8f0; border-radius:4px; }"
            "QLabel { border:none; background:transparent; }"
            "QRadioButton { border:none; background:transparent; color:#355070; font-weight:600; font-size:12px; }"
            "QPushButton { border:1px solid #cbd5e1; border-radius:4px; padding:2px 8px; font-size:11px; background:#ffffff; }"
        )
        src_layout = QVBoxLayout(src_widget)
        src_layout.setContentsMargins(8, 6, 8, 6)
        src_layout.setSpacing(4)
        src_top = QHBoxLayout()
        src_lbl = QLabel("Data source", src_widget)
        src_lbl.setStyleSheet("color:#355070; font-weight:700; font-size:11px; border:none; background:transparent;")
        self.src_notebook_radio = QRadioButton("Notebook", src_widget)
        self.src_notebook_radio.setChecked(True)
        self.src_csv_radio = QRadioButton("CSV file", src_widget)
        self._src_group = QButtonGroup(src_widget)
        self._src_group.addButton(self.src_notebook_radio)
        self._src_group.addButton(self.src_csv_radio)
        src_top.addWidget(src_lbl)
        src_top.addWidget(self.src_notebook_radio)
        src_top.addWidget(self.src_csv_radio)
        src_top.addStretch(1)
        src_layout.addLayout(src_top)
        self._csv_file_widget = QWidget(src_widget)
        self._csv_file_widget.setStyleSheet("QWidget { background:transparent; border:none; }")
        csv_row = QHBoxLayout(self._csv_file_widget)
        csv_row.setContentsMargins(0, 0, 0, 0)
        csv_row.setSpacing(6)
        self._csv_path_label = QLabel("No file selected", self._csv_file_widget)
        self._csv_path_label.setStyleSheet("color:#64748b; font-size:11px; border:none; background:transparent;")
        csv_open_btn = QPushButton("Open…", self._csv_file_widget)
        csv_open_btn.clicked.connect(self._open_csv_file)
        csv_row.addWidget(self._csv_path_label, 1)
        csv_row.addWidget(csv_open_btn)
        self._csv_file_widget.hide()
        src_layout.addWidget(self._csv_file_widget)
        controls_layout.addWidget(src_widget)
        if not show_analysis:
            src_widget.hide()

        # mode + graph size
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(8)
        mode_lbl = QLabel("Mode", self)
        mode_lbl.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        self.mode_combo = AutoCloseComboBox(self)
        self.mode_combo.addItem("Series (1D)", "series")
        self.mode_combo.addItem("Evolution (2D)", "evolution")
        size_lbl = QLabel("Graph size", self)
        size_lbl.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        self.graph_size_combo = AutoCloseComboBox(self)
        for lbl, val in GRAPH_SIZE_OPTIONS:
            self.graph_size_combo.addItem(lbl, val)
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self.mode_combo, 1)
        mode_row.addWidget(size_lbl)
        mode_row.addWidget(self.graph_size_combo, 1)
        controls_layout.addLayout(mode_row)

        # series controls
        self.series_controls = QWidget(self)
        series_layout = QHBoxLayout(self.series_controls)
        series_layout.setContentsMargins(0, 0, 0, 0)
        series_layout.setSpacing(8)

        x_block = QWidget(self.series_controls)
        x_bl = QVBoxLayout(x_block)
        x_bl.setContentsMargins(0, 0, 0, 0)
        x_bl.setSpacing(4)
        self.x_combo = AutoCloseComboBox(self)
        self.x_combo.addItem("Index", "")
        x_lbl = QLabel("X variable", x_block)
        x_lbl.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        x_bl.addWidget(x_lbl)
        x_bl.addWidget(self.x_combo)
        series_layout.addWidget(x_block, 1)

        pt_block = QWidget(self.series_controls)
        pt_bl = QVBoxLayout(pt_block)
        pt_bl.setContentsMargins(0, 0, 0, 0)
        pt_bl.setSpacing(4)
        self.plot_type_combo = AutoCloseComboBox(self)
        for lbl, val in PLOT_TYPE_OPTIONS:
            self.plot_type_combo.addItem(lbl, val)
        pt_lbl = QLabel("Plot type", pt_block)
        pt_lbl.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        pt_bl.addWidget(pt_lbl)
        pt_bl.addWidget(self.plot_type_combo)
        series_layout.addWidget(pt_block, 1)

        y_block = QWidget(self.series_controls)
        y_bl = QVBoxLayout(y_block)
        y_bl.setContentsMargins(0, 0, 0, 0)
        y_bl.setSpacing(4)
        self.y_combo = CheckableComboBox(self)
        y_lbl = QLabel("Y variable(s)", y_block)
        y_lbl.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
        y_bl.addWidget(y_lbl)
        y_bl.addWidget(self.y_combo)
        series_layout.addWidget(y_block, 2)
        controls_layout.addWidget(self.series_controls)

        # per-series style card
        self.series_style_card = QWidget(self)
        self.series_style_card.setStyleSheet("background:#f8fbfe; border:1px solid #dbe5ef;")
        ssl = QVBoxLayout(self.series_style_card)
        ssl.setContentsMargins(8, 8, 8, 8)
        ssl.setSpacing(6)
        ssl_title = QLabel("Per-Series Styles", self.series_style_card)
        ssl_title.setStyleSheet("color:#001f41; font-weight:700; font-size:12px;")
        ssl.addWidget(ssl_title)
        self.series_style_grid = QGridLayout()
        self.series_style_grid.setContentsMargins(0, 0, 0, 0)
        self.series_style_grid.setHorizontalSpacing(8)
        self.series_style_grid.setVerticalSpacing(6)
        ssl.addLayout(self.series_style_grid)
        controls_layout.addWidget(self.series_style_card)

        # evolution controls
        self.evolution_controls = QWidget(self)
        evo_layout = QFormLayout(self.evolution_controls)
        evo_layout.setContentsMargins(0, 0, 0, 0)
        evo_layout.setSpacing(8)
        self.evolution_matrix_combo = AutoCloseComboBox(self)
        self.evolution_matrix_combo.addItem("Select 2D array", "")
        evo_layout.addRow("Evolution array", self.evolution_matrix_combo)
        self.evolution_time_combo = AutoCloseComboBox(self)
        self.evolution_time_combo.addItem("Row index", "")
        evo_layout.addRow("Time axis", self.evolution_time_combo)
        self.evolution_value_combo = AutoCloseComboBox(self)
        self.evolution_value_combo.addItem("Column index", "")
        evo_layout.addRow("Value axis", self.evolution_value_combo)
        self.evolution_step_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.evolution_step_slider.setRange(0, 0)
        evo_layout.addRow("Time step", self.evolution_step_slider)
        self.evolution_step_label = QLabel("Step 0 / 0", self)
        evo_layout.addRow("Selected step", self.evolution_step_label)
        controls_layout.addWidget(self.evolution_controls)

        # plot style
        self.style_controls = QWidget(self)
        scl = QVBoxLayout(self.style_controls)
        scl.setContentsMargins(0, 0, 0, 0)
        scl.setSpacing(6)
        style_title = QLabel("Plot Style", self.style_controls)
        style_title.setStyleSheet("color:#001f41; font-weight:700; font-size:12px;")
        scl.addWidget(style_title)

        style_top = QHBoxLayout()
        style_top.setContentsMargins(0, 0, 0, 0)
        style_top.setSpacing(8)
        self.font_size_combo = AutoCloseComboBox(self)
        for v in FONT_SIZE_OPTIONS:
            self.font_size_combo.addItem(str(v), v)
        self.font_size_combo.setCurrentIndex(self.font_size_combo.findData(16))
        self.line_width_combo = AutoCloseComboBox(self)
        for v in LINE_WIDTH_OPTIONS:
            self.line_width_combo.addItem(str(v), v)
        self.line_width_combo.setCurrentIndex(self.line_width_combo.findData(2))
        self.marker_size_combo = AutoCloseComboBox(self)
        for v in MARKER_SIZE_OPTIONS:
            self.marker_size_combo.addItem(str(v), v)
        mk_idx = self.marker_size_combo.findData(8)
        self.marker_size_combo.setCurrentIndex(max(mk_idx, 0))
        for lbl_text, widget in (
            ("Font size", self.font_size_combo),
            ("Line width", self.line_width_combo),
            ("Marker size", self.marker_size_combo),
        ):
            blk = QWidget(self.style_controls)
            bll = QVBoxLayout(blk)
            bll.setContentsMargins(0, 0, 0, 0)
            bll.setSpacing(4)
            blbl = QLabel(lbl_text, blk)
            blbl.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
            bll.addWidget(blbl)
            bll.addWidget(widget)
            style_top.addWidget(blk, 1)
        scl.addLayout(style_top)

        _cb_ss = (
            "QCheckBox { color:#355070; font-weight:600; font-size:12px; }"
            "QCheckBox::indicator { width:14px; height:14px; border:1.5px solid #000000;"
            " border-radius:3px; background:#ffffff; }"
            "QCheckBox::indicator:checked { border:1.5px solid #000000; background:#d8b4fe; }"
        )
        style_bot = QHBoxLayout()
        style_bot.setContentsMargins(0, 0, 0, 0)
        style_bot.setSpacing(12)
        self.show_grid_check = QCheckBox("Grid", self.style_controls)
        self.show_grid_check.setChecked(True)
        self.show_box_check = QCheckBox("Box", self.style_controls)
        self.show_box_check.setChecked(True)
        self.ticks_inside_check = QCheckBox("Ticks inside", self.style_controls)
        self.ticks_inside_check.setChecked(True)
        self.minor_ticks_check = QCheckBox("Minor ticks", self.style_controls)
        self.minor_ticks_check.setChecked(True)
        for cb in (self.show_grid_check, self.show_box_check,
                   self.ticks_inside_check, self.minor_ticks_check):
            cb.setStyleSheet(_cb_ss)
            style_bot.addWidget(cb)
        style_bot.addStretch(1)
        scl.addLayout(style_bot)
        controls_layout.addWidget(self.style_controls)

        # title / axis labels
        shared_labels = QWidget(self)
        shl = QHBoxLayout(shared_labels)
        shl.setContentsMargins(0, 0, 0, 0)
        shl.setSpacing(8)
        self.title_edit = QLineEdit(self)
        self.x_label_edit = QLineEdit(self)
        self.y_label_edit = QLineEdit(self)
        for lbl_text, widget in (
            ("Title", self.title_edit),
            ("X label", self.x_label_edit),
            ("Y label", self.y_label_edit),
        ):
            sec = QWidget(shared_labels)
            secl = QVBoxLayout(sec)
            secl.setContentsMargins(0, 0, 0, 0)
            secl.setSpacing(4)
            seclbl = QLabel(lbl_text, sec)
            seclbl.setStyleSheet("color:#355070; font-weight:600; font-size:12px;")
            secl.addWidget(seclbl)
            secl.addWidget(widget)
            shl.addWidget(sec, 1)
        controls_layout.addWidget(shared_labels)

        # ── analysis section ──────────────────────────────────────────
        _cb_ss2 = (
            "QCheckBox { color:#355070; font-weight:600; font-size:12px; }"
            "QCheckBox::indicator { width:13px; height:13px; border:1.5px solid #000; border-radius:3px; background:#fff; }"
            "QCheckBox::indicator:checked { border:1.5px solid #000; background:#d8b4fe; }"
        )
        analysis_widget = QWidget(self)
        analysis_widget.setStyleSheet(
            "QWidget { background:#f8fbfe; border:1px solid #dbe5ef; border-radius:4px; }"
            "QLabel { border:none; background:transparent; color:#355070; font-weight:600; font-size:11px; }"
            "QLineEdit { background:#ffffff; border:1px solid #d1dce8; border-radius:4px; padding:2px 5px; font-size:11px; }"
            "QSpinBox { background:#ffffff; border:1px solid #d1dce8; border-radius:4px; padding:1px 3px; font-size:11px; }"
        )
        ana_layout = QVBoxLayout(analysis_widget)
        ana_layout.setContentsMargins(8, 6, 8, 6)
        ana_layout.setSpacing(6)
        ana_title = QLabel("Analysis", analysis_widget)
        ana_title.setStyleSheet("color:#001f41; font-weight:700; font-size:12px; border:none; background:transparent;")
        ana_layout.addWidget(ana_title)

        # smooth row
        smooth_row = QHBoxLayout()
        smooth_row.setSpacing(6)
        self.smooth_check = QCheckBox("Smooth", analysis_widget)
        self.smooth_check.setStyleSheet(_cb_ss2)
        smooth_win_lbl = QLabel("Window", analysis_widget)
        self.smooth_window = QSpinBox(analysis_widget)
        self.smooth_window.setRange(3, 999)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.setValue(5)
        smooth_poly_lbl = QLabel("Poly", analysis_widget)
        self.smooth_poly = QSpinBox(analysis_widget)
        self.smooth_poly.setRange(1, 5)
        self.smooth_poly.setValue(2)
        smooth_row.addWidget(self.smooth_check)
        smooth_row.addStretch(1)
        smooth_row.addWidget(smooth_win_lbl)
        smooth_row.addWidget(self.smooth_window)
        smooth_row.addWidget(smooth_poly_lbl)
        smooth_row.addWidget(self.smooth_poly)
        ana_layout.addLayout(smooth_row)

        # derivative row
        deriv_row = QHBoxLayout()
        deriv_row.setSpacing(6)
        self.deriv_check = QCheckBox("Derivative  dy/dx", analysis_widget)
        self.deriv_check.setStyleSheet(_cb_ss2)
        deriv_row.addWidget(self.deriv_check)
        deriv_row.addStretch(1)
        ana_layout.addLayout(deriv_row)

        # fit rows
        fit_row1 = QHBoxLayout()
        fit_row1.setSpacing(6)
        self.fit_check = QCheckBox("Curve Fit", analysis_widget)
        self.fit_check.setStyleSheet(_cb_ss2)
        self.fit_model_combo = AutoCloseComboBox(analysis_widget)
        for _key, info in BUILTIN_MODELS.items():
            self.fit_model_combo.addItem(info["label"], _key)
        self.fit_model_combo.addItem("Custom formula", "custom")
        fit_row1.addWidget(self.fit_check)
        fit_row1.addWidget(self.fit_model_combo, 1)
        ana_layout.addLayout(fit_row1)

        self._fit_custom_widget = QWidget(analysis_widget)
        self._fit_custom_widget.setStyleSheet("QWidget { background:transparent; border:none; }")
        fit_custom_row = QHBoxLayout(self._fit_custom_widget)
        fit_custom_row.setContentsMargins(0, 0, 0, 0)
        fit_custom_lbl = QLabel("Formula", self._fit_custom_widget)
        self.fit_custom_edit = QLineEdit(self._fit_custom_widget)
        self.fit_custom_edit.setPlaceholderText("e.g. a*exp(-b*x) + c")
        fit_custom_row.addWidget(fit_custom_lbl)
        fit_custom_row.addWidget(self.fit_custom_edit, 1)
        self._fit_custom_widget.hide()
        ana_layout.addWidget(self._fit_custom_widget)

        fit_row2 = QHBoxLayout()
        fit_row2.setSpacing(6)
        fit_xmin_lbl = QLabel("X min", analysis_widget)
        self.fit_xmin_edit = QLineEdit(analysis_widget)
        self.fit_xmin_edit.setPlaceholderText("auto")
        self.fit_xmin_edit.setMaximumWidth(70)
        fit_xmax_lbl = QLabel("X max", analysis_widget)
        self.fit_xmax_edit = QLineEdit(analysis_widget)
        self.fit_xmax_edit.setPlaceholderText("auto")
        self.fit_xmax_edit.setMaximumWidth(70)
        fit_row2.addWidget(fit_xmin_lbl)
        fit_row2.addWidget(self.fit_xmin_edit)
        fit_row2.addWidget(fit_xmax_lbl)
        fit_row2.addWidget(self.fit_xmax_edit)
        fit_row2.addStretch(1)
        ana_layout.addLayout(fit_row2)

        controls_layout.addWidget(analysis_widget)
        if not show_analysis:
            analysis_widget.hide()

        settings_layout.addLayout(controls_layout)
        settings_layout.addStretch(1)

        # ── RIGHT: plot preview ───────────────────────────────────────
        self.preview_panel = QWidget(self)
        pvl = QVBoxLayout(self.preview_panel)
        pvl.setContentsMargins(0, 0, 0, 0)
        pvl.setSpacing(8)
        self.controller_status = QLabel("Waiting for notebook arrays", self.preview_panel)
        self.controller_status.setStyleSheet("color:#64748b; font-size:12px;")
        pvl.addWidget(self.controller_status)
        self.controller_plot = PlotView(self.preview_panel)
        w0, h0 = GRAPH_SIZE_OPTIONS[0][1]
        self.controller_plot.setMinimumHeight(h0)
        self.controller_plot.setMaximumHeight(16777215)
        pvl.addWidget(self.controller_plot, 1)
        self.status_label = self.controller_status
        self.plot_view = self.controller_plot

        card_layout.addWidget(self.settings_panel, 0)
        card_layout.addWidget(self.preview_panel, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # signals
        self.mode_combo.currentIndexChanged.connect(self._sync_mode_ui)
        self.x_combo.currentIndexChanged.connect(self.refresh)
        self.y_combo.checkedItemsChanged.connect(self._handle_y_selection_changed)
        self.plot_type_combo.currentIndexChanged.connect(self.refresh)
        self.evolution_matrix_combo.currentIndexChanged.connect(self._on_evolution_matrix_changed)
        self.evolution_time_combo.currentIndexChanged.connect(self.refresh)
        self.evolution_value_combo.currentIndexChanged.connect(self.refresh)
        self.evolution_step_slider.valueChanged.connect(self._on_evolution_step_changed)
        self.title_edit.textChanged.connect(self.refresh)
        self.x_label_edit.textChanged.connect(self.refresh)
        self.y_label_edit.textChanged.connect(self.refresh)
        self.font_size_combo.currentIndexChanged.connect(self.refresh)
        self.line_width_combo.currentIndexChanged.connect(self.refresh)
        self.marker_size_combo.currentIndexChanged.connect(self.refresh)
        self.graph_size_combo.currentIndexChanged.connect(self._on_size_changed)
        self.show_grid_check.toggled.connect(self.refresh)
        self.show_box_check.toggled.connect(self.refresh)
        self.ticks_inside_check.toggled.connect(self.refresh)
        self.minor_ticks_check.toggled.connect(self.refresh)

        self._src_group.buttonToggled.connect(self._on_source_changed)
        self.smooth_check.toggled.connect(self.refresh)
        self.smooth_window.valueChanged.connect(self.refresh)
        self.smooth_poly.valueChanged.connect(self.refresh)
        self.deriv_check.toggled.connect(self.refresh)
        self.fit_check.toggled.connect(self.refresh)
        self.fit_model_combo.currentIndexChanged.connect(self._on_fit_model_changed)
        self.fit_model_combo.currentIndexChanged.connect(self.refresh)
        self.fit_custom_edit.textChanged.connect(self.refresh)
        self.fit_xmin_edit.textChanged.connect(self.refresh)
        self.fit_xmax_edit.textChanged.connect(self.refresh)

        self._sync_mode_ui()
        self.refresh()
        print("[debug][graph-builder-card] init:done", flush=True)

    # ── public API ───────────────────────────────────────────────────

    def set_namespace(
        self, arrays_1d: dict[str, np.ndarray], arrays_2d: dict[str, np.ndarray]
    ) -> None:
        print(
            f"[debug][graph-builder-card] set_namespace:start title={self._card_title!r} "
            f"count_1d={len(arrays_1d)} count_2d={len(arrays_2d)}",
            flush=True,
        )
        prev_x = self.x_combo.currentData()
        prev_y = set(self._selected_y_vars())
        prev_matrix = self.evolution_matrix_combo.currentData()
        prev_time = self.evolution_time_combo.currentData()
        prev_value = self.evolution_value_combo.currentData()
        prev_mode = self.mode_combo.currentData()
        self._notebook_arrays_1d = arrays_1d
        self._notebook_arrays_2d = arrays_2d
        if self._data_source == "notebook":
            self._arrays_1d = arrays_1d
            self._arrays_2d = arrays_2d
        preferred_mode = "series"
        if arrays_2d and not arrays_1d:
            preferred_mode = "evolution"
        elif prev_mode == "evolution" and arrays_2d:
            preferred_mode = "evolution"
        print(f"[debug][graph-builder-card] set_namespace:preferred_mode mode={preferred_mode!r}", flush=True)
        self.mode_combo.blockSignals(True)
        mode_idx = self.mode_combo.findData(preferred_mode)
        if mode_idx >= 0:
            self.mode_combo.setCurrentIndex(mode_idx)
        self.mode_combo.blockSignals(False)
        for combo in (self.x_combo, self.y_combo, self.evolution_matrix_combo,
                      self.evolution_time_combo, self.evolution_value_combo):
            combo.blockSignals(True)
        self.x_combo.clear()
        self.x_combo.addItem("Index", "")
        self.y_combo.clear()
        self.evolution_matrix_combo.clear()
        self.evolution_matrix_combo.addItem("Select 2D array", "")
        self.evolution_time_combo.clear()
        self.evolution_time_combo.addItem("Row index", "")
        self.evolution_value_combo.clear()
        self.evolution_value_combo.addItem("Column index", "")
        for name in sorted(arrays_1d):
            self.x_combo.addItem(name, name)
            self.evolution_time_combo.addItem(name, name)
            self.evolution_value_combo.addItem(name, name)
            self.y_combo.add_check_item(name, name, checked=name in prev_y)
            print(f"[debug][graph-builder-card] set_namespace:item name={name!r}", flush=True)
        for name in sorted(arrays_2d):
            self.evolution_matrix_combo.addItem(name, name)
            print(f"[debug][graph-builder-card] set_namespace:matrix_item name={name!r}", flush=True)
        for combo, prev in (
            (self.x_combo, prev_x),
            (self.evolution_matrix_combo, prev_matrix),
            (self.evolution_time_combo, prev_time),
            (self.evolution_value_combo, prev_value),
        ):
            idx = combo.findData(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        if not prev_y and arrays_1d:
            default_names = sorted(arrays_1d)
            x_name = self.x_combo.currentData() or ""
            default_y = default_names[1] if len(default_names) > 1 and x_name == default_names[0] else default_names[0]
            self.y_combo.set_checked_values([default_y])
            print(f"[debug][graph-builder-card] set_namespace:default_y name={default_y!r}", flush=True)
        if preferred_mode == "evolution" and self.evolution_matrix_combo.currentIndex() <= 0 and self.evolution_matrix_combo.count() > 1:
            self.evolution_matrix_combo.setCurrentIndex(1)
            print("[debug][graph-builder-card] set_namespace:default_matrix index=1", flush=True)
        for combo in (self.x_combo, self.y_combo, self.evolution_matrix_combo,
                      self.evolution_time_combo, self.evolution_value_combo):
            combo.blockSignals(False)
        self._on_evolution_matrix_changed()
        self._refresh_series_style_rows()
        status = (
            f"{len(arrays_1d)} 1D array(s), {len(arrays_2d)} 2D array(s) available for plotting"
            if arrays_1d or arrays_2d else "No numeric notebook arrays available yet"
        )
        self.controller_status.setText(status)
        print(f"[debug][graph-builder-card] set_namespace:status text={status!r}", flush=True)
        self.refresh()

    def current_figure(self) -> go.Figure:
        print(f"[debug][graph-builder-card] current_figure traces={len(self._current_figure.data)}", flush=True)
        return self._current_figure

    def set_latest_plot(self, title: str, html: str) -> None:
        self._latest_plot_title = title
        self._latest_plot_html = html
        print(
            f"[debug][graph-builder-card] set_latest_plot title={title!r} html_length={len(html)}",
            flush=True,
        )
        self.refresh()

    def set_card_title(self, title: str) -> None:
        self._card_title = title
        self.title_label.setText(title)
        print(f"[debug][graph-builder-card] set_card_title title={title!r}", flush=True)

    def set_remove_enabled(self, enabled: bool) -> None:
        self.remove_label.setVisible(enabled)
        print(f"[debug][graph-builder-card] set_remove_enabled enabled={enabled}", flush=True)

    def refresh_plot(self) -> None:
        print("[debug][graph-builder-card] refresh_plot alias", flush=True)
        self.refresh()

    # ── internal ─────────────────────────────────────────────────────

    def _selected_y_vars(self) -> list[str]:
        selected = [v for v in self.y_combo.checked_values() if isinstance(v, str)]
        print(f"[debug][graph-builder-card] selected_y_vars selected={selected!r}", flush=True)
        return selected

    def _handle_y_selection_changed(self) -> None:
        print("[debug][graph-builder-card] handle_y_selection_changed", flush=True)
        self._refresh_series_style_rows()
        self.refresh()

    def _clear_series_style_rows(self) -> None:
        print(f"[debug][graph-builder-card] clear_series_style_rows count={len(self._series_style_widgets)}", flush=True)
        while self.series_style_grid.count():
            item = self.series_style_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
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
            f"[debug][graph-builder-card] refresh_series_style_rows selected_y={selected_y!r} "
            f"previous_values={previous_values!r}",
            flush=True,
        )
        self._clear_series_style_rows()
        self.series_style_card.setVisible(bool(selected_y))
        if not selected_y:
            return
        for col, header in enumerate(("Y variable", "Plot type", "Line style", "Marker style", "Line color")):
            lbl = QLabel(header, self.series_style_card)
            lbl.setStyleSheet("color:#355070; font-weight:700; font-size:11px;")
            self.series_style_grid.addWidget(lbl, 0, col)
        for row, name in enumerate(selected_y, start=1):
            name_lbl = QLabel(name, self.series_style_card)
            name_lbl.setStyleSheet("color:#0f1b2b; font-weight:600; font-size:12px;")
            pt = AutoCloseComboBox(self.series_style_card)
            for lbl, val in PLOT_TYPE_OPTIONS:
                pt.addItem(lbl, val)
            ls = AutoCloseComboBox(self.series_style_card)
            for lbl, val in LINE_STYLE_OPTIONS:
                ls.addItem(lbl, val)
            ms = AutoCloseComboBox(self.series_style_card)
            for lbl, val in MARKER_STYLE_OPTIONS:
                ms.addItem(lbl, val)
            lc = AutoCloseComboBox(self.series_style_card)
            for lbl, val in LINE_COLOR_OPTIONS:
                lc.addItem(lbl, val)
            prev = previous_values.get(name, {})
            for combo, key, default in (
                (pt, "plot_type", self.plot_type_combo.currentData() or "lines"),
                (ls, "line_style", "solid"),
                (ms, "marker_style", "circle"),
                (lc, "line_color", "auto"),
            ):
                idx = combo.findData(prev.get(key) or default)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                combo.currentIndexChanged.connect(self.refresh)
            self.series_style_grid.addWidget(name_lbl, row, 0)
            self.series_style_grid.addWidget(pt, row, 1)
            self.series_style_grid.addWidget(ls, row, 2)
            self.series_style_grid.addWidget(ms, row, 3)
            self.series_style_grid.addWidget(lc, row, 4)
            self._series_style_widgets[name] = {
                "plot_type": pt, "line_style": ls,
                "marker_style": ms, "line_color": lc,
            }
            print(f"[debug][graph-builder-card] refresh_series_style_rows:item name={name!r}", flush=True)

    def _series_style_map(self) -> dict[str, dict[str, str]]:
        style_map = {
            name: {
                "plot_type": str(w["plot_type"].currentData() or "lines"),
                "line_style": str(w["line_style"].currentData() or "solid"),
                "marker_style": str(w["marker_style"].currentData() or "circle"),
                "line_color": str(w["line_color"].currentData() or "auto"),
            }
            for name, w in self._series_style_widgets.items()
        }
        print(f"[debug][graph-builder-card] series_style_map style_map={style_map!r}", flush=True)
        return style_map

    def _style_options(self) -> dict[str, Any]:
        size = self.graph_size_combo.currentData()
        _w, h = size if isinstance(size, tuple) else (800, 600)
        options = {
            "font_size": int(self.font_size_combo.currentData() or 16),
            "line_width": int(self.line_width_combo.currentData() or 2),
            "marker_size": int(self.marker_size_combo.currentData() or 8),
            "show_grid": self.show_grid_check.isChecked(),
            "show_box": self.show_box_check.isChecked(),
            "ticks_inside": self.ticks_inside_check.isChecked(),
            "show_minor_ticks": self.minor_ticks_check.isChecked(),
            "graph_width": None,
            "graph_height": h,
        }
        print(f"[debug][graph-builder-card] style_options options={options!r}", flush=True)
        return options

    def _sync_mode_ui(self) -> None:
        mode = self.mode_combo.currentData() or "series"
        is_series = mode == "series"
        print(f"[debug][graph-builder-card] sync_mode_ui mode={mode!r}", flush=True)
        self.series_controls.setVisible(is_series)
        self.series_style_card.setVisible(is_series and bool(self._selected_y_vars()))
        self.evolution_controls.setVisible(not is_series)
        self.refresh()

    def _on_evolution_matrix_changed(self) -> None:
        matrix_name = self.evolution_matrix_combo.currentData()
        print(f"[debug][graph-builder-card] evolution_matrix_changed matrix_name={matrix_name!r}", flush=True)
        matrix = (
            self._arrays_2d.get(matrix_name)
            if isinstance(matrix_name, str) and matrix_name else None
        )
        rows = int(matrix.shape[0]) if matrix is not None else 0
        max_index = max(rows - 1, 0)
        self.evolution_step_slider.blockSignals(True)
        self.evolution_step_slider.setRange(0, max_index)
        self.evolution_step_slider.setValue(min(self.evolution_step_slider.value(), max_index))
        self.evolution_step_slider.blockSignals(False)
        self._update_evolution_step_label()
        self.refresh()

    def _on_evolution_step_changed(self, _value: int) -> None:
        print(f"[debug][graph-builder-card] evolution_step_changed value={self.evolution_step_slider.value()}", flush=True)
        self._update_evolution_step_label()
        self.refresh()

    def _update_evolution_step_label(self) -> None:
        cur = self.evolution_step_slider.value()
        tot = self.evolution_step_slider.maximum() + 1
        label = f"Step {cur} / {max(tot - 1, 0)}"
        self.evolution_step_label.setText(label)
        print(f"[debug][graph-builder-card] evolution_step_label label={label!r}", flush=True)

    def _on_size_changed(self) -> None:
        size = self.graph_size_combo.currentData()
        if isinstance(size, tuple):
            self.controller_plot.setMinimumHeight(size[1])
            print(f"[debug][graph-builder-card] on_size_changed height={size[1]}", flush=True)
        self.refresh()

    # ── data source ──────────────────────────────────────────────────

    def _on_source_changed(self, _btn: Any, checked: bool) -> None:
        if not checked:
            return
        self._data_source = "csv" if self.src_csv_radio.isChecked() else "notebook"
        self._csv_file_widget.setVisible(self._data_source == "csv")
        if self._data_source == "notebook":
            self._arrays_1d = self._notebook_arrays_1d
            self._arrays_2d = self._notebook_arrays_2d
            self.mode_combo.setEnabled(True)
        else:
            self._arrays_1d = self._csv_arrays_1d
            self._arrays_2d = {}
            self.mode_combo.blockSignals(True)
            idx = self.mode_combo.findData("series")
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)
            self.mode_combo.setEnabled(False)
        self._repopulate_variable_combos()
        self.refresh()

    def _repopulate_variable_combos(self) -> None:
        arrays_1d = self._arrays_1d
        prev_x = self.x_combo.currentData()
        prev_y = set(self._selected_y_vars())
        for combo in (self.x_combo, self.y_combo):
            combo.blockSignals(True)
        self.x_combo.clear()
        self.x_combo.addItem("Index", "")
        self.y_combo.clear()
        for name in sorted(arrays_1d):
            self.x_combo.addItem(name, name)
            self.y_combo.add_check_item(name, name, checked=name in prev_y)
        idx = self.x_combo.findData(prev_x)
        if idx >= 0:
            self.x_combo.setCurrentIndex(idx)
        for combo in (self.x_combo, self.y_combo):
            combo.blockSignals(False)

    def _open_csv_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open data file", "",
            "Data files (*.csv *.txt *.dat *.tsv);;All files (*)",
        )
        if not path:
            return
        try:
            df = pd.read_csv(path, sep=None, engine="python")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self._csv_arrays_1d = {
                col: df[col].dropna().to_numpy(dtype=float)
                for col in numeric_cols
            }
            self._csv_path_label.setText(os.path.basename(path))
            self.controller_status.setText(
                f"Loaded {os.path.basename(path)} — {len(numeric_cols)} numeric column(s): "
                f"{', '.join(numeric_cols)}"
            )
            if self._data_source == "csv":
                self._arrays_1d = self._csv_arrays_1d
                self._repopulate_variable_combos()
                self.refresh()
        except Exception as exc:
            self.controller_status.setText(f"CSV load error: {exc}")

    # ── fit model ────────────────────────────────────────────────────

    def _on_fit_model_changed(self) -> None:
        self._fit_custom_widget.setVisible(self.fit_model_combo.currentData() == "custom")

    # ── analysis overlays ────────────────────────────────────────────

    def _apply_analysis_overlays(
        self,
        fig: go.Figure,
        arrays_1d: dict[str, np.ndarray],
        x_var: str | None,
        y_vars: list[str],
    ) -> str:
        status_parts: list[str] = []
        for y_name in y_vars:
            y_data = arrays_1d.get(y_name)
            if y_data is None or len(y_data) == 0:
                continue
            x_data = arrays_1d.get(x_var) if x_var else None
            x = (
                x_data
                if x_data is not None and len(x_data) == len(y_data)
                else np.arange(len(y_data), dtype=float)
            )
            # smooth
            if self.smooth_check.isChecked():
                try:
                    from scipy.signal import savgol_filter
                    window = self.smooth_window.value()
                    poly = self.smooth_poly.value()
                    if len(y_data) > window and window > poly:
                        y_smooth = savgol_filter(y_data, window, poly)
                        fig.add_trace(go.Scatter(
                            x=x, y=y_smooth, mode="lines",
                            name=f"{y_name} (smooth)",
                            line={"dash": "dot", "width": 1.5},
                            opacity=0.85,
                        ))
                except Exception as exc:
                    status_parts.append(f"Smooth: {exc}")
            # derivative
            if self.deriv_check.isChecked():
                try:
                    dy = np.gradient(y_data, x)
                    fig.add_trace(go.Scatter(
                        x=x, y=dy, mode="lines",
                        name=f"d({y_name})/dx",
                        line={"dash": "dashdot", "width": 1.5},
                        opacity=0.85,
                    ))
                except Exception as exc:
                    status_parts.append(f"Derivative: {exc}")
            # curve fit
            if self.fit_check.isChecked():
                try:
                    model = self.fit_model_combo.currentData() or "linear"
                    custom = self.fit_custom_edit.text().strip() if model == "custom" else ""
                    xmin_text = self.fit_xmin_edit.text().strip()
                    xmax_text = self.fit_xmax_edit.text().strip()
                    x_min = float(xmin_text) if xmin_text else None
                    x_max = float(xmax_text) if xmax_text else None
                    result = fit_series(x, y_data, model, custom_formula=custom,
                                        x_min=x_min, x_max=x_max)
                    if result.get("error"):
                        status_parts.append(f"Fit: {result['error']}")
                    else:
                        fig.add_trace(go.Scatter(
                            x=result["x_curve"], y=result["y_curve"],
                            mode="lines",
                            name=f"{y_name} fit ({model})",
                            line={"dash": "dash", "width": 2},
                        ))
                        params = result.get("params", {})
                        uncerts = result.get("uncertainties", {})
                        if uncerts:
                            pstr = "  ".join(
                                f"{k}={v:.4g}±{uncerts.get(k, 0):.2g}"
                                for k, v in params.items()
                            )
                        else:
                            pstr = "  ".join(f"{k}={v:.4g}" for k, v in params.items())
                        status_parts.append(
                            f"[{y_name}] R²={result['r_squared']:.4f}  "
                            f"RMSE={result['rmse']:.4g}  {pstr}"
                        )
                except (FitError, Exception) as exc:
                    status_parts.append(f"Fit error: {exc}")
        return "  |  ".join(status_parts)

    def refresh(self) -> None:
        mode = self.mode_combo.currentData() or "series"
        y_vars = self._selected_y_vars()
        print(
            f"[debug][graph-builder-card] refresh:start title={self._card_title!r} mode={mode!r} "
            f"x={self.x_combo.currentData()!r} y={y_vars!r} matrix={self.evolution_matrix_combo.currentData()!r}",
            flush=True,
        )
        if mode == "series" and sorted(self._series_style_widgets) != sorted(y_vars):
            print("[debug][graph-builder-card] refresh:style_rows_out_of_sync", flush=True)
            self._refresh_series_style_rows()
        style = self._style_options()
        graph_width = style["graph_width"]
        graph_height = int(style["graph_height"])
        if mode == "evolution":
            self._current_figure = build_notebook_evolution_figure(
                self._arrays_1d, self._arrays_2d,
                self.evolution_matrix_combo.currentData() or None,
                self.evolution_time_combo.currentData() or None,
                self.evolution_value_combo.currentData() or None,
                self.evolution_step_slider.value(),
                self.plot_type_combo.currentData() or "lines",
                self.title_edit.text().strip(),
                self.x_label_edit.text().strip(),
                self.y_label_edit.text().strip(),
                style,
            )
        else:
            self._current_figure = build_notebook_plot_figure(
                self._arrays_1d,
                self.x_combo.currentData() or None,
                y_vars,
                self.plot_type_combo.currentData() or "lines",
                self.title_edit.text().strip(),
                self.x_label_edit.text().strip(),
                self.y_label_edit.text().strip(),
                self._series_style_map(),
                style,
            )
        has_plot = (
            bool(y_vars)
            if mode == "series"
            else bool(self.evolution_matrix_combo.currentData())
            and self.evolution_matrix_combo.currentData() in self._arrays_2d
        )
        use_latest_plot = (not has_plot) and bool(self._latest_plot_html)
        print(
            f"[debug][graph-builder-card] refresh:availability has_plot={has_plot} "
            f"latest_plot={bool(self._latest_plot_html)} use_latest_plot={use_latest_plot}",
            flush=True,
        )
        if use_latest_plot:
            self.controller_plot.set_html(self._latest_plot_html)
            fallback_status = f"Showing latest executed notebook plot: {self._latest_plot_title or 'Latest notebook plot'}"
            self.controller_status.setText(fallback_status)
            print(f"[debug][graph-builder-card] refresh:fallback_latest_plot status={fallback_status!r}", flush=True)
            return
        # analysis overlays (series mode only)
        if mode == "series":
            overlay_status = self._apply_analysis_overlays(
                self._current_figure,
                self._arrays_1d,
                self.x_combo.currentData() or None,
                y_vars,
            )
        else:
            overlay_status = ""
        html = self._current_figure.to_html(
            include_plotlyjs=True,
            full_html=False,
            config={
                "responsive": graph_width is None,
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "graph",
                    "scale": 3,
                    "width": int(graph_width) if graph_width is not None else 800,
                    "height": graph_height,
                },
            },
        )
        self.controller_plot.setMinimumHeight(graph_height)
        print(f"[debug][graph-builder-card] refresh:plot_height height={graph_height}", flush=True)
        print(f"[debug][graph-builder-card] refresh:html length={len(html)}", flush=True)
        self.controller_plot.set_html(html)
        if mode == "evolution":
            status = (
                f"Evolution: matrix={self.evolution_matrix_combo.currentData() or 'none'} "
                f"step={self.evolution_step_slider.value()}"
            )
        else:
            base = (
                f"[{self._data_source.upper()}] "
                f"x={self.x_combo.currentData() or 'index'}  "
                f"y={', '.join(y_vars) if y_vars else 'none'}"
            )
            status = f"{base}  |  {overlay_status}" if overlay_status else base
        self.controller_status.setText(status)
        print(f"[debug][graph-builder-card] refresh:done status={status!r}", flush=True)


# ── NotebookGraphWorkspace ───────────────────────────────────────────────────

class NotebookGraphWorkspace(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][notebook-graph-workspace] init:start", flush=True)
        self._arrays_1d: dict[str, np.ndarray] = {}
        self._arrays_2d: dict[str, np.ndarray] = {}
        self._latest_plot_title = ""
        self._latest_plot_html = ""
        self._cards: list[GraphBuilderCard] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self._root_layout = root

        header = QHBoxLayout()
        title = QLabel("Notebook Graph Workspace", self)
        title.setStyleSheet("color:#001f41; font-weight:700; font-size:15px;")
        header.addWidget(title)
        header.addStretch(1)
        add_btn = QLabel("+ Add Graph", self)
        add_btn.setStyleSheet(
            "color:#1d4ed8; font-weight:600; font-size:13px;"
            "padding:4px 10px; border:1px solid #93c5fd; border-radius:6px; background:#eff6ff;"
        )
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.mousePressEvent = lambda _e: self.add_graph_card()  # type: ignore[assignment]
        self.add_btn = add_btn
        header.addWidget(add_btn)
        root.addLayout(header)
        root.addStretch(1)

        self.add_graph_card()
        print("[debug][notebook-graph-workspace] init:done", flush=True)

    def _renumber_cards(self) -> None:
        print(f"[debug][notebook-graph-workspace] renumber_cards count={len(self._cards)}", flush=True)
        for index, card in enumerate(self._cards, start=1):
            card.set_card_title(f"Graph {index}")
            card.set_remove_enabled(len(self._cards) > 1)

    def primary_card(self) -> GraphBuilderCard:
        print("[debug][notebook-graph-workspace] primary_card", flush=True)
        return self._cards[0]

    def cards(self) -> list[GraphBuilderCard]:
        print(f"[debug][notebook-graph-workspace] cards count={len(self._cards)}", flush=True)
        return list(self._cards)

    def card_count(self) -> int:
        count = len(self._cards)
        print(f"[debug][notebook-graph-workspace] card_count count={count}", flush=True)
        return count

    def add_graph_card(self) -> GraphBuilderCard:
        print("[debug][notebook-graph-workspace] add_graph_card:start", flush=True)
        card = GraphBuilderCard(self, show_remove=bool(self._cards))
        card.remove_requested.connect(self.remove_graph_card)
        card.set_namespace(self._arrays_1d, self._arrays_2d)
        card.set_latest_plot(self._latest_plot_title, self._latest_plot_html)
        self._cards.append(card)
        self._root_layout.insertWidget(max(self._root_layout.count() - 1, 0), card)
        self._renumber_cards()
        print(f"[debug][notebook-graph-workspace] add_graph_card:done total={len(self._cards)}", flush=True)
        return card

    def remove_graph_card(self, card: GraphBuilderCard) -> None:
        print(f"[debug][notebook-graph-workspace] remove_graph_card:start card_title={card.title_label.text()!r}", flush=True)
        if card not in self._cards:
            print("[debug][notebook-graph-workspace] remove_graph_card:skip_missing", flush=True)
            return
        if len(self._cards) <= 1:
            print("[debug][notebook-graph-workspace] remove_graph_card:skip_last_card", flush=True)
            return
        self._cards.remove(card)
        self._root_layout.removeWidget(card)
        card.hide()
        card.deleteLater()
        self._renumber_cards()
        print(f"[debug][notebook-graph-workspace] remove_graph_card:done total={len(self._cards)}", flush=True)

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        print(f"[debug][notebook-graph-workspace] set_namespace:start count={len(namespace)}", flush=True)
        self._arrays_1d, self._arrays_2d = extract_notebook_array_variables(namespace)
        print(
            f"[debug][notebook-graph-workspace] set_namespace:arrays count_1d={len(self._arrays_1d)} "
            f"count_2d={len(self._arrays_2d)}",
            flush=True,
        )
        for index, card in enumerate(self._cards, start=1):
            print(f"[debug][notebook-graph-workspace] set_namespace:card index={index}", flush=True)
            card.set_namespace(self._arrays_1d, self._arrays_2d)

    def set_latest_plot(self, title: str, html: str) -> None:
        self._latest_plot_title = title
        self._latest_plot_html = html
        print(
            f"[debug][notebook-graph-workspace] set_latest_plot title={title!r} html_length={len(html)}",
            flush=True,
        )
        for index, card in enumerate(self._cards, start=1):
            print(f"[debug][notebook-graph-workspace] set_latest_plot:card index={index}", flush=True)
            card.set_latest_plot(title, html)


# ── NotebookPlotPanel ────────────────────────────────────────────────────────

class NotebookPlotPanel(QWidget):
    def __init__(self, parent: QWidget | None = None, layout_mode: str = "advanced") -> None:
        super().__init__(parent)
        self.layout_mode = layout_mode
        self.setStyleSheet("background:#ffffff;")
        self._arrays_1d: dict[str, np.ndarray] = {}
        self._arrays_2d: dict[str, np.ndarray] = {}
        self._output_widgets: dict[str, tuple[QLabel, QWidget, str]] = {}
        self._output_empty_label: QLabel | None = None
        self._extra_cards: list[GraphBuilderCard] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self._root_layout = root

        # header
        header = QHBoxLayout()
        title = QLabel("Graphs", self)
        title.setStyleSheet("color:#001f41; font-weight:700; font-size:15px;")
        header.addWidget(title)
        header.addStretch(1)
        add_btn = QLabel("+ Add Graph", self)
        add_btn.setStyleSheet(
            "color:#1d4ed8; font-weight:600; font-size:13px;"
            "padding:4px 10px; border:1px solid #93c5fd; border-radius:6px;"
            "background:#eff6ff;"
        )
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.mousePressEvent = lambda _e: self._add_extra_graph()  # type: ignore[assignment]
        header.addWidget(add_btn)
        root.addLayout(header)

        # main graph card (no remove button, full analysis features)
        self.main_card = GraphBuilderCard(self, show_remove=False, show_analysis=True)
        root.addWidget(self.main_card)
        self.mode_combo = self.main_card.mode_combo
        self.graph_size_combo = self.main_card.graph_size_combo
        self.series_controls = self.main_card.series_controls
        self.evolution_controls = self.main_card.evolution_controls
        self.x_combo = self.main_card.x_combo
        self.y_combo = self.main_card.y_combo
        self.plot_type_combo = self.main_card.plot_type_combo
        self.evolution_matrix_combo = self.main_card.evolution_matrix_combo
        self.evolution_time_combo = self.main_card.evolution_time_combo
        self.evolution_value_combo = self.main_card.evolution_value_combo
        self.evolution_step_slider = self.main_card.evolution_step_slider
        self.evolution_step_label = self.main_card.evolution_step_label
        self.font_size_combo = self.main_card.font_size_combo
        self.line_width_combo = self.main_card.line_width_combo
        self.marker_size_combo = self.main_card.marker_size_combo
        self.show_grid_check = self.main_card.show_grid_check
        self.show_box_check = self.main_card.show_box_check
        self.ticks_inside_check = self.main_card.ticks_inside_check
        self.minor_ticks_check = self.main_card.minor_ticks_check
        self.title_edit = self.main_card.title_edit
        self.x_label_edit = self.main_card.x_label_edit
        self.y_label_edit = self.main_card.y_label_edit
        self.controller_plot = self.main_card.controller_plot
        self.controller_status = self.main_card.controller_status
        self.series_style_card = self.main_card.series_style_card
        self.series_style_grid = self.main_card.series_style_grid
        self._series_style_widgets = self.main_card._series_style_widgets

        # cell outputs section
        self.outputs_scroll = QScrollArea(self)
        self.outputs_scroll.setWidgetResizable(True)
        self.outputs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.outputs_scroll.setStyleSheet(
            "QScrollArea { background:#ffffff; border:1px solid #d1dce8; }"
        )
        self.outputs_container = QWidget(self.outputs_scroll)
        self.outputs_container.setStyleSheet("background:#ffffff;")
        self.outputs_layout = QVBoxLayout(self.outputs_container)
        self.outputs_layout.setContentsMargins(0, 0, 0, 0)
        self.outputs_layout.setSpacing(8)
        self.outputs_layout.addStretch(1)
        self.outputs_scroll.setWidget(self.outputs_container)
        self.outputs_scroll.hide()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ── public API ───────────────────────────────────────────────────

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        self._arrays_1d, self._arrays_2d = extract_notebook_array_variables(namespace)
        self.main_card.set_namespace(self._arrays_1d, self._arrays_2d)
        for card in self._extra_cards:
            card.set_namespace(self._arrays_1d, self._arrays_2d)

    def current_controller_figure(self) -> go.Figure:
        return self.main_card.current_figure()

    def refresh_controller_plot(self) -> None:
        print("[debug][notebook-plot-panel] refresh_controller_plot:delegate", flush=True)
        self.main_card.refresh()

    def output_count(self) -> int:
        return len(self._output_widgets)

    def output_titles(self) -> list[str]:
        return [t.text() for t, _w, _h in self._output_widgets.values()]

    # ── extra graph management ────────────────────────────────────────

    def _add_extra_graph(self) -> None:
        card = GraphBuilderCard(self, show_remove=True, show_analysis=True)
        card.remove_requested.connect(self._remove_extra_graph)
        card.set_namespace(self._arrays_1d, self._arrays_2d)
        self._extra_cards.append(card)
        self._root_layout.addWidget(card)

    def _remove_extra_graph(self, card: GraphBuilderCard) -> None:
        if card in self._extra_cards:
            self._extra_cards.remove(card)
        self._root_layout.removeWidget(card)
        card.hide()
        card.deleteLater()

    # ── cell output sync ─────────────────────────────────────────────

    def _graph_title_for_cell(self, cell: Any) -> str:
        source_lines = [ln.strip() for ln in cell.source().splitlines() if ln.strip()]
        title = source_lines[0] if source_lines else getattr(cell, "cell_id", "plot")
        return title[:45] + "..." if len(title) > 48 else title

    def sync_cell_outputs(self, cells: list[Any]) -> None:
        desired_order: list[str] = []
        for cell in cells:
            result = getattr(cell, "last_result", None)
            if result is None:
                continue
            for output_index, output in enumerate(result.outputs):
                if output.kind == "plotly":
                    key = f"{cell.cell_id}:plotly:{output_index}"
                elif output.kind == "html" and "data:image" in output.data.get("html", ""):
                    key = f"{cell.cell_id}:image:{output_index}"
                else:
                    continue
                desired_order.append(key)
                title_text = self._graph_title_for_cell(cell)
                html = output.data["html"]
                existing = self._output_widgets.get(key)
                if existing is None:
                    title = QLabel(title_text, self.outputs_container)
                    title.setStyleSheet("color:#001f41; font-weight:600;")
                    plot = PlotView(self.outputs_container)
                    plot.set_html(html)
                    self._output_widgets[key] = (title, plot, html)
                else:
                    title, plot, old_html = existing
                    title.setText(title_text)
                    if old_html != html:
                        plot.set_html(html)
                        self._output_widgets[key] = (title, plot, html)
        for key in [k for k in self._output_widgets if k not in desired_order]:
            title, widget, _ = self._output_widgets.pop(key)
            title.hide(); widget.hide()
            title.deleteLater(); widget.deleteLater()
        while self.outputs_layout.count():
            item = self.outputs_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        if self._output_empty_label is not None:
            self._output_empty_label.deleteLater()
            self._output_empty_label = None
        if not desired_order:
            empty = QLabel("Run a plotting cell to see output graphs here.", self.outputs_container)
            empty.setStyleSheet("color:#64748b; font-style:italic;")
            self.outputs_layout.addWidget(empty)
            self._output_empty_label = empty
        else:
            for key in desired_order:
                title, widget, _ = self._output_widgets[key]
                self.outputs_layout.addWidget(title)
                self.outputs_layout.addWidget(widget)
        self.outputs_layout.addStretch(1)


# ── QuickGraphPreviewPanel ───────────────────────────────────────────────────

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
        self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        root.addWidget(self.status_label)

        controls_card = QWidget(self)
        controls_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
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
        self.series_controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
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
        self.evolution_controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
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

        self.preview_stack = QStackedWidget(self)
        self.preview_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.plot_view = PlotView(self.preview_stack)
        self.plot_view.setMinimumHeight(360)
        self.plot_view.setMaximumHeight(360)
        self.preview_stack.addWidget(self.plot_view)

        self.empty_state = QWidget(self.preview_stack)
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.setSpacing(0)
        self.empty_label = QLabel("No plot output yet.", self.empty_state)
        self.empty_label.setStyleSheet("color:#64748b; font-style:italic;")
        self.empty_label.setWordWrap(True)
        empty_layout.addWidget(self.empty_label)
        self.preview_stack.addWidget(self.empty_state)
        self.preview_stack.setCurrentWidget(self.empty_state)
        root.addWidget(self.preview_stack)
        root.addStretch(1)

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
        return {
            "font_size": 14,
            "line_width": 2,
            "marker_size": 8,
            "show_grid": True,
            "show_box": True,
            "ticks_inside": True,
            "show_minor_ticks": True,
            "graph_width": None,
            "graph_height": 360,
        }

    def _selected_y_vars(self) -> list[str]:
        return self.y_combo.checked_values()

    def _sync_mode_ui(self) -> None:
        mode = self.mode_combo.currentData() or "series"
        self.series_controls.setVisible(mode == "series")
        self.evolution_controls.setVisible(mode == "evolution")
        self.refresh_preview()

    def _on_evolution_matrix_changed(self) -> None:
        matrix_name = self.evolution_matrix_combo.currentData()
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
        self.evolution_step_label.setText(
            f"Step {self.evolution_step_slider.value()} / {maximum}"
        )

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        print(f"[debug][quick-graph-preview] set_namespace:start count={len(namespace)}", flush=True)
        previous_mode = self.mode_combo.currentData()
        previous_x = self.x_combo.currentData()
        previous_y = self._selected_y_vars()
        previous_matrix = self.evolution_matrix_combo.currentData()
        previous_time = self.evolution_time_combo.currentData()
        previous_value = self.evolution_value_combo.currentData()
        self._arrays_1d, self._arrays_2d = extract_notebook_array_variables(namespace)

        preferred_mode = "series"
        if self._arrays_2d and not self._arrays_1d:
            preferred_mode = "evolution"
        elif previous_mode == "evolution" and self._arrays_2d:
            preferred_mode = "evolution"

        self.mode_combo.blockSignals(True)
        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)
        self.evolution_matrix_combo.blockSignals(True)
        self.evolution_time_combo.blockSignals(True)
        self.evolution_value_combo.blockSignals(True)

        mode_index = self.mode_combo.findData(preferred_mode)
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)

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
            self.y_combo.add_check_item(name, name, checked=name in selected_y)
        if not selected_y and available_series:
            default_y = (
                available_series[1]
                if len(available_series) > 1 and x_data == available_series[0]
                else available_series[0]
            )
            self.y_combo.set_checked_values([default_y])

        self.evolution_matrix_combo.clear()
        self.evolution_matrix_combo.addItem("Select 2D array", "")
        for name in self._arrays_2d:
            self.evolution_matrix_combo.addItem(name, name)
        matrix_data = (
            previous_matrix
            if isinstance(previous_matrix, str) and previous_matrix in self._arrays_2d
            else ""
        )
        matrix_index = self.evolution_matrix_combo.findData(matrix_data)
        if matrix_index <= 0 and preferred_mode == "evolution" and self.evolution_matrix_combo.count() > 1:
            matrix_index = 1
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

        self.mode_combo.blockSignals(False)
        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)
        self.evolution_matrix_combo.blockSignals(False)
        self.evolution_time_combo.blockSignals(False)
        self.evolution_value_combo.blockSignals(False)

        self._sync_mode_ui()
        self._on_evolution_matrix_changed()
        self.refresh_preview()

    def set_latest_plot(self, title: str, html: str) -> None:
        self._latest_plot_title = title
        self._latest_plot_html = html
        self.refresh_preview()

    def current_figure(self) -> go.Figure:
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

        has_series = bool(self._arrays_1d)
        has_evolution = bool(self._arrays_2d)
        use_latest_plot = not has_series and not has_evolution and bool(self._latest_plot_html)
        if use_latest_plot:
            self.plot_title.setVisible(True)
            self.plot_title.setText(self._latest_plot_title or "Latest notebook plot")
            self.preview_stack.setCurrentWidget(self.plot_view)
            self.status_label.setText("Showing latest executed notebook plot.")
            self.plot_view.set_html(self._latest_plot_html)
            return

        if mode == "evolution":
            self._current_figure = build_notebook_evolution_figure(
                self._arrays_1d, self._arrays_2d,
                matrix_var if isinstance(matrix_var, str) else None,
                time_var if isinstance(time_var, str) else None,
                value_var if isinstance(value_var, str) else None,
                step_index,
                plot_type if isinstance(plot_type, str) else "lines",
                "", "", "", style_options,
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
                "", "", "", {}, style_options,
            )
            has_plot = bool(y_vars)
            status = f"Quick preview: x={x_var or 'index'} y={', '.join(y_vars) if y_vars else 'none'}"
            title = ", ".join(y_vars) if y_vars else "Quick preview"

        if not has_plot and self._latest_plot_html:
            self.plot_title.setVisible(True)
            self.plot_title.setText(self._latest_plot_title or "Latest notebook plot")
            self.preview_stack.setCurrentWidget(self.plot_view)
            self.status_label.setText("Showing latest executed notebook plot.")
            self.plot_view.set_html(self._latest_plot_html)
            return

        html = self._current_figure.to_html(
            include_plotlyjs=True,
            full_html=False,
            config={"responsive": True, "displaylogo": False},
        )
        self.plot_title.setVisible(has_plot)
        self.plot_title.setText(title if has_plot else "")
        self.status_label.setText(
            status if has_plot else "Run code to populate arrays or create a plot output."
        )
        if has_plot:
            self.preview_stack.setCurrentWidget(self.plot_view)
            self.plot_view.set_html(html)
        else:
            self.preview_stack.setCurrentWidget(self.empty_state)
