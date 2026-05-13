"""
notebook_plot_panel.py
======================
Graph panel hierarchy for the PySide6 calculation notebook.

Class hierarchy
---------------
NamespaceConsumerMixin          — shared array extraction + combo helpers
BaseGraphPanel(QWidget, mixin)  — abstract: refresh(), current_figure()
  ├─ QuickGraphPreviewPanel     — simple sidebar (Notebook tab)
  └─ GraphBuilderCard           — full panel (Graphs tab), composed of:
        DataSourceWidget        — Notebook / CSV source toggle
        AxisSelectorWidget      — mode, X/Y, plot-type, evolution controls
        SeriesStyleWidget       — per-series line/marker/color rows
        PlotStyleWidget         — font, line width, grid, graph size
        AxisLabelsWidget        — title, X label, Y label
        AnalysisWidget          — smooth, derivative, curve fit

Containers
----------
NotebookPlotPanel               — Graphs-tab wrapper (holds GraphBuilderCards)
NotebookGraphWorkspace          — Notebook-tab workspace (kept for compat.)
"""
from __future__ import annotations

import os
from abc import abstractmethod
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


# ── Options constants ────────────────────────────────────────────────────────

PLOT_TYPE_OPTIONS = (
    ("Lines", "lines"),
    ("Markers", "markers"),
    ("Lines + Markers", "lines+markers"),
    ("Bar", "bar"),
    ("Histogram", "histogram"),
)
LINE_STYLE_OPTIONS = (
    ("Solid", "solid"), ("Dash", "dash"), ("Dot", "dot"),
    ("Dash Dot", "dashdot"), ("Long Dash", "longdash"), ("Long Dash Dot", "longdashdot"),
)
MARKER_STYLE_OPTIONS = (
    ("Circle", "circle"), ("Square", "square"), ("Diamond", "diamond"),
    ("Cross", "cross"), ("X", "x"), ("Triangle Up", "triangle-up"),
    ("Triangle Down", "triangle-down"),
)
LINE_COLOR_OPTIONS = (
    ("Auto", "auto"), ("Blue", "#1f77b4"), ("Red", "#d62728"),
    ("Green", "#2ca02c"), ("Black", "#000000"), ("Orange", "#ff7f0e"),
    ("Purple", "#9467bd"),
)
FONT_SIZE_OPTIONS  = (12, 14, 16, 18, 20, 24, 28, 32)
LINE_WIDTH_OPTIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
MARKER_SIZE_OPTIONS = (4, 6, 8, 10, 12, 14, 16, 18, 20)
GRAPH_SIZE_OPTIONS = (
    ("800 × 700", (800, 700)),
    ("800 × 600", (800, 600)),
    ("Square", (700, 700)),
    ("700 × 700", (700, 700)),
)

_CARD_STYLE = """
    QWidget#graphCard { background:#f8fbfe; border:1px solid #d7e2ec; border-radius:12px; }
    QWidget#graphCard QLabel { background:transparent; border:none; }
    QWidget#graphCard QComboBox,
    QWidget#graphCard QLineEdit,
    QWidget#graphCard QListWidget,
    QWidget#graphCard QSpinBox {
        background:#ffffff; border:1px solid #d1dce8;
        border-radius:8px; padding:5px 8px; font-size:12px; font-weight:400; color:#0f1b2b;
    }
    QWidget#graphCard QListView { background:#ffffff; border:1px solid #d1dce8; }
    QWidget#graphCard QWidget#graphSection,
    QWidget#graphCard QWidget#graphPreviewCard {
        background:#ffffff;
        border:1px solid #d7e2ec;
        border-radius:10px;
    }
    QWidget#graphCard QRadioButton,
    QWidget#graphCard QPushButton {
        font-size:12px; font-weight:400;
    }
    QWidget#graphCard QComboBox QAbstractItemView {
        selection-background-color:#c7def5; selection-color:#0f1b2b; outline:0;
    }
    QWidget#graphCard QComboBox QAbstractItemView::item:hover {
        background:#c7def5; color:#0f1b2b;
    }
"""
_TEXT_SS  = "color:#355070; font-weight:400; font-size:12px;"
_LBL_SS   = "color:#355070; font-weight:700; font-size:12px;"
_TITLE_SS = "color:#001f41; font-weight:700; font-size:14px;"
_SECTION_TITLE_SS = _LBL_SS
_MUTED_SS = "color:#64748b; font-weight:400; font-size:12px;"
_CB_SS = (
    "QCheckBox { color:#355070; font-weight:400; font-size:12px; }"
    "QCheckBox::indicator { width:13px; height:13px; border:1.5px solid #1d4ed8;"
    " border-radius:3px; background:#fff; }"
    "QCheckBox::indicator:checked { border:1.5px solid #1d4ed8; background:#bfdbfe; }"
)


# ── Scientific layout helpers (pure functions) ───────────────────────────────

def _default_style() -> dict[str, Any]:
    """Return the base scientific plotting style used across graph widgets."""
    return {
        "font_size": 16, "line_width": 2, "marker_size": 7,
        "show_grid": True, "show_box": True,
        "ticks_inside": True, "show_minor_ticks": True,
        "graph_width": 800, "graph_height": 700,
    }


def _axis_opts(style: dict[str, Any], axis_title: str) -> dict[str, Any]:
    """Build one Plotly axis configuration from the active style settings."""
    fs = int(style.get("font_size", 16))
    td = "inside" if style.get("ticks_inside", True) else "outside"
    return {
        "title": {"text": axis_title, "font": {"size": fs + 2, "color": "#0f1b2b"}},
        "automargin": True,
        "tickfont": {"size": fs, "color": "#334155"},
        "showgrid": bool(style.get("show_grid", True)),
        "gridcolor": "rgba(200,210,220,0.5)",
        "zeroline": False,
        "showline": bool(style.get("show_box", True)),
        "mirror": "allticks" if style.get("show_box", True) else False,
        "linecolor": "#0f1b2b", "linewidth": 2.5,
        "ticks": td, "ticklen": 10, "tickwidth": 1.5, "tickcolor": "#0f1b2b",
        "minor": {
            "ticks": td if style.get("show_minor_ticks", True) else "",
            "ticklen": 5, "tickcolor": "#475569", "showgrid": False,
        },
    }


def _apply_layout(
    fig: go.Figure, *, title: str | None,
    x_title: str, y_title: str, showlegend: bool,
    style: dict[str, Any] | None = None, barmode: str | None = None,
) -> None:
    """Apply shared scientific layout options to a Plotly figure in-place."""
    s = dict(_default_style())
    if style:
        s.update(style)
    fs = int(s.get("font_size", 16))
    fig.update_layout(
        title={"text": title or None, "font": {"size": fs + 4, "color": "#0f1b2b"}},
        showlegend=showlegend, plot_bgcolor="white", paper_bgcolor="white",
        margin={"l": 60, "r": 20, "t": 35, "b": 90},
        font={"size": fs, "color": "#0f1b2b"},
        legend={"orientation": "v", "x": 0.01, "y": 0.99,
                "xanchor": "left", "yanchor": "top", "font": {"size": fs}},
        barmode=barmode, width=s.get("graph_width"), height=s.get("graph_height"),
    )
    fig.update_xaxes(**_axis_opts(s, x_title))
    fig.update_yaxes(**_axis_opts(s, y_title))


# ── Array extraction (pure functions) ───────────────────────────────────────

def _to_1d(value: Any) -> np.ndarray | None:
    """Normalize a numeric list or array into a 1D float numpy array."""
    arr = np.asarray(value) if isinstance(value, (list, tuple)) else \
          value if isinstance(value, np.ndarray) else None
    if arr is None or arr.ndim != 1 or not np.issubdtype(arr.dtype, np.number):
        return None
    return arr.astype(float)


def _to_2d(value: Any) -> np.ndarray | None:
    """Normalize a numeric list or array into a 2D float numpy array."""
    arr = np.asarray(value) if isinstance(value, (list, tuple)) else \
          value if isinstance(value, np.ndarray) else None
    if arr is None or arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
        return None
    return arr.astype(float)


def extract_notebook_array_variables(
    namespace: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract numeric 1D and 2D arrays from a notebook execution namespace."""
    from types import ModuleType
    arrays_1d: dict[str, np.ndarray] = {}
    arrays_2d: dict[str, np.ndarray] = {}
    for name, value in sorted(namespace.items()):
        if name.startswith("_") or callable(value) or isinstance(value, ModuleType):
            continue
        a1 = _to_1d(value)
        if a1 is not None:
            arrays_1d[name] = a1
            continue
        a2 = _to_2d(value)
        if a2 is not None:
            arrays_2d[name] = a2
    return arrays_1d, arrays_2d


# ── Figure builders (pure functions) ────────────────────────────────────────

def build_notebook_plot_figure(
    arrays: dict[str, np.ndarray],
    x_var: str | None, y_vars: list[str], plot_type: str,
    title: str, x_title: str, y_title: str,
    series_styles: dict[str, dict[str, str]] | None = None,
    style: dict[str, Any] | None = None,
    style_options: dict[str, Any] | None = None,
) -> go.Figure:
    """Build a standard 1D notebook figure from selected x/y arrays and styles."""
    fig = go.Figure()
    s = dict(_default_style())
    if style:
        s.update(style)
    if style_options:
        s.update(style_options)
    lw = int(s.get("line_width", 2))
    ms = int(s.get("marker_size", 7))
    fs = int(s.get("font_size", 16))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    x_data = arrays.get(x_var) if x_var else None
    if not y_vars:
        fig.add_annotation(text="Select Y variable(s) to plot.", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font={"size": fs, "color": "#94a3b8"})
    for i, y_name in enumerate(y_vars):
        y_data = arrays.get(y_name)
        if y_data is None:
            continue
        st = dict((series_styles or {}).get(y_name) or {})
        tp  = st.get("plot_type") or plot_type or "lines"
        ls  = st.get("line_style") or "solid"
        mk  = st.get("marker_style") or "circle"
        col = st.get("line_color") or "auto"
        x_plot = x_data if x_data is not None and len(x_data) == len(y_data) else np.arange(len(y_data))
        c = colors[i % len(colors)] if col == "auto" else col
        if tp == "histogram":
            fig.add_trace(go.Histogram(x=y_data, name=y_name, marker_color=c, opacity=0.75))
        elif tp == "bar":
            fig.add_trace(go.Bar(x=x_plot, y=y_data, name=y_name, marker_color=c))
        else:
            fig.add_trace(go.Scatter(x=x_plot, y=y_data, mode=tp or "lines", name=y_name,
                                     line={"color": c, "width": lw, "dash": ls},
                                     marker={"color": c, "size": ms, "symbol": mk}))
    _apply_layout(fig, title=title or None,
                  x_title=x_title or (x_var or "index"),
                  y_title=y_title or (y_vars[0] if len(y_vars) == 1 else "value"),
                  showlegend=len(y_vars) > 1, style=s,
                  barmode="group" if plot_type == "bar" else None)
    return fig


def build_notebook_evolution_figure(
    arrays_1d: dict[str, np.ndarray], arrays_2d: dict[str, np.ndarray],
    matrix_var: str | None, time_var: str | None, value_var: str | None,
    step_index: int, plot_type: str,
    title: str, x_title: str, y_title: str,
    style: dict[str, Any] | None = None,
) -> go.Figure:
    """Build a figure showing one time-step slice from a 2D evolution matrix."""
    fig = go.Figure()
    s = dict(_default_style())
    if style:
        s.update(style)
    fs = int(s.get("font_size", 16))
    lw = int(s.get("line_width", 2))
    ms = int(s.get("marker_size", 7))
    matrix = arrays_2d.get(matrix_var) if matrix_var else None
    if matrix is None:
        fig.add_annotation(text="Select a 2D array to plot its evolution.",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font={"size": fs, "color": "#94a3b8"})
        return fig
    rows, cols = matrix.shape
    step = min(max(0, step_index), rows - 1)
    t_axis = arrays_1d.get(time_var) if isinstance(time_var, str) and time_var else None
    t_axis = t_axis if t_axis is not None and len(t_axis) == rows else np.arange(rows, dtype=float)
    v_axis = arrays_1d.get(value_var) if isinstance(value_var, str) and value_var else None
    v_axis = v_axis if v_axis is not None and len(v_axis) == cols else np.arange(cols, dtype=float)
    y_data = matrix[step]
    t_val  = t_axis[step] if len(t_axis) > step else step
    tname  = f"{matrix_var or 'matrix'} @ step {step}"
    if plot_type == "bar":
        fig.add_trace(go.Bar(x=v_axis, y=y_data, name=tname, marker_color="#1f77b4"))
    else:
        fig.add_trace(go.Scatter(x=v_axis, y=y_data, name=tname,
                                 mode  =plot_type if plot_type != "histogram" else "lines",
                                 line  ={"color": "#1f77b4", "width": lw},
                                 marker={"color": "#1f77b4", "size": ms}))
    _apply_layout(fig,
                  title=title or f"{matrix_var or '2D array'} @ t={t_val}",
                  x_title=x_title or (value_var or "value index"),
                  y_title=y_title or (matrix_var or "value"),
                  showlegend=False, style=s)
    return fig


# ── NamespaceConsumerMixin ────────────────────────────────────────────────────

class NamespaceConsumerMixin:
    """Shared helpers for namespace management and combo selection preservation."""

    def _init_namespace_state(self) -> None:
        """Initialize cached 1D and 2D array dictionaries for a graph consumer."""
        self._nb_arrays_1d: dict[str, np.ndarray] = {}
        self._nb_arrays_2d: dict[str, np.ndarray] = {}

    @staticmethod
    def _restore_selection(combo: AutoCloseComboBox | CheckableComboBox, previous: Any) -> None:
        """Restore a combo-box selection if the previous value is still present."""
        idx = combo.findData(previous)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    @staticmethod
    def _restore_multi_selection(combo: CheckableComboBox, previous: set[str]) -> None:
        """Restore a multi-select combo selection from a previous checked-value set."""
        if previous:
            combo.set_checked_values(list(previous & {combo.itemData(i)
                                                       for i in range(combo.count())}))


# ── DataSourceWidget ──────────────────────────────────────────────────────────

class DataSourceWidget(QWidget):
    """
    Toggles between Notebook arrays and a loaded CSV file.
    Emits ``changed`` whenever the active source or its arrays change.
    """
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the notebook-vs-CSV data source selector UI."""
        super().__init__(parent)
        self._notebook_arrays: dict[str, np.ndarray] = {}
        self._csv_arrays: dict[str, np.ndarray] = {}

        # Use objectName so the border/background only applies to THIS widget,
        # not to child radio buttons or buttons (avoids invisible-widget bug)
        self.setObjectName("dataSourceBox")
        self.setStyleSheet(
            "#dataSourceBox { background:transparent; border:none; }"
            "#dataSourceBox QRadioButton { " + _TEXT_SS + " }"
            "#dataSourceBox QLabel#dataSourcePath { " + _MUTED_SS + " }"
            "#dataSourceBox QPushButton { border:1px solid #93c5fd; border-radius:4px;"
            " padding:3px 10px; font-size:12px; font-weight:400;"
            " background:#eff6ff; color:#1d4ed8; }"
            "#dataSourceBox QPushButton:hover { background:#dbeafe; }"
        )
        print("[debug][data-source-widget] init shared_font applied=True", flush=True)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        radio_row = QHBoxLayout()
        radio_row.setSpacing(16)
        self._notebook_radio = QRadioButton("Notebook arrays", self)
        self._notebook_radio.setChecked(True)
        self._csv_radio = QRadioButton("CSV / text file", self)
        # Keep QButtonGroup as instance attribute — local vars get garbage-collected
        self._btn_group = QButtonGroup(self)
        self._btn_group.addButton(self._notebook_radio)
        self._btn_group.addButton(self._csv_radio)
        radio_row.addWidget(self._notebook_radio)
        radio_row.addWidget(self._csv_radio)
        radio_row.addStretch(1)
        root.addLayout(radio_row)

        # File row — shown only when CSV is selected
        self._csv_row = QWidget(self)
        cr = QHBoxLayout(self._csv_row)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.setSpacing(8)
        self._path_label = QLabel("No file selected", self._csv_row)
        self._path_label.setObjectName("dataSourcePath")
        self._open_btn = QPushButton("Open file…", self._csv_row)
        self._open_btn.clicked.connect(self._open_file)
        cr.addWidget(self._path_label, 1)
        cr.addWidget(self._open_btn)
        self._csv_row.hide()
        root.addWidget(self._csv_row)

        self._btn_group.buttonToggled.connect(self._on_toggle)

    # ── public ───────────────────────────────────────────────────────

    def active_arrays_1d(self) -> dict[str, np.ndarray]:
        """Return the 1D arrays from the currently active data source."""
        return self._csv_arrays if self._csv_radio.isChecked() else self._notebook_arrays

    def is_csv(self) -> bool:
        """Return whether the widget is currently using CSV-backed data."""
        return self._csv_radio.isChecked()

    def update_notebook_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        """Refresh the notebook-backed array cache and emit when active."""
        self._notebook_arrays = arrays
        if not self._csv_radio.isChecked():
            self.changed.emit()

    # ── private ──────────────────────────────────────────────────────

    def _on_toggle(self, _btn: Any, checked: bool) -> None:
        """Swap visible controls when the active data source changes."""
        if not checked:
            return
        self._csv_row.setVisible(self._csv_radio.isChecked())
        self.changed.emit()

    def _open_file(self) -> None:
        """Load numeric columns from a CSV/text file into the active graph source."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open data file", "",
            "Data files (*.csv *.txt *.dat *.tsv);;All files (*)",
        )
        if not path:
            return
        try:
            df = pd.read_csv(path, sep=None, engine="python")
            df = df.dropna(how="any")          # align all columns to same rows
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self._csv_arrays = {c: df[c].to_numpy(dtype=float) for c in numeric_cols}
            self._path_label.setText(os.path.basename(path))
            self.changed.emit()
        except Exception as exc:
            self._path_label.setText(f"Error: {exc}")


# ── AxisSelectorWidget ────────────────────────────────────────────────────────

class AxisSelectorWidget(QWidget, NamespaceConsumerMixin):
    """
    Mode toggle (Series/Evolution), X/Y variable combos, plot type,
    and evolution controls (matrix, time, value, step slider).
    Emits ``changed`` on any user change.
    """
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the mode-specific axis and variable selection controls."""
        super().__init__(parent)
        self._init_namespace_state()
        self._arrays_2d: dict[str, np.ndarray] = {}
        self.setStyleSheet("QLabel { " + _LBL_SS + " }")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # mode + graph size row (size owned by PlotStyleWidget; here only mode)
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Mode", self)
        mode_lbl.setStyleSheet(_LBL_SS)
        self.mode_combo = AutoCloseComboBox(self)
        self.mode_combo.addItem("Series (1D)", "series")
        self.mode_combo.addItem("Evolution (2D)", "evolution")
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self.mode_combo, 1)
        root.addLayout(mode_row)

        # series controls
        self._series_widget = QWidget(self)
        series_row = QHBoxLayout(self._series_widget)
        series_row.setContentsMargins(0, 0, 0, 0)
        series_row.setSpacing(8)

        self.x_combo = AutoCloseComboBox(self)
        self.x_combo.addItem("Index", "")
        self.plot_type_combo = AutoCloseComboBox(self)
        for lbl, val in PLOT_TYPE_OPTIONS:
            self.plot_type_combo.addItem(lbl, val)
        self.y_combo = CheckableComboBox(self)

        for lbl_text, widget, stretch in (
            ("X variable", self.x_combo, 1),
            ("Plot type", self.plot_type_combo, 1),
            ("Y variable(s)", self.y_combo, 2),
        ):
            blk = QWidget(self._series_widget)
            bl = QVBoxLayout(blk)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(3)
            bl.addWidget(QLabel(lbl_text, blk))
            bl.addWidget(widget)
            series_row.addWidget(blk, stretch)
        root.addWidget(self._series_widget)

        # evolution controls
        self._evo_widget = QWidget(self)
        evo_form = QFormLayout(self._evo_widget)
        evo_form.setContentsMargins(0, 0, 0, 0)
        evo_form.setSpacing(6)
        self.evo_matrix_combo = AutoCloseComboBox(self)
        self.evo_matrix_combo.addItem("Select 2D array", "")
        self.evo_time_combo   = AutoCloseComboBox(self)
        self.evo_time_combo.addItem("Row index", "")
        self.evo_value_combo  = AutoCloseComboBox(self)
        self.evo_value_combo.addItem("Column index", "")
        self.evo_step_slider  = QSlider(Qt.Orientation.Horizontal, self)
        self.evo_step_slider.setRange(0, 0)
        self.evo_step_label   = QLabel("Step 0 / 0", self)
        for row_lbl, widget in (
            ("Evolution array", self.evo_matrix_combo),
            ("Time axis", self.evo_time_combo),
            ("Value axis", self.evo_value_combo),
            ("Time step", self.evo_step_slider),
            ("Selected step", self.evo_step_label),
        ):
            evo_form.addRow(row_lbl, widget)
        self._evo_widget.hide()
        root.addWidget(self._evo_widget)

        # signals
        self.mode_combo.currentIndexChanged.connect(self._sync_mode)
        self.x_combo.currentIndexChanged.connect(self.changed)
        self.y_combo.checkedItemsChanged.connect(self.changed)
        self.plot_type_combo.currentIndexChanged.connect(self.changed)
        self.evo_matrix_combo.currentIndexChanged.connect(self._on_matrix_changed)
        self.evo_time_combo.currentIndexChanged.connect(self.changed)
        self.evo_value_combo.currentIndexChanged.connect(self.changed)
        self.evo_step_slider.valueChanged.connect(self._on_step_changed)

    # ── public ───────────────────────────────────────────────────────

    def populate(
        self,
        arrays_1d: dict[str, np.ndarray],
        arrays_2d: dict[str, np.ndarray],
    ) -> None:
        """Rebuild selector combos from the current 1D and 2D array sets."""
        self._arrays_2d = dict(arrays_2d)
        prev_x      = self.x_combo.currentData()
        prev_y      = set(self.selected_y_vars())
        prev_matrix = self.evo_matrix_combo.currentData()
        prev_time   = self.evo_time_combo.currentData()
        prev_value  = self.evo_value_combo.currentData()

        for combo in (self.x_combo, self.y_combo, self.evo_matrix_combo,
                      self.evo_time_combo, self.evo_value_combo):
            combo.blockSignals(True)

        self.x_combo.clear()
        self.x_combo.addItem("Index", "")
        self.y_combo.clear()
        self.evo_matrix_combo.clear()
        self.evo_matrix_combo.addItem("Select 2D array", "")
        self.evo_time_combo.clear()
        self.evo_time_combo.addItem("Row index", "")
        self.evo_value_combo.clear()
        self.evo_value_combo.addItem("Column index", "")

        for name in sorted(arrays_1d):
            self.x_combo.addItem(name, name)
            self.evo_time_combo.addItem(name, name)
            self.evo_value_combo.addItem(name, name)
            self.y_combo.add_check_item(name, name, checked=name in prev_y)
        for name in sorted(arrays_2d):
            self.evo_matrix_combo.addItem(name, name)

        for combo, prev in (
            (self.x_combo, prev_x), (self.evo_matrix_combo, prev_matrix),
            (self.evo_time_combo, prev_time), (self.evo_value_combo, prev_value),
        ):
            self._restore_selection(combo, prev)

        if not prev_y and arrays_1d:
            names = sorted(arrays_1d)
            x_name = self.x_combo.currentData() or ""
            default = names[1] if len(names) > 1 and x_name == names[0] else names[0]
            self.y_combo.set_checked_values([default])

        for combo in (self.x_combo, self.y_combo, self.evo_matrix_combo,
                      self.evo_time_combo, self.evo_value_combo):
            combo.blockSignals(False)

        self._on_matrix_changed()

    def mode(self) -> str:
        """Return the current graph mode: `series` or `evolution`."""
        return self.mode_combo.currentData() or "series"

    def selected_x(self) -> str | None:
        """Return the selected x-variable name for 1D plotting."""
        v = self.x_combo.currentData()
        return v if v else None

    def selected_y_vars(self) -> list[str]:
        """Return the selected y-variable names for 1D plotting."""
        return [v for v in self.y_combo.checked_values() if isinstance(v, str)]

    def plot_type(self) -> str:
        """Return the currently selected Plotly trace mode/type."""
        return self.plot_type_combo.currentData() or "lines"

    def evolution_params(self) -> tuple[str | None, str | None, str | None, int]:
        """Return the current matrix/time/value/step selection for evolution mode."""
        return (
            self.evo_matrix_combo.currentData() or None,
            self.evo_time_combo.currentData() or None,
            self.evo_value_combo.currentData() or None,
            self.evo_step_slider.value(),
        )

    # ── private ──────────────────────────────────────────────────────

    def _sync_mode(self) -> None:
        """Show the correct control group for the active plotting mode."""
        is_series = self.mode() == "series"
        self._series_widget.setVisible(is_series)
        self._evo_widget.setVisible(not is_series)
        self.changed.emit()

    def _on_matrix_changed(self) -> None:
        """Refresh the evolution slider when the selected matrix changes."""
        name = self.evo_matrix_combo.currentData()
        print(f"[debug][axis-selector] matrix_changed name={name!r}", flush=True)
        self.update_evo_slider_range(self._arrays_2d)
        self._update_step_label()
        self.changed.emit()

    def _on_step_changed(self, _val: int) -> None:
        """Refresh labels and output when the evolution step changes."""
        self._update_step_label()
        self.changed.emit()

    def _update_step_label(self) -> None:
        """Update the human-readable evolution step indicator text."""
        cur = self.evo_step_slider.value()
        tot = self.evo_step_slider.maximum()
        self.evo_step_label.setText(f"Step {cur} / {tot}")

    def update_evo_slider_range(self, arrays_2d: dict[str, np.ndarray]) -> None:
        """Resize the evolution step slider to match the selected matrix height."""
        name = self.evo_matrix_combo.currentData()
        matrix = arrays_2d.get(name) if isinstance(name, str) and name else None
        rows = int(matrix.shape[0]) if matrix is not None else 0
        max_idx = max(rows - 1, 0)
        self.evo_step_slider.blockSignals(True)
        self.evo_step_slider.setRange(0, max_idx)
        self.evo_step_slider.setValue(min(self.evo_step_slider.value(), max_idx))
        self.evo_step_slider.blockSignals(False)
        self._update_step_label()


# ── SeriesStyleWidget ─────────────────────────────────────────────────────────

class SeriesStyleWidget(QWidget):
    """
    Dynamic per-Y-variable style rows (plot type, line, marker, color).
    Visible only when Y variables are selected.
    Emits ``changed`` when the user edits any style control.
    """
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the per-series style editor container."""
        super().__init__(parent)
        self._rows: dict[str, dict[str, AutoCloseComboBox]] = {}

        self.setStyleSheet(
            "background:#ffffff; border:1px solid #d7e2ec; border-radius:8px;"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(4)
        title = QLabel("Per-Series Styles", self)
        title.setStyleSheet(_LBL_SS)
        root.addWidget(title)
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(4)
        root.addLayout(self._grid)
        self.hide()

    # ── public ───────────────────────────────────────────────────────

    def update_y_vars(self, y_vars: list[str], default_plot_type: str) -> None:
        """Rebuild rows for the given Y variables, preserving existing selections."""
        prev = {
            name: {k: w.currentData() for k, w in combos.items()}
            for name, combos in self._rows.items()
        }
        self._clear()
        self.setVisible(bool(y_vars))
        if not y_vars:
            return
        headers = ("Y variable", "Plot type", "Line style", "Marker", "Color")
        for col, hdr in enumerate(headers):
            lbl = QLabel(hdr, self)
            lbl.setStyleSheet(_LBL_SS)
            self._grid.addWidget(lbl, 0, col)
        for row, name in enumerate(y_vars, start=1):
            name_lbl = QLabel(name, self)
            name_lbl.setStyleSheet(_LBL_SS)
            pt = AutoCloseComboBox(self)
            for lbl, val in PLOT_TYPE_OPTIONS:
                pt.addItem(lbl, val)
            ls = AutoCloseComboBox(self)
            for lbl, val in LINE_STYLE_OPTIONS:
                ls.addItem(lbl, val)
            mk = AutoCloseComboBox(self)
            for lbl, val in MARKER_STYLE_OPTIONS:
                mk.addItem(lbl, val)
            lc = AutoCloseComboBox(self)
            for lbl, val in LINE_COLOR_OPTIONS:
                lc.addItem(lbl, val)
            p = prev.get(name, {})
            for combo, key, default in (
                (pt, "plot_type", default_plot_type),
                (ls, "line_style", "solid"),
                (mk, "marker_style", "circle"),
                (lc, "line_color", "auto"),
            ):
                idx = combo.findData(p.get(key) or default)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                combo.currentIndexChanged.connect(self.changed)
            self._grid.addWidget(name_lbl, row, 0)
            self._grid.addWidget(pt,  row, 1)
            self._grid.addWidget(ls,  row, 2)
            self._grid.addWidget(mk,  row, 3)
            self._grid.addWidget(lc,  row, 4)
            self._rows[name] = {"plot_type": pt, "line_style": ls,
                                 "marker_style": mk, "line_color": lc}

    def style_map(self) -> dict[str, dict[str, str]]:
        """Return the currently configured style options for each selected series."""
        return {
            name: {k: str(w.currentData() or "") for k, w in combos.items()}
            for name, combos in self._rows.items()
        }

    # ── private ──────────────────────────────────────────────────────

    def _clear(self) -> None:
        """Remove every style row before rebuilding the per-series editor."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._rows.clear()


# ── PlotStyleWidget ───────────────────────────────────────────────────────────

class PlotStyleWidget(QWidget):
    """
    Graph size presets + font size, line width, marker size,
    grid/box/ticks checkboxes.
    Emits ``changed`` on any user change.
    """
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build appearance controls for graph size and styling toggles."""
        super().__init__(parent)
        self.setStyleSheet("QLabel { " + _LBL_SS + " }")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # size + title row
        size_row = QHBoxLayout()
        size_lbl = QLabel("Graph size", self)
        size_lbl.setStyleSheet(_LBL_SS)
        self.size_combo = AutoCloseComboBox(self)
        for lbl, val in GRAPH_SIZE_OPTIONS:
            self.size_combo.addItem(lbl, val)
        size_row.addWidget(size_lbl)
        size_row.addWidget(self.size_combo, 1)
        root.addLayout(size_row)

        # numeric controls
        num_row = QHBoxLayout()
        num_row.setSpacing(8)
        self.font_size_combo   = AutoCloseComboBox(self)
        self.line_width_combo  = AutoCloseComboBox(self)
        self.marker_size_combo = AutoCloseComboBox(self)
        for val in FONT_SIZE_OPTIONS:
            self.font_size_combo.addItem(str(val), val)
        for val in LINE_WIDTH_OPTIONS:
            self.line_width_combo.addItem(str(val), val)
        for val in MARKER_SIZE_OPTIONS:
            self.marker_size_combo.addItem(str(val), val)
        self.font_size_combo.setCurrentIndex(self.font_size_combo.findData(16))
        self.line_width_combo.setCurrentIndex(self.line_width_combo.findData(2))
        mk_idx = self.marker_size_combo.findData(8)
        self.marker_size_combo.setCurrentIndex(max(mk_idx, 0))
        for lbl_text, widget in (
            ("Font", self.font_size_combo),
            ("Line", self.line_width_combo),
            ("Marker", self.marker_size_combo),
        ):
            blk = QWidget(self)
            bl = QVBoxLayout(blk)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(3)
            bl.addWidget(QLabel(lbl_text, blk))
            bl.addWidget(widget)
            num_row.addWidget(blk, 1)
        root.addLayout(num_row)

        # checkbox row
        cb_row = QHBoxLayout()
        cb_row.setSpacing(10)
        self.grid_check   = QCheckBox("Grid", self)
        self.box_check    = QCheckBox("Box", self)
        self.ticks_check  = QCheckBox("Ticks inside", self)
        self.minor_check  = QCheckBox("Minor ticks", self)
        for cb in (self.grid_check, self.box_check, self.ticks_check, self.minor_check):
            cb.setChecked(True)
            cb.setStyleSheet(_CB_SS)
            cb_row.addWidget(cb)
        cb_row.addStretch(1)
        root.addLayout(cb_row)

        for widget in (self.size_combo, self.font_size_combo,
                        self.line_width_combo, self.marker_size_combo):
            widget.currentIndexChanged.connect(self.changed)
        for cb in (self.grid_check, self.box_check, self.ticks_check, self.minor_check):
            cb.toggled.connect(self.changed)

    # ── public ───────────────────────────────────────────────────────

    def graph_size(self) -> tuple[int, int]:
        """Return the selected graph width and height preset."""
        size = self.size_combo.currentData()
        return size if isinstance(size, tuple) else (800, 700)

    def style_options(self) -> dict[str, Any]:
        """Return the current appearance controls as a style dictionary."""
        w, h = self.graph_size()
        options = {
            "font_size":       int(self.font_size_combo.currentData() or 16),
            "line_width":      int(self.line_width_combo.currentData() or 2),
            "marker_size":     int(self.marker_size_combo.currentData() or 8),
            "show_grid":       self.grid_check.isChecked(),
            "show_box":        self.box_check.isChecked(),
            "ticks_inside":    self.ticks_check.isChecked(),
            "show_minor_ticks": self.minor_check.isChecked(),
            "graph_width":     w,
            "graph_height":    h,
        }
        print(f"[debug][plot-style-widget] style_options options={options!r}", flush=True)
        return options


# ── AxisLabelsWidget ──────────────────────────────────────────────────────────

class AxisLabelsWidget(QWidget):
    """Title, X label, Y label text inputs. Emits ``changed`` on edit."""
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build text inputs for the plot title and axis labels."""
        super().__init__(parent)
        self.setStyleSheet("QLabel { " + _LBL_SS + " }")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.title_edit   = QLineEdit(self)
        self.x_label_edit = QLineEdit(self)
        self.y_label_edit = QLineEdit(self)
        for lbl_text, widget in (
            ("Title", self.title_edit),
            ("X label", self.x_label_edit),
            ("Y label", self.y_label_edit),
        ):
            sec = QWidget(self)
            sl = QVBoxLayout(sec)
            sl.setContentsMargins(0, 0, 0, 0)
            sl.setSpacing(3)
            sl.addWidget(QLabel(lbl_text, sec))
            sl.addWidget(widget)
            row.addWidget(sec, 1)
        for edit in (self.title_edit, self.x_label_edit, self.y_label_edit):
            edit.textChanged.connect(self.changed)

    def title(self) -> str:
        """Return the current plot title text."""
        return self.title_edit.text().strip()

    def x_label(self) -> str:
        """Return the current x-axis label text."""
        return self.x_label_edit.text().strip()

    def y_label(self) -> str:
        """Return the current y-axis label text."""
        return self.y_label_edit.text().strip()


# ── AnalysisWidget ────────────────────────────────────────────────────────────

class AnalysisWidget(QWidget):
    """
    Smooth (Savitzky-Golay), Derivative (np.gradient), Curve Fit.
    ``apply_overlays()`` mutates the figure and returns a status string.
    Emits ``changed`` when any control is toggled or edited.
    """
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build smoothing, derivative, and curve-fit controls."""
        super().__init__(parent)
        self.setStyleSheet(
            "QWidget { background:transparent; border:none; }"
            "QLabel { border:none; background:transparent; color:#355070;"
            " font-weight:700; font-size:12px; }"
            "QLineEdit { background:#fff; border:1px solid #d1dce8;"
            " border-radius:4px; padding:2px 5px; font-size:12px; font-weight:400; }"
            "QSpinBox  { background:#fff; border:1px solid #d1dce8;"
            " border-radius:4px; padding:1px 3px; font-size:12px; font-weight:400; }"
        )
        print("[debug][analysis-widget] init shared_font labels=bold controls=regular", flush=True)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)
        root.addLayout(self._build_smooth_row())
        root.addLayout(self._build_deriv_row())
        root.addLayout(self._build_fit_rows())

    # ── public ───────────────────────────────────────────────────────

    def apply_overlays(
        self,
        fig: go.Figure,
        arrays_1d: dict[str, np.ndarray],
        x_var: str | None,
        y_vars: list[str],
    ) -> str:
        """Add overlay traces to *fig* in-place. Returns status string."""
        parts: list[str] = []
        for y_name in y_vars:
            y_data = arrays_1d.get(y_name)
            if y_data is None or len(y_data) == 0:
                continue
            x_raw = arrays_1d.get(x_var) if x_var else None
            x = (x_raw if x_raw is not None and len(x_raw) == len(y_data)
                 else np.arange(len(y_data), dtype=float))
            parts.extend(self._smooth(fig, x, y_data, y_name))
            parts.extend(self._derivative(fig, x, y_data, y_name))
            parts.extend(self._fit(fig, x, y_data, y_name))
        return "  |  ".join(parts)

    # ── private builders ─────────────────────────────────────────────

    def _build_smooth_row(self) -> QHBoxLayout:
        """Create the smoothing controls row."""
        row = QHBoxLayout()
        row.setSpacing(6)
        self.smooth_check = QCheckBox("Smooth", self)
        self.smooth_check.setStyleSheet(_CB_SS)
        self.smooth_window = QSpinBox(self)
        self.smooth_window.setRange(3, 999)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.setValue(5)
        self.smooth_poly = QSpinBox(self)
        self.smooth_poly.setRange(1, 5)
        self.smooth_poly.setValue(2)
        row.addWidget(self.smooth_check)
        row.addStretch(1)
        row.addWidget(QLabel("Window", self))
        row.addWidget(self.smooth_window)
        row.addWidget(QLabel("Poly", self))
        row.addWidget(self.smooth_poly)
        self.smooth_check.toggled.connect(self.changed)
        self.smooth_window.valueChanged.connect(self.changed)
        self.smooth_poly.valueChanged.connect(self.changed)
        return row

    def _build_deriv_row(self) -> QHBoxLayout:
        """Create the derivative toggle row."""
        row = QHBoxLayout()
        self.deriv_check = QCheckBox("Derivative  dy/dx", self)
        self.deriv_check.setStyleSheet(_CB_SS)
        self.deriv_check.toggled.connect(self.changed)
        row.addWidget(self.deriv_check)
        row.addStretch(1)
        return row

    def _build_fit_rows(self) -> QVBoxLayout:
        """Create the curve-fit controls, model selector, and fit range inputs."""
        col = QVBoxLayout()
        col.setSpacing(4)
        r1 = QHBoxLayout()
        self.fit_check = QCheckBox("Curve Fit", self)
        self.fit_check.setStyleSheet(_CB_SS)
        self.fit_model_combo = AutoCloseComboBox(self)
        for key, info in BUILTIN_MODELS.items():
            self.fit_model_combo.addItem(info["label"], key)
        self.fit_model_combo.addItem("Custom formula", "custom")
        r1.addWidget(self.fit_check)
        r1.addWidget(self.fit_model_combo, 1)
        col.addLayout(r1)

        self._custom_widget = QWidget(self)
        self._custom_widget.setStyleSheet("QWidget { border:none; background:transparent; }")
        cw = QHBoxLayout(self._custom_widget)
        cw.setContentsMargins(0, 0, 0, 0)
        cw.addWidget(QLabel("Formula", self._custom_widget))
        self.fit_custom_edit = QLineEdit(self._custom_widget)
        self.fit_custom_edit.setPlaceholderText("e.g. a*exp(-b*x) + c")
        cw.addWidget(self.fit_custom_edit, 1)
        self._custom_widget.hide()
        col.addWidget(self._custom_widget)

        r2 = QHBoxLayout()
        r2.setSpacing(6)
        self.fit_xmin_edit = QLineEdit(self)
        self.fit_xmin_edit.setPlaceholderText("X min (auto)")
        self.fit_xmin_edit.setMaximumWidth(90)
        self.fit_xmax_edit = QLineEdit(self)
        self.fit_xmax_edit.setPlaceholderText("X max (auto)")
        self.fit_xmax_edit.setMaximumWidth(90)
        r2.addWidget(QLabel("X min", self))
        r2.addWidget(self.fit_xmin_edit)
        r2.addWidget(QLabel("X max", self))
        r2.addWidget(self.fit_xmax_edit)
        r2.addStretch(1)
        col.addLayout(r2)

        self.fit_check.toggled.connect(self.changed)
        self.fit_model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.fit_model_combo.currentIndexChanged.connect(self.changed)
        self.fit_custom_edit.textChanged.connect(self.changed)
        self.fit_xmin_edit.textChanged.connect(self.changed)
        self.fit_xmax_edit.textChanged.connect(self.changed)
        return col

    def _on_model_changed(self) -> None:
        """Show the custom-formula editor only for custom fit models."""
        self._custom_widget.setVisible(self.fit_model_combo.currentData() == "custom")

    # ── overlay helpers ───────────────────────────────────────────────

    def _smooth(self, fig: go.Figure, x: np.ndarray, y: np.ndarray, name: str) -> list[str]:
        """Append a smoothed overlay trace when smoothing is enabled."""
        if not self.smooth_check.isChecked():
            return []
        try:
            from scipy.signal import savgol_filter
            win  = self.smooth_window.value()
            poly = self.smooth_poly.value()
            if len(y) > win and win > poly:
                y_s = savgol_filter(y, win, poly)
                fig.add_trace(go.Scatter(x=x, y=y_s, mode="lines",
                                          name=f"{name} (smooth)",
                                          line={"dash": "dot", "width": 1.5}, opacity=0.85))
        except Exception as exc:
            return [f"Smooth: {exc}"]
        return []

    def _derivative(self, fig: go.Figure, x: np.ndarray, y: np.ndarray, name: str) -> list[str]:
        """Append a derivative overlay trace when derivative mode is enabled."""
        if not self.deriv_check.isChecked():
            return []
        try:
            dy = np.gradient(y, x)
            fig.add_trace(go.Scatter(x=x, y=dy, mode="lines",
                                      name=f"d({name})/dx",
                                      line={"dash": "dashdot", "width": 1.5}, opacity=0.85))
        except Exception as exc:
            return [f"Derivative: {exc}"]
        return []

    def _fit(self, fig: go.Figure, x: np.ndarray, y: np.ndarray, name: str) -> list[str]:
        """Fit the selected model and append the fitted curve overlay."""
        if not self.fit_check.isChecked():
            return []
        try:
            model  = self.fit_model_combo.currentData() or "linear"
            custom = self.fit_custom_edit.text().strip() if model == "custom" else ""
            xmin_t = self.fit_xmin_edit.text().strip()
            xmax_t = self.fit_xmax_edit.text().strip()
            x_min  = float(xmin_t) if xmin_t else None
            x_max  = float(xmax_t) if xmax_t else None
            result = fit_series(x, y, model, custom_formula=custom, x_min=x_min, x_max=x_max)
            # fit_series raises FitError on failure — no "error" key needed
            fig.add_trace(go.Scatter(x=result["x_curve"], y=result["y_curve"],
                                      mode="lines", name=f"{name} fit ({model})",
                                      line={"dash": "dash", "width": 2}))
            params = result.get("params", {})
            uncerts = result.get("uncertainties", {})
            pstr = "  ".join(
                f"{k}={v:.4g}±{uncerts.get(k, 0):.2g}" if uncerts else f"{k}={v:.4g}"
                for k, v in params.items()
            )
            return [f"[{name}] R²={result['r_squared']:.4f}  RMSE={result['rmse']:.4g}  {pstr}"]
        except FitError as exc:
            return [f"Fit: {exc}"]
        except Exception as exc:
            return [f"Fit error: {exc}"]


# ── BaseGraphPanel ────────────────────────────────────────────────────────────

class BaseGraphPanel(QWidget, NamespaceConsumerMixin):
    """
    Abstract base for all graph panels.
    Subclasses implement ``refresh()`` and ``current_figure()``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the shared graph state used by all graph panel variants."""
        super().__init__(parent)
        self._init_namespace_state()
        self._figure: go.Figure = build_notebook_plot_figure({}, None, [], "lines", "", "", "")
        self._latest_title: str = ""
        self._latest_html:  str = ""

    def current_figure(self) -> go.Figure:
        """Return the most recently built figure object."""
        return self._figure

    def set_latest_plot(self, title: str, html: str) -> None:
        """Store the latest executed plot payload and trigger a panel refresh."""
        self._latest_title = title
        self._latest_html  = html
        self.refresh()

    @abstractmethod
    def refresh(self) -> None:
        raise NotImplementedError

    def _render_figure(
        self,
        plot_view   : PlotView,
        fig         : go.Figure,
        graph_width : int,
        graph_height: int,
    ) -> None:
        """Render one Plotly figure into a target `PlotView` widget."""
        html = fig.to_html(
            include_plotlyjs=True,
            full_html=False,
            config={
                "responsive"          : False,
                "displaylogo"         : False,
                "toImageButtonOptions": {
                    "format"          : "png", "filename": "graph",
                    "scale"           : 3, "width": graph_width, "height": graph_height,
                },
            },
        )
        plot_view.setMinimumHeight(graph_height)
        plot_view.set_html(html)


# ── GraphBuilderCard ──────────────────────────────────────────────────────────

class GraphBuilderCard(BaseGraphPanel):
    """
    Full-featured graph builder card for the Graphs tab.
    Composed of six focused sub-widgets; inherits namespace management
    and rendering from BaseGraphPanel.
    """
    remove_requested: Signal = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        show_remove: bool = False,
        show_analysis: bool = True,
    ) -> None:
        """Create one reusable graph-builder card and all of its subcontrols."""
        super().__init__(parent)
        self._card_title = "Notebook Plot Builder"

        self.setObjectName("graphCard")
        self.setStyleSheet(_CARD_STYLE)

        # ── sub-widgets ───────────────────────────────────────────────
        self._data_source   = DataSourceWidget(self)
        self._axis_selector = AxisSelectorWidget(self)
        self._series_style  = SeriesStyleWidget(self)
        self._plot_style    = PlotStyleWidget(self)
        self._axis_labels   = AxisLabelsWidget(self)
        self._analysis      = AnalysisWidget(self)
        self._plot_view     = PlotView(self)
        self._status_label  = QLabel("Waiting for arrays", self)
        self._status_label.setStyleSheet("color:#64748b; font-size:12px;")

        self._build_layout(show_remove, show_analysis)
        self._connect_signals()
        self.refresh()

    # ── public API ───────────────────────────────────────────────────

    def set_namespace(
        self, arrays_1d: dict[str, np.ndarray], arrays_2d: dict[str, np.ndarray]
    ) -> None:
        """Inject the current arrays and refresh every selector from them."""
        self._nb_arrays_1d = arrays_1d
        self._nb_arrays_2d = arrays_2d
        self._data_source.update_notebook_arrays(arrays_1d)
        self._axis_selector.populate(self._data_source.active_arrays_1d(), arrays_2d)
        self._axis_selector.update_evo_slider_range(arrays_2d)
        status = (
            f"{len(arrays_1d)} 1D array(s), {len(arrays_2d)} 2D array(s)"
            if arrays_1d or arrays_2d else "No numeric arrays available"
        )
        self._status_label.setText(status)
        self.refresh()

    def set_card_title(self, title: str) -> None:
        """Update the visible card title shown in the header."""
        self._card_title = title
        self._title_label.setText(title)

    def set_remove_enabled(self, enabled: bool) -> None:
        """Show or hide the remove affordance for this graph card."""
        self._remove_btn.setVisible(enabled)

    def refresh_plot(self) -> None:
        """Public alias for `refresh()` used by compatibility callers."""
        self.refresh()

    # ── BaseGraphPanel override ───────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the current figure from the selected controls and data source."""
        mode      = self._axis_selector.mode()
        arrays_1d = self._data_source.active_arrays_1d()
        style     = self._plot_style.style_options()
        w, h      = self._plot_style.graph_size()

        if mode == "evolution":
            matrix_var, time_var, value_var, step = self._axis_selector.evolution_params()
            self._figure = build_notebook_evolution_figure(
                arrays_1d, self._nb_arrays_2d,
                matrix_var, time_var, value_var, step,
                self._axis_selector.plot_type(),
                self._axis_labels.title(),
                self._axis_labels.x_label(),
                self._axis_labels.y_label(),
                style,
            )
            overlay_status = ""
            base_status = (
                f"Evolution: matrix={matrix_var or 'none'}  step={step}"
            )
        else:
            y_vars = self._axis_selector.selected_y_vars()
            self._series_style.update_y_vars(y_vars, self._axis_selector.plot_type())
            self._figure = build_notebook_plot_figure(
                arrays_1d,
                self._axis_selector.selected_x(),
                y_vars,
                self._axis_selector.plot_type(),
                self._axis_labels.title(),
                self._axis_labels.x_label(),
                self._axis_labels.y_label(),
                self._series_style.style_map(),
                style,
            )
            overlay_status = self._analysis.apply_overlays(
                self._figure, arrays_1d,
                self._axis_selector.selected_x(), y_vars,
            )
            src = "CSV" if self._data_source.is_csv() else "NB"
            base_status = (
                f"[{src}]  x={self._axis_selector.selected_x() or 'index'}"
                f"  y={', '.join(y_vars) if y_vars else 'none'}"
            )

        self._render_figure(self._plot_view, self._figure, w, h)
        full_status = f"{base_status}  |  {overlay_status}" if overlay_status else base_status
        self._status_label.setText(full_status)

    # ── layout builder ────────────────────────────────────────────────

    def _build_layout(self, show_remove: bool, show_analysis: bool) -> None:
        """
        Constructs the layout for the graph builder card, including settings, 
        preview panels, and optional sections for analysis or removal.
        
        Args:
            show_remove (bool): Whether to show the remove button in the card header.
            show_analysis (bool): Whether to include the analysis section in the settings panel.
        """
        card_layout = QHBoxLayout(self)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(12)

        # settings panel (left)
        settings = QWidget(self)
        settings.setStyleSheet("background:transparent;")
        min_w = 640 if show_analysis else 380
        max_w = 780 if show_analysis else 480
        settings.setMinimumWidth(min_w)
        settings.setMaximumWidth(max_w)
        sl = QVBoxLayout(settings)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)

        # header
        hdr = QHBoxLayout()
        self._title_label = QLabel(self._card_title, settings)
        self._title_label.setStyleSheet(_TITLE_SS)
        hdr.addWidget(self._title_label)
        hdr.addStretch(1)
        self._remove_btn = QLabel("✕", settings)
        self._remove_btn.setStyleSheet(
            "color:#94a3b8; font-size:16px; border:none; padding:0 6px; border-radius:4px;"
        )
        self._remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_btn.mousePressEvent = (  # type: ignore[assignment]
            lambda _e: self.remove_requested.emit(self)
        )
        self._remove_btn.setVisible(show_remove)
        hdr.addWidget(self._remove_btn)
        sl.addLayout(hdr)

        plot_setup_body = QWidget(settings)
        plot_setup_layout = QVBoxLayout(plot_setup_body)
        plot_setup_layout.setContentsMargins(0, 0, 0, 0)
        plot_setup_layout.setSpacing(8)
        plot_setup_layout.addWidget(self._axis_selector)
        plot_setup_layout.addWidget(self._series_style)

        if show_analysis:
            sl.addWidget(
                self._make_section(
                    "Data",
                    self._data_source,
                    settings,
                )
            )
        sl.addWidget(
            self._make_section(
                "Plot Setup",
                plot_setup_body,
                settings,
            )
        )
        sl.addWidget(
            self._make_section(
                "Appearance",
                self._plot_style,
                settings,
            )
        )
        sl.addWidget(
            self._make_section(
                "Labels",
                self._axis_labels,
                settings,
            )
        )
        if show_analysis:
            sl.addWidget(
                self._make_section(
                    "Analysis",
                    self._analysis,
                    settings,
                )
            )
        sl.addStretch(1)

        # preview panel (right)
        preview = QWidget(self)
        preview.setObjectName("graphPreviewCard")
        pvl = QVBoxLayout(preview)
        pvl.setContentsMargins(12, 12, 12, 12)
        pvl.setSpacing(6)
        preview_title = QLabel("Live Preview", preview)
        preview_title.setStyleSheet(_SECTION_TITLE_SS)
        pvl.addWidget(preview_title)
        pvl.addWidget(self._status_label)
        pvl.addWidget(self._plot_view, 1)

        card_layout.addWidget(settings, 0)
        card_layout.addWidget(preview, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # expose alias for compat
        self.controller_status = self._status_label
        self.controller_plot   = self._plot_view

    def _make_section(
        self,
        title: str,
        body: QWidget,
        parent: QWidget,
    ) -> QWidget:
        """Wrap one control group in the standard titled section card UI."""
        print(f"[debug][graph-builder-card] make_section title={title!r}", flush=True)
        section = QWidget(parent)
        section.setObjectName("graphSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        title_label = QLabel(title, section)
        title_label.setStyleSheet(_SECTION_TITLE_SS)
        layout.addWidget(title_label)
        layout.addWidget(body)
        return section

    def _connect_signals(self) -> None:
        """
        Connects signals to their respective slots for handling changes 
        from user interactions. Any changes in these components will 
        automatically trigger updates to the graph or internal state.
        """
        self._data_source.changed.connect(self._on_source_changed)
        self._axis_selector.changed.connect(self.refresh)
        self._series_style.changed.connect(self.refresh)
        self._plot_style.changed.connect(self.refresh)
        self._axis_labels.changed.connect(self.refresh)
        self._analysis.changed.connect(self.refresh)

    def _on_source_changed(self) -> None:
        """Reconfigure selectors after switching between notebook and CSV data."""
        is_csv = self._data_source.is_csv()
        if is_csv:
            self._axis_selector.mode_combo.blockSignals(True)
            idx = self._axis_selector.mode_combo.findData("series")
            if idx >= 0:
                self._axis_selector.mode_combo.setCurrentIndex(idx)
            self._axis_selector.mode_combo.blockSignals(False)
            self._axis_selector.mode_combo.setEnabled(False)
        else:
            self._axis_selector.mode_combo.setEnabled(True)
        self._axis_selector.populate(self._data_source.active_arrays_1d(), self._nb_arrays_2d)
        self.refresh()


# ── NotebookGraphWorkspace ────────────────────────────────────────────────────

class NotebookGraphWorkspace(QWidget):
    """Notebook-tab graph workspace — holds one or more GraphBuilderCards."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the notebook-side stacked graph workspace."""
        super().__init__(parent)
        self._nb_arrays_1d: dict[str, np.ndarray] = {}
        self._nb_arrays_2d: dict[str, np.ndarray] = {}
        self._latest_title = ""
        self._latest_html  = ""
        self._cards: list[GraphBuilderCard] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self._root_layout = root

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Graphs", self))
        hdr.addStretch(1)
        root.addLayout(hdr)
        root.addStretch(1)
        self.add_graph_card()

    # ── public ───────────────────────────────────────────────────────

    def primary_card(self) -> GraphBuilderCard:
        """Return the first graph card in the workspace."""
        return self._cards[0]

    def cards(self) -> list[GraphBuilderCard]:
        """Return a snapshot list of all graph cards in the workspace."""
        return list(self._cards)

    def card_count(self) -> int:
        """Return the number of graph cards currently in the workspace."""
        return len(self._cards)

    def add_graph_card(self) -> GraphBuilderCard:
        """Append a new graph card below the existing notebook graph cards."""
        card = GraphBuilderCard(self, show_remove=bool(self._cards), show_analysis=False)
        card.remove_requested.connect(self.remove_graph_card)
        card.set_namespace(self._nb_arrays_1d, self._nb_arrays_2d)
        card.set_latest_plot(self._latest_title, self._latest_html)
        self._cards.append(card)
        self._root_layout.insertWidget(max(self._root_layout.count() - 1, 0), card)
        self._renumber_cards()
        return card

    def remove_graph_card(self, card: GraphBuilderCard) -> None:
        """Remove one graph card while keeping at least one card alive."""
        if card not in self._cards or len(self._cards) <= 1:
            return
        self._cards.remove(card)
        self._root_layout.removeWidget(card)
        card.hide()
        card.deleteLater()
        self._renumber_cards()

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        """Push a new notebook namespace snapshot into every graph card."""
        self._nb_arrays_1d, self._nb_arrays_2d = extract_notebook_array_variables(namespace)
        for card in self._cards:
            card.set_namespace(self._nb_arrays_1d, self._nb_arrays_2d)

    def set_latest_plot(self, title: str, html: str) -> None:
        """Push the latest executed plot payload into every graph card."""
        self._latest_title = title
        self._latest_html  = html
        for card in self._cards:
            card.set_latest_plot(title, html)

    # ── private ──────────────────────────────────────────────────────

    def _renumber_cards(self) -> None:
        """Rename cards sequentially and refresh their remove-button state."""
        for i, card in enumerate(self._cards, start=1):
            card.set_card_title(f"Graph {i}")
            card.set_remove_enabled(len(self._cards) > 1)


# ── NotebookPlotPanel ─────────────────────────────────────────────────────────

class NotebookPlotPanel(QWidget):
    """
    Graphs-tab container.
    Holds the main GraphBuilderCard plus any user-added extra cards.
    """

    def __init__(self, parent: QWidget | None = None, layout_mode: str = "advanced") -> None:
        """Create the graphs-tab container with one main graph card and outputs area."""
        super().__init__(parent)
        self.layout_mode = layout_mode
        self.setStyleSheet("background:#ffffff;")
        self._nb_arrays_1d: dict[str, np.ndarray] = {}
        self._nb_arrays_2d: dict[str, np.ndarray] = {}
        self._extra_cards:  list[GraphBuilderCard] = []
        self._output_widgets: dict[str, tuple[QLabel, QWidget, str]] = {}
        self._output_empty_label: QLabel | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self._root_layout = root

        # header
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Graphs", self))
        hdr.addStretch(1)
        add_btn = QLabel("+ Add Graph", self)
        add_btn.setStyleSheet(
            "color:#1d4ed8; font-weight:600; font-size:13px;"
            "padding:4px 10px; border:1px solid #93c5fd; border-radius:6px; background:#eff6ff;"
        )
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.mousePressEvent = lambda _e: self._add_extra_graph()  # type: ignore[assignment]
        hdr.addWidget(add_btn)
        root.addLayout(hdr)

        # main card (with analysis features)
        # Initialize and configure the main graph card with integrated analysis features
        self.main_card              = GraphBuilderCard(self, show_remove=False, show_analysis=True)
        root.addWidget(self.main_card)
        
        # Extract and expose sub-widgets and components from the main graph card
        # This allows for direct access to these components for further interactions or configurations
        self.mode_combo             = self.main_card._axis_selector.mode_combo
        self.graph_size_combo       = self.main_card._plot_style.size_combo
        self.series_controls        = self.main_card._axis_selector._series_widget
        self.evolution_controls     = self.main_card._axis_selector._evo_widget
        self.x_combo                = self.main_card._axis_selector.x_combo
        self.y_combo                = self.main_card._axis_selector.y_combo
        self.plot_type_combo        = self.main_card._axis_selector.plot_type_combo
        self.evolution_matrix_combo = self.main_card._axis_selector.evo_matrix_combo
        self.evolution_time_combo   = self.main_card._axis_selector.evo_time_combo
        self.evolution_value_combo  = self.main_card._axis_selector.evo_value_combo
        self.evolution_step_slider  = self.main_card._axis_selector.evo_step_slider
        self.evolution_step_label   = self.main_card._axis_selector.evo_step_label
        self.font_size_combo        = self.main_card._plot_style.font_size_combo
        self.line_width_combo       = self.main_card._plot_style.line_width_combo
        self.marker_size_combo      = self.main_card._plot_style.marker_size_combo
        self.show_grid_check        = self.main_card._plot_style.grid_check
        self.show_box_check         = self.main_card._plot_style.box_check
        self.ticks_inside_check     = self.main_card._plot_style.ticks_check
        self.minor_ticks_check      = self.main_card._plot_style.minor_check
        self.title_edit             = self.main_card._axis_labels.title_edit
        self.x_label_edit           = self.main_card._axis_labels.x_label_edit
        self.y_label_edit           = self.main_card._axis_labels.y_label_edit
        self.controller_plot        = self.main_card.controller_plot
        self.controller_status      = self.main_card.controller_status
        self._series_style_widgets  = self.main_card._series_style._rows

        # cell outputs
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
        """Refresh all graph cards from the current notebook namespace."""
        self._nb_arrays_1d, self._nb_arrays_2d = extract_notebook_array_variables(namespace)
        self.main_card.set_namespace(self._nb_arrays_1d, self._nb_arrays_2d)
        for card in self._extra_cards:
            card.set_namespace(self._nb_arrays_1d, self._nb_arrays_2d)

    def current_controller_figure(self) -> go.Figure:
        """Return the figure currently built by the main graph card."""
        return self.main_card.current_figure()

    def refresh_controller_plot(self) -> None:
        """Force the main graph card to rebuild its current plot."""
        self.main_card.refresh()

    def output_count(self) -> int:
        """Return the number of stored notebook cell graph outputs."""
        return len(self._output_widgets)

    def output_titles(self) -> list[str]:
        """Return the visible titles for synced notebook output graphs."""
        return [t.text() for t, _w, _h in self._output_widgets.values()]

    # ── extra graph management ────────────────────────────────────────

    def _add_extra_graph(self) -> None:
        """Append another advanced graph card to the graphs tab."""
        card = GraphBuilderCard(self, show_remove=True, show_analysis=True)
        card.remove_requested.connect(self._remove_extra_graph)
        card.set_namespace(self._nb_arrays_1d, self._nb_arrays_2d)
        self._extra_cards.append(card)
        self._root_layout.addWidget(card)

    def _remove_extra_graph(self, card: GraphBuilderCard) -> None:
        """Remove one extra graph card from the graphs tab."""
        if card in self._extra_cards:
            self._extra_cards.remove(card)
        self._root_layout.removeWidget(card)
        card.hide()
        card.deleteLater()

    # ── cell output sync ─────────────────────────────────────────────

    def sync_cell_outputs(self, cells: list[Any]) -> None:
        """Mirror graph-like notebook cell outputs into the outputs section."""
        desired: list[str] = []
        for cell in cells:
            result = getattr(cell, "last_result", None)
            if result is None:
                continue
            for idx, output in enumerate(result.outputs):
                if output.kind == "plotly":
                    key = f"{cell.cell_id}:plotly:{idx}"
                elif output.kind == "html" and "data:image" in output.data.get("html", ""):
                    key = f"{cell.cell_id}:image:{idx}"
                else:
                    continue
                desired.append(key)
                html = output.data["html"]
                src_lines = [ln.strip() for ln in cell.source().splitlines() if ln.strip()]
                raw = src_lines[0] if src_lines else getattr(cell, "cell_id", "plot")
                title_text = raw[:45] + "..." if len(raw) > 48 else raw
                existing = self._output_widgets.get(key)
                if existing is None:
                    title_lbl = QLabel(title_text, self.outputs_container)
                    title_lbl.setStyleSheet("color:#001f41; font-weight:600;")
                    plot = PlotView(self.outputs_container)
                    plot.set_html(html)
                    self._output_widgets[key] = (title_lbl, plot, html)
                else:
                    title_lbl, plot, old_html = existing
                    title_lbl.setText(title_text)
                    if old_html != html:
                        plot.set_html(html)
                        self._output_widgets[key] = (title_lbl, plot, html)
        for key in [k for k in self._output_widgets if k not in desired]:
            t, w, _ = self._output_widgets.pop(key)
            t.hide(); w.hide()
            t.deleteLater(); w.deleteLater()
        while self.outputs_layout.count():
            item = self.outputs_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        if self._output_empty_label is not None:
            self._output_empty_label.deleteLater()
            self._output_empty_label = None
        if not desired:
            empty = QLabel("Run a plotting cell to see output graphs here.",
                           self.outputs_container)
            empty.setStyleSheet("color:#64748b; font-style:italic;")
            self.outputs_layout.addWidget(empty)
            self._output_empty_label = empty
        else:
            for key in desired:
                t, w, _ = self._output_widgets[key]
                self.outputs_layout.addWidget(t)
                self.outputs_layout.addWidget(w)
        self.outputs_layout.addStretch(1)


# ── QuickGraphPreviewPanel ────────────────────────────────────────────────────

class QuickGraphPreviewPanel(BaseGraphPanel):
    """
    Lightweight graph preview for the Notebook-tab sidebar.
    Inherits namespace management and rendering from BaseGraphPanel.
    Only exposes mode, X/Y variable selection, and a plot view.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the lightweight notebook-side quick graph preview widget."""
        super().__init__(parent)
        self._arrays_1d: dict[str, np.ndarray] = {}
        self._arrays_2d: dict[str, np.ndarray] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # header
        hdr = QHBoxLayout()
        lbl = QLabel("Quick Graph Preview", self)
        lbl.setStyleSheet("color:#001f41; font-weight:700; font-size:15px;")
        hdr.addWidget(lbl)
        hdr.addStretch(1)
        root.addLayout(hdr)

        self._status_label = QLabel("Use the quick controls to preview graphs.", self)
        self._status_label.setStyleSheet("color:#64748b; font-size:12px;")
        root.addWidget(self._status_label)

        # controls card
        card = QWidget(self)
        card.setStyleSheet(
            "QWidget { background:#ffffff; border:1px solid #d1dce8; }"
            "QLabel { border:none; background:transparent; " + _LBL_SS + " }"
            "QComboBox, QLineEdit { background:#ffffff; border:1px solid #d1dce8;"
            " border-radius:6px; padding:4px 6px; }"
        )
        cl = QVBoxLayout(card)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(8)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self._mode_combo = AutoCloseComboBox(card)
        self._mode_combo.addItem("Series (1D)", "series")
        self._mode_combo.addItem("Evolution (2D)", "evolution")
        mode_row.addWidget(QLabel("Mode", card))
        mode_row.addWidget(self._mode_combo, 1)
        cl.addLayout(mode_row)

        # series controls
        self._series_widget = QWidget(card)
        sr = QHBoxLayout(self._series_widget)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(8)
        self._x_combo = AutoCloseComboBox(self._series_widget)
        self._x_combo.addItem("Index", "")
        self._plot_type_combo = AutoCloseComboBox(self._series_widget)
        for lbl_t, val in PLOT_TYPE_OPTIONS:
            self._plot_type_combo.addItem(lbl_t, val)
        self._y_combo = CheckableComboBox(self._series_widget)
        for lbl_text, widget, stretch in (
            ("X variable", self._x_combo, 1),
            ("Plot type", self._plot_type_combo, 1),
            ("Y variable(s)", self._y_combo, 2),
        ):
            blk = QWidget(self._series_widget)
            bl = QVBoxLayout(blk)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(3)
            bl.addWidget(QLabel(lbl_text, blk))
            bl.addWidget(widget)
            sr.addWidget(blk, stretch)
        cl.addWidget(self._series_widget)

        # evolution controls
        self._evo_widget = QWidget(card)
        eg = QGridLayout(self._evo_widget)
        eg.setContentsMargins(0, 0, 0, 0)
        eg.setHorizontalSpacing(8)
        eg.setVerticalSpacing(6)
        self._evo_matrix_combo = AutoCloseComboBox(self._evo_widget)
        self._evo_matrix_combo.addItem("Select 2D array", "")
        self._evo_time_combo = AutoCloseComboBox(self._evo_widget)
        self._evo_time_combo.addItem("Row index", "")
        self._evo_value_combo = AutoCloseComboBox(self._evo_widget)
        self._evo_value_combo.addItem("Column index", "")
        self._evo_step_slider = QSlider(Qt.Orientation.Horizontal, self._evo_widget)
        self._evo_step_slider.setRange(0, 0)
        self._evo_step_label = QLabel("Step 0 / 0", self._evo_widget)
        for row_i, (row_lbl, widget) in enumerate((
            ("Evolution array", self._evo_matrix_combo),
            ("Time axis", self._evo_time_combo),
            ("Value axis", self._evo_value_combo),
            ("Time step", self._evo_step_slider),
            ("Selected step", self._evo_step_label),
        )):
            eg.addWidget(QLabel(row_lbl, self._evo_widget), row_i, 0)
            eg.addWidget(widget, row_i, 1)
        self._evo_widget.hide()
        cl.addWidget(self._evo_widget)
        root.addWidget(card)

        # plot view
        self._plot_view = PlotView(self)
        self._plot_view.setMinimumHeight(360)
        self._plot_view.setMaximumHeight(360)
        self._plot_view.hide()
        root.addWidget(self._plot_view)
        self._empty_label = QLabel("No plot output yet.", self)
        self._empty_label.setStyleSheet("color:#64748b; font-style:italic;")
        root.addWidget(self._empty_label)
        root.addStretch(1)

        # signals
        self._mode_combo.currentIndexChanged.connect(self._sync_mode)
        self._x_combo.currentIndexChanged.connect(self.refresh)
        self._y_combo.checkedItemsChanged.connect(self.refresh)
        self._plot_type_combo.currentIndexChanged.connect(self.refresh)
        self._evo_matrix_combo.currentIndexChanged.connect(self._on_matrix_changed)
        self._evo_time_combo.currentIndexChanged.connect(self.refresh)
        self._evo_value_combo.currentIndexChanged.connect(self.refresh)
        self._evo_step_slider.valueChanged.connect(self._on_step_changed)

    # ── public API ───────────────────────────────────────────────────

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        """Refresh preview selectors from the latest notebook namespace."""
        self._arrays_1d, self._arrays_2d = extract_notebook_array_variables(namespace)
        self._populate_combos()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the quick preview using live selections or fallback plot HTML."""
        mode = self._mode_combo.currentData() or "series"
        style = {"graph_width": None, "graph_height": 360,
                 "font_size": 14, "line_width": 2, "marker_size": 8,
                 "show_grid": True, "show_box": True,
                 "ticks_inside": True, "show_minor_ticks": True}

        if mode == "evolution":
            matrix_var = self._evo_matrix_combo.currentData() or None
            self._figure = build_notebook_evolution_figure(
                self._arrays_1d, self._arrays_2d,
                matrix_var,
                self._evo_time_combo.currentData() or None,
                self._evo_value_combo.currentData() or None,
                self._evo_step_slider.value(),
                self._plot_type_combo.currentData() or "lines",
                "", "", "", style,
            )
            has_plot = bool(matrix_var and matrix_var in self._arrays_2d)
        else:
            y_vars = [v for v in self._y_combo.checked_values() if isinstance(v, str)]
            x_var  = self._x_combo.currentData() or None
            self._figure = build_notebook_plot_figure(
                self._arrays_1d, x_var, y_vars,
                self._plot_type_combo.currentData() or "lines",
                "", "", "", {}, style,
            )
            has_plot = bool(y_vars)

        if not has_plot and self._latest_html:
            self._plot_view.setVisible(True)
            self._empty_label.setVisible(False)
            self._plot_view.set_html(self._latest_html)
            self._status_label.setText(
                f"Latest: {self._latest_title or 'notebook plot'}"
            )
            return

        html = self._figure.to_html(
            include_plotlyjs=True, full_html=False,
            config={"responsive": True, "displaylogo": False},
        )
        self._plot_view.setVisible(has_plot)
        self._empty_label.setVisible(not has_plot)
        if has_plot:
            self._plot_view.set_html(html)
        self._status_label.setText(
            f"x={self._x_combo.currentData() or 'index'}"
            if has_plot else "Run code to populate arrays."
        )

    # ── private ──────────────────────────────────────────────────────

    def _populate_combos(self) -> None:
        """Rebuild every quick-preview combo box from the current arrays."""
        prev_x      = self._x_combo.currentData()
        prev_y      = set(self._y_combo.checked_values())
        prev_matrix = self._evo_matrix_combo.currentData()
        prev_time   = self._evo_time_combo.currentData()
        prev_value  = self._evo_value_combo.currentData()

        for combo in (self._x_combo, self._y_combo, self._evo_matrix_combo,
                      self._evo_time_combo, self._evo_value_combo):
            combo.blockSignals(True)

        self._x_combo.clear()
        self._x_combo.addItem("Index", "")
        self._y_combo.clear()
        self._evo_matrix_combo.clear()
        self._evo_matrix_combo.addItem("Select 2D array", "")
        self._evo_time_combo.clear()
        self._evo_time_combo.addItem("Row index", "")
        self._evo_value_combo.clear()
        self._evo_value_combo.addItem("Column index", "")

        for name in self._arrays_1d:
            self._x_combo.addItem(name, name)
            self._evo_time_combo.addItem(name, name)
            self._evo_value_combo.addItem(name, name)
            self._y_combo.add_check_item(name, name, checked=name in prev_y)
        for name in self._arrays_2d:
            self._evo_matrix_combo.addItem(name, name)

        for combo, prev in (
            (self._x_combo, prev_x), (self._evo_matrix_combo, prev_matrix),
            (self._evo_time_combo, prev_time), (self._evo_value_combo, prev_value),
        ):
            idx = combo.findData(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        if not prev_y and self._arrays_1d:
            names = sorted(self._arrays_1d)
            x_name = self._x_combo.currentData() or ""
            default = names[1] if len(names) > 1 and x_name == names[0] else names[0]
            self._y_combo.set_checked_values([default])

        for combo in (self._x_combo, self._y_combo, self._evo_matrix_combo,
                      self._evo_time_combo, self._evo_value_combo):
            combo.blockSignals(False)

        self._on_matrix_changed()

    def _sync_mode(self) -> None:
        """Swap quick-preview controls between series and evolution layouts."""
        mode = self._mode_combo.currentData() or "series"
        self._series_widget.setVisible(mode == "series")
        self._evo_widget.setVisible(mode == "evolution")
        self.refresh()

    def _on_matrix_changed(self) -> None:
        """Resize the quick-preview evolution slider for the selected matrix."""
        name   = self._evo_matrix_combo.currentData()
        matrix = self._arrays_2d.get(name) if isinstance(name, str) else None
        rows   = int(matrix.shape[0]) if matrix is not None else 0
        max_s  = max(rows - 1, 0)
        self._evo_step_slider.blockSignals(True)
        self._evo_step_slider.setRange(0, max_s)
        if self._evo_step_slider.value() > max_s:
            self._evo_step_slider.setValue(max_s)
        self._evo_step_slider.blockSignals(False)
        self._update_step_label()
        self.refresh()

    def _on_step_changed(self, *_: Any) -> None:
        self._update_step_label()
        self.refresh()

    def _update_step_label(self) -> None:
        cur = self._evo_step_slider.value()
        self._evo_step_label.setText(f"Step {cur} / {self._evo_step_slider.maximum()}")
