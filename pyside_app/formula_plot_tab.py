from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import plotly.graph_objects as go
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from pyside_app.plot_view import PlotView
from utils.formula_parser import evaluate_formula, evaluate_formula_2d, extract_formula_variables


FORMULA_TAB_STYLE = """
#FormulaPlotTab {
    background: #21252b;
}
#FormulaPlotTab QLineEdit, #FormulaPlotTab QComboBox {
    min-height: 32px;
    padding: 6px 10px;
    border: 1px solid #3e4451;
    border-radius: 7px;
    background: #2c313a;
    color: #d7dae0;
}
#FormulaPlotTab QLabel {
    color: #d7dae0;
}
""".strip()

FORMULA_BUTTON_STYLE = """
QPushButton {
    min-height: 32px;
    padding: 6px 14px;
    border: none;
    border-radius: 8px;
    background: #2c313a;
    color: #d7dae0;
    font-weight: 600;
}
QPushButton:hover {
    background: #3e4451;
}
""".strip()

FORMULA_STATUS_READY_STYLE = "color: #d7dae0; font-weight: 600; padding: 6px 0;"
FORMULA_STATUS_UPDATED_STYLE = "color: #98c379; font-weight: 600; padding: 6px 0;"
FORMULA_STATUS_ERROR_STYLE = "color: #e06c75; font-weight: 600; padding: 6px 0;"

FORMULA_SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 6px;
    border-radius: 3px;
    background: #cbd5e1;
}
QSlider::sub-page:horizontal {
    border-radius: 3px;
    background: #61afef;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
    background: #b60021;
}
""".strip()

PARAMETER_PRESETS = {
    "a": {"value": 1.0, "min": -5.0, "max": 5.0, "step": 0.1},
    "b": {"value": 1.0, "min": -5.0, "max": 5.0, "step": 0.1},
    "c": {"value": 0.0, "min": -10.0, "max": 10.0, "step": 0.1},
    "q": {"value": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
    "r": {"value": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
    "sigma": {"value": 1.5, "min": 0.1, "max": 5.0, "step": 0.1},
    "tau": {"value": 2.5, "min": 0.1, "max": 10.0, "step": 0.1},
    "x0": {"value": 0.0, "min": -5.0, "max": 5.0, "step": 0.1},
    "y0": {"value": 0.0, "min": -5.0, "max": 5.0, "step": 0.1},
}

DEFAULT_PARAMETER_PRESET = {"value": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}

PRESET_2D_OPTIONS = [
    ("Gaussian hill", "gaussian_hill"),
    ("Saddle", "saddle"),
    ("Paraboloid", "paraboloid"),
    ("Radial decay", "radial_decay"),
    ("Periodic surface", "periodic_surface"),
]

PRESET_2D_DEFAULTS = {
    "gaussian_hill": {
        "expression_2d": "a*exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))",
        "label_2d": "Gaussian hill",
        "x_min": -6.0,
        "x_max": 6.0,
        "y_min": -6.0,
        "y_max": 6.0,
        "x_points_2d": 100,
        "y_points_2d": 100,
        "x_axis_title": "x",
        "y_axis_title": "y",
        "z_axis_title": "height",
        "surface_colorscale": "Plasma",
        "params": {
            "a": {"value": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
            "x0": {"value": 0.0, "min": -5.0, "max": 5.0, "step": 0.1},
            "y0": {"value": 0.0, "min": -5.0, "max": 5.0, "step": 0.1},
            "sigma": {"value": 1.5, "min": 0.1, "max": 5.0, "step": 0.1},
        },
    },
    "saddle": {
        "expression_2d": "x**2 - y**2",
        "label_2d": "Saddle surface",
        "x_min": -4.0,
        "x_max": 4.0,
        "y_min": -4.0,
        "y_max": 4.0,
        "x_points_2d": 90,
        "y_points_2d": 90,
        "x_axis_title": "x",
        "y_axis_title": "y",
        "z_axis_title": "z",
        "surface_colorscale": "RdBu",
    },
    "paraboloid": {
        "expression_2d": "a*(x**2 + y**2) + c",
        "label_2d": "Paraboloid",
        "x_min": -4.0,
        "x_max": 4.0,
        "y_min": -4.0,
        "y_max": 4.0,
        "x_points_2d": 90,
        "y_points_2d": 90,
        "x_axis_title": "x",
        "y_axis_title": "y",
        "z_axis_title": "z",
        "surface_colorscale": "Cividis",
        "params": {
            "a": {"value": 1.0, "min": -2.0, "max": 2.0, "step": 0.1},
            "c": {"value": 0.0, "min": -5.0, "max": 5.0, "step": 0.1},
        },
    },
    "radial_decay": {
        "expression_2d": "a*exp(-sqrt(x**2 + y**2)/tau)",
        "label_2d": "Radial decay",
        "x_min": -8.0,
        "x_max": 8.0,
        "y_min": -8.0,
        "y_max": 8.0,
        "x_points_2d": 100,
        "y_points_2d": 100,
        "x_axis_title": "x",
        "y_axis_title": "y",
        "z_axis_title": "z",
        "surface_colorscale": "Turbo",
        "params": {
            "a": {"value": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
            "tau": {"value": 2.5, "min": 0.1, "max": 10.0, "step": 0.1},
        },
    },
    "periodic_surface": {
        "expression_2d": "sin(x)*cos(y)",
        "label_2d": "sin(x)*cos(y)",
        "x_min": -5.0,
        "x_max": 5.0,
        "y_min": -5.0,
        "y_max": 5.0,
        "x_points_2d": 80,
        "y_points_2d": 80,
        "x_axis_title": "x",
        "y_axis_title": "y",
        "z_axis_title": "f(x,y)",
        "surface_colorscale": "Viridis",
    },
}


@dataclass
class FormulaPlotState:
    panel_type: str = "1d"
    preset_2d: str = "periodic_surface"
    x_min: float = -10.0
    x_max: float = 10.0
    points: int = 400
    y_min: float = -5.0
    y_max: float = 5.0
    x_points_2d: int = 80
    y_points_2d: int = 80
    expression_2d: str = "sin(x) * cos(y)"
    label_2d: str = "sin(x) * cos(y)"
    x_axis_title: str = "x"
    y_axis_title: str = "f(x)"
    z_axis_title: str = "f(x,y)"
    display_mode_2d: str = "surface"
    surface_colorscale: str = "Viridis"
    show_derivative: bool = False
    show_second_derivative: bool = False
    show_antiderivative: bool = False
    formulas: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "expression": "sin(x)",
                "label": "sin(x)",
                "visible": True,
                "color": "black",
                "dash": "solid",
                "width": 3.0,
            }
        ]
    )
    params: dict[str, dict[str, Any]] = field(default_factory=lambda: {"a": {"value": 1.0}})


def parameter_preset_for_name(name: str) -> dict[str, Any]:
    key = (name or "").strip()
    preset = dict(PARAMETER_PRESETS.get(key, DEFAULT_PARAMETER_PRESET))
    print(f"[debug][formula-plot] parameter_preset_for_name name={key!r} preset={preset!r}", flush=True)
    return preset


def apply_2d_preset(state: FormulaPlotState, preset_name: str) -> None:
    key = preset_name if preset_name in PRESET_2D_DEFAULTS else "periodic_surface"
    preset = PRESET_2D_DEFAULTS[key]
    print(f"[debug][formula-plot] apply_2d_preset start preset={key!r}", flush=True)
    state.preset_2d = key
    state.expression_2d = str(preset["expression_2d"])
    state.label_2d = str(preset["label_2d"])
    state.x_min = float(preset["x_min"])
    state.x_max = float(preset["x_max"])
    state.y_min = float(preset["y_min"])
    state.y_max = float(preset["y_max"])
    state.x_points_2d = int(preset["x_points_2d"])
    state.y_points_2d = int(preset["y_points_2d"])
    state.x_axis_title = str(preset["x_axis_title"])
    state.y_axis_title = str(preset["y_axis_title"])
    state.z_axis_title = str(preset["z_axis_title"])
    state.surface_colorscale = str(preset["surface_colorscale"])
    preset_params = {name: dict(spec) for name, spec in preset.get("params", {}).items()}
    if preset_params:
        state.params.update(preset_params)
    print(
        f"[debug][formula-plot] apply_2d_preset done expression={state.expression_2d!r} "
        f"x_range=({state.x_min}, {state.x_max}) y_range=({state.y_min}, {state.y_max}) params={state.params!r}",
        flush=True,
    )


def _parameter_values(state: FormulaPlotState) -> dict[str, Any]:
    values = {name: spec.get("value", 1.0) for name, spec in (state.params or {}).items()}
    print(f"[debug][formula-plot] parameter_values values={values!r}", flush=True)
    return values


def build_formula_figure(state: FormulaPlotState) -> go.Figure:
    print(f"[debug][formula-plot] build_formula_figure panel_type={state.panel_type!r}", flush=True)
    x_values = np.linspace(state.x_min, state.x_max, int(state.points))
    param_values = _parameter_values(state)
    figure = go.Figure()

    for formula in state.formulas:
        if not formula.get("visible", True):
            continue
        expression = formula.get("expression", "sin(x)")
        label = formula.get("label") or expression
        if label == "sin(x)" and expression != "sin(x)":
            label = expression
        y_values = evaluate_formula(expression, x_values, param_values)
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=label,
                line={"color": formula.get("color", "black"), "dash": formula.get("dash", "solid"), "width": formula.get("width", 3.0)},
            )
        )
        if state.show_derivative:
            derivative = np.gradient(y_values, x_values)
            figure.add_trace(go.Scatter(x=x_values, y=derivative, mode="lines", name=f"Derivative: {label}", line={"dash": "dot"}))
        if state.show_second_derivative:
            second = np.gradient(np.gradient(y_values, x_values), x_values)
            figure.add_trace(go.Scatter(x=x_values, y=second, mode="lines", name=f"Second derivative: {label}", line={"dash": "dash"}))
        if state.show_antiderivative:
            dx = float(x_values[1] - x_values[0]) if len(x_values) > 1 else 1.0
            anti = np.cumsum(y_values) * dx
            figure.add_trace(go.Scatter(x=x_values, y=anti, mode="lines", name=f"Integral: {label}", line={"dash": "longdash"}))

    figure.update_layout(
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        xaxis_title=state.x_axis_title,
        yaxis_title=state.y_axis_title,
    )
    print(f"[debug][formula-plot] build_formula_figure traces={len(figure.data)}", flush=True)
    return figure


def build_formula_figure_2d(state: FormulaPlotState) -> go.Figure:
    print(f"[debug][formula-plot] build_formula_figure_2d mode={state.display_mode_2d!r}", flush=True)
    x_values = np.linspace(state.x_min, state.x_max, int(state.x_points_2d))
    y_values = np.linspace(state.y_min, state.y_max, int(state.y_points_2d))
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_grid = evaluate_formula_2d(state.expression_2d, x_grid, y_grid, _parameter_values(state))
    if state.display_mode_2d == "heatmap":
        trace = go.Heatmap(x=x_values, y=y_values, z=z_grid, colorscale=state.surface_colorscale, name=state.label_2d)
    elif state.display_mode_2d == "contour":
        trace = go.Contour(x=x_values, y=y_values, z=z_grid, colorscale=state.surface_colorscale, name=state.label_2d)
    else:
        trace = go.Surface(x=x_grid, y=y_grid, z=z_grid, colorscale=state.surface_colorscale, name=state.label_2d)
    figure = go.Figure(data=[trace])
    figure.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        scene={
            "xaxis": {"title": state.x_axis_title},
            "yaxis": {"title": state.y_axis_title},
            "zaxis": {"title": state.z_axis_title},
        },
    )
    print(f"[debug][formula-plot] build_formula_figure_2d traces={len(figure.data)}", flush=True)
    return figure


class FormulaPlotTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][formula-plot-tab] init:start", flush=True)
        self.state = FormulaPlotState()
        self._slider_specs: dict[str, tuple[QSlider, QLabel]] = {}
        self.setObjectName("FormulaPlotTab")
        self.setStyleSheet(FORMULA_TAB_STYLE)

        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        left = QFormLayout()
        self.expression_edit = QLineEdit(self.state.formulas[0]["expression"], self)
        left.addRow("1D Formula", self.expression_edit)
        self.preset_2d_combo = QComboBox(self)
        for label, value in PRESET_2D_OPTIONS:
            self.preset_2d_combo.addItem(label, value)
        self.preset_2d_combo.currentIndexChanged.connect(self._on_2d_preset_changed)
        left.addRow("2D Preset", self.preset_2d_combo)
        self.display_mode_2d_combo = QComboBox(self)
        self.display_mode_2d_combo.addItem("Surface", "surface")
        self.display_mode_2d_combo.addItem("Heatmap", "heatmap")
        self.display_mode_2d_combo.addItem("Contour", "contour")
        self.display_mode_2d_combo.currentIndexChanged.connect(self._on_display_mode_2d_changed)
        left.addRow("2D Display", self.display_mode_2d_combo)
        self.expression_2d_edit = QLineEdit(self.state.expression_2d, self)
        left.addRow("2D Formula", self.expression_2d_edit)
        controls.addLayout(left, 2)

        self.slider_panel = QVBoxLayout()
        self.slider_panel.addStretch(1)
        self.refresh_param_controls("1d")
        controls.addLayout(self.slider_panel, 1)

        buttons = QVBoxLayout()
        self.plot_1d_btn = QPushButton("Plot 1D", self)
        self.plot_1d_btn.setStyleSheet(FORMULA_BUTTON_STYLE)
        self.plot_1d_btn.clicked.connect(self.refresh_1d)
        buttons.addWidget(self.plot_1d_btn)
        self.plot_2d_btn = QPushButton("Plot 2D", self)
        self.plot_2d_btn.setStyleSheet(FORMULA_BUTTON_STYLE)
        self.plot_2d_btn.clicked.connect(self.refresh_2d)
        buttons.addWidget(self.plot_2d_btn)
        controls.addLayout(buttons)
        root.addLayout(controls)

        self.status_label = QLabel("Ready", self)
        self.status_label.setStyleSheet(FORMULA_STATUS_READY_STYLE)
        root.addWidget(self.status_label)
        self.plot_view = PlotView(self)
        root.addWidget(self.plot_view, 1)
        self._sync_preset_combo()
        self._sync_display_mode_combo()
        self.refresh_1d()
        print("[debug][formula-plot-tab] init:done", flush=True)

    def _sync_preset_combo(self) -> None:
        print(f"[debug][formula-plot-tab] sync_preset_combo preset={self.state.preset_2d!r}", flush=True)
        index = self.preset_2d_combo.findData(self.state.preset_2d)
        if index >= 0 and index != self.preset_2d_combo.currentIndex():
            self.preset_2d_combo.blockSignals(True)
            self.preset_2d_combo.setCurrentIndex(index)
            self.preset_2d_combo.blockSignals(False)

    def _set_status(self, text: str, style: str) -> None:
        print(f"[debug][formula-plot-tab] set_status text={text!r} style={style!r}", flush=True)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)

    def _sync_display_mode_combo(self) -> None:
        print(f"[debug][formula-plot-tab] sync_display_mode_combo mode={self.state.display_mode_2d!r}", flush=True)
        index = self.display_mode_2d_combo.findData(self.state.display_mode_2d)
        if index >= 0 and index != self.display_mode_2d_combo.currentIndex():
            self.display_mode_2d_combo.blockSignals(True)
            self.display_mode_2d_combo.setCurrentIndex(index)
            self.display_mode_2d_combo.blockSignals(False)

    def _render_current_plot(self) -> None:
        print(f"[debug][formula-plot-tab] render_current_plot panel_type={self.state.panel_type!r}", flush=True)
        try:
            if self.state.panel_type == "2d":
                figure = build_formula_figure_2d(self.state)
                self.plot_view.set_figure(figure)
                self._set_status("Updated 2D plot", FORMULA_STATUS_UPDATED_STYLE)
            else:
                figure = build_formula_figure(self.state)
                self.plot_view.set_figure(figure)
                self._set_status("Updated 1D plot", FORMULA_STATUS_UPDATED_STYLE)
        except Exception as exc:
            print(f"[debug][formula-plot-tab] render_current_plot error={exc!r}", flush=True)
            prefix = "2D" if self.state.panel_type == "2d" else "1D"
            self._set_status(f"{prefix} plot error: {exc}", FORMULA_STATUS_ERROR_STYLE)

    def _on_2d_preset_changed(self, index: int) -> None:
        preset_name = self.preset_2d_combo.itemData(index)
        print(f"[debug][formula-plot-tab] on_2d_preset_changed index={index} preset={preset_name!r}", flush=True)
        if not preset_name:
            return
        apply_2d_preset(self.state, str(preset_name))
        self.expression_2d_edit.setText(self.state.expression_2d)
        self._sync_display_mode_combo()
        self.refresh_param_controls("2d")
        self.refresh_2d()

    def _on_display_mode_2d_changed(self, index: int) -> None:
        mode = self.display_mode_2d_combo.itemData(index)
        print(f"[debug][formula-plot-tab] on_display_mode_2d_changed index={index} mode={mode!r}", flush=True)
        if not mode:
            return
        self.state.display_mode_2d = str(mode)
        self.state.panel_type = "2d"
        self._render_current_plot()

    def _update_param(self, name: str, slider_value: int, label: QLabel) -> None:
        spec = self.state.params.get(name)
        if spec is None:
            spec = parameter_preset_for_name(name)
            self.state.params[name] = spec
        step = float(spec.get("step", 0.1))
        value = round(slider_value * step, 6)
        print(f"[debug][formula-plot-tab] update_param name={name!r} slider_value={slider_value} value={value}", flush=True)
        spec["value"] = value
        label.setText(f"{value:.2f}".rstrip("0").rstrip("."))

    def _on_param_slider(self, name: str, slider_value: int, label: QLabel) -> None:
        self._update_param(name, slider_value, label)
        self._render_current_plot()

    def refresh_param_controls(self, mode: str) -> None:
        print(f"[debug][formula-plot-tab] refresh_param_controls mode={mode!r}", flush=True)
        while self.slider_panel.count():
            item = self.slider_panel.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._slider_specs.clear()

        expression = self.expression_edit.text().strip() if mode == "1d" else self.expression_2d_edit.text().strip()
        names = extract_formula_variables(expression, coordinate_names=("x",) if mode == "1d" else ("x", "y"))
        if not names:
            names = ["a"]
        for param_name in names:
            spec = self.state.params.get(param_name)
            if spec is None:
                spec = parameter_preset_for_name(param_name)
                self.state.params[param_name] = dict(spec)
            else:
                spec = {**parameter_preset_for_name(param_name), **spec}
                self.state.params[param_name] = spec
            row_widget = QWidget(self)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(param_name, row_widget))
            slider = QSlider(Qt.Orientation.Horizontal, row_widget)
            slider.setStyleSheet(FORMULA_SLIDER_STYLE)
            step = float(spec.get("step", 0.1))
            slider.setMinimum(int(round(float(spec.get("min", -10.0)) / step)))
            slider.setMaximum(int(round(float(spec.get("max", 10.0)) / step)))
            slider.setValue(int(round(float(spec.get("value", 1.0)) / step)))
            value_label = QLabel(f"{float(spec.get('value', 1.0)):.2f}".rstrip("0").rstrip("."), row_widget)
            slider.valueChanged.connect(lambda value, name=param_name, label=value_label: self._on_param_slider(name, value, label))
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            self.slider_panel.addWidget(row_widget)
            self._slider_specs[param_name] = (slider, value_label)
        self.slider_panel.addStretch(1)

    def refresh_1d(self) -> None:
        expression = self.expression_edit.text().strip() or "sin(x)"
        print(f"[debug][formula-plot-tab] refresh_1d expression={expression!r}", flush=True)
        self.state.panel_type = "1d"
        self.state.formulas[0]["expression"] = expression
        self.state.formulas[0]["label"] = expression
        self.state.x_axis_title = "x"
        self.state.y_axis_title = "f(x)"
        self.refresh_param_controls("1d")
        self._render_current_plot()

    def refresh_2d(self) -> None:
        expression = self.expression_2d_edit.text().strip() or "sin(x) * cos(y)"
        print(f"[debug][formula-plot-tab] refresh_2d expression={expression!r}", flush=True)
        self.state.panel_type = "2d"
        self.state.expression_2d = expression
        self.state.display_mode_2d = str(self.display_mode_2d_combo.currentData() or self.state.display_mode_2d)
        self.state.preset_2d = str(self.preset_2d_combo.currentData() or self.state.preset_2d)
        preset = PRESET_2D_DEFAULTS.get(self.state.preset_2d, {})
        preset_expression = str(preset.get("expression_2d", ""))
        preset_label = str(preset.get("label_2d", expression))
        self.state.label_2d = preset_label if expression == preset_expression else expression
        print(
            f"[debug][formula-plot-tab] refresh_2d label preset={self.state.preset_2d!r} "
            f"label={self.state.label_2d!r}",
            flush=True,
        )
        self._sync_preset_combo()
        self._sync_display_mode_combo()
        self.refresh_param_controls("2d")
        self._render_current_plot()
