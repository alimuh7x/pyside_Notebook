from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyside_app.formula_plot_tab import (
    FORMULA_BUTTON_STYLE,
    FORMULA_STATUS_READY_STYLE,
    FORMULA_STATUS_UPDATED_STYLE,
    FORMULA_TAB_STYLE,
    FORMULA_STATUS_ERROR_STYLE,
    FormulaPlotState,
    FormulaPlotTab,
    apply_2d_preset,
    build_formula_figure,
    build_formula_figure_2d,
    parameter_preset_for_name,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_build_formula_figure_uses_existing_formula_parser():
    state = FormulaPlotState()
    state.formulas[0]["expression"] = "a * sin(x)"
    state.params["a"]["value"] = 2.0

    figure = build_formula_figure(state)

    assert len(figure.data) == 1
    assert figure.data[0].name == "a * sin(x)"
    assert max(figure.data[0].y) == pytest.approx(2.0, rel=1e-2)


def test_build_formula_figure_supports_derivative_overlay():
    state = FormulaPlotState()
    state.formulas[0]["expression"] = "x**2"
    state.show_derivative = True

    figure = build_formula_figure(state)

    assert len(figure.data) == 2
    assert any("Derivative" in (trace.name or "") for trace in figure.data)


def test_build_formula_figure_2d_supports_surface_mode():
    state = FormulaPlotState(panel_type="2d")
    state.expression_2d = "sin(x) * cos(y)"

    figure = build_formula_figure_2d(state)

    assert len(figure.data) == 1
    assert figure.data[0].type in {"surface", "heatmap", "contour"}


def test_parameter_preset_for_name_returns_named_defaults():
    sigma_spec = parameter_preset_for_name("sigma")
    x0_spec = parameter_preset_for_name("x0")
    fallback_spec = parameter_preset_for_name("custom_gain")

    assert sigma_spec == {"value": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}
    assert x0_spec == {"value": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}
    assert fallback_spec == {"value": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}


def test_apply_2d_preset_populates_expression_axes_and_params():
    state = FormulaPlotState(panel_type="2d")

    result = apply_2d_preset(state, "gaussian_hill")

    assert result is None
    assert state.preset_2d == "gaussian_hill"
    assert state.expression_2d == "a*exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))"
    assert state.label_2d == "Gaussian hill"
    assert (state.x_min, state.x_max, state.y_min, state.y_max) == (-6.0, 6.0, -6.0, 6.0)
    assert state.surface_colorscale == "Plasma"
    assert state.params["sigma"] == {"value": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}


def test_build_formula_figure_2d_uses_preset_axis_titles():
    state = FormulaPlotState(panel_type="2d")

    apply_2d_preset(state, "paraboloid")
    figure = build_formula_figure_2d(state)

    assert figure.layout.scene.xaxis.title.text == "x"
    assert figure.layout.scene.yaxis.title.text == "y"
    assert figure.layout.scene.zaxis.title.text == "z"


def test_formula_plot_tab_exposes_2d_presets_and_formula_styles():
    _app()
    tab = FormulaPlotTab()

    preset_names = [tab.preset_2d_combo.itemData(index) for index in range(tab.preset_2d_combo.count())]

    assert preset_names[:5] == [
        "gaussian_hill",
        "saddle",
        "paraboloid",
        "radial_decay",
        "periodic_surface",
    ]
    assert tab.styleSheet() == FORMULA_TAB_STYLE
    assert tab.plot_1d_btn.styleSheet() == FORMULA_BUTTON_STYLE
    assert tab.plot_2d_btn.styleSheet() == FORMULA_BUTTON_STYLE
    assert tab.display_mode_2d_combo.currentData() == "surface"
    assert tab.status_label.styleSheet() in {
        FORMULA_STATUS_READY_STYLE,
        FORMULA_STATUS_UPDATED_STYLE,
        FORMULA_STATUS_ERROR_STYLE,
    }


def test_slider_drag_updates_plot_without_rebuilding_slider_controls():
    _app()
    tab = FormulaPlotTab()
    slider_before, label_before = tab._slider_specs["a"]

    slider_before.setValue(slider_before.value() + 1)
    slider_after, label_after = tab._slider_specs["a"]

    assert slider_after is slider_before
    assert label_after is label_before
    assert tab.state.params["a"]["value"] == pytest.approx(1.1)


def test_display_mode_combo_switches_2d_trace_type():
    _app()
    tab = FormulaPlotTab()
    tab.preset_2d_combo.setCurrentIndex(tab.preset_2d_combo.findData("periodic_surface"))
    tab.display_mode_2d_combo.setCurrentIndex(tab.display_mode_2d_combo.findData("contour"))

    assert tab.state.display_mode_2d == "contour"
    figure = build_formula_figure_2d(tab.state)
    assert figure.data[0].type == "contour"
