from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication, QScrollArea

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
    build_formula_2d_slice_figures,
    build_formula_figure_2d,
    formula_samples_dataframe,
    formula_surface_dataframe,
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


def test_formula_plot_tab_styles_combo_popups_with_readable_text():
    _app()
    tab = FormulaPlotTab()

    stylesheet = tab.styleSheet()

    assert "QComboBox QAbstractItemView" in stylesheet
    assert "color: #f8fafc" in stylesheet
    assert "selection-background-color: #61afef" in stylesheet
    assert "QCheckBox::indicator:checked" in stylesheet


def test_formula_plot_tab_uses_dash_structure_with_dark_theme():
    _app()
    tab = FormulaPlotTab()

    assert tab.page_scroll.objectName() == "FormulaPageScroll"
    assert tab.page_scroll.widgetResizable()
    assert tab.page_widget.objectName() == "FormulaPage"
    assert tab.top_controls_widget.objectName() == "FormulaTopControls"
    assert tab.main_content_widget.objectName() == "FormulaMainContent"
    assert tab.graph_column_widget.objectName() == "FormulaGraphColumn"
    assert tab.graph_options_widget.objectName() == "FormulaGraphOptions"
    assert tab.settings_sidebar.objectName() == "FormulaSettingsSidebar"
    assert not isinstance(tab.top_controls_widget, QScrollArea)
    assert not isinstance(tab.settings_sidebar, QScrollArea)
    assert tab.top_controls_widget.minimumWidth() >= 1000
    assert tab.settings_sidebar.minimumWidth() >= 340
    assert tab.one_d_panel.parent() is tab.top_controls_widget
    assert tab.range_panel.parent() is tab.top_controls_widget
    assert tab.display_panel.parent() is tab.graph_options_widget
    assert tab.param_panel.parent() is tab.settings_sidebar_contents
    assert tab.one_d_row_layout.count() == 4
    assert tab.formula_rows_layout.count() == 1
    assert tab.range_row_layout.count() == 8
    assert "#FormulaPlotTab" in tab.styleSheet()
    assert "background: #21252b" in tab.styleSheet()
    assert "font-size: 12px" in tab.styleSheet()


def test_build_formula_figure_matches_dash_graph_size_and_ticks():
    state = FormulaPlotState()

    figure = build_formula_figure(state)

    assert figure.layout.width == 1000
    assert figure.layout.height == 700
    assert figure.layout.font.size == 18
    assert figure.layout.font.family == "Arial"
    assert figure.layout.xaxis.mirror == "allticks"
    assert figure.layout.xaxis.ticks == "inside"
    assert figure.layout.xaxis.ticklen == 8
    assert figure.layout.xaxis.tickwidth == 2
    assert figure.layout.xaxis.tickcolor == "black"
    assert figure.layout.yaxis.mirror == "allticks"
    assert figure.layout.yaxis.ticks == "inside"
    assert figure.layout.yaxis.ticklen == 8
    assert figure.layout.yaxis.tickwidth == 2
    assert figure.layout.yaxis.tickcolor == "black"


def test_build_formula_2d_figures_match_dash_graph_sizes_and_fonts():
    state = FormulaPlotState(panel_type="2d")
    state.expression_2d = "x + y"

    surface = build_formula_figure_2d(state)
    x_slice, y_slice = build_formula_2d_slice_figures(state)

    assert surface.layout.width == 1000
    assert surface.layout.height == 760
    assert surface.layout.font.size == 16
    assert surface.layout.font.family == "Arial"
    assert x_slice.layout.height == 280
    assert y_slice.layout.height == 280
    assert x_slice.layout.font.size == 13
    assert y_slice.layout.font.family == "Arial"


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


def test_build_formula_figure_supports_multiple_formulas_and_analysis_markers():
    state = FormulaPlotState()
    state.x_min = -2.0
    state.x_max = 2.0
    state.points = 401
    state.formulas = [
        {"id": "f1", "expression": "x**2 - 1", "label": "roots", "visible": True, "color": "black", "dash": "solid", "width": 3.0},
        {"id": "f2", "expression": "-x", "label": "line", "visible": True, "color": "#d62728", "dash": "dash", "width": 2.0},
    ]
    state.show_derivative = True
    state.show_second_derivative = True
    state.show_antiderivative = True
    state.show_root_markers = True
    state.show_extrema_markers = True
    state.show_intersection_markers = True
    state.analysis_formula = "f1"
    state.analysis_x0 = 0.0
    state.threshold_value = 0.0

    figure = build_formula_figure(state)
    names = [trace.name for trace in figure.data]

    assert "roots" in names
    assert "line" in names
    assert any("Derivative" in (name or "") or "d/dx" in (name or "") for name in names)
    assert any("Second" in (name or "") or "d2/dx2" in (name or "") for name in names)
    assert any("Integral" in (name or "") or "integral" in (name or "") for name in names)
    assert any("root" in (name or "").lower() for name in names)
    assert any("extrema" in (name or "").lower() for name in names)
    assert any("intersection" in (name or "").lower() for name in names)
    assert "2 root" in state.last_summary_text
    assert "turning" in state.last_summary_text


def test_formula_samples_dataframe_uses_notebook_binding_and_all_visible_formulas():
    state = FormulaPlotState()
    state.x_min = 0.0
    state.x_max = 1.0
    state.points = 3
    state.params = {"a": {"value": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "use_notebook": True}}
    state.formulas = [
        {"id": "f1", "expression": "a*x", "label": "scaled", "visible": True, "color": "black", "dash": "solid", "width": 3.0},
        {"id": "f2", "expression": "x + 1", "label": "offset", "visible": True, "color": "black", "dash": "solid", "width": 3.0},
    ]

    df = formula_samples_dataframe(state, {"a": 4.0})

    assert list(df.columns) == ["x", "scaled", "offset"]
    assert df["scaled"].tolist() == pytest.approx([0.0, 2.0, 4.0])
    assert df["offset"].tolist() == pytest.approx([1.0, 1.5, 2.0])


def test_2d_probe_slices_and_surface_export_dataframe():
    state = FormulaPlotState(panel_type="2d")
    state.expression_2d = "x + y"
    state.label_2d = "sum"
    state.x_min = 0.0
    state.x_max = 2.0
    state.y_min = 0.0
    state.y_max = 2.0
    state.x_points_2d = 3
    state.y_points_2d = 3
    state.probe_x_2d = 1.0
    state.probe_y_2d = 2.0

    figure = build_formula_figure_2d(state)
    x_slice, y_slice = build_formula_2d_slice_figures(state)
    df = formula_surface_dataframe(state)

    assert any((trace.name or "") == "Probe" for trace in figure.data)
    assert x_slice.data[0].y.tolist() == pytest.approx([2.0, 3.0, 4.0])
    assert y_slice.data[0].y.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert list(df.columns) == ["x", "y", "z"]
    assert len(df) == 9
    assert state.last_summary_text


def test_formula_plot_tab_accepts_notebook_namespace_provider_for_nb_binding():
    _app()
    tab = FormulaPlotTab(notebook_namespace_provider=lambda: {"a": 3.0})
    tab.expression_edit.setText("a*x")
    tab.refresh_1d()
    tab.state.params["a"]["use_notebook"] = True

    figure = build_formula_figure(tab.state, tab._notebook_vars())

    assert max(figure.data[0].y) == pytest.approx(30.0)


def test_formula_plot_tab_renders_analysis_as_tables():
    _app()
    tab = FormulaPlotTab()
    tab.state.formulas[0]["expression"] = "sin(x)"
    tab.state.formulas[0]["label"] = "sin(x)"
    tab.state.show_root_markers = True
    tab.state.show_extrema_markers = True

    tab.refresh_1d()

    summary_html = tab.summary_browser.toHtml()
    details_html = tab.details_browser.toHtml()
    assert "<table" in summary_html
    assert "<table" in details_html
    assert "Formula" in summary_html
    assert "Roots" in details_html
    assert "Extrema" in details_html
