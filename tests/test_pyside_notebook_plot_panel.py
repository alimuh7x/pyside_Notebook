from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QLabel, QDoubleSpinBox, QRadioButton, QWidget

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyside_app.execution_engine import ExecutionOutput, ExecutionResult
from pyside_app.notebook_plot_panel import (
    NotebookPlotPanel,
    QuickGraphPreviewPanel,
    build_notebook_heatmap_figure,
    build_notebook_contour_figure,
    build_notebook_evolution_animation_figure,
    build_notebook_evolution_figure,
    build_notebook_plot_figure,
    extract_notebook_array_variables,
    extract_notebook_array_variables_with_3d,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeCell:
    def __init__(self, cell_id: str, source: str, result: ExecutionResult | None) -> None:
        self.cell_id = cell_id
        self._source = source
        self.last_result = result

    def source(self) -> str:
        return self._source


def test_extract_notebook_array_variables_filters_to_numeric_series():
    namespace = {
        "x": np.linspace(0.0, 1.0, 5),
        "y": [1.0, 2.0, 3.0],
        "history": np.arange(12.0).reshape(3, 4),
        "cube": np.arange(24.0).reshape(2, 3, 4),
        "labels": ["a", "b", "c"],
        "scalar": 42,
    }

    arrays_1d, arrays_2d = extract_notebook_array_variables(namespace)

    assert sorted(arrays_1d.keys()) == ["x", "y"]
    assert sorted(arrays_2d.keys()) == ["history"]
    assert arrays_1d["x"].shape == (5,)
    assert arrays_1d["y"].shape == (3,)
    assert arrays_2d["history"].shape == (3, 4)
    assert "cube" not in arrays_2d


def test_extract_notebook_array_variables_with_3d_includes_numeric_cubes():
    namespace = {
        "x": np.linspace(0.0, 1.0, 4),
        "field": np.arange(12.0).reshape(3, 4),
        "u_history": np.arange(24.0).reshape(2, 3, 4),
        "bad_cube": np.array([[["a"]]]),
    }

    arrays_1d, arrays_2d, arrays_3d = extract_notebook_array_variables_with_3d(namespace)

    assert sorted(arrays_1d) == ["x"]
    assert sorted(arrays_2d) == ["field"]
    assert sorted(arrays_3d) == ["u_history"]
    assert arrays_3d["u_history"].shape == (2, 3, 4)


def test_notebook_plot_panel_prefers_current_u_over_history_axes_for_series_y():
    _app()
    panel = NotebookPlotPanel()

    panel.set_namespace(
        {
            "n_history": np.array([0.0, 1.0, 2.0]),
            "time_history": np.array([0.0, 0.1, 0.2]),
            "u": np.array([1.0, 2.0, 3.0]),
            "u_history": np.arange(9.0).reshape(3, 3),
        }
    )

    assert "u" in panel.y_combo.checked_values()


def test_build_notebook_heatmap_figure_creates_static_xy_heatmap():
    arrays_1d = {
        "x": np.array([0.0, 0.5, 1.0]),
        "y": np.array([0.0, 1.0]),
    }
    arrays_2d = {"field": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}

    figure = build_notebook_heatmap_figure(
        arrays_1d,
        arrays_2d,
        {},
        "field",
        None,
        "x",
        "y",
        False,
        "Static field",
        "x",
        "y",
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"
    assert list(figure.data[0].x) == [0.0, 0.5, 1.0]
    assert list(figure.data[0].y) == [0.0, 1.0]
    assert np.asarray(figure.data[0].z).shape == (2, 3)
    assert figure.data[0].zsmooth == "best"
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.scaleratio == 1
    assert not figure.frames


def test_build_notebook_heatmap_figure_animates_t_y_x_history():
    arrays_1d = {
        "time_history": np.array([0.0, 0.1, 0.2]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0, 2.0]),
    }
    arrays_3d = {
        "u_history": np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]
        )
    }

    figure = build_notebook_heatmap_figure(
        arrays_1d,
        {},
        arrays_3d,
        "u_history",
        "time_history",
        "x",
        "y",
        True,
        "Field evolution",
        "x",
        "y",
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"
    assert figure.data[0].zauto is True
    assert figure.data[0].colorscale[0] == (0.0, "#00328f")
    assert figure.data[0].colorscale[2] == (0.5, "#fffbdf")
    assert figure.data[0].colorscale[-1] == (1.0, "#a51717")
    assert figure.data[0].zsmooth == "best"
    assert len(figure.frames) == 3
    assert len(figure.layout.sliders) == 1
    assert len(figure.layout.updatemenus) == 1
    assert figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is True
    assert figure.layout.sliders[0].steps[1].args[1]["frame"]["redraw"] is True
    assert figure.frames[1].data[0].zauto is True
    assert figure.frames[1].data[0].colorscale[1] == (0.25, "#00afb8")
    assert figure.frames[1].data[0].colorscale[3] == (0.75, "#ffbc3c")
    assert figure.frames[1].data[0].zsmooth == "best"
    assert figure.layout.yaxis.scaleanchor == "x"
    assert list(figure.frames[-1].data[0].z[0]) == [9.0, 10.0]


def test_build_notebook_heatmap_figure_can_use_fixed_full_history_scale():
    arrays_1d = {
        "time_history": np.array([0.0, 0.1]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0, 1.0]),
    }
    arrays_3d = {
        "u_history": np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[-5.0, 6.0], [7.0, 8.0]],
            ]
        )
    }

    figure = build_notebook_heatmap_figure(
        arrays_1d,
        {},
        arrays_3d,
        "u_history",
        "time_history",
        "x",
        "y",
        True,
        "",
        "",
        "",
        None,
        "fixed",
    )

    assert figure.data[0].zauto is False
    assert figure.data[0].zmin == -5.0
    assert figure.data[0].zmax == 8.0
    assert figure.frames[1].data[0].zauto is False
    assert figure.frames[1].data[0].zmin == -5.0
    assert figure.frames[1].data[0].zmax == 8.0


def test_build_notebook_heatmap_figure_can_use_manual_color_range():
    arrays_1d = {
        "time_history": np.array([0.0, 0.1]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0, 1.0]),
    }
    arrays_3d = {"u_history": np.arange(8.0).reshape(2, 2, 2)}

    figure = build_notebook_heatmap_figure(
        arrays_1d,
        {},
        arrays_3d,
        "u_history",
        "time_history",
        "x",
        "y",
        True,
        "",
        "",
        "",
        None,
        "fixed",
        -2.5,
        4.5,
    )

    assert figure.data[0].zauto is False
    assert figure.data[0].zmin == -2.5
    assert figure.data[0].zmax == 4.5
    assert figure.frames[1].data[0].zmin == -2.5
    assert figure.frames[1].data[0].zmax == 4.5


def test_build_notebook_contour_figure_creates_static_line_contours():
    arrays_1d = {
        "x": np.array([0.0, 0.5, 1.0]),
        "y": np.array([0.0, 1.0]),
    }
    arrays_2d = {"field": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}

    figure = build_notebook_contour_figure(
        arrays_1d,
        arrays_2d,
        {},
        "field",
        None,
        "x",
        "y",
        False,
        "Contours",
        "x",
        "y",
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "contour"
    assert figure.data[0].contours.coloring == "lines"
    assert figure.data[0].contours.showlabels is True
    assert figure.layout.yaxis.scaleanchor == "x"
    assert not figure.frames


def test_build_notebook_contour_figure_animates_filled_contours_with_fixed_scale():
    arrays_1d = {
        "time_history": np.array([0.0, 0.1]),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0, 1.0]),
    }
    arrays_3d = {
        "u_history": np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[-5.0, 6.0], [7.0, 8.0]],
            ]
        )
    }

    figure = build_notebook_contour_figure(
        arrays_1d,
        {},
        arrays_3d,
        "u_history",
        "time_history",
        "x",
        "y",
        True,
        "",
        "",
        "",
        None,
        "fixed",
        filled=True,
    )

    assert figure.data[0].type == "contour"
    assert figure.data[0].contours.coloring == "heatmap"
    assert figure.data[0].zauto is False
    assert figure.data[0].zmin == -5.0
    assert figure.data[0].zmax == 8.0
    assert len(figure.frames) == 2
    assert figure.frames[1].data[0].contours.coloring == "heatmap"
    assert figure.frames[1].data[0].zmin == -5.0


def test_build_notebook_contour_figure_creates_banded_filled_contours():
    arrays_1d = {
        "x": np.array([0.0, 0.5, 1.0]),
        "y": np.array([0.0, 1.0]),
    }
    arrays_2d = {"field": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}

    figure = build_notebook_contour_figure(
        arrays_1d,
        arrays_2d,
        {},
        "field",
        None,
        "x",
        "y",
        False,
        "",
        "",
        "",
        None,
        "fixed",
        filled=True,
        banded=True,
    )

    assert figure.data[0].type == "contour"
    assert figure.data[0].contours.coloring == "fill"
    assert figure.data[0].contours.showlines is True
    assert figure.data[0].line.color == "#1f2937"
    assert figure.data[0].zauto is False


def test_build_notebook_plot_figure_creates_scatter_trace_for_selected_arrays():
    arrays = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([1.0, 3.0, 5.0]),
    }

    figure = build_notebook_plot_figure(
        arrays,
        x_var="x",
        y_vars=["y"],
        plot_type="lines",
        title="Line Plot",
        x_title="time",
        y_title="value",
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "scatter"
    assert figure.data[0].name == "y"
    assert figure.layout.title.text == "Line Plot"
    assert figure.layout.xaxis.title.text == "time"
    assert figure.layout.yaxis.title.text == "value"


def test_build_notebook_plot_figure_applies_per_series_styles():
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
            "phi": {"plot_type": "lines", "line_style": "dash", "marker_style": "square", "line_color": "#000000"},
            "phi_exact": {"plot_type": "markers", "line_style": "solid", "marker_style": "diamond", "line_color": "#d62728"},
        },
    )

    assert len(figure.data) == 2
    assert figure.data[0].mode == "lines"
    assert figure.data[0].line.dash == "dash"
    assert figure.data[0].line.color == "#000000"
    assert figure.data[0].marker.symbol == "square"
    assert figure.data[1].mode == "markers"
    assert figure.data[1].marker.symbol == "diamond"
    assert figure.data[1].marker.color == "#d62728"


def test_build_notebook_plot_figure_applies_scientific_style_options():
    arrays = {
        "x": np.array([0.0, 1.0, 2.0]),
        "phi": np.array([1.0, 3.0, 5.0]),
    }

    figure = build_notebook_plot_figure(
        arrays,
        x_var="x",
        y_vars=["phi"],
        plot_type="lines+markers",
        title="Styled",
        x_title="distance",
        y_title="value",
        style_options={
            "font_size": 18,
            "line_width": 5,
            "marker_size": 12,
            "show_grid": False,
            "show_box": True,
            "ticks_inside": True,
            "show_minor_ticks": True,
            "graph_aspect": "square",
            "graph_width": None,
            "graph_height": 700,
        },
    )

    assert figure.data[0].line.width == 5
    assert figure.data[0].marker.size == 12
    assert figure.layout.font.size == 18
    assert figure.layout.xaxis.showgrid is False
    assert figure.layout.xaxis.showline is True
    assert figure.layout.xaxis.mirror == "allticks"
    assert figure.layout.xaxis.ticks == "inside"
    assert figure.layout.xaxis.ticklen == 10
    assert figure.layout.xaxis.minor.ticks == "inside"
    assert figure.layout.xaxis.minor.ticklen == 5
    assert figure.layout.xaxis.automargin is True
    assert figure.layout.margin.b >= 80
    assert figure.layout.margin.r == 80
    assert figure.layout.width is None
    assert figure.layout.height == 700


def test_notebook_plot_panel_applies_log_axis_controls_in_graphs_tab():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "x": np.array([1.0, 10.0, 100.0]),
            "phi": np.array([0.1, 1.0, 10.0]),
        }
    )

    panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
    panel.y_combo.set_checked_values(["phi"])
    panel.x_axis_scale_combo.setCurrentIndex(panel.x_axis_scale_combo.findData("log"))
    panel.y_axis_scale_combo.setCurrentIndex(panel.y_axis_scale_combo.findData("log"))
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    assert figure.layout.xaxis.type == "log"
    assert figure.layout.yaxis.type == "log"


def test_notebook_plot_panel_updates_dropdowns_and_renders_output_graphs():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "x": np.array([0.0, 1.0, 2.0]),
            "y": np.array([1.0, 4.0, 9.0]),
            "z": np.array([2.0, 3.0, 4.0]),
        }
    )

    assert panel.x_combo.count() >= 4
    assert panel.x_combo.findData("x") >= 0
    assert panel.y_combo.count() == 3
    assert not panel.series_controls.isHidden()
    assert panel.evolution_controls.isHidden()

    panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
    panel.y_combo.set_checked_values(["y"])
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    assert len(figure.data) == 1
    assert figure.data[0].name == "y"

    panel.y_combo.set_checked_values(["x", "z"])
    panel.refresh_controller_plot()

    multi_figure = panel.current_controller_figure()
    assert len(multi_figure.data) == 2
    assert {trace.name for trace in multi_figure.data} == {"x", "z"}

    panel.sync_cell_outputs(
        [
            _FakeCell(
                "cell-1",
                "go.Figure(...)",
                ExecutionResult(
                    outputs=[
                        ExecutionOutput(
                            kind="plotly",
                            data={"html": "<div>plot-a</div>", "text": "Plotly Figure"},
                        )
                    ]
                ),
            )
        ]
    )

    assert "go.Figure" in panel.output_titles()[0]
    assert panel.output_count() == 1
    assert panel.outputs_scroll.isHidden()


def test_notebook_plot_panel_uses_sectioned_visual_layout_for_builder_controls():
    _app()
    panel = NotebookPlotPanel()

    section_boxes = panel.main_card.findChildren(QWidget, "graphSection")
    preview_boxes = panel.main_card.findChildren(QWidget, "graphPreviewCard")
    section_titles = {
        label.text()
        for label in panel.main_card.findChildren(QLabel)
        if label.text() in {"Data", "Plot Setup", "Appearance", "Labels", "Analysis", "Live Preview"}
    }

    assert len(section_boxes) >= 4
    assert len(preview_boxes) == 1
    assert {"Plot Setup", "Appearance", "Labels", "Live Preview"} <= section_titles


def test_notebook_plot_panel_uses_shared_small_heading_font_across_controls():
    _app()
    panel = NotebookPlotPanel()

    data_source_ss = panel.main_card._data_source.styleSheet()
    card_ss = panel.main_card.styleSheet()
    axis_selector_ss = panel.main_card._axis_selector.styleSheet()
    plot_style_ss = panel.main_card._plot_style.styleSheet()
    axis_labels_ss = panel.main_card._axis_labels.styleSheet()
    analysis_ss = panel.main_card._analysis.styleSheet()
    smooth_check_ss = panel.main_card._analysis.smooth_check.styleSheet()

    assert "QRadioButton" in data_source_ss
    assert "font-size:12px" in data_source_ss
    assert "font-weight:400" in data_source_ss
    assert "QComboBox" in card_ss
    assert "font-size:12px" in card_ss
    assert "font-size:12px" in axis_selector_ss
    assert "font-weight:700" in axis_selector_ss
    assert "font-size:12px" in plot_style_ss
    assert "font-weight:700" in plot_style_ss
    assert "font-size:12px" in axis_labels_ss
    assert "font-weight:700" in axis_labels_ss
    assert "font-size:12px" in analysis_ss
    assert "font-weight:700" in analysis_ss
    assert "font-size:12px" in smooth_check_ss
    assert "font-weight:400" in smooth_check_ss


def test_notebook_plot_panel_data_scale_controls_are_readable_and_explicit():
    _app()
    panel = NotebookPlotPanel()

    data_source = panel.main_card._data_source
    scale_labels = {
        label.text(): label.styleSheet()
        for label in data_source.findChildren(QLabel)
        if label.text() in {"X scale:", "Y scale:"}
    }
    scale_radios = {
        radio.text(): radio.styleSheet()
        for radio in data_source.findChildren(QRadioButton)
        if radio.text() in {"Multiply", "Divide"}
    }
    scale_row = data_source.findChild(QWidget, "dataScaleRow")
    scale_spins = data_source.findChildren(QDoubleSpinBox)

    assert scale_row is not None
    assert set(scale_labels) == {"X scale:", "Y scale:"}
    assert all("color:#d7dae0" in style for style in scale_labels.values())
    assert set(scale_radios) == {"Multiply", "Divide"}
    assert all("QRadioButton::indicator:checked" in style for style in scale_radios.values())
    assert len(scale_spins) == 2
    assert all("up-arrow" in spin.styleSheet() and "down-arrow" in spin.styleSheet() for spin in scale_spins)
    assert scale_row.layout() is not None
    assert scale_row.layout().count() >= 2


def test_notebook_plot_panel_appearance_controls_share_one_row():
    _app()
    panel = NotebookPlotPanel()

    row = panel.main_card._plot_style.findChild(QWidget, "appearanceControlRow")
    assert row is not None
    assert panel.graph_size_combo.parentWidget().parentWidget() is row
    assert panel.font_size_combo.parentWidget().parentWidget() is row
    assert panel.line_width_combo.parentWidget().parentWidget() is row
    assert panel.marker_size_combo.parentWidget().parentWidget() is row
    assert panel.x_axis_scale_combo.parentWidget().parentWidget() is row
    assert panel.y_axis_scale_combo.parentWidget().parentWidget() is row


def test_notebook_plot_panel_adds_reference_lines_from_analysis_controls():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "x": np.array([0.0, 1.0, 2.0]),
            "phi": np.array([1.0, 2.0, 4.0]),
        }
    )

    panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
    panel.y_combo.set_checked_values(["phi"])
    analysis = panel.main_card._analysis
    analysis.vertical_line_check.setChecked(True)
    analysis.vertical_line_value.setValue(1.5)
    analysis.horizontal_line_check.setChecked(True)
    analysis.horizontal_line_value.setValue(3.0)
    panel.refresh_controller_plot()

    shapes = list(panel.current_controller_figure().layout.shapes or [])
    assert any(shape.x0 == 1.5 and shape.x1 == 1.5 and shape.yref == "paper" for shape in shapes)
    assert any(shape.y0 == 3.0 and shape.y1 == 3.0 and shape.xref == "paper" for shape in shapes)
    trace_map = {trace.name: trace for trace in panel.current_controller_figure().data}
    crossing = trace_map["phi reference crossing"]
    points = {(round(float(x), 6), round(float(y), 6)) for x, y in zip(crossing.x, crossing.y)}
    assert (1.5, 3.0) in points
    assert crossing.marker.size == 15
    assert crossing.mode == "markers"

    analysis.marker_label_check.setChecked(True)
    panel.refresh_controller_plot()
    trace_map = {trace.name: trace for trace in panel.current_controller_figure().data}
    labeled_crossing = trace_map["phi reference crossing"]
    assert labeled_crossing.mode == "markers+text"
    assert "(1.5, 3)" in labeled_crossing.text
    assert all("<br>" not in text and "x=" not in text and "y=" not in text for text in labeled_crossing.text)
    assert labeled_crossing.textposition


def test_notebook_plot_panel_derivatives_use_right_axes_and_zero_markers():
    _app()
    panel = NotebookPlotPanel()
    x = np.linspace(-2.0, 2.0, 81)
    panel.set_namespace(
        {
            "x": x,
            "phi": x**3 - x,
        }
    )

    panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
    panel.y_combo.set_checked_values(["phi"])
    analysis = panel.main_card._analysis
    analysis.first_deriv_check.setChecked(True)
    analysis.second_deriv_check.setChecked(True)
    analysis.mark_first_zero_check.setChecked(True)
    analysis.mark_second_zero_check.setChecked(True)
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    trace_map = {trace.name: trace for trace in figure.data}
    assert trace_map["d(phi)/dx"].yaxis == "y2"
    assert trace_map["d2(phi)/dx2"].yaxis == "y3"
    assert trace_map["d(phi)/dx"].line.width == 3.2
    assert trace_map["d2(phi)/dx2"].line.width == 3.2
    assert figure.layout.yaxis2.overlaying == "y"
    assert figure.layout.yaxis3.overlaying == "y"
    assert figure.layout.yaxis2.position == 1.0
    assert figure.layout.yaxis3.position == 1.0
    assert figure.layout.yaxis2.shift == 35
    assert figure.layout.yaxis3.shift == 155
    assert figure.layout.margin.r == 320
    assert any("dy/dx = 0" in (trace.name or "") for trace in figure.data)
    assert any("d2y/dx2 = 0" in (trace.name or "") for trace in figure.data)
    zero_markers = [trace for trace in figure.data if "= 0" in (trace.name or "")]
    assert all(trace.marker.size == 16 for trace in zero_markers)
    assert all(trace.mode == "markers" for trace in zero_markers)

    analysis.marker_label_check.setChecked(True)
    panel.refresh_controller_plot()
    figure = panel.current_controller_figure()
    zero_markers = [trace for trace in figure.data if "= 0" in (trace.name or "")]
    assert all(trace.mode == "markers+text" for trace in zero_markers)
    assert any(text.startswith("(") and text.endswith(")") for trace in zero_markers for text in trace.text)
    assert all("x=" not in text and "y=" not in text for trace in zero_markers for text in trace.text)
    summary = analysis.summary_text()
    assert "dy/dx = 0" in summary
    assert "d2y/dx2 = 0" in summary


def test_notebook_plot_panel_post_processing_preset_enables_extrema_and_inflection_review():
    _app()
    panel = NotebookPlotPanel()

    analysis = panel.main_card._analysis
    analysis.post_preset_combo.setCurrentIndex(
        analysis.post_preset_combo.findData("extrema_inflection")
    )

    assert analysis.first_deriv_check.isChecked()
    assert analysis.second_deriv_check.isChecked()
    assert analysis.mark_first_zero_check.isChecked()
    assert analysis.mark_second_zero_check.isChecked()


def test_notebook_plot_panel_applies_per_series_style_controls():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "x": np.array([0.0, 1.0, 2.0]),
            "phi": np.array([1.0, 2.0, 3.0]),
            "phi_exact": np.array([1.5, 2.5, 3.5]),
        }
    )

    panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
    panel.y_combo.set_checked_values(["phi", "phi_exact"])
    panel._series_style_widgets["phi"]["plot_type"].setCurrentIndex(
        panel._series_style_widgets["phi"]["plot_type"].findData("markers")
    )
    panel._series_style_widgets["phi"]["line_style"].setCurrentIndex(
        panel._series_style_widgets["phi"]["line_style"].findData("dot")
    )
    panel._series_style_widgets["phi"]["marker_style"].setCurrentIndex(
        panel._series_style_widgets["phi"]["marker_style"].findData("triangle-up")
    )
    panel._series_style_widgets["phi"]["line_color"].setCurrentIndex(
        panel._series_style_widgets["phi"]["line_color"].findData("#000000")
    )
    panel._series_style_widgets["phi_exact"]["plot_type"].setCurrentIndex(
        panel._series_style_widgets["phi_exact"]["plot_type"].findData("lines")
    )
    panel._series_style_widgets["phi_exact"]["line_style"].setCurrentIndex(
        panel._series_style_widgets["phi_exact"]["line_style"].findData("dash")
    )
    panel._series_style_widgets["phi_exact"]["marker_style"].setCurrentIndex(
        panel._series_style_widgets["phi_exact"]["marker_style"].findData("diamond")
    )
    panel._series_style_widgets["phi_exact"]["line_color"].setCurrentIndex(
        panel._series_style_widgets["phi_exact"]["line_color"].findData("#9467bd")
    )
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    trace_map = {trace.name: trace for trace in figure.data}
    assert trace_map["phi"].mode == "markers"
    assert trace_map["phi"].line.dash == "dot"
    assert trace_map["phi"].marker.symbol == "triangle-up"
    assert trace_map["phi"].line.color == "#000000"
    assert trace_map["phi_exact"].mode == "lines"
    assert trace_map["phi_exact"].line.dash == "dash"
    assert trace_map["phi_exact"].marker.symbol == "diamond"
    assert trace_map["phi_exact"].line.color == "#9467bd"


def test_notebook_plot_panel_uses_auto_colors_by_default_for_multiple_series():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "x": np.array([0.0, 1.0, 2.0]),
            "force": np.array([0.0, 1.0, 0.0]),
            "phi": np.array([0.0, 0.5, 1.0]),
        }
    )

    panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
    panel.y_combo.set_checked_values(["force", "phi"])
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    trace_map = {trace.name: trace for trace in figure.data}
    assert trace_map["force"].line.color == "#1f77b4"
    assert trace_map["phi"].line.color == "#ff7f0e"


def test_notebook_plot_panel_applies_scientific_style_controls():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "x": np.array([0.0, 1.0, 2.0]),
            "phi": np.array([1.0, 2.0, 3.0]),
        }
    )

    panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
    panel.y_combo.set_checked_values(["phi"])
    panel.graph_size_combo.setCurrentIndex(panel.graph_size_combo.findText("Square"))
    panel.font_size_combo.setCurrentIndex(panel.font_size_combo.findData(20))
    panel.line_width_combo.setCurrentIndex(panel.line_width_combo.findData(4))
    panel.marker_size_combo.setCurrentIndex(panel.marker_size_combo.findData(12))
    panel.show_grid_check.setChecked(False)
    panel.show_box_check.setChecked(True)
    panel.ticks_inside_check.setChecked(True)
    panel.minor_ticks_check.setChecked(True)
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    assert figure.layout.font.size == 20
    assert figure.data[0].line.width == 4
    assert figure.data[0].marker.size == 12
    assert figure.layout.xaxis.showgrid is False
    assert figure.layout.xaxis.showline is True
    assert figure.layout.xaxis.mirror == "allticks"
    assert figure.layout.xaxis.ticks == "inside"
    assert figure.layout.xaxis.ticklen == 10
    assert figure.layout.width == 700
    assert figure.layout.height == 700
    assert figure.layout.margin.r == 80
    assert figure.layout.xaxis.minor.ticks == "inside"
    assert panel.controller_plot.minimumHeight() >= 460
    assert figure.layout.xaxis.minor.ticklen == 5
    assert figure.layout.xaxis.automargin is True
    assert figure.layout.margin.b >= 80


def test_notebook_plot_panel_rebuilds_series_style_rows_if_selection_exists_without_signal():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "x": np.array([0.0, 1.0, 2.0]),
            "phi": np.array([1.0, 2.0, 3.0]),
            "phi_exact": np.array([1.5, 2.5, 3.5]),
        }
    )

    panel.x_combo.setCurrentIndex(panel.x_combo.findData("x"))
    panel.y_combo.set_checked_values(["phi", "phi_exact"])
    panel._series_style_widgets.clear()
    panel.refresh_controller_plot()

    assert set(panel._series_style_widgets.keys()) == {"phi", "phi_exact"}


def test_notebook_plot_panel_evolution_mode_builds_trace_from_selected_time_step():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "time": np.array([0.0, 1.0, 2.0]),
            "x": np.array([10.0, 20.0, 30.0, 40.0]),
            "history": np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
        }
    )

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("evolution"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("history"))
    panel.evolution_time_combo.setCurrentIndex(panel.evolution_time_combo.findData("time"))
    panel.evolution_value_combo.setCurrentIndex(panel.evolution_value_combo.findData("x"))
    panel.evolution_step_slider.setValue(1)
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    assert len(figure.data) == 1
    assert figure.data[0].type == "scatter"
    assert list(figure.data[0].x) == [10.0, 20.0, 30.0, 40.0]
    assert list(figure.data[0].y) == [5.0, 6.0, 7.0, 8.0]
    assert panel.evolution_step_slider.maximum() == 2
    assert panel.evolution_step_label.text() == "Step 1 / 2"


def test_notebook_plot_panel_evolution_mode_uses_column_index_when_axis_length_mismatches():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "time": np.array([0.0, 1.0, 2.0]),
            "bad_x": np.array([10.0, 20.0]),
            "history": np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
        }
    )

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("evolution"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("history"))
    panel.evolution_time_combo.setCurrentIndex(panel.evolution_time_combo.findData("time"))
    panel.evolution_value_combo.setCurrentIndex(panel.evolution_value_combo.findData("bad_x"))
    panel.evolution_step_slider.setValue(2)
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    assert list(figure.data[0].x) == [0.0, 1.0, 2.0, 3.0]
    assert list(figure.data[0].y) == [9.0, 10.0, 11.0, 12.0]


def test_build_notebook_evolution_figure_samples_five_static_time_traces():
    arrays_1d = {
        "time": np.linspace(0.0, 9.0, 10),
        "x": np.array([0.0, 1.0, 2.0]),
    }
    arrays_2d = {"history": np.arange(30.0).reshape(10, 3)}

    figure = build_notebook_evolution_figure(
        arrays_1d,
        arrays_2d,
        "history",
        "time",
        "x",
        "lines",
        "",
        "",
        "",
    )

    assert len(figure.data) == 5
    assert [trace.name for trace in figure.data] == ["t=0", "t=2", "t=4", "t=7", "t=9"]


def test_build_notebook_evolution_animation_figure_creates_full_time_playback():
    arrays_1d = {
        "time": np.array([0.0, 0.5, 1.0]),
        "x": np.array([10.0, 20.0, 30.0, 40.0]),
    }
    arrays_2d = {
        "history": np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )
    }

    figure = build_notebook_evolution_animation_figure(
        arrays_1d,
        arrays_2d,
        "history",
        "time",
        "x",
        "lines",
        "",
        "",
        "",
    )

    assert len(figure.data) == 1
    assert len(figure.frames) == 3
    assert list(figure.data[0].x) == [10.0, 20.0, 30.0, 40.0]
    assert list(figure.data[0].y) == [1.0, 2.0, 3.0, 4.0]
    assert list(figure.frames[2].data[0].y) == [9.0, 10.0, 11.0, 12.0]
    assert figure.layout.updatemenus[0].buttons[0].label == "Play"
    assert figure.layout.updatemenus[0].x == 0.78
    assert figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is False
    assert figure.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] == 0
    assert figure.layout.updatemenus[0].buttons[0].args[1]["mode"] == "immediate"
    assert figure.layout.sliders[0].currentvalue.prefix == "Time "
    assert figure.layout.sliders[0].pad.t == 58
    assert figure.layout.sliders[0].steps[1].args[1]["frame"]["redraw"] is False
    assert figure.layout.sliders[0].steps[1].args[1]["transition"]["duration"] == 0
    assert figure.layout.sliders[0].steps[1].args[1]["mode"] == "immediate"
    assert figure.layout.sliders[0].steps[1].label == "t=0.5"
    assert figure.layout.margin.b == 170


def test_build_notebook_evolution_animation_figure_keeps_at_least_100_available_frames():
    rows = 120
    arrays_1d = {
        "time": np.linspace(0.0, 12.0, rows),
        "x": np.array([0.0, 1.0]),
    }
    arrays_2d = {"history": np.arange(rows * 2.0).reshape(rows, 2)}

    figure = build_notebook_evolution_animation_figure(
        arrays_1d,
        arrays_2d,
        "history",
        "time",
        "x",
        "lines",
        "",
        "",
        "",
    )

    assert len(figure.frames) == rows
    assert len(figure.frames) >= 100
    assert list(figure.frames[-1].data[0].y) == [238.0, 239.0]


def test_notebook_plot_panel_animation_toggle_uses_full_evolution_frames():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "time": np.array([0.0, 1.0, 2.0]),
            "x": np.array([10.0, 20.0, 30.0, 40.0]),
            "history": np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ]
            ),
        }
    )

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("evolution"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("history"))
    panel.evolution_time_combo.setCurrentIndex(panel.evolution_time_combo.findData("time"))
    panel.evolution_value_combo.setCurrentIndex(panel.evolution_value_combo.findData("x"))
    panel.evolution_animate_check.setChecked(True)
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    assert len(figure.frames) == 3
    assert "animated" in panel.controller_status.text()


def test_notebook_plot_panel_heatmap_mode_uses_3d_history_frames():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "time_history": np.array([0.0, 0.1, 0.2]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 2.0]),
            "u_history": np.arange(12.0).reshape(3, 2, 2),
        }
    )

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("heatmap"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("u_history"))
    panel.evolution_time_combo.setCurrentIndex(panel.evolution_time_combo.findData("time_history"))
    panel.evolution_value_combo.setCurrentIndex(panel.evolution_value_combo.findData("x"))
    panel.evolution_y_combo.setCurrentIndex(panel.evolution_y_combo.findData("y"))
    panel.evolution_animate_check.setChecked(True)
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()

    assert figure.data[0].type == "heatmap"
    assert len(figure.frames) == 3
    assert "Heatmap" in panel.controller_status.text()


def test_notebook_plot_panel_contour_modes_use_3d_history_frames():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "time_history": np.array([0.0, 0.1]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 2.0]),
            "u_history": np.arange(8.0).reshape(2, 2, 2),
        }
    )

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("contour"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("u_history"))
    panel.evolution_time_combo.setCurrentIndex(panel.evolution_time_combo.findData("time_history"))
    panel.evolution_value_combo.setCurrentIndex(panel.evolution_value_combo.findData("x"))
    panel.evolution_y_combo.setCurrentIndex(panel.evolution_y_combo.findData("y"))
    panel.evolution_animate_check.setChecked(True)
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    assert figure.data[0].type == "contour"
    assert figure.data[0].contours.coloring == "lines"
    assert len(figure.frames) == 2
    assert "Contour" in panel.controller_status.text()

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("contourf"))
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    assert figure.data[0].type == "contour"
    assert figure.data[0].contours.coloring == "heatmap"
    assert len(figure.frames) == 2
    assert "Filled contour" in panel.controller_status.text()

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("contourfb"))
    panel.refresh_controller_plot()

    figure = panel.current_controller_figure()
    assert figure.data[0].type == "contour"
    assert figure.data[0].contours.coloring == "fill"
    assert figure.data[0].contours.showlines is True
    assert len(figure.frames) == 2
    assert "Banded contourf" in panel.controller_status.text()


def test_notebook_plot_panel_reselects_first_matrix_after_stale_evolution_selection():
    _app()
    panel = NotebookPlotPanel()
    panel.set_namespace(
        {
            "history_old": np.ones((2, 3)),
        }
    )
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("evolution"))
    panel.evolution_matrix_combo.setCurrentIndex(panel.evolution_matrix_combo.findData("history_old"))

    panel.set_namespace(
        {
            "u_history": np.arange(12.0).reshape(4, 3),
        }
    )

    assert panel.evolution_matrix_combo.currentData() == "u_history"
    assert len(panel.current_controller_figure().data) == 4


def test_quick_graph_evolution_animation_toggle_uses_full_time_playback():
    _app()
    panel = QuickGraphPreviewPanel()
    panel.set_namespace(
        {
            "time": np.array([0.0, 1.0]),
            "x": np.array([0.0, 1.0, 2.0]),
            "history": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        }
    )

    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("evolution"))
    panel._evo_matrix_combo.setCurrentIndex(panel._evo_matrix_combo.findData("history"))
    panel._evo_time_combo.setCurrentIndex(panel._evo_time_combo.findData("time"))
    panel._evo_value_combo.setCurrentIndex(panel._evo_value_combo.findData("x"))
    panel._evo_animate_check.setChecked(True)
    panel.refresh()

    figure = panel.current_figure()
    assert len(figure.frames) == 2
    assert list(figure.frames[1].data[0].y) == [4.0, 5.0, 6.0]
    assert panel._plot_view.minimumHeight() == 500
    assert panel._plot_view.maximumHeight() == 500
    assert figure.layout.height == 500


def test_quick_graph_heatmap_mode_uses_3d_history_frames():
    _app()
    panel = QuickGraphPreviewPanel()
    panel.set_namespace(
        {
            "time_history": np.array([0.0, 0.1]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 2.0]),
            "u_history": np.arange(8.0).reshape(2, 2, 2),
        }
    )

    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("heatmap"))
    panel._evo_matrix_combo.setCurrentIndex(panel._evo_matrix_combo.findData("u_history"))
    panel._evo_time_combo.setCurrentIndex(panel._evo_time_combo.findData("time_history"))
    panel._evo_value_combo.setCurrentIndex(panel._evo_value_combo.findData("x"))
    panel._evo_y_combo.setCurrentIndex(panel._evo_y_combo.findData("y"))
    panel._evo_animate_check.setChecked(True)
    panel.refresh()

    figure = panel.current_figure()

    assert figure.data[0].type == "heatmap"
    assert len(figure.frames) == 2


def test_quick_graph_heatmap_scale_combo_can_fix_full_history_range():
    _app()
    panel = QuickGraphPreviewPanel()
    panel.set_namespace(
        {
            "time_history": np.array([0.0, 0.1]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 2.0]),
            "u_history": np.array(
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[-4.0, 5.0], [6.0, 7.0]],
                ]
            ),
        }
    )

    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("heatmap"))
    panel._evo_matrix_combo.setCurrentIndex(panel._evo_matrix_combo.findData("u_history"))
    panel._evo_time_combo.setCurrentIndex(panel._evo_time_combo.findData("time_history"))
    panel._evo_value_combo.setCurrentIndex(panel._evo_value_combo.findData("x"))
    panel._evo_y_combo.setCurrentIndex(panel._evo_y_combo.findData("y"))
    panel._evo_animate_check.setChecked(True)
    panel._heatmap_scale_combo.setCurrentIndex(panel._heatmap_scale_combo.findData("fixed"))
    panel.refresh()

    figure = panel.current_figure()

    assert figure.data[0].zauto is False
    assert figure.data[0].zmin == -4.0
    assert figure.data[0].zmax == 7.0


def test_quick_graph_heatmap_range_controls_default_and_manual_lock_scale():
    _app()
    panel = QuickGraphPreviewPanel()
    panel.set_namespace(
        {
            "time_history": np.array([0.0, 0.1]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 1.0]),
            "u_history": np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[-6.0, 7.0], [8.0, 9.0]],
                ]
            ),
        }
    )

    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("heatmap"))
    panel._evo_matrix_combo.setCurrentIndex(panel._evo_matrix_combo.findData("u_history"))

    assert panel._heatmap_min_spin.value() == -6.0
    assert panel._heatmap_max_spin.value() == 9.0

    panel._heatmap_min_spin.setValue(-2.0)
    panel._heatmap_max_spin.setValue(5.0)

    figure = panel.current_figure()

    assert panel._heatmap_scale_combo.currentData() == "fixed"
    assert figure.data[0].zauto is False
    assert figure.data[0].zmin == -2.0
    assert figure.data[0].zmax == 5.0


def test_quick_graph_contour_modes_use_3d_history_frames():
    _app()
    panel = QuickGraphPreviewPanel()
    panel.set_namespace(
        {
            "time_history": np.array([0.0, 0.1]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 2.0]),
            "u_history": np.arange(8.0).reshape(2, 2, 2),
        }
    )

    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("contour"))
    panel._evo_matrix_combo.setCurrentIndex(panel._evo_matrix_combo.findData("u_history"))
    panel._evo_time_combo.setCurrentIndex(panel._evo_time_combo.findData("time_history"))
    panel._evo_value_combo.setCurrentIndex(panel._evo_value_combo.findData("x"))
    panel._evo_y_combo.setCurrentIndex(panel._evo_y_combo.findData("y"))
    panel._evo_animate_check.setChecked(True)
    panel.refresh()

    figure = panel.current_figure()
    assert figure.data[0].type == "contour"
    assert figure.data[0].contours.coloring == "lines"
    assert len(figure.frames) == 2

    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("contourf"))
    panel.refresh()

    figure = panel.current_figure()
    assert figure.data[0].type == "contour"
    assert figure.data[0].contours.coloring == "heatmap"
    assert len(figure.frames) == 2

    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("contourfb"))
    panel.refresh()

    figure = panel.current_figure()
    assert figure.data[0].type == "contour"
    assert figure.data[0].contours.coloring == "fill"
    assert figure.data[0].contours.showlines is True
    assert len(figure.frames) == 2


def test_quick_graph_evolution_controls_use_two_rows():
    _app()
    panel = QuickGraphPreviewPanel()

    layout = panel._evo_widget.layout()

    assert layout is not None
    assert layout.count() == 2
    assert panel._evo_top_row.count() == 4
    assert panel._evo_bottom_row.count() == 6


def test_quick_graph_prefers_current_u_over_history_axes_for_series_y():
    _app()
    panel = QuickGraphPreviewPanel()

    panel.set_namespace(
        {
            "n_history": np.array([0.0, 1.0, 2.0]),
            "time_history": np.array([0.0, 0.1, 0.2]),
            "u": np.array([1.0, 2.0, 3.0]),
            "u_history": np.arange(9.0).reshape(3, 3),
        }
    )

    assert "u" in panel._y_combo.checked_values()


def test_quick_graph_reselects_first_matrix_after_stale_evolution_selection():
    _app()
    panel = QuickGraphPreviewPanel()
    panel.set_namespace({"history_old": np.ones((2, 3))})
    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("evolution"))
    panel._evo_matrix_combo.setCurrentIndex(panel._evo_matrix_combo.findData("history_old"))

    panel.set_namespace({"u_history": np.arange(12.0).reshape(4, 3)})

    assert panel._evo_matrix_combo.currentData() == "u_history"
    assert len(panel.current_figure().data) == 4


def test_quick_graph_evolution_run_slider_uses_saved_parameter_snapshot():
    _app()
    from pyside_app import array_store

    array_store.clear()
    panel = QuickGraphPreviewPanel()
    panel.set_namespace(
        {
            "x": np.array([0.0, 1.0, 2.0, 3.0]),
            "time": np.array([0.0, 1.0]),
            "history": np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        }
    )
    array_store.store_run(
        {
            "x": np.array([10.0, 20.0, 30.0, 40.0]),
            "time": np.array([0.0, 1.0]),
            "history": np.array([[11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]]),
        },
        "D=1",
    )
    array_store.store_run(
        {
            "x": np.array([100.0, 200.0, 300.0, 400.0]),
            "time": np.array([0.0, 1.0]),
            "history": np.array([[21.0, 22.0, 23.0, 24.0], [25.0, 26.0, 27.0, 28.0]]),
        },
        "D=2",
    )

    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("evolution"))
    panel._evo_matrix_combo.setCurrentIndex(panel._evo_matrix_combo.findData("history"))
    panel._evo_time_combo.setCurrentIndex(panel._evo_time_combo.findData("time"))
    panel._evo_value_combo.setCurrentIndex(panel._evo_value_combo.findData("x"))
    panel._update_run_slider(reset_to_current=False)
    panel._evo_run_slider.setValue(0)
    panel.refresh()

    figure = panel.current_figure()
    assert panel._evo_run_label.text() == "D=2"
    assert figure.layout.title.text == "D=2"
    assert list(figure.data[0].x) == [100.0, 200.0, 300.0, 400.0]
    assert list(figure.data[0].y) == [21.0, 22.0, 23.0, 24.0]


def test_quick_graph_cell_run_resets_heatmap_slider_to_current_data():
    _app()
    from pyside_app import array_store

    array_store.clear()
    array_store.store_run(
        {
            "time_history": np.array([0.0, 0.1]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 1.0]),
            "u_history": np.array(
                [
                    [[100.0, 101.0], [102.0, 103.0]],
                    [[104.0, 105.0], [106.0, 107.0]],
                ]
            ),
        },
        "old",
    )

    panel = QuickGraphPreviewPanel()
    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("heatmap"))
    panel.set_namespace(
        {
            "time_history": np.array([0.0, 0.1]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 1.0]),
            "u_history": np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
        }
    )

    figure = panel.current_figure()

    assert panel._evo_run_label.text() == "current"
    assert panel._evo_run_slider.value() == panel._evo_run_slider.maximum()
    assert list(figure.data[0].z[0]) == [1.0, 2.0]
