from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QLabel, QWidget

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyside_app.execution_engine import ExecutionOutput, ExecutionResult
from pyside_app.notebook_plot_panel import (
    NotebookPlotPanel,
    build_notebook_plot_figure,
    extract_notebook_array_variables,
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
    assert figure.layout.width is None
    assert figure.layout.height == 700


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
