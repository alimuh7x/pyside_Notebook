from __future__ import annotations

import plotly.graph_objects as go
import pytest


from pyside_app.execution_engine import ExecutionEngine, detect_cell_parameters


def test_engine_executes_code_and_returns_last_expression():
    engine = ExecutionEngine()

    result = engine.execute("x = 41\nx + 1")

    assert result.error is None
    assert result.outputs
    assert result.outputs[-1].kind == "value"
    assert result.outputs[-1].data["text"] == "42"
    assert engine.get_namespace()["x"] == 41


def test_engine_reuses_namespace_between_runs():
    engine = ExecutionEngine()

    first = engine.execute("scale = 3")
    second = engine.execute("scale * 7")

    assert first.error is None
    assert second.error is None
    assert second.outputs[-1].data["text"] == "21"


def test_engine_captures_stdout():
    engine = ExecutionEngine()

    result = engine.execute("print('hello desktop notebook')")

    assert result.error is None
    assert result.stdout.strip() == "hello desktop notebook"
    assert result.outputs[0].kind == "stdout"


def test_engine_supports_imports_and_functions():
    engine = ExecutionEngine()

    result = engine.execute(
        "import math\n"
        "def area(r):\n"
        "    return math.pi * r**2\n"
        "area(2)"
    )

    assert result.error is None
    assert result.outputs[-1].data["text"].startswith("12.56")


def test_engine_reports_plotly_figures_as_plot_outputs():
    engine = ExecutionEngine()

    result = engine.execute(
        "import plotly.graph_objects as go\n"
        "go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])"
    )

    assert result.error is None
    assert result.outputs[-1].kind == "plotly"
    assert "html" in result.outputs[-1].data
    assert 'src="https://cdn.plot.ly' not in result.outputs[-1].data["html"]
    assert "src='https://cdn.plot.ly" not in result.outputs[-1].data["html"]


def test_engine_captures_matplotlib_figures_after_show():
    engine = ExecutionEngine()

    result = engine.execute(
        "import matplotlib.pyplot as plt\n"
        "plt.plot([0, 1], [0, 1])\n"
        "plt.title('Line')\n"
        "plt.show()"
    )

    assert result.error is None
    plot_outputs = [output for output in result.outputs if output.kind == "plotly"]
    assert plot_outputs
    assert "Plotly.newPlot" in plot_outputs[-1].data["html"]
    assert 'src="https://cdn.plot.ly' not in plot_outputs[-1].data["html"]
    assert "src='https://cdn.plot.ly" not in plot_outputs[-1].data["html"]


def test_engine_falls_back_to_static_html_when_matplotlib_conversion_fails():
    engine = ExecutionEngine()

    result = engine.execute(
        "import matplotlib.pyplot as plt\n"
        "plt.imshow([[0, 1], [1, 0]])\n"
        "plt.colorbar()\n"
        "plt.show()"
    )

    assert result.error is None
    assert any(output.kind in {"plotly", "html"} for output in result.outputs)


def test_engine_supports_sympy_when_available():
    engine = ExecutionEngine()

    result = engine.execute(
        "import sympy as sp\n"
        "x = sp.Symbol('x')\n"
        "sp.expand((x + 1)**2)"
    )

    assert result.error is None
    assert "x**2" in result.outputs[-1].data["text"]


def test_engine_restart_kernel_clears_user_namespace():
    engine = ExecutionEngine()
    engine.execute("value = 99")

    engine.restart()
    result = engine.execute("'value' in globals()")

    assert result.error is None
    assert result.outputs[-1].data["text"] == "False"
    assert "value" not in engine.get_namespace()


def test_engine_returns_traceback_for_errors():
    engine = ExecutionEngine()

    result = engine.execute("1 / 0")

    assert result.error is not None
    assert "ZeroDivisionError" in result.error


def test_engine_value_output_is_json_safe_text_only():
    engine = ExecutionEngine()

    result = engine.execute("x = [1, 2, 3]\nx")

    assert result.error is None
    assert result.outputs[-1].kind == "value"
    assert "text" in result.outputs[-1].data
    assert "value" not in result.outputs[-1].data


def test_engine_does_not_emit_value_output_for_none_result():
    engine = ExecutionEngine()

    result = engine.execute(
        "import matplotlib.pyplot as plt\n"
        "plt.plot([0, 1], [0, 1])\n"
        "plt.show()"
    )

    assert result.error is None
    assert all(output.kind != "value" for output in result.outputs)


def test_engine_blocks_dangerous_builtins_and_imports():
    engine = ExecutionEngine()

    open_result = engine.execute("open('forbidden.txt', 'w')")
    import_result = engine.execute("import os\nos.getcwd()")

    assert open_result.error is not None
    assert "NameError" in open_result.error
    assert import_result.error is not None
    assert "ImportError" in import_result.error


def test_engine_namespace_snapshot_returns_copy():
    engine = ExecutionEngine()
    engine.execute("value = 42")

    snapshot = engine.get_namespace()
    snapshot["value"] = -1

    assert engine.get_namespace()["value"] == 42


def test_detect_cell_parameters_filters_bookkeeping_and_prioritizes_physics_knobs():
    params = detect_cell_parameters(
        "save_id = 0\n"
        "n_save = 12\n"
        "D = 0.1\n"
        "dt = 0.0005\n"
        "Nx = 400\n"
        "omega_phi = 0.1\n"
        "E_alpha = -0.30\n"
    )

    assert "save_id" not in params
    assert "n_save" not in params
    assert list(params)[:3] == ["D", "dt", "E_alpha"]
    assert params["D"]["scale"] == "log"
    assert params["dt"]["scale"] == "log"
    assert params["Nx"]["is_int"] is True
    assert params["omega_phi"]["min"] == 0.0
    assert params["omega_phi"]["max"] == 1.0


def test_detect_cell_parameters_uses_log_scale_for_tiny_positive_values():
    params = detect_cell_parameters("tolerance = 1e-8\nrate_constant = 2500.0")

    assert params["tolerance"]["scale"] == "log"
    assert params["rate_constant"]["scale"] == "log"
