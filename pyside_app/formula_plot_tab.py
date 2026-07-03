from __future__ import annotations

import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import CubicSpline

from pyside_app.plot_view import PlotView
from utils.formula_parser import FormulaValidationError, evaluate_formula, evaluate_formula_2d, extract_formula_variables


_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
_ARROW_UP = str(_ASSET_DIR / "qt_arrow_up.svg").replace("\\", "/")
_ARROW_DOWN = str(_ASSET_DIR / "qt_arrow_down.svg").replace("\\", "/")


FORMULA_TAB_STYLE = """
#FormulaPlotTab {
    background: #21252b;
    font-size: 12px;
}
#FormulaPlotTab QLineEdit, #FormulaPlotTab QComboBox, #FormulaPlotTab QSpinBox {
    min-height: 26px;
    padding: 4px 8px;
    border: 1px solid #3e4451;
    border-radius: 6px;
    background: #2c313a;
    color: #d7dae0;
    font-size: 12px;
}
#FormulaPlotTab QComboBox {
    padding-right: 28px;
}
#FormulaPlotTab QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: right center;
    width: 22px;
    border-left: 1px solid #3e4451;
    background: #3e4451;
}
#FormulaPlotTab QComboBox::down-arrow {
    image: url(__ARROW_DOWN__);
    width: 10px;
    height: 10px;
}
#FormulaPlotTab QSpinBox {
    padding-right: 28px;
}
#FormulaPlotTab QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #3e4451;
    background: #3e4451;
}
#FormulaPlotTab QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 1px solid #3e4451;
    background: #3e4451;
}
#FormulaPlotTab QSpinBox::up-arrow {
    image: url(__ARROW_UP__);
    width: 10px;
    height: 10px;
}
#FormulaPlotTab QSpinBox::down-arrow {
    image: url(__ARROW_DOWN__);
    width: 10px;
    height: 10px;
}
#FormulaPlotTab QComboBox QAbstractItemView {
    background: #282c34;
    color: #f8fafc;
    selection-background-color: #61afef;
    selection-color: #0b1220;
    border: 1px solid #4b5563;
    outline: 0;
    padding: 4px;
}
#FormulaPlotTab QComboBox QAbstractItemView::item {
    min-height: 24px;
    padding: 4px 8px;
}
#FormulaPlotTab QLabel, #FormulaPlotTab QCheckBox {
    color: #d7dae0;
    font-size: 12px;
}
#FormulaPlotTab QCheckBox {
    spacing: 8px;
}
#FormulaPlotTab QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 2px solid #61afef;
    border-radius: 4px;
    background: #21252b;
}
#FormulaPlotTab QCheckBox::indicator:hover {
    border-color: #9cdcfe;
    background: #2c313a;
}
#FormulaPlotTab QCheckBox::indicator:checked {
    background: #61afef;
    border: 2px solid #f8fafc;
}
#FormulaPlotTab QTextBrowser {
    border: 1px solid #3e4451;
    border-radius: 7px;
    background: #282c34;
    color: #d7dae0;
    padding: 8px;
    font-size: 12px;
}
""".strip().replace("__ARROW_UP__", _ARROW_UP).replace("__ARROW_DOWN__", _ARROW_DOWN)

FORMULA_BUTTON_STYLE = """
QPushButton {
    min-height: 26px;
    padding: 4px 10px;
    border: none;
    border-radius: 6px;
    background: #2c313a;
    color: #d7dae0;
    font-size: 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #3e4451;
}
""".strip()

FORMULA_STATUS_READY_STYLE = "color: #d7dae0; font-weight: 600; padding: 6px 0;"
FORMULA_STATUS_UPDATED_STYLE = "color: #98c379; font-weight: 600; padding: 6px 0;"
FORMULA_STATUS_ERROR_STYLE = "color: #e06c75; font-weight: 600; padding: 6px 0;"

FORMULA_GRAPH_WIDTH = 1000
FORMULA_GRAPH_HEIGHT = 700
FORMULA_GRAPH_2D_HEIGHT = 760
FORMULA_SLICE_HEIGHT = 280

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

FORMULA_EXAMPLES = [
    ("Sine Wave", "sin(x)"),
    ("Damped Sine", "a*exp(-b*x) * sin(c*x)"),
    ("Gaussian Peak", "a*exp(-((x-b)**2)/(2*c**2))"),
    ("Logistic Front", "1 / (1 + exp(-a*(x-b)))"),
    ("Exponential Decay", "a*exp(-x/tau) + c"),
    ("Relaxation Growth", "a*(1-exp(-x/tau)) + c"),
    ("Sigmoid Window", "a / (1 + exp(-(x-b)/c)) + d"),
    ("Arrhenius-like", "a*exp(-q/(r*x))"),
    ("Hyperbola", "a/(x-b) + c"),
    ("Parabola", "x**2"),
    ("Cubic", "x**3 - 3*x"),
]

TRACE_COLORS = [
    ("Black", "black"),
    ("Red", "#d62728"),
    ("Blue", "#1f77b4"),
    ("Green", "#2ca02c"),
    ("Orange", "#ff7f0e"),
    ("Purple", "#9467bd"),
]

TRACE_DASHES = [("Solid", "solid"), ("Dash", "dash"), ("Dot", "dot"), ("Dash Dot", "dashdot")]
COLOR_SCALES = ["Viridis", "Cividis", "Plasma", "Turbo", "RdBu"]

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

PRESET_PARAM_DEFAULTS = {
    "a*exp(-b*x) * sin(c*x)": {
        "a": {"value": 1.0, "min": -5.0, "max": 5.0, "step": 0.1},
        "b": {"value": 0.15, "min": 0.0, "max": 2.0, "step": 0.01},
        "c": {"value": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
    },
    "a*exp(-((x-b)**2)/(2*c**2))": {
        "a": {"value": 1.0, "min": -5.0, "max": 5.0, "step": 0.1},
        "b": {"value": 0.0, "min": -10.0, "max": 10.0, "step": 0.1},
        "c": {"value": 1.5, "min": 0.1, "max": 10.0, "step": 0.1},
    },
    "1 / (1 + exp(-a*(x-b)))": {
        "a": {"value": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
        "b": {"value": 0.0, "min": -10.0, "max": 10.0, "step": 0.1},
    },
    "a*exp(-x/tau) + c": {
        "a": {"value": 1.0, "min": -5.0, "max": 5.0, "step": 0.1},
        "tau": {"value": 2.0, "min": 0.1, "max": 20.0, "step": 0.1},
        "c": {"value": 0.0, "min": -5.0, "max": 5.0, "step": 0.1},
    },
    "a*(1-exp(-x/tau)) + c": {
        "a": {"value": 1.0, "min": -5.0, "max": 5.0, "step": 0.1},
        "tau": {"value": 2.0, "min": 0.1, "max": 20.0, "step": 0.1},
        "c": {"value": 0.0, "min": -5.0, "max": 5.0, "step": 0.1},
    },
    "a / (1 + exp(-(x-b)/c)) + d": {
        "a": {"value": 1.0, "min": -5.0, "max": 5.0, "step": 0.1},
        "b": {"value": 0.0, "min": -10.0, "max": 10.0, "step": 0.1},
        "c": {"value": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
        "d": {"value": 0.0, "min": -5.0, "max": 5.0, "step": 0.1},
    },
    "a*exp(-q/(r*x))": {
        "a": {"value": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
        "q": {"value": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
        "r": {"value": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
    },
    "a/(x-b) + c": {
        "a": {"value": 1.0, "min": -10.0, "max": 10.0, "step": 0.1},
        "b": {"value": 0.0, "min": -10.0, "max": 10.0, "step": 0.1},
        "c": {"value": 0.0, "min": -10.0, "max": 10.0, "step": 0.1},
    },
}

PRESET_PANEL_DEFAULTS = {
    "a*exp(-q/(r*x))": {
        "x_min": 200.0,
        "x_max": 1600.0,
        "points": 600,
        "x_axis_title": "Temperature",
        "y_axis_title": "Rate",
        "interval_min": 300.0,
        "interval_max": 1200.0,
        "analysis_x0": 800.0,
    },
}

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


def _new_formula_row_id() -> str:
    row_id = f"formula_{int(time.time() * 1000)}"
    print(f"[debug][formula-plot] new_formula_row_id id={row_id!r}", flush=True)
    return row_id


def _default_formula_row(expression: str = "sin(x)", row_id: str | None = None) -> dict[str, Any]:
    row = {
        "id": row_id or _new_formula_row_id(),
        "expression": expression,
        "label": expression,
        "visible": True,
        "color": "black",
        "dash": "solid",
        "width": 3.0,
    }
    print(f"[debug][formula-plot] default_formula_row row={row!r}", flush=True)
    return row


@dataclass
class FormulaPlotState:
    panel_type: str = "1d"
    preset_2d: str = "periodic_surface"
    example_formula: str = "sin(x)"
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
    contour_levels_2d: int = 12
    contour_style_2d: str = "filled"
    auto_z_range_2d: bool = True
    z_min_2d: float | None = None
    z_max_2d: float | None = None
    probe_x_2d: float = 0.0
    probe_y_2d: float = 0.0
    surface_colorscale: str = "Viridis"
    show_grid: bool = True
    show_legend: bool = True
    show_derivative: bool = False
    show_second_derivative: bool = False
    show_antiderivative: bool = False
    show_tangent: bool = False
    show_normal: bool = False
    show_root_markers: bool = False
    show_extrema_markers: bool = False
    show_intersection_markers: bool = False
    show_area_shading: bool = False
    show_monotonicity_regions: bool = True
    show_concavity_regions: bool = True
    analysis_formula: str = ""
    analysis_x0: float = 0.0
    interval_min: float = -2.0
    interval_max: float = 2.0
    threshold_value: float | None = None
    click_mode: str = "focus"
    pending_interval_start: float | None = None
    formulas: list[dict[str, Any]] = field(default_factory=lambda: [_default_formula_row("sin(x)", "formula_1")])
    params: dict[str, dict[str, Any]] = field(default_factory=lambda: {"a": {"value": 1.0, "min": -5.0, "max": 5.0, "step": 0.1}})
    last_summary_text: str = ""
    last_details_text: str = ""


def parameter_preset_for_name(name: str) -> dict[str, Any]:
    key = (name or "").strip()
    preset = {**DEFAULT_PARAMETER_PRESET, **PARAMETER_PRESETS.get(key, {})}
    print(f"[debug][formula-plot] parameter_preset_for_name name={key!r} preset={preset!r}", flush=True)
    return preset


def _as_float(value: Any, fallback: float) -> float:
    try:
        if value in (None, ""):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        print(f"[debug][formula-plot] as_float fallback value={value!r} fallback={fallback!r}", flush=True)
        return fallback


def _as_int(value: Any, fallback: int) -> int:
    try:
        if value in (None, ""):
            return fallback
        return int(float(value))
    except (TypeError, ValueError):
        print(f"[debug][formula-plot] as_int fallback value={value!r} fallback={fallback!r}", flush=True)
        return fallback


def _filter_real_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = array[np.isfinite(array)]
    print(f"[debug][formula-plot] filter_real_values total={array.size} finite={result.size}", flush=True)
    return result


def _dedupe_points(values: Any, tolerance: float = 1e-7) -> list[float]:
    result: list[float] = []
    for value in sorted(_filter_real_values(values)):
        if not result or abs(float(value) - result[-1]) > tolerance:
            result.append(float(value))
    print(f"[debug][formula-plot] dedupe_points count={len(result)} values={result[:6]!r}", flush=True)
    return result


def _safe_spline(x_values: np.ndarray, y_values: np.ndarray) -> CubicSpline | None:
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(finite_mask) < 4:
        print("[debug][formula-plot] safe_spline skipped not_enough_finite", flush=True)
        return None
    x_valid = x_values[finite_mask]
    y_valid = y_values[finite_mask]
    if np.unique(x_valid).size < 4:
        print("[debug][formula-plot] safe_spline skipped not_enough_unique_x", flush=True)
        return None
    print(f"[debug][formula-plot] safe_spline created points={len(x_valid)}", flush=True)
    return CubicSpline(x_valid, y_valid, extrapolate=False)


def _solve_spline_roots(x_values: np.ndarray, y_values: np.ndarray, limit: int = 12) -> list[float]:
    print(f"[debug][formula-plot] solve_spline_roots start limit={limit}", flush=True)
    spline = _safe_spline(x_values, y_values)
    if spline is not None:
        roots = _dedupe_points(spline.roots(extrapolate=False))
        if roots:
            print(f"[debug][formula-plot] solve_spline_roots spline roots={roots[:limit]!r}", flush=True)
            return roots[:limit]

    roots: list[float] = []
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_valid = x_values[finite_mask]
    y_valid = y_values[finite_mask]
    for idx in np.where(np.diff(np.signbit(y_valid)))[0]:
        x0, x1 = x_valid[idx], x_valid[idx + 1]
        y0, y1 = y_valid[idx], y_valid[idx + 1]
        if np.isclose(y0, y1):
            roots.append(float(x0))
        else:
            roots.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    result = _dedupe_points(roots)[:limit]
    print(f"[debug][formula-plot] solve_spline_roots fallback roots={result!r}", flush=True)
    return result


def _find_extrema(x_values: np.ndarray, y_values: np.ndarray, limit: int = 16) -> tuple[list[dict[str, float | str]], CubicSpline | None]:
    print(f"[debug][formula-plot] find_extrema start limit={limit}", flush=True)
    spline = _safe_spline(x_values, y_values)
    if spline is None:
        return [], None
    first = spline.derivative()
    second = spline.derivative(2)
    extrema: list[dict[str, float | str]] = []
    for x_pos in _dedupe_points(first.roots(extrapolate=False)):
        y_pos = float(spline(x_pos))
        curvature = float(second(x_pos))
        point_type = "minimum" if curvature > 0 else "maximum" if curvature < 0 else "stationary"
        extrema.append({"x": float(x_pos), "y": y_pos, "type": point_type})
    extrema.sort(key=lambda item: float(item["x"]))
    print(f"[debug][formula-plot] find_extrema count={len(extrema)}", flush=True)
    return extrema[:limit], spline


def _find_intersections(x_values: np.ndarray, y_values_a: np.ndarray, y_values_b: np.ndarray, limit: int = 12) -> list[dict[str, float]]:
    print(f"[debug][formula-plot] find_intersections start limit={limit}", flush=True)
    roots = _solve_spline_roots(x_values, y_values_a - y_values_b, limit=limit)
    spline_a = _safe_spline(x_values, y_values_a)
    intersections = []
    for x_pos in roots:
        y_pos = float(spline_a(x_pos)) if spline_a is not None else float(np.interp(x_pos, x_values, y_values_a))
        intersections.append({"x": x_pos, "y": y_pos})
    print(f"[debug][formula-plot] find_intersections count={len(intersections)}", flush=True)
    return intersections


def _find_invalid_intervals(x_values: np.ndarray, y_values: np.ndarray) -> list[tuple[float, float]]:
    print("[debug][formula-plot] find_invalid_intervals start", flush=True)
    invalid_mask = ~np.isfinite(y_values)
    intervals: list[tuple[float, float]] = []
    start_idx = None
    for idx, is_invalid in enumerate(invalid_mask):
        if is_invalid and start_idx is None:
            start_idx = idx
        elif not is_invalid and start_idx is not None:
            intervals.append((float(x_values[start_idx]), float(x_values[idx - 1])))
            start_idx = None
    if start_idx is not None:
        intervals.append((float(x_values[start_idx]), float(x_values[-1])))
    print(f"[debug][formula-plot] find_invalid_intervals count={len(intervals)}", flush=True)
    return intervals


def _normalise_interval(start: Any, end: Any, x_min: float, x_max: float) -> tuple[float, float]:
    left = max(min(_as_float(start, x_min), x_max), x_min)
    right = max(min(_as_float(end, x_max), x_max), x_min)
    if left > right:
        left, right = right, left
    if np.isclose(left, right):
        right = min(x_max, left + max((x_max - x_min) * 0.01, 1e-6))
    print(f"[debug][formula-plot] normalise_interval start={start!r} end={end!r} result=({left}, {right})", flush=True)
    return left, right


def _format_intervals(intervals: list[tuple[float, float]]) -> str:
    return "none" if not intervals else ", ".join(f"[{start:.3g}, {end:.3g}]" for start, end in intervals[:5])


def _analysis_table_html(title: str, text: str) -> str:
    """Render newline/semicolon analysis text as a compact table."""
    print(f"[debug][formula-plot] analysis_table_html title={title!r} length={len(text)}", flush=True)
    rows: list[tuple[str, str]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
        else:
            key, value = "Info", line
        parts = [part.strip() for part in value.split(";") if part.strip()]
        if len(parts) > 1:
            rows.append((key.strip(), "<br>".join(escape(part) for part in parts)))
        else:
            rows.append((key.strip(), escape(value)))
    if not rows:
        rows.append(("Info", "none"))
    row_html = "".join(
        "<tr>"
        f"<th>{escape(key)}</th>"
        f"<td>{value}</td>"
        "</tr>"
        for key, value in rows
    )
    html = (
        "<html><head><style>"
        "body { background:#282c34; color:#d7dae0; font-family:Segoe UI, Arial, sans-serif; font-size:13px; }"
        "h3 { margin:0 0 8px 0; color:#ffffff; font-size:15px; }"
        "table { width:100%; border-collapse:collapse; }"
        "th { width:34%; text-align:left; vertical-align:top; color:#98c379; font-weight:700; padding:6px 8px; border-bottom:1px solid #3e4451; }"
        "td { text-align:left; vertical-align:top; color:#d7dae0; padding:6px 8px; border-bottom:1px solid #3e4451; line-height:1.35; }"
        "</style></head><body>"
        f"<h3>{escape(title)}</h3><table>{row_html}</table>"
        "</body></html>"
    )
    print(f"[debug][formula-plot] analysis_table_html rows={len(rows)}", flush=True)
    return html


def _configure_expanding_text_browser(browser: QTextBrowser) -> None:
    browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)


def _set_expanding_browser_html(browser: QTextBrowser, html: str) -> None:
    browser.setHtml(html)
    width = max(browser.viewport().width(), browser.width(), 320)
    browser.document().setTextWidth(max(1, width - 2 * browser.frameWidth()))
    height = int(browser.document().size().height()) + 2 * browser.frameWidth() + 16
    browser.setMinimumHeight(max(120, height))
    print(f"[debug][formula-plot] expanding_browser_html height={browser.minimumHeight()} width={width}", flush=True)


def _intervals_from_sign(x_values: np.ndarray, signal: np.ndarray, positive_label: str, negative_label: str) -> dict[str, list[tuple[float, float]]]:
    finite_mask = np.isfinite(signal)
    intervals = {positive_label: [], negative_label: [], "flat": []}
    if np.count_nonzero(finite_mask) < 2:
        return intervals
    x_valid = x_values[finite_mask]
    signal_valid = signal[finite_mask]
    signs = np.sign(signal_valid)
    signs[np.abs(signal_valid) < 1e-9] = 0
    start_x = float(x_valid[0])
    current_sign = int(signs[0])
    for idx in range(1, len(x_valid)):
        sign = int(signs[idx])
        if sign == current_sign:
            continue
        end_x = float(x_valid[idx - 1])
        if current_sign > 0:
            intervals[positive_label].append((start_x, end_x))
        elif current_sign < 0:
            intervals[negative_label].append((start_x, end_x))
        else:
            intervals["flat"].append((start_x, end_x))
        start_x = float(x_valid[idx - 1])
        current_sign = sign
    end_x = float(x_valid[-1])
    if current_sign > 0:
        intervals[positive_label].append((start_x, end_x))
    elif current_sign < 0:
        intervals[negative_label].append((start_x, end_x))
    else:
        intervals["flat"].append((start_x, end_x))
    return intervals


def _sync_panel_params(state: FormulaPlotState) -> None:
    print(f"[debug][formula-plot] sync_panel_params panel_type={state.panel_type!r}", flush=True)
    existing = state.params or {}
    discovered: dict[str, dict[str, Any]] = {}
    if state.panel_type == "2d":
        names = extract_formula_variables(state.expression_2d, coordinate_names=("x", "y"))
    else:
        names = []
        for formula in state.formulas:
            names.extend(extract_formula_variables(formula.get("expression", "")))
    for name in sorted(set(names)):
        discovered[name] = {**parameter_preset_for_name(name), **(existing.get(name, {}) or {})}
    if not discovered and state.panel_type == "1d":
        discovered["a"] = {**parameter_preset_for_name("a"), **(existing.get("a", {}) or {})}
    state.params = discovered
    print(f"[debug][formula-plot] sync_panel_params names={list(state.params)}", flush=True)


def _apply_formula_preset_defaults(state: FormulaPlotState, expression: str) -> None:
    print(f"[debug][formula-plot] apply_formula_preset_defaults expression={expression!r}", flush=True)
    _sync_panel_params(state)
    for name, config in PRESET_PARAM_DEFAULTS.get((expression or "").strip(), {}).items():
        state.params[name] = {**DEFAULT_PARAMETER_PRESET, **(state.params.get(name, {}) or {}), **config}
    for key, value in PRESET_PANEL_DEFAULTS.get((expression or "").strip(), {}).items():
        setattr(state, key, value)
    print(f"[debug][formula-plot] apply_formula_preset_defaults params={state.params!r}", flush=True)


def _label_should_follow_expression(label: str, previous_expression: str) -> bool:
    clean_label = (label or "").strip()
    clean_expression = (previous_expression or "").strip()
    return not clean_label or clean_label == clean_expression


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
    state.params = {}
    _sync_panel_params(state)
    for name, spec in preset.get("params", {}).items():
        state.params[name] = {**DEFAULT_PARAMETER_PRESET, **(state.params.get(name, {}) or {}), **dict(spec)}
    print(f"[debug][formula-plot] apply_2d_preset done expression={state.expression_2d!r} params={state.params!r}", flush=True)


def _parameter_values(state: FormulaPlotState, notebook_vars: dict[str, Any] | None = None) -> dict[str, float]:
    notebook_vars = notebook_vars or {}
    values: dict[str, float] = {}
    for name, spec in (state.params or {}).items():
        use_nb = bool((spec or {}).get("use_notebook", False))
        if use_nb and name in notebook_vars:
            try:
                values[name] = float(notebook_vars[name])
                print(f"[debug][formula-plot] parameter_values notebook name={name!r} value={values[name]}", flush=True)
                continue
            except (TypeError, ValueError):
                print(f"[debug][formula-plot] parameter_values notebook_invalid name={name!r} value={notebook_vars[name]!r}", flush=True)
        elif use_nb:
            print(f"[debug][formula-plot] parameter_values notebook_missing name={name!r}", flush=True)
        values[name] = float((spec or {}).get("value", 1.0))
    print(f"[debug][formula-plot] parameter_values values={values!r}", flush=True)
    return values


def _full_param_values(state: FormulaPlotState, notebook_vars: dict[str, Any] | None = None) -> dict[str, Any]:
    nb_numeric = {}
    for name, value in (notebook_vars or {}).items():
        try:
            nb_numeric[name] = float(value)
        except (TypeError, ValueError):
            pass
    values = {**nb_numeric, **_parameter_values(state, notebook_vars)}
    print(f"[debug][formula-plot] full_param_values keys={sorted(values)}", flush=True)
    return values


def build_formula_figure(state: FormulaPlotState, notebook_vars: dict[str, Any] | None = None) -> go.Figure:
    print(f"[debug][formula-plot] build_formula_figure panel_type={state.panel_type!r}", flush=True)
    if state.panel_type == "2d":
        return build_formula_figure_2d(state, notebook_vars)
    try:
        x_min = float(state.x_min)
        x_max = float(state.x_max)
        points = max(10, min(5000, int(state.points)))
        if x_min >= x_max:
            raise FormulaValidationError("X Min must be smaller than X Max")
        x_values = np.linspace(x_min, x_max, points)
        param_values = _full_param_values(state, notebook_vars)
        interval_min, interval_max = _normalise_interval(state.interval_min, state.interval_max, x_min, x_max)
        analysis_x0 = min(max(_as_float(state.analysis_x0, 0.0), x_min), x_max)
        threshold_enabled = state.threshold_value is not None
        threshold_value = _as_float(state.threshold_value, 0.0) if threshold_enabled else None
        fig = go.Figure()
        series_results: list[dict[str, Any]] = []
        error_lines: list[str] = []

        for formula in state.formulas:
            if not formula.get("visible", True):
                print(f"[debug][formula-plot] build_formula_figure skip_hidden row={formula.get('id')!r}", flush=True)
                continue
            expression = (formula.get("expression") or "").strip()
            label = (formula.get("label") or expression or "Formula").strip()
            if label == "sin(x)" and expression != "sin(x)":
                label = expression
            if not expression:
                error_lines.append(f"{label}: Please enter a formula")
                continue
            try:
                y_values = evaluate_formula(expression, x_values, param_values)
            except Exception as exc:
                print(f"[debug][formula-plot] build_formula_figure formula_error label={label!r} error={exc!r}", flush=True)
                error_lines.append(f"{label}: {exc}")
                continue
            spline = _safe_spline(x_values, y_values)
            extrema, extrema_spline = _find_extrema(x_values, y_values, limit=24)
            roots = _solve_spline_roots(x_values, y_values, limit=24)
            threshold_crossings = _solve_spline_roots(x_values, y_values - threshold_value, limit=24) if threshold_enabled else []
            invalid_intervals = _find_invalid_intervals(x_values, y_values)
            derivative_for_hover = spline.derivative()(x_values) if spline is not None else np.gradient(y_values, x_values)
            second_for_hover = spline.derivative(2)(x_values) if spline is not None else np.gradient(derivative_for_hover, x_values)
            cumulative = cumulative_trapezoid(np.nan_to_num(y_values, nan=0.0), x_values, initial=0.0)
            color = formula.get("color", "black")
            dash = formula.get("dash", "solid")
            width = _as_float(formula.get("width", 3.0), 3.0)
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=label,
                line={"color": color, "dash": dash, "width": width},
                customdata=np.column_stack((derivative_for_hover, second_for_hover)),
                hovertemplate="<b>%{fullData.name}</b><br>x=%{x:.6g}<br>y=%{y:.6g}<br>slope=%{customdata[0]:.6g}<br>curvature=%{customdata[1]:.6g}<extra></extra>",
            ))
            series_results.append({
                "id": formula.get("id"),
                "label": label,
                "expression": expression,
                "color": color,
                "y": y_values,
                "spline": spline or extrema_spline,
                "roots": roots,
                "extrema": extrema,
                "invalid_intervals": invalid_intervals,
                "threshold_crossings": threshold_crossings,
                "cumulative": cumulative,
            })

        if not series_results:
            message = "No valid formulas to plot"
            state.last_summary_text = message + ("\n" + "\n".join(error_lines) if error_lines else "")
            state.last_details_text = state.last_summary_text
            fig.update_layout(template="plotly_white", annotations=[{"text": message, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False}])
            return fig

        selected_id = state.analysis_formula or series_results[0]["id"]
        selected = next((item for item in series_results if item["id"] == selected_id), series_results[0])
        root_rows: list[str] = []
        extrema_rows: list[str] = []
        threshold_rows: list[str] = []
        invalid_rows: list[str] = []
        summary_rows: list[str] = []

        for series in series_results:
            y_values = series["y"]
            spline = series["spline"]
            finite_values = _filter_real_values(y_values)
            if finite_values.size:
                summary_rows.append(f"{series['label']}: min={finite_values.min():.4g}, max={finite_values.max():.4g}, mean={finite_values.mean():.4g}")
            derivative_values = spline.derivative()(x_values) if spline is not None else np.gradient(y_values, x_values)
            second_values = spline.derivative(2)(x_values) if spline is not None else np.gradient(derivative_values, x_values)
            if state.show_derivative:
                fig.add_trace(go.Scatter(x=x_values, y=derivative_values, mode="lines", name=f"Derivative: {series['label']}", line={"color": series["color"], "dash": "dash", "width": 2.0}))
            if state.show_second_derivative:
                fig.add_trace(go.Scatter(x=x_values, y=second_values, mode="lines", name=f"Second derivative: {series['label']}", line={"color": series["color"], "dash": "dot", "width": 1.8}))
            if state.show_antiderivative:
                fig.add_trace(go.Scatter(x=x_values, y=series["cumulative"], mode="lines", name=f"Integral: {series['label']}", line={"color": series["color"], "dash": "dashdot", "width": 2.0}))
            if state.show_root_markers and series["roots"]:
                fig.add_trace(go.Scatter(x=series["roots"], y=[0.0] * len(series["roots"]), mode="markers", name=f"{series['label']} roots", marker={"color": series["color"], "size": 10, "symbol": "x"}))
            if state.show_extrema_markers and series["extrema"]:
                fig.add_trace(go.Scatter(x=[item["x"] for item in series["extrema"]], y=[item["y"] for item in series["extrema"]], mode="markers", name=f"{series['label']} extrema", marker={"color": series["color"], "size": 11, "symbol": "diamond"}))
            for root in series["roots"]:
                root_rows.append(f"{series['label']}: x={root:.6g}")
            for item in series["extrema"]:
                extrema_rows.append(f"{series['label']}: {item['type']} at x={float(item['x']):.6g}, y={float(item['y']):.6g}")
            for crossing in series["threshold_crossings"]:
                threshold_rows.append(f"{series['label']}: x={crossing:.6g}, y={threshold_value:.6g}")
            if series["invalid_intervals"]:
                invalid_rows.append(f"{series['label']}: {_format_intervals(series['invalid_intervals'])}")

        intersection_rows: list[str] = []
        intersections: list[tuple[str, str, dict[str, float]]] = []
        for idx in range(len(series_results)):
            for jdx in range(idx + 1, len(series_results)):
                left = series_results[idx]
                right = series_results[jdx]
                for point in _find_intersections(x_values, left["y"], right["y"], limit=12):
                    intersections.append((left["label"], right["label"], point))
                    intersection_rows.append(f"{left['label']} / {right['label']}: x={point['x']:.6g}, y={point['y']:.6g}")
        if state.show_intersection_markers and intersections:
            fig.add_trace(go.Scatter(x=[item[2]["x"] for item in intersections], y=[item[2]["y"] for item in intersections], mode="markers", name="Intersections", marker={"color": "#444", "size": 10, "symbol": "cross"}))

        selected_spline = selected["spline"]
        if selected_spline is not None:
            y0 = float(selected_spline(analysis_x0))
            slope = float(selected_spline.derivative()(analysis_x0))
            second_at_x0 = float(selected_spline.derivative(2)(analysis_x0))
        else:
            y0 = float(np.interp(analysis_x0, x_values, selected["y"]))
            slope = float(np.interp(analysis_x0, x_values, np.gradient(selected["y"], x_values)))
            second_at_x0 = float(np.interp(analysis_x0, x_values, np.gradient(np.gradient(selected["y"], x_values), x_values)))
        if state.show_tangent:
            fig.add_trace(go.Scatter(x=x_values, y=y0 + slope * (x_values - analysis_x0), mode="lines", name=f"{selected['label']} tangent", line={"color": selected["color"], "dash": "dash", "width": 2}))
        if state.show_normal:
            if abs(slope) < 1e-12:
                y_span = float(np.ptp(_filter_real_values(selected["y"]))) or 2.0
                fig.add_trace(go.Scatter(x=[analysis_x0, analysis_x0], y=[y0 - y_span / 2, y0 + y_span / 2], mode="lines", name=f"{selected['label']} normal", line={"color": selected["color"], "dash": "dot", "width": 2}))
            else:
                fig.add_trace(go.Scatter(x=x_values, y=y0 - (x_values - analysis_x0) / slope, mode="lines", name=f"{selected['label']} normal", line={"color": selected["color"], "dash": "dot", "width": 2}))
        if state.show_tangent or state.show_normal or bool(state.analysis_formula):
            fig.add_trace(go.Scatter(x=[analysis_x0], y=[y0], mode="markers", name=f"{selected['label']} focus point", marker={"color": selected["color"], "size": 12, "symbol": "circle-open"}))

        interval_mask = (x_values >= interval_min) & (x_values <= interval_max)
        interval_x = x_values[interval_mask]
        interval_y = selected["y"][interval_mask]
        exact_integral = None
        if interval_x.size >= 2:
            if selected_spline is not None:
                anti = selected_spline.antiderivative()
                exact_integral = float(anti(interval_max) - anti(interval_min))
            else:
                exact_integral = float(trapezoid(interval_y, interval_x))
            if state.show_area_shading:
                fig.add_trace(go.Scatter(x=np.concatenate(([interval_x[0]], interval_x, [interval_x[-1]])), y=np.concatenate(([0.0], interval_y, [0.0])), fill="toself", fillcolor="rgba(31,119,180,0.18)", line={"color": "rgba(31,119,180,0.1)"}, name=f"{selected['label']} area", hoverinfo="skip"))
        selected_derivative = selected_spline.derivative()(x_values) if selected_spline is not None else np.gradient(selected["y"], x_values)
        selected_second = selected_spline.derivative(2)(x_values) if selected_spline is not None else np.gradient(selected_derivative, x_values)
        monotonic = _intervals_from_sign(x_values, selected_derivative, "increasing", "decreasing")
        concavity = _intervals_from_sign(x_values, selected_second, "concave up", "concave down")

        print(f"[debug][formula-plot] applying_dash_1d_layout size={FORMULA_GRAPH_WIDTH}x{FORMULA_GRAPH_HEIGHT}", flush=True)
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            margin={"l": 80, "r": 40, "t": 50, "b": 70},
            font={"size": 18, "family": "Arial"},
            showlegend=state.show_legend,
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=FORMULA_GRAPH_WIDTH,
            height=FORMULA_GRAPH_HEIGHT,
        )
        axis_style = {
            "showgrid": state.show_grid,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "mirror": "allticks",
            "ticks": "inside",
            "ticklen": 8,
            "tickwidth": 2,
            "tickcolor": "black",
            "showline": True,
            "linecolor": "black",
            "linewidth": 2,
        }
        fig.update_xaxes(title=state.x_axis_title or "x", **axis_style)
        fig.update_yaxes(title=state.y_axis_title or "f(x)", **axis_style)
        if state.show_monotonicity_regions:
            for start, end in monotonic["increasing"]:
                fig.add_shape(type="rect", x0=start, x1=end, y0=0.0, y1=0.06, yref="paper", fillcolor="rgba(46,204,113,0.18)", line_width=0, layer="below")
            for start, end in monotonic["decreasing"]:
                fig.add_shape(type="rect", x0=start, x1=end, y0=0.0, y1=0.06, yref="paper", fillcolor="rgba(231,76,60,0.18)", line_width=0, layer="below")
        if state.show_concavity_regions:
            for start, end in concavity["concave up"]:
                fig.add_shape(type="rect", x0=start, x1=end, y0=0.94, y1=1.0, yref="paper", fillcolor="rgba(52,152,219,0.16)", line_width=0, layer="below")
            for start, end in concavity["concave down"]:
                fig.add_shape(type="rect", x0=start, x1=end, y0=0.94, y1=1.0, yref="paper", fillcolor="rgba(155,89,182,0.16)", line_width=0, layer="below")
        fig.add_vrect(x0=interval_min, x1=interval_max, fillcolor="rgba(127,127,127,0.08)", line_width=0, layer="below")
        for series in series_results:
            for start, end in series["invalid_intervals"]:
                fig.add_vrect(x0=start, x1=end, fillcolor="rgba(176,0,32,0.10)", line_width=0, layer="below", annotation_text="invalid")
        if threshold_enabled:
            fig.add_hline(y=threshold_value, line_dash="dot", line_color="#b60021", annotation_text=f"threshold {threshold_value:.4g}")

        summary_lines = [
            f"Formula: {selected['label']}",
            f"Point: x0={analysis_x0:.6g}, y={y0:.6g}",
            f"Slope / Curvature: {slope:.6g} / {second_at_x0:.6g}",
            f"Interval: [{interval_min:.6g}, {interval_max:.6g}]",
            f"Exact integral: {exact_integral:.6g}" if exact_integral is not None else "Exact integral: unavailable",
            f"Counts: {len(selected['roots'])} root(s), {len(selected['extrema'])} turning point(s), {len(selected['threshold_crossings'])} threshold crossing(s)",
            *summary_rows,
        ]
        detail_lines = [
            "Roots: " + ("; ".join(root_rows) if root_rows else "none"),
            "Extrema: " + ("; ".join(extrema_rows) if extrema_rows else "none"),
            "Intersections: " + ("; ".join(intersection_rows) if intersection_rows else "none"),
            "Thresholds: " + ("; ".join(threshold_rows) if threshold_rows else "none"),
            "Invalid regions: " + ("; ".join(invalid_rows) if invalid_rows else "none"),
            "Monotonicity: increasing " + _format_intervals(monotonic["increasing"]) + "; decreasing " + _format_intervals(monotonic["decreasing"]),
            "Concavity: up " + _format_intervals(concavity["concave up"]) + "; down " + _format_intervals(concavity["concave down"]),
        ]
        state.last_summary_text = "\n".join(summary_lines)
        state.last_details_text = "\n".join(detail_lines)
        print(f"[debug][formula-plot] build_formula_figure traces={len(fig.data)} summary={state.last_summary_text!r}", flush=True)
        return fig
    except Exception as exc:
        print(f"[debug][formula-plot] build_formula_figure error={exc!r}", flush=True)
        state.last_summary_text = str(exc) or "Unable to render formula"
        state.last_details_text = state.last_summary_text
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            width=FORMULA_GRAPH_WIDTH,
            height=FORMULA_GRAPH_HEIGHT,
            annotations=[{"text": state.last_summary_text, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False}],
        )
        return fig


def _compute_formula_panel_2d_data(state: FormulaPlotState, notebook_vars: dict[str, Any] | None = None) -> dict[str, Any]:
    print("[debug][formula-plot] compute_2d_data start", flush=True)
    expression = (state.expression_2d or "").strip()
    if not expression:
        raise FormulaValidationError("Please enter a 2D formula")
    x_min, x_max = float(state.x_min), float(state.x_max)
    y_min, y_max = float(state.y_min), float(state.y_max)
    x_points = max(3, int(state.x_points_2d))
    y_points = max(3, int(state.y_points_2d))
    if x_min >= x_max or y_min >= y_max:
        raise FormulaValidationError("2D ranges require min < max")
    x_values = np.linspace(x_min, x_max, x_points)
    y_values = np.linspace(y_min, y_max, y_points)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    param_values = _full_param_values(state, notebook_vars)
    z_grid = evaluate_formula_2d(expression, x_grid, y_grid, param_values)
    finite_mask = np.isfinite(z_grid)
    finite_values = z_grid[finite_mask]
    probe_x = min(max(float(state.probe_x_2d), x_min), x_max)
    probe_y = min(max(float(state.probe_y_2d), y_min), y_max)
    x_index = int(np.abs(x_values - probe_x).argmin())
    y_index = int(np.abs(y_values - probe_y).argmin())
    probe_x = float(x_values[x_index])
    probe_y = float(y_values[y_index])
    probe_z = float(z_grid[y_index, x_index])
    z_min = state.z_min_2d
    z_max = state.z_max_2d
    if finite_values.size and state.auto_z_range_2d:
        z_min = float(finite_values.min())
        z_max = float(finite_values.max())
    elif z_min is not None and z_max is not None and z_min > z_max:
        z_min, z_max = z_max, z_min
    data = {
        "expression": expression,
        "label": (state.label_2d or expression).strip(),
        "x_values": x_values,
        "y_values": y_values,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "z_grid": z_grid,
        "finite_values": finite_values,
        "valid_fraction": 100.0 * np.count_nonzero(finite_mask) / z_grid.size if z_grid.size else 0.0,
        "invalid_cells": int(z_grid.size - np.count_nonzero(finite_mask)),
        "invalid_x": _find_invalid_intervals(x_values, np.any(~np.isfinite(z_grid), axis=0).astype(float)),
        "invalid_y": _find_invalid_intervals(y_values, np.any(~np.isfinite(z_grid), axis=1).astype(float)),
        "probe_x": probe_x,
        "probe_y": probe_y,
        "probe_z": probe_z,
        "x_index": x_index,
        "y_index": y_index,
        "x_slice_values": z_grid[y_index, :],
        "y_slice_values": z_grid[:, x_index],
        "z_min": z_min,
        "z_max": z_max,
        "param_values": _parameter_values(state, notebook_vars),
    }
    print(f"[debug][formula-plot] compute_2d_data done shape={z_grid.shape} probe=({probe_x}, {probe_y}, {probe_z})", flush=True)
    return data


def build_formula_figure_2d(state: FormulaPlotState, notebook_vars: dict[str, Any] | None = None) -> go.Figure:
    print(f"[debug][formula-plot] build_formula_figure_2d mode={state.display_mode_2d!r}", flush=True)
    try:
        data = _compute_formula_panel_2d_data(state, notebook_vars)
        include_probe = not (np.isclose(float(state.probe_x_2d), 0.0) and np.isclose(float(state.probe_y_2d), 0.0))
        if state.display_mode_2d == "heatmap":
            trace = go.Heatmap(x=data["x_values"], y=data["y_values"], z=data["z_grid"], colorscale=state.surface_colorscale, name=data["label"], zmin=data["z_min"], zmax=data["z_max"])
            probe_trace = go.Scatter(x=[data["probe_x"]], y=[data["probe_y"]], mode="markers", name="Probe", marker={"size": 12, "color": "#111827", "symbol": "x"})
            fig = go.Figure(data=[trace, probe_trace] if include_probe else [trace])
        elif state.display_mode_2d == "contour":
            trace = go.Contour(x=data["x_values"], y=data["y_values"], z=data["z_grid"], colorscale=state.surface_colorscale, name=data["label"], zmin=data["z_min"], zmax=data["z_max"], ncontours=max(3, int(state.contour_levels_2d)), contours={"showlabels": state.contour_style_2d == "lines", "coloring": "heatmap" if state.contour_style_2d == "filled" else "lines"})
            probe_trace = go.Scatter(x=[data["probe_x"]], y=[data["probe_y"]], mode="markers", name="Probe", marker={"size": 12, "color": "#111827", "symbol": "x"})
            fig = go.Figure(data=[trace, probe_trace] if include_probe else [trace])
        else:
            trace = go.Surface(x=data["x_grid"], y=data["y_grid"], z=data["z_grid"], colorscale=state.surface_colorscale, name=data["label"], cmin=data["z_min"], cmax=data["z_max"], colorbar={"title": state.z_axis_title})
            probe_trace = go.Scatter3d(x=[data["probe_x"]], y=[data["probe_y"]], z=[data["probe_z"]], mode="markers", name="Probe", marker={"size": 5, "color": "#111827", "symbol": "x"})
            fig = go.Figure(data=[trace, probe_trace] if include_probe else [trace])
        print(f"[debug][formula-plot] applying_dash_2d_layout size={FORMULA_GRAPH_WIDTH}x{FORMULA_GRAPH_2D_HEIGHT}", flush=True)
        fig.update_layout(
            template="plotly_white",
            margin={"l": 40, "r": 40, "t": 70, "b": 40},
            font={"size": 16, "family": "Arial"},
            paper_bgcolor="white",
            plot_bgcolor="white",
            title=f"{data['label']} ({state.display_mode_2d})",
            width=FORMULA_GRAPH_WIDTH,
            height=FORMULA_GRAPH_2D_HEIGHT,
        )
        if state.display_mode_2d == "surface":
            fig.update_layout(scene={"xaxis": {"title": state.x_axis_title}, "yaxis": {"title": state.y_axis_title}, "zaxis": {"title": state.z_axis_title}})
        else:
            fig.update_xaxes(title=state.x_axis_title, showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)")
            fig.update_yaxes(title=state.y_axis_title, showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)")
        state.last_summary_text = (
            f"2D Summary\nFormula: {data['label']}\n"
            f"Domain: x=[{state.x_min:.4g}, {state.x_max:.4g}], y=[{state.y_min:.4g}, {state.y_max:.4g}]\n"
            f"Sampling: {state.x_points_2d} x {state.y_points_2d}\n"
            f"Valid domain: {data['valid_fraction']:.1f}% finite\n"
            f"Probe: ({data['probe_x']:.4g}, {data['probe_y']:.4g}) -> {data['probe_z']:.6g}\n"
            f"Z range: {'Auto' if state.auto_z_range_2d else 'Manual'}"
        )
        if data["finite_values"].size:
            state.last_summary_text += f"\nz min / max: {data['finite_values'].min():.6g} / {data['finite_values'].max():.6g}\nz mean: {data['finite_values'].mean():.6g}"
        state.last_details_text = (
            f"Slices\nx-slice: y={data['probe_y']:.4g}, finite={100.0 * np.count_nonzero(np.isfinite(data['x_slice_values'])) / max(len(data['x_slice_values']), 1):.1f}%\n"
            f"y-slice: x={data['probe_x']:.4g}, finite={100.0 * np.count_nonzero(np.isfinite(data['y_slice_values'])) / max(len(data['y_slice_values']), 1):.1f}%\n"
            f"Invalid cells: {data['invalid_cells']} / {data['z_grid'].size}\n"
            f"Invalid x columns: {_format_intervals(data['invalid_x'])}\nInvalid y rows: {_format_intervals(data['invalid_y'])}"
        )
        print(f"[debug][formula-plot] build_formula_figure_2d traces={len(fig.data)}", flush=True)
        return fig
    except Exception as exc:
        print(f"[debug][formula-plot] build_formula_figure_2d error={exc!r}", flush=True)
        state.last_summary_text = str(exc) or "Unable to render 2D formula"
        state.last_details_text = state.last_summary_text
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            width=FORMULA_GRAPH_WIDTH,
            height=FORMULA_GRAPH_2D_HEIGHT,
            annotations=[{"text": state.last_summary_text, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False}],
        )
        return fig


def _empty_2d_slice_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 50, "b": 45},
        font={"size": 14, "family": "Arial"},
        height=FORMULA_SLICE_HEIGHT,
        annotations=[{"text": message, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 14, "color": "#6b7280"}}],
    )
    return fig


def _build_formula_2d_slice_figure(axis_title: str, coord_values: np.ndarray, z_values: np.ndarray, probe_value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=coord_values, y=z_values, mode="lines", line={"color": color, "width": 3}, connectgaps=False, name=title))
    finite_mask = np.isfinite(z_values)
    if np.count_nonzero(finite_mask):
        probe_index = int(np.abs(coord_values - probe_value).argmin())
        probe_z = z_values[probe_index]
        if np.isfinite(probe_z):
            fig.add_trace(go.Scatter(x=[coord_values[probe_index]], y=[probe_z], mode="markers", marker={"size": 10, "color": "#111827", "symbol": "x"}, name="Probe"))
    for start, end in _find_invalid_intervals(coord_values, z_values):
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(239,68,68,0.12)", line_width=0)
    print(f"[debug][formula-plot] applying_dash_slice_layout title={title!r} height={FORMULA_SLICE_HEIGHT}", flush=True)
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin={"l": 55, "r": 20, "t": 55, "b": 45},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 13, "family": "Arial"},
        showlegend=False,
        height=FORMULA_SLICE_HEIGHT,
    )
    fig.update_xaxes(title=axis_title, showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)")
    fig.update_yaxes(title="z", showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)")
    return fig


def build_formula_2d_slice_figures(state: FormulaPlotState, notebook_vars: dict[str, Any] | None = None) -> tuple[go.Figure, go.Figure]:
    print("[debug][formula-plot] build_formula_2d_slice_figures start", flush=True)
    try:
        data = _compute_formula_panel_2d_data(state, notebook_vars)
        x_slice = _build_formula_2d_slice_figure(state.x_axis_title, data["x_values"], data["x_slice_values"], data["probe_x"], f"x-slice at {state.y_axis_title}={data['probe_y']:.4g}", "#2563eb")
        y_slice = _build_formula_2d_slice_figure(state.y_axis_title, data["y_values"], data["y_slice_values"], data["probe_y"], f"y-slice at {state.x_axis_title}={data['probe_x']:.4g}", "#b60021")
        print("[debug][formula-plot] build_formula_2d_slice_figures done", flush=True)
        return x_slice, y_slice
    except Exception as exc:
        print(f"[debug][formula-plot] build_formula_2d_slice_figures error={exc!r}", flush=True)
        message = str(exc) or "Unable to render slices"
        return _empty_2d_slice_figure(message), _empty_2d_slice_figure(message)


def formula_samples_dataframe(state: FormulaPlotState, notebook_vars: dict[str, Any] | None = None) -> pd.DataFrame:
    print("[debug][formula-plot] formula_samples_dataframe start", flush=True)
    x_values = np.linspace(float(state.x_min), float(state.x_max), max(1, int(state.points)))
    data: dict[str, Any] = {"x": x_values}
    param_values = _full_param_values(state, notebook_vars)
    for formula in state.formulas:
        if not formula.get("visible", True):
            continue
        expression = (formula.get("expression") or "").strip()
        if not expression:
            continue
        label = (formula.get("label") or expression).strip()
        data[label] = evaluate_formula(expression, x_values, param_values)
    df = pd.DataFrame(data)
    print(f"[debug][formula-plot] formula_samples_dataframe done shape={df.shape}", flush=True)
    return df


def formula_surface_dataframe(state: FormulaPlotState, notebook_vars: dict[str, Any] | None = None) -> pd.DataFrame:
    print("[debug][formula-plot] formula_surface_dataframe start", flush=True)
    data = _compute_formula_panel_2d_data(state, notebook_vars)
    df = pd.DataFrame({"x": data["x_grid"].ravel(), "y": data["y_grid"].ravel(), "z": data["z_grid"].ravel()})
    print(f"[debug][formula-plot] formula_surface_dataframe done shape={df.shape}", flush=True)
    return df


class FormulaPlotTab(QWidget):
    def __init__(self, parent: QWidget | None = None, notebook_namespace_provider: Callable[[], dict[str, Any]] | None = None) -> None:
        super().__init__(parent)
        print("[debug][formula-plot-tab] init:start", flush=True)
        self.state = FormulaPlotState()
        self._notebook_namespace_provider = notebook_namespace_provider
        self._slider_specs: dict[str, tuple[QSlider, QLabel]] = {}
        self._param_value_labels: dict[str, QLabel] = {}
        self._suspend_updates = False
        self.setObjectName("FormulaPlotTab")
        self.setStyleSheet(FORMULA_TAB_STYLE)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.page_scroll = QScrollArea(self)
        self.page_scroll.setObjectName("FormulaPageScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setStyleSheet("QScrollArea { border: none; background:#21252b; }")
        outer_layout.addWidget(self.page_scroll)
        self.page_widget = QWidget(self.page_scroll)
        self.page_widget.setObjectName("FormulaPage")
        self.page_widget.setMinimumWidth(FORMULA_GRAPH_WIDTH + 390)
        root = QVBoxLayout(self.page_widget)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        self.page_scroll.setWidget(self.page_widget)
        print("[debug][formula-plot-tab] layout:page_scroll created widget_resizable=True", flush=True)

        self.status_label = QLabel("Ready", self.page_widget)
        self.status_label.setStyleSheet(FORMULA_STATUS_READY_STYLE)
        root.addWidget(self.status_label)

        self.top_controls_widget = QWidget(self.page_widget)
        self.top_controls_widget.setObjectName("FormulaTopControls")
        self.top_controls_widget.setMinimumWidth(FORMULA_GRAPH_WIDTH)
        self.controls_widget = self.top_controls_widget
        self.controls_layout = QVBoxLayout(self.top_controls_widget)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(10)
        root.addWidget(self.top_controls_widget, 0)
        print("[debug][formula-plot-tab] layout:top_controls created local_scroll=False", flush=True)

        self.main_content_widget = QWidget(self.page_widget)
        self.main_content_widget.setObjectName("FormulaMainContent")
        main_content_layout = QHBoxLayout(self.main_content_widget)
        main_content_layout.setContentsMargins(0, 0, 0, 0)
        main_content_layout.setSpacing(10)
        root.addWidget(self.main_content_widget, 1)

        self.graph_column_widget = QWidget(self.main_content_widget)
        self.graph_column_widget.setObjectName("FormulaGraphColumn")
        plot_column = QVBoxLayout(self.graph_column_widget)
        plot_column.setContentsMargins(0, 0, 0, 0)
        plot_column.setSpacing(10)
        main_content_layout.addWidget(self.graph_column_widget, 1)

        self.plot_view = PlotView(self.graph_column_widget)
        self.plot_view.setMinimumHeight(FORMULA_GRAPH_HEIGHT)
        print(f"[debug][formula-plot-tab] plot_view:min_height={FORMULA_GRAPH_HEIGHT}", flush=True)
        plot_column.addWidget(self.plot_view, 3)
        self.graph_options_widget = QWidget(self.graph_column_widget)
        self.graph_options_widget.setObjectName("FormulaGraphOptions")
        self.graph_options_layout = QVBoxLayout(self.graph_options_widget)
        self.graph_options_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_options_layout.setSpacing(10)
        plot_column.addWidget(self.graph_options_widget, 0)
        print("[debug][formula-plot-tab] layout:graph_options created", flush=True)
        self.slice_row = QWidget(self.graph_column_widget)
        slice_layout = QHBoxLayout(self.slice_row)
        slice_layout.setContentsMargins(0, 0, 0, 0)
        self.x_slice_view = PlotView(self.slice_row)
        self.y_slice_view = PlotView(self.slice_row)
        self.x_slice_view.setMinimumHeight(FORMULA_SLICE_HEIGHT)
        self.y_slice_view.setMinimumHeight(FORMULA_SLICE_HEIGHT)
        print(f"[debug][formula-plot-tab] slice_views:min_height={FORMULA_SLICE_HEIGHT}", flush=True)
        self.x_slice_view._toolbar.hide()
        self.y_slice_view._toolbar.hide()
        slice_layout.addWidget(self.x_slice_view)
        slice_layout.addWidget(self.y_slice_view)
        plot_column.addWidget(self.slice_row, 1)
        self.slice_row.hide()

        summary_row = QHBoxLayout()
        self.summary_browser = QTextBrowser(self.graph_column_widget)
        self.details_browser = QTextBrowser(self.graph_column_widget)
        _configure_expanding_text_browser(self.summary_browser)
        _configure_expanding_text_browser(self.details_browser)
        self.summary_browser.setMinimumHeight(120)
        self.details_browser.setMinimumHeight(120)
        summary_row.addWidget(self.summary_browser)
        summary_row.addWidget(self.details_browser)
        plot_column.addLayout(summary_row)

        self.settings_sidebar = QWidget(self.main_content_widget)
        self.settings_sidebar.setObjectName("FormulaSettingsSidebar")
        self.settings_sidebar.setMinimumWidth(360)
        self.settings_sidebar.setMaximumWidth(460)
        settings_sidebar_outer_layout = QVBoxLayout(self.settings_sidebar)
        settings_sidebar_outer_layout.setContentsMargins(0, 0, 0, 0)
        settings_sidebar_outer_layout.setSpacing(0)
        self.settings_sidebar_contents = QWidget(self.settings_sidebar)
        self.settings_sidebar_contents.setObjectName("FormulaSettingsSidebarContents")
        self.sidebar_layout = QVBoxLayout(self.settings_sidebar_contents)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(10)
        settings_sidebar_outer_layout.addWidget(self.settings_sidebar_contents)
        main_content_layout.addWidget(self.settings_sidebar, 0)
        print("[debug][formula-plot-tab] layout:main_content graph_and_sidebar created local_scroll=False", flush=True)

        self._build_controls()
        self._sync_all_controls_from_state()
        self.refresh_1d()
        print("[debug][formula-plot-tab] init:done", flush=True)

    def _notebook_vars(self) -> dict[str, Any]:
        print("[debug][formula-plot-tab] notebook_vars:start", flush=True)
        if self._notebook_namespace_provider is None:
            print("[debug][formula-plot-tab] notebook_vars:none", flush=True)
            return {}
        try:
            namespace = self._notebook_namespace_provider() or {}
        except Exception as exc:
            print(f"[debug][formula-plot-tab] notebook_vars:error error={exc!r}", flush=True)
            return {}
        numeric = {}
        for name, value in namespace.items():
            try:
                if np.asarray(value).shape == ():
                    numeric[name] = float(value)
            except Exception:
                continue
        print(f"[debug][formula-plot-tab] notebook_vars:done keys={sorted(numeric)}", flush=True)
        return numeric

    def _panel(self, title: str, parent: QWidget | None = None) -> QWidget:
        parent_widget = parent or self.controls_widget
        widget = QWidget(parent_widget)
        widget.setStyleSheet("QWidget { background:#282c34; border-radius:7px; }")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        label = QLabel(title, widget)
        label.setStyleSheet("font-weight:700; color:#d7dae0;")
        layout.addWidget(label)
        return widget

    def _build_controls(self) -> None:
        print("[debug][formula-plot-tab] build_controls:start", flush=True)
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("1D Formula", "1d")
        self.mode_combo.addItem("2D Surface", "2d")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.controls_layout.addWidget(self.mode_combo)

        self.one_d_panel = self._panel("1D Formulas")
        self.one_d_layout = self.one_d_panel.layout()
        self.one_d_row_widget = QWidget(self.one_d_panel)
        self.one_d_row_layout = QHBoxLayout(self.one_d_row_widget)
        self.one_d_row_layout.setContentsMargins(0, 0, 0, 0)
        self.one_d_row_layout.setSpacing(10)
        self.example_combo = QComboBox(self.one_d_panel)
        for label, value in FORMULA_EXAMPLES:
            self.example_combo.addItem(label, value)
        self.example_combo.currentIndexChanged.connect(self._on_example_changed)
        self.example_combo.setMinimumWidth(190)
        self.one_d_row_layout.addWidget(self.example_combo, 1)
        self.formula_rows_container = QWidget(self.one_d_panel)
        self.formula_rows_layout = QVBoxLayout(self.formula_rows_container)
        self.formula_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.formula_rows_layout.setSpacing(10)
        self.one_d_row_layout.addWidget(self.formula_rows_container, 6)
        self.add_formula_btn = QPushButton("Add Formula", self.one_d_panel)
        self.add_formula_btn.setStyleSheet(FORMULA_BUTTON_STYLE)
        self.add_formula_btn.clicked.connect(self._add_formula_row)
        self.one_d_row_layout.addWidget(self.add_formula_btn, 1)
        self.reset_params_btn = QPushButton("Reset Params", self.one_d_panel)
        self.reset_params_btn.setStyleSheet(FORMULA_BUTTON_STYLE)
        self.reset_params_btn.clicked.connect(self._reset_params)
        self.one_d_row_layout.addWidget(self.reset_params_btn, 1)
        self.one_d_layout.addWidget(self.one_d_row_widget)
        self.controls_layout.addWidget(self.one_d_panel)
        print("[debug][formula-plot-tab] build_controls:1d_single_row created", flush=True)

        self.range_panel = self._panel("Axes and Sampling")
        self.range_row_widget = QWidget(self.range_panel)
        self.range_row_layout = QHBoxLayout(self.range_row_widget)
        self.range_row_layout.setContentsMargins(0, 0, 0, 0)
        self.range_row_layout.setSpacing(10)
        self.x_min_edit = QLineEdit(str(self.state.x_min), self.range_panel)
        self.x_max_edit = QLineEdit(str(self.state.x_max), self.range_panel)
        self.points_spin = QSpinBox(self.range_panel)
        self.points_spin.setRange(10, 5000)
        self.points_spin.setValue(self.state.points)
        self.x_title_edit = QLineEdit(self.state.x_axis_title, self.range_panel)
        self.y_title_edit = QLineEdit(self.state.y_axis_title, self.range_panel)
        for widget in (self.x_min_edit, self.x_max_edit, self.x_title_edit, self.y_title_edit):
            widget.editingFinished.connect(self._on_1d_settings_changed)
        self.points_spin.valueChanged.connect(self._on_1d_settings_changed)
        range_fields = [
            ("X Min", self.x_min_edit),
            ("X Max", self.x_max_edit),
            ("Points", self.points_spin),
            ("X-Axis Title", self.x_title_edit),
            ("Y-Axis Title", self.y_title_edit),
        ]
        for label_text, editor in range_fields:
            cell = QWidget(self.range_panel)
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(5)
            label = QLabel(label_text, cell)
            label.setStyleSheet("font-weight:700; color:#aab2c0;")
            cell_layout.addWidget(label)
            cell_layout.addWidget(editor, 1)
            self.range_row_layout.addWidget(cell, 1)
        self.range_panel.layout().addWidget(self.range_row_widget)
        self.controls_layout.addWidget(self.range_panel)
        print("[debug][formula-plot-tab] build_controls:range_single_row fields=5", flush=True)

        self.display_panel = self._panel("Figure", self.graph_options_widget)
        self.display_checks: dict[str, QCheckBox] = {}
        grid = QGridLayout()
        options = [
            ("show_legend", "Legend"),
            ("show_grid", "Grid"),
            ("show_derivative", "Derivative"),
            ("show_second_derivative", "2nd Derivative"),
            ("show_antiderivative", "Integral Curve"),
            ("show_tangent", "Tangent"),
            ("show_normal", "Normal"),
            ("show_root_markers", "Root Markers"),
            ("show_extrema_markers", "Extrema Markers"),
            ("show_intersection_markers", "Intersections"),
            ("show_area_shading", "Area Shading"),
            ("show_monotonicity_regions", "Monotonicity"),
            ("show_concavity_regions", "Concavity"),
        ]
        for idx, (attr, label) in enumerate(options):
            check = QCheckBox(label, self.display_panel)
            check.setChecked(bool(getattr(self.state, attr)))
            check.stateChanged.connect(self._on_display_options_changed)
            self.display_checks[attr] = check
            grid.addWidget(check, idx // 2, idx % 2)
        self.display_panel.layout().addItem(grid)
        self.graph_options_layout.addWidget(self.display_panel)

        self.analysis_panel = self._panel("Analysis Tools", self.settings_sidebar_contents)
        analysis_form = QFormLayout()
        self.analysis_formula_combo = QComboBox(self.analysis_panel)
        self.analysis_formula_combo.currentIndexChanged.connect(self._on_analysis_changed)
        self.analysis_x0_edit = QLineEdit(str(self.state.analysis_x0), self.analysis_panel)
        self.threshold_edit = QLineEdit("", self.analysis_panel)
        self.interval_min_edit = QLineEdit(str(self.state.interval_min), self.analysis_panel)
        self.interval_max_edit = QLineEdit(str(self.state.interval_max), self.analysis_panel)
        self.click_mode_combo = QComboBox(self.analysis_panel)
        self.click_mode_combo.addItem("Select tangent point", "focus")
        self.click_mode_combo.addItem("Select integral interval", "interval")
        for widget in (self.analysis_x0_edit, self.threshold_edit, self.interval_min_edit, self.interval_max_edit):
            widget.editingFinished.connect(self._on_analysis_changed)
        self.click_mode_combo.currentIndexChanged.connect(self._on_analysis_changed)
        analysis_form.addRow("Focus Formula", self.analysis_formula_combo)
        analysis_form.addRow("X0", self.analysis_x0_edit)
        analysis_form.addRow("Threshold", self.threshold_edit)
        analysis_form.addRow("Interval Min", self.interval_min_edit)
        analysis_form.addRow("Interval Max", self.interval_max_edit)
        analysis_form.addRow("Click Action", self.click_mode_combo)
        self.analysis_panel.layout().addItem(analysis_form)
        self.sidebar_layout.addWidget(self.analysis_panel)

        self.two_d_panel = self._panel("2D Surface")
        two_form = QFormLayout()
        self.preset_2d_combo = QComboBox(self.two_d_panel)
        for label, value in PRESET_2D_OPTIONS:
            self.preset_2d_combo.addItem(label, value)
        self.preset_2d_combo.currentIndexChanged.connect(self._on_2d_preset_changed)
        self.display_mode_2d_combo = QComboBox(self.two_d_panel)
        self.display_mode_2d_combo.addItem("Surface", "surface")
        self.display_mode_2d_combo.addItem("Heatmap", "heatmap")
        self.display_mode_2d_combo.addItem("Contour", "contour")
        self.display_mode_2d_combo.currentIndexChanged.connect(self._on_display_mode_2d_changed)
        self.expression_2d_edit = QLineEdit(self.state.expression_2d, self.two_d_panel)
        self.label_2d_edit = QLineEdit(self.state.label_2d, self.two_d_panel)
        self.y_min_edit = QLineEdit(str(self.state.y_min), self.two_d_panel)
        self.y_max_edit = QLineEdit(str(self.state.y_max), self.two_d_panel)
        self.x_points_2d_spin = QSpinBox(self.two_d_panel)
        self.x_points_2d_spin.setRange(3, 500)
        self.x_points_2d_spin.setValue(self.state.x_points_2d)
        self.y_points_2d_spin = QSpinBox(self.two_d_panel)
        self.y_points_2d_spin.setRange(3, 500)
        self.y_points_2d_spin.setValue(self.state.y_points_2d)
        self.z_title_edit = QLineEdit(self.state.z_axis_title, self.two_d_panel)
        self.colorscale_combo = QComboBox(self.two_d_panel)
        for colorscale in COLOR_SCALES:
            self.colorscale_combo.addItem(colorscale, colorscale)
        self.contour_levels_spin = QSpinBox(self.two_d_panel)
        self.contour_levels_spin.setRange(3, 50)
        self.contour_levels_spin.setValue(self.state.contour_levels_2d)
        self.contour_style_combo = QComboBox(self.two_d_panel)
        self.contour_style_combo.addItem("Filled", "filled")
        self.contour_style_combo.addItem("Lines Only", "lines")
        self.auto_z_check = QCheckBox("Auto Z Range", self.two_d_panel)
        self.auto_z_check.setChecked(True)
        self.z_min_edit = QLineEdit("", self.two_d_panel)
        self.z_max_edit = QLineEdit("", self.two_d_panel)
        self.probe_x_edit = QLineEdit(str(self.state.probe_x_2d), self.two_d_panel)
        self.probe_y_edit = QLineEdit(str(self.state.probe_y_2d), self.two_d_panel)
        for widget in (self.expression_2d_edit, self.label_2d_edit, self.y_min_edit, self.y_max_edit, self.z_title_edit, self.z_min_edit, self.z_max_edit, self.probe_x_edit, self.probe_y_edit):
            widget.editingFinished.connect(self._on_2d_settings_changed)
        for widget in (self.x_points_2d_spin, self.y_points_2d_spin, self.contour_levels_spin):
            widget.valueChanged.connect(self._on_2d_settings_changed)
        self.colorscale_combo.currentIndexChanged.connect(self._on_2d_settings_changed)
        self.contour_style_combo.currentIndexChanged.connect(self._on_2d_settings_changed)
        self.auto_z_check.stateChanged.connect(self._on_2d_settings_changed)
        two_form.addRow("2D Preset", self.preset_2d_combo)
        two_form.addRow("2D Display", self.display_mode_2d_combo)
        two_form.addRow("2D Formula", self.expression_2d_edit)
        two_form.addRow("Legend", self.label_2d_edit)
        two_form.addRow("Y Min", self.y_min_edit)
        two_form.addRow("Y Max", self.y_max_edit)
        two_form.addRow("X Points", self.x_points_2d_spin)
        two_form.addRow("Y Points", self.y_points_2d_spin)
        two_form.addRow("Z Title", self.z_title_edit)
        two_form.addRow("Colorscale", self.colorscale_combo)
        two_form.addRow("Contour Levels", self.contour_levels_spin)
        two_form.addRow("Contour Style", self.contour_style_combo)
        two_form.addRow("", self.auto_z_check)
        two_form.addRow("Z Min", self.z_min_edit)
        two_form.addRow("Z Max", self.z_max_edit)
        two_form.addRow("Probe X", self.probe_x_edit)
        two_form.addRow("Probe Y", self.probe_y_edit)
        self.two_d_panel.layout().addItem(two_form)
        self.controls_layout.addWidget(self.two_d_panel)

        self.param_panel = self._panel("Parameters", self.settings_sidebar_contents)
        self.param_rows_container = QWidget(self.param_panel)
        self.param_rows_layout = QVBoxLayout(self.param_rows_container)
        self.param_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.param_panel.layout().addWidget(self.param_rows_container)
        self.sidebar_layout.addWidget(self.param_panel)

        self.plot_1d_btn = QPushButton("Plot 1D", self)
        self.plot_1d_btn.setStyleSheet(FORMULA_BUTTON_STYLE)
        self.plot_1d_btn.clicked.connect(self.refresh_1d)
        self.plot_2d_btn = QPushButton("Plot 2D", self)
        self.plot_2d_btn.setStyleSheet(FORMULA_BUTTON_STYLE)
        self.plot_2d_btn.clicked.connect(self.refresh_2d)
        self.export_csv_btn = QPushButton("Export CSV", self)
        self.export_csv_btn.setStyleSheet(FORMULA_BUTTON_STYLE)
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.range_row_layout.addWidget(self.plot_1d_btn, 1)
        self.range_row_layout.addWidget(self.plot_2d_btn, 1)
        self.range_row_layout.addWidget(self.export_csv_btn, 1)
        print(f"[debug][formula-plot-tab] build_controls:range_buttons_attached count={self.range_row_layout.count()}", flush=True)
        self.sidebar_layout.addStretch(1)
        print("[debug][formula-plot-tab] build_controls:done", flush=True)

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)  # type: ignore[arg-type]

    def _sync_formula_rows(self) -> None:
        print(f"[debug][formula-plot-tab] sync_formula_rows count={len(self.state.formulas)}", flush=True)
        self._clear_layout(self.formula_rows_layout)
        for row_index, formula in enumerate(self.state.formulas):
            row_widget = QWidget(self.formula_rows_container)
            grid = QGridLayout(row_widget)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(0)
            expr = QLineEdit(formula.get("expression", ""), row_widget)
            if row_index == 0:
                self.expression_edit = expr
            label = QLineEdit(formula.get("label", ""), row_widget)
            color = QComboBox(row_widget)
            for item_label, value in TRACE_COLORS:
                color.addItem(item_label, value)
            color.setCurrentIndex(max(0, color.findData(formula.get("color", "black"))))
            dash = QComboBox(row_widget)
            for item_label, value in TRACE_DASHES:
                dash.addItem(item_label, value)
            dash.setCurrentIndex(max(0, dash.findData(formula.get("dash", "solid"))))
            width = QSpinBox(row_widget)
            width.setRange(1, 8)
            width.setValue(int(round(_as_float(formula.get("width", 3.0), 3.0))))
            visible = QCheckBox("Show", row_widget)
            visible.setChecked(bool(formula.get("visible", True)))
            remove = QPushButton("Remove", row_widget)
            remove.setStyleSheet(FORMULA_BUTTON_STYLE)
            row_id = str(formula.get("id"))
            for widget in (expr, label):
                widget.editingFinished.connect(lambda rid=row_id, e=expr, l=label, c=color, d=dash, w=width, v=visible: self._update_formula_row(rid, e.text(), l.text(), c.currentData(), d.currentData(), w.value(), v.isChecked()))
            color.currentIndexChanged.connect(lambda _=0, rid=row_id, e=expr, l=label, c=color, d=dash, w=width, v=visible: self._update_formula_row(rid, e.text(), l.text(), c.currentData(), d.currentData(), w.value(), v.isChecked()))
            dash.currentIndexChanged.connect(lambda _=0, rid=row_id, e=expr, l=label, c=color, d=dash, w=width, v=visible: self._update_formula_row(rid, e.text(), l.text(), c.currentData(), d.currentData(), w.value(), v.isChecked()))
            width.valueChanged.connect(lambda _=0, rid=row_id, e=expr, l=label, c=color, d=dash, w=width, v=visible: self._update_formula_row(rid, e.text(), l.text(), c.currentData(), d.currentData(), w.value(), v.isChecked()))
            visible.stateChanged.connect(lambda _=0, rid=row_id, e=expr, l=label, c=color, d=dash, w=width, v=visible: self._update_formula_row(rid, e.text(), l.text(), c.currentData(), d.currentData(), w.value(), v.isChecked()))
            remove.clicked.connect(lambda _=False, rid=row_id: self._remove_formula_row(rid))
            row_items = [
                ("Formula", expr),
                ("Legend", label),
                ("Color", color),
                ("Line", dash),
                ("Width", width),
                ("Visible", visible),
                ("", remove),
            ]
            for column, (label_text, editor) in enumerate(row_items):
                cell = QWidget(row_widget)
                cell_layout = QHBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(4)
                if label_text:
                    header = QLabel(label_text, cell)
                    header.setStyleSheet("font-weight:700; color:#aab2c0;")
                    cell_layout.addWidget(header)
                cell_layout.addWidget(editor, 1)
                grid.addWidget(cell, 0, column)
            grid.setColumnStretch(0, 3)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(2, 1)
            grid.setColumnStretch(3, 1)
            self.formula_rows_layout.addWidget(row_widget)
        print("[debug][formula-plot-tab] sync_formula_rows stacked_rows=True", flush=True)
        self._sync_analysis_formula_combo()

    def _sync_analysis_formula_combo(self) -> None:
        current = self.state.analysis_formula
        self.analysis_formula_combo.blockSignals(True)
        self.analysis_formula_combo.clear()
        for formula in self.state.formulas:
            label = formula.get("label") or formula.get("expression") or "Formula"
            self.analysis_formula_combo.addItem(str(label), formula.get("id"))
        if current:
            index = self.analysis_formula_combo.findData(current)
            if index >= 0:
                self.analysis_formula_combo.setCurrentIndex(index)
        self.analysis_formula_combo.blockSignals(False)

    def _sync_preset_combo(self) -> None:
        print(f"[debug][formula-plot-tab] sync_preset_combo preset={self.state.preset_2d!r}", flush=True)
        index = self.preset_2d_combo.findData(self.state.preset_2d)
        if index >= 0 and index != self.preset_2d_combo.currentIndex():
            self.preset_2d_combo.blockSignals(True)
            self.preset_2d_combo.setCurrentIndex(index)
            self.preset_2d_combo.blockSignals(False)

    def _sync_display_mode_combo(self) -> None:
        print(f"[debug][formula-plot-tab] sync_display_mode_combo mode={self.state.display_mode_2d!r}", flush=True)
        index = self.display_mode_2d_combo.findData(self.state.display_mode_2d)
        if index >= 0 and index != self.display_mode_2d_combo.currentIndex():
            self.display_mode_2d_combo.blockSignals(True)
            self.display_mode_2d_combo.setCurrentIndex(index)
            self.display_mode_2d_combo.blockSignals(False)

    def _sync_all_controls_from_state(self) -> None:
        print("[debug][formula-plot-tab] sync_all_controls_from_state:start", flush=True)
        self._suspend_updates = True
        self.mode_combo.setCurrentIndex(0 if self.state.panel_type == "1d" else 1)
        self.x_min_edit.setText(str(self.state.x_min))
        self.x_max_edit.setText(str(self.state.x_max))
        self.points_spin.setValue(int(self.state.points))
        self.x_title_edit.setText(self.state.x_axis_title)
        self.y_title_edit.setText(self.state.y_axis_title)
        self.analysis_x0_edit.setText(str(self.state.analysis_x0))
        self.threshold_edit.setText("" if self.state.threshold_value is None else str(self.state.threshold_value))
        self.interval_min_edit.setText(str(self.state.interval_min))
        self.interval_max_edit.setText(str(self.state.interval_max))
        for attr, check in self.display_checks.items():
            check.setChecked(bool(getattr(self.state, attr)))
        self._sync_formula_rows()
        self._sync_preset_combo()
        self._sync_display_mode_combo()
        self.expression_2d_edit.setText(self.state.expression_2d)
        self.label_2d_edit.setText(self.state.label_2d)
        self.y_min_edit.setText(str(self.state.y_min))
        self.y_max_edit.setText(str(self.state.y_max))
        self.x_points_2d_spin.setValue(int(self.state.x_points_2d))
        self.y_points_2d_spin.setValue(int(self.state.y_points_2d))
        self.z_title_edit.setText(self.state.z_axis_title)
        self.colorscale_combo.setCurrentIndex(max(0, self.colorscale_combo.findData(self.state.surface_colorscale)))
        self.contour_levels_spin.setValue(int(self.state.contour_levels_2d))
        self.contour_style_combo.setCurrentIndex(max(0, self.contour_style_combo.findData(self.state.contour_style_2d)))
        self.auto_z_check.setChecked(bool(self.state.auto_z_range_2d))
        self.z_min_edit.setText("" if self.state.z_min_2d is None else str(self.state.z_min_2d))
        self.z_max_edit.setText("" if self.state.z_max_2d is None else str(self.state.z_max_2d))
        self.probe_x_edit.setText(str(self.state.probe_x_2d))
        self.probe_y_edit.setText(str(self.state.probe_y_2d))
        self._suspend_updates = False
        self.refresh_param_controls(self.state.panel_type)
        print("[debug][formula-plot-tab] sync_all_controls_from_state:done", flush=True)

    def _set_status(self, text: str, style: str) -> None:
        print(f"[debug][formula-plot-tab] set_status text={text!r} style={style!r}", flush=True)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)

    def _render_current_plot(self) -> None:
        print(f"[debug][formula-plot-tab] render_current_plot panel_type={self.state.panel_type!r}", flush=True)
        if self._suspend_updates:
            return
        try:
            nb_vars = self._notebook_vars()
            if self.state.panel_type == "2d":
                figure = build_formula_figure_2d(self.state, nb_vars)
                self.plot_view.set_figure(figure)
                x_slice, y_slice = build_formula_2d_slice_figures(self.state, nb_vars)
                self.x_slice_view.set_figure(x_slice)
                self.y_slice_view.set_figure(y_slice)
                self.slice_row.show()
                self._set_status("Updated 2D plot", FORMULA_STATUS_UPDATED_STYLE)
            else:
                figure = build_formula_figure(self.state, nb_vars)
                self.plot_view.set_figure(figure)
                self.slice_row.hide()
                self._set_status("Updated 1D plot", FORMULA_STATUS_UPDATED_STYLE)
            _set_expanding_browser_html(self.summary_browser, _analysis_table_html("Summary", self.state.last_summary_text))
            _set_expanding_browser_html(self.details_browser, _analysis_table_html("Details", self.state.last_details_text))
        except Exception as exc:
            print(f"[debug][formula-plot-tab] render_current_plot error={exc!r}", flush=True)
            prefix = "2D" if self.state.panel_type == "2d" else "1D"
            self._set_status(f"{prefix} plot error: {exc}", FORMULA_STATUS_ERROR_STYLE)

    def _on_mode_changed(self, index: int) -> None:
        mode = self.mode_combo.itemData(index)
        print(f"[debug][formula-plot-tab] on_mode_changed index={index} mode={mode!r}", flush=True)
        if mode == "2d":
            self.refresh_2d()
        else:
            self.refresh_1d()

    def _on_example_changed(self, index: int) -> None:
        if self._suspend_updates:
            return
        expression = str(self.example_combo.itemData(index) or "sin(x)")
        print(f"[debug][formula-plot-tab] on_example_changed index={index} expression={expression!r}", flush=True)
        self.state.example_formula = expression
        if self.state.formulas:
            old_expression = self.state.formulas[0].get("expression")
            old_label = self.state.formulas[0].get("label")
            self.state.formulas[0]["expression"] = expression
            if not old_label or old_label == old_expression:
                self.state.formulas[0]["label"] = expression
        _apply_formula_preset_defaults(self.state, expression)
        self._sync_formula_rows()
        self.refresh_param_controls("1d")
        self._render_current_plot()

    def _add_formula_row(self) -> None:
        expression = str(self.example_combo.currentData() or self.state.example_formula or "sin(x)")
        print(f"[debug][formula-plot-tab] add_formula_row expression={expression!r}", flush=True)
        self.state.formulas.append(_default_formula_row(expression))
        _apply_formula_preset_defaults(self.state, expression)
        self._sync_formula_rows()
        self.refresh_param_controls("1d")
        self.refresh_1d()

    def _remove_formula_row(self, row_id: str) -> None:
        print(f"[debug][formula-plot-tab] remove_formula_row row_id={row_id!r}", flush=True)
        self.state.formulas = [formula for formula in self.state.formulas if formula.get("id") != row_id]
        if not self.state.formulas:
            self.state.formulas = [_default_formula_row(self.state.example_formula, "formula_1")]
        _sync_panel_params(self.state)
        self._sync_formula_rows()
        self.refresh_param_controls("1d")
        self.refresh_1d()

    def _update_formula_row(self, row_id: str, expression: str, label: str, color: str, dash: str, width: int, visible: bool) -> None:
        if self._suspend_updates:
            return
        print(f"[debug][formula-plot-tab] update_formula_row row_id={row_id!r} expression={expression!r} label={label!r} visible={visible}", flush=True)
        should_resync_rows = False
        for formula in self.state.formulas:
            if formula.get("id") == row_id:
                previous_expression = str(formula.get("expression") or "")
                row_label = expression if _label_should_follow_expression(label, previous_expression) else label
                formula.update({"expression": expression, "label": label or expression, "color": color, "dash": dash, "width": float(width), "visible": visible})
                formula["label"] = row_label
                should_resync_rows = row_label != label
                break
        _sync_panel_params(self.state)
        if should_resync_rows:
            self._sync_formula_rows()
        else:
            self._sync_analysis_formula_combo()
        self.refresh_param_controls("1d")
        self._render_current_plot()

    def _on_1d_settings_changed(self) -> None:
        if self._suspend_updates:
            return
        print("[debug][formula-plot-tab] on_1d_settings_changed", flush=True)
        self.state.x_min = _as_float(self.x_min_edit.text(), self.state.x_min)
        self.state.x_max = _as_float(self.x_max_edit.text(), self.state.x_max)
        self.state.points = int(self.points_spin.value())
        self.state.x_axis_title = self.x_title_edit.text().strip() or "x"
        self.state.y_axis_title = self.y_title_edit.text().strip() or "f(x)"
        self._render_current_plot()

    def _on_display_options_changed(self) -> None:
        if self._suspend_updates:
            return
        for attr, check in self.display_checks.items():
            setattr(self.state, attr, check.isChecked())
        print(f"[debug][formula-plot-tab] on_display_options_changed options={ {attr: getattr(self.state, attr) for attr in self.display_checks} }", flush=True)
        self._render_current_plot()

    def _on_analysis_changed(self) -> None:
        if self._suspend_updates:
            return
        self.state.analysis_formula = str(self.analysis_formula_combo.currentData() or "")
        self.state.analysis_x0 = _as_float(self.analysis_x0_edit.text(), self.state.analysis_x0)
        threshold_text = self.threshold_edit.text().strip()
        self.state.threshold_value = None if threshold_text == "" else _as_float(threshold_text, 0.0)
        self.state.interval_min = _as_float(self.interval_min_edit.text(), self.state.interval_min)
        self.state.interval_max = _as_float(self.interval_max_edit.text(), self.state.interval_max)
        self.state.click_mode = str(self.click_mode_combo.currentData() or "focus")
        print(f"[debug][formula-plot-tab] on_analysis_changed analysis_formula={self.state.analysis_formula!r} x0={self.state.analysis_x0} threshold={self.state.threshold_value}", flush=True)
        self._render_current_plot()

    def _on_2d_preset_changed(self, index: int) -> None:
        if self._suspend_updates:
            return
        preset_name = self.preset_2d_combo.itemData(index)
        print(f"[debug][formula-plot-tab] on_2d_preset_changed index={index} preset={preset_name!r}", flush=True)
        if not preset_name:
            return
        self.state.panel_type = "2d"
        apply_2d_preset(self.state, str(preset_name))
        self._sync_all_controls_from_state()
        self.refresh_2d()

    def _on_display_mode_2d_changed(self, index: int) -> None:
        mode = self.display_mode_2d_combo.itemData(index)
        print(f"[debug][formula-plot-tab] on_display_mode_2d_changed index={index} mode={mode!r}", flush=True)
        if not mode:
            return
        self.state.display_mode_2d = str(mode)
        self.state.panel_type = "2d"
        self._render_current_plot()

    def _on_2d_settings_changed(self) -> None:
        if self._suspend_updates:
            return
        print("[debug][formula-plot-tab] on_2d_settings_changed", flush=True)
        previous_expression = self.state.expression_2d
        expression = self.expression_2d_edit.text().strip() or "sin(x)*cos(y)"
        label_text = self.label_2d_edit.text().strip()
        self.state.expression_2d = expression
        self.state.label_2d = expression if _label_should_follow_expression(label_text, previous_expression) else label_text
        if self.state.label_2d != label_text:
            self.label_2d_edit.blockSignals(True)
            self.label_2d_edit.setText(self.state.label_2d)
            self.label_2d_edit.blockSignals(False)
        self.state.x_min = _as_float(self.x_min_edit.text(), self.state.x_min)
        self.state.x_max = _as_float(self.x_max_edit.text(), self.state.x_max)
        self.state.y_min = _as_float(self.y_min_edit.text(), self.state.y_min)
        self.state.y_max = _as_float(self.y_max_edit.text(), self.state.y_max)
        self.state.x_points_2d = int(self.x_points_2d_spin.value())
        self.state.y_points_2d = int(self.y_points_2d_spin.value())
        self.state.x_axis_title = self.x_title_edit.text().strip() or "x"
        self.state.y_axis_title = self.y_title_edit.text().strip() or "y"
        self.state.z_axis_title = self.z_title_edit.text().strip() or "f(x,y)"
        self.state.surface_colorscale = str(self.colorscale_combo.currentData() or "Viridis")
        self.state.contour_levels_2d = int(self.contour_levels_spin.value())
        self.state.contour_style_2d = str(self.contour_style_combo.currentData() or "filled")
        self.state.auto_z_range_2d = self.auto_z_check.isChecked()
        self.state.z_min_2d = None if not self.z_min_edit.text().strip() else _as_float(self.z_min_edit.text(), 0.0)
        self.state.z_max_2d = None if not self.z_max_edit.text().strip() else _as_float(self.z_max_edit.text(), 1.0)
        self.state.probe_x_2d = _as_float(self.probe_x_edit.text(), self.state.probe_x_2d)
        self.state.probe_y_2d = _as_float(self.probe_y_edit.text(), self.state.probe_y_2d)
        _sync_panel_params(self.state)
        self.refresh_param_controls("2d")
        self._render_current_plot()

    def _reset_params(self) -> None:
        print("[debug][formula-plot-tab] reset_params:start", flush=True)
        if self.state.panel_type == "2d":
            apply_2d_preset(self.state, self.state.preset_2d)
        else:
            _sync_panel_params(self.state)
            _apply_formula_preset_defaults(self.state, self.state.example_formula)
        self.refresh_param_controls(self.state.panel_type)
        self._render_current_plot()
        print("[debug][formula-plot-tab] reset_params:done", flush=True)

    def _update_param(self, name: str, slider_value: int, label: QLabel) -> None:
        spec = self.state.params.get(name)
        if spec is None:
            spec = parameter_preset_for_name(name)
            self.state.params[name] = spec
        step = max(abs(float(spec.get("step", 0.1))), 1e-9)
        value = round(slider_value * step, 6)
        print(f"[debug][formula-plot-tab] update_param name={name!r} slider_value={slider_value} value={value}", flush=True)
        spec["value"] = value
        label.setText(f"{value:.4g}")

    def _on_param_slider(self, name: str, slider_value: int, label: QLabel) -> None:
        self._update_param(name, slider_value, label)
        self._render_current_plot()

    def _toggle_param_notebook(self, name: str, status: QLabel) -> None:
        spec = self.state.params.setdefault(name, parameter_preset_for_name(name))
        spec["use_notebook"] = not bool(spec.get("use_notebook", False))
        nb_vars = self._notebook_vars()
        if spec["use_notebook"] and name in nb_vars:
            spec["value"] = float(nb_vars[name])
            status.setText(f"NB={float(nb_vars[name]):.4g}")
        elif spec["use_notebook"]:
            status.setText("NB missing")
        else:
            status.setText("local")
        print(f"[debug][formula-plot-tab] toggle_param_notebook name={name!r} use_notebook={spec['use_notebook']} status={status.text()!r}", flush=True)
        self._render_current_plot()

    def _on_param_field_changed(self, name: str, value_edit: QLineEdit, min_edit: QLineEdit, max_edit: QLineEdit, step_edit: QLineEdit, label: QLabel, slider: QSlider) -> None:
        spec = self.state.params.setdefault(name, parameter_preset_for_name(name))
        spec["value"] = _as_float(value_edit.text(), float(spec.get("value", 1.0)))
        spec["min"] = _as_float(min_edit.text(), float(spec.get("min", -10.0)))
        spec["max"] = _as_float(max_edit.text(), float(spec.get("max", 10.0)))
        spec["step"] = max(abs(_as_float(step_edit.text(), float(spec.get("step", 0.1)))), 1e-9)
        if spec["min"] >= spec["max"]:
            spec["max"] = spec["min"] + max(spec["step"], 0.1)
            max_edit.setText(str(spec["max"]))
        spec["value"] = min(max(spec["value"], spec["min"]), spec["max"])
        label.setText(f"{float(spec['value']):.4g}")
        slider.blockSignals(True)
        slider.setMinimum(int(round(float(spec["min"]) / float(spec["step"]))))
        slider.setMaximum(int(round(float(spec["max"]) / float(spec["step"]))))
        slider.setValue(int(round(float(spec["value"]) / float(spec["step"]))))
        slider.blockSignals(False)
        print(f"[debug][formula-plot-tab] on_param_field_changed name={name!r} spec={spec!r}", flush=True)
        self._render_current_plot()

    def refresh_param_controls(self, mode: str) -> None:
        print(f"[debug][formula-plot-tab] refresh_param_controls mode={mode!r}", flush=True)
        self._clear_layout(self.param_rows_layout)
        self._slider_specs.clear()
        self._param_value_labels.clear()
        _sync_panel_params(self.state)
        nb_vars = self._notebook_vars()
        if not self.state.params:
            self.param_rows_layout.addWidget(QLabel("No extra parameters detected.", self.param_rows_container))
            return
        for name in sorted(self.state.params):
            spec = {**parameter_preset_for_name(name), **self.state.params.get(name, {})}
            self.state.params[name] = spec
            row = QWidget(self.param_rows_container)
            grid = QGridLayout(row)
            grid.setContentsMargins(0, 0, 0, 0)
            name_label = QLabel(name, row)
            nb_btn = QPushButton("NB", row)
            nb_btn.setStyleSheet(FORMULA_BUTTON_STYLE)
            status = QLabel("local", row)
            if spec.get("use_notebook") and name in nb_vars:
                status.setText(f"NB={float(nb_vars[name]):.4g}")
            elif spec.get("use_notebook"):
                status.setText("NB missing")
            slider = QSlider(Qt.Orientation.Horizontal, row)
            slider.setStyleSheet(FORMULA_SLIDER_STYLE)
            step = max(abs(float(spec.get("step", 0.1))), 1e-9)
            min_value = float(spec.get("min", -10.0))
            max_value = float(spec.get("max", 10.0))
            value = float(nb_vars[name]) if spec.get("use_notebook") and name in nb_vars else float(spec.get("value", 1.0))
            min_value = min(min_value, value)
            max_value = max(max_value, value)
            if min_value >= max_value:
                max_value = min_value + max(step, 0.1)
            slider.setMinimum(int(round(min_value / step)))
            slider.setMaximum(int(round(max_value / step)))
            slider.setValue(int(round(value / step)))
            value_label = QLabel(f"{value:.4g}", row)
            value_edit = QLineEdit(str(value), row)
            min_edit = QLineEdit(str(min_value), row)
            max_edit = QLineEdit(str(max_value), row)
            step_edit = QLineEdit(str(step), row)
            slider.valueChanged.connect(lambda v, n=name, lab=value_label: self._on_param_slider(n, v, lab))
            nb_btn.clicked.connect(lambda _=False, n=name, st=status: self._toggle_param_notebook(n, st))
            for editor in (value_edit, min_edit, max_edit, step_edit):
                editor.editingFinished.connect(lambda n=name, ve=value_edit, mie=min_edit, mae=max_edit, se=step_edit, lab=value_label, sl=slider: self._on_param_field_changed(n, ve, mie, mae, se, lab, sl))
            grid.addWidget(name_label, 0, 0)
            grid.addWidget(nb_btn, 0, 1)
            grid.addWidget(status, 0, 2)
            grid.addWidget(slider, 1, 0, 1, 3)
            grid.addWidget(value_label, 1, 3)
            grid.addWidget(QLabel("Value", row), 2, 0)
            grid.addWidget(value_edit, 3, 0)
            grid.addWidget(QLabel("Min", row), 2, 1)
            grid.addWidget(min_edit, 3, 1)
            grid.addWidget(QLabel("Max", row), 2, 2)
            grid.addWidget(max_edit, 3, 2)
            grid.addWidget(QLabel("Step", row), 2, 3)
            grid.addWidget(step_edit, 3, 3)
            self.param_rows_layout.addWidget(row)
            self._slider_specs[name] = (slider, value_label)
            self._param_value_labels[name] = value_label
        print(f"[debug][formula-plot-tab] refresh_param_controls done names={list(self._slider_specs)}", flush=True)

    def refresh_1d(self) -> None:
        print("[debug][formula-plot-tab] refresh_1d:start", flush=True)
        self.state.panel_type = "1d"
        if hasattr(self, "expression_edit") and self.state.formulas:
            previous_expression = str(self.state.formulas[0].get("expression") or "")
            expression = self.expression_edit.text().strip() or "sin(x)"
            label = str(self.state.formulas[0].get("label") or "")
            self.state.formulas[0]["expression"] = expression
            if _label_should_follow_expression(label, previous_expression):
                self.state.formulas[0]["label"] = expression
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.blockSignals(False)
        self.one_d_panel.show()
        self.range_panel.show()
        self.display_panel.show()
        self.analysis_panel.show()
        self.two_d_panel.hide()
        self._on_1d_settings_changed()
        self.refresh_param_controls("1d")
        self._render_current_plot()
        print("[debug][formula-plot-tab] refresh_1d:done", flush=True)

    def refresh_2d(self) -> None:
        print("[debug][formula-plot-tab] refresh_2d:start", flush=True)
        self.state.panel_type = "2d"
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.blockSignals(False)
        self.one_d_panel.hide()
        self.display_panel.hide()
        self.analysis_panel.hide()
        self.range_panel.show()
        self.two_d_panel.show()
        self._on_2d_settings_changed()
        self.refresh_param_controls("2d")
        self._render_current_plot()
        print("[debug][formula-plot-tab] refresh_2d:done", flush=True)

    def export_csv(self) -> None:
        print("[debug][formula-plot-tab] export_csv:start", flush=True)
        try:
            filename = "formula_surface_samples.csv" if self.state.panel_type == "2d" else "formula_samples.csv"
            path, _ = QFileDialog.getSaveFileName(self, "Export Formula CSV", filename, "CSV Files (*.csv)")
            print(f"[debug][formula-plot-tab] export_csv path={path!r}", flush=True)
            if not path:
                return
            nb_vars = self._notebook_vars()
            df = formula_surface_dataframe(self.state, nb_vars) if self.state.panel_type == "2d" else formula_samples_dataframe(self.state, nb_vars)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            self._set_status(f"Exported {Path(path).name}", FORMULA_STATUS_UPDATED_STYLE)
            print(f"[debug][formula-plot-tab] export_csv:done path={path!r} shape={df.shape}", flush=True)
        except Exception as exc:
            print(f"[debug][formula-plot-tab] export_csv:error error={exc!r}", flush=True)
            self._set_status(f"CSV export error: {exc}", FORMULA_STATUS_ERROR_STYLE)
