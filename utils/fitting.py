"""Curve fitting utilities for OPView data panels."""
from __future__ import annotations

import ast

import numpy as np
from scipy.optimize import curve_fit

from .formula_parser import ALLOWED_CONSTANTS, ALLOWED_FUNCTIONS, extract_formula_variables


BUILTIN_MODELS: dict[str, dict] = {
    "linear": {
        "label": "Linear  (a·x + b)",
        "formula": "a·x + b",
        "params": ["a", "b"],
        "func": staticmethod(lambda x, a, b: a * x + b),
        "p0": staticmethod(lambda x, y: [1.0, float(np.mean(y))]),
    },
    "poly2": {
        "label": "Polynomial  (a·x² + b·x + c)",
        "formula": "a·x² + b·x + c",
        "params": ["a", "b", "c"],
        "func": staticmethod(lambda x, a, b, c: a * x**2 + b * x + c),
        "p0": staticmethod(lambda x, y: [1.0, 1.0, float(np.mean(y))]),
    },
    "poly3": {
        "label": "Polynomial  (a·x³ + b·x² + c·x + d)",
        "formula": "a·x³ + b·x² + c·x + d",
        "params": ["a", "b", "c", "d"],
        "func": staticmethod(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d),
        "p0": staticmethod(lambda x, y: [1.0, 1.0, 1.0, float(np.mean(y))]),
    },
    "power": {
        "label": "Power Law  (a·xᵇ)",
        "formula": "a · x^b",
        "params": ["a", "b"],
        "func": staticmethod(lambda x, a, b: a * np.abs(x) ** b),
        "p0": staticmethod(lambda x, y: [1.0, 1.0]),
    },
    "exp_growth": {
        "label": "Exponential Growth  (a·eᵇˣ)",
        "formula": "a · exp(b·x)",
        "params": ["a", "b"],
        "func": staticmethod(lambda x, a, b: a * np.exp(b * x)),
        "p0": staticmethod(lambda x, y: [float(np.abs(y).max()) if len(y) else 1.0, 0.01]),
    },
    "exp_decay": {
        "label": "Exponential Decay  (a·e⁻ᵇˣ + c)",
        "formula": "a · exp(-b·x) + c",
        "params": ["a", "b", "c"],
        "func": staticmethod(lambda x, a, b, c: a * np.exp(-b * x) + c),
        "p0": staticmethod(lambda x, y: [float(np.abs(y).max()) if len(y) else 1.0, 0.1, 0.0]),
    },
    "log": {
        "label": "Logarithmic  (a·ln(x) + b)",
        "formula": "a · ln(x) + b",
        "params": ["a", "b"],
        "func": staticmethod(lambda x, a, b: a * np.log(np.abs(x) + 1e-10) + b),
        "p0": staticmethod(lambda x, y: [1.0, 0.0]),
    },
    "gaussian": {
        "label": "Gaussian",
        "formula": "a · exp(-(x−μ)²/(2σ²))",
        "params": ["a", "mu", "sigma"],
        "func": staticmethod(
            lambda x, a, mu, sigma: a * np.exp(-0.5 * ((x - mu) / (abs(sigma) + 1e-10)) ** 2)
        ),
        "p0": staticmethod(
            lambda x, y: [
                float(np.abs(y).max()) if len(y) else 1.0,
                float(np.mean(x)),
                float(np.std(x)) or 1.0,
            ]
        ),
    },
    "sine": {
        "label": "Sine  (a·sin(b·x + c) + d)",
        "formula": "a · sin(b·x + c) + d",
        "params": ["a", "b", "c", "d"],
        "func": staticmethod(lambda x, a, b, c, d: a * np.sin(b * x + c) + d),
        "p0": staticmethod(lambda x, y: [1.0, 1.0, 0.0, float(np.mean(y))]),
    },
    "sigmoid": {
        "label": "Sigmoid / Logistic",
        "formula": "a / (1 + exp(−b·(x−c)))",
        "params": ["a", "b", "c"],
        "func": staticmethod(
            lambda x, a, b, c: a / (1.0 + np.exp(-b * (x - c)))
        ),
        "p0": staticmethod(
            lambda x, y: [
                float(np.abs(y).max()) if len(y) else 1.0,
                1.0,
                float(np.mean(x)),
            ]
        ),
    },
}


class FitError(ValueError):
    """Raised when curve fitting fails."""


def fit_series(
    x: np.ndarray,
    y: np.ndarray,
    model: str,
    custom_formula: str = "",
    x_min: float | None = None,
    x_max: float | None = None,
    n_curve_points: int = 200,
) -> dict:
    """Fit x/y data with the selected model.

    Returns a dict with keys:
        params, uncertainties, r_squared, rmse, formula,
        x_curve, y_curve, error (None on success)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    if x_min is not None:
        mask &= x >= float(x_min)
    if x_max is not None:
        mask &= x <= float(x_max)

    x_data = x[mask]
    y_data = y[mask]

    if len(x_data) < 3:
        raise FitError("Need at least 3 data points in the selected range")

    if model == "custom":
        return _fit_custom(x_data, y_data, custom_formula, n_curve_points)

    if model not in BUILTIN_MODELS:
        raise FitError(f"Unknown model: '{model}'")

    m = BUILTIN_MODELS[model]
    func = m["func"]
    param_names: list[str] = m["params"]
    p0: list[float] = m["p0"](x_data, y_data)

    try:
        popt, pcov = curve_fit(func, x_data, y_data, p0=p0, maxfev=20000)
    except Exception as exc:
        raise FitError(f"Optimisation failed: {exc}") from exc

    return _build_result(
        x_data, y_data, lambda x: func(x, *popt),
        param_names, popt, pcov, m["formula"], n_curve_points,
    )


def _fit_custom(
    x_data: np.ndarray,
    y_data: np.ndarray,
    formula: str,
    n_curve_points: int,
) -> dict:
    """Fit using a user-supplied formula string (same sandbox as formula plot)."""
    if not (formula or "").strip():
        raise FitError("Please enter a formula")

    expr = formula.replace("^", "**").strip()
    param_names = sorted(extract_formula_variables(expr, coordinate_names=("x",)))

    if not param_names:
        raise FitError("No free parameters found. Use symbols like a, b, c alongside x.")
    if len(param_names) > 8:
        raise FitError("Too many parameters (max 8)")

    try:
        parsed = ast.parse(expr, mode="eval")
        compiled = compile(parsed, "<fit>", "eval")
    except SyntaxError as exc:
        raise FitError(f"Invalid formula syntax: {exc}") from exc

    context_base = {**ALLOWED_FUNCTIONS, **ALLOWED_CONSTANTS}

    def model_func(x: np.ndarray, *args) -> np.ndarray:
        ctx = {**context_base, "x": x}
        for name, val in zip(param_names, args):
            ctx[name] = val
        return np.asarray(eval(compiled, {"__builtins__": {}}, ctx), dtype=float)  # noqa: S307

    p0 = [1.0] * len(param_names)
    try:
        popt, pcov = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=30000)
    except Exception as exc:
        raise FitError(f"Optimisation failed: {exc}") from exc

    return _build_result(
        x_data, y_data, lambda x: model_func(x, *popt),
        param_names, popt, pcov, formula, n_curve_points,
    )


def _build_result(
    x_data: np.ndarray,
    y_data: np.ndarray,
    fitted_func,
    param_names: list[str],
    popt: np.ndarray,
    pcov: np.ndarray,
    formula: str,
    n_curve_points: int,
) -> dict:
    perr = np.sqrt(np.abs(np.diag(pcov)))
    y_pred = fitted_func(x_data)
    r2 = _r_squared(y_data, y_pred)
    rmse = float(np.sqrt(np.mean((y_data - y_pred) ** 2)))

    x_curve = np.linspace(float(x_data.min()), float(x_data.max()), n_curve_points)
    y_curve = fitted_func(x_curve)

    def _unc(e: float):
        return float(e) if np.isfinite(e) else None

    return {
        "params": {n: float(v) for n, v in zip(param_names, popt)},
        "uncertainties": {n: _unc(e) for n, e in zip(param_names, perr)},
        "r_squared": r2,
        "rmse": rmse,
        "formula": formula,
        "x_curve": x_curve.tolist(),
        "y_curve": np.where(np.isfinite(y_curve), y_curve, np.nan).tolist(),
        "error": None,
    }


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot
