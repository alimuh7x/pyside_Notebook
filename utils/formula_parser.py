"""
Safe formula parsing utilities for 1D and 2D formula plotting.
"""

from __future__ import annotations

import ast
from typing import Dict, Iterable

import numpy as np


def _np_min(*args):
    """Vectorized min that accepts multiple arguments."""
    if not args:
        raise ValueError("min requires at least one argument")
    arrays = [np.asarray(arg) for arg in args]
    result = arrays[0]
    for arr in arrays[1:]:
        result = np.minimum(result, arr)
    return result


def _np_max(*args):
    """Vectorized max that accepts multiple arguments."""
    if not args:
        raise ValueError("max requires at least one argument")
    arrays = [np.asarray(arg) for arg in args]
    result = arrays[0]
    for arr in arrays[1:]:
        result = np.maximum(result, arr)
    return result


ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "acos": np.arccos,
    "arccos": np.arccos,
    "asin": np.arcsin,
    "arcsin": np.arcsin,
    "atan": np.arctan,
    "arctan": np.arctan,
    "atan2": np.arctan2,
    "cos": np.cos,
    "cosh": np.cosh,
    "exp": np.exp,
    "hypot": np.hypot,
    "log": np.log,
    "log10": np.log10,
    "max": _np_max,
    "min": _np_min,
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
}

ALLOWED_CONSTANTS = {
    "e": np.e,
    "pi": np.pi,
}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


class FormulaValidationError(ValueError):
    """Raised when a formula contains unsupported syntax."""


class _FormulaValidator(ast.NodeVisitor):
    """Validate that the expression only contains allowed syntax."""

    def __init__(self, allowed_names: Iterable[str]):
        self.allowed_names = set(allowed_names)

    def generic_visit(self, node):
        if not isinstance(node, ALLOWED_NODES):
            raise FormulaValidationError(f"Unsupported syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.allowed_names:
            raise FormulaValidationError(f"Unknown symbol: {node.id}")

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise FormulaValidationError("Only approved math functions are allowed")
        for arg in node.args:
            self.visit(arg)
        if node.keywords:
            raise FormulaValidationError("Keyword arguments are not supported")


def extract_formula_variables(expression: str, coordinate_names: Iterable[str] = ("x",)) -> list[str]:
    """Extract user-defined variable names from a formula."""
    if not expression or not str(expression).strip():
        return []

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return []

    excluded = set(coordinate_names) | set(ALLOWED_FUNCTIONS.keys()) | set(ALLOWED_CONSTANTS.keys())
    names = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Name) and node.id not in excluded:
            names.add(node.id)
    return sorted(names)


def evaluate_formula(expression: str, x_values, extra_context: Dict[str, float] | None = None):
    """Evaluate a 1D formula safely against a NumPy x array."""
    if not expression or not str(expression).strip():
        raise FormulaValidationError("Please enter a formula")

    x_array = np.asarray(x_values)
    context = {"x": x_array}
    context.update(ALLOWED_FUNCTIONS)
    context.update(ALLOWED_CONSTANTS)
    if extra_context:
        context.update(extra_context)

    parsed = ast.parse(expression, mode="eval")
    _FormulaValidator(context.keys()).visit(parsed)
    compiled = compile(parsed, "<formula>", "eval")
    result = np.asarray(eval(compiled, {"__builtins__": {}}, context))

    if result.ndim == 0:
        result = np.full_like(x_array, float(result), dtype=float)
    if result.shape != x_array.shape:
        raise FormulaValidationError("Formula must return one y value for each x")
    return result


def evaluate_formula_2d(expression: str, x_grid, y_grid, extra_context: Dict[str, float] | None = None):
    """Evaluate a 2D formula safely against NumPy x/y grids."""
    if not expression or not str(expression).strip():
        raise FormulaValidationError("Please enter a formula")

    x_array = np.asarray(x_grid)
    y_array = np.asarray(y_grid)
    if x_array.shape != y_array.shape:
        raise FormulaValidationError("x and y grids must have the same shape")

    context = {"x": x_array, "y": y_array}
    context.update(ALLOWED_FUNCTIONS)
    context.update(ALLOWED_CONSTANTS)
    if extra_context:
        context.update(extra_context)

    parsed = ast.parse(expression, mode="eval")
    _FormulaValidator(context.keys()).visit(parsed)
    compiled = compile(parsed, "<formula2d>", "eval")
    result = np.asarray(eval(compiled, {"__builtins__": {}}, context))

    if result.ndim == 0:
        result = np.full_like(x_array, float(result), dtype=float)
    if result.shape != x_array.shape:
        raise FormulaValidationError("Formula must return one z value for each (x, y)")
    return result
