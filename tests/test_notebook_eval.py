"""
Tests for utils/notebook_eval.py — the calculation notebook evaluator.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from utils.notebook_eval import evaluate_notebook_rows


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rows(*lines: str) -> list[dict]:
    return [{"id": f"line_{i}", "expression": line} for i, line in enumerate(lines)]


def _run(*lines: str):
    """Evaluate lines and return (evaluated_rows, variables, array_variables)."""
    return evaluate_notebook_rows(_rows(*lines))


def _result(rows, index: int) -> str:
    return rows[index].get("result", "")


def _error(rows, index: int) -> str:
    return rows[index].get("error", "")


# ── Basic arithmetic & scalars ─────────────────────────────────────────────────

class TestArithmetic:
    def test_simple_addition(self):
        ev, _, _ = _run("2 + 3")
        assert _result(ev, 0) == "5"

    def test_power_caret(self):
        ev, _, _ = _run("2^10")
        assert _result(ev, 0) == "1024"

    def test_float_division(self):
        ev, _, _ = _run("7 / 2")
        assert _result(ev, 0) == "3.5"

    def test_unary_minus(self):
        ev, _, _ = _run("-5")
        assert _result(ev, 0) == "-5"

    def test_nested_arithmetic(self):
        ev, _, _ = _run("(3 + 4) * 2 - 1")
        assert _result(ev, 0) == "13"


# ── Variable assignment & carry-forward ────────────────────────────────────────

class TestVariables:
    def test_assignment_shows_value(self):
        ev, variables, _ = _run("x = 42")
        # Assignments display the value in the gutter
        assert _result(ev, 0) == "42"
        assert "x" in variables
        assert variables["x"] == pytest.approx(42)

    def test_carry_forward(self):
        ev, _, _ = _run("a = 5", "b = a * 3", "b")
        assert _result(ev, 2) == "15"

    def test_augmented_assign(self):
        ev, variables, _ = _run("n = 10", "n = n + 5")
        assert variables["n"] == pytest.approx(15)

    def test_multiline_parenthesized_assignment(self):
        ev, variables, _ = _run(
            "x = (1 + 2 +",
            "     3 + 4)",
        )
        assert _error(ev, 0) == ""
        assert _result(ev, 0) == "10"
        assert variables["x"] == pytest.approx(10)
        assert ev[0]["source_span"] == 2


# ── Unit constants ─────────────────────────────────────────────────────────────

class TestUnits:
    def test_gpa(self):
        ev, variables, _ = _run("E = 210 * GPa")
        assert variables["E"] == pytest.approx(210e9)

    def test_deg_conversion(self):
        ev, _, _ = _run("sin(45 * deg)")
        assert float(_result(ev, 0)) == pytest.approx(math.sin(math.pi / 4), rel=1e-6)

    def test_unit_multiply(self):
        """Use explicit multiplication for units: 5 * mm → 5e-3."""
        ev, variables, _ = _run("d = 5 * mm")
        assert variables["d"] == pytest.approx(5e-3)


# ── Built-in math functions ────────────────────────────────────────────────────

class TestMathFunctions:
    def test_sin_cos(self):
        ev, _, _ = _run("sin(pi / 2)")
        assert float(_result(ev, 0)) == pytest.approx(1.0)

    def test_sqrt(self):
        ev, _, _ = _run("sqrt(9)")
        assert _result(ev, 0) == "3"

    def test_log_exp(self):
        ev, _, _ = _run("log(exp(1))")
        assert float(_result(ev, 0)) == pytest.approx(1.0, abs=1e-9)

    def test_abs_is_not_available(self):
        ev, _, _ = _run("abs(-7.5)")
        assert "Only approved functions are allowed" in _error(ev, 0)


# ── Array operations ───────────────────────────────────────────────────────────

class TestArrays:
    def test_linspace(self):
        _, _, arr_vars = _run("t = linspace(0, 1, 5)")
        assert "t" in arr_vars
        assert len(arr_vars["t"]) == 5
        assert arr_vars["t"][0] == pytest.approx(0.0)
        assert arr_vars["t"][-1] == pytest.approx(1.0)

    def test_mean_of_array(self):
        ev, _, _ = _run("v = [2, 4, 6]", "mean(v)")
        assert _result(ev, 1) == "4"

    def test_sum_is_not_available(self):
        ev, _, _ = _run("sum([1, 2, 3, 4])")
        assert "Only approved functions are allowed" in _error(ev, 0)

    def test_zeros_ones(self):
        ev, _, arr_vars = _run("z = zeros(3)", "o = ones(5)")
        assert "z" in arr_vars
        assert "o" in arr_vars
        assert all(v == 0 for v in arr_vars["z"])
        assert all(v == 1 for v in arr_vars["o"])

    def test_array_result_formatting(self):
        ev, _, _ = _run("[1, 2, 3]")
        assert _result(ev, 0) == "[1, 2, 3]"

    def test_list_comprehension(self):
        ev, _, _ = _run("vals = [i * i for i in range(4)]")
        assert _error(ev, 0) == ""
        assert _result(ev, 0) == "[0, 1, 4, 9]"

    def test_list_comprehension_uses_existing_variable(self):
        ev, _, _ = _run("scale = 3", "vals = [scale * i for i in range(4)]")
        assert _error(ev, 1) == ""
        assert _result(ev, 1) == "[0, 3, 6, 9]"

    def test_copy_method_creates_independent_vector_copy(self):
        ev, _, arr_vars = _run(
            "a = [1, 2, 3]",
            "b = a.copy()",
            "b[0] = 99",
        )
        assert _error(ev, 1) == ""
        assert arr_vars["a"] == pytest.approx([1, 2, 3])
        assert arr_vars["b"] == pytest.approx([99, 2, 3])


class TestDictionaries:
    def test_dictionary_literal(self):
        ev, _, _ = _run("params = {'a': 1, 'b': 2}")
        assert _error(ev, 0) == ""
        assert "'a': 1" in _result(ev, 0)
        assert "'b': 2" in _result(ev, 0)

    def test_multiline_dictionary_literal(self):
        ev, _, _ = _run(
            "params = {",
            "    'a': 1,",
            "    'b': 2,",
            "}",
        )
        assert _error(ev, 0) == ""
        assert "'a': 1" in _result(ev, 0)


# ── Matrix operations ──────────────────────────────────────────────────────────

class TestMatrix:
    def test_det(self):
        ev, _, _ = _run("det([[1, 2], [3, 4]])")
        assert float(_result(ev, 0)) == pytest.approx(-2.0)

    def test_eye(self):
        ev, _, _ = _run("I = eye(3)")
        # 3×3 identity matrix shown in result
        assert "1" in _result(ev, 0)
        assert "0" in _result(ev, 0)

    def test_matmul(self):
        # Lists are not numpy arrays: must assign to variables for @
        ev, _, arr_vars = _run(
            "A = eye(2)",
            "B = linspace(1, 2, 2)",
            "C = A @ B",
        )
        assert _result(ev, 2) != "" or _error(ev, 2) == ""

    def test_solve(self):
        ev, variables, arr_vars = _run("x_sol = solve([[2, 0], [0, 3]], [4, 9])")
        # solve returns [2, 3]
        assert "x_sol" in arr_vars
        assert arr_vars["x_sol"] == pytest.approx([2.0, 3.0])

    def test_multiline_matrix_assignment(self):
        ev, _, _ = _run(
            "n_all = [[ 1, 1, 1],[ 1, 1, 1],[ 1, 1, 1],[-1, 1, 1],",
            "         [-1, 1, 1],[-1, 1, 1],[ 1,-1, 1],[ 1,-1, 1]]",
        )
        assert _error(ev, 0) == ""
        assert _result(ev, 0) != ""
        assert _error(ev, 1) == ""
        assert ev[0]["source_span"] == 2

    def test_copy_method_creates_independent_matrix_copy(self):
        ev, _, _ = _run(
            "A = eye(2)",
            "B = A.copy()",
            "B[0][0] = 5",
            "A",
            "B",
        )
        assert _error(ev, 1) == ""
        assert _result(ev, 3) == "[[1, 0], [0, 1]]"
        assert _result(ev, 4) == "[[5, 0], [0, 1]]"


class TestBlocks:
    def test_for_block_carries_source_span(self):
        ev, _, _ = _run(
            "for i in range(3):",
            "    x = i",
        )
        assert ev[0]["source_span"] == 2


# ── Engineering functions ──────────────────────────────────────────────────────

class TestEngineering:
    def test_von_mises_uniaxial(self):
        """Von Mises of uniaxial stress = stress itself."""
        ev, _, _ = _run("von_mises(100, 0, 0)")
        assert float(_result(ev, 0)) == pytest.approx(100.0)

    def test_shear_modulus(self):
        ev, _, _ = _run("shear_modulus(210e9, 0.3)")
        expected = 210e9 / (2 * 1.3)
        assert float(_result(ev, 0)) == pytest.approx(expected, rel=1e-5)

    def test_diffusion_dt_stable(self):
        """dt should be ≤ dx²/(2D) for stability."""
        ev, variables, _ = _run("dt = diffusion_dt(0.01, 1e-4)")
        assert variables["dt"] == pytest.approx(0.5 * 0.01**2 / 1e-4)


# ── For loops ──────────────────────────────────────────────────────────────────

class TestForLoops:
    def test_simple_loop(self):
        ev, variables, _ = _run("acc = 0", "for i in range(5):", "    acc = acc + i")
        # 0+1+2+3+4 = 10
        assert variables["acc"] == pytest.approx(10)

    def test_loop_result_label(self):
        ev, _, _ = _run("for k in range(3):", "    k")
        # Header row shows [done — N iter]
        assert "done" in _result(ev, 0)

    def test_loop_fills_array(self):
        _, _, arr_vars = _run(
            "v = zeros(4)",
            "for i in range(4):",
            "    v[i] = i * 2",
        )
        assert arr_vars["v"] == pytest.approx([0, 2, 4, 6])


# ── Error handling ─────────────────────────────────────────────────────────────

class TestErrors:
    def test_unknown_symbol(self):
        ev, _, _ = _run("unknown_var")
        assert _error(ev, 0) != "" or "Error" in _result(ev, 0)

    def test_division_by_zero(self):
        ev, _, _ = _run("1 / 0")
        # Should produce inf or an error, not crash
        result = _result(ev, 0)
        assert "inf" in result.lower() or _error(ev, 0) != ""

    def test_comment_lines_are_empty(self):
        ev, _, _ = _run("# this is a comment", "// also a comment")
        assert _result(ev, 0) == ""
        assert _result(ev, 1) == ""
        assert _error(ev, 0) == ""
        assert _error(ev, 1) == ""

    def test_empty_line(self):
        ev, _, _ = _run("")
        assert _result(ev, 0) == ""
        assert _error(ev, 0) == ""
