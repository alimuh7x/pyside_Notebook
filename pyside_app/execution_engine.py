from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import io
import threading
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import scipy

from utils.notebook_eval import ALLOWED_NOTEBOOK_FUNCTIONS, PHYSICAL_CONSTANTS, UNIT_CONSTANTS


SAFE_IMPORT_PREFIXES = (
    "math",
    "numpy",
    "pandas",
    "scipy",
    "plotly",
    "matplotlib",
    "sympy",
)

SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bool",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "globals",
    "hasattr",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "locals",
    "map",
    "max",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "__build_class__",
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "ZeroDivisionError",
    "NameError",
    "ImportError",
)


@dataclass
class ExecutionOutput:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    outputs: list[ExecutionOutput] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


def _format_value(value: Any) -> str:
    if isinstance(value, go.Figure):
        return "Plotly Figure"
    if isinstance(value, pd.DataFrame):
        return value.to_string(index=False)
    if isinstance(value, np.ndarray):
        return np.array2string(value, threshold=20)
    return repr(value) if isinstance(value, str) else str(value)


def _plotly_html(figure: go.Figure) -> str:
    print("[debug][execution-engine] plotly:html:start", flush=True)
    html = figure.to_html(
        include_plotlyjs=True,
        full_html=False,
        config={"responsive": True, "displaylogo": False},
    )
    has_external_cdn = 'src="https://cdn.plot.ly' in html or "src='https://cdn.plot.ly" in html
    print(
        f"[debug][execution-engine] plotly:html:done length={len(html)} external_cdn={has_external_cdn}",
        flush=True,
    )
    return html


def _output_from_value(value: Any) -> ExecutionOutput | None:
    if value is None:
        print("[debug][execution-engine] value_output:skip_none", flush=True)
        return None
    if isinstance(value, go.Figure):
        return ExecutionOutput(
            kind="plotly",
            data={
                "figure": value,
                "html": _plotly_html(value),
                "text": "Plotly Figure",
            },
        )
    if isinstance(value, pd.DataFrame):
        return ExecutionOutput(
            kind="table",
            data={
                "shape": value.shape,
                "text": value.to_string(index=False),
                "html": value.to_html(index=False),
            },
        )
    return ExecutionOutput(kind="value", data={"text": _format_value(value)})


def _prepare_matplotlib() -> Any | None:
    try:
        import matplotlib

        backend = matplotlib.get_backend().lower()
        if backend != "agg":
            matplotlib.use("Agg")
            print("[debug][execution-engine] matplotlib:backend_set backend='Agg'", flush=True)
        import matplotlib.pyplot as plt

        print("[debug][execution-engine] matplotlib:prepared", flush=True)
        return plt
    except Exception as exc:
        print(f"[debug][execution-engine] matplotlib:unavailable error={exc!r}", flush=True)
        return None


def _convert_matplotlib_to_plotly(figure: Any) -> go.Figure | None:
    try:
        from plotly.tools import mpl_to_plotly
    except Exception as exc:
        print(f"[debug][execution-engine] matplotlib:converter_unavailable error={exc!r}", flush=True)
        return None
    try:
        converted = mpl_to_plotly(figure)
        converted.update_layout(autosize=True)
        print("[debug][execution-engine] matplotlib:converter_success", flush=True)
        return converted
    except Exception as exc:
        print(f"[debug][execution-engine] matplotlib:converter_failed error={exc!r}", flush=True)
        return None


def _collect_matplotlib_outputs(plt: Any | None) -> list[ExecutionOutput]:
    if plt is None:
        return []
    outputs: list[ExecutionOutput] = []
    figure_numbers = list(plt.get_fignums())
    print(f"[debug][execution-engine] matplotlib:figures count={len(figure_numbers)}", flush=True)
    for figure_number in figure_numbers:
        figure = plt.figure(figure_number)
        print(f"[debug][execution-engine] matplotlib:figure:start number={figure_number}", flush=True)
        plotly_figure = _convert_matplotlib_to_plotly(figure)
        if plotly_figure is not None:
            print(f"[debug][execution-engine] matplotlib:figure:converted number={figure_number}", flush=True)
            outputs.append(
                ExecutionOutput(
                    kind="plotly",
                    data={
                        "figure": plotly_figure,
                        "html": _plotly_html(plotly_figure),
                        "text": "Matplotlib Figure",
                    },
                )
            )
            continue
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        print(f"[debug][execution-engine] matplotlib:figure:fallback_png number={figure_number}", flush=True)
        outputs.append(
            ExecutionOutput(
                kind="html",
                data={
                    "text": "Matplotlib Figure",
                    "html": (
                        "<div style='padding: 4px 0;'>"
                        f"<img src='data:image/png;base64,{encoded}' "
                        "alt='Matplotlib Figure' style='max-width: 100%; height: auto;' />"
                        "</div>"
                    ),
                },
            )
        )
    return outputs


def _split_last_expression(source: str) -> tuple[ast.Module, ast.expr | None]:
    parsed = ast.parse(source, mode="exec")
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        last_expr = parsed.body[-1].value
        parsed.body = parsed.body[:-1]
        return parsed, last_expr
    return parsed, None


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] | list[str] | None = (),
    level: int = 0,
) -> Any:
    normalized_fromlist = tuple(fromlist or ())
    print(
        f"[debug][execution-engine] safe_import:start name={name!r} fromlist={normalized_fromlist!r} level={level}",
        flush=True,
    )
    if level != 0:
        print("[debug][execution-engine] safe_import:blocked relative_import", flush=True)
        raise ImportError("Relative imports are not allowed in desktop notebook")
    if not any(name == prefix or name.startswith(prefix + ".") for prefix in SAFE_IMPORT_PREFIXES):
        print(f"[debug][execution-engine] safe_import:blocked name={name!r}", flush=True)
        raise ImportError(f"Import of module {name!r} is not allowed in desktop notebook")
    imported = builtins.__import__(name, globals, locals, normalized_fromlist, level)
    print(f"[debug][execution-engine] safe_import:done name={name!r}", flush=True)
    return imported


def _safe_builtins() -> dict[str, Any]:
    safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    safe["__import__"] = _safe_import
    print(f"[debug][execution-engine] safe_builtins:count count={len(safe)}", flush=True)
    return safe


def build_base_namespace() -> dict[str, Any]:
    callable_min = UNIT_CONSTANTS["min"]
    namespace: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "np": np,
        "pd": pd,
        "scipy": scipy,
        "plotly": plotly,
        "go": go,
        "math": __import__("math"),
        "tau": 2 * np.pi,
        "deg": np.pi / 180.0,
        "inf": float("inf"),
        **PHYSICAL_CONSTANTS,
        **UNIT_CONSTANTS,
        **ALLOWED_NOTEBOOK_FUNCTIONS,
        "min": callable_min,
    }
    return namespace


class ExecutionEngine:
    def __init__(self, initial_namespace: dict[str, Any] | None = None) -> None:
        self._base_namespace = build_base_namespace()
        self.namespace: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._restart_pending = False
        self.restart()
        if initial_namespace:
            with self._lock:
                self.namespace.update(initial_namespace)

    def _reset_namespace_unlocked(self) -> None:
        self.namespace = dict(self._base_namespace)

    def restart(self) -> bool:
        if self._lock.acquire(timeout=0.05):
            try:
                self._restart_pending = False
                self._reset_namespace_unlocked()
            finally:
                self._lock.release()
            return True
        self._restart_pending = True
        return False

    def _apply_pending_restart_if_needed(self) -> None:
        if not self._restart_pending:
            return
        with self._lock:
            if self._restart_pending:
                self._reset_namespace_unlocked()
                self._restart_pending = False

    def execute(self, source: str) -> ExecutionResult:
        self._apply_pending_restart_if_needed()
        print(f"[debug][execution-engine] execute: source={source!r}", flush=True)
        result = ExecutionResult()
        if not (source or "").strip():
            print("[debug][execution-engine] execute: empty_source", flush=True)
            return result

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with self._lock:
                plt = _prepare_matplotlib()
                exec_tree, last_expr = _split_last_expression(source)
                if exec_tree.body:
                    compiled = compile(exec_tree, "<desktop-notebook>", "exec")
                    with (
                        contextlib.redirect_stdout(stdout_buffer),
                        contextlib.redirect_stderr(stderr_buffer),
                        warnings.catch_warnings(),
                    ):
                        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")
                        exec(compiled, self.namespace, self.namespace)  # noqa: S102
                if last_expr is not None:
                    compiled_expr = compile(ast.Expression(last_expr), "<desktop-notebook-expr>", "eval")
                    with (
                        contextlib.redirect_stdout(stdout_buffer),
                        contextlib.redirect_stderr(stderr_buffer),
                        warnings.catch_warnings(),
                    ):
                        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")
                        value = eval(compiled_expr, self.namespace, self.namespace)  # noqa: S307
                    output = _output_from_value(value)
                    if output is not None:
                        result.outputs.append(output)
                result.outputs.extend(_collect_matplotlib_outputs(plt))
                if plt is not None:
                    plt.close("all")
                    print("[debug][execution-engine] matplotlib:closed_all", flush=True)
        except Exception:
            result.error = traceback.format_exc()
            print(f"[debug][execution-engine] execute:error error={result.error!r}", flush=True)

        result.stdout = stdout_buffer.getvalue()
        result.stderr = stderr_buffer.getvalue()
        if result.stdout:
            result.outputs.insert(0, ExecutionOutput(kind="stdout", data={"text": result.stdout.rstrip()}))
        if result.stderr:
            result.outputs.append(ExecutionOutput(kind="stderr", data={"text": result.stderr.rstrip()}))
        self._apply_pending_restart_if_needed()
        print(
            f"[debug][execution-engine] execute:done outputs={len(result.outputs)} "
            f"stdout_len={len(result.stdout)} stderr_len={len(result.stderr)} has_error={result.error is not None}",
            flush=True,
        )
        return result

    def get_namespace(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self.namespace)
        print(f"[debug][execution-engine] get_namespace count={len(snapshot)}", flush=True)
        return snapshot
