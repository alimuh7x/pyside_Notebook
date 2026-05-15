from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import inspect
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
    "pint",
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
class InteractParamSpec:
    """Describes one slider or dropdown parameter for an interact() call."""
    type: str  # "slider" or "choice"
    value: Any = 0.0
    min: float = -10.0
    max: float = 10.0
    step: float = 0.1
    options: list[Any] = field(default_factory=list)


@dataclass
class InteractRequest:
    """Captures an interact() call for rendering as a slider panel in the cell output."""
    fn: Any
    param_specs: dict[str, InteractParamSpec]
    initial_output: "ExecutionOutput | None"


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
    namespace_snapshot: dict[str, Any] | None = None  # set by execute_with_overrides


def _format_value(value: Any) -> str:
    """Convert a Python value into the text shown in notebook outputs."""
    if isinstance(value, go.Figure):
        return "Plotly Figure"
    if isinstance(value, pd.DataFrame):
        return value.to_string(index=False)
    if isinstance(value, np.ndarray):
        return np.array2string(value, threshold=20)
    return repr(value) if isinstance(value, str) else str(value)


def _plotly_html(figure: go.Figure) -> str:
    """Render a Plotly figure into embeddable HTML for the UI."""
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
    """Convert the final expression value into a structured notebook output."""
    if value is None:
        print("[debug][execution-engine] value_output:skip_none", flush=True)
        return None
    if isinstance(value, InteractRequest):
        return ExecutionOutput(
            kind="interact",
            data={"request": value, "text": "Interactive widget"},
        )
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
    """Prepare Matplotlib for headless notebook execution if it is available."""
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
    """Try to convert one Matplotlib figure into an equivalent Plotly figure."""
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
    """Collect every Matplotlib figure created during execution as notebook outputs."""
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
    """Split code into exec statements plus an optional trailing expression."""
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
    """Restrict notebook imports to a predefined safe allowlist."""
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
    """Build the reduced builtin namespace available to notebook code."""
    safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    safe["__import__"] = _safe_import
    print(f"[debug][execution-engine] safe_builtins:count count={len(safe)}", flush=True)
    return safe


def _build_interact_fn() -> Any:
    """Build the interact() function for the notebook namespace."""
    def interact(fn, **kwargs):
        """Create an interactive slider panel for a function.

        Usage::

            def plot(E=200e9, nu=0.3):
                eps = np.linspace(0, 0.01, 200)
                fig = go.Figure()
                fig.add_scatter(x=eps, y=E*eps/1e6, name=f"E={E/1e9:.0f} GPa")
                return fig

            interact(plot, E=(70e9, 210e9, 1e9), nu=(0.1, 0.5, 0.01))

        Each keyword arg can be:
        - (min, max)          — continuous slider, 100 steps
        - (min, max, step)    — slider with given step
        - [a, b, c, ...]      — discrete dropdown choices
        """
        sig = inspect.signature(fn)
        param_specs: dict[str, InteractParamSpec] = {}

        for name, param in sig.parameters.items():
            default = param.default if param.default is not inspect.Parameter.empty else 0.0

            if name in kwargs:
                spec_raw = kwargs[name]
                if isinstance(spec_raw, (list, tuple)) and not isinstance(spec_raw, str):
                    if isinstance(spec_raw, list):
                        val = default if default in spec_raw else spec_raw[0]
                        param_specs[name] = InteractParamSpec(type="choice", value=val, options=list(spec_raw))
                    else:
                        lo = float(spec_raw[0])
                        hi = float(spec_raw[1])
                        step = float(spec_raw[2]) if len(spec_raw) > 2 else round((hi - lo) / 100, 12)
                        val = float(default) if isinstance(default, (int, float)) else lo
                        val = max(lo, min(hi, val))
                        param_specs[name] = InteractParamSpec(type="slider", value=val, min=lo, max=hi, step=step)
                else:
                    val = float(spec_raw) if isinstance(spec_raw, (int, float)) else 1.0
                    param_specs[name] = InteractParamSpec(type="slider", value=val, min=val - 10, max=val + 10, step=0.1)
            else:
                val = float(default) if isinstance(default, (int, float)) else 0.0
                half = max(abs(val) * 2, 5.0)
                param_specs[name] = InteractParamSpec(type="slider", value=val, min=val - half, max=val + half, step=round(half / 50, 12))

        initial_values = {name: spec.value for name, spec in param_specs.items()}
        try:
            result = fn(**initial_values)
        except Exception as exc:
            result = f"Error computing initial output: {exc}"

        initial_output = _output_from_value(result)
        print(f"[debug][interact] fn={fn.__name__!r} params={list(param_specs.keys())}", flush=True)
        return InteractRequest(fn=fn, param_specs=param_specs, initial_output=initial_output)

    return interact


def _build_animate_frames_fn() -> Any:
    """Build the animate_frames() helper for animated Plotly figures."""
    def animate_frames(
        x_data,
        y_data_list,
        frame_labels=None,
        title="Animation",
        xaxis_title="x",
        yaxis_title="y",
        mode="lines",
        duration=100,
    ):
        """Create an animated Plotly figure from a sequence of y-data arrays.

        Args:
            x_data: 1-D x-axis values shared across all frames.
            y_data_list: Iterable of 1-D y arrays (one per frame).
            frame_labels: Optional labels shown in the animation slider.
            title, xaxis_title, yaxis_title: Layout strings.
            mode: Plotly trace mode — 'lines', 'markers', 'lines+markers'.
            duration: Milliseconds between frames.

        Returns:
            go.Figure with animation frames and play/pause controls.
        """
        frames_list = list(y_data_list)
        n = len(frames_list)
        labels = list(frame_labels) if frame_labels is not None else [str(i) for i in range(n)]

        frames = [
            go.Frame(
                data=[go.Scatter(x=x_data, y=frames_list[i], mode=mode)],
                name=labels[i],
            )
            for i in range(n)
        ]

        fig = go.Figure(
            data=[go.Scatter(x=x_data, y=frames_list[0], mode=mode)],
            frames=frames,
            layout=go.Layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                template="plotly_white",
                updatemenus=[{
                    "type": "buttons",
                    "showactive": False,
                    "y": 0,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "buttons": [
                        {"label": "Play", "method": "animate",
                         "args": [None, {"frame": {"duration": duration, "redraw": True}, "fromcurrent": True}]},
                        {"label": "Pause", "method": "animate",
                         "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                    ],
                }],
                sliders=[{
                    "steps": [
                        {"args": [[lbl], {"frame": {"duration": duration, "redraw": True}, "mode": "immediate"}],
                         "label": lbl, "method": "animate"}
                        for lbl in labels
                    ],
                    "y": 0, "len": 0.9, "x": 0.05,
                }],
            ),
        )
        print(f"[debug][animate_frames] frames={n}", flush=True)
        return fig

    return animate_frames


def _ast_literal_number(node: ast.expr) -> float | int | None:
    """Return the numeric value if node is a plain literal or unary-minus literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        if isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float)) and not isinstance(inner.value, bool):
            return -inner.value
    return None


_HIDDEN_PARAMETER_NAMES = {
    "i",
    "j",
    "k",
    "n",
    "m",
    "idx",
    "index",
    "row",
    "col",
    "save_id",
    "run_id",
    "plot_id",
    "figure_id",
    "n_save",
}

_LOG_PARAMETER_PARTS = (
    "diff",
    "rate",
    "kappa",
    "conduct",
    "i0",
    "dt",
    "dx",
    "dy",
    "dz",
    "tol",
    "eps",
)

_INTEGER_PARAMETER_PARTS = (
    "nx",
    "ny",
    "nz",
    "nt",
    "iter",
    "steps",
    "points",
    "samples",
)


def _parameter_priority(name: str) -> int:
    """Rank likely physics knobs before bookkeeping constants."""
    lowered = name.lower()
    if lowered in {"d", "dt", "dx", "dy", "dz", "kappa", "nu", "rho"}:
        return 0
    if any(part in lowered for part in ("alpha", "beta", "gamma", "omega", "phi", "eta", "potential", "voltage")):
        return 1
    if any(part in lowered for part in ("i0", "rate", "diff", "conduct", "modulus", "young", "poisson")):
        return 1
    if any(part in lowered for part in ("nx", "ny", "nz", "nt", "iter", "step", "time")):
        return 2
    return 5


def _is_bookkeeping_parameter(name: str, val: float | int) -> bool:
    """Return True for literal counters that create noisy sliders."""
    lowered = name.lower()
    if lowered in _HIDDEN_PARAMETER_NAMES:
        return True
    if lowered.endswith("_id") or lowered.endswith("_idx") or lowered.endswith("_index"):
        return True
    if lowered.startswith(("save_", "plot_", "figure_")) and abs(float(val)) < 2:
        return True
    return False


def _looks_like_log_parameter(name: str, val: float | int) -> bool:
    """Return True when multiplicative changes are more useful than linear changes."""
    if float(val) <= 0:
        return False
    lowered = name.lower()
    if any(part in lowered for part in _LOG_PARAMETER_PARTS):
        return True
    return abs(float(val)) >= 1e4 or abs(float(val)) < 1e-2


def _looks_like_integer_parameter(name: str, val: float | int) -> bool:
    """Return True for integer mesh/time-count parameters."""
    if not isinstance(val, int):
        return False
    lowered = name.lower()
    return any(part in lowered for part in _INTEGER_PARAMETER_PARTS)


def _param_spec(name: str, val: float | int) -> dict[str, Any] | None:
    """Build a smart slider spec centred on a detected literal value."""
    if _is_bookkeeping_parameter(name, val):
        return None
    is_int = isinstance(val, int)
    v = float(val)
    lowered = name.lower()
    scale = "linear"

    if "omega" in lowered or "relax" in lowered:
        lo, hi = 0.0, 1.0
        step = 0.01
    elif _looks_like_log_parameter(name, val):
        scale = "log"
        lo = max(v / 100.0, 1e-30)
        hi = max(v * 100.0, lo * 10.0)
        step = 1.0
    elif _looks_like_integer_parameter(name, val):
        lo = max(1.0, round(v * 0.25))
        hi = max(lo + 1.0, round(v * 3.0))
        step = max(1.0, round((hi - lo) / 120))
    elif any(part in lowered for part in ("potential", "voltage", "phi", "eta")) or lowered.startswith("e_"):
        span = max(abs(v) * 1.5, 0.25)
        lo, hi = v - span, v + span
        step = (hi - lo) / 200
    elif v == 0.0:
        lo, hi = -5.0, 5.0
        step = 0.05
    elif v > 0:
        lo, hi = v * 0.1, v * 3.0
        step = (hi - lo) / 200
    else:
        lo, hi = v * 3.0, abs(v) * 0.5
        step = (hi - lo) / 200

    if is_int:
        lo = max(1.0, round(lo))
        hi = max(lo + 1.0, round(hi))
        step = max(1.0, round(step))
        scale = "linear"
    return {
        "value": val,
        "min": lo,
        "max": hi,
        "step": step,
        "is_int": is_int,
        "scale": scale,
        "priority": _parameter_priority(name),
    }


def _apply_overrides_to_source(source: str, overrides: dict[str, Any]) -> str:
    """Replace top-level literal assignments in AST so overrides take effect.

    Prepending `Nx = 420` before source fails when source contains `Nx = 400`
    later — the original assignment overwrites the override. Instead, walk the
    parsed AST and replace the RHS of matching top-level assignments in-place,
    then unparse back to source.
    """
    if not overrides:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    replaced: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name not in overrides:
                continue
            val = overrides[name]
            if isinstance(val, bool):
                continue
            if isinstance(val, int):
                node.value = ast.Constant(value=val)
            else:
                node.value = ast.Constant(value=float(val))
            ast.fix_missing_locations(node)
            replaced.add(name)

    # Prepend any overrides whose name wasn't found as a top-level assignment
    prefix_lines: list[str] = []
    for name, val in overrides.items():
        if name not in replaced:
            if isinstance(val, int) and not isinstance(val, bool):
                prefix_lines.append(f"{name} = {val!r}")
            else:
                prefix_lines.append(f"{name} = {float(val)!r}")

    modified = ast.unparse(tree)
    if prefix_lines:
        modified = "\n".join(prefix_lines) + "\n" + modified
    return modified


def detect_cell_parameters(source: str) -> dict[str, dict[str, Any]]:
    """Scan source for top-level assignments from number literals.

    Returns a dict mapping variable name → slider spec:
    ``{value, min, max, step, is_int}``.
    Only top-level (module-level) simple ``name = literal`` statements are
    considered — loop variables, augmented assignments, and computed values
    are ignored.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    params: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        val = _ast_literal_number(node.value)
        if val is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                spec = _param_spec(target.id, val)
                if spec is not None:
                    params[target.id] = spec
    params = dict(sorted(params.items(), key=lambda item: (item[1]["priority"], item[0].lower())))
    print(f"[debug][execution-engine] detect_cell_parameters found={list(params.keys())}", flush=True)
    return params


def build_base_namespace() -> dict[str, Any]:
    """Create the default execution namespace shared by notebook cells."""
    callable_min = UNIT_CONSTANTS["min"]

    try:
        from plotly.subplots import make_subplots as _make_subplots
    except Exception:
        _make_subplots = None

    try:
        import plotly.figure_factory as _ff
        _quiver = _ff.create_quiver
        _streamline = _ff.create_streamline
    except Exception:
        _quiver = None
        _streamline = None

    try:
        import pint as _pint
        _ureg = _pint.UnitRegistry()
        _Q = _ureg.Quantity
    except Exception:
        _ureg = None
        _Q = None

    extra: dict[str, Any] = {}
    if _make_subplots is not None:
        extra["make_subplots"] = _make_subplots
    if _quiver is not None:
        extra["quiver"] = _quiver
        extra["streamline"] = _streamline
    if _ureg is not None:
        extra["ureg"] = _ureg
        extra["Q_"] = _Q

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
        "interact": _build_interact_fn(),
        "animate_frames": _build_animate_frames_fn(),
        **PHYSICAL_CONSTANTS,
        **UNIT_CONSTANTS,
        **ALLOWED_NOTEBOOK_FUNCTIONS,
        "min": callable_min,
        **extra,
    }
    return namespace


class ExecutionEngine:
    """Execute notebook code inside a persistent, thread-safe Python namespace."""

    def __init__(self, initial_namespace: dict[str, Any] | None = None) -> None:
        """Initialize the engine, base namespace, and restart state."""
        self._base_namespace = build_base_namespace()
        self.namespace: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._restart_pending = False
        self.restart()
        if initial_namespace:
            with self._lock:
                self.namespace.update(initial_namespace)

    def _reset_namespace_unlocked(self) -> None:
        """Reset the runtime namespace back to the base environment."""
        self.namespace = dict(self._base_namespace)

    def restart(self) -> bool:
        """Reset the namespace immediately or mark a restart for the next safe point."""
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
        """Apply a deferred restart request before or after execution."""
        if not self._restart_pending:
            return
        with self._lock:
            if self._restart_pending:
                self._reset_namespace_unlocked()
                self._restart_pending = False

    def execute(self, source: str) -> ExecutionResult:
        """Run one notebook cell and return normalized outputs and errors."""
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

    def execute_with_overrides(self, source: str, overrides: dict[str, Any]) -> ExecutionResult:
        """Re-run source with scalar overrides in an isolated namespace snapshot.

        The shared notebook namespace is **never modified** — a copy is taken
        under the lock, the overrides are prepended as assignments, and execution
        runs in that private copy.  Safe to call from a background thread while
        the main engine is idle.
        """
        result = ExecutionResult()
        if not (source or "").strip():
            return result

        full_source = _apply_overrides_to_source(source, overrides)

        # Take a snapshot without holding the lock during execution.
        # Deep-copy numpy arrays so that in-place writes inside exec() cannot
        # alias with self.namespace or with earlier slider-run snapshots.
        with self._lock:
            ns_snap = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in self.namespace.items()
            }

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            plt = _prepare_matplotlib()
            exec_tree, last_expr = _split_last_expression(full_source)
            if exec_tree.body:
                compiled = compile(exec_tree, "<param-explore>", "exec")
                with (
                    contextlib.redirect_stdout(stdout_buf),
                    contextlib.redirect_stderr(stderr_buf),
                    warnings.catch_warnings(),
                ):
                    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
                    exec(compiled, ns_snap, ns_snap)  # noqa: S102
            if last_expr is not None:
                compiled_expr = compile(ast.Expression(last_expr), "<param-explore-expr>", "eval")
                with (
                    contextlib.redirect_stdout(stdout_buf),
                    contextlib.redirect_stderr(stderr_buf),
                    warnings.catch_warnings(),
                ):
                    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
                    value = eval(compiled_expr, ns_snap, ns_snap)  # noqa: S307
                out = _output_from_value(value)
                if out is not None:
                    result.outputs.append(out)
            result.outputs.extend(_collect_matplotlib_outputs(plt))
            if plt is not None:
                plt.close("all")
        except Exception:
            result.error = traceback.format_exc()

        result.stdout = stdout_buf.getvalue()
        result.stderr = stderr_buf.getvalue()
        if result.stdout:
            result.outputs.insert(0, ExecutionOutput(kind="stdout", data={"text": result.stdout.rstrip()}))
        if result.stderr:
            result.outputs.append(ExecutionOutput(kind="stderr", data={"text": result.stderr.rstrip()}))
        result.namespace_snapshot = ns_snap
        print(
            f"[debug][execution-engine] execute_with_overrides:done overrides={list(overrides.keys())} "
            f"outputs={len(result.outputs)} has_error={result.error is not None}",
            flush=True,
        )
        return result

    def get_namespace(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of the current execution namespace."""
        with self._lock:
            snapshot = dict(self.namespace)
        print(f"[debug][execution-engine] get_namespace count={len(snapshot)}", flush=True)
        return snapshot
