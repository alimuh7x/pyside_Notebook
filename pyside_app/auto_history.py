from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import numpy as np


RECORDER_NAMESPACE_NAME = "__auto_history_recorder__"
GENERATED_HISTORY_NAMES = "__auto_history_generated_names__"
_TEMP_SUFFIXES = ("_new", "_tmp", "_temp", "_old", "_copy")


@dataclass
class _ArraySeries:
    frames: list[np.ndarray] = field(default_factory=list)
    changed: bool = False


class AutoHistoryRecorder:
    """Record numeric arrays that evolve inside instrumented range loops."""

    def __init__(self, max_saved_frames: int = 5, max_memory_mb: int = 256) -> None:
        self.max_saved_frames = max(1, int(max_saved_frames))
        self.max_memory_bytes = max(1, int(max_memory_mb)) * 1024 * 1024
        self._arrays: dict[str, _ArraySeries] = {}
        self._loop_values: dict[str, list[Any]] = {}
        print(
            f"[debug][auto-history] init max_saved_frames={self.max_saved_frames} "
            f"max_memory_mb={max_memory_mb}",
            flush=True,
        )

    def instrument(self, source: str) -> ast.Module:
        """Parse and instrument source code for auto-history sampling."""
        print("[debug][auto-history] scan:start", flush=True)
        tree = ast.parse(source)
        return self.instrument_tree(tree)

    def instrument_tree(self, tree: ast.Module) -> ast.Module:
        """Insert sampling calls after eligible loop bodies."""
        print("[debug][auto-history] instrument_tree:start", flush=True)
        transformed = _HistoryLoopTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        print("[debug][auto-history] instrument_tree:done", flush=True)
        return transformed

    def sample(self, loop_name: str, loop_value: Any, namespace: dict[str, Any]) -> None:
        """Capture current numeric arrays from one loop iteration."""
        self._loop_values.setdefault(loop_name, []).append(loop_value)
        print(f"[debug][auto-history] sample loop={loop_name!r} value={loop_value!r}", flush=True)
        for name, value in list(namespace.items()):
            if not self._should_consider_name(name):
                continue
            if not isinstance(value, np.ndarray):
                continue
            if value.ndim not in (1, 2) or value.size == 0:
                continue
            if not np.issubdtype(value.dtype, np.number):
                continue
            frame = np.asarray(value, dtype=float).copy()
            series = self._arrays.setdefault(name, _ArraySeries())
            if series.frames:
                previous = series.frames[-1]
                if previous.shape != frame.shape:
                    print(
                        f"[debug][auto-history] drop shape_changed name={name!r} "
                        f"old={previous.shape} new={frame.shape}",
                        flush=True,
                    )
                    self._arrays.pop(name, None)
                    continue
                if not np.array_equal(previous, frame):
                    series.changed = True
            series.frames.append(frame)
            print(f"[debug][auto-history] captured name={name!r} shape={frame.shape}", flush=True)

    def finalize(self, namespace: dict[str, Any]) -> dict[str, np.ndarray]:
        """Create history arrays in namespace and return the generated arrays."""
        print("[debug][auto-history] finalize:start", flush=True)
        generated: dict[str, np.ndarray] = {}
        primary_loop_name, loop_values = self._primary_loop()
        keep_indices_by_length: dict[int, np.ndarray] = {}

        for name, series in sorted(self._arrays.items()):
            if not series.changed or len(series.frames) < 2:
                print(f"[debug][auto-history] skip unchanged name={name!r}", flush=True)
                continue
            frame_count = len(series.frames)
            keep_indices = keep_indices_by_length.get(frame_count)
            if keep_indices is None:
                keep_indices = self._keep_indices(series.frames)
                keep_indices_by_length[frame_count] = keep_indices
            history = np.stack([series.frames[int(i)] for i in keep_indices], axis=0)
            history_name = self._history_name(name, namespace)
            namespace[history_name] = history
            self._mark_generated(namespace, history_name)
            generated[history_name] = history
            print(
                f"[debug][auto-history] created name={history_name!r} source={name!r} "
                f"shape={history.shape}",
                flush=True,
            )

        if generated and primary_loop_name and loop_values:
            sample_count = next(iter(generated.values())).shape[0]
            keep_indices = self._matching_keep_indices(len(loop_values), sample_count)
            sampled_loop_values = np.asarray([loop_values[int(i)] for i in keep_indices], dtype=float)
            loop_history_name = self._history_name(primary_loop_name, namespace)
            namespace[loop_history_name] = sampled_loop_values
            self._mark_generated(namespace, loop_history_name)
            generated[loop_history_name] = sampled_loop_values
            print(
                f"[debug][auto-history] created name={loop_history_name!r} "
                f"shape={sampled_loop_values.shape}",
                flush=True,
            )
            if "dt" in namespace:
                try:
                    time_values = sampled_loop_values * float(namespace["dt"])
                    time_name = self._history_name("time", namespace)
                    namespace[time_name] = time_values
                    self._mark_generated(namespace, time_name)
                    generated[time_name] = time_values
                    print(f"[debug][auto-history] created name={time_name!r} shape={time_values.shape}", flush=True)
                except (TypeError, ValueError):
                    print("[debug][auto-history] time_history skipped invalid_dt", flush=True)

        print(f"[debug][auto-history] finalize:done generated={list(generated)}", flush=True)
        return generated

    @staticmethod
    def clear_generated(namespace: dict[str, Any]) -> None:
        """Remove histories generated by a previous execution from a namespace."""
        names = namespace.get(GENERATED_HISTORY_NAMES)
        if not isinstance(names, set):
            return
        print(f"[debug][auto-history] clear_generated names={sorted(names)!r}", flush=True)
        for name in list(names):
            namespace.pop(name, None)
        names.clear()

    @staticmethod
    def _mark_generated(namespace: dict[str, Any], name: str) -> None:
        names = namespace.setdefault(GENERATED_HISTORY_NAMES, set())
        if isinstance(names, set):
            names.add(name)

    def _primary_loop(self) -> tuple[str, list[Any]]:
        if not self._loop_values:
            return "", []
        name = max(self._loop_values, key=lambda key: len(self._loop_values[key]))
        return name, self._loop_values[name]

    @staticmethod
    def _should_consider_name(name: str) -> bool:
        return (
            bool(name)
            and not name.startswith("_")
            and not name.endswith("_history")
            and not name.endswith("_auto_history")
            and not name.endswith(_TEMP_SUFFIXES)
        )

    @staticmethod
    def _history_name(name: str, namespace: dict[str, Any]) -> str:
        preferred = f"{name}_history"
        if preferred not in namespace:
            return preferred
        fallback = f"{name}_auto_history"
        if fallback not in namespace:
            return fallback
        suffix = 2
        while f"{fallback}_{suffix}" in namespace:
            suffix += 1
        return f"{fallback}_{suffix}"

    def _keep_indices(self, frames: list[np.ndarray]) -> np.ndarray:
        frame_count = len(frames)
        bytes_per_frame = max(1, int(frames[0].nbytes))
        memory_limited_frames = max(1, self.max_memory_bytes // bytes_per_frame)
        target_count = min(frame_count, self.max_saved_frames, memory_limited_frames)
        if target_count < frame_count:
            print(
                f"[debug][auto-history] downsample frames={frame_count} target={target_count} "
                f"bytes_per_frame={bytes_per_frame}",
                flush=True,
            )
        return self._matching_keep_indices(frame_count, target_count)

    @staticmethod
    def _matching_keep_indices(frame_count: int, target_count: int) -> np.ndarray:
        if target_count >= frame_count:
            return np.arange(frame_count, dtype=int)
        if target_count <= 1:
            return np.asarray([frame_count - 1], dtype=int)
        return np.unique(np.round(np.linspace(0, frame_count - 1, target_count)).astype(int))


class _HistoryLoopTransformer(ast.NodeTransformer):
    """Insert recorder sample calls after loops that assign named variables."""

    def visit_For(self, node: ast.For) -> ast.AST:
        node = self.generic_visit(node)
        if not self._is_range_loop(node):
            return node
        if not isinstance(node.target, ast.Name):
            return node
        assigned_names = self._assigned_names(node)
        print(
            f"[debug][auto-history] loop_found target={node.target.id!r} "
            f"assigned={sorted(assigned_names)!r}",
            flush=True,
        )
        if not assigned_names:
            return node
        node.body.append(self._sample_expr(node.target.id))
        return node

    @staticmethod
    def _is_range_loop(node: ast.For) -> bool:
        call = node.iter
        return (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
        )

    @staticmethod
    def _assigned_names(node: ast.For) -> set[str]:
        names: set[str] = set()
        for child in node.body:
            for subnode in ast.walk(child):
                if isinstance(subnode, ast.Assign):
                    for target in subnode.targets:
                        if isinstance(target, ast.Name) and AutoHistoryRecorder._should_consider_name(target.id):
                            names.add(target.id)
                elif isinstance(subnode, ast.AnnAssign):
                    target = subnode.target
                    if isinstance(target, ast.Name) and AutoHistoryRecorder._should_consider_name(target.id):
                        names.add(target.id)
                elif isinstance(subnode, ast.AugAssign):
                    target = subnode.target
                    if isinstance(target, ast.Name) and AutoHistoryRecorder._should_consider_name(target.id):
                        names.add(target.id)
        return names

    @staticmethod
    def _sample_expr(loop_name: str) -> ast.Expr:
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=RECORDER_NAMESPACE_NAME, ctx=ast.Load()),
                    attr="sample",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Constant(loop_name),
                    ast.Name(id=loop_name, ctx=ast.Load()),
                    ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
                ],
                keywords=[],
            )
        )
