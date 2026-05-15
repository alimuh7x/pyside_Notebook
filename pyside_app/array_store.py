"""Temp-file-based array run history.

Each execution run saves all numeric numpy arrays to a temp directory.
Arrays are stored as .npy files inside per-run subdirectories.
The store is cleared on kernel restart.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any

import numpy as np

_store_dir: str = ""
_run_counter: int = 0


def _get_dir() -> str:
    global _store_dir
    if not _store_dir or not os.path.isdir(_store_dir):
        _store_dir = tempfile.mkdtemp(prefix="nb_arrays_")
    return _store_dir


def get_store_dir() -> str:
    """Return the current temp directory where array files are stored."""
    return _get_dir()


def store_run(namespace: dict[str, Any], label: str = "") -> int:
    """Persist all 1-D/2-D numeric arrays from namespace. Returns run number."""
    global _run_counter
    _run_counter += 1
    run_num = _run_counter
    run_dir = os.path.join(_get_dir(), f"run_{run_num:04d}")
    os.makedirs(run_dir, exist_ok=True)
    for name, value in namespace.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim not in (1, 2) or value.size == 0:
            continue
        if not np.issubdtype(value.dtype, np.number):
            continue
        try:
            np.save(os.path.join(run_dir, f"{name}.npy"), value.astype(float))
        except Exception:
            pass
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({"label": label, "run": run_num}, f)
    return run_num


def get_history(array_name: str) -> list[tuple[str, np.ndarray]]:
    """Return [(label, array), ...] for array_name, newest run first."""
    base = _get_dir()
    try:
        run_dirs = sorted(
            [d for d in os.listdir(base) if d.startswith("run_")],
            reverse=True,
        )
    except OSError:
        return []
    results: list[tuple[str, np.ndarray]] = []
    for run_dir_name in run_dirs:
        run_dir = os.path.join(base, run_dir_name)
        arr_path = os.path.join(run_dir, f"{array_name}.npy")
        if not os.path.exists(arr_path):
            continue
        try:
            arr = np.load(arr_path)
        except Exception:
            continue
        label = run_dir_name
        meta_path = os.path.join(run_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                label = meta.get("label", "") or run_dir_name
            except Exception:
                pass
        results.append((label, arr))
    return results


def get_run_snapshots(array_name: str) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Return saved run snapshots that contain array_name, newest run first."""
    base = _get_dir()
    try:
        run_dirs = sorted(
            [d for d in os.listdir(base) if d.startswith("run_")],
            reverse=True,
        )
    except OSError:
        return []
    results: list[tuple[str, dict[str, np.ndarray]]] = []
    for run_dir_name in run_dirs:
        run_dir = os.path.join(base, run_dir_name)
        if not os.path.exists(os.path.join(run_dir, f"{array_name}.npy")):
            continue
        arrays: dict[str, np.ndarray] = {}
        for file_name in os.listdir(run_dir):
            if not file_name.endswith(".npy"):
                continue
            var_name = file_name[:-4]
            try:
                arr = np.load(os.path.join(run_dir, file_name))
            except Exception:
                continue
            if arr.ndim in (1, 2):
                arrays[var_name] = arr
        if array_name not in arrays:
            continue
        label = run_dir_name
        meta_path = os.path.join(run_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                label = meta.get("label", "") or run_dir_name
            except Exception:
                pass
        results.append((label, arrays))
    return results


def clear() -> None:
    """Delete all stored runs. Call on kernel restart."""
    global _store_dir, _run_counter
    if _store_dir and os.path.isdir(_store_dir):
        shutil.rmtree(_store_dir, ignore_errors=True)
    _store_dir = tempfile.mkdtemp(prefix="nb_arrays_")
    _run_counter = 0
