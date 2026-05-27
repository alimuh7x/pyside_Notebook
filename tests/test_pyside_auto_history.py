from __future__ import annotations

import numpy as np

from pyside_app.auto_history import AutoHistoryRecorder
from pyside_app.auto_history import RECORDER_NAMESPACE_NAME
from pyside_app import array_store
from pyside_app.execution_engine import ExecutionEngine


DIFFUSION_SOURCE = """
import numpy as np
L = 1.0
Nx = 50
dx = L / (Nx - 1)
D = 0.1
dt = 0.0005
Nt = 20
alpha = D * dt / dx**2
x = np.linspace(0, L, Nx)
u = np.exp(-100 * (x - 0.5)**2)
for n in range(Nt):
    u_new = u.copy()
    for i in range(1, Nx - 1):
        u_new[i] = u[i] + alpha * (u[i+1] - 2*u[i] + u[i-1])
    u_new[0] = 0
    u_new[-1] = 0
    u = u_new
"""


FUNCTION_CALL_SOURCE = """
import numpy as np

def solve_wave():
    Nx = 30
    Nt = 12
    dt = 0.01
    x = np.linspace(0.0, 1.0, Nx)
    u = np.exp(-100 * (x - 0.5) ** 2)
    for n in range(Nt):
        u = np.sin(np.pi * x) * np.cos(n * dt)

solve_wave()
"""


def test_auto_history_recorder_creates_1d_history_from_time_loop():
    namespace: dict[str, object] = {"np": np}
    recorder = AutoHistoryRecorder(max_saved_frames=20)
    namespace[RECORDER_NAMESPACE_NAME] = recorder
    tree = recorder.instrument(DIFFUSION_SOURCE)

    exec(compile(tree, "<test-auto-history>", "exec"), namespace, namespace)
    recorder.finalize(namespace)

    assert "u_history" in namespace
    assert namespace["u_history"].shape == (20, 50)
    assert namespace["n_history"].shape == (20,)
    assert namespace["time_history"].shape == (20,)
    assert "u_new_history" not in namespace
    assert "x_history" not in namespace


def test_auto_history_recorder_creates_3d_history_for_2d_field():
    namespace: dict[str, object] = {"np": np}
    recorder = AutoHistoryRecorder(max_saved_frames=6)
    namespace[RECORDER_NAMESPACE_NAME] = recorder
    tree = recorder.instrument(
        """
import numpy as np
Nt = 6
u = np.zeros((4, 5))
for n in range(Nt):
    u_new = u + n + 1
    u = u_new
"""
    )

    exec(compile(tree, "<test-auto-history-2d>", "exec"), namespace, namespace)
    recorder.finalize(namespace)

    assert namespace["u_history"].shape == (6, 4, 5)
    assert np.allclose(namespace["u_history"][-1], namespace["u"])


def test_auto_history_recorder_does_not_print_sampling_debugs(capsys):
    namespace: dict[str, object] = {"np": np}
    recorder = AutoHistoryRecorder(max_saved_frames=6)
    namespace[RECORDER_NAMESPACE_NAME] = recorder
    tree = recorder.instrument(
        """
import numpy as np
Nt = 4
u = np.zeros((2, 3))
for n in range(Nt):
    u = u + n + 1
"""
    )

    exec(compile(tree, "<test-auto-history-silent>", "exec"), namespace, namespace)
    recorder.finalize(namespace)

    captured = capsys.readouterr()
    assert "[debug][auto-history]" not in captured.out
    assert captured.err == ""


def test_execution_engine_execute_adds_auto_history_to_namespace():
    engine = ExecutionEngine()

    result = engine.execute(DIFFUSION_SOURCE)
    namespace = engine.get_namespace()

    assert result.error is None
    assert namespace["u_history"].shape == (20, 50)
    assert namespace["time_history"].shape == (20,)


def test_execution_engine_auto_history_works_inside_final_function_call():
    engine = ExecutionEngine()

    result = engine.execute(FUNCTION_CALL_SOURCE)
    namespace = engine.get_namespace()

    assert result.error is None
    assert namespace["u"].shape == (30,)
    assert namespace["u_history"].shape == (12, 30)
    assert namespace["time_history"].shape == (12,)
    assert namespace["time_history"][-1] == 0.11


def test_execution_engine_override_run_keeps_auto_history_in_snapshot_only():
    engine = ExecutionEngine()
    baseline = engine.execute(DIFFUSION_SOURCE)

    result = engine.execute_with_overrides(DIFFUSION_SOURCE, {"D": 0.2})
    shared_namespace = engine.get_namespace()

    assert baseline.error is None
    assert result.error is None
    assert result.namespace_snapshot is not None
    assert result.namespace_snapshot["D"] == 0.2
    assert result.namespace_snapshot["u_history"].shape == (20, 50)
    assert shared_namespace["D"] == 0.1
    assert not np.allclose(result.namespace_snapshot["u_history"], shared_namespace["u_history"])


def test_execution_engine_override_auto_history_works_inside_final_function_call():
    engine = ExecutionEngine()

    result = engine.execute_with_overrides(FUNCTION_CALL_SOURCE, {"unused": 1.0})

    assert result.error is None
    assert result.namespace_snapshot is not None
    assert result.namespace_snapshot["u"].shape == (30,)
    assert result.namespace_snapshot["u_history"].shape == (12, 30)
    assert result.namespace_snapshot["time_history"].shape == (12,)


def test_auto_history_exposes_final_local_2d_field_for_static_heatmap():
    engine = ExecutionEngine()

    result = engine.execute(
        """
import numpy as np

def solve_field():
    u = np.zeros((3, 4))
    for n in range(5):
        u = np.full((3, 4), float(n))

solve_field()
"""
    )
    namespace = engine.get_namespace()

    assert result.error is None
    assert namespace["u"].shape == (3, 4)
    assert np.allclose(namespace["u"], 4.0)
    assert namespace["u_history"].shape == (5, 3, 4)


def test_execution_engine_auto_history_keeps_100_frames_for_long_time_series():
    engine = ExecutionEngine()
    source = DIFFUSION_SOURCE.replace("Nt = 20", "Nt = 2000")

    result = engine.execute(source)
    namespace = engine.get_namespace()

    assert result.error is None
    assert namespace["u_history"].shape == (100, 50)
    assert namespace["time_history"].shape == (100,)
    assert namespace["n_history"][0] == 0
    assert namespace["n_history"][-1] == 1999


def test_execution_engine_repeated_runs_replace_generated_history_names():
    engine = ExecutionEngine()

    first = engine.execute(DIFFUSION_SOURCE)
    second = engine.execute(DIFFUSION_SOURCE.replace("D = 0.1", "D = 0.2"))
    namespace = engine.get_namespace()

    assert first.error is None
    assert second.error is None
    assert "u_history" in namespace
    assert "u_auto_history" not in namespace
    assert namespace["D"] == 0.2


def test_array_store_saves_auto_history_for_2d_field_parameter_runs():
    array_store.clear()
    field_history = np.arange(24.0).reshape(2, 3, 4)

    array_store.store_run({"u_history": field_history}, label="D=0.2")
    snapshots = array_store.get_run_snapshots("u_history")

    assert snapshots
    assert snapshots[0][0] == "D=0.2"
    assert snapshots[0][1]["u_history"].shape == (2, 3, 4)
