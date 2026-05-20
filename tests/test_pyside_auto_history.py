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


def test_execution_engine_execute_adds_auto_history_to_namespace():
    engine = ExecutionEngine()

    result = engine.execute(DIFFUSION_SOURCE)
    namespace = engine.get_namespace()

    assert result.error is None
    assert namespace["u_history"].shape == (5, 50)
    assert namespace["time_history"].shape == (5,)


def test_execution_engine_override_run_keeps_auto_history_in_snapshot_only():
    engine = ExecutionEngine()
    baseline = engine.execute(DIFFUSION_SOURCE)

    result = engine.execute_with_overrides(DIFFUSION_SOURCE, {"D": 0.2})
    shared_namespace = engine.get_namespace()

    assert baseline.error is None
    assert result.error is None
    assert result.namespace_snapshot is not None
    assert result.namespace_snapshot["D"] == 0.2
    assert result.namespace_snapshot["u_history"].shape == (5, 50)
    assert shared_namespace["D"] == 0.1
    assert not np.allclose(result.namespace_snapshot["u_history"], shared_namespace["u_history"])


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
