from __future__ import annotations

from pyside_app.execution_engine import detect_cell_parameters


source = """
save_id = 0
n_save = 12
Nx = 400
D = 0.1
dt = 0.0005
kappa = 1.0
E_alpha = -0.30
omega_phi = 0.1
"""

print("[verify][smart-sliders] source loaded", flush=True)
params = detect_cell_parameters(source)
print(f"[verify][smart-sliders] sliders={list(params)}", flush=True)
for name, spec in params.items():
    print(
        f"[verify][smart-sliders] {name}: value={spec['value']} min={spec['min']} "
        f"max={spec['max']} scale={spec['scale']} is_int={spec['is_int']}",
        flush=True,
    )
print("[verify][smart-sliders] PASS", flush=True)
