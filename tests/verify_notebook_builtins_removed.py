from __future__ import annotations

from utils.notebook_eval import ALLOWED_NOTEBOOK_FUNCTIONS, evaluate_notebook_rows


def _rows(*lines: str) -> list[dict]:
    return [{"id": f"line_{i}", "expression": line} for i, line in enumerate(lines)]


print(f"[verify][notebook-builtins] has_abs={'abs' in ALLOWED_NOTEBOOK_FUNCTIONS}", flush=True)
print(f"[verify][notebook-builtins] has_sum={'sum' in ALLOWED_NOTEBOOK_FUNCTIONS}", flush=True)
print(f"[verify][notebook-builtins] has_min={'min' in ALLOWED_NOTEBOOK_FUNCTIONS}", flush=True)
print(f"[verify][notebook-builtins] has_max={'max' in ALLOWED_NOTEBOOK_FUNCTIONS}", flush=True)
print(f"[verify][notebook-builtins] has_round={'round' in ALLOWED_NOTEBOOK_FUNCTIONS}", flush=True)
print(f"[verify][notebook-builtins] has_range={'range' in ALLOWED_NOTEBOOK_FUNCTIONS}", flush=True)

evaluated_rows, _variables, _arrays = evaluate_notebook_rows(_rows("abs(-1)", "sum([1, 2, 3])", "[i*i for i in range(4)]"))
print(f"[verify][notebook-builtins] abs_error={evaluated_rows[0].get('error')!r}", flush=True)
print(f"[verify][notebook-builtins] sum_error={evaluated_rows[1].get('error')!r}", flush=True)
print(f"[verify][notebook-builtins] range_result={evaluated_rows[2].get('result')!r}", flush=True)
