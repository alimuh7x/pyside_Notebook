from __future__ import annotations

import numpy as np

from utils.notebook_eval import evaluate_notebook_rows


def rows(*lines: str) -> list[dict]:
    return [{"id": f"line_{i}", "expression": line} for i, line in enumerate(lines)]


def main() -> None:
    print("[debug][notebook-copy] starting vector copy verification")
    vector_lines = rows(
        "a = [1, 2, 3]",
        "b = a.copy()",
        "b[0] = 99",
    )
    evaluated, scalars, arrays = evaluate_notebook_rows(vector_lines)
    print(f"[debug][notebook-copy] vector evaluated rows: {evaluated}")
    print(f"[debug][notebook-copy] vector scalars: {scalars}")
    print(f"[debug][notebook-copy] vector arrays: {arrays}")
    assert evaluated[1]["error"] == ""
    assert np.allclose(arrays["a"], [1, 2, 3])
    assert np.allclose(arrays["b"], [99, 2, 3])
    print("[debug][notebook-copy] vector copy verification passed")

    print("[debug][notebook-copy] starting matrix copy verification")
    matrix_lines = rows(
        "A = eye(2)",
        "B = A.copy()",
        "B[0][0] = 5",
        "A",
        "B",
    )
    evaluated, scalars, arrays = evaluate_notebook_rows(matrix_lines)
    print(f"[debug][notebook-copy] matrix evaluated rows: {evaluated}")
    print(f"[debug][notebook-copy] matrix scalars: {scalars}")
    print(f"[debug][notebook-copy] matrix arrays: {arrays}")
    assert evaluated[1]["error"] == ""
    assert evaluated[3]["result"] == "[[1, 0], [0, 1]]"
    assert evaluated[4]["result"] == "[[5, 0], [0, 1]]"
    print("[debug][notebook-copy] matrix copy verification passed")

    print("[debug][notebook-copy] all copy verifications complete")


if __name__ == "__main__":
    main()
