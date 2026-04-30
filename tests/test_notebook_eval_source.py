"""
Source-level regression tests for notebook evaluator conflicts.
"""
from __future__ import annotations

import pathlib
import unittest


NOTEBOOK_EVAL = pathlib.Path(__file__).resolve().parents[1] / "utils" / "notebook_eval.py"


class NotebookEvalSourceTests(unittest.TestCase):
    def test_min_is_not_wrapped_as_callable_function(self):
        source = NOTEBOOK_EVAL.read_text(encoding="utf-8")

        self.assertIn("class _CallableFloat(float):", source)
        self.assertNotIn('"min": _CallableFloat(UNIT_CONSTANTS["min"], _min_arr)', source)

    def test_notebook_eval_allows_exact_copy_method_only(self):
        source = NOTEBOOK_EVAL.read_text(encoding="utf-8")

        self.assertIn("def _is_allowed_method_call(node: ast.Call, allowed_names: set[str]) -> bool:", source)
        self.assertIn('if node.func.attr != "copy":', source)
        self.assertIn('return isinstance(base, ast.Name) and base.id in allowed_names', source)


if __name__ == "__main__":
    unittest.main()
