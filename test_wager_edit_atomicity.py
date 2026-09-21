"""Wager batch-edit atomicity regression (audit F22).

app.py can't be imported outside Streamlit (module-level side effects), so — like the
audit's supplemental contract — this AST-extracts _apply_wager_edits + its pure
helpers and runs them against a fake wagers backend.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_wager_edit_atomicity -v
"""
import ast
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parent


def _extract():
    names = {"_apply_wager_edits", "_wager_ids", "_coerce_int", "_coerce_float"}
    src = io.open(ROOT / "app.py", encoding="utf-8").read()
    parsed = ast.parse(src)
    mod = ast.Module(
        body=[n for n in parsed.body
              if isinstance(n, ast.FunctionDef) and n.name in names],
        type_ignores=[])
    ns = {"pd": pd, "st": MagicMock()}
    exec(compile(mod, "app.py:extracted", "exec"), ns)
    return ns


class F22BatchEditAtomicityTests(unittest.TestCase):
    def test_failed_update_does_not_leave_a_committed_deletion(self):
        ns = _extract()
        st = ns["st"]
        original = pd.DataFrame([
            {"Delete": False, "Price": -110, "Line": 4.5, "Stake": 10.},
            {"Delete": False, "Price": -110, "Line": 4.5, "Stake": 10.},
        ], index=["a", "b"])
        edited = original.copy()
        edited.loc["a", "Delete"] = True     # delete a
        edited.loc["b", "Stake"] = 20.       # edit b (this write will fail)
        surviving = {"a", "b"}

        def delete(ids):
            surviving.difference_update(ids)
            return len(ids)

        wagers = MagicMock()
        wagers.delete_wagers.side_effect = delete
        wagers.update_wagers.side_effect = OSError("synthetic second-write failure")
        with patch.dict(sys.modules, {"wagers": wagers}):
            ns["_apply_wager_edits"](original, edited, editable=True)
        # Non-destructive update is attempted first; its failure aborts BEFORE the
        # delete, so nothing was committed and the "unchanged" message is accurate.
        self.assertIn("unchanged", st.error.call_args[0][0])
        self.assertEqual(surviving, {"a", "b"})

    def test_partial_commit_is_reported_not_claimed_unchanged(self):
        ns = _extract()
        st = ns["st"]
        original = pd.DataFrame([
            {"Delete": False, "Price": -110, "Line": 4.5, "Stake": 10.},
            {"Delete": False, "Price": -110, "Line": 4.5, "Stake": 10.},
        ], index=["a", "b"])
        edited = original.copy()
        edited.loc["a", "Delete"] = True     # delete a (this write will fail)
        edited.loc["b", "Stake"] = 20.       # edit b (succeeds first)

        wagers = MagicMock()
        wagers.update_wagers.return_value = 1
        wagers.delete_wagers.side_effect = OSError("synthetic delete failure")
        with patch.dict(sys.modules, {"wagers": wagers}):
            ns["_apply_wager_edits"](original, edited, editable=True)
        msg = st.error.call_args[0][0]
        self.assertIn("Partially saved", msg)
        self.assertNotIn("unchanged", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
