"""nfl_data layer-cache regression (audit F24).

Run: PYTHONIOENCODING=utf-8 python -m unittest test_nfl_data -v
"""
import unittest
from unittest.mock import patch

import pandas as pd

import nfl_data


class F24LayerCacheFreshnessTests(unittest.TestCase):
    def test_absent_layer_is_not_cached_for_process_lifetime(self):
        # First read: file absent -> None (must NOT be cached). Second read after the
        # file becomes real -> the DataFrame, not a stale cached absence.
        with patch.object(nfl_data, "_CACHE", {}), \
             patch.object(nfl_data, "_is_real", side_effect=[False, True]), \
             patch("pandas.read_parquet", return_value=pd.DataFrame({"fixture": [1]})):
            self.assertIsNone(nfl_data._read_layer("audit_fixture", 2026))
            self.assertIsNotNone(nfl_data._read_layer("audit_fixture", 2026))

    def test_real_layer_is_cached(self):
        # A successfully-read layer is cached (second call does not re-stat/read).
        with patch.object(nfl_data, "_CACHE", {}), \
             patch.object(nfl_data, "_is_real", side_effect=[True]) as is_real, \
             patch("pandas.read_parquet", return_value=pd.DataFrame({"x": [1]})) as rp:
            a = nfl_data._read_layer("audit_fixture", 2026)
            b = nfl_data._read_layer("audit_fixture", 2026)
        self.assertIsNotNone(a)
        self.assertIs(a, b)
        self.assertEqual(is_real.call_count, 1)   # cached: not re-checked
        self.assertEqual(rp.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
