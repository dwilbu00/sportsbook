"""Guards for the parallel props-sweep merge/chunk primitives (pure functions — no
process spawn). The full serial-vs-parallel byte-identical parity was validated
end-to-end on the real mirror (12 players / focused grid → IDENTICAL)."""
import unittest

import backtest as bt


class ContiguousChunksTests(unittest.TestCase):
    def test_order_preserving_and_complete(self):
        items = list(range(10))
        chunks = bt._contiguous_chunks(items, 3)
        # contiguous + in order + covers everything exactly once
        self.assertEqual([x for c in chunks for x in c], items)
        self.assertTrue(all(c == list(range(c[0], c[0] + len(c))) for c in chunks))

    def test_more_workers_than_items(self):
        chunks = bt._contiguous_chunks([1, 2], 8)
        self.assertEqual(chunks, [[1], [2]])   # no empty chunks

    def test_single(self):
        self.assertEqual(bt._contiguous_chunks([5], 4), [[5]])


def _cell(errors, n, hits, decisive, safe=None, calib=None):
    return {"errors": list(errors), "n": n, "hits": hits, "decisive": decisive,
            "safe": safe or {}, "quantile": {}, "calib_obs": calib}


class MergePropsResultsTests(unittest.TestCase):
    def test_merge_concats_in_order_and_sums(self):
        p0 = {"vA": {"batter_hits": _cell([1.0, 2.0], 2, 1, 2,
                                          safe={0.5: {"hits": 1, "n": 2}},
                                          calib=[("x", 1, 0.5, 1, 0.6, "2025-04-01")])}}
        p1 = {"vA": {"batter_hits": _cell([3.0], 1, 0, 1,
                                          safe={0.5: {"hits": 1, "n": 1}},
                                          calib=[("y", 2, 1.5, 0, 0.4, "2025-04-02")])}}
        m = bt._merge_props_results([p0, p1])["vA"]["batter_hits"]
        self.assertEqual(m["errors"], [1.0, 2.0, 3.0])          # order preserved
        self.assertEqual((m["n"], m["hits"], m["decisive"]), (3, 1, 3))
        self.assertEqual(m["safe"][0.5], {"hits": 2, "n": 3})
        self.assertEqual([o[0] for o in m["calib_obs"]], ["x", "y"])

    def test_merge_none_calib(self):
        p0 = {"v": {"p": _cell([1.0], 1, 0, 1, calib=None)}}
        p1 = {"v": {"p": _cell([2.0], 1, 0, 1, calib=None)}}
        m = bt._merge_props_results([p0, p1])["v"]["p"]
        self.assertEqual(m["errors"], [1.0, 2.0])
        self.assertIsNone(m["calib_obs"])

    def test_merge_empty(self):
        self.assertEqual(bt._merge_props_results([]), {})
        self.assertEqual(bt._merge_props_results([None]), {})


if __name__ == "__main__":
    unittest.main()
