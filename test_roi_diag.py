"""Hermetic tests for the --roi-diag profitability lens in refit_calibration.py.

--roi-diag is a NO-WRITE diagnostic that replays each calibration method (A/B/C/D/E)
through the LIVE edge+EV recommendation gate at best-of-book / de-vigged consensus
prices and reports realized flat-1u ROI ALONGSIDE holdout Brier — so a method that
narrowly fails the 0.002 Brier gate but lifts betting ROI becomes visible.

Three layers, all pure stdlib (no live ESPN / Statcast / SQL / secrets):
  * _roi_sim_method  — the load-bearing gate replay: edge>=threshold AND +EV, side
    pick, flat-1u payoff. The negative-EV case proves the EV leg bites (the key
    difference vs _cc_stratum_table, which backs any +edge side).
  * _roi_build_rows  — prop-generic priced-row builder (de-vig + decimals), with
    blc.project_and_empirical patched.
  * diagnose_roi     — end-to-end, hermetic (mirrors DiagnoseNegbinCaveatTests):
    caveats, per-method table, incumbent tag, xBA caveat, thin-prop skip, no write.

Run: PYTHONIOENCODING=utf-8 python test_roi_diag.py
"""

import io
import re
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import book_line_calibration as blc
import refit_calibration as rc
from odds_client import american_to_decimal


# ── _roi_sim_method: the live edge+EV gate replay ────────────────────────────
def _srow(o, p, mkt_over, over_am=100, under_am=100):
    """A priced sim row: outcome ``o`` (1=over hit), model P(over) ``p``, de-vigged
    consensus ``mkt_over``, and both american prices with their decimal payouts (as
    _roi_build_rows would carry them)."""
    return {
        "o": o, "p": p, "mkt_over": mkt_over,
        "over_price": over_am, "under_price": under_am,
        "over_dec": american_to_decimal(over_am),
        "under_dec": american_to_decimal(under_am),
    }


def _P(r):
    return r["p"]


class RoiSimMethodTests(unittest.TestCase):
    def test_over_side_value_bet_roi_and_hit(self):
        # p=0.70 vs mkt_over=0.55 -> back the OVER, edge +0.15 (>=0.05), +100 (dec
        # 2.0) so EV = 0.70*2.0-1 = +0.40 > 0 -> every row is a bet. 3 overs win /
        # 2 lose: pnl = 3*(+1) + 2*(-1) = +1.0, roi = +0.20, hit = 0.60.
        rows = [_srow(o, 0.70, 0.55) for o in (1, 1, 1, 0, 0)]
        s = rc._roi_sim_method(rows, _P, 0.05)
        self.assertEqual(s["n_bets"], 5)
        self.assertAlmostEqual(s["pnl"], 1.0, places=9)
        self.assertAlmostEqual(s["roi"], 0.20, places=9)
        self.assertAlmostEqual(s["hit"], 0.60, places=9)
        self.assertAlmostEqual(s["avg_edge"], 0.15, places=9)

    def test_under_side_value_bet(self):
        # p=0.30 vs mkt_over=0.55 -> over_edge -0.25 <= 0 -> back the UNDER, edge
        # +0.25, side_prob 0.70, under +100 (dec 2.0) -> EV +0.40 -> bet. The under
        # wins when o==0: 3 wins / 1 loss -> pnl = +2.0, roi +0.50, hit 0.75.
        rows = [_srow(o, 0.30, 0.55) for o in (0, 0, 0, 1)]
        s = rc._roi_sim_method(rows, _P, 0.05)
        self.assertEqual(s["n_bets"], 4)
        self.assertAlmostEqual(s["pnl"], 2.0, places=9)
        self.assertAlmostEqual(s["roi"], 0.50, places=9)
        self.assertAlmostEqual(s["hit"], 0.75, places=9)
        self.assertAlmostEqual(s["avg_edge"], 0.25, places=9)

    def test_ties_break_to_under(self):
        # p exactly == mkt_over -> over_edge 0.0, NOT > 0 -> the UNDER branch (edge
        # 0.0), matching props/_cc_stratum_table's sign convention. Edge 0 < any
        # positive threshold, so it never actually bets -- but the branch is taken.
        rows = [_srow(0, 0.50, 0.50)]
        s = rc._roi_sim_method(rows, _P, 0.05)
        self.assertEqual(s["n_bets"], 0)

    def test_below_threshold_no_bet(self):
        # edge +0.02 < threshold 0.05 -> skipped even though EV would be positive.
        rows = [_srow(o, 0.57, 0.55) for o in (1, 0, 1)]
        s = rc._roi_sim_method(rows, _P, 0.05)
        self.assertEqual(s["n_bets"], 0)
        self.assertIsNone(s["roi"])
        self.assertIsNone(s["hit"])
        self.assertIsNone(s["avg_edge"])
        self.assertEqual(s["pnl"], 0.0)

    def test_negative_ev_no_bet_even_with_edge(self):
        # THE key case: edge clears threshold (+0.10 >= 0.05) but the price is bad.
        # Over at -300 (dec 1.3333): EV = 0.65*1.3333 - 1 = -0.133 < 0 -> the EV leg
        # of _prop_is_value REJECTS it. An edge-only gate (_cc_stratum_table) would
        # have taken this bet; the recommendation gate does not.
        rows = [_srow(o, 0.65, 0.55, over_am=-300) for o in (1, 1, 0)]
        s = rc._roi_sim_method(rows, _P, 0.05)
        self.assertEqual(s["n_bets"], 0)
        # sanity: the SAME edge at a +EV price DOES bet (isolates the EV leg).
        good = [_srow(o, 0.65, 0.55, over_am=100) for o in (1, 1, 0)]
        self.assertEqual(rc._roi_sim_method(good, _P, 0.05)["n_bets"], 3)

    def test_unpriced_rows_skipped(self):
        rows = [
            {"o": 1, "p": 0.70, "mkt_over": None,
             "over_price": None, "under_price": None,
             "over_dec": None, "under_dec": None},
            _srow(1, 0.70, 0.55, over_am=100),          # priced, would bet
            {"o": 1, "p": 0.70, "mkt_over": 0.55,        # priced-prob but no decimals
             "over_price": 100, "under_price": 100,
             "over_dec": None, "under_dec": None},
        ]
        s = rc._roi_sim_method(rows, _P, 0.05)
        self.assertEqual(s["n_bets"], 1)                 # only the fully-priced row

    def test_flat_unit_payoff_sign_at_nonzero_price(self):
        # +150 (dec 2.5): a win pays +1.5 units, a loss -1.0 (flat 1u convention).
        win = rc._roi_sim_method([_srow(1, 0.70, 0.55, over_am=150)], _P, 0.05)
        self.assertAlmostEqual(win["pnl"], 1.5, places=9)
        loss = rc._roi_sim_method([_srow(0, 0.70, 0.55, over_am=150)], _P, 0.05)
        self.assertAlmostEqual(loss["pnl"], -1.0, places=9)

    def test_empty_returns_none_metrics(self):
        s = rc._roi_sim_method([], _P, 0.05)
        self.assertEqual(s["n_bets"], 0)
        self.assertEqual(s["pnl"], 0.0)
        self.assertIsNone(s["roi"])


# ── _roi_build_rows: priced-row construction ─────────────────────────────────
def _bobs(prop_key, over_price=-110, under_price=-110, line=1.5, actual=2):
    return {"prop_key": prop_key, "game_date": "2026-07-01",
            "line": line, "actual": actual,
            "over_price": over_price, "under_price": under_price}


class RoiBuildRowsTests(unittest.TestCase):
    def _build(self, enriched, proj_emp=(1.4, 0.6), prop_key="batter_hits"):
        with patch.object(blc, "project_and_empirical",
                          side_effect=lambda *a, **k: proj_emp):
            return rc._roi_build_rows(enriched, {}, "baseball_mlb", prop_key)

    def test_devig_and_decimals_for_priced_row(self):
        rows = self._build([_bobs("batter_hits", -110, -110)])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertAlmostEqual(r["mkt_over"], 0.5, places=9)       # two -110s de-vig
        self.assertAlmostEqual(r["over_dec"], american_to_decimal(-110), places=9)
        self.assertAlmostEqual(r["under_dec"], american_to_decimal(-110), places=9)
        self.assertEqual(r["over_price"], -110)
        self.assertEqual(r["under_price"], -110)
        self.assertAlmostEqual(r["projected"], 1.4, places=9)
        self.assertAlmostEqual(r["empirical_over"], 0.6, places=9)
        self.assertEqual(r["line"], 1.5)
        self.assertEqual(r["actual"], 2)
        self.assertEqual(r["game_date"], "2026-07-01")

    def test_asymmetric_prices_devig(self):
        # -140 over / +120 under: fair over prob > 0.5. Raw implieds 0.5833 / 0.4545
        # sum 1.0379; de-vigged over = 0.5833/1.0379 ~ 0.562.
        rows = self._build([_bobs("batter_hits", -140, 120)])
        self.assertGreater(rows[0]["mkt_over"], 0.5)
        self.assertAlmostEqual(rows[0]["mkt_over"], 0.5833 / 1.03788, places=3)

    def test_unpriced_row_kept_with_none_market(self):
        rows = self._build([_bobs("batter_hits", None, None)])
        self.assertEqual(len(rows), 1)                             # kept, not dropped
        r = rows[0]
        self.assertIsNone(r["mkt_over"])
        self.assertIsNone(r["over_dec"])
        self.assertIsNone(r["under_dec"])

    def test_one_sided_price_is_unpriced(self):
        # Only an over price -> can't de-vig a two-way market -> mkt_over None.
        rows = self._build([_bobs("batter_hits", -110, None)])
        self.assertIsNone(rows[0]["mkt_over"])

    def test_other_prop_excluded(self):
        rows = self._build([_bobs("batter_hits"), _bobs("pitcher_outs")],
                           prop_key="batter_hits")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["obs"]["prop_key"], "batter_hits")

    def test_empirical_clamped(self):
        hi = self._build([_bobs("batter_hits")], proj_emp=(1.4, 1.3))
        self.assertEqual(hi[0]["empirical_over"], 1.0)
        lo = self._build([_bobs("batter_hits")], proj_emp=(1.4, -0.2))
        self.assertEqual(lo[0]["empirical_over"], 0.0)

    def test_none_projection_dropped(self):
        rows = self._build([_bobs("batter_hits")], proj_emp=(None, None))
        self.assertEqual(rows, [])

    def test_carries_raw_obs(self):
        obs = _bobs("batter_hits")
        rows = self._build([obs])
        self.assertIs(rows[0]["obs"], obs)                         # method D reads it


# ── diagnose_roi: end-to-end, hermetic ───────────────────────────────────────
def _eobs(prop_key, i, line, proj, actual, emp=0.5, priced=True):
    """A synthetic enriched observation. game_date is derived from ``i`` so the
    chronological 50/50 split is deterministic; prices are -110/-110 when priced."""
    return {
        "prop_key": prop_key,
        "game_date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}",
        "line": line, "actual": actual, "_proj": proj, "_emp": emp,
        "over_price": (-110 if priced else None),
        "under_price": (-110 if priced else None),
    }


def _strikeout_obs(n=50):
    # line 5.5 -> integer actuals never push; projected/actual vary so the pooled
    # residual has a real sigma and the NegBin scale is non-degenerate.
    return [_eobs("pitcher_strikeouts", i, 5.5,
                  proj=5.0 + (i % 3) * 0.5, actual=4 + (i % 4)) for i in range(n)]


def _hits_obs(n=50):
    return [_eobs("batter_hits", i, 1.5,
                  proj=1.2 + (i % 3) * 0.2, actual=1 + (i % 2)) for i in range(n)]


class DiagnoseRoiEndToEndTests(unittest.TestCase):
    def _run(self, existing, enriched, xstats_strength=0.0):
        """Drive diagnose_roi through hermetic patches; return (stdout, save_mock).
        project_and_empirical is fed each obs's stashed (_proj, _emp);
        project_distributional is neutralized (method D falls away) so the table is
        the leakage-safe A/B/C(+E) set."""
        buf = io.StringIO()
        with patch.object(rc, "load_calibration", return_value=existing), \
             patch.object(rc, "save_calibration") as save_mock, \
             patch.object(blc, "harvest_real_line_book_lines",
                          return_value=([{}], 1, 0)), \
             patch.object(blc, "join_book_lines_to_actuals",
                          return_value=enriched), \
             patch.object(blc, "project_and_empirical",
                          side_effect=lambda obs, *a, **k: (obs["_proj"], obs["_emp"])), \
             patch.object(blc, "project_distributional", return_value=None), \
             redirect_stdout(buf):
            rc.diagnose_roi("mlb", xstats_strength=xstats_strength)
        return buf.getvalue(), save_mock

    def _methods_in_table(self, out):
        """Method letters that got a row: 4 spaces + letter + whitespace + a digit
        (the n column). Excludes the '    A/B/C/E here run...' caveat (a '/' follows)
        and the header/prose caveats."""
        return {m for m in "ABCDE" if re.search(rf"^    {m}\s+\d", out, re.M)}

    def test_prints_all_three_caveats_and_writes_nothing(self):
        out, save_mock = self._run(
            {"pitcher_strikeouts": {"method": "C"}}, _strikeout_obs())
        self.assertIn("BEST-OF-BOOK", out)      # best-of-book / not-DK caveat
        self.assertIn("TRAIN half", out)        # train-fit-vs-deployed caveat
        self.assertIn("nothing written", out)   # no-write footer
        save_mock.assert_not_called()           # the invariant: diagnostic writes nothing

    def test_table_lists_ABCE_and_marks_incumbent(self):
        out, _ = self._run(
            {"pitcher_strikeouts": {"method": "C"}}, _strikeout_obs())
        shown = self._methods_in_table(out)
        self.assertTrue({"A", "B", "C", "E"}.issubset(shown), shown)
        self.assertNotIn("D", shown)            # not batter_hits + patched to None
        self.assertIn("<- incumbent", out)      # method C tagged
        # column header + coverage line present
        self.assertIn("Brier", out)
        self.assertIn("ROI%", out)
        self.assertIn("priced=", out)

    def test_thin_prop_skipped_before_table(self):
        out, _ = self._run(
            {"pitcher_earned_runs": {"method": "C"}},
            [_eobs("pitcher_earned_runs", i, 2.5, proj=2.4, actual=1 + (i % 4))
             for i in range(10)])
        self.assertIn("too thin", out)
        self.assertNotIn("n_test=", out)        # never reached the per-prop table

    def test_xba_caveat_when_shipped_prop_blends_xba(self):
        # batter_hits is xstats-kind; a shipped xstats_strength>0 means the live
        # incumbent runs on an xBA basis while A/B/C/E here use plain projections.
        out, _ = self._run(
            {"batter_hits": {"method": "C", "xstats_strength": 0.6}}, _hits_obs())
        self.assertIn("xBA", out)
        self.assertIn("blends xBA", out)

    def test_no_xba_caveat_when_prop_does_not_ship_xba(self):
        out, _ = self._run(
            {"batter_hits": {"method": "C", "xstats_strength": None}}, _hits_obs())
        self.assertNotIn("blends xBA", out)

    def test_no_calibrated_props_short_circuits(self):
        out, save_mock = self._run({}, [])
        self.assertIn("No calibrated props", out)
        save_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
