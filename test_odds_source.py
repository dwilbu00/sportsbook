"""Tests for the team-market backtest's Azure-warehouse odds source selection
and the prediction-log holdout supplement (backtest.py Phase B).

Hermetic — no SQL and no network: the warehouse/store/log readers are patched, so
these exercise only the selection logic, the supplement mapping/dedup, and the
extra-obs folding into the shrink holdout.
"""

import unittest
from unittest.mock import patch

import backtest


class LoadOddsStoreTests(unittest.TestCase):
    """_load_odds_store: auto / warehouse / store selection."""

    def _wh(self, games):
        return {"sport_key": "baseball_mlb", "games": games,
                "bookmaker": "warehouse (best-of-book, closing)"}

    def _local(self, games):
        return {"sport_key": "baseball_mlb", "games": games}

    def test_auto_prefers_warehouse_when_it_has_games(self):
        with patch("warehouse.load_team_market_store",
                   return_value=self._wh({"g1": {}})), \
             patch.object(backtest.hist_store, "load_store",
                          return_value=self._local({"g2": {}})) as local_load:
            store, used = backtest._load_odds_store("baseball_mlb", "", "auto")
        self.assertEqual(used, "warehouse")
        self.assertIn("g1", store["games"])
        local_load.assert_not_called()

    def test_auto_falls_back_to_store_when_warehouse_empty(self):
        with patch("warehouse.load_team_market_store", return_value=self._wh({})), \
             patch.object(backtest.hist_store, "load_store",
                          return_value=self._local({"g2": {}})):
            store, used = backtest._load_odds_store("baseball_mlb", "", "auto")
        self.assertEqual(used, "store")
        self.assertIn("g2", store["games"])

    def test_store_label_forces_local_even_in_auto(self):
        with patch("warehouse.load_team_market_store",
                   return_value=self._wh({"g1": {}})), \
             patch.object(backtest.hist_store, "load_store",
                          return_value=self._local({"g2": {}})):
            _store, used = backtest._load_odds_store("baseball_mlb", "morning",
                                                     "auto")
        self.assertEqual(used, "store")

    def test_force_warehouse(self):
        with patch("warehouse.load_team_market_store",
                   return_value=self._wh({"g1": {}})):
            _store, used = backtest._load_odds_store("baseball_mlb", "",
                                                     "warehouse")
        self.assertEqual(used, "warehouse")

    def test_force_store(self):
        with patch.object(backtest.hist_store, "load_store",
                          return_value=self._local({"g2": {}})) as local_load:
            _store, used = backtest._load_odds_store("baseball_mlb", "", "store")
        self.assertEqual(used, "store")
        local_load.assert_called_once()


class MarketLogSupplementTests(unittest.TestCase):
    """_market_log_supplement: raw_prob/outcome, dedup, push-drop, graded-skip."""

    def _row(self, **kw):
        base = {"sport_key": "baseball_mlb", "resolved": True,
                "bet_type": "moneyline", "event_id": "e1", "outcome": 1,
                "raw_prob": 0.6, "model_prob": 0.65, "price": 120,
                "is_value": True, "commence_time": "2026-07-22T23:00:00Z",
                "home_team": "Rockies", "away_team": "Astros"}
        base.update(kw)
        return base

    def _supplement(self, rows, graded=None):
        with patch("recalibration._read_market_log", return_value=rows):
            return backtest._market_log_supplement("baseball_mlb", graded or set())

    def test_includes_resolved_prefers_raw_prob(self):
        out = self._supplement([self._row()])
        self.assertEqual(out["moneyline"]["obs"], [(0.6, 1)])
        self.assertEqual(out["moneyline"]["roi"], [(True, 120, 1)])

    def test_falls_back_to_model_prob(self):
        out = self._supplement([self._row(raw_prob=None)])
        self.assertEqual(out["moneyline"]["obs"], [(0.65, 1)])

    def test_drops_push_and_out_of_range_prob(self):
        out = self._supplement([self._row(outcome=None),               # push
                                self._row(event_id="e2", raw_prob=1.5)])  # bad
        self.assertEqual(out["moneyline"]["obs"], [])

    def test_skips_warehouse_graded_keys(self):
        out = self._supplement([self._row()], graded={("e1", "moneyline")})
        self.assertEqual(out["moneyline"]["obs"], [])

    def test_dedup_via_game_key_when_store_lacked_event_id(self):
        # The store graded under a game_key token (no event_id); the log row
        # carries an event_id. They must still dedup via the shared game_key.
        import historical_odds
        gk = historical_odds.game_key("2026-07-22T23:00:00Z", "Rockies", "Astros")
        out = self._supplement([self._row()], graded={(gk, "moneyline")})
        self.assertEqual(out["moneyline"]["obs"], [])

    def test_dedup_within_log(self):
        out = self._supplement([self._row(), self._row(raw_prob=0.9)])
        self.assertEqual(len(out["moneyline"]["obs"]), 1)

    def test_bet_type_maps_to_market_names(self):
        out = self._supplement([self._row(bet_type="spread", event_id="s1"),
                                self._row(bet_type="total", event_id="t1")])
        self.assertEqual(len(out["spreads"]["obs"]), 1)
        self.assertEqual(len(out["totals"]["obs"]), 1)

    def test_event_id_fallback_to_game_key(self):
        # No event_id → keyed by game_key(commence, home, away); still included.
        out = self._supplement([self._row(event_id=None)])
        self.assertEqual(out["moneyline"]["obs"], [(0.6, 1)])

    def test_date_window_scopes_out_of_range_rows(self):
        rows = [self._row(event_id="in", game_date="2026-07-15"),
                self._row(event_id="out", game_date="2023-04-01"),
                self._row(event_id="nodate", game_date=None)]
        with patch("recalibration._read_market_log", return_value=rows):
            out = backtest._market_log_supplement(
                "baseball_mlb", set(),
                date_from="2026-07-01", date_to="2026-07-31")
        # Only the in-window row survives; out-of-range and undated are dropped.
        self.assertEqual(out["moneyline"]["obs"], [(0.6, 1)])

    def test_no_window_keeps_all(self):
        rows = [self._row(event_id="a", game_date="2026-07-15"),
                self._row(event_id="b", game_date="2023-04-01")]
        with patch("recalibration._read_market_log", return_value=rows):
            out = backtest._market_log_supplement("baseball_mlb", set())
        self.assertEqual(len(out["moneyline"]["obs"]), 2)


class ShrinkCalibrationExtraObsTests(unittest.TestCase):
    """_write_shrink_calibration folds the log supplement into the holdout, and
    the fold is flip-symmetric (a one-sided (p,o) row == its (1-p,1-o) form)."""

    def _results(self, blend):
        buckets = {m: backtest._empty_market_bucket() for m in backtest.MARKETS}
        buckets["moneyline"]["blend"] = blend
        return {"live": buckets}

    def test_extra_obs_counted_in_holdout(self):
        results = self._results([(0.7, 0.55, 1), (0.6, 0.52, 0)])
        extra = {"moneyline": [(0.8, 1)], "spreads": [], "totals": []}
        captured = {}

        def fake_save(sport_key, shrink, holdout=None, meta=None):
            captured["holdout"] = holdout

        with patch.object(backtest, "save_prob_shrink", fake_save):
            backtest._write_shrink_calibration("baseball_mlb", results,
                                               extra_obs=extra)
        ml = captured["holdout"]["moneyline"]
        self.assertEqual(ml["n"], 3)
        self.assertEqual(ml["n_warehouse"], 2)
        self.assertEqual(ml["n_log"], 1)

    def test_thin_sample_withholds_shrink_but_publishes_holdout(self):
        # A blend where shrinking clearly helps (overconfident + wrong), but the
        # sample is below the guard → shrink withheld, holdout still published.
        results = self._results([(0.9, 0.55, 0)] * 5)
        captured = {}

        def fake_save(sport_key, shrink, holdout=None, meta=None):
            captured["shrink"] = shrink
            captured["holdout"] = holdout

        with patch.object(backtest, "save_prob_shrink", fake_save):
            backtest._write_shrink_calibration("baseball_mlb", results,
                                               min_shrink_n=100)
        self.assertEqual(captured["shrink"], {})            # withheld (n=5<100)
        self.assertIn("moneyline", captured["holdout"])     # holdout still published

    def test_shrink_persisted_when_sample_meets_min(self):
        results = self._results([(0.9, 0.55, 0)] * 5)
        captured = {}

        def fake_save(sport_key, shrink, holdout=None, meta=None):
            captured["shrink"] = shrink

        with patch.object(backtest, "save_prob_shrink", fake_save):
            backtest._write_shrink_calibration("baseball_mlb", results,
                                               min_shrink_n=5)
        self.assertEqual(captured["shrink"].get("moneyline"), 0.0)  # n=5>=5

    def test_flip_symmetry_of_best_shrink(self):
        # (p, o) folds into the same raw/shrunk Brier as its flip (1-p, 1-o) —
        # equal up to float summation noise, so compare with tolerance.
        a = backtest._best_shrink([(0.7, None, 1)])
        b = backtest._best_shrink([(0.3, None, 0)])
        self.assertEqual(a[0], b[0])                    # same best_s
        self.assertAlmostEqual(a[1], b[1], places=9)    # same best_brier
        self.assertAlmostEqual(a[2], b[2], places=9)    # same raw_brier


if __name__ == "__main__":
    unittest.main()
