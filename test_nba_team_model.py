"""NBA team-market model (nba_team_model) + its serving hooks:
  * model math: predict / fit_offsets / walk-forward offsets use only earlier seasons;
  * build_matchup_features: inert without a fitted block, None on an unresolved team, the
    pregame-index (backtest) path;
  * analysis: _predict_margin + analyze_totals_value use the model's margin / total / sigmas
    for NBA only;
  * backtest: NBA warehouse schedules (odds-feed names) + _matchup_features_for dispatch;
  * app: the NBA warehouse team dim / schedule helpers (extracted from app.py source, since
    app.py is the Streamlit entry point and isn't importable in tests).
Synthetic rows; no network/DB.
"""
import ast
import math
import os
import unittest
from unittest.mock import patch

import analysis
import backtest
import nba_team_model as TM

_BLOCK = {"params": {"half_life": 20.0, "opp_adjust": True, "regress": 0.5,
                     "prior_games": 10.0, "league_prior_games": 60.0},
          "hca": 2.0, "tot_off": 1.5, "beta_rest": -2.5, "sigma_margin": 14.0,
          "sigma_total": 18.5}

_H = {"pace": 100.0, "ortg": 116.0, "drtg": 110.0, "lg_pace": 100.0, "lg_rtg": 113.0,
      "n": 20, "rest_days": 2}
_A = {"pace": 100.0, "ortg": 112.0, "drtg": 114.0, "lg_pace": 100.0, "lg_rtg": 113.0,
      "n": 20, "rest_days": 1}                          # away on a back-to-back


class ModelMathTests(unittest.TestCase):
    def test_predict_adds_hca_rest_offset(self):
        margin, sm, total, st = TM.predict(_H, _A, _BLOCK)
        base_h = 100.0 * (116 * 114 / 113) / 100
        base_a = 100.0 * (112 * 110 / 113) / 100
        self.assertAlmostEqual(margin, (base_h - base_a) + 2.0 + (-2.5) * (0 - 1))
        self.assertAlmostEqual(total, base_h + base_a + 1.5)
        self.assertEqual((sm, st), (14.0, 18.5))

    def test_fit_offsets_recovers_hca_and_rest(self):
        games, raw = [], {}
        for k in range(399):                               # multiple of 3 -> mean b2b 0
            b2b = (k % 3) - 1                              # -1, 0, +1
            m_raw, t_raw = 0.0, 220.0
            gid = f"g{k}"
            games.append({"game_id": gid, "season": 2024, "hp": 110 + 2.0 - 1.5 * b2b,
                          "ap": 110.0})
            raw[gid] = (m_raw, t_raw, b2b)
        f = TM.fit_offsets(games, raw, [2024])
        self.assertAlmostEqual(f["hca"], 2.0, places=6)    # b2b mean is 0 by design
        self.assertAlmostEqual(f["beta_rest"], -1.5, places=6)
        self.assertAlmostEqual(f["tot_off"], (220.0 + 2.0) - 220.0, places=6)
        self.assertAlmostEqual(f["sig_m"], 0.0, places=6)

    def test_fit_offsets_needs_minimum_games(self):
        self.assertIsNone(TM.fit_offsets([], {}, [2024]))


class FeaturesTests(unittest.TestCase):
    def test_inert_without_block(self):
        with patch("nba_team_model.model", return_value=None):
            self.assertIsNone(TM.build_matchup_features("A", "B", "2025-01-10"))

    def test_unresolved_team_none(self):
        with patch("nba_warehouse.resolve_team_id", return_value=None):
            self.assertIsNone(TM.build_matchup_features("A", "B", "2025-01-10", pregame={},
                                                        m=_BLOCK))

    def test_pregame_index_path(self):
        idx = {("2025-01-10", 1): _H, ("2025-01-10", 2): _A}
        with patch("nba_warehouse.resolve_team_id", side_effect=lambda n: {"Home": 1,
                                                                           "Away": 2}[n]):
            f = TM.build_matchup_features("Home", "Away", "2025-01-10", pregame=idx, m=_BLOCK)
        margin, sm, total, st = TM.predict(_H, _A, _BLOCK)
        self.assertAlmostEqual(f["nba_pred_margin"], margin)
        self.assertAlmostEqual(f["nba_pred_total"], total)
        self.assertEqual(f["nba_pred_std"], 14.0)
        self.assertEqual(f["nba_total_std"], 18.5)
        self.assertEqual(f["nba_away"]["rest_days"], 1)

    def test_model_requires_all_keys(self):
        TM._MODEL_CACHE.clear()
        with patch("calibration_loader.load_starter_adjustment",
                   return_value={"nba_team_model": {"hca": 2.0}}):
            self.assertIsNone(TM.model())
        TM._MODEL_CACHE.clear()
        with patch("calibration_loader.load_starter_adjustment",
                   return_value={"nba_team_model": _BLOCK}):
            self.assertEqual(TM.model()["hca"], 2.0)
        TM._MODEL_CACHE.clear()

    def test_model_memoized_until_file_changes(self):
        TM._MODEL_CACHE.clear()
        with patch("calibration_loader.load_starter_adjustment",
                   return_value={"nba_team_model": _BLOCK}) as load:
            TM.model()
            TM.model()
        self.assertEqual(load.call_count, 1)                 # second call hit the cache
        TM._MODEL_CACHE.clear()


class WalkForwardTests(unittest.TestCase):
    def test_offsets_use_only_earlier_seasons(self):
        # 2023 has enough games to fit; 2024 is scored from 2023 only; 2023 itself has no
        # earlier season -> unscored.
        rows = []
        for season, yr in ((2023, "2023"), (2024, "2024")):
            for k in range(220):
                d = f"{yr}-{1 + k // 28:02d}-{1 + k % 28:02d}"
                for tid, opp, pts, side in ((1, 2, 112.0, "home"), (2, 1, 108.0, "away")):
                    rows.append({"game_id": f"{yr}-{k}", "team_id": tid,
                                 "opponent_team_id": opp, "season": season,
                                 "season_type": "regular-season", "game_date": d,
                                 "home_away": side, "game_minutes": 48.0,
                                 "possessions": 100.0, "points": pts})
        games = TM.games_from_rows(rows)
        preds, fits = TM.walk_forward_predictions(games, rows, _BLOCK["params"], [2023, 2024])
        self.assertIsNone(fits[2023])
        self.assertIsNotNone(fits[2024])
        self.assertTrue(all(g["season"] == 2024 for g in games if g["game_id"] in preds))


class AnalysisHookTests(unittest.TestCase):
    def _stats(self, team):
        return {"recent_games": [{"home_team": team, "away_team": "X", "home_score": 100,
                                  "away_score": 90, "total_score": 190}],
                "recent": {"avg_scored": 100, "avg_allowed": 90}}

    def test_predict_margin_uses_model_for_nba(self):
        feats = {"nba_pred_margin": 4.5, "nba_pred_std": 14.0}
        out = analysis._predict_margin({"home_team": "H", "away_team": "A"}, self._stats("H"),
                                       self._stats("A"), "basketball_nba",
                                       matchup_features=feats)
        self.assertEqual((out[0], out[1]), (4.5, 14.0))

    def test_predict_margin_ignores_nba_keys_for_other_sports(self):
        feats = {"nba_pred_margin": 4.5, "nba_pred_std": 14.0}
        out = analysis._predict_margin({"home_team": "H", "away_team": "A"}, self._stats("H"),
                                       self._stats("A"), "icehockey_nhl",
                                       matchup_features=feats)
        self.assertNotEqual(out[0], 4.5)

    def test_totals_use_model_projection_and_sigma(self):
        odds = {"home_team": "H", "away_team": "A",
                "totals": {"Over": [{"line": 225.5, "price": -110, "book": "DraftKings"}],
                           "Under": [{"line": 225.5, "price": -110, "book": "DraftKings"}]}}
        feats = {"nba_pred_total": 230.0, "nba_total_std": 18.0}
        cands = analysis.analyze_totals_value(odds, self._stats("H"), self._stats("A"),
                                              sport_key="basketball_nba",
                                              matchup_features=feats)
        self.assertEqual(cands[0]["projected_total"], 230.0)
        p_over = 1 - 0.5 * (1 + math.erf(((225.5 - 230.0) / 18.0) / math.sqrt(2)))
        self.assertAlmostEqual(cands[0]["model_over_hit_rate"], round(p_over * 100, 2))


class BacktestTests(unittest.TestCase):
    def _rows(self):
        out = []
        for gid, d, h, a, hp, ap in (("g1", "2025-01-02", 1, 2, 110, 100),
                                     ("g2", "2025-01-04", 2, 1, 120, 105)):
            out.append({"game_id": gid, "game_date": d, "team_id": h, "opponent_team_id": a,
                        "home_away": "home", "points": hp, "opponent_points": ap})
            out.append({"game_id": gid, "game_date": d, "team_id": a, "opponent_team_id": h,
                        "home_away": "away", "points": ap, "opponent_points": hp})
        return out

    def test_nba_warehouse_schedules_use_odds_names(self):
        store = {"games": {"k": {"home_team": "Los Angeles Clippers",
                                 "away_team": "Boston Celtics"}}}
        names = {1: "LA Clippers", 2: "Boston Celtics"}
        resolve = {"Los Angeles Clippers": 1, "Boston Celtics": 2}
        with patch("nba_warehouse.team_games", return_value=self._rows()), \
             patch("nba_warehouse.team_names", return_value=names), \
             patch("nba_warehouse.resolve_team_id", side_effect=resolve.get):
            teams, sched = backtest._nba_warehouse_team_schedules([2025], store)
        self.assertEqual(teams["Los Angeles Clippers"]["id"], 1)     # odds spelling wins
        self.assertNotIn("LA Clippers", teams)
        g = sched[1][0]
        self.assertEqual((g["date"], g["home_team"], g["away_team"]),
                         ("2025-01-02", "Los Angeles Clippers", "Boston Celtics"))
        self.assertEqual((g["home_score"], g["away_score"], g["total_score"]),
                         (110.0, 100.0, 210.0))
        self.assertEqual(len(sched[1]), 2)
        self.assertEqual(teams["Los Angeles Clippers"]["win_pct"], 0.5)

    def test_matchup_features_dispatch(self):
        with patch("backtest._nba_matchup_features", return_value={"x": 1}) as f:
            self.assertEqual(backtest._matchup_features_for("H", "A", "2025-01-02",
                                                            "basketball_nba"), {"x": 1})
            f.assert_called_once()


def _app_funcs(*names):
    """Exec selected top-level functions from app.py's source in an isolated namespace."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    ns = {"find_team": lambda teams, name: {"id": "espn", "display_name": name},
          "get_team_schedule": lambda *a: ["espn"]}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns


class AppHelperTests(unittest.TestCase):
    def setUp(self):
        self.ns = _app_funcs("_resolve_team_dim", "_fetch_team_schedule")

    def test_nba_dim_is_warehouse_str_id(self):
        with patch("nba_warehouse.resolve_team_id", return_value=1610612747):
            d = self.ns["_resolve_team_dim"]("basketball_nba", "Los Angeles Lakers", {})
        self.assertEqual(d["id"], "1610612747")
        self.assertEqual(d["display_name"], "Los Angeles Lakers")

    def test_nba_never_uses_espn(self):
        # ESPN retired for NBA: the ESPN find_team stub is never consulted
        with patch("nba_warehouse.resolve_team_id", return_value=None):
            self.assertIsNone(self.ns["_resolve_team_dim"]("basketball_nba", "Lakers", {}))

    def test_nba_unresolved_team_skips(self):
        with patch("nba_warehouse.resolve_team_id", return_value=None):
            self.assertIsNone(self.ns["_resolve_team_dim"]("basketball_nba", "Nope", {}))

    def test_nba_schedule_from_warehouse(self):
        team = {"id": "1610612747", "display_name": "Los Angeles Lakers"}
        with patch("nba_team_stats.team_stats",
                   return_value={"recent_games": [{"date": "2025-01-02"}]}):
            s = self.ns["_fetch_team_schedule"]("basketball_nba", team, "basketball", "nba")
        self.assertEqual(s, [{"date": "2025-01-02"}])


if __name__ == "__main__":
    unittest.main()
