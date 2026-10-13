"""NBA prop pace + defense context (nba_prop_context) and its three consumers:
  * the adjustment math (pace / defense factors, combos, fail-open, strengths 0 = identity);
  * build_index: (date, team_id) keys, PRE-GAME contexts (a game never sees its own result);
  * the real-line projector: Bayesian shrinkage now applied (fit == serve) + the context
    adjustment for basketball only;
  * refit plumbing: 8-part variant labels -> nba_* knobs; the NBA focused grid;
  * live pricing: the knobs gate the adjustment and the adjusted values drive avg_stat.
Synthetic data; no network/DB.
"""
import unittest
from unittest.mock import patch

import nba_prop_context as C
import nba_ratings as R


def _ctx(poss, poss_hat, pts_allowed, lg_pts=110.0, ast_allowed=25.0, lg_ast=25.0):
    allowed = {st: (10.0, 10.0) for st in R.ALLOWED_STATS}
    allowed["points"] = (pts_allowed, lg_pts)
    allowed["assists"] = (ast_allowed, lg_ast)
    return {"poss": poss, "poss_hat": poss_hat, "allowed": allowed}


class AdjustMathTests(unittest.TestCase):
    def test_strengths_zero_is_identity(self):
        vals = [10.0, 20.0]
        self.assertIs(C.adjust_values(vals, [None, None], _ctx(None, 100, 110), ("points",),
                                      0.0, 0.0), vals)

    def test_pace_factor(self):
        up = _ctx(None, 104.0, 110.0)
        past = [_ctx(100.0, 0, 110.0), _ctx(98.0, 0, 110.0)]
        out = C.adjust_values([20.0, 30.0], past, up, ("points",), 1.0, 0.0)
        self.assertAlmostEqual(out[0], 20.0 * 104 / 100)
        self.assertAlmostEqual(out[1], 30.0 * 104 / 98)
        half = C.adjust_values([20.0], past[:1], up, ("points",), 0.5, 0.0)
        self.assertAlmostEqual(half[0], 20.0 * (104 / 100) ** 0.5)

    def test_defense_factor_direction(self):
        # tonight's opponent allows 121/100 (+10% vs league 110); the past game was vs an
        # average defense -> value scales UP by 1.10 at full strength.
        up = _ctx(None, 100.0, 121.0)
        out = C.adjust_values([20.0], [_ctx(100.0, 0, 110.0)], up, ("points",), 0.0, 1.0)
        self.assertAlmostEqual(out[0], 22.0)
        # past game was vs an elite D (allows 99 = -10%): normalizing UP, tonight average ->
        out2 = C.adjust_values([20.0], [_ctx(100.0, 0, 99.0)], _ctx(None, 100.0, 110.0),
                               ("points",), 0.0, 1.0)
        self.assertAlmostEqual(out2[0], 20.0 * (1.0 / 0.9))

    def test_combo_sums_components(self):
        up = _ctx(None, 100.0, 121.0, ast_allowed=25.0)         # pts +11, ast +0
        rel_t = (121.0 + 25.0) / (110.0 + 25.0)
        out = C.adjust_values([30.0], [_ctx(100.0, 0, 110.0)], up, ("points", "assists"),
                              0.0, 1.0)
        self.assertAlmostEqual(out[0], 30.0 * rel_t)

    def test_missing_context_fails_open(self):
        out = C.adjust_values([20.0, 30.0], [None, _ctx(100.0, 0, 110.0)],
                              _ctx(None, 110.0, 110.0), ("points",), 1.0, 1.0)
        self.assertEqual(out[0], 20.0)
        self.assertAlmostEqual(out[1], 30.0 * 1.1)
        self.assertEqual(C.adjust_values([5.0], [None], None, ("points",), 1.0, 1.0), [5.0])


def _team_rows():
    def pair(gid, date, h, a, hp, ap, poss):
        base = {"game_id": gid, "season": 2025, "season_type": "regular-season",
                "game_date": date, "game_minutes": 48.0, "possessions": poss}
        extra = {st: 10.0 for st in R.ALLOWED_STATS if st != "points"}
        return [{**base, **extra, "team_id": h, "opponent_team_id": a, "home_away": "home",
                 "points": hp},
                {**base, **extra, "team_id": a, "opponent_team_id": h, "home_away": "away",
                 "points": ap}]
    return (pair("g1", "2025-01-01", 1, 2, 110, 100, 100.0)
            + pair("g2", "2025-01-03", 1, 3, 130, 90, 104.0))


class IndexTests(unittest.TestCase):
    def test_keys_and_pregame(self):
        idx = C.build_index(_team_rows(), {"prior_games": 0, "league_prior_games": 0,
                                           "opp_adjust": False, "half_life": 1e9})
        self.assertIn(("2025-01-03", 1), idx)
        self.assertEqual(idx[("2025-01-03", 1)]["poss"], 104.0)     # actual possessions
        # team 1's pre-game context on 01-03 must not include the 130-point game itself:
        # the opponent (team 3) has no history -> league fallback, so poss_hat uses only g1.
        self.assertIsNotNone(idx[("2025-01-03", 1)]["poss_hat"])

    def test_adjust_games_uses_index(self):
        idx = {("2025-01-03", 1): _ctx(None, 104.0, 110.0),
               ("2025-01-01", 1): _ctx(100.0, 0, 110.0)}
        games = [{"game_date": "2025-01-01", "team_id": 1}]
        out = C.adjust_games([20.0], games, "2025-01-03", 1, idx, ("points",), 1.0, 0.0)
        self.assertAlmostEqual(out[0], 20.0 * 1.04)
        # unknown priced game -> unchanged
        self.assertEqual(C.adjust_games([20.0], games, "2025-02-01", 1, idx, ("points",),
                                        1.0, 0.0), [20.0])


class LiveAdjustTests(unittest.TestCase):
    def test_resolves_opponent_and_adjusts(self):
        mine = {"pace": 102.0, "lg_pace": 100.0, "allowed100": {}, "lg_allowed100": {}}
        theirs = {"pace": 100.0, "lg_pace": 100.0,
                  "allowed100": {st: 10.0 for st in R.ALLOWED_STATS},
                  "lg_allowed100": {st: 10.0 for st in R.ALLOWED_STATS}}
        index = {("2025-01-01", 1): _ctx(100.0, 0, 110.0)}
        with patch("nba_warehouse.resolve_team_id", side_effect={"Home": 1, "Away": 2}.get), \
             patch("nba_prop_context.live_context", return_value=(index, {1: mine, 2: theirs})):
            out = C.live_adjust([20.0], ["2025-01-01"], [1], 1, "Home", "Away", "2025-01-05",
                                "player_points", 1.0, 0.0)
        self.assertAlmostEqual(out[0], 20.0 * 102.0 / 100.0)

    def test_player_not_on_either_team_unchanged(self):
        with patch("nba_warehouse.resolve_team_id", side_effect={"Home": 1, "Away": 2}.get):
            out = C.live_adjust([20.0], ["2025-01-01"], [9], 9, "Home", "Away", "2025-01-05",
                                "player_points", 1.0, 1.0)
        self.assertEqual(out, [20.0])


class RealLineProjectorTests(unittest.TestCase):
    def _obs(self, prop="player_points"):
        prior = [{"points": v, "MIN": 30.0, "is_home": True, "opponent": 5,
                  "game_date": f"2025-01-{d:02d}", "team_id": 1}
                 for v, d in ((30.0, 10), (10.0, 8), (20.0, 6))]
        return {"prior_games": prior, "stat_label": "points", "line": 19.5,
                "test_game": {"is_home": True, "game_date": "2025-01-12", "team_id": 1,
                              "season": 2025},
                "prop_key": prop, "game_date": "2025-01-12"}

    def test_shrinkage_now_applied(self):
        import book_line_calibration as blc
        p0, _ = blc.project_and_empirical(self._obs(), {"half_life": 1.0}, "basketball_nba")
        p1, _ = blc.project_and_empirical(self._obs(), {"half_life": 1.0, "shrinkage_k": 10},
                                          "basketball_nba")
        w = [1.0, 0.5, 0.25]
        wm = (30 * 1 + 10 * 0.5 + 20 * 0.25) / sum(w)
        self.assertAlmostEqual(p0, wm)
        self.assertAlmostEqual(p1, (sum(w) * wm + 10 * 20.0) / (sum(w) + 10))

    def test_context_basketball_only(self):
        import book_line_calibration as blc
        params = {"half_life": None, "nba_pace_strength": 1.0}
        with patch("nba_prop_context.index_for_seasons", return_value={"x": 1}), \
             patch("nba_prop_context.adjust_games",
                   side_effect=lambda v, *a, **k: [x * 2 for x in v]):
            nba, _ = blc.project_and_empirical(self._obs(), params, "basketball_nba")
            mlb, _ = blc.project_and_empirical(self._obs(), params, "baseball_mlb")
        self.assertAlmostEqual(nba, 40.0)              # mean 20 doubled
        self.assertAlmostEqual(mlb, 20.0)              # ignored outside basketball


class RefitPlumbingTests(unittest.TestCase):
    def test_parse_and_cfg(self):
        import refit_calibration as rc
        name = "hl3/opp0.0/defadj0.0/shrink10/ven0.0/rest0.0/npace1.0/ndef0.5"
        p = rc._parse_variant_name(name)
        self.assertEqual((p["nba_pace_strength"], p["nba_def_strength"]), (1.0, 0.5))
        self.assertEqual(p["shrink_k"], 10.0)
        self.assertNotIn("nba_pace_strength",
                         rc._parse_variant_name("hl3/opp0.0/defadj0.0/shrink10/ven0.0/rest0.0"))
        self.assertIsNone(rc._parse_variant_name(
            "hl3/opp0.0/defadj0.0/shrink10/ven0.0/rest0.0/bogus1/ndef0.5"))

    def test_nba_focused_grid(self):
        import backtest
        g = backtest._build_focused_props_grid(half_lifes=[None, 3], nba_context=True)
        ctx = [k for k in g if "/npace" in k]
        self.assertEqual(len(ctx), 2 * 4 * len(backtest.NBA_CONTEXT_COMBOS))
        self.assertFalse(any("/opp0.5/" in k or "/defadj1.0/" in k for k in g))
        self.assertEqual(g[ctx[0]]["nba_pace"] + g[ctx[0]]["nba_def"] > 0, True)
        plain = backtest._build_focused_props_grid(half_lifes=[None, 3])
        self.assertTrue(any("/opp0.5/" in k for k in plain))     # other sports unchanged
        self.assertFalse(any("/npace" in k for k in plain))


class LivePropsWiringTests(unittest.TestCase):
    def _run(self, cfg, adjust=None):
        import props
        prop_data = {"commence_time": "2025-01-15T00:30:00Z", "home_team": "Home",
                     "away_team": "Away", "game_id": "evt1",
                     "props": {"player_points": {"Star Guard": {
                         "line": 20.5, "over_implied": 0.5, "under_implied": 0.5,
                         "over_price": -110, "under_price": -110,
                         "over_book": "DK", "under_book": "DK"}}}}
        dates = [f"2025-01-{d:02d}" for d in range(1, 15)]
        hist = {"Star Guard": {"player_points": {
            "found": True, "values": [20.0] * 14, "game_dates": list(reversed(dates)),
            "minutes": [34.0] * 14, "team_id": 1, "team_ids": [1] * 14}}}
        with patch.object(props, "load_calibration", return_value={"player_points": cfg}), \
             patch.object(props, "load_recalibration", return_value={}), \
             patch.object(props, "maybe_auto_refit"), \
             patch.object(props, "log_prediction_rows"), \
             patch.object(props, "log_prediction"), \
             patch("nba_prop_context.live_adjust",
                   side_effect=adjust or (lambda v, *a, **k: [x * 2 for x in v])) as la:
            cands = props.analyze_player_props_value(prop_data, hist, threshold_pct=1.0,
                                                     sport_key="basketball_nba")
        return cands[0], la

    def test_knobs_on_adjust_values(self):
        base = {"method": "A", "half_life": None, "shrinkage_k": 0}
        off, la_off = self._run(base)
        on, la_on = self._run({**base, "nba_pace_strength": 1.0, "nba_def_strength": 0.5})
        la_off.assert_not_called()
        la_on.assert_called_once()
        args = la_on.call_args[0]
        # per-game team ids, one per value that survived the live window (recent_n) +
        # reliability filter
        self.assertEqual(args[2], [1] * len(args[0]))
        self.assertEqual(args[1], sorted(args[1], reverse=True))  # dates newest-first
        self.assertEqual((args[8], args[9]), (1.0, 0.5))        # the strengths
        self.assertAlmostEqual(on["avg_stat"], 2 * off["avg_stat"])


if __name__ == "__main__":
    unittest.main()
