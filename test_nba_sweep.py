"""NBA backtest-engine warehouse integration: the player-props sweep (synthetic refit /
recency sweep / eval_min_streak all route through run_player_props_backtest -> fetch_player_data)
grades NBA off nba_warehouse (all 16 props incl. combos) instead of ESPN's 3. SQLite; no network.
"""
import os
import shutil
import tempfile
import unittest

import db_store
import backtest
import nba_warehouse as nw


class SweepModeTests(unittest.TestCase):
    """ESPN retired for NBA: basketball always sweeps off the warehouse (mirror-first, so even
    with SQL off); no kill switch."""

    def tearDown(self):
        db_store.configure_engine(None)

    def test_basketball_always_warehouse(self):
        db_store.configure_engine(None)
        self.assertTrue(backtest._nba_sweep_mode("basketball"))
        db_store.configure_engine("sqlite://")
        self.assertTrue(backtest._nba_sweep_mode("basketball"))

    def test_non_basketball_off(self):
        self.assertFalse(backtest._nba_sweep_mode("baseball"))
        self.assertFalse(backtest._nba_sweep_mode("football"))


class SweepStatLabelTests(unittest.TestCase):
    def _gl(self):
        return [{"points": 30.0, "rebounds": 8.0, "assists": 9.0, "played": True},
                {"points": 0.0, "rebounds": 0.0, "assists": 0.0, "played": False}]  # DNP

    def test_single_prop_uses_column(self):
        self.assertEqual(backtest._nba_sweep_stat_label("player_points", self._gl()), "points")

    def test_combo_injects_sum_and_dnp_none(self):
        gl = self._gl()
        label = backtest._nba_sweep_stat_label("player_points_rebounds_assists", gl)
        self.assertEqual(label, "player_points_rebounds_assists")
        self.assertEqual(gl[0][label], 47.0)                 # 30+8+9
        self.assertIsNone(gl[1][label])                      # DNP -> None (skipped, no phantom 0)

    def test_unmapped_prop_none(self):
        self.assertIsNone(backtest._nba_sweep_stat_label("player_double_double", self._gl()))


class FetchPlayerDataNbaTests(unittest.TestCase):
    def setUp(self):
        self._mdir = tempfile.mkdtemp()                  # empty mirror -> fall through to SQLite
        os.environ["ODI_MIRROR_DIR"] = self._mdir
        db_store.configure_engine("sqlite://")
        nw.create_all()
        with db_store.get_engine().begin() as c:
            c.execute(nw.nba_player.insert().values(
                person_id=20, name_norm="shooter guy", full_name="Shooter Guy"))
            for k in range(1, 4):
                c.execute(nw.nba_player_game.insert().values(
                    person_id=20, game_id=f"g{k}", season=2024,
                    game_date=f"2024-01-0{k}", team_id=13, home_away="home",
                    opponent_team_id=99, played=True, minutes=30.0,
                    points=float(20 + k), rebounds=5.0, assists=4.0,
                    three_point_field_goals_attempted=8.0))

    def tearDown(self):
        db_store.configure_engine(None)
        os.environ.pop("ODI_MIRROR_DIR", None)
        shutil.rmtree(self._mdir, ignore_errors=True)

    def test_fetch_routes_to_warehouse(self):
        data = backtest.fetch_player_data("basketball", "nba", ["Shooter Guy"],
                                          season_year=2024, quiet=True)
        self.assertIn("Shooter Guy", data)
        gl = data["Shooter Guy"]
        self.assertEqual(len(gl), 3)
        self.assertEqual(gl[0]["game_date"], "2024-01-03")    # newest first
        self.assertEqual(gl[0]["points"], 23.0)
        self.assertEqual(gl[0]["team_id"], 13)                # team_id present (layoff detect)
        self.assertIn("three_point_field_goals_attempted", gl[0])
        # the sweep's actual-extraction then resolves all props off these rows
        self.assertEqual(backtest._nba_sweep_stat_label("player_threes", gl),
                         "three_point_field_goals_made")

    def test_fetch_missing_player_skipped(self):
        data = backtest.fetch_player_data("basketball", "nba", ["Nobody Here"],
                                          season_year=2024, quiet=True)
        self.assertEqual(data, {})


class NbaDefenseTests(unittest.TestCase):
    """Per-prop, id-keyed, ESPN-free opponent-defense from the warehouse corpus."""

    def _corpus(self):
        # Opponents are stats.nba.com team_id INTS (int64) — as the warehouse/mirror carry them.
        # team 13 allowed (g1): pts 30+20=50, 3pm 5+3=8; team 1 allowed (g2): pts 10, 3pm 1.
        return {
            "player a": [{"opponent": 13, "game_pk": "g1", "game_date": "2024-01-01",
                          "points": 30.0, "three_point_field_goals_made": 5.0}],
            "player b": [{"opponent": 13, "game_pk": "g1", "game_date": "2024-01-01",
                          "points": 20.0, "three_point_field_goals_made": 3.0}],
            "player c": [{"opponent": 1, "game_pk": "g2", "game_date": "2024-01-02",
                          "points": 10.0, "three_point_field_goals_made": 1.0}],
        }

    def _agg(self):
        from unittest.mock import patch
        with patch("nba_warehouse.calib_gamelogs_bulk", return_value=self._corpus()):
            return backtest._nba_warehouse_opp_allowed_agg([2024])

    def test_opp_allowed_aggregation(self):
        agg = self._agg()
        self.assertEqual(set(agg), {13, 1})                   # native int keys
        d13 = dict(agg[13][0][1])
        self.assertEqual(d13["points"], 50.0)                 # opposing players summed = allowed
        self.assertEqual(d13["three_point_field_goals_made"], 8.0)

    def test_per_prop_slice_single_and_combo(self):
        agg = self._agg()
        avg_p, series_p, league_p = backtest._nba_def_for_prop(agg, "points")
        self.assertEqual(avg_p[13], 50.0)
        self.assertEqual(avg_p[1], 10.0)
        self.assertEqual(league_p, 30.0)                      # (50+10)/2
        avg_c, _, _ = backtest._nba_def_for_prop(
            agg, ("points", "three_point_field_goals_made"))
        self.assertEqual(avg_c[13], 58.0)                     # 50 + 8 per-game combo sum

    def test_int_id_resolve_no_crash(self):
        # The migration bug: ESPN ids were str, stats.nba.com are int. The shared resolvers must
        # exact-match on int keys and NOT fall into the .lower() substring path (crash on int).
        from backtest import _resolve_opp_pts_allowed, _resolve_opp_pa_asof
        avg_p, series_p, _ = backtest._nba_def_for_prop(self._agg(), "points")
        self.assertEqual(_resolve_opp_pts_allowed(1, avg_p), 10.0)   # int lookup, exact
        self.assertEqual(_resolve_opp_pts_allowed(13, avg_p), 50.0)
        self.assertIsNone(_resolve_opp_pts_allowed(999, avg_p))      # int miss -> None (no crash)
        # output-side as-of resolver on an int key (this is the one that crashed)
        self.assertEqual(_resolve_opp_pa_asof(13, "2024-06-01", series_p), 50.0)
        self.assertIsNone(_resolve_opp_pa_asof(999, "2024-06-01", series_p))  # miss -> None


class HalfLifeGridTests(unittest.TestCase):
    """The half-life axis is overridable (fan recency out both ways) without touching the
    per-sport default [None, 5, 10, 15]."""

    def test_parse(self):
        import refit_calibration as rc
        self.assertEqual(rc._parse_half_lives("none,3,4,5,20"), [None, 3, 4, 5, 20])
        self.assertIsNone(rc._parse_half_lives(None))
        self.assertIsNone(rc._parse_half_lives(""))

    def test_default_grid_unchanged(self):
        keys = " ".join(backtest._build_focused_props_grid().keys())
        self.assertIn("hl5/", keys)
        self.assertIn("hl15/", keys)
        self.assertNotIn("hl3/", keys)              # default has no 3

    def test_focused_override_fans_out(self):
        keys = " ".join(backtest._build_focused_props_grid(
            half_lifes=[None, 3, 4, 5, 20]).keys())
        self.assertIn("hl3/", keys)
        self.assertIn("hl4/", keys)
        self.assertIn("hl20/", keys)
        self.assertNotIn("hl10/", keys)             # 10 not in the override

    def test_full_grid_override(self):
        keys = list(backtest._build_props_sweep_grid(half_lifes=[3]).keys())
        self.assertTrue(keys and all(k.startswith("hl3/") for k in keys))


if __name__ == "__main__":
    unittest.main()
