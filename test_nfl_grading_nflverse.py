"""NFL prop grading via the nflverse player-week feed (phase 1 of the cutover).

Guards recalibration._resolve_nfl_actual + nfl_schedule.season_week_for_date + the
resolve_one_prop wiring (nflverse-first, ESPN fallback). See
notes/NFL_GRADING_NFLVERSE_CUTOVER_2026-09-25.md.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_nfl_grading_nflverse -v
"""
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

import recalibration as rc
import nfl_schedule
import nfl_props_scan as scan
import nfl_opportunity_serving as nos


def _df(rows):
    return pd.DataFrame(rows)


class ResolveNflActualTests(unittest.TestCase):
    def _run(self, prop_key, player, df, sw=("2026", 3), game_date="2026-09-21"):
        with patch("nfl_schedule.season_week_for_date", return_value=sw), \
                patch("nfl_opportunity_serving._load", return_value=df):
            return rc._resolve_nfl_actual(prop_key, player, game_date)

    def test_yardage_reads_the_mapped_column(self):
        df = _df([{"player_norm": scan._norm("Jared Goff"), "week": 3,
                   "passing_yards": 327.0, "rushing_yards": 5.0}])
        self.assertEqual(self._run("player_pass_yds", "Jared Goff", df), 327.0)

    def test_receptions_column(self):
        df = _df([{"player_norm": scan._norm("Malik Nabers"), "week": 3,
                   "receptions": 7.0}])
        self.assertEqual(self._run("player_receptions", "Malik Nabers", df), 7.0)

    def test_anytime_td_sums_rush_rec_and_return(self):
        # DK/FD anytime-TD pays rushing + receiving + return TDs -> sum all three.
        df = _df([{"player_norm": scan._norm("Kyren Williams"), "week": 3,
                   "rushing_tds": 1.0, "receiving_tds": 1.0, "special_teams_tds": 1.0}])
        self.assertEqual(self._run("player_anytime_td", "Kyren Williams", df), 3.0)

    def test_zero_is_a_real_actual_not_pending(self):
        # a player who PLAYED but caught 0 has a row -> actual 0.0, NOT None(pending).
        df = _df([{"player_norm": scan._norm("Player One"), "week": 3, "receptions": 0.0}])
        self.assertEqual(self._run("player_receptions", "Player One", df), 0.0)

    def test_unmapped_prop_is_none(self):
        df = _df([{"player_norm": scan._norm("X"), "week": 3, "passing_yards": 1.0}])
        self.assertIsNone(self._run("player_field_goals_made", "X", df))

    def test_dnp_no_row_is_none(self):
        df = _df([{"player_norm": scan._norm("Someone Else"), "week": 3, "receptions": 5.0}])
        self.assertIsNone(self._run("player_receptions", "Ghost Player", df))

    def test_wrong_week_is_none(self):
        # sw resolves to week 3 but the player's only row is week 2 -> pending.
        df = _df([{"player_norm": scan._norm("Player One"), "week": 2, "receptions": 5.0}])
        self.assertIsNone(self._run("player_receptions", "Player One", df))

    def test_unresolvable_week_is_none(self):
        df = _df([{"player_norm": scan._norm("Player One"), "week": 3, "receptions": 5.0}])
        self.assertIsNone(self._run("player_receptions", "Player One", df, sw=None))

    def test_none_feed_is_none(self):
        with patch("nfl_schedule.season_week_for_date", return_value=("2026", 3)), \
                patch("nfl_opportunity_serving._load", return_value=None):
            self.assertIsNone(rc._resolve_nfl_actual("player_receptions", "X", "2026-09-21"))

    def test_nickname_resolves_in_grading(self):
        # odds feed "Joshua Palmer" grades against nflverse "Josh Palmer".
        df = _df([{"player_norm": scan._norm("Josh Palmer"), "week": 3,
                   "receiving_yards": 43.0}])
        self.assertEqual(self._run("player_reception_yds", "Joshua Palmer", df), 43.0)


class CanonicalNormTests(unittest.TestCase):
    """nfl_opportunity_serving._canonical_norm — the shared exact-then-nickname resolver
    used by BOTH serving and grading."""

    def _df(self, norms):
        return pd.DataFrame({"player_norm": norms, "week": [1] * len(norms)})

    def test_exact_match(self):
        self.assertEqual(nos._canonical_norm(self._df(["josh allen", "jared goff"]),
                                             "josh allen"), "josh allen")

    def test_nickname_truncation_forward(self):
        self.assertEqual(nos._canonical_norm(self._df(["josh palmer", "davante adams"]),
                                             "joshua palmer"), "josh palmer")

    def test_nickname_truncation_reverse(self):
        self.assertEqual(nos._canonical_norm(self._df(["joshua palmer"]),
                                             "josh palmer"), "joshua palmer")

    def test_non_prefix_decoy_rejected(self):
        # "mike williams" must NOT bind "marvin williams" (neither first name a prefix).
        self.assertIsNone(nos._canonical_norm(self._df(["marvin williams"]), "mike williams"))

    def test_ambiguous_abstains(self):
        self.assertIsNone(nos._canonical_norm(self._df(["josh allen", "jordan allen"]),
                                              "jo allen"))

    def test_absent_returns_none(self):
        self.assertIsNone(nos._canonical_norm(self._df(["josh allen"]), "davante adams"))

    def test_single_token_returns_none(self):
        self.assertIsNone(nos._canonical_norm(self._df(["josh allen"]), "cher"))

    def test_none_df_returns_none(self):
        self.assertIsNone(nos._canonical_norm(None, "josh allen"))


class DidPlayTests(unittest.TestCase):
    """nfl_opportunity_serving.did_play — the snap-count DNP signal."""

    def _snaps(self, rows):
        return pd.DataFrame(rows)

    def test_snap_row_present_is_true(self):
        df = self._snaps([{"player_norm": scan._norm("Bijan Robinson"), "week": 4,
                           "offense_snaps": 55}])
        with patch("nfl_opportunity_serving._load_snaps", return_value=df):
            self.assertIs(nos.did_play(scan._norm("Bijan Robinson"), 2026, 4), True)

    def test_week_posted_player_absent_is_false(self):
        df = self._snaps([{"player_norm": scan._norm("Someone Else"), "week": 4,
                           "offense_snaps": 10}])
        with patch("nfl_opportunity_serving._load_snaps", return_value=df):
            self.assertIs(nos.did_play(scan._norm("Tyler Higbee"), 2026, 4), False)

    def test_week_not_posted_is_none(self):
        df = self._snaps([{"player_norm": scan._norm("X"), "week": 3, "offense_snaps": 5}])
        with patch("nfl_opportunity_serving._load_snaps", return_value=df):
            self.assertIsNone(nos.did_play(scan._norm("X"), 2026, 4))   # week 4 not present

    def test_feed_unavailable_is_none(self):
        with patch("nfl_opportunity_serving._load_snaps", return_value=None):
            self.assertIsNone(nos.did_play(scan._norm("X"), 2026, 4))

    def test_nickname_counts_as_played(self):
        df = self._snaps([{"player_norm": scan._norm("Josh Palmer"), "week": 4,
                           "offense_snaps": 40}])
        with patch("nfl_opportunity_serving._load_snaps", return_value=df):
            self.assertIs(nos.did_play(scan._norm("Joshua Palmer"), 2026, 4), True)


class NflDnpVoidTests(unittest.TestCase):
    """recalibration._nfl_is_dnp: void a graded NFL prop only when the game is >=24h old
    AND snap counts confirm the player took no snap that (regular-season) week."""

    OLD = "2020-09-20T17:00:00Z"          # far past -> age >> STALE_DNP_HOURS

    def _void(self, did_play_val, prop="player_receptions", commence=OLD, sw=("2026", 4)):
        with patch("nfl_schedule.season_week_for_date", return_value=sw), \
                patch("nfl_opportunity_serving.did_play", return_value=did_play_val):
            return rc._nfl_is_dnp(prop, "Tyler Higbee", "2026-09-28", commence)

    def test_confirmed_dnp_voids(self):
        self.assertTrue(self._void(False))

    def test_played_does_not_void(self):
        self.assertFalse(self._void(True))

    def test_unknown_does_not_void(self):
        self.assertFalse(self._void(None))

    def test_unmapped_prop_does_not_void(self):
        self.assertFalse(self._void(False, prop="player_kicking_points"))

    def test_recent_game_does_not_void(self):
        recent = datetime.now(timezone.utc).isoformat()
        self.assertFalse(self._void(False, commence=recent))

    def test_playoff_week_does_not_void(self):
        self.assertFalse(self._void(False, sw=("2026", 20)))

    def test_routes_through_is_stale_dnp(self):
        with patch("nfl_schedule.season_week_for_date", return_value=("2026", 4)), \
                patch("nfl_opportunity_serving.did_play", return_value=False):
            self.assertTrue(rc._is_stale_dnp("americanfootball_nfl", "player_receptions",
                                             "Tyler Higbee", "2026-09-28", self.OLD))

    def test_mlb_path_unaffected(self):
        # NFL branch must not disturb the MLB stale-DNP routing.
        with patch("recalibration._nfl_is_dnp") as nfl:
            rc._is_stale_dnp("baseball_mlb", "batter_hits", "X", "2026-08-09", self.OLD)
        nfl.assert_not_called()


class SeasonWeekForDateTests(unittest.TestCase):
    def _patch_games(self, recs):
        return patch("nfl_schedule.load_games", side_effect=lambda seasons=None, **k: recs)

    def test_exact_gameday(self):
        recs = [{"season": "2026", "week": "3", "gameday": "2026-09-21"},
                {"season": "2026", "week": "2", "gameday": "2026-09-14"}]
        with self._patch_games(recs):
            self.assertEqual(nfl_schedule.season_week_for_date("2026-09-21"), ("2026", 3))

    def test_plus_minus_one_day_slippage(self):
        recs = [{"season": "2026", "week": "3", "gameday": "2026-09-21"}]
        with self._patch_games(recs):
            self.assertEqual(nfl_schedule.season_week_for_date("2026-09-22"), ("2026", 3))

    def test_no_game_near_date_is_none(self):
        recs = [{"season": "2026", "week": "3", "gameday": "2026-09-21"}]
        with self._patch_games(recs):
            self.assertIsNone(nfl_schedule.season_week_for_date("2026-10-15"))

    def test_january_belongs_to_prior_season(self):
        # a Jan 2027 playoff game is season 2026, week 19.
        recs = [{"season": "2026", "week": "19", "gameday": "2027-01-10"}]
        with self._patch_games(recs):
            self.assertEqual(nfl_schedule.season_week_for_date("2027-01-10"), ("2026", 19))

    def test_bad_date_is_none(self):
        with self._patch_games([]):
            self.assertIsNone(nfl_schedule.season_week_for_date("not-a-date"))


class ResolveOnePropWiringTests(unittest.TestCase):
    """resolve_one_prop wiring: gated OFF by default (ESPN-primary); when the gate is
    ON, nflverse is preferred with ESPN as the fallback on a miss."""

    def _resolve(self):
        return rc.resolve_one_prop(
            "americanfootball_nfl", "Player", "player_receptions", 5.5,
            "2026-09-21", "2026-09-21T17:00:00Z")

    def test_default_on_prefers_nflverse(self):
        # Post-flip (2026-09-25, parity clean): default gate ON -> nflverse is primary.
        self.assertTrue(rc._nfl_grade_from_nflverse())      # documents the default
        with patch("recalibration._resolve_mlb_actual", return_value=None), \
                patch("recalibration._resolve_nfl_actual", return_value=42.0) as nfl, \
                patch("recalibration._load_player_gamelog") as espn:
            out = self._resolve()
        self.assertEqual(out, 42.0)
        nfl.assert_called_once()
        espn.assert_not_called()          # nflverse primary -> ESPN not touched on a hit

    def test_kill_switch_env_forces_espn(self):
        # ODI_NFL_GRADE_NFLVERSE=0 reverts to ESPN-primary even with the default ON.
        import os
        with patch.dict(os.environ, {"ODI_NFL_GRADE_NFLVERSE": "0"}), \
                patch("recalibration._resolve_mlb_actual", return_value=None), \
                patch("recalibration._resolve_nfl_actual", return_value=42.0) as nfl, \
                patch("recalibration._load_player_gamelog", return_value=(None, {})) as espn:
            out = self._resolve()
        self.assertIsNone(out)            # ESPN path (empty) -> pending
        nfl.assert_not_called()           # kill-switch -> nflverse skipped entirely
        espn.assert_called_once()

    def test_gate_on_prefers_nflverse_and_skips_espn_on_hit(self):
        with patch("recalibration._nfl_grade_from_nflverse", return_value=True), \
                patch("recalibration._resolve_mlb_actual", return_value=None), \
                patch("recalibration._resolve_nfl_actual", return_value=42.0) as nfl, \
                patch("recalibration._load_player_gamelog") as espn:
            out = self._resolve()
        self.assertEqual(out, 42.0)
        nfl.assert_called_once()
        espn.assert_not_called()          # nflverse hit -> never touches the ESPN path

    def test_gate_on_falls_back_to_espn_on_nflverse_miss(self):
        with patch("recalibration._nfl_grade_from_nflverse", return_value=True), \
                patch("recalibration._resolve_mlb_actual", return_value=None), \
                patch("recalibration._resolve_nfl_actual", return_value=None) as nfl, \
                patch("recalibration._load_player_gamelog", return_value=(None, {})) as espn:
            out = self._resolve()
        self.assertIsNone(out)            # ESPN also empty -> stays pending
        nfl.assert_called_once()
        espn.assert_called_once()         # fell through to the ESPN fallback

    def test_env_var_overrides_default(self):
        import os
        with patch.dict(os.environ, {"ODI_NFL_GRADE_NFLVERSE": "1"}):
            self.assertTrue(rc._nfl_grade_from_nflverse())
        with patch.dict(os.environ, {"ODI_NFL_GRADE_NFLVERSE": "0"}):
            self.assertFalse(rc._nfl_grade_from_nflverse())


if __name__ == "__main__":
    unittest.main(verbosity=2)
