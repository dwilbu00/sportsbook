"""Tests for the game_pk team-market grading fast path (Tier A #2).

Commit 1 covers the two new primitives (mlb_warehouse.final_game_by_pk +
game_results.grade_team_bet_by_game_pk / GRADE_PENDING). All hermetic — no SQL, no
network: final_game_by_pk is exercised by patching get_game (snake_case rows), the
grader by patching final_game_by_pk + team_id_for_name_tolerant.
"""
import unittest
from unittest.mock import patch

import game_results as gr
import mlb_warehouse as mw


class FinalGameByPkTests(unittest.TestCase):
    def _game(self, **kw):
        base = {"game_pk": 1, "status": "Final", "detailed_state": "Final",
                "home_score": 5.0, "away_score": 3.0,
                "home_team_id": "111", "away_team_id": "222",
                "game_date": "2024-04-01T23:05:00Z"}
        base.update(kw)
        return base

    def test_final(self):
        with patch.object(mw, "get_game", return_value=self._game()):
            fg = mw.final_game_by_pk(1)
        self.assertEqual(fg["state"], "final")
        self.assertEqual(fg["home_score"], 5.0)
        self.assertEqual(fg["home_team_id"], "111")
        self.assertEqual(fg["commence_time"], "2024-04-01T23:05:00Z")

    def test_legit_zero_zero_is_final(self):
        # A real 0-0 must grade — the check is `is None`, not falsy.
        with patch.object(mw, "get_game",
                          return_value=self._game(home_score=0.0, away_score=0.0)):
            self.assertEqual(mw.final_game_by_pk(1)["state"], "final")

    def test_terminal_postponed(self):
        # status can read 'Final' with a postponed detailed_state — snake_case; must
        # NOT pass as final (the _is_genuine_final camelCase trap).
        with patch.object(mw, "get_game",
                          return_value=self._game(detailed_state="Postponed")):
            self.assertEqual(mw.final_game_by_pk(1)["state"], "terminal")

    def test_terminal_suspended_and_cancelled(self):
        for ds in ("Suspended: Rain", "Cancelled"):
            with patch.object(mw, "get_game",
                              return_value=self._game(detailed_state=ds)):
                self.assertEqual(mw.final_game_by_pk(1)["state"], "terminal")

    def test_live_when_in_progress(self):
        with patch.object(mw, "get_game",
                          return_value=self._game(status="In Progress",
                                                  detailed_state="In Progress",
                                                  home_score=None, away_score=None)):
            self.assertEqual(mw.final_game_by_pk(1)["state"], "live")

    def test_final_status_but_score_missing_is_live(self):
        with patch.object(mw, "get_game",
                          return_value=self._game(home_score=None)):
            self.assertEqual(mw.final_game_by_pk(1)["state"], "live")

    def test_none_when_no_row(self):
        with patch.object(mw, "get_game", return_value=None):
            self.assertIsNone(mw.final_game_by_pk(999))


class GradeByGamePkTests(unittest.TestCase):
    _FINAL = {"state": "final", "home_score": 5.0, "away_score": 3.0,
              "home_team_id": "111", "away_team_id": "222",
              "commence_time": "2024-04-01T23:05:00Z"}

    def test_non_mlb_returns_none(self):
        self.assertIsNone(gr.grade_team_bet_by_game_pk(
            "basketball_nba", 1, "moneyline", "home", "X", None))

    def test_no_game_pk_returns_none(self):
        self.assertIsNone(gr.grade_team_bet_by_game_pk(
            "baseball_mlb", None, "moneyline", "home", "X", None))

    def test_total_is_orientation_invariant(self):
        # 5+3=8 over 7.5 -> won; graded without any team_id resolution.
        with patch.object(mw, "final_game_by_pk", return_value=self._FINAL):
            self.assertEqual(
                gr.grade_team_bet_by_game_pk("baseball_mlb", 1, "total", "over",
                                             None, 7.5),
                ("won", "5-3"))

    def test_moneyline_home_from_team_id(self):
        with patch.object(mw, "final_game_by_pk", return_value=self._FINAL), \
             patch.object(mw, "team_id_for_name_tolerant", return_value="111"):
            self.assertEqual(
                gr.grade_team_bet_by_game_pk("baseball_mlb", 1, "moneyline", None,
                                             "Home Team", None),
                ("won", "5-3"))            # home won 5-3

    def test_moneyline_away_from_team_id(self):
        with patch.object(mw, "final_game_by_pk", return_value=self._FINAL), \
             patch.object(mw, "team_id_for_name_tolerant", return_value="222"):
            self.assertEqual(
                gr.grade_team_bet_by_game_pk("baseball_mlb", 1, "moneyline", None,
                                             "Away Team", None),
                ("lost", "5-3"))           # away lost

    def test_unmappable_team_falls_back(self):
        with patch.object(mw, "final_game_by_pk", return_value=self._FINAL), \
             patch.object(mw, "team_id_for_name_tolerant", return_value=None):
            self.assertIsNone(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "moneyline", None, "???", None))

    def test_live_fresh_returns_pending(self):
        fresh = dict(self._FINAL, state="live",
                     commence_time="2099-01-01T00:00:00Z")   # future -> not stale
        with patch.object(mw, "final_game_by_pk", return_value=fresh):
            self.assertIs(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "moneyline", "home", "X", None),
                gr.GRADE_PENDING)

    def test_live_but_stale_falls_back(self):
        stale = dict(self._FINAL, state="live",
                     commence_time="2020-01-01T00:00:00Z")   # long past -> stale
        with patch.object(mw, "final_game_by_pk", return_value=stale):
            self.assertIsNone(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "moneyline", "home", "X", None))

    def test_terminal_falls_back(self):
        with patch.object(mw, "final_game_by_pk",
                          return_value=dict(self._FINAL, state="terminal")):
            self.assertIsNone(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "moneyline", "home", "X", None))

    def test_no_warehouse_row_falls_back(self):
        with patch.object(mw, "final_game_by_pk", return_value=None):
            self.assertIsNone(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "total", "over", None, 7.5))

    def test_distinct_pks_grade_distinct_scores(self):
        # DH exactness: the score comes from the resolved game, not name+date.
        g1 = dict(self._FINAL, home_score=5.0, away_score=3.0)
        g2 = dict(self._FINAL, home_score=1.0, away_score=2.0)
        with patch.object(mw, "final_game_by_pk", side_effect=lambda pk: g1 if pk == 1 else g2):
            self.assertEqual(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "total", "over", None, 7.5), ("won", "5-3"))
            self.assertEqual(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 2, "total", "over", None, 7.5), ("lost", "1-2"))


class GradeWagerRoutingTests(unittest.TestCase):
    """wagers._grade_wager routing: fast path vs the unchanged name+date fallback."""
    def _row(self, **kw):
        base = {"sport_key": "baseball_mlb", "bet_type": "moneyline", "game_pk": 1,
                "side": "home", "team": "NYY", "point": None,
                "commence_time": "2024-04-01T23:05:00Z", "game_date": "2024-04-01"}
        base.update(kw)
        return base

    def test_fast_result_short_circuits_name_date(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk",
                          return_value=("won", "5-3")), \
             patch.object(gr, "final_score") as fsm:
            self.assertEqual(wagers._grade_wager(self._row()), ("won", "5-3"))
            fsm.assert_not_called()          # game_pk path won -> never hit name+date

    def test_pending_stays_pending_no_fallback(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk",
                          return_value=gr.GRADE_PENDING), \
             patch.object(gr, "final_score") as fsm:
            self.assertIsNone(wagers._grade_wager(self._row()))
            fsm.assert_not_called()          # critical: do NOT fall back on PENDING

    def test_fast_none_falls_back_to_name_date(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk", return_value=None), \
             patch.object(gr, "final_score", return_value=(5.0, 3.0)) as fsm, \
             patch.object(gr, "side_for_team", return_value="home"):
            self.assertEqual(wagers._grade_wager(self._row()), ("won", "5-3"))
            fsm.assert_called_once()          # fell through to name+date

    def test_absent_game_pk_skips_fast_path(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk") as fastm, \
             patch.object(gr, "final_score", return_value=(5.0, 3.0)), \
             patch.object(gr, "side_for_team", return_value="home"):
            self.assertEqual(wagers._grade_wager(self._row(game_pk=None)),
                             ("won", "5-3"))
            fastm.assert_not_called()          # byte-identical to pre-#2

    def test_non_mlb_skips_fast_path(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk") as fastm, \
             patch.object(gr, "final_score", return_value=(110.0, 100.0)), \
             patch.object(gr, "side_for_team", return_value="home"):
            wagers._grade_wager(self._row(sport_key="basketball_nba",
                                          bet_type="moneyline"))
            fastm.assert_not_called()          # NBA never enters the fast path


class WagerEnrichStampTests(unittest.TestCase):
    """wagers._enrich_ids stamps a DH-safe game_pk on MLB team wagers (Tier A #2)."""
    def _row(self, **kw):
        base = {"sport_key": "baseball_mlb", "bet_type": "moneyline",
                "home_team": "NYY", "away_team": "BOS", "team": "NYY",
                "commence_time": "2024-04-01T23:05:00Z"}
        base.update(kw)
        return base

    def test_team_wager_gets_game_pk(self):
        import wagers
        with patch.object(mw, "team_id_for_name_tolerant", side_effect=["111", "222"]), \
             patch.object(mw, "find_game_pk_by_commence", return_value=777) as f:
            row = wagers._enrich_ids(self._row())
        self.assertEqual(row.get("game_pk"), 777)
        f.assert_called_once()

    def test_ambiguous_dh_leaves_game_pk_none(self):
        import wagers
        with patch.object(mw, "team_id_for_name_tolerant", side_effect=["111", "222"]), \
             patch.object(mw, "find_game_pk_by_commence", return_value=None):
            self.assertIsNone(wagers._enrich_ids(self._row()).get("game_pk"))

    def test_missing_commence_not_stamped(self):
        import wagers
        with patch.object(mw, "team_id_for_name_tolerant", side_effect=["111", "222"]), \
             patch.object(mw, "find_game_pk_by_commence") as f:
            self.assertIsNone(
                wagers._enrich_ids(self._row(commence_time=None)).get("game_pk"))
        f.assert_not_called()

    def test_non_baseball_untouched(self):
        import wagers
        with patch.object(mw, "find_game_pk_by_commence") as f:
            self.assertIsNone(
                wagers._enrich_ids(self._row(sport_key="basketball_nba")).get("game_pk"))
        f.assert_not_called()


class WarehouseEnrichStampTests(unittest.TestCase):
    """warehouse._enrich_ids stamps one shared game_pk on all team odds lines."""
    def test_team_lines_share_one_game_pk(self):
        import warehouse
        meta = {"home": "NYY", "away": "BOS", "game_date": "2024-04-01",
                "commence_time": "2024-04-01T23:05:00Z"}
        lines = [{"bet_type": "moneyline", "selection": "NYY"},
                 {"bet_type": "total", "selection": "Over"},
                 {"bet_type": "player_prop", "player": "Aaron Judge",
                  "prop_key": "batter_hits"}]
        with patch.object(mw, "team_id_for_name_tolerant", side_effect=["111", "222"]), \
             patch.object(mw, "find_game_pk_by_commence", return_value=777), \
             patch("entity_resolver.resolve",
                   return_value={"mlb_player_id": "5", "game_pk": 777}), \
             patch("mlb_starters.warm_player_index"):
            _meta, out = warehouse._enrich_ids("baseball_mlb", meta, lines)
        self.assertEqual(out[0]["game_pk"], 777)     # moneyline (team branch)
        self.assertEqual(out[1]["game_pk"], 777)     # total (team branch)
        self.assertEqual(out[2]["game_pk"], 777)     # prop (entity_resolver branch)

    def test_unresolved_team_leaves_lines_null(self):
        import warehouse
        meta = {"home": "NYY", "away": "BOS", "game_date": "2024-04-01",
                "commence_time": "2024-04-01T23:05:00Z"}
        lines = [{"bet_type": "spread", "selection": "NYY"}]
        with patch.object(mw, "team_id_for_name_tolerant", return_value=None), \
             patch.object(mw, "find_game_pk_by_commence") as f, \
             patch("mlb_starters.warm_player_index"):
            _meta, out = warehouse._enrich_ids("baseball_mlb", meta, lines)
        self.assertIsNone(out[0].get("game_pk"))
        f.assert_not_called()                        # both team ids None -> no resolve


if __name__ == "__main__":
    unittest.main()
