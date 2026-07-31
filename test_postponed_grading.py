"""Postponed / suspended / cancelled games must never grade a bet.

statsapi can report abstractGameState "Final" for a game that never truly
completed (a postponed game surfaces as "Final" with a 0-0 line; a suspended
game with a partial box score). Three grading paths trusted that flag and would
settle a rained-out game's bets as WIN/LOSS off a bogus line:

  1. mlb_starters.resolve_player_game_stat  (statsapi hard-ID prop resolver)
  2. game_results._mlb_scores_for_date       (team-market score reader)
  3. recalibration.resolve_one_prop          (ESPN ±1-day fallback)

These pin the fix: only a genuine completion (abstractGameState "Final" AND a
detailedState that is not postponed/suspended/cancelled) grades; a
rain-SHORTENED but official game ("Completed Early") still grades.
"""

import unittest
from unittest.mock import patch

import mlb_starters
import game_results
import recalibration


class IsGenuineFinalTests(unittest.TestCase):
    def test_final_final_is_genuine(self):
        self.assertTrue(mlb_starters._is_genuine_final(
            {"status": "Final", "detailedState": "Final"}))

    def test_completed_early_still_grades(self):
        # A rain-shortened OFFICIAL game — real result, must grade.
        self.assertTrue(mlb_starters._is_genuine_final(
            {"status": "Final", "detailedState": "Completed Early"}))

    def test_missing_detailed_trusts_abstract(self):
        # Older cached index (pre-upgrade) has no detailedState.
        self.assertTrue(mlb_starters._is_genuine_final({"status": "Final"}))

    def test_postponed_not_genuine(self):
        self.assertFalse(mlb_starters._is_genuine_final(
            {"status": "Final", "detailedState": "Postponed"}))

    def test_suspended_not_genuine(self):
        self.assertFalse(mlb_starters._is_genuine_final(
            {"status": "Final", "detailedState": "Suspended: Rain"}))

    def test_cancelled_not_genuine(self):
        self.assertFalse(mlb_starters._is_genuine_final(
            {"status": "Final", "detailedState": "Cancelled"}))

    def test_live_not_genuine(self):
        self.assertFalse(mlb_starters._is_genuine_final(
            {"status": "Live", "detailedState": "In Progress"}))

    def test_none_not_genuine(self):
        self.assertFalse(mlb_starters._is_genuine_final(None))

    def test_all_final_false_when_one_postponed(self):
        idx = {"1": {"status": "Final", "detailedState": "Final"},
               "2": {"status": "Final", "detailedState": "Postponed"}}
        self.assertFalse(mlb_starters._all_final(idx))


class ResolvePlayerGameStatGateTests(unittest.TestCase):
    """The statsapi hard-ID prop resolver keeps a suspended game PENDING even
    though it carries a partial box score reporting abstractGameState 'Final'."""

    SPLITS = [{"game": {"gamePk": 777}, "stat": {"hits": 1}}]  # 1 hit so far

    def _resolve(self, detailed):
        index = {"777": {"gameDate": "2025-07-01T23:00:00Z",
                         "status": "Final", "detailedState": detailed}}
        with patch.object(mlb_starters, "find_player_id",
                          return_value=("123", False)), \
             patch.object(mlb_starters, "_player_gamelog_splits",
                          return_value=self.SPLITS), \
             patch.object(mlb_starters, "get_schedule_index",
                          side_effect=lambda d: index):
            return mlb_starters.resolve_player_game_stat(
                "B. Batter", "2025-07-01T23:00:00Z", "2025-07-01",
                "hitting", "hits", 2025)

    def test_suspended_returns_pending(self):
        self.assertIs(self._resolve("Suspended: Rain"),
                      mlb_starters.GAME_NOT_FINAL)

    def test_postponed_returns_pending(self):
        self.assertIs(self._resolve("Postponed"), mlb_starters.GAME_NOT_FINAL)

    def test_genuine_final_grades(self):
        self.assertEqual(self._resolve("Final"), 1.0)


class MlbScoresExcludePostponedTests(unittest.TestCase):
    """The team-market score reader drops postponed/suspended games so a
    rained-out game's moneyline/spread/total bets stay pending."""

    def _game(self, detailed, hs, as_):
        return {
            "status": {"abstractGameState": "Final", "detailedState": detailed},
            "teams": {"home": {"team": {"name": "Guardians"}, "score": hs},
                      "away": {"team": {"name": "Tigers"}, "score": as_}},
            "gameDate": "2025-07-01T23:00:00Z",
        }

    def _live_game(self, hs, as_):
        return {
            "status": {"abstractGameState": "Live", "detailedState": "In Progress"},
            "teams": {"home": {"team": {"name": "Dodgers"}, "score": hs},
                      "away": {"team": {"name": "Padres"}, "score": as_}},
            "gameDate": "2025-07-02T02:00:00Z",  # late (west-coast) game
        }

    def _scores(self, *games):
        data = {"dates": [{"games": list(games)}]}
        with patch.object(mlb_starters, "_read_cache", return_value=None), \
             patch.object(mlb_starters, "_write_cache", return_value=None), \
             patch.object(mlb_starters, "_get", return_value=data):
            return game_results._mlb_scores_for_date("2025-07-01")

    def _slate(self, *games):
        data = {"dates": [{"games": list(games)}]}
        with patch.object(mlb_starters, "_read_cache", return_value=None), \
             patch.object(mlb_starters, "_write_cache", return_value=None), \
             patch.object(mlb_starters, "_get", return_value=data):
            return game_results._mlb_slate_for_date("2025-07-01")

    def test_slate_complete_when_all_final(self):
        games, complete = self._slate(self._game("Final", 5, 3))
        self.assertTrue(complete)          # every game final → immutable slate
        self.assertEqual(len(games), 1)

    def test_slate_incomplete_when_a_game_is_live(self):
        games, complete = self._slate(self._game("Final", 5, 3),
                                      self._live_game(2, 1))
        self.assertFalse(complete)         # a live game → slate can still change
        self.assertEqual(len(games), 1)    # only the final game has a usable score

    def test_slate_incomplete_when_postponed(self):
        games, complete = self._slate(self._game("Postponed", 0, 0),
                                      self._game("Final", 5, 3))
        self.assertFalse(complete)         # postponed makeup lands on another date
        self.assertEqual(len(games), 1)

    def test_empty_slate_is_not_complete(self):
        games, complete = self._slate()
        self.assertEqual(games, [])
        self.assertFalse(complete)         # no games ≠ immutable (guards bad fetch)

    def test_postponed_excluded(self):
        rows = self._scores(self._game("Postponed", 0, 0))
        self.assertEqual(rows, [])

    def test_suspended_excluded(self):
        rows = self._scores(self._game("Suspended", 3, 2))
        self.assertEqual(rows, [])

    def test_genuine_final_included(self):
        rows = self._scores(self._game("Final", 5, 3))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["home_score"], 5.0)

    def test_mixed_slate_keeps_only_genuine(self):
        rows = self._scores(self._game("Postponed", 0, 0),
                            self._game("Final", 5, 3))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["home_score"], 5.0)


class EspnAdjacentGuardTests(unittest.TestCase):
    """resolve_one_prop's ±1-day ESPN fallback must not grade a postponed game
    against the prior night's DIFFERENT game (~24h away); genuine UTC/local date
    slippage (same start time) still resolves."""

    def _resolve(self, gamelog, by_date, game_date, commence):
        with patch.object(recalibration, "_resolve_mlb_actual",
                          return_value=None), \
             patch.object(recalibration, "_load_player_gamelog",
                          return_value=(gamelog, by_date)):
            return recalibration.resolve_one_prop(
                "baseball_mlb", "B", "batter_hits", 0.5, game_date, commence)

    def test_adjacent_prior_night_not_graded(self):
        # Bet on a game postponed on 2025-07-02 (~23:10Z). ESPN has no row that
        # day, but the player DID play the prior night (2025-07-01 ~23:10Z).
        gamelog = [{"H": 2.0, "game_date": "2025-07-01T23:10:00Z",
                    "completed": True}]
        by_date = {"2025-07-01": [("2025-07-01T23:10:00Z", 0)]}
        actual = self._resolve(gamelog, by_date, "2025-07-02",
                               "2025-07-02T23:10:00Z")
        self.assertIsNone(actual)  # ~24h away -> rejected -> stays pending

    def test_genuine_slippage_resolves(self):
        # Same physical game, filed a calendar day off (commence == row start).
        gamelog = [{"H": 2.0, "game_date": "2025-07-01T23:30:00Z",
                    "completed": True}]
        by_date = {"2025-07-01": [("2025-07-01T23:30:00Z", 0)]}
        actual = self._resolve(gamelog, by_date, "2025-07-02",
                               "2025-07-02T00:00:00Z")
        self.assertEqual(actual, 2.0)  # 30min away -> genuine slippage -> graded


if __name__ == "__main__":
    unittest.main()
