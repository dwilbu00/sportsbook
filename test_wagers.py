"""Tests for the actual-bets ledger (Submit Picks -> realized ROI).

Hermetic: no live statsapi / ESPN / Azure. Row construction from analysis
candidates, team-market grading math, the profit formula, the NDJSON append +
grade round-trip (local tempdir store), and the stake-weighted ROI summary are
all exercised on fixtures with the resolvers mocked.
"""
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import game_results
import pricing_common
import recalibration
import wagers

_META = {
    "sport_key": "baseball_mlb",
    "event_id": "E1",
    "commence_time": "2026-07-16T18:00:00Z",
    "game_date": "2026-07-16",
    "home_team": "Boston Red Sox",
    "away_team": "New York Yankees",
    "stake": 10.0,
    "placed_at": "2026-07-16T12:00:00+00:00",
    "seq": 0,
}


def _meta(seq=0):
    m = dict(_META)
    m["seq"] = seq
    return m


class BuildWagerRowTests(unittest.TestCase):
    def test_moneyline_row(self):
        cand = {"team": "Boston Red Sox", "opponent": "New York Yankees",
                "home_away": "HOME", "best_price": 120, "best_book": "DK",
                "blended_prob": 58.0, "best_edge_pct": 6.0, "event_id": "E1"}
        row = wagers.build_wager_row("moneyline", None, cand, _meta())
        self.assertEqual(row["bet_type"], "moneyline")
        self.assertEqual(row["side"], "home")
        self.assertEqual(row["executed_price"], 120)
        self.assertEqual(row["model_price"], 120)
        self.assertEqual(row["stake"], 10.0)
        self.assertAlmostEqual(row["model_prob"], 0.58)
        self.assertEqual(row["game_date"], "2026-07-16")
        self.assertEqual(row["home_team"], "Boston Red Sox")
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["wager_id"], "2026-07-16T12:00:00+00:00#0")

    def test_spread_row(self):
        cand = {"team": "Boston Red Sox", "opponent": "New York Yankees",
                "home_away": "AWAY", "spread": -1.5, "price": -110,
                "cover_rate": 55.0, "edge_pct": 4.0, "event_id": "E1"}
        row = wagers.build_wager_row("spread", None, cand, _meta())
        self.assertEqual(row["side"], "away")
        self.assertEqual(row["point"], -1.5)
        self.assertEqual(row["executed_price"], -110)
        self.assertAlmostEqual(row["model_prob"], 0.55)

    def test_total_over_and_under_rows(self):
        cand = {"matchup": "New York Yankees @ Boston Red Sox", "line": 8.5,
                "over_price": -105, "under_price": -115, "over_hit_rate": 57.0,
                "over_edge_pct": 5.0, "under_edge_pct": -2.0, "event_id": "E1"}
        over = wagers.build_wager_row("total", "OVER", cand, _meta())
        self.assertEqual(over["side"], "over")
        self.assertEqual(over["point"], 8.5)
        self.assertEqual(over["executed_price"], -105)
        self.assertAlmostEqual(over["model_prob"], 0.57)
        under = wagers.build_wager_row("total", "UNDER", cand, _meta(1))
        self.assertEqual(under["side"], "under")
        self.assertEqual(under["executed_price"], -115)
        self.assertAlmostEqual(under["model_prob"], 0.43)

    def test_player_prop_row(self):
        cand = {"player": "Rafael Devers", "prop": "batter_hits",
                "prop_label": "Hits", "line": 1.5, "direction": "OVER",
                "over_price": -110, "under_price": -110, "over_rate": 60.0,
                "edge_pct": 7.0, "matchup": "NYY @ BOS",
                "team": "Boston Red Sox", "event_id": "E1"}
        row = wagers.build_wager_row("player_prop", None, cand, _meta())
        self.assertEqual(row["player"], "Rafael Devers")
        self.assertEqual(row["prop_key"], "batter_hits")
        self.assertEqual(row["line"], 1.5)
        self.assertEqual(row["executed_price"], -110)
        self.assertEqual(row["direction"], "OVER")
        self.assertAlmostEqual(row["model_prob"], 0.60)

    def test_safe_mode_prop_uses_alt_line_and_price(self):
        cand = {"player": "Aaron Judge", "prop": "batter_hits",
                "prop_label": "Hits", "safe_mode": True, "safe_alt_line": 0.5,
                "safe_alt_price": -200, "model_hit_at_safe": 80.0,
                "direction": "OVER", "edge_pct": 5.0, "matchup": "NYY @ BOS",
                "team": "New York Yankees", "event_id": "E1"}
        row = wagers.build_wager_row("player_prop", None, cand, _meta())
        self.assertEqual(row["line"], 0.5)
        self.assertEqual(row["executed_price"], -200)
        self.assertEqual(row["direction"], "OVER")
        self.assertAlmostEqual(row["model_prob"], 0.80)


class GradeTeamBetTests(unittest.TestCase):
    def test_moneyline(self):
        self.assertEqual(
            game_results.grade_team_bet("moneyline", "home", None, 5, 3), "won")
        self.assertEqual(
            game_results.grade_team_bet("moneyline", "away", None, 5, 3), "lost")

    def test_spread(self):
        # Home -1.5, wins by 2 -> covers.
        self.assertEqual(
            game_results.grade_team_bet("spread", "home", -1.5, 5, 3), "won")
        # Home -2.0, wins by exactly 2 -> push.
        self.assertEqual(
            game_results.grade_team_bet("spread", "home", -2.0, 5, 3), "push")
        # Away +1.5, loses by 2 -> does not cover.
        self.assertEqual(
            game_results.grade_team_bet("spread", "away", 1.5, 5, 3), "lost")

    def test_total(self):
        self.assertEqual(
            game_results.grade_team_bet("total", "over", 7.5, 5, 3), "won")
        self.assertEqual(
            game_results.grade_team_bet("total", "under", 8.5, 5, 3), "won")
        self.assertEqual(
            game_results.grade_team_bet("total", "over", 8.0, 5, 3), "push")

    def test_bad_inputs_return_none(self):
        self.assertIsNone(
            game_results.grade_team_bet("total", "over", None, 5, 3))
        self.assertIsNone(
            game_results.grade_team_bet("spread", "home", -1.5, None, 3))


class ProfitTests(unittest.TestCase):
    def test_win_loss_push(self):
        self.assertAlmostEqual(pricing_common.profit(120, 10, True), 12.0)
        self.assertAlmostEqual(pricing_common.profit(-110, 10, False), -10.0)
        self.assertAlmostEqual(pricing_common.profit(-110, 10, None), 0.0)

    def test_negative_price_win(self):
        self.assertAlmostEqual(
            pricing_common.profit(-200, 10, True), 5.0)


class _LocalLedger:
    """Context manager: force wagers/recalibration onto a tempdir NDJSON store."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()

    def __enter__(self):
        self._p1 = patch.object(recalibration, "PRED_DIR", self._tmp.name)
        self._p2 = patch.object(recalibration, "_prediction_log_blob_url",
                                return_value="")
        self._p1.start(); self._p2.start()
        return self

    def __exit__(self, *exc):
        self._p1.stop(); self._p2.stop()
        self._tmp.cleanup()


class RoundTripAndGradeTests(unittest.TestCase):
    def test_submit_read_and_resolve(self):
        prop = wagers.build_wager_row("player_prop", None, {
            "player": "Rafael Devers", "prop": "batter_hits",
            "prop_label": "Hits", "line": 1.5, "direction": "OVER",
            "over_price": -110, "over_rate": 60.0, "edge_pct": 7.0,
            "matchup": "NYY @ BOS", "team": "Boston Red Sox", "event_id": "E1"},
            _meta(0))
        ml = wagers.build_wager_row("moneyline", None, {
            "team": "Boston Red Sox", "opponent": "New York Yankees",
            "home_away": "HOME", "best_price": 120, "best_book": "DK",
            "blended_prob": 58.0, "best_edge_pct": 6.0, "event_id": "E1"},
            _meta(1))

        with _LocalLedger():
            self.assertEqual(wagers.submit_wagers([prop, ml]), 2)
            # Idempotent: re-submitting the same wager_ids adds nothing.
            self.assertEqual(wagers.submit_wagers([prop, ml]), 0)
            self.assertEqual(len(wagers.read_wagers()), 2)

            now = datetime(2026, 7, 20, tzinfo=timezone.utc)
            with patch.object(recalibration, "resolve_one_prop",
                              return_value=2.0), \
                 patch.object(game_results, "final_score", return_value=(5, 3)):
                graded = wagers.resolve_pending_wagers(now=now)
            self.assertEqual(graded, 2)

            rows = {r["bet_type"]: r for r in wagers.read_wagers()}
            self.assertEqual(rows["player_prop"]["status"], "won")
            self.assertAlmostEqual(rows["player_prop"]["profit"],
                                   pricing_common.profit(-110, 10, True))
            self.assertEqual(rows["moneyline"]["status"], "won")
            self.assertAlmostEqual(rows["moneyline"]["profit"], 12.0)

    def test_pending_future_game_not_graded(self):
        row = wagers.build_wager_row("moneyline", None, {
            "team": "Boston Red Sox", "opponent": "New York Yankees",
            "home_away": "HOME", "best_price": 120, "event_id": "E1"},
            _meta(0))
        with _LocalLedger():
            wagers.submit_wagers([row])
            # "now" is before the game date -> nothing gradable yet.
            now = datetime(2026, 7, 15, tzinfo=timezone.utc)
            with patch.object(game_results, "final_score", return_value=(5, 3)):
                self.assertEqual(wagers.resolve_pending_wagers(now=now), 0)
            self.assertEqual(wagers.read_wagers()[0]["status"], "pending")


class SummaryTests(unittest.TestCase):
    def test_stake_weighted_roi(self):
        rows = [
            {"sport_key": "baseball_mlb", "bet_type": "moneyline",
             "status": "won", "stake": 10.0, "profit": 12.0},
            {"sport_key": "baseball_mlb", "bet_type": "player_prop",
             "status": "lost", "stake": 10.0, "profit": -10.0},
            {"sport_key": "baseball_mlb", "bet_type": "total",
             "status": "push", "stake": 10.0, "profit": 0.0},
            {"sport_key": "baseball_mlb", "bet_type": "moneyline",
             "status": "pending", "stake": 10.0, "profit": None},
        ]
        summary = wagers.summarize_wagers(rows)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["resolved"], 3)
        self.assertEqual(summary["pending"], 1)
        self.assertAlmostEqual(summary["total_staked"], 30.0)
        self.assertAlmostEqual(summary["realized_profit"], 2.0)
        self.assertAlmostEqual(summary["roi"], 2.0 / 30.0)
        self.assertEqual((summary["won"], summary["lost"], summary["push"]),
                         (1, 1, 1))
        self.assertAlmostEqual(summary["hit_rate"], 0.5)
        self.assertAlmostEqual(summary["pending_stake"], 10.0)
        self.assertTrue(summary["by_bet_type"])
        self.assertTrue(summary["by_sport"])


if __name__ == "__main__":
    unittest.main()
