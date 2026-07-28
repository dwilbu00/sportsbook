"""Tests for team-market forward tracking (moneyline / spread / total):

- build_market_prediction_rows picks the model's favored side per (event, market)
  with the correct side/point/price/is_value mapping (ML higher blended_prob;
  spread higher cover_rate; total over/under lean).
- log_market_prediction_rows upserts by (sport, event, bet_type) identity: newest
  supersedes an unresolved row, a resolved row is never overwritten.
- resolve_pending_market_outcomes grades via the shared team graders, mapping
  won/lost/push -> 1/0/None, stays pending when final_score is None, and skips
  future games.
- summarize_market_prediction_rows hit-rate / Brier / ROI math.
- Best-effort contract: the builder returns [] and the logger returns 0 on error
  rather than raising (analysis must never break).
"""

import unittest
from unittest.mock import patch

import recalibration


def _ar():
    """A minimal analysis result with one event and all three team markets.

    Favored sides: ML -> Astros (blended 62 > 38); spread -> Astros
    (cover 55 > 45); total -> over (over_hit 58 >= 50)."""
    return {
        "events": {
            "e1": {"commence_time": "2020-05-01T23:00:00Z",
                   "game_date": "2020-05-01",
                   "home_team": "Rockies", "away_team": "Astros"},
        },
        "all_ml": [
            {"event_id": "e1", "type": "moneyline", "team": "Astros",
             "opponent": "Rockies", "home_away": "AWAY", "blended_prob": 62.0,
             "model_prob": 60.0, "best_price": -140, "best_book": "DK",
             "is_value": True},
            {"event_id": "e1", "type": "moneyline", "team": "Rockies",
             "opponent": "Astros", "home_away": "HOME", "blended_prob": 38.0,
             "model_prob": 40.0, "best_price": 120, "best_book": "DK",
             "is_value": False},
        ],
        "all_spreads": [
            {"event_id": "e1", "type": "spread", "team": "Astros",
             "opponent": "Rockies", "home_away": "AWAY", "spread": -1.5,
             "cover_rate": 55.0, "model_cover_rate": 53.0, "price": -110,
             "is_value": False},
            {"event_id": "e1", "type": "spread", "team": "Rockies",
             "opponent": "Astros", "home_away": "HOME", "spread": 1.5,
             "cover_rate": 45.0, "model_cover_rate": 47.0, "price": -110,
             "is_value": False},
        ],
        "all_totals": [
            {"event_id": "e1", "type": "total_over", "matchup": "Astros @ Rockies",
             "line": 9.5, "over_hit_rate": 58.0, "model_over_hit_rate": 56.0,
             "over_price": -105, "under_price": -115,
             "is_over_value": True, "is_under_value": False},
        ],
    }


class _InMemoryLog:
    """Patch mutate_market_prediction_log to run the mutator on a row list."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, mutator, max_retries=5, where=None):
        return mutator(self.rows)


class BuildRowsTests(unittest.TestCase):
    def setUp(self):
        self.rows = recalibration.build_market_prediction_rows(
            _ar(), "baseball_mlb")
        self.by_type = {r["bet_type"]: r for r in self.rows}

    def test_one_row_per_market(self):
        self.assertEqual(set(self.by_type), {"moneyline", "spread", "total"})
        self.assertEqual(len(self.rows), 3)

    def test_common_event_fields(self):
        for r in self.rows:
            self.assertEqual(r["sport_key"], "baseball_mlb")
            self.assertEqual(r["event_id"], "e1")
            self.assertEqual(r["home_team"], "Rockies")
            self.assertEqual(r["away_team"], "Astros")
            self.assertEqual(r["game_date"], "2020-05-01")
            self.assertFalse(r["resolved"])
            self.assertIsNone(r["outcome"])

    def test_moneyline_picks_higher_blended_prob(self):
        ml = self.by_type["moneyline"]
        self.assertEqual(ml["team"], "Astros")
        self.assertEqual(ml["side"], "away")
        self.assertAlmostEqual(ml["model_prob"], 0.62)
        self.assertAlmostEqual(ml["raw_prob"], 0.60)
        self.assertEqual(ml["price"], -140)
        self.assertIsNone(ml["point"])
        self.assertIs(ml["is_value"], True)

    def test_spread_picks_higher_cover_rate(self):
        sp = self.by_type["spread"]
        self.assertEqual(sp["team"], "Astros")
        self.assertEqual(sp["side"], "away")
        self.assertAlmostEqual(sp["model_prob"], 0.55)
        self.assertAlmostEqual(sp["raw_prob"], 0.53)
        self.assertEqual(sp["point"], -1.5)
        self.assertEqual(sp["price"], -110)

    def test_total_takes_over_under_lean(self):
        tot = self.by_type["total"]
        self.assertEqual(tot["side"], "over")
        self.assertAlmostEqual(tot["model_prob"], 0.58)
        self.assertAlmostEqual(tot["raw_prob"], 0.56)
        self.assertEqual(tot["point"], 9.5)
        self.assertEqual(tot["price"], -105)
        self.assertIs(tot["is_value"], True)
        # Totals carry no team keys but still resolve via the event's home/away.
        self.assertIsNone(tot["team"])
        self.assertEqual(tot["home_team"], "Rockies")

    def test_total_under_lean_when_over_below_50(self):
        ar = _ar()
        ar["all_totals"][0].update({
            "over_hit_rate": 42.0, "model_over_hit_rate": 44.0,
            "is_over_value": False, "is_under_value": True})
        tot = {r["bet_type"]: r for r in
               recalibration.build_market_prediction_rows(ar, "baseball_mlb")}["total"]
        self.assertEqual(tot["side"], "under")
        self.assertAlmostEqual(tot["model_prob"], 0.58)   # 1 - 0.42
        self.assertAlmostEqual(tot["raw_prob"], 0.56)     # 1 - 0.44
        self.assertEqual(tot["price"], -115)              # under_price
        self.assertIs(tot["is_value"], True)

    def test_malformed_input_returns_empty(self):
        self.assertEqual(recalibration.build_market_prediction_rows(None, "x"), [])
        self.assertEqual(recalibration.build_market_prediction_rows({}, ""), [])


class UpsertTests(unittest.TestCase):
    def test_relogging_supersedes_unresolved(self):
        rows = []
        with patch.object(recalibration, "mutate_market_prediction_log",
                          side_effect=_InMemoryLog(rows)):
            first = recalibration.build_market_prediction_rows(_ar(), "baseball_mlb")
            recalibration.log_market_prediction_rows(first)
            self.assertEqual(len(rows), 3)
            second = recalibration.build_market_prediction_rows(_ar(), "baseball_mlb")
            for r in second:                       # simulate a line move
                r["model_prob"] = 0.70
            recalibration.log_market_prediction_rows(second)
        self.assertEqual(len(rows), 3)             # still one row per market
        self.assertTrue(all(r["model_prob"] == 0.70 for r in rows))

    def test_resolved_row_not_overwritten(self):
        graded = recalibration.build_market_prediction_rows(_ar(), "baseball_mlb")
        for r in graded:
            r.update({"resolved": True, "outcome": 1, "actual": "5-3"})
        rows = list(graded)
        with patch.object(recalibration, "mutate_market_prediction_log",
                          side_effect=_InMemoryLog(rows)):
            relog = recalibration.build_market_prediction_rows(_ar(), "baseball_mlb")
            for r in relog:
                r["model_prob"] = 0.99
            changed = recalibration.log_market_prediction_rows(relog)
        self.assertEqual(changed, 0)               # nothing changed
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["resolved"] and r["outcome"] == 1 for r in rows))

    def test_distinct_markets_all_appended(self):
        rows = []
        with patch.object(recalibration, "mutate_market_prediction_log",
                          side_effect=_InMemoryLog(rows)):
            added = recalibration.log_market_prediction_rows(
                recalibration.build_market_prediction_rows(_ar(), "baseball_mlb"))
        self.assertEqual(added, 3)
        self.assertEqual({r["bet_type"] for r in rows},
                         {"moneyline", "spread", "total"})

    def test_logger_is_best_effort_on_error(self):
        def boom(mutator, max_retries=5, where=None):
            raise RuntimeError("backend down")
        with patch.object(recalibration, "mutate_market_prediction_log",
                          side_effect=boom):
            self.assertEqual(
                recalibration.log_market_prediction_rows(
                    recalibration.build_market_prediction_rows(_ar(), "baseball_mlb")),
                0)


def _row(bet_type="moneyline", side="away", ts="t1", **extra):
    row = {
        "ts": ts, "sport_key": "baseball_mlb", "event_id": "e1",
        "bet_type": bet_type, "team": "Astros", "opponent": "Rockies",
        "home_team": "Rockies", "away_team": "Astros", "home_away": "AWAY",
        "side": side, "point": None, "model_prob": 0.6, "raw_prob": 0.58,
        "price": -140, "matchup": "Astros @ Rockies",
        "game_date": "2020-05-01", "commence_time": "2020-05-01T23:00:00Z",
        "resolved": False, "actual": None, "outcome": None, "resolved_at": None,
    }
    row.update(extra)
    return row


class ResolverTests(unittest.TestCase):
    def _run(self, rows):
        import game_results
        with patch.object(recalibration, "_read_market_log",
                          return_value=[dict(r) for r in rows]), \
             patch.object(recalibration, "mutate_market_prediction_log",
                          side_effect=_InMemoryLog(rows)), \
             patch.object(game_results, "final_score", return_value=(5.0, 3.0)), \
             patch.object(game_results, "side_for_team", return_value="away"), \
             patch.object(game_results, "grade_team_bet",
                          side_effect=self._grade):
            return recalibration.resolve_pending_market_outcomes("baseball_mlb")

    @staticmethod
    def _grade(bet_type, side, point, home_score, away_score):
        # Away team lost the game 5-3 for moneyline; spread pushes; total wins.
        return {"moneyline": "lost", "spread": "push", "total": "won"}[bet_type]

    def test_maps_won_lost_push_to_outcomes(self):
        rows = [_row("moneyline", "away", ts="t1"),
                _row("spread", "away", ts="t2", point=-1.5),
                _row("total", "over", ts="t3", point=9.5)]
        n = self._run(rows)
        self.assertEqual(n, 3)
        by_type = {r["bet_type"]: r for r in rows}
        self.assertEqual(by_type["moneyline"]["outcome"], 0)   # lost
        self.assertIsNone(by_type["spread"]["outcome"])        # push
        self.assertEqual(by_type["total"]["outcome"], 1)       # won
        for r in rows:
            self.assertTrue(r["resolved"])
            self.assertEqual(r["actual"], "5-3")

    def test_stays_pending_when_no_final_score(self):
        import game_results
        rows = [_row("moneyline", "away")]
        with patch.object(recalibration, "_read_market_log",
                          return_value=[dict(r) for r in rows]), \
             patch.object(recalibration, "mutate_market_prediction_log",
                          side_effect=_InMemoryLog(rows)), \
             patch.object(game_results, "final_score", return_value=None):
            n = recalibration.resolve_pending_market_outcomes("baseball_mlb")
        self.assertEqual(n, 0)
        self.assertFalse(rows[0]["resolved"])

    def test_skips_future_games(self):
        import game_results
        rows = [_row("moneyline", "away", game_date="2999-01-01")]
        called = {"n": 0}

        def _fs(*a, **k):
            called["n"] += 1
            return (5.0, 3.0)

        with patch.object(recalibration, "_read_market_log",
                          return_value=[dict(r) for r in rows]), \
             patch.object(recalibration, "mutate_market_prediction_log",
                          side_effect=_InMemoryLog(rows)), \
             patch.object(game_results, "final_score", side_effect=_fs):
            n = recalibration.resolve_pending_market_outcomes("baseball_mlb")
        self.assertEqual(n, 0)
        self.assertEqual(called["n"], 0)       # never fetched a future game


class SummaryTests(unittest.TestCase):
    def test_hit_rate_brier_roi(self):
        rows = [
            # moneyline win at +100 (model_prob 0.60)
            _row("moneyline", ts="a", resolved=True, outcome=1,
                 model_prob=0.60, price=100),
            # moneyline loss at -110 (model_prob 0.40)
            _row("moneyline", ts="b", event_id="e2", resolved=True, outcome=0,
                 model_prob=0.40, price=-110),
            # total push at -110 (returns stake)
            _row("total", ts="c", side="over", point=9.5, resolved=True,
                 outcome=None, model_prob=0.50, price=-110),
            # pending (not counted)
            _row("spread", ts="d", event_id="e3", resolved=False,
                 model_prob=0.55, price=-110),
        ]
        summary = recalibration.summarize_market_prediction_rows(rows)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["resolved"], 3)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["pushes"], 1)
        self.assertEqual(summary["graded"], 2)          # only 0/1 outcomes
        self.assertAlmostEqual(summary["hit_rate"], 0.5)  # 1 of 2 decided
        # Brier over the two decided: (0.60-1)^2 + (0.40-0)^2 = 0.16 + 0.16, /2
        self.assertAlmostEqual(summary["brier"], 0.16)
        # ROI over win (+1.0), loss (-1.0), push (0.0): mean = 0.0
        self.assertAlmostEqual(summary["roi"], 0.0)
        self.assertEqual(summary["priced_resolved"], 3)

    def test_by_market_split_and_sport_filter(self):
        rows = [_row("moneyline", ts="a", resolved=True, outcome=1),
                _row("total", ts="b", side="over", point=9.5, resolved=True,
                     outcome=1, model_prob=0.6)]
        summary = recalibration.summarize_market_prediction_rows(rows)
        markets = {m["bet_type"] for m in summary["by_market"]}
        self.assertEqual(markets, {"moneyline", "total"})
        # Sport filter excludes non-matching rows.
        empty = recalibration.summarize_market_prediction_rows(
            rows, sport_key="basketball_nba")
        self.assertEqual(empty["total"], 0)


if __name__ == "__main__":
    unittest.main()
