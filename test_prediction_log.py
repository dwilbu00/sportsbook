"""Tests for prediction-log durability fixes (P2 batch 2):

- log_prediction_rows upserts by forecast identity so repeated same-slate
  logging cannot grow the log without bound, while preserving graded outcomes.
- compact_prediction_log collapses historical duplicates.
- seed_from_book_line_cache counts only current-season prior games for warmup
  blending, matching the runtime pipeline.
"""

import unittest
from unittest.mock import patch

import recalibration


def _row(player="Player One", ts="2024-04-01T10:00:00Z", raw=0.6,
         resolved=False, **extra):
    row = {
        "ts": ts,
        "sport_key": "basketball_nba",
        "event_id": "evt-1",
        "prop_key": "player_points",
        "player": player,
        "game_date": "2024-04-01",
        "line": 20.5,
        "raw_prob": raw,
        "direction": "OVER",
        "resolved": resolved,
        "outcome": None,
        "actual": None,
    }
    row.update(extra)
    return row


class _InMemoryLog:
    """Patch mutate_prediction_log to operate on an in-memory row list."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, mutator, max_retries=5):
        return mutator(self.rows)


class LogPredictionFieldTests(unittest.TestCase):
    """log_prediction persists the pick-rules ROI lens inputs (team,
    batting_order) without changing the forecast identity."""

    def _build(self, **extra):
        return recalibration.log_prediction(
            sport_key="baseball_mlb", prop_key="batter_hits", player="Slugger",
            game_date="2026-07-20", line=0.5, raw_prob=0.6, direction="OVER",
            price=-110, event_id="e1", write=False, **extra)

    def test_team_and_batting_order_persisted(self):
        row = self._build(team="Yankees", batting_order=3)
        self.assertEqual(row["team"], "Yankees")
        self.assertEqual(row["batting_order"], 3)

    def test_missing_fields_default_to_none(self):
        row = self._build()
        self.assertIsNone(row["team"])
        self.assertIsNone(row["batting_order"])

    def test_non_numeric_batting_order_nulls_field_not_row(self):
        # A malformed batting_order must NOT drop the forecast (calibration needs
        # it); the auxiliary field just nulls out.
        row = self._build(batting_order="not-a-number")
        self.assertIsNotNone(row)
        self.assertIsNone(row["batting_order"])

    def test_identity_unaffected_by_new_fields(self):
        row = self._build(team="Yankees", batting_order=3)
        self.assertEqual(
            recalibration.prediction_identity(row),
            ("baseball_mlb", "e1", "batter_hits", "Slugger", 0.5))


class UpsertDeduplicationTests(unittest.TestCase):
    def test_relogging_supersedes_stale_unresolved_duplicate(self):
        rows = []
        with patch.object(recalibration, "mutate_prediction_log",
                          side_effect=_InMemoryLog(rows)):
            recalibration.log_prediction_rows([_row(ts="t1", raw=0.60)])
            recalibration.log_prediction_rows([_row(ts="t2", raw=0.65)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ts"], "t2")
        self.assertEqual(rows[0]["raw_prob"], 0.65)

    def test_resolved_forecast_is_not_overwritten_by_a_relog(self):
        rows = [_row(ts="t1", raw=0.60, resolved=True, outcome=1, actual=25.0)]
        with patch.object(recalibration, "mutate_prediction_log",
                          side_effect=_InMemoryLog(rows)):
            changed = recalibration.log_prediction_rows(
                [_row(ts="t2", raw=0.90)])
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["resolved"])
        self.assertEqual(rows[0]["outcome"], 1)
        self.assertEqual(rows[0]["raw_prob"], 0.60)  # graded forecast preserved
        self.assertEqual(changed, 0)  # a no-op re-log triggers no write

    def test_distinct_forecasts_are_all_appended(self):
        rows = []
        with patch.object(recalibration, "mutate_prediction_log",
                          side_effect=_InMemoryLog(rows)):
            added = recalibration.log_prediction_rows(
                [_row(player="A"), _row(player="B")])
        self.assertEqual(added, 2)
        self.assertEqual({r["player"] for r in rows}, {"A", "B"})


class CompactionTests(unittest.TestCase):
    def test_compaction_merges_outcome_and_preserves_order(self):
        rows = [
            _row(player="A", ts="t1", raw=0.60),
            _row(player="A", ts="t2", raw=0.65,
                 resolved=True, outcome=1, actual=25.0),
            _row(player="B", ts="t3", raw=0.40),
        ]
        with patch.object(recalibration, "mutate_prediction_log",
                          side_effect=_InMemoryLog(rows)):
            removed = recalibration.compact_prediction_log()
        self.assertEqual(removed, 1)
        self.assertEqual([r["player"] for r in rows], ["A", "B"])
        merged = rows[0]
        self.assertTrue(merged["resolved"])
        self.assertEqual(merged["outcome"], 1)
        self.assertEqual(merged["raw_prob"], 0.65)         # resolved base

    def test_compaction_is_a_no_op_on_a_clean_log(self):
        rows = [_row(player="A"), _row(player="B")]
        with patch.object(recalibration, "mutate_prediction_log",
                          side_effect=_InMemoryLog(rows)):
            removed = recalibration.compact_prediction_log()
        self.assertEqual(removed, 0)
        self.assertEqual(len(rows), 2)


class SeedWarmupSeasonCountTests(unittest.TestCase):
    def test_seed_counts_only_current_season_prior_games(self):
        import book_line_calibration as blc
        import calibration_loader as cl

        obs = {
            "prop_key": "player_points",
            "line": 20.5,
            "actual": 25.0,
            "game_date": "2024-12-15",
            "prior_games": [
                {"game_date": "2024-11-01"},   # current 2024-25 NBA season
                {"game_date": "2024-12-01"},   # current 2024-25 NBA season
                {"game_date": "2023-03-01"},   # prior season
                {"game_date": "2023-04-01"},   # prior season
                {"game_date": "2024-01-15"},   # prior season (before Oct 2024)
            ],
        }
        captured = {}

        def fake_warmup(prop_cfg, projection, line, current_season_games,
                        empirical_over=None):
            captured["curr"] = current_season_games
            return None

        with patch.object(blc, "harvest_book_lines_from_store", return_value=[]), \
             patch.object(blc, "harvest_book_lines_from_prediction_log",
                          return_value=[]), \
             patch.object(blc, "harvest_book_lines", return_value=[1]), \
             patch.object(blc, "join_book_lines_to_actuals", return_value=[obs]), \
             patch.object(blc, "project_and_empirical", return_value=(22.0, 0.6)), \
             patch.object(blc, "_team_defense_lookup",
                          return_value=(None, None, None)), \
             patch.object(cl, "load_calibration",
                          return_value={"player_points": {"method": "B"}}), \
             patch.object(cl, "apply_calibration_with_warmup",
                          side_effect=fake_warmup):
            recalibration.seed_from_book_line_cache(
                "nba", "basketball", "nba", "basketball_nba",
                ["player_points"])

        # Two of the five prior games fall inside the current season; the naive
        # len(prior_games) count would have passed 5.
        self.assertEqual(captured["curr"], 2)


if __name__ == "__main__":
    unittest.main()
