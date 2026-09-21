"""Prop-serving probability-pipeline regressions (audit F17-F19).

Exercises analyze_player_props_value with calibration/gate/recal/logging patched to
no-ops, so these pin the SERVING contract (source routing, view parity, push
handling) without live data.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_props_serving -v
"""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import props


class _PropsServingTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        for name, value in (("load_calibration", {}), ("load_value_gate", {}),
                            ("load_recalibration", {}), ("maybe_auto_refit", None),
                            ("log_prediction_rows", 0)):
            self.stack.enter_context(patch.object(props, name, return_value=value))

    def tearDown(self):
        self.stack.close()

    def board(self, prop="player_receptions", line=4.5):
        return dict(game_id="e", home_team="A", away_team="B", props={prop: {
            "Fixture": dict(line=line, over_implied=.5, under_implied=.5,
                            over_price=-110, under_price=-110,
                            dk_over_price=-110, dk_under_price=-110,
                            dk_over_book="DraftKings", dk_under_book="DraftKings")}})


class F17NflModelHistoryGateTests(_PropsServingTest):
    def test_valid_model_without_espn_history_is_not_discarded(self):
        # F17: a frozen NFL model with n_prior=20 must serve even when ESPN has no
        # history for the player — the legacy short-history filter must not drop it.
        model = dict(proj=6., sd=2., p_over=.7, p_at=lambda line: .7, n_prior=20)
        with patch.object(props, "_nfl_model_override", return_value=model):
            r = props.analyze_player_props_value(
                self.board(), {}, sport_key="americanfootball_nfl")[0]
        self.assertIsNotNone(r.get("avg_stat"), r.get("skip_reason"))
        self.assertEqual(r["avg_stat"], 6.0)
        self.assertEqual(r["games_sampled"], 20)   # the model cohort, not the synthetic 1


class F18ProbabilityParityTests(_PropsServingTest):
    def test_nfl_safe_probability_matches_standard_final_calibration(self):
        # F18: Safe Mode returned the RAW model prob (0.80) while Standard showed the
        # recalibrated 66.67% at the same line. Both must now report the same final
        # calibrated probability at a given line.
        model = dict(proj=5.5, sd=1., p_over=.8,
                     p_at=lambda line: .8 if line <= 4.5 else .4, n_prior=20)
        hist = {"Fixture": {"player_receptions": dict(found=True, values=[8.] * 20)}}
        with patch.object(props, "_nfl_model_override", return_value=model), \
             patch.object(props, "load_recalibration",
                          return_value={"player_receptions": {"a": .5, "b": 0.}}):
            standard = props.analyze_player_props_value(
                self.board(), hist, sport_key="americanfootball_nfl")[0]
            safe = props.analyze_player_props_value(
                self.board(), hist, sport_key="americanfootball_nfl",
                safe_mode=True, safe_target=.75)[0]
        self.assertEqual(safe["safe_threshold"], 5)
        self.assertAlmostEqual(safe["model_hit_at_safe"], standard["over_rate"], places=2)


class F19IntegerLinePushTests(_PropsServingTest):
    def test_all_push_integer_line_is_not_an_under_win(self):
        # F19: 20 outcomes all exactly == an integer line of 5 -> every result PUSHES,
        # so UNDER must not be recommended as value (was 1 - over_rate = 1.0).
        hist = {"Fixture": {"player_points": dict(found=True, values=[5.] * 20)}}
        with patch.object(props, "load_calibration", return_value={"player_points": {
                "method": "A", "shrinkage_k": 0, "half_life": None}}):
            r = props.analyze_player_props_value(
                self.board("player_points", 5.), hist, sport_key="basketball_nba")[0]
        self.assertFalse(r["is_value"],
                         "all-push integer line should not recommend UNDER as value")

    def test_half_point_line_unaffected(self):
        # Half-point lines can't push: a clear over must still evaluate normally.
        hist = {"Fixture": {"player_points": dict(found=True, values=[6.] * 20)}}
        with patch.object(props, "load_calibration", return_value={"player_points": {
                "method": "A", "shrinkage_k": 0, "half_life": None}}):
            r = props.analyze_player_props_value(
                self.board("player_points", 4.5), hist, sport_key="basketball_nba")[0]
        self.assertEqual(r["over_rate"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
