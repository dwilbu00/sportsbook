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


if __name__ == "__main__":
    unittest.main(verbosity=2)
