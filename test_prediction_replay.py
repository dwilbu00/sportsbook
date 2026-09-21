"""F28 P2 — prediction replay harness parity tests.

Proves: a stored NFL prediction row replays to its recorded RAW prob from provenance
alone (event season/week, no wall-clock); replay REFUSES across a serving-version
change (artifacts_changed); replay abstains when it can't reproduce (no week, other
sport). Deterministic — nfl_prop_serving/nfl_props_scan are injected as fakes so the
test needs neither the nflverse stack nor the frozen artifacts.
"""
import sys
import types
import unittest
from unittest.mock import patch

import prediction_replay

# The provenance the row was stamped with; the parity gate passes iff the CURRENT
# build reports the same calibration_hash + model_schema.
_CAL = "cal-abc123"
_SCHEMA = "fs7:art0011223344"


def _row(**over):
    r = {
        "sport_key": "americanfootball_nfl",
        "commence_time": "2024-09-15T17:00:00+00:00",
        "player": "A. Receiver",
        "prop_key": "player_reception_yds",
        "line": 60.5,
        "raw_prob": 0.5,
        "final_prob": 0.52,
        "method": "nfl_model",
        "event_season": 2024,
        "event_week": 2,
        "calibration_hash": _CAL,
        "model_schema": _SCHEMA,
        "reference_book_policy": "pinnacle_devig",
        "context_fingerprint": "fp-original",
    }
    r.update(over)
    return r


def _ctx(cal=_CAL, schema=_SCHEMA, fp="fp-current"):
    return {"calibration_hash": cal, "model_schema": schema,
            "context_fingerprint": fp}


def _inject_nfl(p_over):
    """Fakes for the serving path; project() returns a fixed p_over (or None)."""
    serving = types.ModuleType("nfl_prop_serving")
    serving.project = lambda pn, s, w, prop, line=None: (
        None if p_over is None else {"p_over": p_over})
    scan = types.ModuleType("nfl_props_scan")
    scan._norm = lambda n: (n or "").strip().lower()
    return {"nfl_prop_serving": serving, "nfl_props_scan": scan}


class ReplayHarnessTests(unittest.TestCase):

    def test_no_provenance_not_replayable(self):
        self.assertEqual(prediction_replay.replay(_row(context_fingerprint=None))
                         ["status"], "no_provenance")
        legacy = _row()
        del legacy["context_fingerprint"]            # pre-F28 row: key absent entirely
        self.assertEqual(prediction_replay.replay(legacy)["status"], "no_provenance")

    @patch("prediction_provenance.build_context")
    def test_calibration_change_refuses(self, bc):
        bc.return_value = _ctx(cal="cal-DIFFERENT")
        out = prediction_replay.replay(_row())
        self.assertEqual(out["status"], "artifacts_changed")
        self.assertEqual(out["stored_fingerprint"], "fp-original")

    @patch("prediction_provenance.build_context")
    def test_model_schema_change_refuses(self, bc):
        bc.return_value = _ctx(schema="fs8:artNEWNEWNEW")   # artifact/schema drift
        self.assertEqual(prediction_replay.replay(_row())["status"],
                         "artifacts_changed")

    @patch("prediction_provenance.build_context")
    def test_nfl_replay_reproduces_raw_prob(self, bc):
        bc.return_value = _ctx()                            # artifacts unchanged
        with patch.dict(sys.modules, _inject_nfl(0.6321)):
            out = prediction_replay.replay(_row(raw_prob=0.6321))
        self.assertEqual(out["status"], "ok")
        self.assertTrue(out["matches"])
        self.assertAlmostEqual(out["recomputed"], 0.6321)
        self.assertEqual(out["stored"], 0.6321)

    @patch("prediction_provenance.build_context")
    def test_nfl_replay_flags_divergence(self, bc):
        # Same artifacts, but recompute != stored raw_prob → matches False (drift, not
        # a version change): the harness surfaces it rather than silently passing.
        bc.return_value = _ctx()
        with patch.dict(sys.modules, _inject_nfl(0.70)):
            out = prediction_replay.replay(_row(raw_prob=0.50))
        self.assertEqual(out["status"], "ok")
        self.assertFalse(out["matches"])

    @patch("prediction_provenance.build_context")
    def test_nfl_missing_week_abstains(self, bc):
        # No event_week recorded → cannot reproduce the cohort deterministically.
        bc.return_value = _ctx()
        out = prediction_replay.replay(_row(event_week=None))
        self.assertEqual(out["status"], "unsupported")

    @patch("prediction_provenance.build_context")
    def test_nfl_serving_none_abstains(self, bc):
        # project() returns nothing (player not in mirror) → abstain, never a false OK.
        bc.return_value = _ctx()
        with patch.dict(sys.modules, _inject_nfl(None)):
            out = prediction_replay.replay(_row())
        self.assertEqual(out["status"], "unsupported")

    @patch("prediction_provenance.build_context")
    def test_other_sport_unsupported(self, bc):
        bc.return_value = _ctx()
        out = prediction_replay.replay(
            _row(sport_key="baseball_mlb", method="model_c", event_week=None))
        self.assertEqual(out["status"], "unsupported")
        self.assertIn("baseball_mlb", out["reason"])

    @patch("prediction_provenance.build_context")
    def test_replay_many_tally(self, bc):
        # A mixed batch: one reproduces, one diverges, one has no provenance.
        bc.return_value = _ctx()
        rows = [_row(raw_prob=0.6321), _row(raw_prob=0.10),
                _row(context_fingerprint=None)]
        with patch.dict(sys.modules, _inject_nfl(0.6321)):
            s = prediction_replay.replay_many(rows)
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["ok"], 2)
        self.assertEqual(s["matched"], 1)
        self.assertEqual(s["mismatched"], 1)
        self.assertEqual(s["no_provenance"], 1)
        self.assertEqual(len(s["mismatches"]), 1)
        self.assertEqual(s["mismatches"][0]["stored"], 0.10)


if __name__ == "__main__":
    unittest.main()
