"""NBA prop overconfidence response (2026-10-16): honest displayed P + never flagged.
  * recalibration.apply_recal_cfg: one entry, Platt OR a base-method ISOTONIC seed;
  * refit_calibration._stage_base_recal stages either winner into the committed seed
    (keeping the other props + their provenance);
  * the runtime loop tolerates an isotonic seed (no Platt incumbent; the seed is not
    blended with a Platt loop fit at load);
  * serving: props applies an isotonic seed as the final calibration slot, and a prop in
    the value_gate ``suppress`` list is never is_value (still displayed);
  * the shipped NBA config: all 10 props suppressed + a validated seed entry for each.
Synthetic data; no network/DB (the last class reads the committed calibration files).
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import recalibration as rc

NBA_PROPS = ["player_points", "player_rebounds", "player_assists", "player_threes",
             "player_steals", "player_blocks", "player_turnovers", "player_points_rebounds",
             "player_points_assists", "player_points_rebounds_assists"]


class ApplyRecalCfgTests(unittest.TestCase):
    def test_platt_matches_apply_platt(self):
        cfg = {"a": 0.4, "b": -0.1}
        self.assertAlmostEqual(rc.apply_recal_cfg(0.7, cfg), rc.apply_platt(0.7, 0.4, -0.1))

    def test_isotonic_matches_apply_isotonic(self):
        cfg = {"kind": "isotonic", "kx": [0.2, 0.5, 0.9], "ky": [0.45, 0.5, 0.55]}
        self.assertAlmostEqual(rc.apply_recal_cfg(0.7, cfg),
                               rc.apply_isotonic(0.7, cfg["kx"], cfg["ky"]))
        self.assertAlmostEqual(rc.apply_recal_cfg(0.99, cfg), 0.55)   # flat beyond knots

    def test_identity_when_no_usable_map(self):
        for cfg in (None, {}, {"b": 0.2}, {"kind": "isotonic", "kx": [], "ky": []}):
            self.assertEqual(rc.apply_recal_cfg(0.7, cfg), 0.7)
        self.assertIsNone(rc.apply_recal_cfg(None, {"a": 1.0, "b": 0.0}))

    def test_clamped(self):
        cfg = {"kind": "isotonic", "kx": [0.0, 1.0], "ky": [-0.2, 1.3]}
        self.assertEqual(rc.apply_recal_cfg(0.0, cfg), 0.0)
        self.assertEqual(rc.apply_recal_cfg(1.0, cfg), 1.0)


class StageBaseRecalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._p = patch.object(rc, "CALIB_DIR", self.tmp)
        self._p.start()
        rc._LOAD_CACHE.pop("basketball_nba", None)

    def tearDown(self):
        self._p.stop()
        rc._LOAD_CACHE.pop("basketball_nba", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_isotonic_and_platt_both_staged_with_provenance(self):
        import refit_calibration as R
        R._stage_base_recal("basketball_nba", "player_blocks", {"A": {
            "method": "A", "winner": "isotonic",
            "iso": {"kx": [0.1, 0.5, 0.9], "ky": [0.40, 0.45, 0.50], "n": 4000}}})
        R._stage_base_recal("basketball_nba", "player_points", {"B": {
            "method": "B", "winner": "platt", "platt": {"a": 0.05, "b": -0.02, "n": 9000}}})
        R._stage_base_recal("basketball_nba", "player_steals", {"B": {
            "method": "B", "winner": "raw"}})                        # raw -> nothing
        with open(rc.recalibration_path("basketball_nba"), encoding="utf-8") as f:
            blob = json.load(f)
        props = blob["props"]
        self.assertEqual(sorted(props), ["player_blocks", "player_points"])
        blk = props["player_blocks"]
        self.assertEqual((blk["kind"], blk["kx"], blk["ky"], blk["n_fit"], blk["validated"]),
                         ("isotonic", [0.1, 0.5, 0.9], [0.4, 0.45, 0.5], 4000, True))
        self.assertEqual((props["player_points"]["a"], props["player_points"]["b"]),
                         (0.05, -0.02))
        # the second save kept the first prop's provenance note
        self.assertEqual(sorted(blob["meta"]), ["player_blocks_recal", "player_points_recal"])
        # and the loader serves both
        loaded = rc.load_recalibration("basketball_nba")
        self.assertEqual(sorted(loaded), ["player_blocks", "player_points"])


class LoopToleratesIsotonicSeedTests(unittest.TestCase):
    ISO = {"kind": "isotonic", "kx": [0.0, 1.0], "ky": [0.48, 0.52], "n_fit": 5000,
           "validated": True}

    def tearDown(self):
        rc._LOAD_CACHE.pop("basketball_nba", None)

    def test_sql_merge_keeps_isotonic_seed_and_blends_platt_seed(self):
        seed = {"player_blocks": dict(self.ISO),
                "player_points": {"a": 0.1, "b": 0.0, "n_fit": 1000, "validated": True}}
        loop = {"props": {
            "player_blocks": {"a": 0.9, "b": 0.0, "n_fit": 200, "validated": True},
            "player_points": {"a": 0.9, "b": 0.0, "n_fit": 200, "validated": True}}}
        rc._LOAD_CACHE.pop("basketball_nba", None)
        with patch.object(rc, "_sql", return_value=True), \
             patch.object(rc._db, "load_recal", return_value=loop), \
             patch.object(rc, "_read_local_recal", return_value=(None, seed)):
            _, props = rc._load_recal_cached("basketball_nba")
        self.assertEqual(props["player_blocks"], self.ISO)          # seed stands
        self.assertIn("blend_weight", props["player_points"])       # Platt seed still blends

    def test_refit_sport_has_no_platt_incumbent_for_isotonic_seed(self):
        rows = [{"sport_key": "basketball_nba", "resolved": True, "outcome": 1,
                 "prop_key": "player_blocks", "raw_prob": 0.6, "line": 1.5,
                 "player": "A", "game_date": "2025-01-01"}]
        seen = {}

        def _fit(records, incumbent=None):
            seen["incumbent"] = incumbent
            return None

        with patch.object(rc, "_read_log", return_value=rows), \
             patch.object(rc, "_read_local_recal",
                          return_value=(None, {"player_blocks": dict(self.ISO)})), \
             patch("calibration_loader.load_calibration", return_value={}), \
             patch.object(rc, "fit_platt_chronological", side_effect=_fit), \
             patch.object(rc, "save_recalibration") as save:
            self.assertEqual(rc.refit_sport("basketball_nba", resolve_first=False), {})
        self.assertIsNone(seen["incumbent"])
        save.assert_not_called()


class ServingTests(unittest.TestCase):
    def _run(self, recal=None, gate=None):
        import props
        prop_data = {"commence_time": "2025-01-15T00:30:00Z", "home_team": "Home",
                     "away_team": "Away", "game_id": "evt1",
                     "props": {"player_points": {"Star Guard": {
                         "line": 20.5, "over_implied": 0.5, "under_implied": 0.5,
                         "over_price": -110, "under_price": -110,
                         "over_book": "draftkings", "under_book": "draftkings",
                         "dk_over_price": -110, "dk_under_price": -110}}}}
        dates = [f"2025-01-{d:02d}" for d in range(1, 15)]
        hist = {"Star Guard": {"player_points": {
            "found": True, "values": [30.0] * 10 + [10.0] * 4,
            "game_dates": list(reversed(dates)), "minutes": [34.0] * 14,
            "team_id": 1, "team_ids": [1] * 14}}}
        cfg = {"method": "A", "half_life": None, "shrinkage_k": 0}
        with patch.object(props, "load_calibration", return_value={"player_points": cfg}), \
             patch.object(props, "load_recalibration", return_value=recal or {}), \
             patch.object(props, "load_value_gate", return_value=gate or {}), \
             patch.object(props, "maybe_auto_refit"), \
             patch.object(props, "log_prediction_rows"), \
             patch.object(props, "log_prediction"):
            return props.analyze_player_props_value(prop_data, hist, threshold_pct=1.0,
                                                    sport_key="basketball_nba")[0]

    def test_control_flags_the_big_raw_edge(self):
        c = self._run()
        self.assertEqual(c["over_rate"], 100.0)
        self.assertTrue(c["is_value"])
        self.assertIsNone(c["recalibration"])

    def test_isotonic_seed_is_the_final_calibration_slot(self):
        c = self._run(recal={"player_points": {"kind": "isotonic", "kx": [0.0, 1.0],
                                               "ky": [0.52, 0.52], "validated": True}})
        self.assertEqual(c["over_rate"], 52.0)                    # honest, flattened P
        self.assertEqual(c["recalibration"]["kind"], "isotonic")
        self.assertEqual(c["recalibration"]["raw_prob"], 100.0)

    def test_suppressed_prop_is_never_value_but_still_shown(self):
        c = self._run(gate={"suppress": NBA_PROPS})
        self.assertFalse(c["is_value"])
        self.assertEqual(c["over_rate"], 100.0)                   # still displayed


class ShippedNbaConfigTests(unittest.TestCase):
    """The committed NBA config: every calibrated prop suppressed + a validated recal map."""

    def test_all_ten_props_suppressed(self):
        from calibration_loader import load_calibration, load_value_gate
        gate = load_value_gate("basketball_nba")
        self.assertEqual(sorted(gate.get("suppress") or []), sorted(NBA_PROPS))
        self.assertIsNone(gate.get("ev_floor"))   # suppress-only: no ROI gate re-tune
        self.assertEqual(sorted(load_calibration("basketball_nba")), sorted(NBA_PROPS))

    def test_every_prop_has_a_validated_seed_map(self):
        path = os.path.join(os.path.dirname(os.path.abspath(rc.__file__)), "calibration",
                            "recalibration_basketball_nba.json")
        with open(path, encoding="utf-8") as f:
            _, props = rc._parse_recal_blob(json.load(f))
        self.assertEqual(sorted(props), sorted(NBA_PROPS))
        for k, cfg in props.items():
            ok = (cfg.get("kind") == "isotonic" and cfg.get("kx") and cfg.get("ky")) or (
                cfg.get("a") is not None and cfg.get("b") is not None)
            self.assertTrue(ok, k)
            # every map pulls the overconfident raw P toward a coin flip
            self.assertLess(abs(rc.apply_recal_cfg(0.85, cfg) - 0.5), 0.35 - 1e-9, k)


if __name__ == "__main__":
    unittest.main()
