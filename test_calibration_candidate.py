"""
Candidate-file staging for calibration writes (calibration_loader).

A refit stages to calibration/<sport>.candidate.json and the live file the app
serves is never touched until promote_calibration() atomically swaps it in.
These tests pin the invariants that make that safe:
  * default (mode off) writes the live file, byte-for-byte as before;
  * candidate mode redirects every save_* helper to the candidate and leaves
    live untouched, seeded from live so other blocks/props survive;
  * successive saves in one cycle accumulate into a single candidate;
  * serving readers ALWAYS read live, even mid-staging;
  * promote archives the old live then swaps; discard throws the candidate away.

Hermetic: CALIBRATION_DIR/ARCHIVE_DIR are redirected to a temp dir per test; no
network, no SQL.
"""
import json
import os
import tempfile
import unittest

import calibration_loader as cl

SK = "baseball_mlb"


class AdditiveCandidateReadTests(unittest.TestCase):
    """Candidate-aware SERVING reads (set_serving_candidate) — so an OFFLINE backtest can
    grade a staged team-market calibration (additive / prob_shrink / market_blend /
    challenger) on a holdout WITHOUT promoting. The live app never enables it (always
    reads live), and it's SEPARATE from set_candidate_mode (writes) so a refit's serving
    reads stay live mid-staging."""

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._orig_dir = cl.CALIBRATION_DIR
        cl.CALIBRATION_DIR = self._dir
        cl.set_serving_candidate(False)

    def tearDown(self):
        cl.CALIBRATION_DIR = self._orig_dir
        cl.set_serving_candidate(False)

    def _write(self, path, park_weight):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sport_key": SK, "expected_runs_additive":
                       {"enabled": True, "run_env": {"park_weight": park_weight}}}, f)

    def _park(self):
        return (cl.load_expected_runs_additive(SK).get("run_env") or {}).get(
            "park_weight")

    def test_off_reads_live_even_with_candidate(self):
        self._write(cl.calibration_path(SK), 0.0)
        self._write(cl.candidate_path(SK), 1.0)
        self.assertEqual(self._park(), 0.0)               # serving = live (default)

    def test_write_staging_alone_does_not_change_serving(self):
        # set_candidate_mode (writes) must NOT flip serving to the candidate.
        self._write(cl.calibration_path(SK), 0.0)
        self._write(cl.candidate_path(SK), 1.0)
        cl.set_candidate_mode(True)
        self.assertEqual(self._park(), 0.0)               # still live mid-staging
        cl.set_candidate_mode(False)

    def test_serving_candidate_reads_candidate(self):
        self._write(cl.calibration_path(SK), 0.0)
        self._write(cl.candidate_path(SK), 1.0)
        cl.set_serving_candidate(True)
        self.assertEqual(self._park(), 1.0)               # backtest = candidate

    def test_serving_candidate_no_candidate_falls_back_to_live(self):
        self._write(cl.calibration_path(SK), 0.0)
        cl.set_serving_candidate(True)                     # no candidate file
        self.assertEqual(self._park(), 0.0)

    def test_toggle_does_not_cross_serve_from_cache(self):
        self._write(cl.calibration_path(SK), 0.0)
        self._write(cl.candidate_path(SK), 1.0)
        self.assertEqual(self._park(), 0.0)               # live (caches live block)
        cl.set_serving_candidate(True)
        self.assertEqual(self._park(), 1.0)               # cache cleared -> candidate
        cl.set_serving_candidate(False)
        self.assertEqual(self._park(), 0.0)               # back to live

    def _write_blob(self, path, blob):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(blob, sport_key=SK), f)

    def test_team_readers_serving_candidate_aware(self):
        # prob_shrink / market_blend / expected_runs_challenger all serve candidate-aware.
        self._write_blob(cl.calibration_path(SK), {
            "prob_shrink": {"totals": 0.6},
            "market_blend": {"totals": {"w": 1.0}},
            "expected_runs_challenger": {"enabled": False}})
        self._write_blob(cl.candidate_path(SK), {
            "prob_shrink": {"totals": 0.2},
            "market_blend": {"totals": {"w": 0.2}},
            "expected_runs_challenger": {"enabled": True}})
        cl.set_serving_candidate(False)
        self.assertEqual(cl.load_prob_shrink(SK)["totals"], 0.6)
        self.assertEqual(cl.load_market_blend(SK)["totals"]["w"], 1.0)
        self.assertFalse(cl.load_expected_runs_challenger(SK)["enabled"])
        cl.set_serving_candidate(True)
        self.assertEqual(cl.load_prob_shrink(SK)["totals"], 0.2)
        self.assertEqual(cl.load_market_blend(SK)["totals"]["w"], 0.2)
        self.assertTrue(cl.load_expected_runs_challenger(SK)["enabled"])

    def test_set_serving_candidate_clears_pricing_cache(self):
        import pricing_common as pc
        self._write_blob(cl.calibration_path(SK), {"prob_shrink": {"totals": 0.6}})
        self._write_blob(cl.candidate_path(SK), {"prob_shrink": {"totals": 0.2}})
        cl.set_serving_candidate(False)
        self.assertAlmostEqual(pc._shrink_factor(SK, "totals"), 0.6)   # caches live
        cl.set_serving_candidate(True)                                 # clears the cache
        self.assertAlmostEqual(pc._shrink_factor(SK, "totals"), 0.2)   # re-reads candidate

    # ── PROPS (load_calibration) candidate-awareness ──────────────────────────
    # Regression guard: load_calibration (the PROPS method-selection reader) must
    # honor serving-candidate too, else a staged --seasons props refit silently
    # grades on LIVE (the C->E flip that read identical-to-live and tripped the
    # holdout-validation sanity check). Mirrors the team-reader invariants above.
    def _write_props(self, path, method):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sport_key": SK,
                       "props": {"pitcher_earned_runs": {"method": method}}}, f)

    def _method(self):
        return (cl.load_calibration(SK).get("pitcher_earned_runs") or {}).get("method")

    def test_props_off_reads_live_even_with_candidate(self):
        self._write_props(cl.calibration_path(SK), "C")
        self._write_props(cl.candidate_path(SK), "E")
        self.assertEqual(self._method(), "C")             # serving = live (default)

    def test_props_write_staging_alone_does_not_change_serving(self):
        self._write_props(cl.calibration_path(SK), "C")
        self._write_props(cl.candidate_path(SK), "E")
        cl.set_candidate_mode(True)
        self.assertEqual(self._method(), "C")             # still live mid-staging
        cl.set_candidate_mode(False)

    def test_props_serving_candidate_reads_candidate(self):
        self._write_props(cl.calibration_path(SK), "C")
        self._write_props(cl.candidate_path(SK), "E")
        cl.set_serving_candidate(True)
        self.assertEqual(self._method(), "E")             # backtest = candidate

    def test_props_serving_candidate_no_candidate_falls_back_to_live(self):
        self._write_props(cl.calibration_path(SK), "C")
        cl.set_serving_candidate(True)                     # no candidate file
        self.assertEqual(self._method(), "C")


class CandidateStagingTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._orig_dir = cl.CALIBRATION_DIR
        self._orig_arch = cl.ARCHIVE_DIR
        cl.CALIBRATION_DIR = self._dir
        cl.ARCHIVE_DIR = os.path.join(self._dir, "archive")
        cl.set_candidate_mode(False)

    def tearDown(self):
        cl.CALIBRATION_DIR = self._orig_dir
        cl.ARCHIVE_DIR = self._orig_arch
        cl.set_candidate_mode(False)

    def _live(self):
        return cl._read_json(cl.calibration_path(SK))

    def _cand(self):
        return cl._read_json(cl.candidate_path(SK))

    def _seed_live(self):
        """A representative live file: props (incl. an E incumbent) + a block."""
        cl.set_candidate_mode(False)
        cl.save_calibration(SK, {"batter_hits": {"method": "C", "n_obs": 500},
                                 "batter_tb": {"method": "E", "n_obs": 300}})
        cl.save_prob_shrink(SK, {"totals": 0.5})

    # ── default (mode off) ─────────────────────────────────────────────
    def test_default_writes_live_not_candidate(self):
        cl.save_calibration(SK, {"batter_hits": {"method": "B", "n_obs": 10}})
        self.assertTrue(os.path.exists(cl.calibration_path(SK)))
        self.assertFalse(cl.has_candidate(SK))
        self.assertEqual(cl.active_write_label(SK), f"{SK}.json")

    # ── candidate mode ─────────────────────────────────────────────────
    def test_candidate_write_leaves_live_untouched(self):
        self._seed_live()
        cl.set_candidate_mode(True)
        # A bare re-sweep would reset the E incumbent to A.
        cl.save_calibration(SK, {"batter_hits": {"method": "C", "n_obs": 520},
                                 "batter_tb": {"method": "A", "n_obs": 320},
                                 "batter_rbi": {"method": "B", "n_obs": 150}})
        self.assertTrue(cl.has_candidate(SK))
        self.assertEqual(cl.active_write_label(SK), f"{SK}.candidate.json")
        # LIVE must be untouched — incumbent E preserved.
        self.assertEqual(self._live()["props"]["batter_tb"]["method"], "E")
        self.assertNotIn("batter_rbi", self._live()["props"])

    def test_candidate_seeds_from_live_preserving_blocks(self):
        self._seed_live()
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"batter_hits": {"method": "C", "n_obs": 520}})
        cand = self._cand()
        # Candidate started as a full copy of live, so the prob_shrink block and
        # the un-refit prop survive.
        self.assertEqual(cand.get("prob_shrink"), {"totals": 0.5})
        self.assertEqual(cand["props"]["batter_tb"]["method"], "E")

    def test_successive_saves_accumulate_in_one_candidate(self):
        self._seed_live()
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"batter_rbi": {"method": "B", "n_obs": 150}})
        cl.save_prob_shrink(SK, {"spreads": 0.7})
        cand = self._cand()
        self.assertEqual(cand["prob_shrink"], {"totals": 0.5, "spreads": 0.7})
        self.assertIn("batter_rbi", cand["props"])
        self.assertIn("batter_hits", cand["props"])
        # Live still untouched by either write.
        self.assertNotIn("spreads", self._live()["prob_shrink"])
        self.assertNotIn("batter_rbi", self._live()["props"])

    def test_serving_readers_always_read_live(self):
        self._seed_live()
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"batter_tb": {"method": "A", "n_obs": 320}})
        cl.save_prob_shrink(SK, {"totals": 0.99})
        # Even mid-staging, the app-facing readers return the LIVE values.
        self.assertEqual(load_method(SK, "batter_tb"), "E")
        self.assertEqual(cl.load_prob_shrink(SK), {"totals": 0.5})

    # ── diff ───────────────────────────────────────────────────────────
    def test_diff_reports_changes(self):
        self._seed_live()
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"batter_hits": {"method": "C", "n_obs": 520},
                                 "batter_tb": {"method": "A", "n_obs": 320},
                                 "batter_rbi": {"method": "B", "n_obs": 150}})
        d = cl.diff_calibration(SK)
        self.assertTrue(d["has_candidate"])
        self.assertEqual(d["props"]["added"], ["batter_rbi"])
        self.assertEqual(d["props"]["removed"], [])
        changed = {c["prop"]: (c["live_method"], c["candidate_method"])
                   for c in d["props"]["changed"]}
        self.assertEqual(changed["batter_tb"], ("E", "A"))
        # batter_hits: method unchanged (C->C) but n_obs moved → still reported.
        self.assertEqual(changed["batter_hits"], ("C", "C"))

    def test_diff_no_candidate(self):
        self._seed_live()
        self.assertEqual(cl.diff_calibration(SK), {"has_candidate": False})

    def test_diff_survives_non_dict_prop(self):
        # A malformed prop value (bare string) must not crash --diff.
        self._seed_live()
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"weird_prop": "not-a-dict"})
        d = cl.diff_calibration(SK)  # would AttributeError without the guard
        self.assertIn("weird_prop", d["props"]["added"])

    # ── staging-aware refit read (composition) ─────────────────────────
    def test_refit_read_composes_with_staged_candidate(self):
        self._seed_live()
        cl.set_candidate_mode(True)
        # A staged sweep changes a method + residual in the candidate.
        cl.save_calibration(SK, {"batter_tb": {"method": "B", "n_obs": 999}})
        # A subsequent staged pass must read the CANDIDATE's props, not live.
        props = cl.load_calibration_for_refit(SK)
        self.assertEqual(props["batter_tb"]["method"], "B")
        self.assertEqual(props["batter_tb"]["n_obs"], 999)

    def test_refit_read_falls_back_to_live_when_no_candidate(self):
        self._seed_live()
        cl.set_candidate_mode(True)  # staging on, but nothing staged yet
        props = cl.load_calibration_for_refit(SK)
        self.assertEqual(props["batter_tb"]["method"], "E")  # live incumbent

    def test_refit_read_is_live_when_mode_off(self):
        self._seed_live()
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"batter_tb": {"method": "B", "n_obs": 999}})
        cl.set_candidate_mode(False)
        # Not staging → serving-style read is live regardless of the candidate.
        self.assertEqual(cl.load_calibration_for_refit(SK)["batter_tb"]["method"],
                         "E")

    # ── carryover notice + mode accessor ───────────────────────────────
    def test_existing_candidate_notice(self):
        self._seed_live()
        self.assertIsNone(cl.existing_candidate_notice(SK))
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"batter_tb": {"method": "A", "n_obs": 1}})
        note = cl.existing_candidate_notice(SK)
        self.assertIsNotNone(note)
        self.assertIn("ACCUMULATES", note)

    def test_candidate_mode_accessor(self):
        self.assertFalse(cl.candidate_mode_active())
        cl.set_candidate_mode(True)
        self.assertTrue(cl.candidate_mode_active())
        cl.set_candidate_mode(False)
        self.assertFalse(cl.candidate_mode_active())

    # ── promote / discard ──────────────────────────────────────────────
    def test_promote_archives_and_swaps(self):
        self._seed_live()
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"batter_tb": {"method": "A", "n_obs": 320}})
        cl.set_candidate_mode(False)
        archived = cl.promote_calibration(SK)
        # Candidate consumed; live now carries the candidate's value.
        self.assertFalse(cl.has_candidate(SK))
        self.assertEqual(self._live()["props"]["batter_tb"]["method"], "A")
        # Archive holds the OLD live (the E incumbent) as a rollback point.
        self.assertTrue(os.path.exists(archived))
        self.assertEqual(cl._read_json(archived)["props"]["batter_tb"]["method"],
                         "E")

    def test_promote_without_live_returns_none_archive(self):
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"batter_hits": {"method": "B", "n_obs": 10}})
        cl.set_candidate_mode(False)
        archived = cl.promote_calibration(SK)
        self.assertIsNone(archived)
        self.assertEqual(self._live()["props"]["batter_hits"]["method"], "B")

    def test_promote_without_candidate_raises(self):
        self._seed_live()
        with self.assertRaises(FileNotFoundError):
            cl.promote_calibration(SK)

    def test_discard_removes_candidate_leaves_live(self):
        self._seed_live()
        cl.set_candidate_mode(True)
        cl.save_calibration(SK, {"batter_tb": {"method": "A", "n_obs": 320}})
        cl.set_candidate_mode(False)
        self.assertTrue(cl.discard_candidate(SK))
        self.assertFalse(cl.has_candidate(SK))
        # Live intact.
        self.assertEqual(self._live()["props"]["batter_tb"]["method"], "E")
        # Discarding again is a no-op.
        self.assertFalse(cl.discard_candidate(SK))

    # ── Tier A #1d: additive expected-runs block (candidate-staged) ────────
    def test_additive_save_stages_candidate_preserves_blocks(self):
        # save_expected_runs_additive stages to the CANDIDATE (not live) and preserves
        # the other blocks; the serving loader reads LIVE until promote.
        self._seed_live()                      # props (C/E) + prob_shrink block
        model = {"enabled": True, "feature_keys": ["xwobacon", "k9"],
                 "model": {"coef": [1.0, 2.0], "league_rate9": 4.3}}
        cl.set_candidate_mode(True)
        cl.save_expected_runs_additive(SK, model, meta={"seasons": [2024, 2025]})
        cl.set_candidate_mode(False)
        # Live untouched (no additive block yet); serving load reads live -> {}.
        self.assertNotIn("expected_runs_additive", self._live())
        self.assertEqual(cl.load_expected_runs_additive(SK), {})
        # Candidate carries the additive block AND preserves props + prob_shrink.
        cand = self._cand()
        self.assertEqual(cand["expected_runs_additive"], model)
        self.assertEqual(cand["props"]["batter_hits"]["method"], "C")
        self.assertEqual(cand["prob_shrink"]["totals"], 0.5)
        self.assertEqual(cand["meta"]["expected_runs_additive"]["seasons"],
                         [2024, 2025])

    def test_additive_promote_makes_it_live_and_preserves_props(self):
        self._seed_live()
        model = {"enabled": True, "feature_keys": ["xwobacon", "k9"]}
        cl.set_candidate_mode(True)
        cl.save_expected_runs_additive(SK, model)
        cl.set_candidate_mode(False)
        cl.promote_calibration(SK)
        self.assertEqual(cl.load_expected_runs_additive(SK), model)
        self.assertEqual(self._live()["props"]["batter_hits"]["method"], "C")


def load_method(sport_key, prop):
    return (cl.load_calibration(sport_key).get(prop) or {}).get("method")


class TeamMarketResetTests(unittest.TestCase):
    """stage_team_market_reset stages a clean team-market slate in the CANDIDATE
    (live untouched), clears exactly the blocks the fresh RUN regenerates, keeps
    valid challenger shares so the spreads projection can still fire, and preserves
    props + the auxiliary live levers."""

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._orig = cl.CALIBRATION_DIR
        cl.CALIBRATION_DIR = self._dir
        cl.set_candidate_mode(False)

    def tearDown(self):
        cl.CALIBRATION_DIR = self._orig
        cl.set_candidate_mode(False)

    def _live_blob(self):
        return {
            "sport_key": SK,
            "props": {"batter_hits": {"method": "E"}},
            "prob_shrink": {"moneyline": 0.25, "spreads": 0.6, "totals": 0.7},
            "market_blend": {"moneyline": {"w": 0.5}},
            "starter_adjustment": {"enabled": True, "moneyline": 0.35},
            "lineup_adjustment": {"enabled": True, "props": {}},
            "value_gate": {"ev_floor": 0.05, "edge_floor": 0.02},
            "expected_runs_additive": {"enabled": True, "model": {"coef": [1]}},
            "expected_runs_challenger": {
                "enabled": True,
                "live_markets": {"spreads": True},
                "final_2025_validation": {
                    "model": {"offense_weight": 0.5, "pitching_weight": 0.5},
                    "ensemble_challenger_share": {
                        "moneyline": 0.75, "home_minus_1_5": 0.7, "margin": 0.9}}},
            "meta": {"prob_shrink": {"x": 1}, "prob_shrink_holdout": {"y": 2},
                     "current_season": 2026},
        }

    def _write_live(self):
        with open(cl.calibration_path(SK), "w", encoding="utf-8") as f:
            json.dump(self._live_blob(), f)

    def test_reset_clears_team_market_and_preserves_rest(self):
        self._write_live()
        path = cl.stage_team_market_reset(SK)
        self.assertEqual(path, cl.candidate_path(SK))       # candidate, not live
        with open(cl.candidate_path(SK), encoding="utf-8") as f:
            cand = json.load(f)
        # cleared
        self.assertEqual(cand["prob_shrink"], {})
        self.assertNotIn("market_blend", cand)
        share = cand["expected_runs_challenger"]["final_2025_validation"][
            "ensemble_challenger_share"]
        self.assertEqual(share, {"home_minus_1_5": 0.5, "margin": 0.5})  # neutral
        self.assertNotIn("moneyline", share)                # stale sibling stripped
        self.assertNotIn("prob_shrink", cand["meta"])
        self.assertNotIn("prob_shrink_holdout", cand["meta"])
        # preserved
        self.assertEqual(cand["props"], {"batter_hits": {"method": "E"}})
        self.assertEqual(cand["starter_adjustment"], {"enabled": True,
                                                      "moneyline": 0.35})
        self.assertEqual(cand["lineup_adjustment"], {"enabled": True, "props": {}})
        self.assertEqual(cand["value_gate"], {"ev_floor": 0.05, "edge_floor": 0.02})
        chal = cand["expected_runs_challenger"]
        self.assertTrue(chal["enabled"])
        self.assertEqual(chal["live_markets"], {"spreads": True})
        self.assertEqual(chal["final_2025_validation"]["model"],
                         {"offense_weight": 0.5, "pitching_weight": 0.5})

    def test_reset_leaves_live_untouched(self):
        self._write_live()
        cl.stage_team_market_reset(SK)
        cl.set_candidate_mode(False)
        with open(cl.calibration_path(SK), encoding="utf-8") as f:
            live = json.load(f)
        self.assertEqual(live["prob_shrink"],
                         {"moneyline": 0.25, "spreads": 0.6, "totals": 0.7})
        self.assertEqual(live["expected_runs_challenger"]["final_2025_validation"][
            "ensemble_challenger_share"]["home_minus_1_5"], 0.7)

    def test_reset_shares_stay_valid_for_projection(self):
        # The neutral shares MUST be present + in [0,1] so the spreads expected-runs
        # projection can fire (else --fit-shares captures nothing).
        self._write_live()
        cl.stage_team_market_reset(SK)
        cl.set_serving_candidate(True)
        try:
            chal = cl.load_expected_runs_challenger(SK)
            share = chal["final_2025_validation"]["ensemble_challenger_share"]
            self.assertTrue(0.0 <= share["home_minus_1_5"] <= 1.0)
            self.assertTrue(0.0 <= share["margin"] <= 1.0)
        finally:
            cl.set_serving_candidate(False)


if __name__ == "__main__":
    unittest.main()
