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


def load_method(sport_key, prop):
    return (cl.load_calibration(sport_key).get(prop) or {}).get("method")


if __name__ == "__main__":
    unittest.main()
