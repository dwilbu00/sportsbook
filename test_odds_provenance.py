"""Tests for odds provenance: source tagging + the seed retag/prune maintenance."""
import io
import contextlib
import unittest

from sqlalchemy import select, insert

import db_store
import odds_provenance as op


class OddsProvenanceTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        self.t = db_store.odds_snapshot
        self.eng = db_store.get_engine()
        # odds_provenance._engine() gates on SQL secrets; point it at the test engine.
        self._orig = op._engine
        op._engine = db_store.get_engine
        self.addCleanup(lambda: setattr(op, "_engine", self._orig))

    def _legacy(self, eid, kind, source=None, gd="2024-06-01", sport="baseball_mlb"):
        # Direct insert = a pre-column legacy row (source NULL), bypassing
        # capture_odds_snapshot's None->'live' default.
        with self.eng.begin() as c:
            c.execute(insert(self.t).values(
                sport=sport, game_date=gd, event_id=eid, kind=kind,
                snapshot_hour="h" + eid, captured_at=gd + "T18:00:00Z",
                commence_time=gd + "T18:00:00Z", source=source))

    def _sources(self):
        with self.eng.connect() as c:
            return dict(c.execute(select(self.t.c.event_id, self.t.c.source)).all())

    def test_capture_defaults_live_and_stores_explicit_source(self):
        db_store.capture_odds_snapshot(
            {"sport": "baseball_mlb", "game_date": "2024-06-01", "event_id": "A",
             "kind": "team", "snapshot_hour": "hA", "captured_at": "x",
             "commence_time": "x"}, [])   # no source -> defaults 'live'
        db_store.capture_odds_snapshot(
            {"sport": "baseball_mlb", "game_date": "2024-06-01", "event_id": "B",
             "kind": "team", "snapshot_hour": "hB", "captured_at": "x",
             "commence_time": "x", "source": "backfill"}, [])
        s = self._sources()
        self.assertEqual(s["A"], "live")
        self.assertEqual(s["B"], "backfill")

    def test_classify_positive_rules_and_date_gate(self):
        # seed by kind, sbr by id prefix — regardless of live_since
        self.assertEqual(op._classify("seed", "x", "2024-01-01", None), "seed")
        self.assertEqual(op._classify("team", "sbr-x", "2024-01-01", None), "sbr")
        # ambiguous non-seed/non-sbr with NO boundary -> left untagged (never live)
        self.assertIsNone(op._classify("team", "abc", "2024-01-01", None))
        self.assertIsNone(op._classify("props", "abc", "2026-06-01", None))
        # date gate: >= live_since -> live, earlier -> backfill_close
        self.assertEqual(op._classify("team", "abc", "2026-06-01", "2026-01-01"), "live")
        self.assertEqual(op._classify("team", "abc", "2025-06-01", "2026-01-01"),
                         "backfill_close")

    def test_retag_without_live_since_leaves_ambiguous_null(self):
        self._legacy("L-seed", "seed", gd="2024-06-01")
        self._legacy("L-team", "team", gd="2024-06-01")       # ambiguous, no boundary
        self._legacy("sbr-L", "team", gd="2024-06-01")
        self._legacy("L-live", "team", source="live", gd="2026-06-01")  # tagged -> untouched
        with contextlib.redirect_stdout(io.StringIO()):
            op._retag(apply=True)                              # no live_since
        s = self._sources()
        self.assertEqual(s["L-seed"], "seed")
        self.assertIsNone(s["L-team"])                        # left NULL, NOT 'live'
        self.assertEqual(s["sbr-L"], "sbr")
        self.assertEqual(s["L-live"], "live")

    def test_retag_with_live_since_splits_by_date(self):
        self._legacy("seed24", "seed", gd="2024-06-01")
        self._legacy("hist23", "team", gd="2023-06-01")       # < boundary -> backfill_close
        self._legacy("hist25", "props", gd="2025-06-01")      # < boundary -> backfill_close
        self._legacy("live26", "team", gd="2026-06-01")       # >= boundary -> live
        with contextlib.redirect_stdout(io.StringIO()):
            op._retag(apply=True, live_since="2026-01-01")
        s = self._sources()
        self.assertEqual(s["seed24"], "seed")
        self.assertEqual(s["hist23"], "backfill_close")
        self.assertEqual(s["hist25"], "backfill_close")
        self.assertEqual(s["live26"], "live")

    def test_all_backfill_tags_every_ambiguous_row_without_live_since(self):
        # a sport never live-captured: no date gate, everything non-seed/non-sbr
        # becomes backfill_close (incl. rows dated 2026).
        self._legacy("soc-seed", "seed", gd="2025-06-01", sport="soccer_epl")
        self._legacy("soc-a", "team", gd="2024-06-01", sport="soccer_epl")
        self._legacy("soc-b", "props", gd="2026-06-01", sport="soccer_epl")
        with contextlib.redirect_stdout(io.StringIO()):
            op._retag(apply=True, sport_key="soccer_epl", all_backfill=True)
        s = self._sources()
        self.assertEqual(s["soc-seed"], "seed")
        self.assertEqual(s["soc-a"], "backfill_close")
        self.assertEqual(s["soc-b"], "backfill_close")   # 2026 too — no live gate

    def test_classify_all_backfill_overrides_date(self):
        self.assertEqual(
            op._classify("team", "abc", "2026-06-01", None, all_backfill=True),
            "backfill_close")
        self.assertEqual(  # seed/sbr still win over all_backfill
            op._classify("seed", "abc", "2026-06-01", None, all_backfill=True), "seed")

    def test_resolve_sport_aliases_all_and_raw_key(self):
        self.assertEqual(op._resolve_sport("mlb"), "baseball_mlb")
        self.assertIsNone(op._resolve_sport("all"))
        self.assertIsNone(op._resolve_sport(None))
        self.assertEqual(op._resolve_sport("soccer_epl"), "soccer_epl")  # raw passthrough

    def test_retag_scoped_by_sport_and_years(self):
        self._legacy("mlb26", "team", gd="2026-06-01", sport="baseball_mlb")
        self._legacy("nba26", "team", gd="2026-06-01", sport="basketball_nba")
        self._legacy("mlb25", "team", gd="2025-06-01", sport="baseball_mlb")
        with contextlib.redirect_stdout(io.StringIO()):
            # only MLB, only 2026 -> just mlb26 is touched
            op._retag(apply=True, sport_key="baseball_mlb", years=["2026"],
                      live_since="2026-01-01")
        s = self._sources()
        self.assertEqual(s["mlb26"], "live")
        self.assertIsNone(s["nba26"])      # out of sport scope
        self.assertIsNone(s["mlb25"])      # out of year scope

    def test_dry_run_writes_nothing(self):
        self._legacy("d-team", "team", gd="2026-06-01")
        with contextlib.redirect_stdout(io.StringIO()):
            op._retag(apply=False, live_since="2026-01-01")
        self.assertIsNone(self._sources()["d-team"])

    def test_prune_seed_scoped_and_cascades(self):
        self._legacy("mlb-seed-24", "seed", gd="2024-06-01")
        self._legacy("mlb-seed-25", "seed", gd="2025-06-01")
        self._legacy("mlb-seed-23", "seed", gd="2023-06-01")     # out of year scope
        self._legacy("mlb-team-24", "team", gd="2024-06-01")     # not seed
        self._legacy("nba-seed-24", "seed", gd="2024-06-01", sport="basketball_nba")  # wrong sport
        # attach a line to a to-be-deleted snapshot to prove cascade
        with self.eng.connect() as c:
            sid = c.execute(select(self.t.c.id)
                            .where(self.t.c.event_id == "mlb-seed-24")).scalar()
        with self.eng.begin() as c:
            c.execute(insert(db_store.odds_line).values(
                snapshot_id=sid, bet_type="moneyline", selection="H", price=-120))
        with contextlib.redirect_stdout(io.StringIO()):
            op._prune("baseball_mlb", ["2024", "2025"], apply=True, yes=True, kind="seed")
        remaining = set(self._sources())
        self.assertEqual(
            remaining, {"mlb-seed-23", "mlb-team-24", "nba-seed-24"})
        with self.eng.connect() as c:
            n_line = c.execute(select(db_store.odds_line)).all()
        self.assertEqual(len(n_line), 0)   # the seed snapshot's line cascaded away

    def test_prune_all_years_removes_every_seed_for_sport(self):
        self._legacy("s24", "seed", gd="2024-06-01")
        self._legacy("s26", "seed", gd="2026-06-01")
        self._legacy("t24", "team", gd="2024-06-01")
        self._legacy("nba-seed", "seed", gd="2024-06-01", sport="basketball_nba")
        with contextlib.redirect_stdout(io.StringIO()):
            op._prune("baseball_mlb", None, apply=True, yes=True, kind="seed")  # all years
        self.assertEqual(set(self._sources()), {"t24", "nba-seed"})

    def test_prune_dry_run_deletes_nothing(self):
        self._legacy("mlb-seed-24", "seed", gd="2024-06-01")
        with contextlib.redirect_stdout(io.StringIO()):
            op._prune("baseball_mlb", ["2024", "2025"], apply=False, yes=False, kind="seed")
        self.assertIn("mlb-seed-24", self._sources())   # still there

    def test_prune_by_source_drops_2026_live_only(self):
        # clean-slate: prune the thin 2026 pre-relaunch live odds, keep the corpus.
        self._legacy("live26", "team", source="live", gd="2026-06-01")
        self._legacy("bf25", "team", source="backfill_close", gd="2025-06-01")
        self._legacy("live25", "team", source="live", gd="2025-06-01")  # out of year scope
        with contextlib.redirect_stdout(io.StringIO()):
            op._prune("baseball_mlb", ["2026"], apply=True, yes=True, source="live")
        remaining = set(self._sources())
        self.assertEqual(remaining, {"bf25", "live25"})   # only 2026-live dropped

    def test_prune_source_null_drops_only_untagged(self):
        # clean-slate: prune the untagged (source IS NULL) legacy cruft, keep tagged.
        self._legacy("cruft1", "props", gd="2025-06-01")               # source NULL
        self._legacy("cruft2", "team", gd="2024-06-01")                # source NULL
        self._legacy("tagged", "props", source="backfill_close", gd="2025-06-01")
        with contextlib.redirect_stdout(io.StringIO()):
            op._prune("baseball_mlb", None, apply=True, yes=True, source="null")
        remaining = set(self._sources())
        self.assertEqual(remaining, {"tagged"})   # only NULL-source rows dropped

    def test_prune_without_filter_refuses(self):
        self._legacy("x", "team", gd="2026-06-01")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                op._prune("baseball_mlb", None, apply=True, yes=True)  # no kind/source
        self.assertIn("x", self._sources())   # nothing deleted


if __name__ == "__main__":
    unittest.main()
