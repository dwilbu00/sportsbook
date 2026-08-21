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

    def test_retag_maps_null_rows_and_leaves_tagged(self):
        self._legacy("L-seed", "seed")
        self._legacy("L-team", "team")
        self._legacy("sbr-L", "team")
        self._legacy("L-live", "team", source="live")     # already tagged -> untouched
        with contextlib.redirect_stdout(io.StringIO()):
            op._retag(apply=True)
        s = self._sources()
        self.assertEqual(s["L-seed"], "seed")
        self.assertEqual(s["L-team"], "live")
        self.assertEqual(s["sbr-L"], "sbr")
        self.assertEqual(s["L-live"], "live")             # non-null, non-backfill -> untouched

    def test_retag_splits_backfill_by_timing(self):
        # close: captured at commence (0h). early: captured 12h before (morning).
        self._legacy("bf-close", "team", source="backfill")   # cap == commence (18Z)
        with self.eng.begin() as c:
            c.execute(insert(self.t).values(
                sport="baseball_mlb", game_date="2024-06-02", event_id="bf-early",
                kind="team", snapshot_hour="he", captured_at="2024-06-02T11:00:00Z",
                commence_time="2024-06-02T23:00:00Z", source="backfill"))
        with contextlib.redirect_stdout(io.StringIO()):
            op._retag(apply=True)
        s = self._sources()
        self.assertEqual(s["bf-close"], "backfill_close")
        self.assertEqual(s["bf-early"], "backfill_early")

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
            op._prune_seed("baseball_mlb", ["2024", "2025"], apply=True, yes=True)
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
            op._prune_seed("baseball_mlb", None, apply=True, yes=True)  # None = all years
        self.assertEqual(set(self._sources()), {"t24", "nba-seed"})

    def test_prune_dry_run_deletes_nothing(self):
        self._legacy("mlb-seed-24", "seed", gd="2024-06-01")
        with contextlib.redirect_stdout(io.StringIO()):
            op._prune_seed("baseball_mlb", ["2024", "2025"], apply=False, yes=False)
        self.assertIn("mlb-seed-24", self._sources())   # still there


if __name__ == "__main__":
    unittest.main()
