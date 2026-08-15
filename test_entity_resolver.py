"""Tests for the odds-boundary MLB player identity resolver (entity_resolver).

Post-Commit-C/P5 the id-core is UNCONDITIONALLY delegated to
mlb_starters.resolve_mlbam_id (the old ODI_MLB_STAMP_RESOLVER gate + SFBB-only
id-core + player_alias write are retired); it is monkeypatched here while the team +
game_pk envelope runs against real fixtures on SQLite. Covers: MLB gating, the
delegated resolve/miss, is_pitcher passthrough, prop_key/lineup/probables threading,
crash-vs-unresolved (None), the schedule gap-fill, game_pk independence, the dict
contract, and the never-raises guarantee.
"""

import unittest
from unittest import mock

from sqlalchemy import insert, select

import db_store
import entity_resolver as er
import mlb_starters
import mlb_warehouse


def _rows(table):
    with db_store.get_engine().connect() as conn:
        return conn.execute(select(table)).fetchall()


def _count(table):
    return len(_rows(table))


class _Backend:
    def setUp(self):
        db_store.configure_engine("sqlite://")
        mlb_warehouse.create_all()
        mlb_warehouse._TEAMS_ENSURED.clear()

    def tearDown(self):
        db_store.configure_engine(None)
        mlb_warehouse._TEAMS_ENSURED.clear()


class ResolveTests(_Backend, unittest.TestCase):
    def setUp(self):
        super().setUp()
        for tid, nm in (("147", "New York Yankees"), ("111", "Boston Red Sox")):
            with db_store.get_engine().begin() as conn:
                conn.execute(insert(mlb_warehouse.mlb_team), {
                    "team_id": tid, "name": nm,
                    "name_norm": db_store.normalize_name(nm)})
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": 700, "official_date": "2026-08-09",
                "game_date": "2026-08-09T23:05:00Z", "home_team_id": "147",
                "away_team_id": "111", "season": 2026})

    def _resolve(self, name="Aaron Judge", **kw):
        return er.resolve(name, "baseball_mlb", "New York Yankees",
                          "Boston Red Sox", game_date="2026-08-09",
                          commence="2026-08-09T23:05:00Z", **kw)

    def test_non_mlb_unresolved(self):
        r = er.resolve("LeBron James", "basketball_nba", "LAL", "BOS")
        self.assertFalse(r["resolved"])
        self.assertEqual(r["method"], "unresolved")
        self.assertEqual(r["reason"], "non_mlb_or_no_name")

    def test_blank_name_unresolved(self):
        self.assertFalse(er.resolve("", "baseball_mlb", "A", "B")["resolved"])

    def test_resolved_delegates_to_game_context_core(self):
        with mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(592450, False)) as core:
            r = self._resolve()
        core.assert_called_once()
        self.assertTrue(r["resolved"])
        self.assertEqual(r["mlb_player_id"], "592450")   # int → str
        self.assertEqual(r["game_pk"], 700)              # nearest game to commence
        self.assertIs(r["is_pitcher"], False)            # statsapi-authoritative
        self.assertEqual(r["method"], "game_context_resolver")
        self.assertEqual(_count(mlb_warehouse.player_alias), 0)  # alias write retired

    def test_is_pitcher_passthrough(self):
        with mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(543037, True)):
            r = self._resolve(name="Gerrit Cole")
        self.assertIs(r["is_pitcher"], True)

    def test_threads_prop_key_teams_lineup_probables(self):
        lineup = {"players": {}}
        probs = {"new york yankees": {"pitcher_id": 1, "name": "Gerrit Cole"}}
        with mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(543037, True)) as core:
            self._resolve(name="Gerrit Cole", prop_key="pitcher_strikeouts",
                          confirmed_lineup=lineup, probable_starters=probs)
        kw = core.call_args.kwargs
        self.assertEqual(kw["prop_key"], "pitcher_strikeouts")
        self.assertEqual(kw["teams"], ["New York Yankees", "Boston Red Sox"])
        self.assertIs(kw["confirmed_lineup"], lineup)
        self.assertIs(kw["probable_starters"], probs)

    def test_miss_is_fail_closed_but_game_pk_still_resolves(self):
        with mock.patch.object(mlb_starters, "resolve_mlbam_id", return_value=None):
            r = self._resolve(name="Ambiguous Nobody")
        self.assertFalse(r["resolved"])
        self.assertIsNone(r["mlb_player_id"])
        self.assertEqual(r["reason"], "ambiguous_or_unknown")
        self.assertEqual(r["game_pk"], 700)              # game resolves independently

    def test_delegate_raise_is_swallowed_to_unresolved(self):
        # A raise from the delegate is an ordinary unresolved (fail-open shadow row),
        # NOT the envelope's resolver_error — and game_pk still resolves.
        with mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               side_effect=RuntimeError("boom")):
            r = self._resolve()
        self.assertFalse(r["resolved"])
        self.assertEqual(r["reason"], "ambiguous_or_unknown")
        self.assertEqual(r["game_pk"], 700)

    def test_envelope_error_is_resolver_error(self):
        # An error in the ENVELOPE (game_pk derivation) → resolver_error.
        with mock.patch.object(mlb_warehouse, "team_id_for_name",
                               side_effect=RuntimeError("boom")):
            r = self._resolve()
        self.assertFalse(r["resolved"])
        self.assertEqual(r["reason"], "resolver_error")

    def test_no_commence_leaves_game_pk_none(self):
        with mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(592450, False)):
            r = er.resolve("Aaron Judge", "baseball_mlb", "New York Yankees",
                           "Boston Red Sox", game_date="2026-08-09")
        self.assertTrue(r["resolved"])
        self.assertIsNone(r["game_pk"])

    def test_dict_contract_resolved_vs_unresolved(self):
        # Every return (resolved / unresolved) carries the identical key-set, so
        # every stamp consumer reads the same keys.
        with mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(592450, False)):
            ok = self._resolve()
        with mock.patch.object(mlb_starters, "resolve_mlbam_id", return_value=None):
            miss = self._resolve(name="Nobody")
        self.assertEqual(set(ok.keys()), set(miss.keys()))


class GapFillTests(_Backend, unittest.TestCase):
    """On a game_pk miss, resolve ingests that date's schedule once, then retries.
    The id-core is mocked; game_pk is derived by the envelope independent of the id."""

    def test_gap_fill_on_miss_then_retry(self):
        er._GAP_FILLED.clear()
        seq = iter([None, 700])            # miss, then hit after the gap-fill ingest
        with mock.patch.object(mlb_warehouse, "team_id_for_name", return_value="147"), \
             mock.patch.object(mlb_warehouse, "find_game_pk_by_commence",
                               side_effect=lambda *a: next(seq)), \
             mock.patch.object(mlb_warehouse, "ingest_date") as ing, \
             mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(592450, False)):
            r = er.resolve("Aaron Judge", "baseball_mlb", "New York Yankees",
                           "Boston Red Sox", commence="2026-08-09T23:00:00Z")
        self.assertEqual(r["game_pk"], 700)
        ing.assert_called()                # gap-fill ingested the schedule (no boxscores)
        self.assertFalse(ing.call_args.kwargs.get("with_boxscores", True))

    def test_gap_fill_deduped_per_process(self):
        er._GAP_FILLED.clear()
        with mock.patch.object(mlb_warehouse, "team_id_for_name", return_value="147"), \
             mock.patch.object(mlb_warehouse, "find_game_pk_by_commence",
                               return_value=None), \
             mock.patch.object(mlb_warehouse, "ingest_date") as ing, \
             mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(592450, False)):
            er.resolve("X", "baseball_mlb", "New York Yankees", "Boston Red Sox",
                       commence="2026-08-09T23:00:00Z")
            n1 = ing.call_count
            er.resolve("Y", "baseball_mlb", "New York Yankees", "Boston Red Sox",
                       commence="2026-08-09T23:00:00Z")
            n2 = ing.call_count
        self.assertEqual(n1, n2)           # same date already gap-filled → no re-ingest

    def test_no_gap_fill_when_game_pk_hits(self):
        er._GAP_FILLED.clear()
        with mock.patch.object(mlb_warehouse, "team_id_for_name", return_value="147"), \
             mock.patch.object(mlb_warehouse, "find_game_pk_by_commence",
                               return_value=700), \
             mock.patch.object(mlb_warehouse, "ingest_date") as ing, \
             mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(592450, False)):
            r = er.resolve("Aaron Judge", "baseball_mlb", "New York Yankees",
                           "Boston Red Sox", commence="2026-08-09T23:00:00Z")
        self.assertEqual(r["game_pk"], 700)
        ing.assert_not_called()            # already resolvable → no ingest on the hot path


if __name__ == "__main__":
    unittest.main()
