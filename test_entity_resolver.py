"""Tests for the P3 odds-boundary MLB player identity resolver (entity_resolver).

The MLBAM/SFBB identity lookups (mlb_starters.find_player_id,
player_id_map.mlb_id_for_name) are monkeypatched; the team + game resolution runs
against real fixtures on SQLite. Covers: MLB gating, fail-closed misses, the
globally-unique alias write vs the shared-name no-write, game_pk independence,
and the never-raises contract.
"""

import os
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

    def test_hit_globally_unique_writes_alias(self):
        # mlb_id_for_name resolves regardless of the team hint (globally unique).
        with mock.patch("player_id_map.mlb_id_for_name", return_value="592450"):
            r = self._resolve()
        self.assertTrue(r["resolved"])
        self.assertEqual(r["mlb_player_id"], "592450")
        self.assertEqual(r["game_pk"], 700)              # nearest game to commence
        self.assertEqual(r["method"], "sfbb_unique")
        rows = [x._mapping for x in _rows(mlb_warehouse.player_alias)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "oddsapi")
        self.assertEqual(rows[0]["provider_key"], "aaron judge")
        self.assertEqual(rows[0]["mlb_player_id"], "592450")

    def test_hit_shared_name_hinted_writes_no_bare_alias(self):
        # Bare name ambiguous (teams=None → None); the two-team hint disambiguates
        # → resolved but NOT bare-aliased (would serve the wrong namesake later).
        def _by_team(name, teams=None):
            return None if teams is None else "111111"
        with mock.patch("player_id_map.mlb_id_for_name", side_effect=_by_team):
            r = self._resolve(name="Will Smith")
        self.assertTrue(r["resolved"])
        self.assertEqual(r["mlb_player_id"], "111111")
        self.assertEqual(r["method"], "sfbb_hinted")
        self.assertEqual(_count(mlb_warehouse.player_alias), 0)

    def test_miss_is_fail_closed_but_game_pk_still_resolves(self):
        with mock.patch("player_id_map.mlb_id_for_name", return_value=None):
            r = self._resolve(name="Ambiguous Nobody")
        self.assertFalse(r["resolved"])
        self.assertIsNone(r["mlb_player_id"])
        self.assertEqual(r["reason"], "ambiguous_or_unknown")
        self.assertEqual(r["game_pk"], 700)              # game resolves independently
        self.assertEqual(_count(mlb_warehouse.player_alias), 0)

    def test_is_pitcher_not_derived(self):
        with mock.patch("player_id_map.mlb_id_for_name", return_value="543037"):
            r = self._resolve(name="Gerrit Cole")
        self.assertIsNone(r["is_pitcher"])               # P3 resolver doesn't derive role

    def test_never_raises(self):
        with mock.patch("player_id_map.mlb_id_for_name",
                        side_effect=RuntimeError("boom")):
            r = self._resolve()
        self.assertFalse(r["resolved"])
        self.assertEqual(r["reason"], "resolver_error")

    def test_no_commence_leaves_game_pk_none(self):
        with mock.patch("player_id_map.mlb_id_for_name", return_value="592450"):
            r = er.resolve("Aaron Judge", "baseball_mlb", "New York Yankees",
                           "Boston Red Sox", game_date="2026-08-09")
        self.assertTrue(r["resolved"])
        self.assertIsNone(r["game_pk"])


class GapFillTests(_Backend, unittest.TestCase):
    """On a game_pk miss, resolve ingests that date's schedule once, then retries."""

    def test_gap_fill_on_miss_then_retry(self):
        er._GAP_FILLED.clear()
        seq = iter([None, 700])            # miss, then hit after the gap-fill ingest
        with mock.patch.object(mlb_warehouse, "team_id_for_name", return_value="147"), \
             mock.patch.object(mlb_warehouse, "find_game_pk_by_commence",
                               side_effect=lambda *a: next(seq)), \
             mock.patch.object(mlb_warehouse, "ingest_date") as ing, \
             mock.patch("player_id_map.mlb_id_for_name", return_value="592450"):
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
             mock.patch("player_id_map.mlb_id_for_name", return_value="592450"):
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
             mock.patch("player_id_map.mlb_id_for_name", return_value="592450"):
            r = er.resolve("Aaron Judge", "baseball_mlb", "New York Yankees",
                           "Boston Red Sox", commence="2026-08-09T23:00:00Z")
        self.assertEqual(r["game_pk"], 700)
        ing.assert_not_called()            # already resolvable → no ingest on the hot path


class StampResolverGateTests(_Backend, unittest.TestCase):
    """Commit C Phase 1: the ODI_MLB_STAMP_RESOLVER gate. OFF (default) is
    byte-identical (SFBB-only id-core, mlb_starters untouched); ON delegates the
    id-core to mlb_starters.resolve_mlbam_id while the envelope (game_pk, gap-fill,
    dict contract) is unchanged."""

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

    def _resolve(self, name="Aaron Judge", on=True, **kw):
        env = {er._STAMP_RESOLVER_ENV: "1" if on else ""}
        with mock.patch.dict(os.environ, env):
            return er.resolve(name, "baseball_mlb", "New York Yankees",
                              "Boston Red Sox", game_date="2026-08-09",
                              commence="2026-08-09T23:05:00Z", **kw)

    def test_off_uses_sfbb_core_and_never_calls_new_resolver(self):
        # Gate OFF: the legacy SFBB id-core runs; the game-context resolver is not
        # touched (byte-identical to pre-Commit-C).
        with mock.patch.object(mlb_starters, "resolve_mlbam_id") as new, \
             mock.patch("player_id_map.mlb_id_for_name", return_value="592450"):
            r = self._resolve(on=False)
        new.assert_not_called()
        self.assertTrue(r["resolved"])
        self.assertEqual(r["mlb_player_id"], "592450")
        self.assertEqual(r["method"], "sfbb_unique")

    def test_on_delegates_and_bypasses_sfbb_core(self):
        # Gate ON: delegate resolves; the SFBB player_id_map lookup is not used.
        with mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(592450, False)) as new, \
             mock.patch("player_id_map.mlb_id_for_name") as sfbb:
            r = self._resolve()
        new.assert_called_once()
        sfbb.assert_not_called()
        self.assertTrue(r["resolved"])
        self.assertEqual(r["mlb_player_id"], "592450")   # int → str
        self.assertIs(r["is_pitcher"], False)            # statsapi-authoritative
        self.assertEqual(r["method"], "game_context_resolver")
        self.assertEqual(r["game_pk"], 700)              # envelope unchanged
        self.assertEqual(_count(mlb_warehouse.player_alias), 0)  # no bare alias

    def test_on_threads_prop_key_lineup_probables(self):
        lineup = {"players": {}}
        probs = {"new york yankees": {"pitcher_id": 1, "name": "Gerrit Cole"}}
        with mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               return_value=(543037, True)) as new:
            self._resolve(name="Gerrit Cole", prop_key="pitcher_strikeouts",
                          confirmed_lineup=lineup, probable_starters=probs)
        kw = new.call_args.kwargs
        self.assertEqual(kw["prop_key"], "pitcher_strikeouts")
        self.assertEqual(kw["teams"], ["New York Yankees", "Boston Red Sox"])
        self.assertIs(kw["confirmed_lineup"], lineup)
        self.assertIs(kw["probable_starters"], probs)

    def test_on_none_is_fail_closed_but_game_pk_resolves(self):
        # Delegate fails closed (None) → unresolved, but game_pk still resolves.
        with mock.patch.object(mlb_starters, "resolve_mlbam_id", return_value=None):
            r = self._resolve(name="Ambiguous Nobody")
        self.assertFalse(r["resolved"])
        self.assertIsNone(r["mlb_player_id"])
        self.assertEqual(r["reason"], "ambiguous_or_unknown")
        self.assertEqual(r["game_pk"], 700)

    def test_on_delegate_raise_is_swallowed_to_unresolved(self):
        # resolve_mlbam_id already swallows infra errors to None, but belt-and-
        # suspenders: a raise from the delegate is treated as an ordinary
        # unresolved (fail-open shadow row), NOT the envelope's resolver_error.
        with mock.patch.object(mlb_starters, "resolve_mlbam_id",
                               side_effect=RuntimeError("boom")):
            r = self._resolve()
        self.assertFalse(r["resolved"])
        self.assertEqual(r["reason"], "ambiguous_or_unknown")
        self.assertEqual(r["game_pk"], 700)              # envelope still ran


if __name__ == "__main__":
    unittest.main()
