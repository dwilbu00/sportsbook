"""Tests for the SFBB player/team ID cross-maps (player_id_map), exercised against
in-memory SQLite so pymssql and the live Azure DB are never touched. Mirrors
test_gamelog_store.py: the SQL backend is enabled via configure_engine (tests never
set SQL_* env), and tearDown clears the override. The network is always mocked —
requests.get is patched to serve fixed CSV text.
"""

import unittest
from unittest.mock import patch

import db_store
import player_id_map


# Real SFBB headers (subset the parser reads; csv.DictReader keys by header name so
# omitted columns are simply absent → .get() returns None). A leading BOM (﻿)
# on the header exercises the utf-8-sig decode. Cases embedded:
#   * Mike Trout        — unique, active (round-trip anchor)
#   * Will Smith ×2      — two ACTIVE namesakes, different MLBIDs → ambiguous → None
#   * Jose Ramirez ×2    — active + inactive namesake → inactive dropped, active resolves
#   * Shohei Ohtani ×2   — two-way: two rows, SAME MLBID → resolves (not ambiguous)
#   * Prospect Guy       — active, blank MLBID → mlb lookups None, name still stored
#   * Retired Guy        — ACTIVE=N → dropped entirely (active-only load)
#   * Burn Dupe ×2       — the real 'burneaj01' crash: two rows, SAME IDPLAYER, BOTH
#                          inactive → both dropped (so they can't collide on UNIQUE)
#   * Twin Guy ×2        — two rows, SAME IDPLAYER, BOTH active → deduped to one,
#                          keeping the MLBID-bearing row (defensive UNIQUE guard)
_PLAYERS_CSV = "﻿" + "\n".join([
    "IDPLAYER,PLAYERNAME,MLBNAME,MLBID,ESPNID,BREFID,IDFANGRAPHS,TEAM,POS,ALLPOS,BATS,THROWS,DRAFTKINGSNAME,ACTIVE",
    "sfbb001,Mike Trout,Mike Trout,545361,30836,troutmi01,10155,LAA,OF,OF,R,R,Mike Trout,Y",
    "sfbb002,Will Smith,Will Smith,669257,42403,smithwi05,17137,LAD,C,C,R,R,Will Smith,Y",
    "sfbb003,Will Smith,Will Smith,519293,32821,smithwi02,10664,KC,P,P,L,L,Will Smith,Y",
    "sfbb004,Jose Ramirez,Jose Ramirez,608070,5361,ramirjo01,13510,CLE,3B,3B,S,R,Jose Ramirez,Y",
    "sfbb005,Jose Ramirez,Jose Ramirez,000001,9999,ramirjo99,99999,FA,P,P,R,R,Jose Ramirez,N",
    "sfbb006,Shohei Ohtani,Shohei Ohtani,660271,39832,ohtansh01,19755,LAD,DH,DH,L,R,Shohei Ohtani,Y",
    "sfbb007,Shohei Ohtani,Shohei Ohtani,660271,39832,ohtansh01,19755,LAD,P,P,L,R,Shohei Ohtani,Y",
    "sfbb008,Prospect Guy,Prospect Guy,,88888,,,SD,SS,SS,R,R,Prospect Guy,Y",
    "sfbb009,Retired Guy,Retired Guy,400001,44444,retgu01,40001,FA,OF,OF,R,R,Retired Guy,N",
    "burndup1,Burn Dupe,Burn Dupe,410001,45001,burndup1,41001,FA,P,P,R,R,,N",
    "burndup1,Burn Dupe,Burn Dupe,410002,45002,burndup1,41002,FA,P,P,R,R,,N",
    "twindup1,Twin Guy,Twin Guy,,46001,twindup1,42001,NYY,OF,OF,R,R,Twin Guy,Y",
    "twindup1,Twin Guy,Twin Guy,777001,46002,twindup1,42002,NYY,OF,OF,R,R,Twin Guy,Y",
])

# Real SFBB team headers (subset). CLE nickname is deliberately the stale "Indians";
# WAS uses ESPN code WSH; ATH is the relocated Athletics.
_TEAMS_CSV = "\n".join([
    "SFBBTEAM,DKTEAM,ESPNTEAM,BBREFTEAM,FANGRAPHSABBR,RETROSHEET,FANGRAPHSTEAM",
    "ARI,ARI,ARI,ARI,ARI,ARI,Diamondbacks",
    "CLE,CLE,CLE,CLE,CLE,CLE,Indians",
    "ATH,ATH,ATH,ATH,ATH,SAC,Athletics",
    "WAS,WAS,WSH,WSN,WAS,WAS,Nationals",
    "CHW,CWS,CHW,CHW,CHW,CHA,White Sox",
    "LAA,LAA,LAA,LAA,LAA,ANA,Angels",
])


class _FakeResp:
    def __init__(self, text):
        self.content = text.encode("utf-8")

    def raise_for_status(self):
        pass


class _Backend:
    def setUp(self):
        db_store.configure_engine("sqlite://")
        player_id_map.create_all()
        self._reset_module_state()

    def tearDown(self):
        db_store.configure_engine(None)     # → enabled() False (no SQL_* env)
        self._reset_module_state()

    @staticmethod
    def _reset_module_state():
        player_id_map._invalidate_index("player")
        player_id_map._invalidate_index("team")
        player_id_map._LAST_FRESH_CHECK = {"player": 0.0, "team": 0.0}
        player_id_map._KEY_LOCKS.clear()

    def _load(self, players=_PLAYERS_CSV, teams=_TEAMS_CSV):
        def fake_get(url, headers=None, timeout=None, allow_redirects=None):
            return _FakeResp(players if "PLAYER" in url.upper() else teams)
        with patch.object(player_id_map.requests, "get", side_effect=fake_get):
            player_id_map.refresh_players()
            player_id_map.refresh_teams()
        self._reset_module_state()          # force lookups to rebuild from SQL


class SchemaParityTests(unittest.TestCase):
    """Guard the column-name SPECs against the Table definitions so they can't
    silently drift from sql/schema.sql (which mirrors these for the Azure DDL)."""

    def test_player_cols_match_table(self):
        self.assertEqual({c.name for c in player_id_map.player_id_map.columns},
                         set(player_id_map._PLAYER_COLS))

    def test_team_cols_match_table(self):
        self.assertEqual({c.name for c in player_id_map.team_id_map.columns},
                         set(player_id_map._TEAM_COLS))

    def test_meta_cols_match_table(self):
        self.assertEqual({c.name for c in player_id_map.id_map_meta.columns},
                         set(player_id_map._META_COLS))


class FetchParseTests(_Backend, unittest.TestCase):

    def test_browser_ua_and_redirects(self):
        captured = {}

        def fake_get(url, headers=None, timeout=None, allow_redirects=None):
            captured["headers"] = headers
            captured["allow_redirects"] = allow_redirects
            return _FakeResp(_PLAYERS_CSV)

        with patch.object(player_id_map.requests, "get", side_effect=fake_get):
            player_id_map.refresh_players()
        self.assertIn("Mozilla/5.0", captured["headers"]["User-Agent"])
        self.assertTrue(captured["allow_redirects"])

    def test_row_counts_and_bom_stripped(self):
        self._load()
        engine = db_store.get_engine()
        with engine.connect() as conn:
            from sqlalchemy import func, select
            n_players = conn.execute(
                select(func.count()).select_from(player_id_map.player_id_map)).scalar()
            n_teams = conn.execute(
                select(func.count()).select_from(player_id_map.team_id_map)).scalar()
        # 13 CSV rows → 8 stored: 4 inactive dropped (Jose 000001, Retired Guy, and
        # both Burn Dupe rows), the two active Twin Guy rows deduped to one. Trout,
        # both Will Smiths, active Jose, both Ohtani rows, Prospect Guy (active,
        # blank MLBID), and the deduped Twin Guy remain.
        self.assertEqual(n_players, 8)
        self.assertEqual(n_teams, 6)
        # BOM stripped → the first row's IDPLAYER key is clean, so Trout is found.
        self.assertEqual(player_id_map.mlb_id_for_name("Mike Trout"), "545361")

    def test_meta_written(self):
        self._load()
        engine = db_store.get_engine()
        with engine.connect() as conn:
            meta = player_id_map._read_meta(conn, "player")
        self.assertEqual(meta["row_count"], 8)
        self.assertIsNotNone(meta["last_fetched_at"])


class PlayerLookupTests(_Backend, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._load()

    def test_unique_name_resolves(self):
        self.assertEqual(player_id_map.mlb_id_for_name("Mike Trout"), "545361")
        self.assertEqual(player_id_map.espn_id_for_name("Mike Trout"), "30836")

    def test_two_active_namesakes_are_ambiguous(self):
        # Two Will Smiths, both active, different MLBIDs → refuse (drop-ambiguous).
        self.assertIsNone(player_id_map.mlb_id_for_name("Will Smith"))

    def test_active_wins_over_inactive_namesake(self):
        # The inactive Jose Ramirez (000001) is dropped at load, so only the active
        # 608070 row remains; the accented odds spelling still folds to that match.
        self.assertEqual(player_id_map.mlb_id_for_name("José Ramírez"),
                         "608070")

    def test_two_way_player_same_id_resolves(self):
        # Ohtani appears twice (DH + P) with the SAME MLBID → one distinct id → ok.
        self.assertEqual(player_id_map.mlb_id_for_name("Shohei Ohtani"), "660271")

    def test_blank_mlb_id_is_none_but_espn_present(self):
        # Prospect Guy is active with a blank MLBID → stored (name/espn resolve),
        # but mlb lookups return None.
        self.assertIsNone(player_id_map.mlb_id_for_name("Prospect Guy"))
        self.assertEqual(player_id_map.espn_id_for_name("Prospect Guy"), "88888")

    def test_inactive_players_are_not_stored(self):
        # ACTIVE=N rows (retired) are dropped at load: noise for a live-slate
        # bettor, and SFBB duplicates some retired IDPLAYERs (the crash guarded
        # against below). Retired Guy has a real MLBID yet must not resolve.
        self.assertIsNone(player_id_map.mlb_id_for_name("Retired Guy"))
        self.assertIsNone(player_id_map.get_row("Retired Guy"))

    def test_inactive_duplicate_idplayer_does_not_crash_load(self):
        # The real 'burneaj01' case: two rows share one IDPLAYER and are BOTH
        # inactive. The active filter drops them before they can collide on
        # uq_player_id_map — _load in setUp would have raised on write otherwise.
        self.assertIsNone(player_id_map.mlb_id_for_name("Burn Dupe"))

    def test_active_duplicate_idplayer_deduped_preferring_mlb_id(self):
        # Two ACTIVE rows share one IDPLAYER; the write anchors UNIQUE on it, so the
        # loader collapses them to one, keeping the MLBID-bearing row so the pipeline
        # key survives. Regressing the dedup would raise a UNIQUE violation on write.
        self.assertEqual(player_id_map.mlb_id_for_name("Twin Guy"), "777001")

    def test_espn_mlb_bridge_round_trip(self):
        self.assertEqual(player_id_map.espn_id_for_mlb_id("545361"), "30836")
        self.assertEqual(player_id_map.mlb_id_for_espn_id("30836"), "545361")

    def test_get_row_returns_full_record(self):
        row = player_id_map.get_row("Mike Trout")
        self.assertEqual(row["dk_name"], "Mike Trout")
        self.assertEqual(row["team"], "LAA")
        self.assertTrue(row["active"])

    def test_unknown_name_is_none(self):
        self.assertIsNone(player_id_map.mlb_id_for_name("Nobody Here"))
        self.assertIsNone(player_id_map.get_row("Nobody Here"))


class TeamLookupTests(_Backend, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._load()

    def test_abbr_resolves(self):
        self.assertEqual(player_id_map.team_code_for_abbr("ari"), "ARI")
        self.assertEqual(player_id_map.team_code_for_name("ARI"), "ARI")

    def test_espn_abbr_maps_to_sfbb_code(self):
        # ESPN uses WSH; the canonical SFBB code is WAS.
        self.assertEqual(player_id_map.team_code_for_name("WSH"), "WAS")

    def test_full_name_and_nickname(self):
        self.assertEqual(player_id_map.team_code_for_name("Arizona Diamondbacks"),
                         "ARI")
        self.assertEqual(player_id_map.team_code_for_name("Diamondbacks"), "ARI")

    def test_cle_guardians_override(self):
        # Stale nickname "Indians" still resolves; the current "Guardians" too.
        self.assertEqual(player_id_map.team_code_for_name("Indians"), "CLE")
        self.assertEqual(player_id_map.team_code_for_name("Cleveland Guardians"),
                         "CLE")

    def test_athletics_relocated(self):
        self.assertEqual(player_id_map.team_code_for_name("Athletics"), "ATH")
        self.assertEqual(player_id_map.team_code_for_name("Oakland Athletics"),
                         "ATH")

    def test_nba_team_falls_through(self):
        self.assertIsNone(player_id_map.team_code_for_name("Los Angeles Lakers"))
        self.assertIsNone(player_id_map.team_code_for_abbr("LAL"))


class FailOpenTests(_Backend, unittest.TestCase):

    def test_sql_off_returns_none(self):
        self._load()
        db_store.configure_engine(None)       # SQL off
        self._reset_module_state()
        self.assertIsNone(player_id_map.mlb_id_for_name("Mike Trout"))
        self.assertIsNone(player_id_map.team_code_for_name("Diamondbacks"))
        self.assertEqual(player_id_map.get_row("Mike Trout"), None)


class LazyRefreshTests(_Backend, unittest.TestCase):

    def _age_meta(self, map_name, hours):
        engine = db_store.get_engine()
        with engine.begin() as conn:
            conn.execute(
                player_id_map.id_map_meta.update()
                .where(player_id_map.id_map_meta.c.map_name == map_name)
                .values(last_fetched_at=player_id_map._now() - hours * 3600))

    def test_stale_map_is_refreshed_on_lookup(self):
        self._load()
        self._age_meta("player", 100)         # past the 24h TTL
        self._reset_module_state()
        updated = _PLAYERS_CSV.replace("545361", "999999")  # Trout's MLBID changes
        with patch.object(player_id_map.requests, "get",
                          side_effect=lambda *a, **k: _FakeResp(updated)):
            self.assertEqual(player_id_map.mlb_id_for_name("Mike Trout"), "999999")

    def test_refresh_error_serves_stale(self):
        self._load()
        self._age_meta("player", 100)
        self._reset_module_state()
        with patch.object(player_id_map.requests, "get",
                          side_effect=ConnectionError("SFBB down")):
            # Refresh fails → the stale SQL rows are still served.
            self.assertEqual(player_id_map.mlb_id_for_name("Mike Trout"), "545361")

    def test_fresh_map_does_not_refetch(self):
        self._load()                          # meta is fresh (just written)
        self._reset_module_state()
        with patch.object(player_id_map.requests, "get",
                          side_effect=AssertionError("must not fetch when fresh")):
            self.assertEqual(player_id_map.mlb_id_for_name("Mike Trout"), "545361")


if __name__ == "__main__":
    unittest.main()
