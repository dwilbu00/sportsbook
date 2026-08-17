"""Tests for the game_pk team-market grading fast path (Tier A #2).

Commit 1 covers the two new primitives (mlb_warehouse.final_game_by_pk +
game_results.grade_team_bet_by_game_pk / GRADE_PENDING). All hermetic — no SQL, no
network: final_game_by_pk is exercised by patching get_game (snake_case rows), the
grader by patching final_game_by_pk + team_id_for_name_tolerant.
"""
import unittest
from unittest.mock import patch

from sqlalchemy import insert, select

import db_store
import game_results as gr
import mlb_warehouse as mw


class FinalGameByPkTests(unittest.TestCase):
    def _game(self, **kw):
        base = {"game_pk": 1, "status": "Final", "detailed_state": "Final",
                "home_score": 5.0, "away_score": 3.0,
                "home_team_id": "111", "away_team_id": "222",
                "game_date": "2024-04-01T23:05:00Z"}
        base.update(kw)
        return base

    def test_final(self):
        with patch.object(mw, "get_game", return_value=self._game()):
            fg = mw.final_game_by_pk(1)
        self.assertEqual(fg["state"], "final")
        self.assertEqual(fg["home_score"], 5.0)
        self.assertEqual(fg["home_team_id"], "111")
        self.assertEqual(fg["commence_time"], "2024-04-01T23:05:00Z")

    def test_legit_zero_zero_is_final(self):
        # A real 0-0 must grade — the check is `is None`, not falsy.
        with patch.object(mw, "get_game",
                          return_value=self._game(home_score=0.0, away_score=0.0)):
            self.assertEqual(mw.final_game_by_pk(1)["state"], "final")

    def test_terminal_postponed(self):
        # status can read 'Final' with a postponed detailed_state — snake_case; must
        # NOT pass as final (the _is_genuine_final camelCase trap).
        with patch.object(mw, "get_game",
                          return_value=self._game(detailed_state="Postponed")):
            self.assertEqual(mw.final_game_by_pk(1)["state"], "terminal")

    def test_terminal_suspended_and_cancelled(self):
        for ds in ("Suspended: Rain", "Cancelled"):
            with patch.object(mw, "get_game",
                              return_value=self._game(detailed_state=ds)):
                self.assertEqual(mw.final_game_by_pk(1)["state"], "terminal")

    def test_live_when_in_progress(self):
        with patch.object(mw, "get_game",
                          return_value=self._game(status="In Progress",
                                                  detailed_state="In Progress",
                                                  home_score=None, away_score=None)):
            self.assertEqual(mw.final_game_by_pk(1)["state"], "live")

    def test_final_status_but_score_missing_is_live(self):
        with patch.object(mw, "get_game",
                          return_value=self._game(home_score=None)):
            self.assertEqual(mw.final_game_by_pk(1)["state"], "live")

    def test_none_when_no_row(self):
        with patch.object(mw, "get_game", return_value=None):
            self.assertIsNone(mw.final_game_by_pk(999))


class GradeByGamePkTests(unittest.TestCase):
    _FINAL = {"state": "final", "home_score": 5.0, "away_score": 3.0,
              "home_team_id": "111", "away_team_id": "222",
              "commence_time": "2024-04-01T23:05:00Z"}

    def test_non_mlb_returns_none(self):
        self.assertIsNone(gr.grade_team_bet_by_game_pk(
            "basketball_nba", 1, "moneyline", "home", "X", None))

    def test_no_game_pk_returns_none(self):
        self.assertIsNone(gr.grade_team_bet_by_game_pk(
            "baseball_mlb", None, "moneyline", "home", "X", None))

    def test_total_is_orientation_invariant(self):
        # 5+3=8 over 7.5 -> won; graded without any team_id resolution.
        with patch.object(mw, "final_game_by_pk", return_value=self._FINAL):
            self.assertEqual(
                gr.grade_team_bet_by_game_pk("baseball_mlb", 1, "total", "over",
                                             None, 7.5),
                ("won", "5-3"))

    def test_moneyline_home_from_team_id(self):
        with patch.object(mw, "final_game_by_pk", return_value=self._FINAL), \
             patch.object(mw, "team_id_for_name_tolerant", return_value="111"):
            self.assertEqual(
                gr.grade_team_bet_by_game_pk("baseball_mlb", 1, "moneyline", None,
                                             "Home Team", None),
                ("won", "5-3"))            # home won 5-3

    def test_moneyline_away_from_team_id(self):
        with patch.object(mw, "final_game_by_pk", return_value=self._FINAL), \
             patch.object(mw, "team_id_for_name_tolerant", return_value="222"):
            self.assertEqual(
                gr.grade_team_bet_by_game_pk("baseball_mlb", 1, "moneyline", None,
                                             "Away Team", None),
                ("lost", "5-3"))           # away lost

    def test_unmappable_team_falls_back(self):
        with patch.object(mw, "final_game_by_pk", return_value=self._FINAL), \
             patch.object(mw, "team_id_for_name_tolerant", return_value=None):
            self.assertIsNone(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "moneyline", None, "???", None))

    def test_live_fresh_returns_pending(self):
        fresh = dict(self._FINAL, state="live",
                     commence_time="2099-01-01T00:00:00Z")   # future -> not stale
        with patch.object(mw, "final_game_by_pk", return_value=fresh):
            self.assertIs(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "moneyline", "home", "X", None),
                gr.GRADE_PENDING)

    def test_live_but_stale_falls_back(self):
        stale = dict(self._FINAL, state="live",
                     commence_time="2020-01-01T00:00:00Z")   # long past -> stale
        with patch.object(mw, "final_game_by_pk", return_value=stale):
            self.assertIsNone(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "moneyline", "home", "X", None))

    def test_terminal_falls_back(self):
        with patch.object(mw, "final_game_by_pk",
                          return_value=dict(self._FINAL, state="terminal")):
            self.assertIsNone(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "moneyline", "home", "X", None))

    def test_no_warehouse_row_falls_back(self):
        with patch.object(mw, "final_game_by_pk", return_value=None):
            self.assertIsNone(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "total", "over", None, 7.5))

    def test_distinct_pks_grade_distinct_scores(self):
        # DH exactness: the score comes from the resolved game, not name+date.
        g1 = dict(self._FINAL, home_score=5.0, away_score=3.0)
        g2 = dict(self._FINAL, home_score=1.0, away_score=2.0)
        with patch.object(mw, "final_game_by_pk", side_effect=lambda pk: g1 if pk == 1 else g2):
            self.assertEqual(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 1, "total", "over", None, 7.5), ("won", "5-3"))
            self.assertEqual(gr.grade_team_bet_by_game_pk(
                "baseball_mlb", 2, "total", "over", None, 7.5), ("lost", "1-2"))


class GradeWagerRoutingTests(unittest.TestCase):
    """wagers._grade_wager routing: fast path vs the unchanged name+date fallback."""
    def _row(self, **kw):
        base = {"sport_key": "baseball_mlb", "bet_type": "moneyline", "game_pk": 1,
                "side": "home", "team": "NYY", "point": None,
                "commence_time": "2024-04-01T23:05:00Z", "game_date": "2024-04-01"}
        base.update(kw)
        return base

    def test_fast_result_short_circuits_name_date(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk",
                          return_value=("won", "5-3")), \
             patch.object(gr, "final_score") as fsm:
            self.assertEqual(wagers._grade_wager(self._row()), ("won", "5-3"))
            fsm.assert_not_called()          # game_pk path won -> never hit name+date

    def test_pending_stays_pending_no_fallback(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk",
                          return_value=gr.GRADE_PENDING), \
             patch.object(gr, "final_score") as fsm:
            self.assertIsNone(wagers._grade_wager(self._row()))
            fsm.assert_not_called()          # critical: do NOT fall back on PENDING

    def test_fast_none_falls_back_to_name_date(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk", return_value=None), \
             patch.object(gr, "final_score", return_value=(5.0, 3.0)) as fsm, \
             patch.object(gr, "side_for_team", return_value="home"):
            self.assertEqual(wagers._grade_wager(self._row()), ("won", "5-3"))
            fsm.assert_called_once()          # fell through to name+date

    def test_absent_game_pk_skips_fast_path(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk") as fastm, \
             patch.object(gr, "final_score", return_value=(5.0, 3.0)), \
             patch.object(gr, "side_for_team", return_value="home"):
            self.assertEqual(wagers._grade_wager(self._row(game_pk=None)),
                             ("won", "5-3"))
            fastm.assert_not_called()          # byte-identical to pre-#2

    def test_non_mlb_skips_fast_path(self):
        import wagers
        with patch.object(gr, "grade_team_bet_by_game_pk") as fastm, \
             patch.object(gr, "final_score", return_value=(110.0, 100.0)), \
             patch.object(gr, "side_for_team", return_value="home"):
            wagers._grade_wager(self._row(sport_key="basketball_nba",
                                          bet_type="moneyline"))
            fastm.assert_not_called()          # NBA never enters the fast path


class WagerEnrichStampTests(unittest.TestCase):
    """wagers._enrich_ids stamps a DH-safe game_pk on MLB team wagers (Tier A #2)."""
    def _row(self, **kw):
        base = {"sport_key": "baseball_mlb", "bet_type": "moneyline",
                "home_team": "NYY", "away_team": "BOS", "team": "NYY",
                "commence_time": "2024-04-01T23:05:00Z"}
        base.update(kw)
        return base

    def test_team_wager_gets_game_pk(self):
        import wagers
        with patch.object(mw, "team_id_for_name_tolerant", side_effect=["111", "222"]), \
             patch.object(mw, "find_game_pk_by_commence", return_value=777) as f:
            row = wagers._enrich_ids(self._row())
        self.assertEqual(row.get("game_pk"), 777)
        f.assert_called_once()

    def test_ambiguous_dh_leaves_game_pk_none(self):
        import wagers
        with patch.object(mw, "team_id_for_name_tolerant", side_effect=["111", "222"]), \
             patch.object(mw, "find_game_pk_by_commence", return_value=None):
            self.assertIsNone(wagers._enrich_ids(self._row()).get("game_pk"))

    def test_missing_commence_not_stamped(self):
        import wagers
        with patch.object(mw, "team_id_for_name_tolerant", side_effect=["111", "222"]), \
             patch.object(mw, "find_game_pk_by_commence") as f:
            self.assertIsNone(
                wagers._enrich_ids(self._row(commence_time=None)).get("game_pk"))
        f.assert_not_called()

    def test_non_baseball_untouched(self):
        import wagers
        with patch.object(mw, "find_game_pk_by_commence") as f:
            self.assertIsNone(
                wagers._enrich_ids(self._row(sport_key="basketball_nba")).get("game_pk"))
        f.assert_not_called()


class WarehouseEnrichStampTests(unittest.TestCase):
    """warehouse._enrich_ids stamps one shared game_pk on all team odds lines."""
    def test_team_lines_share_one_game_pk(self):
        import warehouse
        meta = {"home": "NYY", "away": "BOS", "game_date": "2024-04-01",
                "commence_time": "2024-04-01T23:05:00Z"}
        lines = [{"bet_type": "moneyline", "selection": "NYY"},
                 {"bet_type": "total", "selection": "Over"},
                 {"bet_type": "player_prop", "player": "Aaron Judge",
                  "prop_key": "batter_hits"}]
        with patch.object(mw, "team_id_for_name_tolerant", side_effect=["111", "222"]), \
             patch.object(mw, "find_game_pk_by_commence", return_value=777), \
             patch("entity_resolver.resolve",
                   return_value={"mlb_player_id": "5", "game_pk": 777}), \
             patch("mlb_starters.warm_player_index"):
            _meta, out = warehouse._enrich_ids("baseball_mlb", meta, lines)
        self.assertEqual(out[0]["game_pk"], 777)     # moneyline (team branch)
        self.assertEqual(out[1]["game_pk"], 777)     # total (team branch)
        self.assertEqual(out[2]["game_pk"], 777)     # prop (entity_resolver branch)

    def test_unresolved_team_leaves_lines_null(self):
        import warehouse
        meta = {"home": "NYY", "away": "BOS", "game_date": "2024-04-01",
                "commence_time": "2024-04-01T23:05:00Z"}
        lines = [{"bet_type": "spread", "selection": "NYY"}]
        with patch.object(mw, "team_id_for_name_tolerant", return_value=None), \
             patch.object(mw, "find_game_pk_by_commence") as f, \
             patch("mlb_starters.warm_player_index"):
            _meta, out = warehouse._enrich_ids("baseball_mlb", meta, lines)
        self.assertIsNone(out[0].get("game_pk"))
        f.assert_not_called()                        # both team ids None -> no resolve


class BackfillTeamGamePkTests(unittest.TestCase):
    """Team-anchor game_pk backfill (Tier A #2) over wagers + odds_line: DH-safe,
    non-destructive (fill-NULL-only), idempotent, dry-run vs apply. team_id_for_name_
    tolerant is patched (name->id is resolver-tested elsewhere); find_game_pk_by_commence
    / find_game_pk run for real against seeded mlb_game so DH resolution is end-to-end."""
    _NAMES = {"Yankees": "147", "Red Sox": "111"}

    def setUp(self):
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        mw.create_all()
        mw._TEAMS_ENSURED.clear()

    def tearDown(self):
        db_store.configure_engine(None)
        mw._TEAMS_ENSURED.clear()

    def _game(self, pk, official_date, game_date, home="147", away="111"):
        with db_store.get_engine().begin() as conn:
            for tid in (home, away):
                if not conn.execute(select(mw.mlb_team).where(
                        mw.mlb_team.c.team_id == tid)).first():
                    conn.execute(insert(mw.mlb_team), {"team_id": tid, "name": tid})
            conn.execute(insert(mw.mlb_game), {
                "game_pk": pk, "official_date": official_date, "game_date": game_date,
                "season": 2026, "home_team_id": home, "away_team_id": away})

    def _wager(self, wid, **kw):
        row = {"wager_id": wid, "sport_key": "baseball_mlb", "bet_type": "moneyline",
               "home_team": "Yankees", "away_team": "Red Sox", "team": "Yankees",
               "player": None, "game_pk": None, "game_date": "2026-08-01",
               "commence_time": "2026-08-01T18:00:00Z"}
        row.update(kw)
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(db_store.wagers), row)

    def _snapshot(self, event_id, commence, lines, game_date="2026-08-01"):
        # Every line dict must carry the SAME keys (executemany compiles from the
        # first row) — full-key with None defaults.
        cols = ("bet_type", "selection", "point", "player", "prop_key")
        with db_store.get_engine().begin() as conn:
            sid = conn.execute(insert(db_store.odds_snapshot), {
                "sport": "baseball_mlb", "game_date": game_date, "event_id": event_id,
                "kind": "team", "snapshot_hour": event_id, "commence_time": commence,
                "home": "Yankees", "away": "Red Sox"}).inserted_primary_key[0]
            conn.execute(insert(db_store.odds_line),
                         [{**{c: None for c in cols}, "snapshot_id": sid, **ln}
                          for ln in lines])

    def _wager_gpks(self):
        with db_store.get_engine().connect() as conn:
            return {r._mapping["wager_id"]: r._mapping["game_pk"]
                    for r in conn.execute(select(db_store.wagers))}

    def _oline_gpks(self):
        with db_store.get_engine().connect() as conn:
            return {r._mapping["bet_type"]: r._mapping["game_pk"]
                    for r in conn.execute(select(db_store.odds_line))}

    def _seed(self):
        self._game(700, "2026-08-01", "2026-08-01T18:00:00Z")
        self._game(800, "2026-08-02", "2026-08-02T17:00:00Z")
        self._game(801, "2026-08-02", "2026-08-02T21:00:00Z")   # split DH
        self._game(810, "2026-08-04", "2026-08-04T18:00:00Z")
        self._game(811, "2026-08-04", "2026-08-04T18:00:00Z")   # same-timestamp DH
        self._wager("w-single")                                 # -> 700
        self._wager("w-dh", game_date="2026-08-02",
                    commence_time="2026-08-02T20:55:00Z")       # -> 801 (nearest)
        self._wager("w-tie", game_date="2026-08-04",
                    commence_time="2026-08-04T18:00:00Z")       # -> None (DH tie)
        self._wager("w-stamped", game_pk=999)                   # already set
        self._wager("w-nba", sport_key="basketball_nba")        # non-MLB
        self._wager("w-prop", player="Aaron Judge")             # player row (excluded)
        self._snapshot("evt-1", "2026-08-01T18:00:00Z", [
            {"bet_type": "moneyline", "selection": "Yankees"},
            {"bet_type": "total", "selection": "Over", "point": 8.5},
            {"bet_type": "player_prop", "player": "Aaron Judge",
             "prop_key": "batter_hits"}])

    def _run(self, dry_run):
        with patch.object(mw, "team_id_for_name_tolerant",
                          side_effect=lambda nm: self._NAMES.get(nm)):
            return mw.backfill_team_game_pk(dry_run=dry_run)

    def test_dry_run_reports_but_does_not_write(self):
        self._seed()
        s = self._run(dry_run=True)
        self.assertEqual(s["wagers"]["candidates"], 3)          # single, dh, tie
        self.assertEqual(s["wagers"]["matched"], 2)            # single + dh
        self.assertEqual(s["odds_line"]["candidates"], 2)      # ml + total (not prop)
        self.assertEqual(s["odds_line"]["matched"], 2)
        self.assertIsNone(self._wager_gpks()["w-single"])      # nothing written
        self.assertTrue(all(v is None for v in self._oline_gpks().values()))

    def test_apply_fills_dh_safe_and_nondestructive(self):
        self._seed()
        self._run(dry_run=False)
        w = self._wager_gpks()
        self.assertEqual(w["w-single"], 700)
        self.assertEqual(w["w-dh"], 801)                       # nearest commence
        self.assertIsNone(w["w-tie"])                          # same-timestamp DH -> NULL
        self.assertEqual(w["w-stamped"], 999)                  # not overwritten
        self.assertIsNone(w["w-nba"])                          # non-MLB untouched
        self.assertIsNone(w["w-prop"])                         # player row untouched
        ol = self._oline_gpks()
        self.assertEqual(ol["moneyline"], 700)
        self.assertEqual(ol["total"], 700)
        self.assertIsNone(ol["player_prop"])                  # prop line untouched

    def test_idempotent(self):
        self._seed()
        self._run(dry_run=False)
        again = self._run(dry_run=False)
        self.assertEqual(again["wagers"]["matched"], 0)
        self.assertEqual(again["odds_line"]["matched"], 0)

    def test_naive_commence_unique_date_resolves_via_fallback(self):
        self._game(700, "2026-08-01", "2026-08-01T18:00:00Z")
        self._wager("w-naive", commence_time="2026-08-01T18:00:00")   # no Z (tz-naive)
        self._run(dry_run=False)                               # must not raise
        self.assertEqual(self._wager_gpks()["w-naive"], 700)  # find_game_pk fallback

    def test_naive_commence_dh_left_null_no_crash(self):
        self._game(810, "2026-08-04", "2026-08-04T17:00:00Z")
        self._game(811, "2026-08-04", "2026-08-04T21:00:00Z")
        self._wager("w-dhnaive", game_date="2026-08-04",
                    commence_time="2026-08-04T20:55:00")       # no Z (tz-naive)
        self._run(dry_run=False)                               # must not raise
        self.assertIsNone(self._wager_gpks()["w-dhnaive"])    # DH + naive -> NULL


if __name__ == "__main__":
    unittest.main()
