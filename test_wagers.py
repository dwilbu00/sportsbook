"""Tests for the actual-bets ledger (Submit Picks -> realized ROI).

Hermetic: no live statsapi / ESPN / Azure. Row construction from analysis
candidates, team-market grading math, the profit formula, the NDJSON append +
grade round-trip (local tempdir store), and the stake-weighted ROI summary are
all exercised on fixtures with the resolvers mocked.
"""
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import game_results
import pricing_common
import recalibration
import wagers

_META = {
    "sport_key": "baseball_mlb",
    "event_id": "E1",
    "commence_time": "2026-07-16T18:00:00Z",
    "game_date": "2026-07-16",
    "home_team": "Boston Red Sox",
    "away_team": "New York Yankees",
    "stake": 10.0,
    "placed_at": "2026-07-16T12:00:00+00:00",
    "seq": 0,
}


def _meta(seq=0):
    m = dict(_META)
    m["seq"] = seq
    return m


class BuildWagerRowTests(unittest.TestCase):
    def test_moneyline_row(self):
        cand = {"team": "Boston Red Sox", "opponent": "New York Yankees",
                "home_away": "HOME", "best_price": 120, "best_book": "DK",
                "blended_prob": 58.0, "best_edge_pct": 6.0, "event_id": "E1"}
        row = wagers.build_wager_row("moneyline", None, cand, _meta())
        self.assertEqual(row["bet_type"], "moneyline")
        self.assertEqual(row["side"], "home")
        self.assertEqual(row["executed_price"], 120)
        self.assertEqual(row["model_price"], 120)
        self.assertEqual(row["stake"], 10.0)
        self.assertAlmostEqual(row["model_prob"], 0.58)
        self.assertEqual(row["game_date"], "2026-07-16")
        self.assertEqual(row["home_team"], "Boston Red Sox")
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["wager_id"], "2026-07-16T12:00:00+00:00#0")

    def test_spread_row(self):
        cand = {"team": "Boston Red Sox", "opponent": "New York Yankees",
                "home_away": "AWAY", "spread": -1.5, "price": -110,
                "cover_rate": 55.0, "edge_pct": 4.0, "event_id": "E1"}
        row = wagers.build_wager_row("spread", None, cand, _meta())
        self.assertEqual(row["side"], "away")
        self.assertEqual(row["point"], -1.5)
        self.assertEqual(row["executed_price"], -110)
        self.assertAlmostEqual(row["model_prob"], 0.55)

    def test_total_over_and_under_rows(self):
        cand = {"matchup": "New York Yankees @ Boston Red Sox", "line": 8.5,
                "over_price": -105, "under_price": -115, "over_hit_rate": 57.0,
                "over_edge_pct": 5.0, "under_edge_pct": -2.0, "event_id": "E1"}
        over = wagers.build_wager_row("total", "OVER", cand, _meta())
        self.assertEqual(over["side"], "over")
        self.assertEqual(over["point"], 8.5)
        self.assertEqual(over["executed_price"], -105)
        self.assertAlmostEqual(over["model_prob"], 0.57)
        under = wagers.build_wager_row("total", "UNDER", cand, _meta(1))
        self.assertEqual(under["side"], "under")
        self.assertEqual(under["executed_price"], -115)
        self.assertAlmostEqual(under["model_prob"], 0.43)

    def test_player_prop_row(self):
        cand = {"player": "Rafael Devers", "prop": "batter_hits",
                "prop_label": "Hits", "line": 1.5, "direction": "OVER",
                "over_price": -110, "under_price": -110, "over_rate": 60.0,
                "edge_pct": 7.0, "matchup": "NYY @ BOS",
                "team": "Boston Red Sox", "event_id": "E1"}
        row = wagers.build_wager_row("player_prop", None, cand, _meta())
        self.assertEqual(row["player"], "Rafael Devers")
        self.assertEqual(row["prop_key"], "batter_hits")
        self.assertEqual(row["line"], 1.5)
        self.assertEqual(row["executed_price"], -110)
        self.assertEqual(row["direction"], "OVER")
        self.assertAlmostEqual(row["model_prob"], 0.60)

    def test_player_prop_stakes_at_dk_price_when_present(self):
        # P1.1b: over_price is the best-across-books price (value/EV); the
        # ledger must record the DraftKings price the user actually bets.
        cand = {"player": "Rafael Devers", "prop": "batter_hits",
                "prop_label": "Hits", "line": 1.5, "direction": "OVER",
                "over_price": 150, "under_price": -110,
                "dk_over_price": 100, "dk_under_price": -120,
                "over_rate": 60.0, "edge_pct": 7.0, "matchup": "NYY @ BOS",
                "team": "Boston Red Sox", "event_id": "E1"}
        row = wagers.build_wager_row("player_prop", None, cand, _meta())
        self.assertEqual(row["executed_price"], 100)  # DK, not the +150 best

    def test_player_prop_falls_back_to_best_when_dk_absent(self):
        cand = {"player": "Rafael Devers", "prop": "batter_hits",
                "prop_label": "Hits", "line": 1.5, "direction": "OVER",
                "over_price": 150, "under_price": -110,
                "dk_over_price": None, "dk_under_price": None,
                "over_rate": 60.0, "edge_pct": 7.0, "matchup": "NYY @ BOS",
                "team": "Boston Red Sox", "event_id": "E1"}
        row = wagers.build_wager_row("player_prop", None, cand, _meta())
        self.assertEqual(row["executed_price"], 150)  # falls back to best

    def test_safe_mode_prop_uses_alt_line_and_price(self):
        cand = {"player": "Aaron Judge", "prop": "batter_hits",
                "prop_label": "Hits", "safe_mode": True, "safe_alt_line": 0.5,
                "safe_alt_price": -200, "model_hit_at_safe": 80.0,
                "direction": "OVER", "edge_pct": 5.0, "matchup": "NYY @ BOS",
                "team": "New York Yankees", "event_id": "E1"}
        row = wagers.build_wager_row("player_prop", None, cand, _meta())
        self.assertEqual(row["line"], 0.5)
        self.assertEqual(row["executed_price"], -200)
        self.assertEqual(row["direction"], "OVER")
        self.assertAlmostEqual(row["model_prob"], 0.80)

    def test_blank_row_derives_et_local_game_date(self):
        # 02:30 UTC on 7/21 is 10:30 PM ET on 7/20 -> official game date is 7/20,
        # NOT the raw UTC 7/21. Meta omits game_date so the fallback is exercised.
        meta = {"sport_key": "baseball_mlb", "event_id": "E9",
                "commence_time": "2026-07-21T02:30:00Z",
                "home_team": "H", "away_team": "A", "stake": 10.0,
                "placed_at": "2026-07-20T20:00:00+00:00", "seq": 0}
        cand = {"team": "H", "opponent": "A", "home_away": "HOME",
                "best_price": 120, "event_id": "E9"}
        row = wagers.build_wager_row("moneyline", None, cand, meta)
        self.assertEqual(row["game_date"], "2026-07-20")


class GradeTeamBetTests(unittest.TestCase):
    def test_moneyline(self):
        self.assertEqual(
            game_results.grade_team_bet("moneyline", "home", None, 5, 3), "won")
        self.assertEqual(
            game_results.grade_team_bet("moneyline", "away", None, 5, 3), "lost")

    def test_spread(self):
        # Home -1.5, wins by 2 -> covers.
        self.assertEqual(
            game_results.grade_team_bet("spread", "home", -1.5, 5, 3), "won")
        # Home -2.0, wins by exactly 2 -> push.
        self.assertEqual(
            game_results.grade_team_bet("spread", "home", -2.0, 5, 3), "push")
        # Away +1.5, loses by 2 -> does not cover.
        self.assertEqual(
            game_results.grade_team_bet("spread", "away", 1.5, 5, 3), "lost")

    def test_total(self):
        self.assertEqual(
            game_results.grade_team_bet("total", "over", 7.5, 5, 3), "won")
        self.assertEqual(
            game_results.grade_team_bet("total", "under", 8.5, 5, 3), "won")
        self.assertEqual(
            game_results.grade_team_bet("total", "over", 8.0, 5, 3), "push")

    def test_bad_inputs_return_none(self):
        self.assertIsNone(
            game_results.grade_team_bet("total", "over", None, 5, 3))
        self.assertIsNone(
            game_results.grade_team_bet("spread", "home", -1.5, None, 3))


class ProfitTests(unittest.TestCase):
    def test_win_loss_push(self):
        self.assertAlmostEqual(pricing_common.profit(120, 10, True), 12.0)
        self.assertAlmostEqual(pricing_common.profit(-110, 10, False), -10.0)
        self.assertAlmostEqual(pricing_common.profit(-110, 10, None), 0.0)

    def test_negative_price_win(self):
        self.assertAlmostEqual(
            pricing_common.profit(-200, 10, True), 5.0)


class _LocalLedger:
    """Context manager: force wagers/recalibration onto a tempdir NDJSON store."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()

    def __enter__(self):
        self._p1 = patch.object(recalibration, "PRED_DIR", self._tmp.name)
        self._p2 = patch.object(recalibration, "_prediction_log_blob_url",
                                return_value="")
        self._p1.start(); self._p2.start()
        return self

    def __exit__(self, *exc):
        self._p1.stop(); self._p2.stop()
        self._tmp.cleanup()


class RoundTripAndGradeTests(unittest.TestCase):
    def test_submit_read_and_resolve(self):
        prop = wagers.build_wager_row("player_prop", None, {
            "player": "Rafael Devers", "prop": "batter_hits",
            "prop_label": "Hits", "line": 1.5, "direction": "OVER",
            "over_price": -110, "over_rate": 60.0, "edge_pct": 7.0,
            "matchup": "NYY @ BOS", "team": "Boston Red Sox", "event_id": "E1"},
            _meta(0))
        ml = wagers.build_wager_row("moneyline", None, {
            "team": "Boston Red Sox", "opponent": "New York Yankees",
            "home_away": "HOME", "best_price": 120, "best_book": "DK",
            "blended_prob": 58.0, "best_edge_pct": 6.0, "event_id": "E1"},
            _meta(1))

        with _LocalLedger():
            self.assertEqual(wagers.submit_wagers([prop, ml]), 2)
            # Idempotent: re-submitting the same wager_ids adds nothing.
            self.assertEqual(wagers.submit_wagers([prop, ml]), 0)
            self.assertEqual(len(wagers.read_wagers()), 2)

            now = datetime(2026, 7, 20, tzinfo=timezone.utc)
            with patch.object(recalibration, "resolve_one_prop",
                              return_value=2.0), \
                 patch.object(game_results, "final_score", return_value=(5, 3)):
                graded = wagers.resolve_pending_wagers(now=now)
            self.assertEqual(graded, 2)

            rows = {r["bet_type"]: r for r in wagers.read_wagers()}
            self.assertEqual(rows["player_prop"]["status"], "won")
            self.assertAlmostEqual(rows["player_prop"]["profit"],
                                   pricing_common.profit(-110, 10, True))
            self.assertEqual(rows["moneyline"]["status"], "won")
            self.assertAlmostEqual(rows["moneyline"]["profit"], 12.0)

    def test_pending_future_game_not_graded(self):
        row = wagers.build_wager_row("moneyline", None, {
            "team": "Boston Red Sox", "opponent": "New York Yankees",
            "home_away": "HOME", "best_price": 120, "event_id": "E1"},
            _meta(0))
        with _LocalLedger():
            wagers.submit_wagers([row])
            # "now" is before the game date -> nothing gradable yet.
            now = datetime(2026, 7, 15, tzinfo=timezone.utc)
            with patch.object(game_results, "final_score", return_value=(5, 3)):
                self.assertEqual(wagers.resolve_pending_wagers(now=now), 0)
            self.assertEqual(wagers.read_wagers()[0]["status"], "pending")

    def _live_prop(self):
        return wagers.build_wager_row("player_prop", None, {
            "player": "Live Bat", "prop": "batter_hits", "prop_label": "Hits",
            "line": 1.5, "direction": "OVER", "over_price": -110,
            "over_rate": 60.0, "edge_pct": 7.0, "matchup": "A @ H",
            "team": "H", "event_id": "E1"},
            {"sport_key": "baseball_mlb", "event_id": "E1",
             "commence_time": "2026-07-21T23:10:00Z", "home_team": "H",
             "away_team": "A", "stake": 10.0,
             "placed_at": "2026-07-21T20:00:00+00:00", "seq": 0})

    def test_in_progress_game_not_attempted_within_buffer(self):
        # Bug 1: 30 min after first pitch the game is clearly live -> the
        # commence+buffer pre-filter must skip it before any resolver fetch.
        with _LocalLedger():
            wagers.submit_wagers([self._live_prop()])
            now = datetime(2026, 7, 21, 23, 40, tzinfo=timezone.utc)
            with patch.object(recalibration, "resolve_one_prop") as rp:
                self.assertEqual(wagers.resolve_pending_wagers(now=now), 0)
                rp.assert_not_called()
            self.assertEqual(wagers.read_wagers()[0]["status"], "pending")

    def test_unresolvable_live_game_stays_pending(self):
        # Bug 1: past the buffer but the resolver can't confirm a final stat
        # (returns None, e.g. an extra-innings game still going) -> stay pending,
        # never grade a partial line as a loss.
        with _LocalLedger():
            wagers.submit_wagers([self._live_prop()])
            now = datetime(2026, 7, 22, 4, 0, tzinfo=timezone.utc)  # 5h later
            with patch.object(recalibration, "resolve_one_prop", return_value=None):
                self.assertEqual(wagers.resolve_pending_wagers(now=now), 0)
            self.assertEqual(wagers.read_wagers()[0]["status"], "pending")

    def test_late_us_game_with_utc_date_grades_and_heals(self):
        # Bug 2: a 7/20-night game legacy-stored with the raw UTC game_date 7/21
        # (commence 02:30Z 7/21 = 10:30 PM ET 7/20). The old UTC guard treated it
        # as "not finished" the next day forever; the new guard grades it, and
        # settling heals game_date to the correct ET-local 7/20.
        prop = wagers.build_wager_row("player_prop", None, {
            "player": "Bat", "prop": "batter_hits", "prop_label": "Hits",
            "line": 1.5, "direction": "OVER", "over_price": -110,
            "over_rate": 60.0, "edge_pct": 7.0, "matchup": "A @ H",
            "team": "H", "event_id": "E1"},
            {"sport_key": "baseball_mlb", "event_id": "E1",
             "commence_time": "2026-07-21T02:30:00Z", "game_date": "2026-07-21",
             "home_team": "H", "away_team": "A", "stake": 10.0,
             "placed_at": "2026-07-20T20:00:00+00:00", "seq": 0})
        self.assertEqual(prop["game_date"], "2026-07-21")  # legacy stored value
        with _LocalLedger():
            wagers.submit_wagers([prop])
            now = datetime(2026, 7, 21, 14, 0, tzinfo=timezone.utc)  # next morning
            with patch.object(recalibration, "resolve_one_prop", return_value=2.0):
                self.assertEqual(wagers.resolve_pending_wagers(now=now), 1)
            row = wagers.read_wagers()[0]
            self.assertEqual(row["status"], "won")
            self.assertEqual(row["game_date"], "2026-07-20")  # healed to ET-local


class DeleteAndEditTests(unittest.TestCase):
    def _seed(self):
        prop = wagers.build_wager_row("player_prop", None, {
            "player": "Rafael Devers", "prop": "batter_hits",
            "prop_label": "Hits", "line": 1.5, "direction": "OVER",
            "over_price": -110, "over_rate": 60.0, "edge_pct": 7.0,
            "matchup": "NYY @ BOS", "team": "Boston Red Sox", "event_id": "E1"},
            _meta(0))
        ml = wagers.build_wager_row("moneyline", None, {
            "team": "Boston Red Sox", "opponent": "New York Yankees",
            "home_away": "HOME", "best_price": 120, "best_book": "DK",
            "blended_prob": 58.0, "best_edge_pct": 6.0, "event_id": "E1"},
            _meta(1))
        return prop, ml

    def test_delete_by_id_is_idempotent(self):
        prop, ml = self._seed()
        with _LocalLedger():
            wagers.submit_wagers([prop, ml])
            self.assertEqual(wagers.delete_wagers([prop["wager_id"]]), 1)
            self.assertEqual([r["wager_id"] for r in wagers.read_wagers()],
                             [ml["wager_id"]])
            # Deleting an already-gone id removes nothing.
            self.assertEqual(wagers.delete_wagers([prop["wager_id"]]), 0)

    def test_delete_empty_is_noop(self):
        prop, ml = self._seed()
        with _LocalLedger():
            wagers.submit_wagers([prop, ml])
            self.assertEqual(wagers.delete_wagers([]), 0)
            self.assertEqual(len(wagers.read_wagers()), 2)

    def test_edit_price_line_stake_and_point_sync(self):
        prop, _ = self._seed()
        with _LocalLedger():
            wagers.submit_wagers([prop])
            n = wagers.update_wagers({prop["wager_id"]: {
                "executed_price": -120, "line": 2.5, "stake": 25.0}})
            self.assertEqual(n, 1)
            row = wagers.read_wagers()[0]
            self.assertEqual(row["executed_price"], -120)
            self.assertEqual(row["line"], 2.5)
            self.assertEqual(row["point"], 2.5)   # line edit syncs the point
            self.assertEqual(row["stake"], 25.0)

    def test_edit_skips_settled_rows(self):
        prop, _ = self._seed()
        with _LocalLedger():
            wagers.submit_wagers([prop])
            now = datetime(2026, 7, 20, tzinfo=timezone.utc)
            with patch.object(recalibration, "resolve_one_prop", return_value=2.0):
                wagers.resolve_pending_wagers(now=now)
            self.assertEqual(wagers.read_wagers()[0]["status"], "won")
            # A settled bet's realized fields must not be editable.
            self.assertEqual(
                wagers.update_wagers({prop["wager_id"]: {"stake": 99.0}}), 0)
            self.assertEqual(wagers.read_wagers()[0]["stake"], 10.0)

    def test_regrade_resets_settled_to_pending_then_regrades(self):
        prop, _ = self._seed()
        with _LocalLedger():
            wagers.submit_wagers([prop])
            now = datetime(2026, 7, 20, tzinfo=timezone.utc)
            # Simulate the OLD bug: graded "lost" off a live/partial 0 hits.
            with patch.object(recalibration, "resolve_one_prop", return_value=0.0):
                wagers.resolve_pending_wagers(now=now)
            self.assertEqual(wagers.read_wagers()[0]["status"], "lost")

            # Re-grade resets it to pending and clears the realized fields.
            self.assertEqual(wagers.regrade_wagers([prop["wager_id"]]), 1)
            row = wagers.read_wagers()[0]
            self.assertEqual(row["status"], "pending")
            self.assertIsNone(row["profit"])
            self.assertIsNone(row["actual"])
            self.assertIsNone(row["resolved_at"])

            # Next pass re-grades with the true final stat -> now a win.
            with patch.object(recalibration, "resolve_one_prop", return_value=2.0):
                self.assertEqual(wagers.resolve_pending_wagers(now=now), 1)
            self.assertEqual(wagers.read_wagers()[0]["status"], "won")

    def test_regrade_ignores_pending_and_empty(self):
        prop, _ = self._seed()
        with _LocalLedger():
            wagers.submit_wagers([prop])  # still pending
            self.assertEqual(wagers.regrade_wagers([prop["wager_id"]]), 0)
            self.assertEqual(wagers.regrade_wagers([]), 0)
            self.assertEqual(wagers.read_wagers()[0]["status"], "pending")


class AttachClvTests(unittest.TestCase):
    def test_clv_queries_warehouse_by_utc_commence_date(self):
        # The row's game_date is US-local (7/20) but the warehouse partitions by
        # the UTC commence date (7/21). attach_clv must query by the UTC date or
        # it misses the snapshot folder and CLV never populates.
        import warehouse
        row = {
            "bet_type": "player_prop", "sport_key": "baseball_mlb",
            "event_id": "E1", "player": "Bat", "prop_key": "batter_hits",
            "direction": "OVER", "line": 1.5, "point": 1.5,
            "game_date": "2026-07-20",
            "commence_time": "2026-07-21T02:30:00Z",
            "executed_price": -110,
        }
        seen = {}

        def fake_closing(**kwargs):
            seen.update(kwargs)
            return {"price": -120, "implied_prob": 0.545, "captured_at": "x"}

        with patch.object(warehouse, "closing_line_for", side_effect=fake_closing):
            wagers.attach_clv([row])
        self.assertEqual(seen.get("game_date"), "2026-07-21")  # UTC, not 7/20
        self.assertEqual(row["close_price"], -120)
        self.assertIsNotNone(row["clv_pct"])


class PersistClvTests(unittest.TestCase):
    def _prop(self, commence, seq=0):
        meta = {"sport_key": "baseball_mlb", "event_id": "E1",
                "commence_time": commence, "game_date": commence[:10],
                "home_team": "H", "away_team": "A", "stake": 10.0,
                "placed_at": "2026-07-20T12:00:00+00:00", "seq": seq}
        return wagers.build_wager_row("player_prop", None, {
            "player": "Bat", "prop": "batter_hits", "prop_label": "Hits",
            "line": 1.5, "direction": "OVER", "over_price": -110,
            "over_rate": 60.0, "edge_pct": 7.0, "matchup": "A @ H",
            "team": "H", "event_id": "E1"}, meta)

    def test_persists_clv_for_started_game_and_is_idempotent(self):
        import warehouse
        row = self._prop("2026-07-21T02:30:00Z")
        now = datetime(2026, 7, 21, 6, 0, tzinfo=timezone.utc)  # after commence
        with _LocalLedger():
            wagers.submit_wagers([row])
            with patch.object(warehouse, "closing_line_for",
                              return_value={"price": -120, "implied_prob": 0.545,
                                            "captured_at": "x"}):
                self.assertEqual(wagers.persist_clv(now=now), 1)
                saved = wagers.read_wagers()[0]
                self.assertEqual(saved["close_price"], -120)
                self.assertIsNotNone(saved["clv_pct"])
                # Already persisted -> nothing more to write.
                self.assertEqual(wagers.persist_clv(now=now), 0)

    def test_skips_pregame_rows(self):
        import warehouse
        row = self._prop("2026-07-21T02:30:00Z")
        now = datetime(2026, 7, 21, 1, 0, tzinfo=timezone.utc)  # before commence
        with _LocalLedger():
            wagers.submit_wagers([row])
            with patch.object(warehouse, "closing_line_for",
                              return_value={"price": -120, "implied_prob": 0.5}) as m:
                self.assertEqual(wagers.persist_clv(now=now), 0)
                m.assert_not_called()  # pre-commence -> not even queried
            self.assertIsNone(wagers.read_wagers()[0]["close_price"])

    def test_returns_zero_when_warehouse_has_no_line(self):
        import warehouse
        row = self._prop("2026-07-21T02:30:00Z")
        now = datetime(2026, 7, 21, 6, 0, tzinfo=timezone.utc)
        with _LocalLedger():
            wagers.submit_wagers([row])
            with patch.object(warehouse, "closing_line_for", return_value=None):
                self.assertEqual(wagers.persist_clv(now=now), 0)
            self.assertIsNone(wagers.read_wagers()[0]["close_price"])


class ResetClvTests(unittest.TestCase):
    def test_reset_clears_clv_fields_and_is_idempotent(self):
        with _LocalLedger():
            wagers.submit_wagers([{
                "wager_id": "w1", "status": "won", "stake": 10.0,
                "close_price": -105, "close_line": 9.5, "clv_pct": 3.2}])
            self.assertEqual(wagers.reset_clv(), 1)
            row = wagers.read_wagers()[0]
            self.assertIsNone(row["close_price"])
            self.assertIsNone(row["close_line"])
            self.assertIsNone(row["clv_pct"])
            self.assertEqual(wagers.reset_clv(), 0)  # nothing left to clear

    def test_reset_limited_to_ids(self):
        with _LocalLedger():
            wagers.submit_wagers([
                {"wager_id": "a", "status": "won", "close_price": -110,
                 "clv_pct": 1.0},
                {"wager_id": "b", "status": "won", "close_price": -120,
                 "clv_pct": 2.0}])
            self.assertEqual(wagers.reset_clv(["a"]), 1)
            rows = {r["wager_id"]: r for r in wagers.read_wagers()}
            self.assertIsNone(rows["a"]["close_price"])
            self.assertEqual(rows["b"]["close_price"], -120)


class SummaryTests(unittest.TestCase):
    def test_stake_weighted_roi(self):
        rows = [
            {"sport_key": "baseball_mlb", "bet_type": "moneyline",
             "status": "won", "stake": 10.0, "profit": 12.0},
            {"sport_key": "baseball_mlb", "bet_type": "player_prop",
             "status": "lost", "stake": 10.0, "profit": -10.0},
            {"sport_key": "baseball_mlb", "bet_type": "total",
             "status": "push", "stake": 10.0, "profit": 0.0},
            {"sport_key": "baseball_mlb", "bet_type": "moneyline",
             "status": "pending", "stake": 10.0, "profit": None},
        ]
        summary = wagers.summarize_wagers(rows)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["resolved"], 3)
        self.assertEqual(summary["pending"], 1)
        self.assertAlmostEqual(summary["total_staked"], 30.0)
        self.assertAlmostEqual(summary["realized_profit"], 2.0)
        self.assertAlmostEqual(summary["roi"], 2.0 / 30.0)
        self.assertEqual((summary["won"], summary["lost"], summary["push"]),
                         (1, 1, 1))
        self.assertAlmostEqual(summary["hit_rate"], 0.5)
        self.assertAlmostEqual(summary["pending_stake"], 10.0)
        self.assertTrue(summary["by_bet_type"])
        self.assertTrue(summary["by_sport"])


if __name__ == "__main__":
    unittest.main()
