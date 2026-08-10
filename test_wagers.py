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
from unittest.mock import MagicMock, patch

import db_store
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
        self._p1.start()
        return self

    def __exit__(self, *exc):
        self._p1.stop()
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


class DnpAutoVoidTests(unittest.TestCase):
    """A player prop whose player is a confirmed, stale DNP (listed but never
    played) is un-gradable forever. resolve_pending_wagers must VOID it (stake
    refunded, ROI-neutral) instead of stranding it as pending every tick — the
    same stale-DNP sweep the prediction resolver already does."""

    def _dnp_prop(self, seq=0):
        # Commence is comfortably past the 3h MLB buffer relative to the test
        # clock, so the row reaches _grade_wager (the age gate lives inside the
        # mocked _is_stale_dnp, so this fixture doesn't depend on wall time).
        return wagers.build_wager_row("player_prop", None, {
            "player": "Reynaldo Lopez", "prop": "pitcher_outs",
            "prop_label": "Outs", "line": 15.5, "direction": "UNDER",
            "under_price": -110, "under_rate": 55.0, "edge_pct": 5.0,
            "matchup": "WSH @ ATL", "team": "Atlanta Braves", "event_id": "E1"},
            {"sport_key": "baseball_mlb", "event_id": "E1",
             "commence_time": "2026-08-01T23:16:00Z", "game_date": "2026-08-01",
             "home_team": "Atlanta Braves", "away_team": "Washington Nationals",
             "stake": 10.0, "placed_at": "2026-08-01T20:00:00+00:00", "seq": seq})

    def _now(self):
        return datetime(2026, 8, 3, 13, 0, tzinfo=timezone.utc)  # ~38h later

    def test_confirmed_stale_dnp_is_voided(self):
        with _LocalLedger():
            wagers.submit_wagers([self._dnp_prop()])
            with patch.object(recalibration, "resolve_one_prop",
                              return_value=None), \
                 patch.object(recalibration, "_is_stale_dnp", return_value=True):
                self.assertEqual(wagers.resolve_pending_wagers(now=self._now()), 1)
            row = wagers.read_wagers()[0]
            self.assertEqual(row["status"], "void")
            self.assertIsNone(row["actual"])
            self.assertAlmostEqual(row["profit"], 0.0)  # stake refunded
            self.assertIsNotNone(row["resolved_at"])

    def test_not_yet_stale_dnp_stays_pending(self):
        # Un-gradable but NOT yet a confirmed stale DNP (same-day data lag): the
        # resolver returns None and the void gate says "not stale" -> stay pending
        # and retry next tick, never void prematurely.
        with _LocalLedger():
            wagers.submit_wagers([self._dnp_prop()])
            with patch.object(recalibration, "resolve_one_prop",
                              return_value=None), \
                 patch.object(recalibration, "_is_stale_dnp", return_value=False):
                self.assertEqual(wagers.resolve_pending_wagers(now=self._now()), 0)
            self.assertEqual(wagers.read_wagers()[0]["status"], "pending")

    def test_void_survives_the_sql_where_filtered_path(self):
        with _SqlLedger():
            wagers.submit_wagers([self._dnp_prop()])
            with patch.object(recalibration, "resolve_one_prop",
                              return_value=None), \
                 patch.object(recalibration, "_is_stale_dnp", return_value=True):
                self.assertEqual(wagers.resolve_pending_wagers(now=self._now()), 1)
            self.assertEqual(wagers.read_wagers()[0]["status"], "void")

    def test_team_bet_never_auto_voids(self):
        # The DNP void path is player-prop-only; a team market that can't resolve
        # (no final score) stays pending, it is never voided as a scratch.
        row = wagers.build_wager_row("moneyline", None, {
            "team": "Atlanta Braves", "opponent": "Washington Nationals",
            "home_away": "HOME", "best_price": 120, "event_id": "E1"},
            {"sport_key": "baseball_mlb", "event_id": "E1",
             "commence_time": "2026-08-01T23:16:00Z", "game_date": "2026-08-01",
             "home_team": "Atlanta Braves", "away_team": "Washington Nationals",
             "stake": 10.0, "placed_at": "2026-08-01T20:00:00+00:00", "seq": 0})
        with _LocalLedger():
            wagers.submit_wagers([row])
            with patch.object(game_results, "final_score", return_value=None), \
                 patch.object(recalibration, "_is_stale_dnp", return_value=True):
                self.assertEqual(wagers.resolve_pending_wagers(now=self._now()), 0)
            self.assertEqual(wagers.read_wagers()[0]["status"], "pending")


class TeamIdentityGradingTests(unittest.TestCase):
    """A moneyline/spread bet must grade off the team it was placed on, even if
    the stored `side` is stale/flipped (the Yankees-graded-as-Phillies bug)."""

    def test_side_for_team(self):
        self.assertEqual(game_results.side_for_team(
            "New York Yankees", "Philadelphia Phillies", "New York Yankees"),
            "away")
        self.assertEqual(game_results.side_for_team(
            "Philadelphia Phillies", "Philadelphia Phillies", "New York Yankees"),
            "home")
        self.assertIsNone(game_results.side_for_team(
            "Boston Red Sox", "Philadelphia Phillies", "New York Yankees"))

    def test_moneyline_grades_by_team_not_stale_side(self):
        # Yankees (away) ML; final = Phillies 11, Yankees 4 → the bet LOST.
        row = wagers.build_wager_row("moneyline", None, {
            "team": "New York Yankees", "opponent": "Philadelphia Phillies",
            "home_away": "AWAY", "best_price": 120, "event_id": "E1"},
            {"sport_key": "baseball_mlb", "event_id": "E1",
             "commence_time": "2026-07-26T23:20:00Z", "game_date": "2026-07-26",
             "home_team": "Philadelphia Phillies",
             "away_team": "New York Yankees", "stake": 10.0,
             "placed_at": "2026-07-26T20:00:00+00:00", "seq": 0})
        row["side"] = "home"   # simulate a stale/flipped stored side
        with patch.object(game_results, "final_score", return_value=(11, 4)):
            self.assertEqual(wagers._grade_wager(row), ("lost", "11-4"))


class FinalScoreDisambiguationTests(unittest.TestCase):
    """final_score must pick the exact night in a series and never settle a bet
    off an adjacent-day game (stale-slate / series wrong-night bug)."""

    def setUp(self):
        game_results._SCORE_CACHE.clear()

    def tearDown(self):
        game_results._SCORE_CACHE.clear()

    _SERIES = {
        "2026-07-25": [{"home_team": "Philadelphia Phillies",
                        "away_team": "New York Yankees", "home_score": 1.0,
                        "away_score": 3.0,
                        "commence_time": "2026-07-25T22:05:00Z"}],
        "2026-07-26": [{"home_team": "Philadelphia Phillies",
                        "away_team": "New York Yankees", "home_score": 11.0,
                        "away_score": 4.0,
                        "commence_time": "2026-07-26T23:20:00Z"}],
    }

    def test_picks_exact_night_in_series(self):
        with patch.object(game_results, "_scores_for_date",
                          side_effect=lambda sk, d: self._SERIES.get(str(d)[:10], [])):
            score = game_results.final_score(
                "baseball_mlb", "2026-07-26", "Philadelphia Phillies",
                "New York Yankees", "2026-07-26T23:20:00Z")
        self.assertEqual(score, (11.0, 4.0))     # the 7/26 game, not 7/25

    def test_wrong_night_only_match_is_rejected(self):
        # 7/26 missing from the slate; only 7/25 matches → >20h off → don't grade.
        only_25 = {"2026-07-25": self._SERIES["2026-07-25"]}
        with patch.object(game_results, "_scores_for_date",
                          side_effect=lambda sk, d: only_25.get(str(d)[:10], [])):
            score = game_results.final_score(
                "baseball_mlb", "2026-07-26", "Philadelphia Phillies",
                "New York Yankees", "2026-07-26T23:20:00Z")
        self.assertIsNone(score)


class SlateCompletenessCacheTests(unittest.TestCase):
    """A slate is memoized for the process lifetime only once COMPLETE. An
    incomplete slate (games still live) must re-fetch after the short memo TTL so
    a game that goes final is picked up — the UTC-vs-Eastern immutability bug that
    stranded late games' bets and forecasts as 'pending' until a process restart."""

    def setUp(self):
        game_results._SCORE_CACHE.clear()

    def tearDown(self):
        game_results._SCORE_CACHE.clear()

    _PARTIAL = [{"home_team": "A", "away_team": "B", "home_score": 1.0,
                 "away_score": 2.0, "commence_time": "2026-07-30T23:00:00Z"}]
    _FULL = _PARTIAL + [{"home_team": "C", "away_team": "D", "home_score": 4.0,
                         "away_score": 3.0, "commence_time": "2026-07-31T02:00:00Z"}]

    def _run(self, slates, times):
        """Drive _scores_for_date over a scripted [(games, complete), ...] fetch
        sequence and a scripted clock; return (results, fetch_count)."""
        calls = []

        def fake_slate(_gd):
            calls.append(_gd)
            return slates[min(len(calls) - 1, len(slates) - 1)]

        clock = {"t": 0.0}
        fake_time = MagicMock()
        fake_time.time.side_effect = lambda: clock["t"]
        results = []
        with patch.object(game_results, "_mlb_slate_for_date",
                          side_effect=fake_slate), \
             patch.object(game_results, "time", fake_time):
            for t in times:
                clock["t"] = t
                results.append(
                    game_results._scores_for_date("baseball_mlb", "2026-07-30"))
        return results, len(calls)

    def test_incomplete_slate_refetches_after_ttl(self):
        ttl = game_results._RECENT_MEMO_TTL
        results, fetches = self._run(
            slates=[(self._PARTIAL, False), (self._FULL, True)],
            times=[1000.0, 1000.0 + ttl - 1, 1000.0 + ttl + 1])
        self.assertEqual(results[0], self._PARTIAL)   # first fetch: still partial
        self.assertEqual(results[1], self._PARTIAL)   # within TTL: served from memo
        self.assertEqual(results[2], self._FULL)      # past TTL: re-fetched → full
        self.assertEqual(fetches, 2)                  # exactly one re-fetch

    def test_complete_slate_never_refetches(self):
        results, fetches = self._run(
            slates=[(self._FULL, True)],
            times=[1000.0, 1000.0 + game_results._RECENT_MEMO_TTL * 100])
        self.assertEqual(results[0], self._FULL)
        self.assertEqual(results[1], self._FULL)
        self.assertEqual(fetches, 1)                  # complete → cached indefinitely


class ReadStatusTests(unittest.TestCase):
    """read_wagers_with_status surfaces a backend outage so My Bets can show an
    'unreachable' banner instead of 'no submitted picks yet'."""

    def test_surfaces_backend_error(self):
        with patch.object(recalibration, "_read_ndjson_blob",
                          side_effect=RuntimeError("timeout")):
            rows, err = wagers.read_wagers_with_status()
            self.assertEqual(rows, [])
            self.assertIsInstance(err, RuntimeError)
            # Legacy read_wagers still swallows to [] for non-UI callers.
            self.assertEqual(wagers.read_wagers(), [])

    def test_ok_returns_rows_and_no_error(self):
        with patch.object(recalibration, "_read_ndjson_blob",
                          return_value=([{"wager_id": "w1"}], None)):
            rows, err = wagers.read_wagers_with_status()
        self.assertIsNone(err)
        self.assertEqual(rows, [{"wager_id": "w1"}])


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

    def test_reset_empty_id_list_clears_nothing(self):
        # An explicitly EMPTY selection must clear NOTHING (not everything) —
        # a caller that computed "rows to reset" and got none (e.g. migrate on a
        # props-only ledger) must not accidentally wipe every CLV value.
        with _LocalLedger():
            wagers.submit_wagers([
                {"wager_id": "a", "status": "won", "close_price": -110,
                 "clv_pct": 1.0}])
            self.assertEqual(wagers.reset_clv([]), 0)
            self.assertEqual(wagers.read_wagers()[0]["close_price"], -110)


class _SqlLedger:
    """Context manager: route wagers/recalibration onto a fresh in-memory SQL
    store (so the ``where``-filtered read/mutate paths are actually exercised —
    the local NDJSON path ignores ``where``)."""

    def __enter__(self):
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        return self

    def __exit__(self, *exc):
        db_store.configure_engine(None)
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()


class FilteredWagerDmlSqlTests(unittest.TestCase):
    """The by-id/IS-NULL ``where`` filters the wager DML now pass must produce
    the same results on the SQL path as the (where-ignoring) Blob path."""

    def _prop(self, wid_seq, team="Boston Red Sox"):
        return wagers.build_wager_row("player_prop", None, {
            "player": f"P{wid_seq}", "prop": "batter_hits", "prop_label": "Hits",
            "line": 1.5, "direction": "OVER", "over_price": -110,
            "over_rate": 60.0, "edge_pct": 7.0, "matchup": "NYY @ BOS",
            "team": team, "event_id": "E1"}, _meta(wid_seq))

    def test_delete_targets_only_given_ids(self):
        a, b, c = self._prop(0), self._prop(1), self._prop(2)
        with _SqlLedger():
            wagers.submit_wagers([a, b, c])
            self.assertEqual(wagers.delete_wagers([b["wager_id"]]), 1)
            self.assertEqual({r["wager_id"] for r in wagers.read_wagers()},
                             {a["wager_id"], c["wager_id"]})
            self.assertEqual(wagers.delete_wagers([b["wager_id"]]), 0)  # gone

    def test_update_edits_only_the_targeted_pending_row(self):
        a, b = self._prop(0), self._prop(1)
        with _SqlLedger():
            wagers.submit_wagers([a, b])
            self.assertEqual(
                wagers.update_wagers({a["wager_id"]: {"stake": 25.0}}), 1)
            rows = {r["wager_id"]: r for r in wagers.read_wagers()}
            self.assertEqual(rows[a["wager_id"]]["stake"], 25.0)
            self.assertEqual(rows[b["wager_id"]]["stake"], 10.0)  # untouched

    def test_regrade_resets_only_targeted_settled_row(self):
        a, b = self._prop(0), self._prop(1)
        now = datetime(2026, 7, 20, tzinfo=timezone.utc)
        with _SqlLedger():
            wagers.submit_wagers([a, b])
            with patch.object(recalibration, "resolve_one_prop", return_value=0.0):
                wagers.resolve_pending_wagers(now=now)
            rows = {r["wager_id"]: r for r in wagers.read_wagers()}
            self.assertEqual(rows[a["wager_id"]]["status"], "lost")
            self.assertEqual(wagers.regrade_wagers([a["wager_id"]]), 1)
            rows = {r["wager_id"]: r for r in wagers.read_wagers()}
            self.assertEqual(rows[a["wager_id"]]["status"], "pending")
            self.assertEqual(rows[b["wager_id"]]["status"], "lost")  # untouched

    def test_apply_clv_updates_is_null_filter_fills_and_is_idempotent(self):
        # apply_clv_updates is the sole CLV writer (fed by backfill_dk_clv.py).
        # It must fill only close_price IS NULL rows via the SQL ``where`` filter,
        # so a re-run with the same input writes nothing.
        meta = {"sport_key": "baseball_mlb", "event_id": "E1",
                "commence_time": "2026-07-21T02:30:00Z", "game_date": "2026-07-21",
                "home_team": "H", "away_team": "A", "stake": 10.0,
                "placed_at": "2026-07-20T12:00:00+00:00", "seq": 0}
        row = wagers.build_wager_row("total", "OVER", {
            "line": 8.5, "over_price": -110, "under_price": -105,
            "over_hit_rate": 55.0, "over_edge_pct": 6.0,
            "matchup": "A @ H", "event_id": "E1"}, meta)
        with _SqlLedger():
            wagers.submit_wagers([row])
            wid = row["wager_id"]
            filled = {wid: {"close_price": -120, "close_line": 8.5,
                            "clv_pct": 3.2}}
            # First pass fills the close_price IS NULL row via the SQL filter.
            self.assertEqual(wagers.apply_clv_updates(filled), 1)
            saved = wagers.read_wagers()[0]
            self.assertEqual(saved["close_price"], -120)
            self.assertEqual(saved["clv_pct"], 3.2)
            # Second pass: the IS NULL filter now returns no rows -> no work.
            self.assertEqual(wagers.apply_clv_updates(filled), 0)


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

    def test_void_counts_as_resolved_but_roi_neutral(self):
        # A voided (scratch/DNP) bet is SETTLED, not pending, and refunds the
        # stake: it carries no won/lost/push, no staked amount, and no realized
        # P/L, but it must not sit in the pending bucket forever.
        rows = [
            {"sport_key": "baseball_mlb", "bet_type": "player_prop",
             "status": "won", "stake": 10.0, "profit": 9.0},
            {"sport_key": "baseball_mlb", "bet_type": "player_prop",
             "prop_key": "pitcher_outs", "status": "void", "stake": 10.0,
             "profit": 0.0},
            {"sport_key": "baseball_mlb", "bet_type": "moneyline",
             "status": "pending", "stake": 10.0, "profit": None},
        ]
        summary = wagers.summarize_wagers(rows)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["resolved"], 2)   # won + void
        self.assertEqual(summary["pending"], 1)     # the void is NOT pending
        self.assertEqual(summary["void"], 1)
        self.assertAlmostEqual(summary["total_staked"], 10.0)   # void excluded
        self.assertAlmostEqual(summary["realized_profit"], 9.0)  # void excluded
        self.assertEqual((summary["won"], summary["lost"], summary["push"]),
                         (1, 0, 0))
        self.assertAlmostEqual(summary["pending_stake"], 10.0)   # only the pending
        # The void surfaces in its own by-bet-type bucket as resolved, not pending.
        void_bucket = next(b for b in summary["by_bet_type"]
                           if b.get("prop_key") == "pitcher_outs")
        self.assertEqual((void_bucket["resolved"], void_bucket["pending"],
                          void_bucket["void"]), (1, 0, 1))

    def test_by_bet_type_splits_props_by_market(self):
        rows = [
            {"sport_key": "baseball_mlb", "bet_type": "player_prop",
             "prop_key": "batter_hits", "prop_label": "Batter Hits",
             "status": "won", "stake": 10.0, "profit": 9.0},
            {"sport_key": "baseball_mlb", "bet_type": "player_prop",
             "prop_key": "pitcher_strikeouts", "prop_label": "Pitcher Ks",
             "status": "lost", "stake": 10.0, "profit": -10.0},
            {"sport_key": "baseball_mlb", "bet_type": "spread",
             "status": "won", "stake": 10.0, "profit": 9.0},
        ]
        by_type = wagers.summarize_wagers(rows)["by_bet_type"]
        labels = {b["label"] for b in by_type}
        # Two DISTINCT prop markets, not one pooled "Player Prop" bucket.
        self.assertIn("Player Prop — Batter Hits", labels)
        self.assertIn("Player Prop — Pitcher Ks", labels)
        self.assertIn("Spread", labels)
        hits = next(b for b in by_type if b["label"] == "Player Prop — Batter Hits")
        self.assertEqual(hits["prop_key"], "batter_hits")
        self.assertEqual((hits["won"], hits["lost"]), (1, 0))

    def test_by_bet_type_prop_label_falls_back_to_prop_key(self):
        rows = [{"sport_key": "baseball_mlb", "bet_type": "player_prop",
                 "prop_key": "batter_total_bases", "status": "won",
                 "stake": 10.0, "profit": 9.0}]
        by_type = wagers.summarize_wagers(rows)["by_bet_type"]
        self.assertEqual(by_type[0]["label"], "Player Prop — Batter Total Bases")


if __name__ == "__main__":
    unittest.main()
