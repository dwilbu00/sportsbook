"""Tests for the team-market top-up gap diff (topup_team_odds.compute_gap).

Pure/offline — no SQL, no network. The diff decides which DATES get an Odds-API
snapshot (real credits), so the multiset match, the ±1-day tolerance, doubleheader
handling, and unresolved-code skipping are all covered.
"""

import unittest

import topup_team_odds as tu


def _g(pk, od, codeset, commence=None):
    return {"game_pk": pk, "official_date": od, "codeset": codeset,
            "commence": commence or f"{od}T20:00:00Z"}


def _s(date10, codeset):
    return {"date10": date10, "codeset": codeset, "event_id": f"e{date10}"}


NYY_TOR = frozenset({"NYY", "TOR"})
BOS_BAL = frozenset({"BOS", "BAL"})


class ComputeGapTests(unittest.TestCase):
    def test_all_missing_when_no_snapshots(self):
        games = [_g(1, "2025-08-17", NYY_TOR), _g(2, "2025-08-17", BOS_BAL)]
        miss, dates, unresolved = tu.compute_gap(games, [])
        self.assertEqual(len(miss), 2)
        self.assertEqual(dates, {"2025-08-17": 2})
        self.assertEqual(unresolved, [])

    def test_exact_match_covers(self):
        games = [_g(1, "2025-08-17", NYY_TOR)]
        snaps = [_s("2025-08-17", NYY_TOR)]
        miss, dates, _ = tu.compute_gap(games, snaps)
        self.assertEqual(miss, [])
        self.assertEqual(dates, {})

    def test_plus_minus_one_day_tolerance(self):
        # UTC snapshot lands the day AFTER the ET official date.
        games = [_g(1, "2025-08-17", NYY_TOR)]
        snaps = [_s("2025-08-18", NYY_TOR)]
        miss, dates, _ = tu.compute_gap(games, snaps)
        self.assertEqual(miss, [])

    def test_wrong_codeset_not_covered(self):
        games = [_g(1, "2025-08-17", NYY_TOR)]
        snaps = [_s("2025-08-17", BOS_BAL)]
        miss, dates, _ = tu.compute_gap(games, snaps)
        self.assertEqual(len(miss), 1)
        self.assertEqual(dates, {"2025-08-17": 1})

    def test_two_days_off_not_covered(self):
        games = [_g(1, "2025-08-17", NYY_TOR)]
        snaps = [_s("2025-08-19", NYY_TOR)]
        miss, _, _ = tu.compute_gap(games, snaps)
        self.assertEqual(len(miss), 1)

    def test_doubleheader_needs_two_snapshots(self):
        # Two games, same date + same code-set; only ONE snapshot → one still missing.
        games = [_g(1, "2025-08-17", NYY_TOR), _g(2, "2025-08-17", NYY_TOR)]
        snaps = [_s("2025-08-17", NYY_TOR)]
        miss, dates, _ = tu.compute_gap(games, snaps)
        self.assertEqual(len(miss), 1)
        self.assertEqual(dates, {"2025-08-17": 1})
        # Two snapshots → both covered.
        miss2, _, _ = tu.compute_gap(games, [_s("2025-08-17", NYY_TOR),
                                             _s("2025-08-17", NYY_TOR)])
        self.assertEqual(miss2, [])

    def test_snapshot_not_double_counted_across_games(self):
        # One snapshot must not cover two different games on nearby dates.
        games = [_g(1, "2025-08-17", NYY_TOR), _g(2, "2025-08-18", NYY_TOR)]
        snaps = [_s("2025-08-17", NYY_TOR)]
        miss, dates, _ = tu.compute_gap(games, snaps)
        # The 08-17 game consumes the snapshot; 08-18 has none within ±1? 08-18
        # would look at 08-17 (already consumed), 08-18, 08-19 → missing.
        self.assertEqual(len(miss), 1)
        self.assertEqual(miss[0]["game_pk"], 2)

    def test_unresolved_codeset_skipped_not_fetched(self):
        games = [_g(1, "2025-08-17", None)]
        miss, dates, unresolved = tu.compute_gap(games, [])
        self.assertEqual(miss, [])
        self.assertEqual(dates, {})
        self.assertEqual(len(unresolved), 1)

    def test_snapshot_ts_picks_first_tipoff(self):
        ts = tu.snapshot_ts_for_date(
            ["2025-08-17T23:05:00Z", "2025-08-17T17:10:00Z", None])
        self.assertEqual(ts, "2025-08-17T17:10:00Z")
        self.assertIsNone(tu.snapshot_ts_for_date([None, None]))


if __name__ == "__main__":
    unittest.main()
