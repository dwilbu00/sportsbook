"""Tests for the multi-sport historical-odds backfill routing logic
(credit-windfall backfill: --category / --snapshot / props floor)."""

import unittest

import backfill_historical_odds as bf


class ResolveCategoryTests(unittest.TestCase):
    def test_team_forces_props_off_keeps_markets(self):
        self.assertEqual(
            bf._resolve_category("team", "h2h,spreads,totals", "ignored", "basketball_nba"),
            ("h2h,spreads,totals", ""))

    def test_props_fills_broad_set_and_clears_featured(self):
        markets, props = bf._resolve_category("props", "h2h,spreads,totals", "",
                                              "basketball_nba")
        self.assertEqual(markets, "")
        self.assertEqual(props, ",".join(bf.BACKFILL_PROPS_BY_SPORT["basketball_nba"]))

    def test_props_respects_explicit_props(self):
        self.assertEqual(
            bf._resolve_category("props", "x", "player_points", "basketball_nba"),
            ("", "player_points"))

    def test_none_is_passthrough(self):
        self.assertEqual(
            bf._resolve_category(None, "h2h", "player_points", "baseball_mlb"),
            ("h2h", "player_points"))

    def test_props_unknown_sport_raises(self):
        with self.assertRaises(ValueError):
            bf._resolve_category("props", "", "", "icehockey_nhl")


class ResolveSnapshotModeTests(unittest.TestCase):
    def test_early_sets_daily_time_label(self):
        self.assertEqual(
            bf._resolve_snapshot_mode("early", None, "", "13:00", "commence"),
            ("daily", "13:00", "morning"))

    def test_close_is_unchanged(self):
        self.assertEqual(
            bf._resolve_snapshot_mode("close", None, "", "13:00", "commence"),
            ("commence", None, ""))

    def test_early_keeps_explicit_time_and_label(self):
        self.assertEqual(
            bf._resolve_snapshot_mode("early", "09:30", "custom", "13:00", "commence"),
            ("daily", "09:30", "custom"))


class PropLabelCoverageTests(unittest.TestCase):
    """Spend-review #1 regression: an Odds-API prop key NOT in
    odds_client.PROP_LABELS is silently dropped by parse_player_props at the
    durable-write layer (paid credits -> zero stored lines). So every broad-corpus
    backfill key MUST have a PROP_LABELS entry, and the CLI pre-flight guard must
    reject any that don't."""

    def test_every_backfill_prop_key_has_a_label(self):
        from odds_client import PROP_LABELS
        for sk, keys in bf.BACKFILL_PROPS_BY_SPORT.items():
            for k in keys:
                self.assertIn(k, PROP_LABELS,
                              f"{k} ({sk}) not in PROP_LABELS -> would be silently "
                              f"dropped at the durable-write layer")


class PropsFloorTests(unittest.TestCase):
    def test_floor_is_the_vendor_props_start(self):
        self.assertEqual(bf.PROPS_MIN_DATE, "2023-05-03")

    def test_backfill_prop_sets_are_broad_and_present(self):
        # capture-broad corpus sets exist for the 3 modeled sports
        for sk in ("baseball_mlb", "americanfootball_nfl", "basketball_nba"):
            self.assertGreaterEqual(len(bf.BACKFILL_PROPS_BY_SPORT[sk]), 7)


if __name__ == "__main__":
    unittest.main()
