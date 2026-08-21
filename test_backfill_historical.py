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


class WarehouseGapFillTests(unittest.TestCase):
    """--gap-fill warehouse-coverage diff: only games MISSING a landed snapshot
    for the (kind, source) role should be planned for (re)fetch."""

    def setUp(self):
        import db_store
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        self.t = db_store.odds_snapshot
        self.ln = db_store.odds_line
        self.eng = db_store.get_engine()

    def _snap(self, eid, gd, home, away, kind="team",
              source="backfill_close", with_line=True):
        from sqlalchemy import insert
        with self.eng.begin() as c:
            r = c.execute(insert(self.t).values(
                sport="baseball_mlb", game_date=gd, event_id=eid, kind=kind,
                snapshot_hour="h" + eid, captured_at=gd + "T23:00:00Z",
                commence_time=gd + "T23:00:00Z", home=home, away=away, source=source))
            sid = r.inserted_primary_key[0]
            if with_line:
                c.execute(insert(self.ln).values(
                    snapshot_id=sid, bet_type="moneyline", selection=home, price=-120))
        return sid

    def test_covered_indexes_landed_snapshots(self):
        self._snap("e1", "2024-06-01", "New York Yankees", "Boston Red Sox")
        cov = bf._warehouse_covered("baseball_mlb", "team", "backfill_close",
                                    ["2024-06-01"])
        self.assertIn(("New York Yankees", "Boston Red Sox"), cov["2024-06-01"])

    def test_lineless_snapshot_still_counts_as_covered(self):
        # capture writes snapshot+lines atomically, so a lineless snapshot is a
        # faithful 'fetched, book had nothing' record — NOT a load failure. It must
        # count as covered so gap-fill doesn't perpetually re-attempt (and can't
        # repair it anyway: write-once blocks re-landing lines into the bucket).
        self._snap("e1", "2024-06-01", "Yankees", "Red Sox", with_line=False)
        cov = bf._warehouse_covered("baseball_mlb", "team", "backfill_close",
                                    ["2024-06-01"])
        self.assertIn(("Yankees", "Red Sox"), cov["2024-06-01"])

    def test_source_and_kind_are_scoped(self):
        self._snap("e1", "2024-06-01", "Yankees", "Red Sox", source="backfill_early")
        self._snap("e2", "2024-06-01", "Cubs", "Mets", kind="props")
        # asking for team/backfill_close matches neither the early nor the props row
        self.assertEqual(
            bf._warehouse_covered("baseball_mlb", "team", "backfill_close",
                                  ["2024-06-01"]), {})
        self.assertIn(("Yankees", "Red Sox"),
                      bf._warehouse_covered("baseball_mlb", "team", "backfill_early",
                                            ["2024-06-01"])["2024-06-01"])

    def test_date_range_scoped(self):
        self._snap("e1", "2024-06-01", "Yankees", "Red Sox")
        self._snap("e2", "2024-07-01", "Cubs", "Mets")
        cov = bf._warehouse_covered("baseball_mlb", "team", "backfill_close",
                                    ["2024-06-01"])
        self.assertNotIn("2024-07-01", cov)

    def test_is_covered_fuzzy_matches_espn_naming(self):
        cov = {"2024-06-01": [("New York Yankees", "Boston Red Sox")]}
        self.assertTrue(bf._is_covered(
            {"date": "2024-06-01T23:05:00Z", "home_team": "Yankees",
             "away_team": "Red Sox"}, cov))
        self.assertFalse(bf._is_covered(  # wrong opponent
            {"date": "2024-06-01T23:05:00Z", "home_team": "Yankees",
             "away_team": "Orioles"}, cov))
        self.assertFalse(bf._is_covered(  # right teams, wrong date
            {"date": "2024-06-02T23:05:00Z", "home_team": "Yankees",
             "away_team": "Red Sox"}, cov))

    def test_no_dates_short_circuits_empty(self):
        self.assertEqual(
            bf._warehouse_covered("baseball_mlb", "team", "backfill_close", []), {})


if __name__ == "__main__":
    unittest.main()
