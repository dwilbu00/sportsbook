"""Unit tests for r2_data — the pure DK/Pinnacle pairing + close-selection core and
the outcome lookup. No warehouse: synthetic warehouse-shaped rows."""
import unittest

import r2_data as d


def _row(book, sid, cap, com, prop="pitcher_earned_runs", side="OVER", pt=0.5, price=-110,
         player="Gunnar Henderson", mlbid="683002", eid="E1", gpk=700):
    return {"book": book, "snapshot_id": sid, "captured_at": cap,
            "commence_time": com, "event_id": eid, "player": player,
            "player_mlb_id": mlbid, "prop_key": prop, "direction": side,
            "point": pt, "price": price, "game_pk": gpk, "game_date": com[:10]}


def _two_sided(book, sid, cap, com, pt=0.5, over=-110, under=-110, **kw):
    return [_row(book, sid, cap, com, side="OVER", pt=pt, price=over, **kw),
            _row(book, sid, cap, com, side="UNDER", pt=pt, price=under, **kw)]


COM = "2024-06-26T23:10:00Z"


class ParseTsTests(unittest.TestCase):
    def test_z_and_offset_and_bad(self):
        self.assertIsNotNone(d._parse_ts("2024-06-26T23:05:38Z"))
        self.assertIsNotNone(d._parse_ts("2024-06-26T23:05:38+00:00"))
        self.assertIsNone(d._parse_ts(None))
        self.assertIsNone(d._parse_ts("not-a-date"))

    def test_z_equals_offset(self):
        self.assertEqual(d._parse_ts("2024-06-26T23:05:38Z"),
                         d._parse_ts("2024-06-26T23:05:38+00:00"))


class CloseSelectionTests(unittest.TestCase):
    def test_picks_latest_pre_commence_snapshot_with_both_books(self):
        rows = []
        # open snapshot sid=1 @ 12:00Z (both books), close sid=2 @ 23:05Z (both books)
        for b in ("draftkings", "pinnacle"):
            rows += _two_sided(b, 1, "2024-06-26T12:00:00Z", COM, over=100, under=-120)
            rows += _two_sided(b, 2, "2024-06-26T23:05:00Z", COM, over=-110, under=-110)
        legs, stats = d.select_prop_legs(rows)
        self.assertEqual(len(legs), 1)
        self.assertEqual(legs[0].snapshot_id, 2)         # the close, not the open
        self.assertEqual(legs[0].dk_over_price, -110)    # from sid=2

    def test_post_commence_snapshot_excluded(self):
        rows = []
        for b in ("draftkings", "pinnacle"):
            rows += _two_sided(b, 1, "2024-06-26T22:00:00Z", COM)          # valid close
            rows += _two_sided(b, 2, "2024-06-26T23:40:00Z", COM, over=250)  # in-play (post-commence)
        legs, _ = d.select_prop_legs(rows)
        self.assertEqual(len(legs), 1)
        self.assertEqual(legs[0].snapshot_id, 1)         # NOT the later in-play snap

    def test_never_pairs_across_snapshots(self):
        # DK only in sid=2, Pinnacle only in sid=1 -> no snapshot has BOTH -> drop.
        rows = _two_sided("pinnacle", 1, "2024-06-26T12:00:00Z", COM)
        rows += _two_sided("draftkings", 2, "2024-06-26T23:05:00Z", COM)
        legs, stats = d.select_prop_legs(rows)
        self.assertEqual(legs, [])
        self.assertEqual(stats["events_dropped_no_both_book_close"], 1)

    def test_event_without_pinnacle_dropped(self):
        rows = _two_sided("draftkings", 1, "2024-06-26T23:05:00Z", COM)
        legs, stats = d.select_prop_legs(rows)
        self.assertEqual(legs, [])
        self.assertEqual(stats["events_dropped_no_both_book_close"], 1)


class OfferAssemblyTests(unittest.TestCase):
    def test_cross_line_offers_preserved(self):
        # Pinnacle posts BOTH 0.5 and 1.5 (two-sided); DK posts 0.5 only.
        rows = _two_sided("pinnacle", 1, "2024-06-26T23:05:00Z", COM, pt=0.5)
        rows += _two_sided("pinnacle", 1, "2024-06-26T23:05:00Z", COM, pt=1.5)
        rows += _two_sided("draftkings", 1, "2024-06-26T23:05:00Z", COM, pt=0.5)
        legs, _ = d.select_prop_legs(rows)
        self.assertEqual(len(legs), 1)
        self.assertEqual(legs[0].dk_point, 0.5)
        pts = sorted(o["point"] for o in legs[0].pinnacle_offers)
        self.assertEqual(pts, [0.5, 1.5])                # both survive for projection

    def test_one_sided_pinnacle_dropped(self):
        # Pinnacle posts only OVER (no UNDER) -> can't devig -> leg dropped.
        rows = [_row("pinnacle", 1, "2024-06-26T23:05:00Z", COM, side="OVER")]
        rows += _two_sided("draftkings", 1, "2024-06-26T23:05:00Z", COM)
        legs, stats = d.select_prop_legs(rows)
        self.assertEqual(legs, [])
        self.assertEqual(stats["leg_dropped_no_pinnacle_twosided"], 1)

    def test_dk_and_pin_join_on_mlb_id_despite_name_diff(self):
        # Books spell the name differently but share the MLBAM id -> one leg.
        rows = _two_sided("pinnacle", 1, "2024-06-26T23:05:00Z", COM,
                          player="G. Henderson", mlbid="683002")
        rows += _two_sided("draftkings", 1, "2024-06-26T23:05:00Z", COM,
                           player="Gunnar Henderson", mlbid="683002")
        legs, _ = d.select_prop_legs(rows)
        self.assertEqual(len(legs), 1)
        self.assertIsNotNone(legs[0].dk_over_price)
        self.assertTrue(legs[0].pinnacle_offers)


class SynonymTests(unittest.TestCase):
    """DK batter_hits priced off Pinnacle batter_total_bases (TB>=1 <=> H>=1)."""

    def _rows(self, dk_hits_point, pin_tb_points):
        # DK posts hits at dk_hits_point; Pinnacle posts TB at the given points.
        rows = _two_sided("draftkings", 1, "2024-06-26T23:05:00Z", COM,
                          prop="batter_hits", pt=dk_hits_point)
        for pt in pin_tb_points:
            rows += _two_sided("pinnacle", 1, "2024-06-26T23:05:00Z", COM,
                               prop="batter_total_bases", pt=pt)
        return rows

    def test_hits_0p5_priced_off_pinnacle_tb_0p5(self):
        legs, _ = d.select_prop_legs(self._rows(0.5, [0.5, 1.5]))
        hits = [lg for lg in legs if lg.prop_key == "batter_hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].dk_point, 0.5)
        self.assertEqual(hits[0].ref_prop, "batter_total_bases")   # synonym-priced
        pts = sorted(o["point"] for o in hits[0].pinnacle_offers)
        self.assertEqual(pts, [0.5, 1.5])          # Pinnacle TB offers, not hits

    def test_hits_1p5_dropped_no_tb_identity(self):
        # hits 1.5 has NO TB identity (P(TB>=2) != P(H>=2)) -> no synonym leg.
        legs, stats = d.select_prop_legs(self._rows(1.5, [0.5, 1.5]))
        hits = [lg for lg in legs if lg.prop_key == "batter_hits"]
        self.assertEqual(hits, [])
        self.assertEqual(stats["leg_dropped_synonym_bad_point"], 1)

    def test_hits_priced_even_when_pinnacle_only_posts_tb_1p5(self):
        # Only TB 1.5 posted -> projector back-solves P(TB>=1) at the 0.5 target.
        legs, _ = d.select_prop_legs(self._rows(0.5, [1.5]))
        hits = [lg for lg in legs if lg.prop_key == "batter_hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual([o["point"] for o in hits[0].pinnacle_offers], [1.5])

    def test_same_prop_tb_still_priced_directly(self):
        # A DK TB leg still uses Pinnacle TB directly (ref_prop None = same prop).
        rows = _two_sided("draftkings", 1, "2024-06-26T23:05:00Z", COM,
                          prop="batter_total_bases", pt=1.5)
        rows += _two_sided("pinnacle", 1, "2024-06-26T23:05:00Z", COM,
                           prop="batter_total_bases", pt=1.5)
        legs, _ = d.select_prop_legs(rows)
        tb = [lg for lg in legs if lg.prop_key == "batter_total_bases"]
        self.assertEqual(len(tb), 1)
        self.assertIsNone(tb[0].ref_prop)


class TeamMLTests(unittest.TestCase):
    def _ml_row(self, book, sid, cap, com, selection, price, home="NYM", away="NYY",
                eid="G1", gpk=555):
        return {"book": book, "snapshot_id": sid, "captured_at": cap,
                "commence_time": com, "event_id": eid, "home": home, "away": away,
                "bet_type": "moneyline", "selection": selection, "point": None,
                "price": price, "game_pk": gpk, "game_date": com[:10]}

    def test_pairs_home_away_per_book_at_close(self):
        com = "2024-06-26T23:10:00Z"
        rows = [
            self._ml_row("draftkings", 2, "2024-06-26T23:00:00Z", com, "NYM", +105),
            self._ml_row("draftkings", 2, "2024-06-26T23:00:00Z", com, "NYY", -125),
            self._ml_row("pinnacle", 2, "2024-06-26T23:00:00Z", com, "NYM", +108),
            self._ml_row("pinnacle", 2, "2024-06-26T23:00:00Z", com, "NYY", -120),
        ]
        legs, stats = d.select_team_ml_legs(rows)
        self.assertEqual(len(legs), 1)
        lg = legs[0]
        self.assertEqual((lg.dk_home, lg.dk_away), (105, -125))
        self.assertEqual((lg.pin_home, lg.pin_away), (108, -120))

    def test_incomplete_moneyline_dropped(self):
        com = "2024-06-26T23:10:00Z"
        rows = [  # DK missing the away side
            self._ml_row("draftkings", 2, "2024-06-26T23:00:00Z", com, "NYM", +105),
            self._ml_row("pinnacle", 2, "2024-06-26T23:00:00Z", com, "NYM", +108),
            self._ml_row("pinnacle", 2, "2024-06-26T23:00:00Z", com, "NYY", -120),
        ]
        legs, stats = d.select_team_ml_legs(rows)
        self.assertEqual(legs, [])
        self.assertEqual(stats["legs_dropped_incomplete_moneyline"], 1)


class TeamTriadTests(unittest.TestCase):
    def _tr(self, book, bt, selection, point, price, sid=2,
            cap="2024-06-26T23:00:00Z", com="2024-06-26T23:10:00Z",
            home="NYM", away="NYY", eid="G1", gpk=500):
        return {"book": book, "snapshot_id": sid, "captured_at": cap,
                "commence_time": com, "event_id": eid, "home": home, "away": away,
                "kind": "team", "bet_type": bt, "selection": selection,
                "point": point, "price": price, "game_pk": gpk}

    def test_extracts_all_three_markets(self):
        rows = [
            self._tr("draftkings", "moneyline", "NYM", None, -150),
            self._tr("draftkings", "moneyline", "NYY", None, +130),
            self._tr("draftkings", "spread", "NYM", -1.5, +120),
            self._tr("draftkings", "spread", "NYY", +1.5, -140),
            self._tr("draftkings", "total", "Over", 8.5, -110),
            self._tr("draftkings", "total", "Under", 8.5, -105),
        ]
        triads, stats = d.select_team_triad(rows)
        self.assertEqual(len(triads), 1)
        t = triads[0]
        self.assertEqual((t.ml_home, t.ml_away), (-150, 130))
        self.assertEqual((t.rl_home_point, t.rl_home, t.rl_away), (-1.5, 120, -140))
        self.assertEqual((t.total_line, t.total_over, t.total_under), (8.5, -110, -105))

    def test_incomplete_triad_dropped(self):
        rows = [  # missing the total market
            self._tr("draftkings", "moneyline", "NYM", None, -150),
            self._tr("draftkings", "moneyline", "NYY", None, +130),
            self._tr("draftkings", "spread", "NYM", -1.5, +120),
            self._tr("draftkings", "spread", "NYY", +1.5, -140),
        ]
        triads, stats = d.select_team_triad(rows)
        self.assertEqual(triads, [])
        self.assertEqual(stats["events_dropped_incomplete_triad"], 1)


class F5MLTests(unittest.TestCase):
    def _r(self, book, bt, sel, pt, price, sid=2, cap="2024-06-26T23:00:00Z",
           com="2024-06-26T23:10:00Z", home="Mets", away="Yankees", eid="F1", gpk=700):
        return {"book": book, "snapshot_id": sid, "captured_at": cap,
                "commence_time": com, "event_id": eid, "home": home, "away": away,
                "kind": "first_five", "bet_type": bt, "selection": sel,
                "point": pt, "price": price, "game_pk": gpk}

    def test_extracts_dk_ml_pin_spread0_fd_ml(self):
        rows = [
            self._r("draftkings", "moneyline", "Mets", None, -120),
            self._r("draftkings", "moneyline", "Yankees", None, -110),
            self._r("pinnacle", "spread", "Mets", 0.0, +102),      # 0.0 spread == F5 ML
            self._r("pinnacle", "spread", "Yankees", 0.0, -114),
            self._r("fanduel", "moneyline", "Mets", None, +102),
            self._r("fanduel", "moneyline", "Yankees", None, -128),
        ]
        legs, stats = d.select_f5_ml_legs(rows)
        self.assertEqual(len(legs), 1)
        lg = legs[0]
        self.assertEqual((lg.dk_home, lg.dk_away), (-120, -110))
        self.assertEqual((lg.pin_home, lg.pin_away), (102, -114))
        self.assertEqual((lg.fd_home, lg.fd_away), (102, -128))

    def test_pinnacle_nonzero_spread_ignored(self):
        # A Pinnacle F5 spread NOT at 0.0 is not the ML -> DK+Pin incomplete -> drop.
        rows = [
            self._r("draftkings", "moneyline", "Mets", None, -120),
            self._r("draftkings", "moneyline", "Yankees", None, -110),
            self._r("pinnacle", "spread", "Mets", 1.5, +102),
            self._r("pinnacle", "spread", "Yankees", -1.5, -114),
        ]
        legs, stats = d.select_f5_ml_legs(rows)
        self.assertEqual(legs, [])
        self.assertEqual(stats["legs_dropped_incomplete_dk_pin"], 1)

    def test_fanduel_optional(self):
        rows = [
            self._r("draftkings", "moneyline", "Mets", None, -120),
            self._r("draftkings", "moneyline", "Yankees", None, -110),
            self._r("pinnacle", "spread", "Mets", 0.0, +102),
            self._r("pinnacle", "spread", "Yankees", 0.0, -114),
        ]
        legs, _ = d.select_f5_ml_legs(rows)
        self.assertEqual(len(legs), 1)
        self.assertIsNone(legs[0].fd_home)


class OutcomeValueTests(unittest.TestCase):
    def test_game_pk_exact_hit_and_miss(self):
        idx = {"batter": {("683002", 700): {"H": 2.0, "SO": 1.0}}}
        self.assertEqual(d.outcome_value(idx, "batter_hits", "683002", 700), 2.0)
        self.assertEqual(d.outcome_value(idx, "batter_strikeouts", "683002", 700), 1.0)
        self.assertIsNone(d.outcome_value(idx, "batter_hits", "683002", 999))  # wrong game
        self.assertIsNone(d.outcome_value(idx, "batter_hits", "999", 700))     # wrong player

    def test_none_inputs(self):
        idx = {"batter": {}}
        self.assertIsNone(d.outcome_value(idx, "batter_hits", None, 700))
        self.assertIsNone(d.outcome_value(idx, "batter_hits", "683002", None))

    def test_ip_to_outs_xform(self):
        idx = {"pitcher": {("605483", 700): {"IP": 6.1}}}   # 6.1 IP = 19 outs
        self.assertEqual(d.outcome_value(idx, "pitcher_outs", "605483", 700), 19)


if __name__ == "__main__":
    unittest.main()
