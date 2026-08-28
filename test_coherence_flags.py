"""Unit tests for coherence_flags.flag_games — the pure daily-flag selection."""
import unittest

import coherence_flags as cf
import r2_data


def _triad(ml_home=-150, ml_away=+130, rl_point=-1.5, rl_home=+120, rl_away=-140,
           total_over=-110, total_under=-110, total_line=8.5):
    return r2_data.TeamTriad(
        event_id="G1", game_date="2026-08-27", commence_time="x", snapshot_id=2,
        game_pk=500, home="Mets", away="Yankees", ml_home=ml_home, ml_away=ml_away,
        rl_home_point=rl_point, rl_home=rl_home, rl_away=rl_away,
        total_line=total_line, total_over=total_over, total_under=total_under)


class FlagGamesTests(unittest.TestCase):
    def test_flags_have_actionable_fields(self):
        # Low floor so at least one side flags; check the record is bettable.
        flags = cf.flag_games([_triad()], offset=0.0, ev_floor=-10.0)
        self.assertTrue(flags)
        f = flags[0]
        for k in ("side", "team", "point", "dk_price", "ev", "coherent_fair"):
            self.assertIn(k, f)
        self.assertIn(f["side"], ("home", "away"))
        # away side carries the mirrored run-line point
        away = [x for x in flags if x["side"] == "away"]
        home = [x for x in flags if x["side"] == "home"]
        if away and home:
            self.assertAlmostEqual(away[0]["point"], -home[0]["point"], places=9)

    def test_high_floor_flags_nothing(self):
        self.assertEqual(cf.flag_games([_triad()], offset=0.0, ev_floor=0.95), [])

    def test_offset_shifts_fair(self):
        # A positive offset lowers implied home cover -> raises the away-side fair by
        # the same amount. Assert the calibrated fair actually moves with the offset.
        base = cf.flag_games([_triad()], offset=0.0, ev_floor=-10.0)
        shifted = cf.flag_games([_triad()], offset=0.10, ev_floor=-10.0)
        b_away = next(f for f in base if f["side"] == "away")
        s_away = next(f for f in shifted if f["side"] == "away")
        self.assertAlmostEqual(s_away["coherent_fair"] - b_away["coherent_fair"],
                               0.10, places=3)

    def test_sorted_by_ev_desc(self):
        flags = cf.flag_games([_triad(), _triad(rl_home=+180, rl_away=-220)],
                              offset=0.0, ev_floor=-10.0)
        evs = [f["ev"] for f in flags]
        self.assertEqual(evs, sorted(evs, reverse=True))


if __name__ == "__main__":
    unittest.main()
