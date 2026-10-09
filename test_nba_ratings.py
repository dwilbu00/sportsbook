"""nba_ratings — as-of pace/efficiency ratings. Guards the leakage rules (a game never sees its
own result; same-date games are simultaneous; ratings_as_of excludes the date), the math
(pace/ORtg/DRtg from decayed sums, prior-season regression, opponent adjustment direction,
project_matchup), and rest days. Synthetic team-game rows; no network/DB.
"""
import unittest

import nba_ratings as R


def _pair(gid, date, home, away, hp, ap, season=2025, poss=100.0, minutes=48.0, **extra):
    """Two team-game rows (canonical warehouse shape) for one game."""
    def row(tid, opp, pts, side):
        r = {"game_id": gid, "team_id": tid, "opponent_team_id": opp, "season": season,
             "season_type": "regular-season", "game_date": date, "home_away": side,
             "game_minutes": minutes, "possessions": poss, "points": pts}
        for s in R.ALLOWED_STATS:
            if s != "points":
                r[s] = extra.get(s, 10.0)
        return r
    return [row(home, away, hp, "home"), row(away, home, ap, "away")]


NOPRIOR = {"prior_games": 0, "league_prior_games": 0, "opp_adjust": False,
           "half_life": 1e9}


class LeakageTests(unittest.TestCase):
    def test_game_never_sees_its_own_result(self):
        base = _pair("g1", "2025-01-01", 1, 2, 110, 100)
        a = _pair("g2", "2025-01-03", 1, 2, 110, 100)
        b = _pair("g2", "2025-01-03", 1, 2, 180, 60)          # wildly different result
        pa = R.pregame_ratings(base + a, NOPRIOR)[("g2", 1)]
        pb = R.pregame_ratings(base + b, NOPRIOR)[("g2", 1)]
        self.assertEqual(pa["ortg"], pb["ortg"])
        self.assertEqual(pa["lg_rtg"], pb["lg_rtg"])

    def test_same_date_games_are_simultaneous(self):
        hist = _pair("g0", "2025-01-01", 3, 4, 100, 100)
        day = _pair("g1", "2025-01-02", 3, 4, 100, 100)
        other = _pair("g2", "2025-01-02", 1, 2, 150, 140)      # same date, different teams
        with_other = R.pregame_ratings(hist + day + other, NOPRIOR)[("g1", 3)]
        without = R.pregame_ratings(hist + day, NOPRIOR)[("g1", 3)]
        self.assertEqual(with_other["lg_rtg"], without["lg_rtg"])   # league avg unaffected
        self.assertEqual(with_other["lg_pace"], without["lg_pace"])

    def test_ratings_as_of_excludes_the_date(self):
        rows = (_pair("g1", "2025-01-01", 1, 2, 100, 100)
                + _pair("g2", "2025-01-05", 1, 2, 140, 100))
        r = R.ratings_as_of(rows, "2025-01-05", 2025, team_ids=[1], params=NOPRIOR)[1]
        self.assertAlmostEqual(r["ortg_raw"], 100.0)            # only g1 counted
        self.assertEqual(r["n"], 1)


class MathTests(unittest.TestCase):
    def test_pace_and_ratings_from_sums(self):
        rows = (_pair("g1", "2025-01-01", 1, 2, 110, 100, poss=100.0)
                + _pair("g2", "2025-01-02", 1, 3, 120, 90, poss=96.0, minutes=53.0))
        r = R.ratings_as_of(rows, "2025-01-03", 2025, team_ids=[1], params=NOPRIOR)[1]
        self.assertAlmostEqual(r["pace_raw"], 48.0 * 196.0 / 101.0)   # OT minutes count
        self.assertAlmostEqual(r["ortg_raw"], 100.0 * 230.0 / 196.0)
        self.assertAlmostEqual(r["drtg_raw"], 100.0 * 190.0 / 196.0)
        self.assertAlmostEqual(r["allowed100"]["assists"], 100.0 * 20.0 / 196.0)

    def test_decay_weights_recent_games_more(self):
        rows = (_pair("g1", "2025-01-01", 1, 2, 90, 100)
                + _pair("g2", "2025-01-02", 1, 2, 130, 100))
        fast = R.ratings_as_of(rows, "2025-01-03", 2025, [1],
                               {**NOPRIOR, "half_life": 1.0})[1]
        flat = R.ratings_as_of(rows, "2025-01-03", 2025, [1], NOPRIOR)[1]
        self.assertAlmostEqual(flat["ortg_raw"], 110.0)
        self.assertGreater(fast["ortg_raw"], 110.0)              # leans toward the 130 game

    def test_prior_season_regressed_toward_league(self):
        # season 2024: team 1 scores 120, team 2 scores 100. The 2025 opener must start each
        # team at last season's final RELATIVE rating, regressed half-way to 1.0.
        params = {"prior_games": 10, "league_prior_games": 60, "regress": 0.5,
                  "opp_adjust": False, "half_life": 1e9}
        walk = R.RatingsWalk(params)
        for k in range(10):
            g = _pair(f"p{k}", f"2024-01-{k + 1:02d}", 1, 2, 120, 100, season=2024)
            walk.feed(f"2024-01-{k + 1:02d}", [(2024, f"p{k}", g[0], g[1])])
        prev_rel = walk.final[(2024, 1)]["ortg"] / walk.league_final[2024]["rtg"]
        self.assertGreater(prev_rel, 1.05)                       # team 1 was clearly better
        r = walk.rating(2025, 1, "2024-11-01")
        self.assertEqual(r["n"], 0)                              # no 2025 games yet
        self.assertAlmostEqual(r["ortg"] / r["lg_rtg"], 1 + 0.5 * (prev_rel - 1), places=6)

    def test_opponent_adjustment_direction(self):
        # team 1 = elite defense (allows 90), team 2 = poor defense (allows 110). Team 3 scores
        # the SAME 100 against one or the other: raw ORtg is identical, but the adjusted ORtg
        # must be higher after facing the elite defense.
        hist = []
        for k in range(8):
            hist += _pair(f"a{k}", f"2025-01-{k + 1:02d}", 1, 2, 110, 90)
        params = {"prior_games": 0, "league_prior_games": 0, "half_life": 1e9,
                  "opp_adjust": True}
        vs_elite = R.ratings_as_of(hist + _pair("x", "2025-01-20", 3, 1, 100, 100),
                                   "2025-01-21", 2025, [3], params)[3]
        vs_poor = R.ratings_as_of(hist + _pair("x", "2025-01-20", 3, 2, 100, 100),
                                  "2025-01-21", 2025, [3], params)[3]
        self.assertAlmostEqual(vs_elite["ortg_raw"], vs_poor["ortg_raw"])
        self.assertGreater(vs_elite["ortg"], vs_elite["ortg_raw"])   # scaled up vs elite D
        self.assertLess(vs_poor["ortg"], vs_poor["ortg_raw"])        # scaled down vs poor D

    def test_project_matchup(self):
        h = {"pace": 102.0, "ortg": 118.0, "drtg": 110.0, "lg_pace": 100.0, "lg_rtg": 114.0}
        a = {"pace": 98.0, "ortg": 112.0, "drtg": 116.0, "lg_pace": 100.0, "lg_rtg": 114.0}
        pj = R.project_matchup(h, a, hca=2.0)
        poss = 102.0 * 98.0 / 100.0
        self.assertAlmostEqual(pj["possessions"], poss)
        self.assertAlmostEqual(pj["home_pts"], poss * (118 * 116 / 114) / 100 + 1.0)
        self.assertAlmostEqual(pj["away_pts"], poss * (112 * 110 / 114) / 100 - 1.0)
        self.assertAlmostEqual(pj["margin"], pj["home_pts"] - pj["away_pts"])

    def test_rest_days(self):
        rows = (_pair("g1", "2025-01-01", 1, 2, 100, 100)
                + _pair("g2", "2025-01-02", 1, 3, 100, 100)
                + _pair("g3", "2025-01-05", 1, 2, 100, 100))
        pre = R.pregame_ratings(rows, NOPRIOR)
        self.assertIsNone(pre[("g1", 1)]["rest_days"])
        self.assertEqual(pre[("g2", 1)]["rest_days"], 1)        # back-to-back
        self.assertEqual(pre[("g3", 1)]["rest_days"], 3)
        self.assertEqual(pre[("g3", 2)]["rest_days"], 4)

    def test_unpaired_game_ignored(self):
        rows = _pair("g1", "2025-01-01", 1, 2, 100, 100)[:1]    # one row only
        self.assertEqual(R.pregame_ratings(rows, NOPRIOR), {})


if __name__ == "__main__":
    unittest.main()
