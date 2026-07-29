"""Tests for bet_selector.select_top_bets (auto-pick top-N value bets).

Pure synthetic pools with hand-set event_id. Pool entries mirror what
app._iter_wager_candidates yields: (selection_key, bet_type, side, candidate),
bet_type in {moneyline, spread, total, player_prop}, side in {OVER, UNDER, None}
(props carry their direction on the candidate, not in `side`).
"""

import unittest

import bet_selector


# ── pool-entry factories ───────────────────────────────────────────────────

def _prop(key, event_id, team, prop, direction, ev,
          edge=5.0, over_rate=60.0, batting_order=None, player=None):
    return (key, "player_prop", None, {
        "event_id": event_id, "team": team, "player": player or key,
        "prop": prop, "direction": direction,
        "expected_roi_pct": ev, "edge_pct": edge, "over_rate": over_rate,
        "batting_order": batting_order, "games_sampled": 20,
        "matchup": "AWAY @ HOME",
    })


def _total(key, event_id, side, ev, edge=5.0, over_hit_rate=55.0):
    return (key, "total", side, {
        "event_id": event_id, "matchup": "AWAY @ HOME",
        "over_expected_roi_pct": ev if side == "OVER" else None,
        "under_expected_roi_pct": ev if side == "UNDER" else None,
        "over_edge_pct": edge if side == "OVER" else None,
        "under_edge_pct": edge if side == "UNDER" else None,
        "over_hit_rate": over_hit_rate,
    })


def _ml(key, event_id, team, ev, edge=5.0, prob=60.0):
    return (key, "moneyline", None, {
        "event_id": event_id, "team": team,
        "blended_prob": prob, "edge_pct": edge, "expected_roi_pct": ev,
    })


def _spread(key, event_id, team, ev, edge=5.0, cover=55.0):
    return (key, "spread", None, {
        "event_id": event_id, "team": team,
        "cover_rate": cover, "edge_pct": edge, "expected_roi_pct": ev,
        "games_sampled": 20,
    })


MLB = "baseball_mlb"
NBA = "basketball_nba"


class RankingTests(unittest.TestCase):

    def test_rank_by_ev_and_cap(self):
        # Distinct games → no conflicts; top-2 by EV, best first.
        pool = [_ml("k1", "g1", "A", 3.0),
                _ml("k2", "g2", "B", 10.0),
                _ml("k3", "g3", "C", 7.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 2, "ev"), ["k2", "k3"])

    def test_n_larger_than_pool_returns_all_ranked(self):
        pool = [_ml("k1", "g1", "A", 3.0),
                _ml("k2", "g2", "B", 10.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 10, "ev"), ["k2", "k1"])

    def test_empty_pool_and_zero_n(self):
        self.assertEqual(bet_selector.select_top_bets([], MLB, 5, "ev"), [])
        pool = [_ml("k1", "g1", "A", 3.0)]
        self.assertEqual(bet_selector.select_top_bets(pool, MLB, 0, "ev"), [])

    def test_metric_prob(self):
        # 'a' has lower EV but higher win prob → wins under the prob metric.
        pool = [_ml("a", "g1", "A", ev=1.0, prob=80.0),
                _ml("b", "g2", "B", ev=20.0, prob=51.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 1, "prob"), ["a"])

    def test_metric_edge(self):
        pool = [_ml("a", "g1", "A", ev=1.0, edge=30.0),
                _ml("b", "g2", "B", ev=20.0, edge=2.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 1, "edge"), ["a"])

    def test_metric_balanced_prefers_all_round(self):
        # 'bal' is 2nd on EV and 2nd on prob → lowest Borda sum, beats the
        # one-dimensional extremes and the all-round-bad bet.
        pool = [_ml("bal", "g1", "A", ev=9.0, prob=75.0),
                _ml("hev", "g2", "B", ev=20.0, prob=30.0),
                _ml("hpr", "g3", "C", ev=2.0, prob=95.0),
                _ml("bad", "g4", "D", ev=3.0, prob=35.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 1, "balanced"), ["bal"])

    def test_totals_use_side_specific_ev(self):
        # OVER reads over_expected_roi_pct (2), UNDER reads under_ (15).
        pool = [_total("o", "g1", "OVER", ev=2.0),
                _total("u", "g2", "UNDER", ev=15.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 1, "ev"), ["u"])


class StructuralRuleTests(unittest.TestCase):
    """L1 hard conflicts + L2 anti-correlation (all sports)."""

    def test_l1_opposite_moneyline_same_game_blocked(self):
        pool = [_ml("a", "g1", "A", 10.0), _ml("b", "g1", "B", 9.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 5, "ev"), ["a"])

    def test_l2_mlb_pitcher_k_over_plus_total_over_blocked(self):
        # pitcher_strikeouts OVER + game-total OVER = -0.30 correlation.
        pool = [_prop("k", "g1", "A", "pitcher_strikeouts", "OVER", 10.0),
                _total("to", "g1", "OVER", 9.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 5, "ev"), ["k"])

    def test_l2_nba_same_team_overs_blocked(self):
        # Two same-team player overs in NBA = -0.20 (shared usage cap).
        pool = [_prop("p1", "g1", "LAL", "player_points", "OVER", 10.0),
                _prop("p2", "g1", "LAL", "player_rebounds", "OVER", 9.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, NBA, 5, "ev"), ["p1"])

    def test_cross_game_not_blocked(self):
        # Same structural pair but different games → both allowed.
        pool = [_ml("a", "g1", "A", 10.0), _ml("b", "g2", "B", 9.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 5, "ev"), ["a", "b"])


class RuleOfThreeTests(unittest.TestCase):
    """Rule (a): <= 3 batter_hits OVER per team; team=None exempt."""

    def test_fourth_same_team_hits_over_excluded(self):
        pool = [_prop(f"h{i}", "g1", "A", "batter_hits", "OVER", 10.0 - i,
                      player=f"P{i}") for i in range(4)]
        res = bet_selector.select_top_bets(pool, MLB, 5, "ev")
        self.assertEqual(res, ["h0", "h1", "h2"])

    def test_across_teams_allowed(self):
        pool = ([_prop(f"a{i}", "g1", "A", "batter_hits", "OVER", 20.0 - i,
                       player=f"A{i}") for i in range(3)]
                + [_prop(f"b{i}", "g1", "B", "batter_hits", "OVER", 10.0 - i,
                         player=f"B{i}") for i in range(3)])
        res = bet_selector.select_top_bets(pool, MLB, 6, "ev")
        self.assertEqual(len(res), 6)

    def test_team_none_exempt(self):
        pool = [_prop(f"h{i}", "g1", None, "batter_hits", "OVER", 10.0 - i,
                      player=f"P{i}") for i in range(5)]
        res = bet_selector.select_top_bets(pool, MLB, 5, "ev")
        self.assertEqual(len(res), 5)


class PitcherVsHitterUnderTests(unittest.TestCase):
    """Rule (b): pitcher_* UNDER + opposing batter_hits UNDER blocked."""

    def test_opposing_under_pair_blocked(self):
        pool = [_prop("pu", "g1", "A", "pitcher_strikeouts", "UNDER", 10.0),
                _prop("hu", "g1", "B", "batter_hits", "UNDER", 9.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 5, "ev"), ["pu"])

    def test_same_team_under_pair_allowed(self):
        pool = [_prop("pu", "g1", "A", "pitcher_strikeouts", "UNDER", 10.0),
                _prop("hu", "g1", "A", "batter_hits", "UNDER", 9.0)]
        self.assertEqual(
            len(bet_selector.select_top_bets(pool, MLB, 5, "ev")), 2)

    def test_different_game_under_pair_allowed(self):
        pool = [_prop("pu", "g1", "A", "pitcher_strikeouts", "UNDER", 10.0),
                _prop("hu", "g2", "B", "batter_hits", "UNDER", 9.0)]
        self.assertEqual(
            len(bet_selector.select_top_bets(pool, MLB, 5, "ev")), 2)


class OverUnderMixTests(unittest.TestCase):
    """Rule (c): game-total OVER + player-prop UNDER same game blocked."""

    def test_total_over_plus_prop_under_same_game_blocked(self):
        pool = [_total("to", "g1", "OVER", 10.0),
                _prop("pu", "g1", "A", "batter_hits", "UNDER", 9.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 5, "ev"), ["to"])

    def test_total_over_plus_prop_under_cross_game_allowed(self):
        pool = [_total("to", "g1", "OVER", 10.0),
                _prop("pu", "g2", "A", "batter_hits", "UNDER", 9.0)]
        self.assertEqual(
            len(bet_selector.select_top_bets(pool, MLB, 5, "ev")), 2)


class BattingOrderTests(unittest.TestCase):
    """Rule (d): batter_hits OVER excluded only for confirmed slot > 4."""

    def test_confirmed_slot_5_excluded(self):
        pool = [_prop("h", "g1", "A", "batter_hits", "OVER", 10.0,
                      batting_order=5)]
        self.assertEqual(bet_selector.select_top_bets(pool, MLB, 5, "ev"), [])

    def test_slot_3_allowed(self):
        pool = [_prop("h", "g1", "A", "batter_hits", "OVER", 10.0,
                      batting_order=3)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 5, "ev"), ["h"])

    def test_unconfirmed_none_allowed_fail_open(self):
        pool = [_prop("h", "g1", "A", "batter_hits", "OVER", 10.0,
                      batting_order=None)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 5, "ev"), ["h"])

    def test_non_mlb_ignores_batting_order(self):
        # Non-MLB → the whole L3 layer is skipped even with a confirmed slot.
        pool = [_prop("h", "g1", "A", "batter_hits", "OVER", 10.0,
                      batting_order=5)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, NBA, 5, "ev"), ["h"])


class DoubleheaderTests(unittest.TestCase):

    def test_same_matchup_different_event_id_are_separate_games(self):
        # Opposite MLs that would conflict if keyed on matchup, but the two DH
        # games have distinct event_ids → both survive.
        pool = [_ml("a", "dh1", "A", 10.0), _ml("b", "dh2", "B", 9.0)]
        self.assertEqual(
            bet_selector.select_top_bets(pool, MLB, 5, "ev"), ["a", "b"])


class EarnedRunsStrikeoutsConflictTests(unittest.TestCase):
    """ER-over + K-over on the same pitcher are self-cancelling (negatively
    correlated); the shared _pair_correlation entry must block co-selection."""

    def test_same_pitcher_er_over_and_k_over_not_both_selected(self):
        pool = [
            _prop("er", "e1", "H", "pitcher_earned_runs", "OVER", 12.0,
                  player="P"),
            _prop("k", "e1", "H", "pitcher_strikeouts", "OVER", 11.0,
                  player="P"),
        ]
        picks = bet_selector.select_top_bets(pool, MLB, 5, "ev")
        self.assertEqual(picks, ["er"])   # higher EV kept; K dropped as conflict

    def test_different_pitchers_er_and_k_both_allowed(self):
        pool = [
            _prop("er", "e1", "H", "pitcher_earned_runs", "OVER", 12.0,
                  player="P"),
            _prop("k", "e1", "A", "pitcher_strikeouts", "OVER", 11.0,
                  player="Q"),
        ]
        self.assertEqual(
            sorted(bet_selector.select_top_bets(pool, MLB, 5, "ev")), ["er", "k"])


if __name__ == "__main__":
    unittest.main()
