"""test_nfl_signs.py — SIGN + direction guards for the NFL EPA/margin/totals code.

Two defensive-EPA sign bugs shipped this cycle (nfl_totals collapsed to net-EPA and
inverted the defense effect; nfl_power's opponent adjustment ran the offense update
the wrong way). def_epa is EPA ALLOWED, so a good defense is NEGATIVE — an easy sign
to flip. These tests assert the *direction* of every def-sign-dependent computation
on real data, so a future flip fails loudly instead of silently degrading the model.

Runs off the committed mirror (2024). Skips cleanly if the mirror is absent.
Run:  python test_nfl_signs.py   (or: python -m unittest test_nfl_signs)
"""
import unittest

import nfl_epa
import nfl_totals

SEASON = 2024


def _corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def _have_pbp():
    return bool(nfl_epa.load_plays(SEASON))


class NflSignTests(unittest.TestCase):

    # ── totals expected-scoring signal ────────────────────────────────────────
    def test_totals_signal_rises_with_weaker_opponent_defense(self):
        """expected_total_signal must INCREASE as an opposing defense weakens
        (def_epa allowed rises). This is the exact bug that shipped in nfl_totals."""
        base = nfl_totals.expected_total_signal(0.0, 0.0, 0.0, 0.0)
        weaker_away_d = nfl_totals.expected_total_signal(0.0, 0.0, 0.0, 0.10)
        self.assertGreater(weaker_away_d, base,
                           "weaker away defense (higher def_epa) must raise expected total")
        better_home_o = nfl_totals.expected_total_signal(0.10, 0.0, 0.0, 0.0)
        self.assertGreater(better_home_o, base,
                           "better home offense must raise expected total")

    # ── team_epa net-EPA orientation ──────────────────────────────────────────
    def test_net_epa_predicts_margin_positively(self):
        """A higher home-minus-away net_epa must go with a larger home margin. A flip
        in the off/def/net sign convention would drive this correlation negative."""
        if not _have_pbp():
            self.skipTest("no pbp mirror")
        ratings = nfl_epa.team_epa(SEASON, prior_shrink=False)
        plays = nfl_epa.load_plays(SEASON)
        games = {p["game_id"]: p for p in plays if "home_score" in p}
        edges, margins = [], []
        for p in games.values():
            h, a = ratings.get(p["home_team"]), ratings.get(p["away_team"])
            if not h or not a:
                continue
            edges.append(h["net_epa"] - a["net_epa"])
            margins.append(p["home_score"] - p["away_score"])
        c = _corr(edges, margins)
        self.assertGreater(c, 0.2, f"net_epa edge vs margin corr={c:.3f} (sign inverted?)")

    def test_def_epa_is_allowed_convention(self):
        """def_epa is EPA ALLOWED: a team's def_epa should correlate POSITIVELY with
        points it gives up. Higher def_epa = weaker defense."""
        if not _have_pbp():
            self.skipTest("no pbp mirror")
        ratings = nfl_epa.team_epa(SEASON, prior_shrink=False)
        plays = nfl_epa.load_plays(SEASON)
        # accumulate points allowed per team from unique games
        seen = set()
        pts_allowed = {}
        for p in plays:
            if "home_score" not in p or p["game_id"] in seen:
                continue
            seen.add(p["game_id"])
            pts_allowed[p["home_team"]] = pts_allowed.get(p["home_team"], 0) + p["away_score"]
            pts_allowed[p["away_team"]] = pts_allowed.get(p["away_team"], 0) + p["home_score"]
        teams = [t for t in ratings if t in pts_allowed]
        c = _corr([ratings[t]["def_epa"] for t in teams],
                  [pts_allowed[t] for t in teams])
        self.assertGreater(c, 0.3, f"def_epa vs points-allowed corr={c:.3f} "
                           "(def_epa should be 'allowed': higher = more points given up)")

    # ── opponent-adjustment must refine, not invert ───────────────────────────
    def test_power_adjustment_does_not_invert(self):
        """Opponent-adjusted net must stay strongly POSITIVELY correlated with raw net
        — the adjustment refines strength-of-schedule, it must not flip the ranking
        (the nfl_power offense-update sign bug tanked this)."""
        try:
            import nfl_power
        except Exception:
            self.skipTest("nfl_power unavailable")
        if not _have_pbp():
            self.skipTest("no pbp mirror")
        raw = nfl_epa.team_epa(SEASON, prior_shrink=False)
        adj = nfl_power.team_power(SEASON, prior_shrink=False)
        teams = [t for t in raw if t in adj]
        c = _corr([raw[t]["net_epa"] for t in teams], [adj[t]["net_epa"] for t in teams])
        self.assertGreater(c, 0.7, f"raw vs opponent-adjusted net corr={c:.3f} "
                           "(adjustment inverted the ranking — sign bug?)")

    def test_power_def_sign_preserved(self):
        """The best raw defenses (lowest def_epa) must remain below-average after
        opponent adjustment — the adjustment must not flip defensive quality."""
        try:
            import nfl_power
        except Exception:
            self.skipTest("nfl_power unavailable")
        if not _have_pbp():
            self.skipTest("no pbp mirror")
        raw = nfl_epa.team_epa(SEASON, prior_shrink=False)
        adj = nfl_power.team_power(SEASON, prior_shrink=False)
        best_raw = sorted(raw, key=lambda t: raw[t]["def_epa"])[:8]  # 8 best raw D
        adj_mean = sum(v["def_epa"] for v in adj.values()) / len(adj)
        better = sum(1 for t in best_raw if t in adj and adj[t]["def_epa"] < adj_mean)
        self.assertGreaterEqual(better, 6, f"only {better}/8 top raw defenses stay "
                                "below-average after adjustment (sign preserved?)")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
