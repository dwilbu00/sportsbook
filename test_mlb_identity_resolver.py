"""P6 GAME-CONTEXT-FIRST MLB identity (mlb_starters.resolve_mlbam_id).

The single StatsAPI-native resolver: today's posted lineup / announced probables
(role-partitioned, trade-aware, namesake-safe) → season-roster unique-exact →
role-verified SFBB fallback. These tests pin the caches + SFBB seam so no network or
DB is touched; they assert the resolution ORDER and the two behaviors the rework
exists for: (1) trade-safety — a uniquely-named player resolves by id regardless of
today's team; (2) drift rejection — the SFBB pitcher-id-for-a-batter-prop bug
(Luis Garcia Jr. 677651 vs 671277) can never bind.
"""

import unittest
from unittest import mock

import mlb_starters as m


SEASON = 2026


def _lineup(players):
    """players: {norm_name: (mlbam_id, side)} → the get_confirmed_lineup shape."""
    return {"players": {k: {"player_id": pid, "side": side, "batting_order": 1}
                        for k, (pid, side) in players.items()},
            "home_confirmed": True, "away_confirmed": True}


def _probables(by_team):
    """by_team: {norm_team: (pitcher_id, fullName)} → get_probable_starters shape."""
    return {t: {"pitcher_id": pid, "name": nm, "team_id": 0}
            for t, (pid, nm) in by_team.items()}


class ResolveMlbamIdTests(unittest.TestCase):
    def setUp(self):
        # Season roster (_player_index) + inverted is_pitcher index (_is_pitcher_index /
        # _resolve_is_pitcher) primed so nothing hits the network.
        m._PLAYER_INDEX_CACHE[SEASON] = {
            "aaron judge": [(592450, False)],
            "traded slugger": [(555000, False)],          # unique → roster tier
            "callup batter": [(555111, False)],            # unique → roster tier (F1)
            "luis garcia jr": [(671277, False)],           # statsapi KEEPS the suffix
            "luis garcia": [(671277, False), (677651, True)],  # true same-spell namesake
        }
        m._PITCHER_BY_ID_CACHE[SEASON] = {
            "592450": False, "555000": False, "555111": False, "555222": False,
            "671277": False, "677651": True, "111": True,
        }

    def tearDown(self):
        m._PLAYER_INDEX_CACHE.pop(SEASON, None)
        m._PITCHER_BY_ID_CACHE.pop(SEASON, None)

    def _no_sfbb(self):
        return mock.patch.object(m, "_player_id_map", return_value=None)

    def _sfbb(self, mlb_id):
        pim = mock.Mock()
        pim.mlb_id_for_name.return_value = mlb_id
        pim.get_row.return_value = None
        return mock.patch.object(m, "_player_id_map", return_value=pim)

    # ── Tier 1: today's posted game (authoritative, trade-aware, namesake-safe) ──

    def test_batter_from_lineup_beats_pitcher_namesake(self):
        # The Luis Garcia Jr. fix: batter prop binds the LINEUP id (671277), never the
        # probable pitcher's id (677651), even though both share the base name.
        lineup = _lineup({"luis garcia jr": (671277, "home")})
        probs = _probables({"new york yankees": (677651, "Luis García")})
        with self._no_sfbb():
            self.assertEqual(
                m.resolve_mlbam_id("Luis Garcia Jr.", SEASON, prop_key="batter_hits",
                                   teams=["New York Yankees", "Boston Red Sox"],
                                   confirmed_lineup=lineup, probable_starters=probs),
                (671277, False))

    def test_lineup_tolerates_dropped_suffix(self):
        # Odds feed drops "Jr." but StatsAPI keeps it → suffix-stripped fallback hits.
        lineup = _lineup({"luis garcia jr": (671277, "home")})
        with self._no_sfbb():
            self.assertEqual(
                m.resolve_mlbam_id("Luis Garcia", SEASON, prop_key="batter_hits",
                                   teams=["New York Yankees"],
                                   confirmed_lineup=lineup),
                (671277, False))

    def test_pitcher_from_probables_scoped_to_matchup(self):
        probs = _probables({"new york yankees": (677651, "Luis García")})
        with self._no_sfbb():
            self.assertEqual(
                m.resolve_mlbam_id("Luis Garcia", SEASON,
                                   prop_key="pitcher_strikeouts",
                                   teams=["New York Yankees", "Boston Red Sox"],
                                   probable_starters=probs),
                (677651, True))                    # is_pitcher forced True for a probable

    def test_probables_not_matched_off_the_matchup_teams(self):
        # A same-named probable pitcher in a DIFFERENT game must not bind.
        probs = _probables({"chicago cubs": (677651, "Luis García")})
        with self._no_sfbb():
            self.assertIsNone(
                m.resolve_mlbam_id("Luis Garcia", SEASON,
                                   prop_key="pitcher_strikeouts",
                                   teams=["New York Yankees", "Boston Red Sox"],
                                   probable_starters=probs))

    def test_probables_bind_across_divergent_team_names(self):
        # F7: probables are keyed by the StatsAPI team name ("oakland athletics") but
        # the matchup hint carries the ODDS short name ("Athletics"). Tolerant matching
        # (not a strict probs.get) still binds the authoritative probable rather than
        # silently no-op'ing and falling through to the drift-prone SFBB tier.
        probs = _probables({"oakland athletics": (111, "Ace Pitcher")})
        with self._no_sfbb():
            self.assertEqual(
                m.resolve_mlbam_id("Ace Pitcher", SEASON,
                                   prop_key="pitcher_strikeouts",
                                   teams=["Athletics", "New York Yankees"],
                                   probable_starters=probs),
                (111, True))

    # ── Tier 2: season-roster unique-exact (trade-safe, pre-lineup) ──

    def test_trade_safe_unique_name_resolves_without_context(self):
        # No lineup/probables (analyzed early) and a DIFFERENT team than any history:
        # a uniquely-named player still resolves by id (history-by-id then sweeps all
        # games regardless of the team each was played for).
        with self._no_sfbb():
            self.assertEqual(
                m.resolve_mlbam_id("Traded Slugger", SEASON, prop_key="batter_hits",
                                   teams=["Some New Team", "Opponent"]),
                (555000, False))

    def test_suffix_keeps_namesake_distinct_in_roster(self):
        # Pre-lineup: "luis garcia jr" is a UNIQUE roster key (statsapi keeps suffix),
        # so it resolves to the batter (671277) without any game context — the drift
        # only ever came from SFBB storing the suffix-stripped name.
        with self._no_sfbb():
            self.assertEqual(
                m.resolve_mlbam_id("Luis Garcia Jr.", SEASON, prop_key="batter_hits"),
                (671277, False))

    def test_roster_role_mismatch_falls_through(self):
        # A unique roster hit whose role contradicts the prop is not returned from
        # tier 2 (falls through to SFBB, here absent → None).
        with self._no_sfbb():
            self.assertIsNone(
                m.resolve_mlbam_id("Traded Slugger", SEASON,
                                   prop_key="pitcher_strikeouts"))

    def test_tier2_defers_to_team_aware_sfbb_on_same_role_conflict(self):
        # F1: the season roster is teams-BLIND. A unique index hit (555111) that a
        # team-hinted SFBB lookup contradicts with a DIFFERENT SAME-ROLE id (555222)
        # is deferred to the team-aware SFBB tier — the ~weekly index can miss a
        # recent add while a namesake is its sole roster entry.
        with self._sfbb(555222):
            self.assertEqual(
                m.resolve_mlbam_id("Callup Batter", SEASON, prop_key="batter_hits",
                                   teams=["New York Yankees", "Boston Red Sox"]),
                (555222, False))

    def test_tier2_kept_when_sfbb_conflict_is_cross_role(self):
        # Counter-case to F1: a CROSS-role SFBB disagreement (a pitcher id for a
        # batter prop) is the very drift the index is meant to beat, so it does NOT
        # defer — the unique roster batter (555111) is kept, not the SFBB pitcher.
        with self._sfbb(677651):
            self.assertEqual(
                m.resolve_mlbam_id("Callup Batter", SEASON, prop_key="batter_hits",
                                   teams=["New York Yankees", "Boston Red Sox"]),
                (555111, False))

    def test_tier2_kept_when_no_team_hint(self):
        # The deferral is team-hint gated: with no ``teams`` there is nothing to make
        # the SFBB lookup more trustworthy than the index, so the unique roster hit
        # stands even against a different same-role SFBB id.
        with self._sfbb(555222):
            self.assertEqual(
                m.resolve_mlbam_id("Callup Batter", SEASON, prop_key="batter_hits"),
                (555111, False))

    # ── Tier 3: SFBB fallback, role-verified (never trusted blind) ──

    def test_sfbb_drift_rejected_on_role_contradiction(self):
        # No context; "luis garcia" is an ambiguous roster name (tier 2 skips); SFBB
        # drifts to the PITCHER id (677651) for a BATTER prop → rejected (the bug).
        with self._sfbb(677651):
            self.assertIsNone(
                m.resolve_mlbam_id("Luis Garcia", SEASON, prop_key="batter_hits",
                                   teams=["New York Yankees", "Boston Red Sox"]))

    def test_sfbb_accepted_when_role_matches(self):
        # Ambiguous roster name, no context, SFBB resolves to a role-consistent id.
        with self._sfbb(111):
            self.assertEqual(
                m.resolve_mlbam_id("Luis Garcia", SEASON,
                                   prop_key="pitcher_strikeouts",
                                   teams=["New York Yankees"]),
                (111, True))

    def test_sfbb_role_agnostic_when_prop_key_none(self):
        # find_player_id contract: no prop_key → role-agnostic, SFBB accepted as-is
        # (the caller role-gates the tuple).
        with self._sfbb(677651):
            self.assertEqual(
                m.resolve_mlbam_id("Luis Garcia", SEASON, prop_key=None),
                (677651, True))

    def test_sfbb_unverifiable_role_rejected_for_role_known_prop(self):
        # F2: an SFBB id ABSENT from the season index (role unverifiable) with no SFBB
        # position is rejected for a role-known prop. An unverifiable role is exactly
        # the drift risk — a batter prop could otherwise bind an unroster'd pitcher id
        # whose role can never be contradicted. (999999 is in neither primed cache;
        # _sfbb's get_row returns None → no ALLPOS to fall back on.)
        with self._sfbb(999999):
            self.assertIsNone(
                m.resolve_mlbam_id("Luis Garcia", SEASON, prop_key="batter_hits",
                                   teams=["New York Yankees", "Boston Red Sox"]))

    def test_sfbb_role_verified_via_row_allpos_when_unrostered(self):
        # Complement to F2: an unrostered SFBB id whose role CAN be established from the
        # SFBB row's ALLPOS is accepted when it matches the prop — the fallback only
        # rejects when the role is genuinely unknowable, not merely absent from the
        # (~weekly) statsapi roster snapshot.
        pim = mock.Mock()
        pim.mlb_id_for_name.return_value = 999999      # unrostered
        pim.get_row.return_value = {"allpos": "SS/2B"}  # batter positions
        with mock.patch.object(m, "_player_id_map", return_value=pim):
            self.assertEqual(
                m.resolve_mlbam_id("Luis Garcia", SEASON, prop_key="batter_hits",
                                   teams=["New York Yankees"]),
                (999999, False))

    # ── Role-agnostic context (find_player_id path) ──

    def test_role_agnostic_defers_same_spelling_cross_role(self):
        # Same base name in BOTH lineup (batter) and probables (pitcher), no prop_key:
        # ambiguous → tier 1 defers; roster "luis garcia" is also ambiguous → tier 2
        # skips; no SFBB → None (never a coin-flip bind).
        lineup = _lineup({"luis garcia": (671277, "home")})
        probs = _probables({"new york yankees": (677651, "Luis Garcia")})
        with self._no_sfbb():
            self.assertIsNone(
                m.resolve_mlbam_id("Luis Garcia", SEASON, prop_key=None,
                                   teams=["New York Yankees"],
                                   confirmed_lineup=lineup, probable_starters=probs))

    # ── find_player_id delegation + robustness ──

    def test_find_player_id_delegates_and_keeps_tuple_contract(self):
        with self._no_sfbb():
            self.assertEqual(
                m.find_player_id("Aaron Judge", SEASON), (592450, False))

    def test_find_player_id_forwards_game_context(self):
        lineup = _lineup({"luis garcia jr": (671277, "home")})
        probs = _probables({"new york yankees": (111, "Somebody")})
        with mock.patch.object(m, "resolve_mlbam_id",
                               return_value=(671277, False)) as r:
            m.find_player_id("Luis Garcia Jr.", SEASON,
                             teams=["NYY"], lineup=lineup, probables=probs)
        _, kw = r.call_args
        self.assertIsNone(kw.get("prop_key"))
        self.assertIs(kw.get("confirmed_lineup"), lineup)
        self.assertIs(kw.get("probable_starters"), probs)

    def test_empty_name_returns_none(self):
        with self._no_sfbb():
            self.assertIsNone(m.resolve_mlbam_id("", SEASON, prop_key="batter_hits"))

    def test_never_raises(self):
        # SFBB stubbed out too (hermetic): the only failure under test is the index
        # blowing up, which the outer fail-open guard must swallow to None.
        with self._no_sfbb(), \
                mock.patch.object(m, "_player_index",
                                  side_effect=RuntimeError("boom")):
            self.assertIsNone(
                m.resolve_mlbam_id("Aaron Judge", SEASON, prop_key="batter_hits"))


if __name__ == "__main__":
    unittest.main()
