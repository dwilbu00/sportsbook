"""Tests for the final two P2 audit items.

1. Savant team-key mismatch (mlb_starters) — the expected-runs ensemble
   challenger keyed its offense/bullpen dicts by the team string Savant returns
   (``player_name`` under group_by=team) but looked them up by the StatsAPI
   ``abbreviation``. A live probe confirmed these diverge for Oakland (StatsAPI
   ``OAK`` vs Savant ``ATH``), silently disabling the challenger for that team.
   The fix normalizes Savant keys into the StatsAPI namespace (self-correcting
   across seasons) and warns loudly on any residual coverage gap.

2. Athlete-name collisions (espn_client) — ``search_athlete`` took the first of
   up to 5 name matches with no team filter, so a common name could resolve to
   the wrong (or wrong-sport) athlete. It now prefers a candidate on one of the
   matchup's two teams when a ``team_ids`` hint is supplied, falling back to the
   first result otherwise (never worse than before).
"""

import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

import db_store
import espn_client
import mlb_starters
import savant_history as sh


# ── Item 1: Savant team-key normalization ────────────────────────────────────
class CanonicalTeamKeyTests(unittest.TestCase):
    def test_alias_applied_when_target_present(self):
        # Savant 'ATH' maps to StatsAPI 'OAK' when OAK is the season's abbr.
        abbrs = {"OAK", "NYY", "SEA"}
        self.assertEqual(mlb_starters._canonical_team_key("ATH", abbrs), "OAK")

    def test_raw_key_kept_when_it_is_canonical(self):
        # If a future season's StatsAPI itself uses 'ATH', do NOT remap to OAK.
        abbrs = {"ATH", "NYY", "SEA"}
        self.assertEqual(mlb_starters._canonical_team_key("ATH", abbrs), "ATH")

    def test_unknown_key_unchanged(self):
        abbrs = {"OAK", "NYY"}
        self.assertEqual(mlb_starters._canonical_team_key("ZZZ", abbrs), "ZZZ")

    def test_alias_not_applied_when_target_absent(self):
        # No OAK in this index -> can't safely remap; leave raw for validation.
        abbrs = {"NYY", "SEA"}
        self.assertEqual(mlb_starters._canonical_team_key("ATH", abbrs), "ATH")


def _savant_rows(teams):
    """Grouped-by-team CSV rows (only the columns the parser reads)."""
    return [{"player_name": t, "xwoba": "0.320", "pa": "200"} for t in teams]


def _fake_index(abbrs):
    return {f"norm_{a.lower()}": {"id": i, "name": a, "abbr": a}
            for i, a in enumerate(sorted(abbrs), start=1)}


class ExpectedRunsTeamKeyTests(unittest.TestCase):
    def _run(self, savant_teams, index_abbrs):
        # 3 CSV fetches: offense vs L, offense vs R, bullpen RP.
        rows = _savant_rows(savant_teams)
        stderr = io.StringIO()
        with patch.object(mlb_starters, "_read_cache", return_value=None), \
                patch.object(mlb_starters, "_write_cache"), \
                patch.object(mlb_starters, "_get_savant_csv",
                             side_effect=[rows, rows, rows]), \
                patch.object(mlb_starters, "get_team_index",
                             return_value=_fake_index(index_abbrs)), \
                redirect_stderr(stderr):
            out = mlb_starters.get_expected_runs_team_factors(2024, "2024-09-01")
        return out, stderr.getvalue()

    def test_savant_key_normalized_to_statsapi_abbr(self):
        # Savant emits ATH; consumer looks up OAK. After normalization the
        # offense/bullpen dicts must be keyed by OAK, and coverage is complete
        # so nothing is warned.
        out, log = self._run(["ATH", "NYY"], {"OAK", "NYY"})
        self.assertIsNotNone(out)
        self.assertIn("OAK", out["offense_vs_hand"]["L"])
        self.assertNotIn("ATH", out["offense_vs_hand"]["L"])
        self.assertIn("OAK", out["bullpen_xwoba"])
        self.assertEqual(log, "")

    def test_unmapped_key_warns_loudly(self):
        # 'ZZZ' has no alias and isn't in the index -> stays unmapped; SEA in the
        # index has no Savant data -> missing. Either way the challenger is
        # disabled for those teams, so a coverage-gap warning must fire.
        out, log = self._run(["ATH", "NYY", "ZZZ"], {"OAK", "NYY", "SEA"})
        self.assertIsNotNone(out)
        self.assertIn("OAK", out["offense_vs_hand"]["L"])   # ATH still fixed
        self.assertIn("ZZZ", out["offense_vs_hand"]["L"])   # left for visibility
        self.assertIn("coverage gap", log)
        self.assertIn("ZZZ", log)
        self.assertIn("SEA", log)


# ── Item 2: team-aware athlete resolution ────────────────────────────────────
class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _athlete(aid, name, team_id):
    return {"id": aid, "displayName": name, "team": {"id": team_id}}


class SearchAthleteTeamHintTests(unittest.TestCase):
    def _site_payload(self):
        # Two different real players share a display name, on different teams.
        return _Resp({"athletes": [
            _athlete("100", "Will Smith", "5"),    # ESPN's first result
            _athlete("200", "Will Smith", "12"),   # the one in our game
        ]})

    def test_prefers_candidate_on_matchup_team(self):
        with patch("espn_client.requests.get", return_value=self._site_payload()):
            res = espn_client.search_athlete(
                "baseball", "mlb", "Will Smith", team_ids=["12", "9"])
        self.assertEqual(res["id"], "200")
        self.assertEqual(res["team_id"], "12")

    def test_falls_back_to_first_without_hint(self):
        with patch("espn_client.requests.get", return_value=self._site_payload()):
            res = espn_client.search_athlete("baseball", "mlb", "Will Smith")
        self.assertEqual(res["id"], "100")

    def test_falls_back_to_first_when_no_team_matches(self):
        # Hint given but neither candidate is on those teams -> first (unchanged).
        with patch("espn_client.requests.get", return_value=self._site_payload()):
            res = espn_client.search_athlete(
                "baseball", "mlb", "Will Smith", team_ids=["999"])
        self.assertEqual(res["id"], "100")

    def test_team_ids_ints_are_coerced(self):
        # Callers pass str ids, but be robust to ints too.
        with patch("espn_client.requests.get", return_value=self._site_payload()):
            res = espn_client.search_athlete(
                "baseball", "mlb", "Will Smith", team_ids=[12])
        self.assertEqual(res["id"], "200")


def _web_item(aid, name, sport, league, team_core_id):
    """A web-search /search item. Matches the real ESPN shape probed live: team
    is exposed via teamRelationships[].core.id (NOT a top-level team.id), and the
    feed is cross-sport for a shared name."""
    return {"id": aid, "displayName": name, "sport": sport, "league": league,
            "teamRelationships": [{"type": "team", "core": {"id": team_core_id}}]}


class SearchAthleteWebFallbackTests(unittest.TestCase):
    """Exercises the web-search fallback — the branch the site API's 404 forces
    for MLB (verified live), which the site-API-shaped tests above never reach.
    Covers sport/league filtering, teamRelationships-based team-id extraction,
    team preference among same-name MLB players, and cross-sport isolation."""

    def _responses(self):
        # 1st call = site API (404 for MLB -> falls through); 2nd = web search.
        site = _Resp({}, status=404)
        web = _Resp({"items": [
            _web_item("38309", "Will Smith", "baseball", "mlb", "19"),  # first MLB
            _web_item("900", "Will Smith", "hockey", "nhl", "4"),       # cross-sport
            _web_item("31549", "Will Smith", "baseball", "mlb", "4"),   # in our game
            _web_item("777", "Will Smith", "football", "college-football", "88"),
        ]})
        return [site, web]

    def test_prefers_matchup_team_via_team_relationships(self):
        with patch("espn_client.requests.get", side_effect=self._responses()):
            res = espn_client.search_athlete(
                "baseball", "mlb", "Will Smith", team_ids=["4", "9"])
        # The MLB Will Smith on team 4 (id 31549), not the NHL one also on "4".
        self.assertEqual(res["id"], "31549")
        self.assertEqual(res["team_id"], "4")

    def test_first_mlb_result_without_hint(self):
        with patch("espn_client.requests.get", side_effect=self._responses()):
            res = espn_client.search_athlete("baseball", "mlb", "Will Smith")
        self.assertEqual(res["id"], "38309")
        self.assertEqual(res["team_id"], "19")

    def test_sport_filter_blocks_cross_sport_team_id_collision(self):
        # Hint = the NHL player's team id. Because team preference is confined to
        # sport/league-matched candidates, we must NOT return the NHL athlete —
        # we return the first MLB result instead.
        with patch("espn_client.requests.get", side_effect=self._responses()):
            res = espn_client.search_athlete(
                "baseball", "mlb", "Will Smith", team_ids=["999"])
        self.assertEqual(res["id"], "38309")


class GetPlayerStatHistoryForwardsHintTests(unittest.TestCase):
    def test_team_ids_forwarded_to_search(self):
        search = Mock(return_value={"id": "", "name": "x", "team_id": None})
        with patch.object(espn_client, "search_athlete", search):
            espn_client.get_player_stat_history(
                "basketball", "nba", "Some Player", "player_points",
                team_ids=["1", "2"])
        search.assert_called_once()
        self.assertEqual(search.call_args.kwargs.get("team_ids"), ["1", "2"])


class WarehouseTeamOffenseTests(unittest.TestCase):
    """ODI_MLB_WAREHOUSE_OFFENSE: team OFFENSE factors from the statcast_pitch warehouse
    instead of the live Savant HTTP endpoint (works on a Savant-unreachable box)."""

    def setUp(self):
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        sh.create_all()
        mlb_starters._COVERAGE_WARNED.clear()

    def tearDown(self):
        db_store.configure_engine(None)

    def _pitch(self, team, p_throws, xwoba, date="2024-06-01"):
        with db_store.get_engine().begin() as c:
            c.execute(sh.statcast_pitch.insert(), {
                "game_date": date, "pitcher": "1", "batter": "9",
                "p_throws": p_throws, "batting_team": team, "stand": "R",
                "xwoba": xwoba})

    # ── _warehouse_team_factors (the derivation) ──
    def test_warehouse_offense_from_statcast(self):
        for _ in range(60):
            self._pitch("NYY", "R", 0.340)
            self._pitch("SEA", "R", 0.300)
            self._pitch("NYY", "L", 0.360)
        with patch.object(mlb_starters, "get_team_index",
                          return_value=_fake_index({"NYY", "SEA"})):
            out = mlb_starters._warehouse_team_factors(2024, "2024-09-01")
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["offense_vs_hand"]["R"]["NYY"], 0.340, places=6)
        self.assertAlmostEqual(out["offense_vs_hand"]["R"]["SEA"], 0.300, places=6)
        self.assertAlmostEqual(out["offense_vs_hand"]["L"]["NYY"], 0.360, places=6)
        self.assertIsNotNone(out["league_xwoba"])
        self.assertEqual(out["bullpen_xwoba"], {})            # v1: empty
        self.assertIsNone(out["league_bullpen_xwoba"])        # -> starter-only staff

    def test_min_pa_filter_drops_thin_teams(self):
        for _ in range(60):
            self._pitch("NYY", "R", 0.340)
        for _ in range(10):                                   # < min_pa=40 -> dropped
            self._pitch("SEA", "R", 0.300)
        with patch.object(mlb_starters, "get_team_index",
                          return_value=_fake_index({"NYY", "SEA"})):
            out = mlb_starters._warehouse_team_factors(2024, "2024-09-01")
        self.assertIn("NYY", out["offense_vs_hand"]["R"])
        self.assertNotIn("SEA", out["offense_vs_hand"]["R"])

    def test_leakage_safe_excludes_as_of_and_after(self):
        for _ in range(60):
            self._pitch("NYY", "R", 0.200, date="2024-05-01")   # before cutoff
        for _ in range(60):
            self._pitch("NYY", "R", 0.900, date="2024-09-01")   # on/after as-of
        with patch.object(mlb_starters, "get_team_index",
                          return_value=_fake_index({"NYY"})):
            out = mlb_starters._warehouse_team_factors(2024, "2024-06-01")  # cutoff 05-31
        self.assertAlmostEqual(out["offense_vs_hand"]["R"]["NYY"], 0.200, places=6)

    def test_savant_key_normalized_from_warehouse(self):
        for _ in range(60):
            self._pitch("ATH", "R", 0.320)                    # Savant abbr
        with patch.object(mlb_starters, "get_team_index",
                          return_value=_fake_index({"OAK", "NYY"})):
            out = mlb_starters._warehouse_team_factors(2024, "2024-09-01")
        self.assertIn("OAK", out["offense_vs_hand"]["R"])     # normalized
        self.assertNotIn("ATH", out["offense_vs_hand"]["R"])

    def test_empty_statcast_returns_none(self):
        with patch.object(mlb_starters, "get_team_index",
                          return_value=_fake_index({"NYY"})):
            self.assertIsNone(mlb_starters._warehouse_team_factors(2024, "2024-09-01"))

    def test_sql_off_returns_none(self):
        db_store.configure_engine(None)
        self.assertIsNone(mlb_starters._warehouse_team_factors(2024, "2024-09-01"))

    # ── the env flag ──
    def test_flag_env_truth_table(self):
        for val, exp in [("1", True), ("true", True), ("on", True), ("YES", True),
                         ("", False), ("0", False), ("off", False)]:
            with patch.dict(os.environ, {"ODI_MLB_WAREHOUSE_OFFENSE": val}):
                self.assertEqual(mlb_starters._mlb_warehouse_offense_enabled(), exp)

    # ── get_expected_runs_team_factors routing ──
    def test_flag_off_uses_savant_not_warehouse(self):
        rows = _savant_rows(["NYY", "SEA"])
        with patch.object(mlb_starters, "_mlb_warehouse_offense_enabled",
                          return_value=False), \
                patch.object(mlb_starters, "_warehouse_team_factors") as wf, \
                patch.object(mlb_starters, "_read_cache", return_value=None), \
                patch.object(mlb_starters, "_write_cache"), \
                patch.object(mlb_starters, "_get_savant_csv",
                             side_effect=[rows, rows, rows]), \
                patch.object(mlb_starters, "get_team_index",
                             return_value=_fake_index({"NYY", "SEA"})):
            out = mlb_starters.get_expected_runs_team_factors(2024, "2024-09-01")
        wf.assert_not_called()                                # warehouse never touched
        self.assertIsNotNone(out)

    def test_flag_on_prefers_warehouse_over_savant(self):
        wh = {"league_xwoba": 0.320, "league_bullpen_xwoba": None,
              "offense_vs_hand": {"L": {"NYY": 0.33}, "R": {"NYY": 0.31}},
              "bullpen_xwoba": {}}
        with patch.object(mlb_starters, "_mlb_warehouse_offense_enabled",
                          return_value=True), \
                patch.object(mlb_starters, "_read_cache", return_value=None), \
                patch.object(mlb_starters, "_write_cache"), \
                patch.object(mlb_starters, "_warehouse_team_factors",
                             return_value=wh), \
                patch.object(mlb_starters, "_get_savant_csv") as savant:
            out = mlb_starters.get_expected_runs_team_factors(2024, "2024-09-01")
        self.assertEqual(out, wh)
        savant.assert_not_called()                            # no Savant HTTP

    def test_flag_on_falls_back_to_savant_when_warehouse_none(self):
        rows = _savant_rows(["NYY", "SEA"])
        with patch.object(mlb_starters, "_mlb_warehouse_offense_enabled",
                          return_value=True), \
                patch.object(mlb_starters, "_read_cache", return_value=None), \
                patch.object(mlb_starters, "_write_cache"), \
                patch.object(mlb_starters, "_warehouse_team_factors",
                             return_value=None), \
                patch.object(mlb_starters, "_get_savant_csv",
                             side_effect=[rows, rows, rows]), \
                patch.object(mlb_starters, "get_team_index",
                             return_value=_fake_index({"NYY", "SEA"})):
            out = mlb_starters.get_expected_runs_team_factors(2024, "2024-09-01")
        self.assertIsNotNone(out)                             # Savant fallback fired
        self.assertIn("NYY", out["offense_vs_hand"]["R"])


if __name__ == "__main__":
    unittest.main()
