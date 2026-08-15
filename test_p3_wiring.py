"""P3 part-2 wiring: the resolver's (mlb_player_id, game_pk) is stamped onto new
prediction / wager / odds_line rows, and a fail-closed MISS stays NULL — it must
NEVER fall back to the weaker single-team lookup. entity_resolver / player_id_map
are monkeypatched; these exercise the pure row-builders (no SQL needed)."""

import os
import unittest
from unittest import mock


class PredictionStampTests(unittest.TestCase):
    """recalibration.log_prediction(write=False) + _enrich_prediction_ids."""

    def _row(self, **kw):
        import recalibration
        return recalibration.log_prediction(
            sport_key="baseball_mlb", prop_key="batter_hits", player="Aaron Judge",
            game_date="2026-08-09", line=0.5, raw_prob=0.6, write=False, **kw)

    def test_resolved_ids_stamped_and_fallback_suppressed(self):
        # ids_resolved=True → keep the resolver id verbatim; the weak single-team
        # fallback (mocked to a WRONG value) must NOT be consulted.
        with mock.patch("player_id_map.mlb_id_for_name", return_value="WRONG"):
            row = self._row(mlb_player_id="592450", game_pk=777, ids_resolved=True,
                            team="New York Yankees")
        self.assertEqual(row["player_mlb_id"], "592450")
        self.assertEqual(row["game_pk"], 777)
        self.assertEqual(row["player_key"], "mlb:592450")

    def test_resolved_miss_stays_null_no_fallback(self):
        # The crux: a fail-closed miss (ids_resolved=True, id=None) stays NULL and
        # does NOT fall back to mlb_id_for_name (which would re-guess).
        with mock.patch("player_id_map.mlb_id_for_name", return_value="WRONG"):
            row = self._row(mlb_player_id=None, game_pk=777, ids_resolved=True,
                            team="New York Yankees")
        self.assertIsNone(row["player_mlb_id"])
        self.assertEqual(row["game_pk"], 777)
        self.assertTrue(row["player_key"].startswith("name:"))

    def test_legacy_path_still_uses_fallback(self):
        # No ids_resolved → the legacy single-team enricher runs (back-compat).
        with mock.patch("player_id_map.mlb_id_for_name", return_value="123"), \
             mock.patch("player_id_map.team_code_for_name", return_value="NYY"):
            row = self._row(team="New York Yankees")
        self.assertEqual(row["player_mlb_id"], "123")
        self.assertIsNone(row["game_pk"])                # not provided → NULL


class PropsStampThreadingTests(unittest.TestCase):
    """Commit C P2: analyze_player_props_value threads prop_key + confirmed_lineup +
    probable_starters into the identity stamp, and role-partitions the per-event
    cache so a same-name batter and pitcher (a two-way player offered as both)
    resolve independently (one resolver call per role)."""

    def _prop_data(self):
        od = {"line": 0.5, "over_implied": 0.5, "under_implied": 0.5,
              "over_price": -110, "under_price": -110,
              "over_book": "DK", "under_book": "DK"}
        return {"commence_time": "2026-08-10T23:10:00Z",
                "home_team": "Los Angeles Angels", "away_team": "Houston Astros",
                "game_id": "evt-c",
                "props": {"batter_hits": {"Shohei Ohtani": dict(od)},
                          "pitcher_strikeouts": {"Shohei Ohtani": dict(od)}}}

    def test_prop_key_and_game_context_threaded_role_partitioned(self):
        import props
        calls = []

        def _resolve(name, *a, **k):
            calls.append(k)
            return {"resolved": True, "mlb_player_id": "660271", "game_pk": 1}

        lineup = {"players": {}}
        probs = {"houston astros": {"pitcher_id": 1, "name": "x"}}
        # Enforce ON makes the slate pre-scan resolve every (name, prop_key) pair
        # deterministically (candidate production aside), so the role-partition is
        # observable as exactly one call per role.
        with mock.patch.dict(os.environ, {props._MLB_ENFORCE_IDENTITY_ENV: "1"}), \
             mock.patch("entity_resolver.resolve", side_effect=_resolve), \
             mock.patch.object(props, "maybe_auto_refit"), \
             mock.patch.object(props, "load_recalibration", return_value={}), \
             mock.patch.object(props, "log_prediction_rows"):
            props.analyze_player_props_value(
                self._prop_data(), {}, threshold_pct=1.0, sport_key="baseball_mlb",
                confirmed_lineup=lineup, probable_starters=probs)
        # Same name, two roles → two distinct resolver calls (cache is (name, role)).
        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(c.get("prop_key") for c in calls),
                         ["batter_hits", "pitcher_strikeouts"])
        for c in calls:                             # game context forwarded each call
            self.assertIs(c.get("confirmed_lineup"), lineup)
            self.assertIs(c.get("probable_starters"), probs)


class WagerStampTests(unittest.TestCase):
    """wagers.build_wager_row → _enrich_ids stamps via the resolver."""

    def test_player_prop_stamped_via_resolver(self):
        import wagers
        meta = {"sport_key": "baseball_mlb", "placed_at": "t", "event_id": "e1",
                "commence_time": "2026-08-09T23:05:00Z", "game_date": "2026-08-09",
                "home_team": "New York Yankees", "away_team": "Boston Red Sox"}
        candidate = {"player": "Aaron Judge", "prop": "batter_hits",
                     "prop_label": "Hits", "direction": "OVER", "line": 0.5,
                     "over_price": -120, "dk_over_price": -115, "over_rate": 60.0,
                     "team": "New York Yankees", "matchup": "BOS @ NYY"}
        rc = mock.Mock(return_value={"mlb_player_id": "592450", "game_pk": 777})
        with mock.patch("entity_resolver.resolve", rc), \
             mock.patch("player_id_map.team_code_for_name", return_value="NYY"):
            row = wagers.build_wager_row("player_prop", "OVER", candidate, meta)
        self.assertEqual(row["player_mlb_id"], "592450")
        self.assertEqual(row["game_pk"], 777)
        # Commit C: the prop_key is threaded so the resolver role-partitions.
        self.assertEqual(rc.call_args.kwargs.get("prop_key"), "batter_hits")


class OddsLineStampTests(unittest.TestCase):
    """warehouse._enrich_ids stamps prop lines via the resolver (cached per player)."""

    def test_prop_line_stamped_and_cached(self):
        import warehouse
        meta = {"home": "New York Yankees", "away": "Boston Red Sox",
                "game_date": "2026-08-09", "commence_time": "2026-08-09T23:05:00Z"}
        lines = [
            {"bet_type": "player_prop", "player": "Aaron Judge", "prop_key": "batter_hits"},
            {"bet_type": "player_prop", "player": "Aaron Judge", "prop_key": "batter_hits"},
            {"bet_type": "moneyline", "selection": "New York Yankees"},
        ]
        rc = mock.Mock(return_value={"mlb_player_id": "592450", "game_pk": 777})
        with mock.patch("entity_resolver.resolve", rc), \
             mock.patch("player_id_map.team_code_for_name", return_value="NYY"):
            warehouse._enrich_ids("baseball_mlb", meta, lines)
        self.assertEqual(lines[0]["player_mlb_id"], "592450")
        self.assertEqual(lines[0]["game_pk"], 777)
        self.assertEqual(lines[1]["game_pk"], 777)          # OVER/UNDER share
        self.assertEqual(lines[2]["team_code"], "NYY")      # team line untouched by resolver
        self.assertEqual(rc.call_count, 1)                  # resolved once, then cached

    def test_non_mlb_sport_no_resolve(self):
        import warehouse
        lines = [{"bet_type": "player_prop", "player": "LeBron James"}]
        rc = mock.Mock(return_value={"mlb_player_id": "x", "game_pk": 1})
        with mock.patch("entity_resolver.resolve", rc):
            warehouse._enrich_ids("basketball_nba", {"home": "LAL", "away": "BOS"}, lines)
        self.assertEqual(rc.call_count, 0)                  # gated off for non-baseball
        self.assertIsNone(lines[0].get("game_pk"))

    def test_dual_role_same_name_resolves_per_role(self):
        # Commit C: a two-way player offered as BOTH a batter and pitcher prop must
        # resolve once PER ROLE (the cache is (name, role)), each with its prop_key.
        import warehouse
        meta = {"home": "Los Angeles Angels", "away": "Houston Astros",
                "game_date": "2026-08-09", "commence_time": "2026-08-09T23:05:00Z"}
        lines = [
            {"bet_type": "player_prop", "player": "Shohei Ohtani",
             "prop_key": "batter_hits"},
            {"bet_type": "player_prop", "player": "Shohei Ohtani",
             "prop_key": "pitcher_strikeouts"},
        ]
        rc = mock.Mock(return_value={"mlb_player_id": "660271", "game_pk": 5})
        with mock.patch("entity_resolver.resolve", rc):
            warehouse._enrich_ids("baseball_mlb", meta, lines)
        self.assertEqual(rc.call_count, 2)                  # one per role, not cached across
        self.assertEqual(sorted(c.kwargs.get("prop_key") for c in rc.call_args_list),
                         ["batter_hits", "pitcher_strikeouts"])


if __name__ == "__main__":
    unittest.main()
