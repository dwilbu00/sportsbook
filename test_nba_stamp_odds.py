"""P5b — the NBA odds_line id-stamp (nba_warehouse.prop_id_index/resolve_prop_ids +
nba_stamp_odds.stamp). Verifies: unambiguous resolve; namesake/no-match left unstamped;
FILL + OVERWRITE-drift + NEVER-null-a-good-stamp policy; dry-run writes nothing; apply is
idempotent; non-NBA / non-prop rows are ignored. SQLite warehouse; no network.
"""
import unittest

import db_store
import nba_warehouse as nw
import nba_stamp_odds as stamp


def _snap(c, sport, gdate, eid, commence):
    return c.execute(db_store.odds_snapshot.insert().values(
        sport=sport, game_date=gdate, event_id=eid, kind="props",
        snapshot_hour=f"{gdate.replace('-', '')}T00Z", commence_time=commence)).inserted_primary_key[0]


def _line(c, snap_id, player, bet_type="player_prop", pid=None, gid=None):
    return c.execute(db_store.odds_line.insert().values(
        snapshot_id=snap_id, bet_type=bet_type, selection="Over", player=player,
        prop_key="player_points", direction="Over", price=-110,
        nba_person_id=pid, nba_game_id=gid)).inserted_primary_key[0]


class StampTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        nw.create_all()
        with db_store.get_engine().begin() as c:
            # two "star guard" namesakes + one solo player
            for pid, nn in ((10, "star guard"), (11, "star guard"), (20, "solo player")):
                c.execute(nw.nba_player.insert().values(
                    person_id=pid, name_norm=nn, full_name=nn.title()))
            for pid, gid, d in ((10, "0022400001", "2024-01-05"),   # star guard unique 01-05
                                (20, "0022400050", "2024-01-06"),   # solo player 01-06
                                (10, "0022400010", "2024-01-10"),   # namesake collision 01-10
                                (11, "0022400011", "2024-01-10")):
                c.execute(nw.nba_player_game.insert().values(
                    person_id=pid, game_id=gid, season=2024, game_date=d, played=True,
                    minutes=30.0, points=20.0))
            s1 = _snap(c, "basketball_nba", "2024-01-05", "e1", "2024-01-05T19:00:00Z")
            s2 = _snap(c, "basketball_nba", "2024-01-06", "e2", "2024-01-06T19:00:00Z")
            s3 = _snap(c, "basketball_nba", "2024-01-10", "e3", "2024-01-10T19:00:00Z")
            sb = _snap(c, "baseball_mlb", "2024-01-05", "eb", "2024-01-05T19:00:00Z")
            self.L1 = _line(c, s1, "Star Guard")                       # -> fill (10, 0022400001)
            self.L2 = _line(c, s2, "Solo Player")                      # -> fill (20, 0022400050)
            self.L3 = _line(c, s3, "Star Guard")                       # -> unresolved (namesake)
            self.L4 = _line(c, s1, "Star Guard", pid=10, gid="0022400001")   # already correct
            self.L5 = _line(c, s1, "Star Guard", pid=999, gid="0022409999")  # overwrite drift
            self.L6 = _line(c, s3, "Star Guard", pid=10, gid="0022400010")   # unresolved but STAMPED
            self.Lb = _line(c, sb, "Some Batter")                      # non-NBA -> ignored
            self.Lt = _line(c, s1, "Lakers", bet_type="moneyline")     # non-prop -> ignored

    def tearDown(self):
        db_store.configure_engine(None)

    def _get(self, rid):
        ol = db_store.odds_line
        with db_store.get_engine().connect() as c:
            r = c.execute(select_row(ol, rid)).one()
        return r.nba_person_id, r.nba_game_id

    # ---- resolver ----
    def test_index_and_resolve(self):
        idx = nw.prop_id_index([2024])
        self.assertEqual(nw.resolve_prop_ids(idx, "Star Guard", "2024-01-05"),
                         (10, "0022400001"))                      # unambiguous
        self.assertEqual(nw.resolve_prop_ids(idx, "Star Guard", "2024-01-10"),
                         (None, None))                            # namesake collision -> skip
        self.assertEqual(nw.resolve_prop_ids(idx, "Nobody", "2024-01-05"),
                         (None, None))                            # no such player
        self.assertEqual(nw.resolve_prop_ids(idx, "Solo Player", "2024-01-05"),
                         (None, None))                            # right player, wrong date

    # ---- plan counts ----
    def test_plan_counts(self):
        s = stamp.stamp(dry_run=True)["stats"]
        self.assertEqual(s["scanned"], 6)          # L1..L6 (Lb non-NBA, Lt non-prop excluded)
        self.assertEqual(s["already_correct"], 1)  # L4
        self.assertEqual(s["fill"], 2)             # L1, L2
        self.assertEqual(s["overwrite_drift"], 1)  # L5
        self.assertEqual(s["unresolved"], 2)       # L3, L6
        self.assertEqual(s["to_update_rows"], 3)   # L1, L2, L5
        self.assertEqual(s["to_update_games"], 2)  # (10,0022400001)x2 + (20,0022400050)

    def test_dry_run_writes_nothing(self):
        stamp.stamp(dry_run=True)
        self.assertEqual(self._get(self.L1), (None, None))        # still unstamped
        self.assertEqual(self._get(self.L5), (999, "0022409999"))  # drift untouched

    def test_apply_and_idempotent(self):
        stamp.stamp(dry_run=False)
        self.assertEqual(self._get(self.L1), (10, "0022400001"))  # filled
        self.assertEqual(self._get(self.L2), (20, "0022400050"))  # filled
        self.assertEqual(self._get(self.L5), (10, "0022400001"))  # drift overwritten
        self.assertEqual(self._get(self.L4), (10, "0022400001"))  # unchanged
        # NEVER null a good stamp on a no-resolve:
        self.assertEqual(self._get(self.L6), (10, "0022400010"))  # preserved
        self.assertEqual(self._get(self.L3), (None, None))        # stayed null
        # non-NBA / non-prop never touched
        self.assertEqual(self._get(self.Lb), (None, None))
        # second run = no-op
        s2 = stamp.stamp(dry_run=False)["stats"]
        self.assertEqual(s2["to_update_rows"], 0)
        self.assertEqual(s2["already_correct"], 4)                # L1,L2,L4,L5


def select_row(table, rid):
    from sqlalchemy import select
    return select(table).where(table.c.id == rid)


if __name__ == "__main__":
    unittest.main()
