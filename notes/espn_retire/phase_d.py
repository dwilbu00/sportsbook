"""One-off edit script (ESPN-for-NBA retirement, phase D): drop the NBA pieces of the ESPN gamelog
cache (+ its DDL; optional owner-run drop script), the dead ESPN athlete-list helper, the dead NBA
output-defense default, and stale comments. Cache tests move to NFL rows (still served)."""
import re

ROOT = r"c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/"


def edit(rel, pairs):
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


# ── gamelog_store: NFL-only ESPN cache now ───────────────────────────────────────────────
edit("gamelog_store.py", [
    ("* **MLB + NBA + NFL.** Other sports (e.g. NHL) have no fact table -> the store",
     "* **MLB + NFL** (NBA dropped 2026-10: warehouse-only, ESPN retired). Other sports\n"
     "  (e.g. NHL) have no fact table -> the store"),
    ('_NBA_STATS = ("MIN", "PTS", "REB", "AST")\n', ""),
    ('nba_gamelog = _fact_table("nba_gamelog", _NBA_STATS)\n', ""),
    ('    "nba": (nba_gamelog, _NBA_STATS),\n', ""),
    ('''def _sport_has_table(sport):
    return sport in ("basketball", "football")''',
     '''def _sport_has_table(sport):
    return sport == "football"'''),
    ('''    """Fetch the raw ESPN gamelog for NBA/NFL. (MLB is warehouse-only post-P6, so the
    old empty-gamelog pitcher/synth fallback is gone.)"""''',
     '''    """Fetch the raw ESPN gamelog for NFL. (MLB and NBA are warehouse-only, so the old
    empty-gamelog pitcher/synth fallback is gone.)"""'''),
    ('''    """player_type for the fetched rows (NBA/NFL only — MLB is warehouse-only)."""
    if sport == "basketball":
        return "nba"
    if sport == "football":''',
     '''    """player_type for the fetched rows (NFL only — MLB and NBA are warehouse-only)."""
    if sport == "football":'''),
])

# ── its tests: generic cache mechanics now exercised with NFL rows ───────────────────────
p = ROOT + "test_gamelog_store.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''def _nba_row(game_date, pts=25.0, reb=8.0, ast=6.0, minutes=34.0):
    return {"MIN": minutes, "FG": 9.0, "3PT": 3.0, "FT": 4.0, "REB": reb,
            "AST": ast, "BLK": 1.0, "STL": 2.0, "PF": 3.0, "TO": 2.0,
            "PTS": pts, "opponent": "Celtics", "is_home": True, "team_id": "2",
            "game_date": game_date, "completed": True}


''', "")
start = s.index("    def test_nba_roundtrip(self):")
end = s.index("    def test_nfl_roundtrip_reduced_shape(self):")
s = s[:start] + s[end:]
start = s.index("    def test_nba_columns(self):")
end = s.index("    def test_nfl_columns(self):")
s = s[:start] + s[end:]
s = s.replace('"basketball", "nba"', '"football", "nfl"')
s = s.replace('sport="basketball", league="nba"', 'sport="football", league="nfl"')
s = s.replace("_nba_row(", "_nfl_row(")
s = s.replace('_read_rows(conn, "nba",', '_read_rows(conn, "nfl",')
s = s.replace('self.assertEqual(meta["player_type"], "nba")',
              'self.assertEqual(meta["player_type"], "nfl")')
s = s.replace("    # NBA fixtures (the store is NBA/NFL-only post-P4b; MLB is warehouse-only).",
              "    # NFL fixtures (the store is NFL-only: MLB and NBA are warehouse-only).")
s = s.replace("    # generic store round-trip is covered by the NBA/NFL cases below.)",
              "    # generic store round-trip is covered by the NFL case below; NBA is warehouse-only.)")
s = s.replace("    # (MLB batter/pitcher classification removed in P4b — _classify is NBA/NFL-only.)",
              "    # (MLB batter/pitcher classification removed in P4b — _classify is NFL-only.)")
assert "basketball" not in s and "_nba_row" not in s, "leftover NBA refs in test_gamelog_store"
open(p, "w", encoding="utf-8").write(s)

# ── DDL: drop the nba_gamelog block from the hand-run schema; optional owner drop script ─
p = ROOT + "sql/schema.sql"
s = open(p, encoding="utf-8").read()
start = s.index("--------------------------------------------------------------------- nba_gamelog")
end = s.index("--------------------------------------------------------------------- nfl_gamelog")
s = s[:start] + s[end:]
s = s.replace("    player_type     NVARCHAR(16),                   -- batter|pitcher|nba",
              "    player_type     NVARCHAR(16),                   -- batter|pitcher|nfl (nba retired)")
open(p, "w", encoding="utf-8").write(s)
open(ROOT + "sql/drop_nba_gamelog.sql", "w", encoding="utf-8").write(
    """-- drop_nba_gamelog.sql — OPTIONAL owner-run cleanup after the ESPN-for-NBA retirement
-- (2026-10-16). dbo.nba_gamelog was a RE-FETCHABLE ESPN cache (replace-all per athlete/season;
-- never irreplaceable data) that nothing reads anymore — NBA history + grading come from the
-- stats.nba.com warehouse. Safe to leave in place; run this only to reclaim the space.
-- Run as owner/admin in Azure (the app login has no DDL).
IF OBJECT_ID('dbo.nba_gamelog', 'U') IS NOT NULL
    DROP TABLE dbo.nba_gamelog;
GO
-- bookkeeping rows for the dropped cache (NFL rows in these shared tables are untouched)
DELETE FROM dbo.gamelog_fetch_meta WHERE sport = 'basketball';
DELETE FROM dbo.athlete_id_cache WHERE sport = 'basketball';
GO
""")

# ── espn_client: the ESPN season athlete listing only the old NBA refit pool used ────────
p = ROOT + "espn_client.py"
s = open(p, encoding="utf-8").read()
start = s.index("def list_season_athletes(")
m = re.search(r"\n(def |class |[A-Z_]+ = )", s[s.index("def _parse_byathlete_entry("):])
end = s.index("def _parse_byathlete_entry(") + m.start() + 1
s = s[:start] + s[end:]
open(p, "w", encoding="utf-8").write(s)

# ── props: the NBA output-defense default is dead (raw points-allowed retired for NBA) ───
p = ROOT + "props.py"
s = open(p, encoding="utf-8").read()
m = re.search(r'PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH = \{[^}]*\}', s)
assert m, "output defense dict"
block = m.group(0)
new_block = re.sub(r'\n\s*"basketball_nba":\s*[0-9.]+,?[^\n]*', "", block)
assert new_block != block, "NBA entry not found in output-defense dict"
s = s.replace(block, new_block)
open(p, "w", encoding="utf-8").write(s)

# ── stale comment ────────────────────────────────────────────────────────────────────────
edit("recalibration.py", [(
    "This supersedes the ephemeral ESPN nba_gamelog cache.",
    "NBA is warehouse-only (the ESPN nba_gamelog cache was retired 2026-10).")])
print("ok")
