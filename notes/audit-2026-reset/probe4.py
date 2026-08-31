"""READ-ONLY probe 4: namesake collision detail."""
import db_store
from sqlalchemy import text
db_store.promote_secrets_from_toml()
eng = db_store.get_engine()

def q(label, sql):
    print("\n=== " + label + " ===")
    try:
        with eng.connect() as c:
            for row in c.execute(text(sql)):
                print(" ", tuple(row))
    except Exception as e:
        print("  ERROR:", type(e).__name__, str(e)[:300])

q("odds_line 'Luis Garcia Jr.' distinct mlb_id x snapshot home/away", """
SELECT l.player_mlb_id, s.home, s.away, COUNT(*) n
FROM odds_line l JOIN odds_snapshot s ON l.snapshot_id=s.id
WHERE l.player='Luis Garcia Jr.' AND l.bet_type='player_prop'
GROUP BY l.player_mlb_id, s.home, s.away ORDER BY l.player_mlb_id
""")

q("pred_log 'Luis Garcia Jr.' key x mlb_id x team x prop_role", """
SELECT player_key, player_mlb_id, team,
   CASE WHEN prop_key LIKE 'pitcher_%' THEN 'P' ELSE 'B' END role, COUNT(*) n
FROM prediction_log WHERE player='Luis Garcia Jr.'
GROUP BY player_key, player_mlb_id, team,
   CASE WHEN prop_key LIKE 'pitcher_%' THEN 'P' ELSE 'B' END
ORDER BY player_key
""")

q("pred_log 'Max Muncy' key x mlb_id x team", """
SELECT player_key, player_mlb_id, team, COUNT(*) n
FROM prediction_log WHERE player='Max Muncy'
GROUP BY player_key, player_mlb_id, team ORDER BY player_key
""")

print("\nDONE")
