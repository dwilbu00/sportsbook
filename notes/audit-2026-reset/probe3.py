"""READ-ONLY probe 3: wagers, market_pred, odds warehouse ranges."""
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

q("wagers: sport x mlb_id presence (player bets) + date range", """
SELECT sport_key,
       CASE WHEN player IS NULL OR player='' THEN 'noplayer'
            WHEN player_mlb_id IS NULL THEN 'noid' ELSE 'hasid' END AS idp,
       COUNT(*) AS n, MIN(game_date), MAX(game_date)
FROM wagers GROUP BY sport_key,
       CASE WHEN player IS NULL OR player='' THEN 'noplayer'
            WHEN player_mlb_id IS NULL THEN 'noid' ELSE 'hasid' END
ORDER BY sport_key, idp
""")

q("market_prediction_log: sport x count + game_date range", """
SELECT sport_key, COUNT(*) AS n, MIN(game_date), MAX(game_date)
FROM market_prediction_log GROUP BY sport_key ORDER BY sport_key
""")

q("odds_snapshot: sport x count + game_date range", """
SELECT sport, COUNT(*) AS n, MIN(game_date), MAX(game_date)
FROM odds_snapshot GROUP BY sport ORDER BY sport
""")

q("odds_line total by bet_type (baseball)", """
SELECT l.bet_type, COUNT(*) AS n
FROM odds_line l JOIN odds_snapshot s ON l.snapshot_id=s.id
WHERE s.sport LIKE 'baseball%'
GROUP BY l.bet_type ORDER BY l.bet_type
""")

# name-collision surface: distinct normalized names that map to >1 distinct mlb_id in odds_line
q("odds_line baseball: player names carrying >1 distinct mlb_id (true namesake collisions)", """
SELECT player, COUNT(DISTINCT player_mlb_id) AS ids
FROM odds_line l JOIN odds_snapshot s ON l.snapshot_id=s.id
WHERE s.sport LIKE 'baseball%' AND l.bet_type='player_prop' AND l.player_mlb_id IS NOT NULL
GROUP BY player HAVING COUNT(DISTINCT player_mlb_id) > 1
""")

# pred_log: same player NAME carrying both a name: and mlb: key (enrichment gap)
q("pred_log baseball: names carrying BOTH name-key and mlb-key rows", """
SELECT player, COUNT(*) n,
   SUM(CASE WHEN player_key LIKE 'mlb:%' THEN 1 ELSE 0 END) mlbk,
   SUM(CASE WHEN player_key LIKE 'name:%' THEN 1 ELSE 0 END) namek
FROM prediction_log WHERE sport_key LIKE 'baseball%'
GROUP BY player
HAVING SUM(CASE WHEN player_key LIKE 'mlb:%' THEN 1 ELSE 0 END) > 0
   AND SUM(CASE WHEN player_key LIKE 'name:%' THEN 1 ELSE 0 END) > 0
""")

print("\nDONE")
