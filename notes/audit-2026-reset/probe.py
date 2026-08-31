"""READ-ONLY audit probe. COUNT/GROUP BY only. No writes."""
import db_store
from sqlalchemy import text

ok = db_store.promote_secrets_from_toml()
print("promote_secrets_from_toml:", ok)
if not db_store.enabled():
    raise SystemExit("SQL not enabled")

eng = db_store.get_engine()

def q(label, sql):
    print("\n=== " + label + " ===")
    try:
        with eng.connect() as c:
            for row in c.execute(text(sql)):
                print(" ", tuple(row))
    except Exception as e:
        print("  ERROR:", type(e).__name__, str(e)[:300])

# 1. prediction_log: key-prefix by sport + resolved
q("pred_log: sport / keyprefix / resolved counts", """
SELECT sport_key,
       CASE WHEN player_key IS NULL THEN 'NULL'
            WHEN player_key LIKE 'mlb:%' THEN 'mlb'
            WHEN player_key LIKE 'name:%' THEN 'name'
            ELSE 'other' END AS keypfx,
       resolved, COUNT(*) AS n
FROM prediction_log
GROUP BY sport_key,
       CASE WHEN player_key IS NULL THEN 'NULL'
            WHEN player_key LIKE 'mlb:%' THEN 'mlb'
            WHEN player_key LIKE 'name:%' THEN 'name'
            ELSE 'other' END,
       resolved
ORDER BY sport_key, keypfx, resolved
""")

# 2. baseball pred_log by season year + keyprefix (resolved only)
q("pred_log baseball: season(year) / keyprefix (resolved=1)", """
SELECT LEFT(game_date,4) AS yr,
       CASE WHEN player_key LIKE 'mlb:%' THEN 'mlb'
            WHEN player_key LIKE 'name:%' THEN 'name' ELSE 'other' END AS keypfx,
       COUNT(*) AS n
FROM prediction_log
WHERE sport_key LIKE 'baseball%' AND resolved=1
GROUP BY LEFT(game_date,4),
       CASE WHEN player_key LIKE 'mlb:%' THEN 'mlb'
            WHEN player_key LIKE 'name:%' THEN 'name' ELSE 'other' END
ORDER BY yr, keypfx
""")

# 3. baseball pred_log resolved rows: player_mlb_id present vs NULL, by prop role
q("pred_log baseball resolved: prop-role x mlb_id presence", """
SELECT CASE WHEN prop_key LIKE 'pitcher_%' THEN 'pitcher'
            WHEN prop_key LIKE 'batter_%' THEN 'batter' ELSE 'other' END AS role,
       CASE WHEN player_mlb_id IS NULL THEN 'noid' ELSE 'hasid' END AS idp,
       COUNT(*) AS n
FROM prediction_log
WHERE sport_key LIKE 'baseball%' AND resolved=1
GROUP BY CASE WHEN prop_key LIKE 'pitcher_%' THEN 'pitcher'
              WHEN prop_key LIKE 'batter_%' THEN 'batter' ELSE 'other' END,
         CASE WHEN player_mlb_id IS NULL THEN 'noid' ELSE 'hasid' END
ORDER BY role, idp
""")

# 4. odds_line player_prop rows (calibration book-line source): mlb_id presence by year
q("odds_line player_prop baseball: year x mlb_id presence", """
SELECT LEFT(s.game_date,4) AS yr,
       CASE WHEN l.player_mlb_id IS NULL THEN 'noid' ELSE 'hasid' END AS idp,
       COUNT(*) AS n
FROM odds_line l JOIN odds_snapshot s ON l.snapshot_id = s.id
WHERE s.sport LIKE 'baseball%' AND l.bet_type='player_prop'
GROUP BY LEFT(s.game_date,4),
         CASE WHEN l.player_mlb_id IS NULL THEN 'noid' ELSE 'hasid' END
ORDER BY yr, idp
""")

# 5. odds_line player_prop by prop-role x mlb_id (cross-role visibility)
q("odds_line player_prop baseball: role x mlb_id presence", """
SELECT CASE WHEN l.prop_key LIKE 'pitcher_%' THEN 'pitcher'
            WHEN l.prop_key LIKE 'batter_%' THEN 'batter' ELSE 'other' END AS role,
       CASE WHEN l.player_mlb_id IS NULL THEN 'noid' ELSE 'hasid' END AS idp,
       COUNT(*) AS n
FROM odds_line l JOIN odds_snapshot s ON l.snapshot_id = s.id
WHERE s.sport LIKE 'baseball%' AND l.bet_type='player_prop'
GROUP BY CASE WHEN l.prop_key LIKE 'pitcher_%' THEN 'pitcher'
              WHEN l.prop_key LIKE 'batter_%' THEN 'batter' ELSE 'other' END,
         CASE WHEN l.player_mlb_id IS NULL THEN 'noid' ELSE 'hasid' END
ORDER BY role, idp
""")

# 6. distinct name-keyed resolved baseball players (potential ambiguity surface)
q("pred_log baseball resolved name-keyed: distinct player_key count", """
SELECT COUNT(DISTINCT player_key) AS distinct_name_keys,
       COUNT(*) AS rows
FROM prediction_log
WHERE sport_key LIKE 'baseball%' AND resolved=1 AND player_key LIKE 'name:%'
""")

# 7. total prediction_log rows + resolved
q("pred_log totals", """
SELECT sport_key, COUNT(*) AS total, SUM(CASE WHEN resolved=1 THEN 1 ELSE 0 END) AS resolved
FROM prediction_log GROUP BY sport_key ORDER BY sport_key
""")

print("\nDONE")
