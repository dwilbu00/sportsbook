"""READ-ONLY probe 2: resolved_at timeline + name-keyed row detail."""
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

# resolved_at month distribution for baseball resolved rows
q("pred_log baseball resolved: resolved_at YYYY-MM x keyprefix", """
SELECT LEFT(resolved_at,7) AS mon,
       CASE WHEN player_key LIKE 'mlb:%' THEN 'mlb'
            WHEN player_key LIKE 'name:%' THEN 'name' ELSE 'other' END AS keypfx,
       COUNT(*) AS n
FROM prediction_log
WHERE sport_key LIKE 'baseball%' AND resolved=1
GROUP BY LEFT(resolved_at,7),
         CASE WHEN player_key LIKE 'mlb:%' THEN 'mlb'
              WHEN player_key LIKE 'name:%' THEN 'name' ELSE 'other' END
ORDER BY mon, keypfx
""")

# resolved rows graded BEFORE 2026-08-06 (role-gate on fallbacks) with outcome set
q("pred_log baseball resolved+graded (outcome in 0/1): before vs after 2026-08-06", """
SELECT CASE WHEN resolved_at < '2026-08-06' THEN 'before_0806' ELSE 'on_after_0806' END AS era,
       COUNT(*) AS n
FROM prediction_log
WHERE sport_key LIKE 'baseball%' AND resolved=1 AND outcome IN (0,1)
GROUP BY CASE WHEN resolved_at < '2026-08-06' THEN 'before_0806' ELSE 'on_after_0806' END
ORDER BY era
""")

# the 22 distinct name-keyed players (resolved) -- list names + prop role + outcome presence
q("pred_log baseball resolved name-keyed rows: player, role, count", """
SELECT player,
       CASE WHEN prop_key LIKE 'pitcher_%' THEN 'P' WHEN prop_key LIKE 'batter_%' THEN 'B' ELSE '?' END AS role,
       COUNT(*) AS n
FROM prediction_log
WHERE sport_key LIKE 'baseball%' AND resolved=1 AND player_key LIKE 'name:%'
GROUP BY player,
       CASE WHEN prop_key LIKE 'pitcher_%' THEN 'P' WHEN prop_key LIKE 'batter_%' THEN 'B' ELSE '?' END
ORDER BY n DESC
""")

# how many graded rows have NULL outcome (push/void) vs 0/1
q("pred_log baseball resolved: outcome value distribution", """
SELECT CASE WHEN outcome IS NULL THEN 'NULL' ELSE CAST(outcome AS varchar) END AS oc, COUNT(*) AS n
FROM prediction_log WHERE sport_key LIKE 'baseball%' AND resolved=1
GROUP BY CASE WHEN outcome IS NULL THEN 'NULL' ELSE CAST(outcome AS varchar) END
ORDER BY oc
""")

# earliest / latest resolved_at + game_date
q("pred_log baseball date range", """
SELECT MIN(game_date), MAX(game_date), MIN(resolved_at), MAX(resolved_at)
FROM prediction_log WHERE sport_key LIKE 'baseball%' AND resolved=1
""")

print("\nDONE")
