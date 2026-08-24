"""Throwaway diagnostic: why does warehouse.load_prop_lines('baseball_mlb')
return 0 when odds_line clearly holds 1.5M MLB prop lines? Run from the deploy
dir:  python diag_props.py   (safe to delete afterward)."""
import traceback

import db_store
import warehouse
from sqlalchemy import text

db_store.promote_secrets_from_toml()
print("db_store.enabled():", db_store.enabled())
try:
    print("warehouse._sql() :", warehouse._sql())
except Exception as e:
    print("warehouse._sql() : ERROR", type(e).__name__, e)
try:
    print("storage_backend  :", warehouse.storage_backend())
except Exception as e:
    print("storage_backend  : ERROR", type(e).__name__, e)

eng = db_store.get_engine()
print("engine url       :", str(eng.url))

sql = ("SELECT COUNT(*) FROM odds_line ol "
       "JOIN odds_snapshot os ON ol.snapshot_id = os.id "
       "WHERE os.sport = 'baseball_mlb' AND ol.bet_type = 'player_prop'")
try:
    with eng.connect() as c:
        n = c.execute(text(sql)).scalar()
    print("prop lines via THIS engine:", n)
except Exception:
    print("direct count raised:")
    traceback.print_exc()

print("--- db_store.player_prop_lines('baseball_mlb') ---")
try:
    rows = db_store.player_prop_lines("baseball_mlb")
    print("player_prop_lines rows:", len(rows))
    if rows:
        r = rows[0]
        print("sample:", {k: r.get(k) for k in
              ("event_id", "player", "prop_key", "point", "direction",
               "commence_time", "snapshot_id", "captured_at")})
except Exception:
    traceback.print_exc()

print("--- warehouse.load_prop_lines('baseball_mlb') ---")
try:
    lp = warehouse.load_prop_lines("baseball_mlb")
    print("load_prop_lines rows:", len(lp))
    if lp:
        print("sample:", lp[0])
except Exception:
    traceback.print_exc()
