"""Check the FIXED, chunked player-prop read (warehouse.load_prop_lines).
Run from the deploy dir:  python diag_props.py   (safe to delete afterward).

NOTE: this deliberately does NOT call db_store.player_prop_lines('baseball_mlb')
with no args — that's the OLD unchunked full-table read (~1.5M rows) that hangs.
The refit uses load_prop_lines, which reads one season at a time."""
import time
import inspect

import db_store
import warehouse

db_store.promote_secrets_from_toml()
print("SQL enabled:", db_store.enabled())

# Is the scale fix present on THIS machine? (exclude_early is new in commit 802814e)
has_fix = "exclude_early" in inspect.signature(
    db_store.player_prop_lines).parameters
print("scale fix present (exclude_early param):", has_fix)
if not has_fix:
    print("  -> This machine does NOT have the fix. Pull the latest code "
          "(through commit 802814e) here, then re-run.")

t0 = time.time()
rows = warehouse.load_prop_lines("baseball_mlb", prop_keys=["batter_hits"])
dt = time.time() - t0
print(f"load_prop_lines(batter_hits) rows: {len(rows):,}  in {dt:.1f}s")
if rows:
    r = rows[0]
    print("sample:", {k: r.get(k) for k in
          ("game_date", "player", "prop_key", "line",
           "over_price", "under_price")})
    from collections import Counter
    print("by year:", Counter((str(r.get("game_date"))[:4] for r in rows))
          .most_common())
