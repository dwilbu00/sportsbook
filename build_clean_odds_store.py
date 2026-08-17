"""
Assemble a CLEAN multi-season team-market odds store for backtesting, free of
the SBR-poisoned closes.

Background
----------
The team-market backtest reads odds from ONE source: either the SQL warehouse
(`--source warehouse`) or a local historical_odds JSON store (`--source store
--store-label X`). Outcomes/schedules always come from the StatsAPI warehouse;
the store only supplies the prices.

The warehouse's 2023 -> ~2025-08-16 closes were ingested from SBR, which we
found systematically poisons ROI backtests (same games/model: ML SBR -8.58% vs
clean Odds-API +1.19%, Brier identical). We reloaded clean Odds-API DK closes
for those windows into labeled stores (api2024, api2025). 2026+ in the warehouse
is already clean (live-captured, pre-close).

Why a store and NOT a warehouse promote
---------------------------------------
warehouse.load_team_market_store picks each event's CLOSING snapshot (nearest
at-or-before commence via _closing_sort_key). If we promoted the reloaded ~noon
API lines alongside the existing SBR closes, SBR (closer to commence) would keep
winning that selection — the poison would never be replaced. A dedicated clean
store sidesteps this entirely and keeps the live warehouse untouched.

What this does
--------------
Merges the reloaded API stores (default api2024 + api2025) with a warehouse
export of the live-captured seasons (>= --warehouse-from, default 2026-01-01)
into a single labeled store (default label 'apiclean'). Non-destructive: reads
the source stores + warehouse; writes ONLY the new labeled file. Idempotent —
re-run any time (e.g. after reloading 2023 post-refill) to rebuild.

Usage
-----
    python build_clean_odds_store.py --sport mlb --dry-run
    python build_clean_odds_store.py --sport mlb            # writes apiclean

Then backtest on it:
    python backtest.py --team-market --sport mlb --engine live \
        --source store --store-label apiclean --limit 100000
"""
import argparse
import os

import historical_odds as store_mod

SPORT_MAP = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
}


def _year_of(key):
    """Year prefix of a game_key '<YYYY-MM-DD>|away @ home' (or '' if unparseable)."""
    return (key or "")[:4]


def _breakdown(games):
    """{year: count} over a games dict, keyed by the game_key date prefix."""
    out = {}
    for k in games:
        out[_year_of(k)] = out.get(_year_of(k), 0) + 1
    return dict(sorted(out.items()))


def main():
    p = argparse.ArgumentParser(
        description="Build a clean multi-season team-market odds store.")
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), default="mlb")
    p.add_argument("--label", default="apiclean",
                   help="Output store label (historical_odds/<key>__<label>.json).")
    p.add_argument("--store-sources", default="api2024,api2025",
                   help="Comma-separated labeled stores to merge (reloaded API "
                        "closes). Default: api2024,api2025.")
    p.add_argument("--warehouse-from", default="2026-01-01",
                   help="Include warehouse games with an ET play date >= this "
                        "(the clean, live-captured seasons). Empty '' to skip "
                        "the warehouse export entirely.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report the plan + per-year counts without writing.")
    args = p.parse_args()

    sport_key = SPORT_MAP[args.sport]
    print(f"\n=== Build clean team-market store: {sport_key} "
          f"-> label '{args.label}' ===")

    merged = {}          # game_key -> entry
    collisions = []      # (key, from_source, over_source)
    provenance = {}      # game_key -> source label (for collision reporting)

    # ── Reloaded API stores ────────────────────────────────────────────────
    src_labels = [s.strip() for s in args.store_sources.split(",") if s.strip()]
    for lbl in src_labels:
        store = store_mod.load_store(sport_key, lbl)
        games = store.get("games") or {}
        if not games:
            print(f"  [store {lbl}] EMPTY or missing "
                  f"({os.path.basename(store_mod.store_path(sport_key, lbl))}) "
                  f"-- skipping.")
            continue
        for k, entry in games.items():
            if k in merged:
                collisions.append((k, lbl, provenance.get(k, "?")))
                continue
            merged[k] = entry
            provenance[k] = lbl
        print(f"  [store {lbl}] {len(games)} games  {_breakdown(games)}")

    # ── Warehouse export (clean, live-captured seasons) ─────────────────────
    if args.warehouse_from:
        try:
            import db_store
            db_store.promote_secrets_from_toml()
        except Exception:
            pass
        try:
            import warehouse
            wh = warehouse.load_team_market_store(sport_key)
            wh_games = {k: v for k, v in (wh.get("games") or {}).items()
                        if _year_of(k) and k[:10] >= args.warehouse_from}
        except Exception as e:
            print(f"  [warehouse] export FAILED ({type(e).__name__}: {e}); "
                  "continuing with store sources only.")
            wh_games = {}
        added = 0
        for k, entry in wh_games.items():
            if k in merged:
                collisions.append((k, "warehouse", provenance.get(k, "?")))
                continue
            merged[k] = entry
            provenance[k] = "warehouse"
            added += 1
        print(f"  [warehouse >= {args.warehouse_from}] {added} games  "
              f"{_breakdown(wh_games)}")

    if collisions:
        print(f"\n  [!] {len(collisions)} game-key collision(s) across sources "
              "(kept the first, earlier-listed source):")
        for k, dropped, kept in collisions[:10]:
            print(f"        {k}  (dropped {dropped}, kept {kept})")
        if len(collisions) > 10:
            print(f"        ... and {len(collisions) - 10} more.")

    print(f"\n  === Merged total: {len(merged)} games  {_breakdown(merged)} ===")

    if args.dry_run:
        print("\n  [dry-run] Nothing written. Re-run without --dry-run to save.")
        return
    if not merged:
        print("\n  Nothing to write (no games merged). Done.")
        return

    out = {"sport_key": sport_key,
           "bookmaker": "draftkings (clean Odds-API + warehouse, pre-close)",
           "markets": "h2h,spreads,totals",
           "games": merged}
    store_mod.save_store(sport_key, out, args.label)
    print(f"\n  Wrote {store_mod.store_path(sport_key, args.label)}")
    print(f"  Backtest with:  python backtest.py --team-market --sport "
          f"{args.sport} --engine live --source store --store-label "
          f"{args.label} --limit 100000")


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
