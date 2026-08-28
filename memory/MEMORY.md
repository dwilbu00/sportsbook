# Memory index

Auto-memory for the MLB (+NBA/NFL) sportsbook betting model. Owner = Doug; assistant = Cal.

**Structure (consolidated 2026-08-28):** 4 domain files hold durable knowledge, `ACTIVE.md` holds in-flight work, `wishlist.md` is the parking lot. Read the domain(s) relevant to the task; read `ACTIVE.md` to resume.

**Maintenance discipline:**
- One fact, one place. When a **result supersedes an idea** ("test X" vs "tested X → Y"), delete the idea, keep the result — so it's never revisited.
- Put **in-flight** work in `ACTIVE.md`; when it ships, delete it there and fold the outcome into its domain file.
- `wishlist.md` is **append/transfer-only** — add ideas or move them out to a domain/ACTIVE when worked; never expand a wishlist entry in place.
- Drop stale/superseded/abandoned notes (leave a one-line tombstone only if it'd otherwise be re-proposed).

## Files
- [preferences](preferences.md) — How to work with Doug (owner): who he is, commit/push rules, betting books (DK+FD), spend confirmation, backtest handoff, Alpha status, and his defaults-audit methodology.
- [ACTIVE](ACTIVE.md) — in-flight work only; read this to resume. Links each item to its domain.
- [edges-and-backtests](edges-and-backtests.md) — What the edge hunt found: the validated coherence run-line + cv_floor variance edge, the exhausted sharp-staleness/mean-edge nulls, and the warehouse backtest tooling.
- [data-and-architecture](data-and-architecture.md) — THE SYSTEM: Azure SQL warehouse + MLB StatsAPI medallion (ESPN fully removed for MLB), MLBAM/game_pk identity, as-of feature stores, additive runs model, CLV, and the active 5M-credit odds backfill + relaunch reset.
- [modeling-and-calibration](modeling-and-calibration.md) — The model itself: live calibration state, refit/candidate workflow, method A-E bake-off outcomes, Kelly sizing, online Platt, team-market audit verdict, cross-sport parity, and why forward Brier lagged backtest.
- [wishlist](wishlist.md) — parked owner ideas; append/transfer-only (never expanded per-item as work is done).
