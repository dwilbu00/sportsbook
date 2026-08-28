# Project: MLB (+NBA/NFL) sportsbook betting model

Owner **Doug** is the sole user and developer; the assistant goes by **Cal**.

## Project memory — read this first (it's in the repo so it travels across machines)

Persistent memory lives in **`memory/`** (version-controlled here, not in `~/.claude`,
so any machine gets it via `git pull`). At the start of substantive work:

1. Read `memory/MEMORY.md` — the index + maintenance discipline.
2. Read `memory/ACTIVE.md` — what's in flight (resume here).
3. Read the relevant domain file(s): `memory/edges-and-backtests.md`,
   `memory/data-and-architecture.md`, `memory/modeling-and-calibration.md`,
   `memory/preferences.md`. `memory/wishlist.md` = parked ideas.

**Write/maintain project memory in `memory/` (NOT `~/.claude`).** Discipline:
- One fact, one place. When a **result supersedes an idea**, delete the idea, keep the result.
- In-flight work goes in `ACTIVE.md`; fold it into a domain file when it ships.
- `wishlist.md` is **append/transfer-only** — never expand an entry in place as work is done.
- Commit memory changes like code (they're in the repo now).

## Standing rules (safety-critical — full versions in `memory/preferences.md`)

- **Commit** proactively to `main` after verified changes; **NEVER `git push`** (Doug pushes).
- Bets execute at **DraftKings AND FanDuel** only; all other books (Pinnacle, etc.) are
  analysis-only — never recommend/display their prices or size off them.
- **Confirm before any real spend** (Odds API credits, paid backfills): one explicit
  "firing now — ~N credits, go?" beat even after a general go-ahead. Dry-runs / reads are free.
- Doug runs **backtests on his own (faster) machine** — write the exact command block +
  what to look for, he pastes the output back.
- App is **Alpha** (single user, ship fast) — but the **never-destroy-irreplaceable-data**
  constraint is absolute (wagers, bankroll ledger, prediction/calibration corpus, learned
  fits → translate/re-key in place, never drop).
- End commit messages with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer.
