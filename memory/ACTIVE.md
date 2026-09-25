---
name: active
description: In-flight work only — current state + next candidates, each linking to its domain file. When an item ships, fold the durable outcome into its domain and delete it here; do not let this file accrete history.
metadata:
  type: project
---

# ACTIVE — in-flight work

One line per open item. When an item ships, delete it here and fold the durable outcome into its domain file (`one fact, one place`). **Last reconciled 2026-09-24** — memory maintenance pass: folded the shipped Astra audit, bonus harvest, and NFL-props-model arc into their domain files + the new [[bonus-strategy]]; slimmed ACTIVE from 86 mega-line blockquotes to this. History lives in the domain files + git, not here.

## Current state (2026-09-24)
- **Product = the bonus/promo +EV engine** — the project's ONE real edge; live on Streamlit Cloud (💰 Bonuses + 🎰 Parlays). Thesis / system / schema / fixes → [[bonus-strategy]].
- **NFL in-season** (Week 3, 2026). Team margin model + 8 frozen props models shipped, both calibrated fair-value + opportunity-abstain tools — **no straight-bet edge** (efficient at the close) → [[modeling-and-calibration]], [[edges-and-backtests]].
- **MLB parked → spring 2027** (efficient-for-us on the clean corpus; Doug doesn't bet the tail) → [[edges-and-backtests]].
- **Infra healthy:** Azure SQL system-of-record; provenance stamping live (F28); full test suite runs order-independently green (`python -m unittest discover`); `memory_lint.py` for memory hygiene → [[data-and-architecture]].

## Recently shipped (outcomes folded to domain files; pointers only)
- **Astra audit F01–F28 COMPLETE + live-verified** (origin/main): concurrency CAS (F23, verified under 12-way Azure load), prediction provenance + replay (F28 P1/P2, 49/49 on the live corpus), auth gate (F20). F28 P3 deferred by design. → [[data-and-architecture]] + git + notes/AUDIT_F28_PROVENANCE_DESIGN_2026-09-21.md.
- **Bonus-engine harvest + fixes** (2026-09-22→24): manual parlay-add, boosted-straight grading, anytime-TD grading, SGP min-total-odds bugfix, min_odds_leg default -300 → [[bonus-strategy]].
- **Hermetic test sweep** (2026-09-24): 2101 tests order-independent green; fixed 1 secret-leaker (test_coherence_flags) + 6 mirror-leakers.

## Next candidates (Doug's call which to pick up)
- **NFL prop grading → nflverse cutover (SCOPED 2026-09-25, not built)** — replace the ESPN per-player gamelog path with the already-live `nfl_opportunity_serving._load` (bulk nflverse player-week parquet) + `nfl_schedule` games.csv for week/scores; kills the per-player fetch loop AND the whole class of ESPN grading bugs (only-YDS+TD cache, name→id ambiguity). The hard part (dep-free live nflverse fetch on Cloud) already exists. Full scope + prop→column map + parity gate: `notes/NFL_GRADING_NFLVERSE_CUTOVER_2026-09-25.md`. → [[data-and-architecture]].
- **NBA before tip-off (~late Oct 2026)** — extend the bonus engine to NBA (nba_api warehouse + opportunity gate + calibrated de-vig legs) → [[bonus-strategy]], [[wishlist]].
- **Transient soft-line logger** — the ONE untested edge hypothesis (catch momentary pre-game mispricings; needs live minute-level polling + modest Odds-API credits) → [[edges-and-backtests]].
- **Bonus polish** — boosted-straight capture at Value-Finder submit time; retroactive boost on settled bets → [[bonus-strategy]].
- **F28 P3** — validation-domain partitioned replay reports (deferred).
- **MLB spring-2027 loose end** — recalibrate over-confident live `pitcher_strikeouts` (cv 0.266 > 0.25; `refit --recalibrate`, ~+0.0126 Brier) + audit other un-recalibrated props → [[edges-and-backtests]], [[modeling-and-calibration]].

## Owner action items
- ✅ **NFL forward-grading fix DONE + VERIFIED (2026-09-25, `61a7510`):** migration ran + `forward_tracker --resolve` re-graded **165/165 NFL rows, 0 stuck**; spot-checks confirm CORRECT actuals (Goff pass_yds 327 not the old ambiguous −1; Stafford pass_attempts 31 / completions 22; Nabers receptions 1 — all match the live gamelog). Count props (receptions/attempts/completions) now gradable; yardage/TD no longer mis-graded. Root cause: the `nfl_gamelog` cache stored only YDS+TD. → [[data-and-architecture]] gamelog note.
- ✅ **Forward-grading perf DONE + PUSHED (2026-09-25, `e10255d`):** `_load_player_gamelog` memoized (resolve loop 113.7s→0.1s, `80f64f3`) + `MAX_RESOLVE_PER_LAUNCH` 80→250 so a single hourly in-app pass drains a full slate. Decided AGAINST an external scheduler (a GitHub Action cron was built then dropped) — the memo made the in-app daemon fast enough. Residual caveat = Community Cloud still sleeps when idle, so grading only runs during active sessions; but one fast pass on the next visit now catches the whole backlog. → [[data-and-architecture]] gamelog note.
- None else blocking (app_password set, Cloud rebooted, SQL DDL applied). Optional: `python memory_lint.py` periodically for memory hygiene.
