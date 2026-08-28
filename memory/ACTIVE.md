---
name: active
description: In-flight work only — what's being worked on now, linking to the domain files. Move an item to its domain when done; do not let this file accrete history.
metadata:
  type: project
---

# ACTIVE — in-flight work

One line per open item, grouped by domain. When an item ships, delete it here and fold the durable outcome into its `[[domain]]` file (result-supersedes-idea).

## [[edges-and-backtests]]
- **Forward-track the sharpened coherence run-line** — The in-app forward log (bet_type runline_coherence, dog+1.5 at [0.60,0.70) fav band) is the confirmation instrument for a marginal t≈1.66 edge — bet small, let the sample grow; part of [[edges-and-backtests]]
- **Forward-monitor cv_floor (earned_runs×CV) + confirm fillability** — Edge is LIVE (method E + cv_floor 1.3); remaining = forward-monitor for decay + confirm DK/FanDuel book volatile SPs at size before scaling; part of [[edges-and-backtests]]
- **Pick the next strategic direction (both edge theses exhausted)** — Doug to steer: Monte Carlo one-distribution sim for correlation/coherence edges, soft/niche markets, NBA/NFL, or consolidate+operate — sharp-staleness + cross-market probes are spent; part of [[edges-and-backtests]]
- **Variance-mispricing → team markets (owner hypothesis, unbuilt)** — Test whether the validated SP-ER-CV feeds team totals/dog-ML (fat right tail underpriced); needs a new team-market volatility diagnostic that doesn't exist yet; part of [[edges-and-backtests]]

## [[data-and-architecture]]
- **5M-credit historical-odds backfill** — ACTIVE + staged: tooling built + spend-review cleared; per-(sport×category) dry-run→confirm→--apply runbook is owner's to execute before the 5M reverts month-end — see [[data-and-architecture]]
- **Clean-slate relaunch (archive-then-epoch)** — archive_app_data.py built + reviewed; execute at relaunch with app STOPPED + explicit confirm (zero bankroll, prune thin 2026 live odds, clean early+close corpus load) — [[data-and-architecture]]
- **Deep real-line re-refit + sweep bulk-index rewrite** — Join/read-scale fixes shipped; IN PROGRESS at compaction = rewrite backtest.fetch_player_data MLB path to get_calib_gamelogs_bulk + re-point ~6 test mocks + commit, then owner runs the --real-lines re-refit → --promote — [[data-and-architecture]]
- **Commit C legacy re-stamp RUN** — Code complete + pushed (gate retired); owner still to run restamp_legacy_ids.py --apply --odds → mlb_warehouse.py --backfill-game-pk --apply to correct corpus drift (SQL backup first) — [[data-and-architecture]]
- **Pending prod DDL / index creates** — Owner hand-runs guarded DDL on Azure: statcast tables+seed, perf indexes (ONLINE=ON), source ALTER+coverage views, pitcher BB/BF/HR/HBP/GS ALTERs + re-derive pitcher facts — [[data-and-architecture]]

## [[modeling-and-calibration]]
- **value_gate auto-tune into offline refit** — OPEN: fold a confirmation-gated --gate-diag selection into the refit as advisory + --promote (never live-auto; threshold argmax overfits unlike smooth Platt) — [[modeling-and-calibration]]
- **bet_selector 'balanced' metric** — OPEN: honest-small-EV markets (moneyline) get crowded out of top-N by miscalibrated high-EV props even after the ROI-gate fix; consider a balanced ranking metric — [[modeling-and-calibration]]
- **Tune EV floor 4%→3% for DK/FD** — OPEN: value_gate EV floor was chosen on consensus prices (optimistic vs real books); may lower to 3% for DK/FanDuel — [[modeling-and-calibration]]
- **Pitcher props: leave synthetic_sweep + seed pitcher Platt** — OPEN remedy for the forward-Brier gap: grow pitcher REAL-line obs so they exit synthetic-line fits, and seed/enable online Platt for pitcher props (currently zero fits) — [[modeling-and-calibration]]
- **Online Platt Blob-mode hardening** — OPEN (low pri): seed-pristine invariant + per-key merge are SQL-only; mirror per-key merge into the Blob branch + make save guard mode-agnostic before any non-SQL deployment — [[modeling-and-calibration]]
- **Verify bankroll DDL ran in prod** — VERIFY: sql/schema.sql for bankroll_ledger + app_settings must be hand-run in prod or Kelly sizes against $0 — confirm it executed — [[modeling-and-calibration]]
- **Market-upset (situational) study for team markets** — NEXT team-market step (mostly edge-strategy domain): global knob-tuning exhausted, hunt where the close is systematically wrong using calibration-junction disagreements as per-game features — [[team-market-audit]]

## [[preferences]]
- **STEP 2 recalibration refit — the release gate** — OPEN: Doug runs on his machine `refit_calibration.py --sport mlb --seasons 2024,2025,2026 --props batter_total_bases,batter_strikeouts,pitcher_outs` (stages a candidate), pastes --diff; I review for the INCUMBENT-HYSTERESIS trap (splice winners back as incumbents first) + whether TB moves A→E confirmed → if so un-suppress TB + re-check batter_strikeouts, THEN --promote. Statcast SQL cache prereq. See [[calibration-candidate-workflow]]. ⚠ the 2026-08-19 'app is shut down until this completes' premise may be stale — verify against current live state before assuming.
