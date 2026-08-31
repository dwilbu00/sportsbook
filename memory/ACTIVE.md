---
name: active
description: In-flight work only — what's being worked on now, linking to the domain files. Move an item to its domain when done; do not let this file accrete history.
metadata:
  type: project
---

# ACTIVE — in-flight work

One line per open item. When an item ships, delete it here and fold the durable outcome into its `[[domain]]` file (result-supersedes-idea). Last reconciled 2026-08-28 (verified against code + owner-confirmed prod state).

## Ongoing monitors (not tasks — just accumulate + watch)
- **Coherence run-line forward-track** — the in-app forward log (`runline_coherence`, dog +1.5 at [0.60,0.70) fav band) is the confirmation instrument for the marginal t≈1.66 edge. Bet small, let the sample grow. [[edges-and-backtests]]
- **cv_floor forward-monitor** — LIVE (earned_runs method E + cv_floor 1.3); watch for decay + confirm DK/FanDuel book volatile SPs at size before scaling. [[edges-and-backtests]]

## [[edges-and-backtests]]
- **prop_roi RUN (2026-08-28) → over-bias confirmed but vig-eaten; 1 marginal cell.** Verdict in [[edges-and-backtests]]: only `batter_strikeouts UNDER 1.5` (+2.2%, t=1.93, replicates) crosses +EV; ER UNDER 2.5 breakeven; everything else vig-negative. OPEN owner decision: forward-track K-under-1.5 small (borderline, clears the coherence t≈1.66 precedent) or record + move on. Don't build a system on it.
- **Line-timing DONE (2026-08-31) — result in [[edges-and-backtests]].** ML: bet EARLY (~2% CLV paired, `+++` 3/3 seasons); RL/totals unanswerable offline (n=4/6, warehouse captures them near game-time only). OPEN follow-ups: (1) adopt the ML-early timing rule in the live workflow; (2) **coherence RL timing (Doug's plan 2026-08-31): NO code needed now — the auto-log already banks the EARLY side** (`recalibration.build_market_prediction_rows` runline_coherence rows carry `ts`=analysis time, `price`=early dog+1.5 price, `point`, `commence_time`, event identity). Let it accumulate; WHEN sample sufficient (~few hundred flagged games), do a ONE-TIME **targeted** Odds-API historical pull of CLOSING dog+1.5 for only those game_pks (small confirmed spend, not a full backfill) + a CLV grader → per-bet early-vs-close ΔROI on the REAL live edge. Streamlit has no scheduler, so this defers the paid close capture instead of cron-ing it. Parked directions still open: bet-disqualification (partly folded into datamine). Monte Carlo = no-go.
- **Upset datamine (Doug's pick 2026-08-31) — BUILT, awaiting run (commit a4bbb58).** `scenario_backtest.py --scenario upset_datamine`: tags each game with market-blind factors from warehouse meta (home-plate UMPIRE, doubleheader, favorite/underdog days-rest, favorite/underdog travel) + stratifies realized ROI of base bets (dog ML, dog +1.5, total over/under) by factor bucket; per-season replication = honesty gate (★ = +ROI & positive every season at n>=min). Batch 1 = schedule/venue/umpire (all clean from mlb_game, no park/weather-table dependency). RUN: `python scenario_backtest.py --scenario upset_datamine --seasons 2024,2025,2026` (add `--datamine-min-n N`). Read a ★ cell (bet it) or a clearly -ROI replicating cell (disqualify) as the deliverable; VERIFY any ★ with a finer cut (umpire cells overfit). Batch 2 (if batch 1 promising): park/weather run-env, day-after-night (needs tz), bullpen fatigue, lineup-vs-typical.
- **Sharpen coherence w/ stable-favorite-SP filter (Doug's pick 2026-08-31) — DONE + SHIPPED LIVE (commit a5aabad).** favorite-SP-CV_MAX=1.0 gate live in `coherence_flags` (CLI + in-app), fails closed, tested. Full detail in [[edges-and-backtests]] (★★). Now a MONITOR: the in-app `runline_coherence` forward log tracks the SHARPENED (stable-SP) edge — watch it grow; expected ~+11% but bet small (t-ceiling ~3.37 on 3 backtested seasons). Verify live once: `python coherence_flags.py --live` shows the gate line + skip counts. FOLLOW-UP still open: the free early/close RL capture for the timing question (see line-timing item above).

## [[modeling-and-calibration]]
- **value_gate auto-tune** — fold a confirmation-gated `--gate-diag` selection into the offline refit as advisory + `--promote` (never live-auto; threshold argmax overfits vs smooth Platt). `refit_calibration` still never calls `save_value_gate`.
- **EV floor 4%→3% for DK/FD** — `value_gate.ev_floor` is still 0.04 (chosen on optimistic consensus prices); A/B 0.04 vs 0.03 on DK/FanDuel fills via `--gate-diag`.
- **Pitcher props: seed Platt + exit synthetic-line fits** — grow pitcher REAL-line obs so they leave synthetic-line fits, and seed/enable online Platt for pitcher props (zero fits today) → closes the forward-Brier gap.
- **Online Platt Blob-mode hardening (low pri)** — seed-pristine invariant + per-key merge are SQL-only; mirror per-key merge into the Blob branch + make save guard mode-agnostic before any non-SQL deploy.
