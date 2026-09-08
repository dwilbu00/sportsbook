# Codebase handoff for independent review (2026-09-08)

You're being asked to give an **impartial second opinion** on this sports-betting modeling repo —
both the **MLB** system (mature, the bulk of the code) and the **NFL** work (built this session).
The author (an AI assistant, "Cal") and the owner (Doug) are both deep in it and want fresh eyes —
especially on whether the conclusions are *sound* and the methodology is *honest*, not just whether
the code runs. **Please be adversarial.** The most useful thing you can do is find where we fooled
ourselves. The two big claims to stress-test are the same for both sports: **(a) is the model
leakage-free and validated out-of-sample honestly, and (b) is the "no durable edge / at the
efficient-market floor" verdict correct, or did we stop short / mis-measure?**

## Project in one line
A sportsbook betting model (MLB, now pivoting to NFL). Bets execute at DraftKings + FanDuel only;
all other books are analysis-only. Persistent project memory lives in **`memory/`** — read
`memory/MEMORY.md` (index), `memory/ACTIVE.md` (current state), and the domain files
`memory/modeling-and-calibration.md` + `memory/data-and-architecture.md` + `memory/edges-and-backtests.md`.
Those contain the full reasoning trail; this file is the review-oriented summary.

## What was built this session (NFL team model)
Clean-slate NFL data foundation on **nflverse** + a dedicated game-margin model, now **shipped live**.

Key files (all in `deploy/`):
- `nfl_schedule.py` — canonical `game_id` spine + fail-closed odds→game join (837/837 clean).
- `nfl_ingest.py` — offline nflverse→mirror-parquet ingest (all layers; pbp 2016-2025).
- `nfl_data.py` — dep-free mirror readers + `game_context()` (rest/bye/division/primetime).
- `nfl_epa.py` — team EPA (leakage-safe as-of), now reads the pbp mirror; `build_matchup_features`.
- `nfl_qb_asof.py` — starting-QB EPA layer (per-passer dropback EPA, starter projection).
- `nfl_injury_impact.py` — snap-based injury burden (offense EPA-out + defense/O-line counts).
- `nfl_model.py` — assembles + fits + predicts the live margin model.
- `nfl_totals.py` — total-points model (diagnostic, not shipped).
- `nfl_power.py` — opponent-adjusted EPA (**tested and rejected** — kept as documented negative).
- `nfl_market_scan.py`, `nfl_situational_scan.py`, `nfl_props_scan.py`, `nfl_edge_calibration.py`,
  `nfl_qb_edge.py`, `nfl_injury_edge.py` — the diagnostic/edge-hunt harnesses.
- `test_nfl_epa.py` — leakage + abbreviation-parity guards.

**Live model** (`calibration/americanfootball_nfl.json` → `starter_adjustment.nfl_margin_model`):
`margin = 2.13(HFA) + 32.5·net_epa_edge + 10.9·qb_edge + 0.065·off_inj + 0.65·def_inj + 0.35·rest`,
sigma 13.12. Wired via `nfl_epa.build_matchup_features` → `analysis._predict_margin`.

## The central claims (verify these)
1. **The model is leakage-safe** — all features use only data strictly before each game.
   (`test_nfl_epa.py` checks EPA; please also scrutinize the QB, injury, and rest as-of paths.)
2. **Matchup granularity does not beat the coarse aggregate OOS** — tested 7 ways (opponent
   adjustment; offense/defense split; pass/rush phase split; position-aware injuries; interaction
   terms; a regularized λ-blend; Pythagorean/points). All neutral-to-worse. Conclusion: aggregate
   net-EPA + HFA + QB + aggregate-injuries + rest is the robust optimum.
3. **We cannot out-predict the sharp closing line** — margin RMSE 13.19 vs close 12.67; win-prob
   Brier 0.226 vs market 0.211, and *worse on the games where we disagree* (the conditional test).
4. Therefore the remaining edge, if any, is in **transient soft lines** (not out-predicting the
   close) and/or **NFL props** (the MLB machinery was props-side), not more team-model features.

## Please scrutinize (our likeliest self-deceptions)
- **OOS is thin.** Odds-graded tests are **3 seasons only** (Odds-API limit); leave-one-season-out
  = 3 folds. The 10-season tests are pbp/margin-only (no odds). Are conclusions over-claimed for
  the power available? Multiple-comparisons across ~a dozen tests — is any "win" just noise?
- **"Game-actual starter" in backtests.** The QB/model validation used the starter who actually
  played (public pre-game, so we argue it's not outcome leakage). Is that defensible, or does it
  flatter the model vs the live *projected* starter? (Live path = `projected_starter`; the gap we
  measured was small: 13.123 game-actual → 13.155 projected.)
- **One test had a real sign bug** (`nfl_power`/phase-matched, first pass showed 14.33 due to a
  flipped defense-EPA sign; corrected to ~13.33). We caught and fixed it — but it means the test
  harness deserves a careful read. Are the *other* edge/RMSE comparisons sign-correct?
- **The live wiring.** `analysis._predict_margin` now branches for NFL to use `nfl_pred_margin`.
  We smoke-tested `_predict_margin` directly and claim MLB is byte-identical — verify the full app
  path (`app.py` ~3137) and the backtest/`feature_store` fit==serve caching aren't broken.
- **Fixed sigma = 13.12** drives every win probability (Brier). Is a constant sigma appropriate,
  or should it vary (weather, spread magnitude)? Our Brier-vs-market comparison depends on it.
- **Devig + closing-line RMSE/Brier** — we compare to the DK closing line devigged via
  `r2_sharp.fair_two_way`. Is the "closing line RMSE 12.67 / Brier 0.211" benchmark computed fairly?
- **The props scan** killed a candidate as "DNP survivorship" (scratched low-line players dropped
  from grading inflate OVER). Verify that reasoning + the drill in `nfl_props_scan.py`.
- **Calibration props section is a known toy** (`player_pass_yds` ≡ `player_rush_yds`, byte-identical)
  — not yet rebuilt.

## How to run (Windows, `PYTHONIOENCODING=utf-8`)
- Tests: `python test_nfl_epa.py`
- Margin model fit + OOS: `python nfl_model.py --seasons 2023,2024,2025`
- QB OOS lift: `python backtest_nfl_epa.py --seasons 2023-2025 --oos-qb`
- Beat-the-close: `python nfl_qb_edge.py --seasons 2023,2024,2025`
- Injury OOS + beat-close: `python nfl_injury_edge.py --seasons 2023,2024,2025 --beat-close`
- Required-edge curve: `python nfl_edge_calibration.py --seasons 2023,2024,2025`
- Totals: `python nfl_totals.py --seasons 2023,2024,2025 --beat-close`
- Spine audit: `python nfl_schedule.py --audit --seasons 2023,2024,2025,2026`

nflverse data pulls are FREE (no paid API). Odds-API pulls cost credits — do not run those without
Doug's explicit go. The mirror parquets are in `warehouse_mirror_data/` (Git LFS).

---

# MLB system (the mature system — also in scope)

MLB is the older, larger part of the codebase and the source of most of the shared infrastructure
the NFL work reuses. It is currently **parked/efficient** — every candidate edge dissolved on a
clean data corpus (see below). Full reasoning is in `memory/modeling-and-calibration.md`,
`memory/edges-and-backtests.md`, `memory/data-and-architecture.md`; this is the review summary.

## What's there (MLB)
- **Props model** — the heavy statistical machinery. Per-prop residual method bake-off (A=raw,
  B=Gaussian, C=empirical-CDF, D=binomial P(≥k) distributional, E=NegBin), plus an **online Platt**
  sigmoid overlay (seed-as-Bayesian-prior, champion-gated). Live: `batter_hits` = pooled C +
  per-line-bucket **D+xBA blend**, `batter_strikeouts`=C, `pitcher_strikeouts`=C,
  `batter_total_bases`=E, `batter_rbis`=A, `pitcher_earned_runs`=E, `pitcher_outs`=A (suppressed).
- **Team markets** — shared `analysis._predict_margin` (recency + Pythagorean + starter/pitcher
  adjustment) → `prob_shrink` + `market_blend` calibration. Verdict: well-calibrated but **at the
  efficient-market variance floor (~0.24-0.25 Brier), no durable global edge**.
- **Bet sizing** — vig-aware fractional-Kelly (half-Kelly, 5% per-bet cap, 25% slate cap); durable
  bankroll ledger (`bankroll.py`, SQL/NDJSON, idempotent txn reconcile).
- **Data** — Azure SQL warehouse (2024-2026; 2023 purged) + mirror-parquet cache (Git LFS),
  leakage-safe as-of primitives (`savant_history`, `pitcher_asof`, `book_line_calibration`).
- **Refit workflow** — candidate-staging (`refit_calibration.py` → `*.candidate.json` → `--diff`
  → `--promote`/`--discard`); live file never touched mid-refit.
- Key files: `analysis.py`, `props.py`, `mlb_starters.py`, `savant_history.py`, `pitcher_asof.py`,
  `backtest.py` / `backtest_props.py` / `backtest_starters.py`, `calibration_loader.py`,
  `recalibration.py`, `warehouse_mirror.py`, `pricing_common.py`, `wagers.py`, `db_store.py`,
  `r2_sharp.py` (devig).

## Central MLB claims (verify these)
1. **No look-ahead leakage** in the as-of/backtest machinery (four adversarial verifiers agreed;
   every primitive is strict-before-date). Re-audit if you can.
2. **No durable team-market edge** — model Brier ≥ market on all 3 markets; overconfidence is a
   point-estimate/shrinkage issue, not variance. The only claimed edges are situational.
3. **Every "structural" edge died on the clean corpus** — coherence run-line (was a raw-band/dirty-
   capture artifact), the under/f5 family. **Lone survivor: `batter_strikeouts UNDER 1.5`** (+1.26%,
   t=1.69, replicates every season; an inverted public over-bias). `cv_floor` (earned_runs) is
   flagged SUSPECT/unre-validated.
4. **The forward-vs-backtest Brier gap is a pitcher-prop artifact** (pitchers fit on synthetic
   season-avg lines, graded on real book lines, and have no online Platt), NOT regime change or
   grading corruption — so a "2026-only reset" was judged a NO-OP and abandoned.
5. **The SBR odds-source confound** — pre-mid-2025 backtests used SBR prices that were unfairly
   pessimistic (same games, identical model Brier, ML swung −8.6% SBR → +1.2% clean API). A durable
   methodological lesson: judge on clean-API prices.

## Please scrutinize (MLB self-deceptions to challenge)
- **The `batter_strikeouts UNDER 1.5` survivor** — +1.26% at t=1.69 across a large multiple-
  comparisons search. Is it real, or the one cell that survived by chance? (Same skepticism we
  applied to the NFL props artifact.)
- **The "no team edge" verdict** — is the model genuinely at the variance floor, or under-tuned /
  mis-specified in a way that hides a real edge? The OOS ML "+3.3% at EV≥12%" is claimed unstable
  across seasons — agree?
- **Incumbent hysteresis** in the refit sweep (a bare re-sweep resets non-A incumbents to A and can
  silently lose a within-band real-line method). Does the candidate-staging actually contain it?
- **value_gate EV floor (4%)** was chosen on de-vigged *consensus* prices, optimistic vs the DK/FD
  prices we actually bet — is the ROI story robust to real book prices?
- The claim that calibration is a single slot (shrink OR Platt OR blend, never stacked) — is that
  respected everywhere, or is there hidden double-anchoring?

## Standing constraints (apply to any run)
Bets execute at **DraftKings + FanDuel only** (Pinnacle/others analysis-only). **Odds-API pulls
cost paid credits — do not run any ingest/backfill/`--source live` path that hits it.** MLB
backtests read the warehouse/mirror; nflverse + committed mirror data are free. Never commit
`secrets.toml`.

---

# NFL bottom line we reached (challenge it)
The NFL team model is genuinely good and at its noise-limited ceiling (~0.5 RMSE behind the sharp
market, which is irreducible from box-score data). No amount of matchup cleverness closed the gap
(7 tests). Like MLB, it lands at "no durable team-market edge, market is efficient for us." Both
verdicts are offered as *robust* — if you can break either, that's exactly the value of this review.
