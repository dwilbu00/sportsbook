---
name: preferences
description: How to work with Doug (owner): who he is, commit/push rules, betting books (DK+FD), spend confirmation, backtest handoff, Alpha status, and his defaults-audit methodology.
metadata:
  type: feedback
---

## Who: Doug + Cal
Owner is **Doug** — sole user AND developer of this MLB (+NBA/NFL) sportsbook betting model. Technical, hands-on, rigor-driven: runs backtests/DDL/backfills himself, makes the modeling calls, wants adversarial verification before spends, OOS validation, no silent regressions. Treat this as an ongoing partnership, not one-off tasks. He likes personalizing the relationship and invited me to pick a name (2026-08-18). I go by **Cal** (nod to *calibration*, the spine of the project) — use it if he addresses me by it.

## Commit / push discipline
- **Commit proactively.** After a self-contained, verified change (py_compile / tests green), commit directly to `main` without asking. Doug reviews commits after the fact. (This superseded the old "never commit unless asked" rule.)
- **NEVER push** — Doug runs `git push` himself; commits accumulate locally unpushed until he does.
- Commit message ends with the `Co-Authored-By` trailer.
- **Never commit `.streamlit/secrets.toml`** (gitignored live Azure + Odds API creds).
- **Push = deploy** (Streamlit Cloud) → it's the durable sync point. When Doug says he pushed, update the relevant topic memory to the now-pushed/deployed state (record shipped SHAs + what's live vs still pending).

## Betting books — DK AND FanDuel (both executable)
- Doug bets at **DraftKings AND FanDuel** (FD added 2026-08-27). A bet best-priced at FD is actionable; **line-shopping DK vs FD is a legit profit lever** (take the edge at whichever prices it better).
- All other books (Pinnacle/BetMGM/etc.) are **analysis-only** — never recommend, never display their price/payout, never drive staking. Pinnacle = the R2 sharp reference, never bet.
- De-vigged multi-book/Pinnacle consensus IS the preferred edge *baseline* — that's analysis, not execution.
- ⚠ **LIVE APP CODE is still DK-only** (DK-anchoring, DK-only EV/display). Surfacing FD prices / a DK-vs-FD line-shop is a future code change, not built. DK-line anchoring shipped commit 7ea856b: prop analysis anchors on the line DK actually posts (not cross-book modal); DK-alone guard raises the edge bar 2× (`_DK_SELFDEVIG_EDGE_MULT`) when <3 peers and no sharp at DK's line.
- ⚠ **FD data caveat (warehouse, 2026-08-27):** FanDuel fully covers TEAM markets (ML/spread/total, 6k+ events) + pitcher_strikeouts/pitcher_outs, but has NO batter props or earned_runs captured — so FD can't line-shop the batter over-bias or ER under (our strongest signals) without a credit re-pull.

## Spend confirmation
Before EXECUTING any large/irreversible spend (Odds API credits, paid backfills, real money/quota), give one final explicit "firing now — ~N credits, go?" beat — **even after a general go-ahead during scoping**. A planning "let's do it" is NOT consent to spend at the moment of execution (2026-08-15: approved ~10.5k-cr backfill during scoping, fired it, Doug immediately asked to pause). Dry-runs / pricing / diagnostics / reads / code+tests are free — no confirm needed; only the real spend does.

## Verifying prod state (do this instead of asking)
I can read prod Azure SQL **directly** (read-only, no spend) by calling
`db_store.promote_secrets_from_toml()` first, then querying — so verify prod state
(tables, DDL applied, row counts, live calibration) MYSELF rather than asking Doug.
Never commit `secrets.toml`. Coords/mechanism in [[data-and-architecture]].

## Backtest execution — handoff to Doug's machine
Doug runs all backtests on his own (much faster) machine. **Do NOT run long/grading backtests here.** Write the exact copy-pasteable command block + a short "what I'm looking for in the output" note, hand off, wait for pasted output to interpret. Keep doing the CODE side here (build/commit INERT + opt-in features behind their own sweep flags). Cheap dry-runs / store-builds / non-grading utilities are fine to run here.

## Alpha status + the one hard constraint
App is officially **Alpha** (2026-08-10): no users but Doug, in active development → ship live and iterate fast. Don't gate routine forward progress on "is this safe for users?" — there are none. Prefer cheap dual-run/parity windows over big-bang cutovers, but don't wait for calendar boundaries to go live.
- **The relaxation is ONLY about user-facing breakage.** The data-integrity / never-destroy-irreplaceable-data constraint is UNCHANGED and absolute: wagers, bankroll ledger, prediction/calibration corpus, and learned JSON fits → translate/re-key in place, never drop.

## Doug's defaults-audit methodology
Doug's instinct (2026-08-19): "how many default values are we assuming are OK when they could be hurting us — games-per-player, min_sample, half-life, lookback?" A 4-agent audit found ~40 magic numbers, ~half never validated; his **lookback suspicion was the bullseye**. Durable working preference — his **corrected dependency-ordered sequence** for any default/calibration work:
1. **Projection defaults first** (the INPUTS): recent_n × half_life JOINT (biggest knob), market_prior_k, min_sample/warmup/shrinkage_k, per-team variance floor.
2. **Recalibrate on the corrected basis** (base methods, prob_shrink, pythag, blend).
3. **Re-validate market verdicts + team value gates** on the corrected+recalibrated basis (totals-off? spreads? edge thresholds).
4. **Sizing** (kelly_fraction).
Guiding pattern he confirmed: calibration GATES are grounded in evidence; the PROJECTION + SELECTION knobs are mostly on vibes. Category-A rule: where LIVE serving contradicts what we already proved, FIX it (not a study).

### Defaults work — what's DONE (durable outcomes)
- **STEP 1 COMPLETE — recency windows locked + shipped** (commits 7e9b35c mechanism + 70f69dc values): decay OFF held 9/9; batter_hits / batter_strikeouts / pitcher_strikeouts = **full season, decay off**; pitcher_ER **recent_n=20**; pitcher_outs no stable window (suppressed); TB/RBI not swept → default 20 (revisit step 3). recent_n is now a per-prop calib knob. This CLOSED a fit≠serve gap (methods were fit at full, served at 20).
- **Weather validated → OFF for props** (commit 50739f2, 3 seasons): density weather HURTS batter_hits 3/3; ER not robust (method-coupled). `PLAYER_PROP_WEATHER_STRENGTH` mlb **0.5→0.0**. air_density / density_factor / `--weather-sweep` machinery BANKED for the STEP-3 team-runs run_env (run-environment effect belongs there, where the additive team model already consumes `game_weather_map`).
- **Totals suppressed + team-market suppress mechanism** (commit 6958848): `pricing_common._market_suppressed` now gates TEAM markets (not just props); 'totals' added to suppress (−4.9%/−5.1% both seasons + overconfident). Stale comments fixed same commit.
- **Enablers committed (unpushed as of register):** method E (NegBin) selectable in the SYNTHETIC sweep = the TB fix (1b1aa7e; `stats.fit_negbin_params` shared fitter); multi-season pooling `--seasons 2024,2025,2026` (812773b, per-season sweep, no cross-season leakage, pooled calib_obs, union player pool).

### Defaults work — deferred Category-B sweeps (not yet run)
Ranked never-swept knobs still on vibes: `market_prior_k=0` (market-anchor OFF; sweep {0,3,5,10,20,40}/prop); team value threshold flat 5% (likely too stringent vs props' 4%EV/1%edge — sweep per-market); per-team **variance floor=1.0** (prime suspect for forward>backtest overconfidence; ignores within-game correlation); `kelly_fraction=0.5` (sizing_sweep is diagnostic-only, never pinned; uncertainty-Kelly built, not wired live); RECENCY_HALF_LIFE team=7/props=0; DEFAULT_SPREAD_DISPERSION=0. Spreads still gated flat 5% live (study said high-conviction ≥10%) — deferred into the team-gate sweep.

### Calibration upgrades to adopt when we recalibrate (from BP-clone survey)
Genuinely higher-value ideas only: (1) dedicated fit→calibrate→final-test 3-way chronological partition w/ 1-day embargo (deployed Platt is refit on 100% of rows → mildly optimistic); (2) per-season expanding walk-forward folds (replicate EACH season, not one pooled cut); (3) Beta-Binomial empirical-Bayes shrinkage for rate props (cheap TB/RBI win, emits posterior_sd for uncertainty-Kelly); (4) a calibrator BAKE-OFF (Platt/Beta/Isotonic/Temperature) on the final recal layer — build only if the refit diff shows a need. (Note: our NegBin method E already beats BP's plain-Poisson count model.)
