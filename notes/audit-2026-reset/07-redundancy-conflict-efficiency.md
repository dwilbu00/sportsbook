# Audit 07 — Redundancy / Conflict / Efficiency / Blockers

Area owner focus: working-against-ourselves, double-counting, conflicting adjustments, duplicate logic across
props.py / pricing_common.py / recalibration.py / refit_calibration.py, dead code, expensive recomputation,
and anything that BLOCKS a clean MLB-2026-only calibration reset.

Scope discipline: READ-ONLY. Every claim below is anchored to file:line. "PROVED" = read directly in code /
committed JSON. "INFERRED" = deduced from code + shipped config but not runtime-verified. I did NOT connect to
Azure SQL, so warehouse/prediction-log/recal-table *row counts by season* are UNVERIFIED (see Open Questions).

Date of audit: 2026-08-07. Shipped calibration read: `calibration/baseball_mlb.json` (fit_timestamp 2026-08-07T10:36Z).

---

## HEADLINE (read this first)

**The offline calibration fit and the live deployment score DIFFERENT projections.** Several projection
multipliers are applied at *runtime* but are NOT reconstructed in the *offline* fit that selects the method,
fits the residual distribution, and advertises the backtest Brier. This is a genuine "working against
ourselves" defect and it is the most parsimonious explanation for the owner's core symptom — *forward Brier
worse than backtest Brier across most markets*. **A 2026-only reset re-runs the SAME pipeline and reproduces
the SAME mismatch, so the reset alone will not close the gap.** Details in Finding 1.

---

## FINDING 1 [HIGH / CONFLICT] — Offline fit omits live multipliers (matchup, lineup, weather, and park for synthetic props)

The live projection is `combined_mult = output_def_mult * matchup_mult * lineup_mult * park_mult *
weather_mult * rest_mult` then `avg_stat = base_proj * combined_mult`; the calibration line is shifted by
`effective_line = line / combined_mult` (props.py:1251-1258). Two offline fitters must mirror this:

1. **Real-line fitter** (`book_line_calibration.project_and_empirical`, ships batter_hits) reconstructs ONLY
   `combined_mult = feat_mult * park_mult` (book_line_calibration.py:854). `feat_mult` is prop_features
   (rest/platoon/gamecontext — all OFF). It does NOT reconstruct `matchup_mult`, `lineup_mult`, or
   `weather_mult` (grep for "weather"/"matchup_mult" in that file: absent). PROVED.
2. **Synthetic sweep** (`backtest._build_props_sweep_grid` → `run_player_props_backtest`, ships all pitcher
   props + batter_strikeouts) sweeps half_life/opp_defense/def_adj/shrink/venue/rest but its `_preset` default
   is `park_strength=0.0` and park is NOT a grid axis (backtest.py:1980-2000, 1862). It also never applies
   weather (explicit comment "no weather knob … LIVE-ONLY", backtest.py:1878-1880) nor matchup_mult /
   lineup_mult (those are separate starter_adjustment / lineup_adjustment fits, not in the props sweep). PROVED.

Now cross against what is ON live in the shipped config (baseball_mlb.json + props.py module defaults):

| prop | ships on | live multipliers ON | reconstructed offline? |
|------|----------|---------------------|------------------------|
| batter_hits | real_line (method E) | matchup 0.5, lineup ON, park 1.0, weather 0.5 | park YES; matchup/lineup/weather **NO** |
| pitcher_earned_runs | synthetic (method A) | park 1.0, weather 0.5 | **park NO** (grid park=0), weather **NO** |
| batter_strikeouts | synthetic (method A) | matchup 0.5, opp_defense 1.0 (weight) | opp_defense YES; matchup **NO** |
| pitcher_strikeouts | synthetic (method B) | venue 0.25 | venue YES → consistent |
| pitcher_outs | synthetic (method A) | none | consistent |

- matchup 0.5 is live because `starter_adjustment.props` selected `{batter_hits:0.5, batter_strikeouts:0.5, …}`
  (baseball_mlb.json meta) and `_starter_adjustment(sport,"props",prop)` returns it (pricing_common.py:201-222);
  applied via `_mlb_prop_matchup_mult` (props.py:1200).
- park 1.0 / weather 0.5 are live via module defaults `PLAYER_PROP_PARK_STRENGTH={"baseball_mlb":1.0}` /
  `PLAYER_PROP_WEATHER_STRENGTH={"baseball_mlb":0.5}` (props.py:299,317) — NOT calibration-JSON knobs, so a
  refit never re-decides them.

**Consequence:** the calibration's method choice, residual μ/σ/ECDF, mean_scale/dispersion, AND the advertised
`fit_brier` were all computed against a projection that live then shifts by up to ~10-20% (matchup ±5-15%,
weather bounds 0.88-1.15, park bounds 0.85-1.20). The backtest Brier is therefore not comparable to forward
Brier — a structural gap, not necessarily model rot.

**Weather is structurally unfittable offline** (no historical per-game weather stored — backtest.py:1878).
It will ALWAYS be a live-only mismatch on batter_hits/pitcher_earned_runs unless disabled live or historical
weather is captured.

**Bearing on reset: UNDERMINES.** The reset re-runs this exact pipeline; the mismatch is regenerated. Real
fix = make offline==online (reconstruct matchup/lineup/weather/park in both fitters, OR turn the live-only
signals off), independent of which season the data comes from.

## FINDING 2 [HIGH / BLOCKER] — "2026 reset" via refit only touches props{}; matchup + team-market + lineup fits are PRESERVED stale (pre-ABS)

`refit_sport` ends with `save_calibration(sport_key, props_cfg, meta)` (refit_calibration.py:762).
`save_calibration` loads the existing blob and preserves every non-props top-level block
(calibration_loader.py:105-132: "a props refit preserves every other top-level block"). PROVED.

So a bare `refit_calibration.py --sport mlb` re-fits `props{}` but leaves untouched, on pre-2026 data:
- `starter_adjustment` (matchup weights) — `source: backtest_props:2024,2025` (baseball_mlb.json meta). This
  is the matchup_mult applied LIVE to batter_hits/batter_strikeouts (0.5). A "2026-only" batter model is thus
  still multiplied by a 2024-2025-fit signal.
- `expected_runs_challenger` — `fit_window 2024-03-20…2024-06-30` (team markets).
- `prob_shrink` (spreads/totals/moneyline), `lineup_adjustment` — older fits.

**Bearing on reset: UNDERMINES / QUALIFIES.** The reset is not actually "2026-only" in deployment; pre-ABS
signals re-enter through preserved blocks and through matchup_mult. A true 2026 reset also needs
backtest_starters / backtest_props / prob-shrink / expected-runs re-runs on 2026 data — which may not have
enough volume yet (Open Question).

## FINDING 3 [HIGH / BLOCKER — operational landmine] — Incumbent hysteresis: a bare refit DEMOTES batter_hits from real-line E to synthetic method and DROPS real_line_fit / line_methods

`refit_sport` builds each prop cfg fresh from the synthetic-sweep winner (`_build_prop_cfg`,
refit_calibration.py:733) and `save_calibration(merge_props=True)` does `existing_props.update(props_cfg)`
(calibration_loader.py:119-121) — which REPLACES the whole per-prop value. The synthetic winner carries no
`real_line_fit` and no `line_methods`, so those keys (and the real-line method E for batter_hits, and the
per-line-bucket Platt selection) are lost; batter_hits reverts to its synthetic method. PROVED (matches the
"INCUMBENT HYSTERESIS" gotcha already in MEMORY.md). batter_hits is the ONLY prop with a real 2026 fit
(fit_basis=real_line; all others synthetic_sweep), so this is the prop most at risk.

Correct reset sequence must be: bare `refit_sport` → `refit_sport_real_lines` (reads durable store; preserves
synthetic where thin) → and, because batter_hits E beats A by only ~0.0019 < the 0.002 flip-gate, splice E
back as incumbent FIRST so MIN_REAL_LINE_OVERRIDE_OBS / _incumbent_protected keeps it. Error-prone; easy to
silently ship a worse batter_hits.

**Bearing on reset: NEUTRAL-to-UNDERMINES** (it is a hazard in *executing* the reset, not a reason for/against
resetting the data).

## FINDING 4 [MEDIUM] — Two stacked calibration layers, but the Platt safety-net covers only ONE bucket in the committed seed

Live applies the residual method (B/C/D/E) OR empirical (A) to get `raw_over_rate` (props.py:1276-1334), then
`_apply_final_recalibration` (Platt) on top (props.py:1344-1359). This is a PROPER cascade, not a
double-application: the prediction log stores `raw_prob=raw_over_rate` i.e. the PRE-Platt value
(props.py:1655), so Platt is fit on exactly the input it corrects (recalibration.py fits on raw_prob). CLEARED
as a double-count.

BUT the committed recal seed `calibration/recalibration_baseball_mlb.json` contains a Platt fit for exactly
ONE key: `batter_hits@le_0.5` (a=0.375, b=0.194, n_fit=2144, holdout 2026-07-26/28, source
book_line_cache_seed). The a=0.375 slope is a HEAVY correction — consistent with raw E probs being
systematically off live (which Finding 1 predicts). Every OTHER market (pitcher props, batter_strikeouts,
batter_hits alt-lines ≥1.5) has NO Platt net in the seed, so they eat the full Finding-1 mismatch plus any ABS
regime shift with no correction. This is a coherent mechanism for "forward worse than backtest across MOST
markets, but batter_hits main line least affected."

**Bearing on reset: SUPPORTS that forward degradation is real for the uncovered markets — but the cheaper fix
may be broadening Platt coverage, not a full data reset.** Open Question: the LIVE recal is loaded from SQL,
not this file; how many buckets the SQL table has actually accrued is UNVERIFIED.

## FINDING 5 [MEDIUM] — Weather and park partially model the SAME offensive-environment physics (mild double-count)

For batter_hits / pitcher_earned_runs both `park_mult` and `weather_mult` fold into combined_mult
(props.py:1224-1234). The weather nudge is baseline-relative to 70 F / no-wind (props.py:321,
WEATHER_BASELINE_TEMP_F), NOT relative to the park's own average conditions. Park factors already embed each
venue's typical air-density/carry environment (e.g. Coors altitude). Applying a 70 F-relative temp/wind
multiplier on top re-introduces a slice of the same carry physics the park factor already priced, producing a
small systematic bias at hot/cold/high-altitude parks. The code comment (props.py:305-316) argues weather
doesn't double-count the *player's 15-game sample* — true — but it does not address overlap with the *park
factor*. Magnitude is small (strength 0.5, bounds ±12/15%) and unvalidated (no historical weather).

**Bearing on reset: NEUTRAL** (small; and weather is unvalidatable offline regardless of season).

## FINDING 6 [LOW] — Duplicate multiplier implementations across offline/online (drift risk)

`backtest.venue_mult` / `backtest.opp_defense_mult` (backtest.py:92-121) are byte-for-byte equivalent to
`pricing_common._venue_match_multiplier` (numeric-strength branch) / `_opponent_defense_multiplier`
(pricing_common.py:279-336). Verified identical TODAY. But they are two hand-maintained copies of the same
formula (the docstring even says "Tunable version of analysis._venue_match_multiplier") — a future edit to one
silently diverges offline fit from live. Same pattern for `_resolve_opp_pts_allowed` (backtest) vs
`_resolve_team_defense` (pricing_common). **Bearing: NEUTRAL** (maintenance hazard, not a live bug).

## FINDING 7 [LOW / INFO] — Per-candidate uncached SQL reads in the live hot loop

`statcast_asof.get_rates` opens a fresh `engine.connect()` per call (statcast_asof.py:84-104) with no
in-process memo; it is called per batter_hits candidate via the xstats path (props.py:1146-1150), and
`mlb_starters.find_player_id` is likewise called per candidate. For a full slate this is O(batters × lines)
SQL round-trips + player-id lookups per refresh. Fail-open, correctness-neutral; a per-process cache keyed by
(player_id, season) would remove redundant queries. **Bearing: NEUTRAL** (efficiency only).

## FINDING 8 [INFO] — Config surface: many inert knobs; and fit_season is a LABEL, not proof of data vintage

Shipped MLB props carry several knobs that are OFF/inert (output_def_strength=0 all props, market_prior_k
empty, rest/platoon/gamecontext off, NegBin E ships only batter_hits). These are intentional
"free-recheckable" scaffolding per MEMORY.md, not dead code, but they enlarge the reasoning surface for a
reset. Separately: every prop shows `fit_season: 2026`, but that field is just `refit_sport`'s `--season`
default (refit_calibration.py:735 sets it from the arg, which defaults to `datetime.now().year`). It is NOT
evidence the underlying gamelog/warehouse rows are 2026-only. batter_hits real_line_fit + the recal seed ARE
demonstrably 2026 (holdout dates 2026-07). Pitcher synthetic n_obs=333 being 2026-only is INFERRED
(cross_season="strict" + season=2026 at refit_calibration.py:698-701) but not proven from the artifact.

---

## Adversarial checks I ran that CLEARED (no defect)

- **opp_defense double-count (weight-side AND output-side)?** props.py applies `_opponent_defense_multiplier`
  to prior-game weights (1092-1095, knob opp_defense_strength) AND `_output_defense_multiplier` to the level
  (1112-1118, knob output_def_strength). In the shipped MLB config NO prop has both > 0 (batter_strikeouts has
  opp_defense 1.0 weight-side, output 0.0; everything else 0/0). Not double-counted today. There is no
  separate opp_defense entry in prop_features, so no weight-vs-feature overlap either. CLEARED (but the two
  knobs being independently settable is a latent trap if a future refit turns both on for one prop).
- **park applied twice (weight AND multiplier)?** Park appears only in the output combined_mult
  (props.py:1213-1226); the weight loop (1090-1105) has only opp_defense + venue. The road-context DELTA
  design (divide by the recency-weighted average of the player's recent parks) explicitly avoids
  double-counting his home park. CLEARED.
- **Platt applied to an already-Platt'd probability (feedback loop)?** No — trained and applied on pre-Platt
  raw_over_rate. CLEARED (see Finding 4).
- **venue/opp_defense formula drift offline vs online?** Identical today. CLEARED (flagged as Finding 6 drift
  risk).

---

## Open Questions (could NOT determine from code — worth a read-only SQL COUNT before deciding)

1. Warehouse / prediction-log / statcast_asof row counts BY SEASON — is there enough 2026 volume for a clean
   2026-only refit of pitcher props and (critically) team markets? Not verifiable without a DB connection.
2. Live recalibration SQL table: how many (prop@bucket) Platt fits has the online loop actually accrued beyond
   the committed `batter_hits@le_0.5` seed? Determines how exposed the "uncovered markets" really are
   (Finding 4).
3. Does `book_line_calibration.harvest_real_line_book_lines` filter the durable store by season? If the store
   is already 2026-only by construction, Finding 2's "2026-only" concern narrows to the preserved
   team-market/matchup blocks only.
4. Are the ABS-sensitive props (K/BB-driven: batter_strikeouts, pitcher_strikeouts) the ones whose forward
   Brier degraded most? That would corroborate the ABS-regime motivation over the Finding-1 methodology
   artifact. Needs the forward-tracker numbers, not code.

---

## Net read for the reset decision

- The reset's PREMISE (forward < backtest) is at least PARTLY a self-inflicted methodology artifact
  (Findings 1 + 4), which a same-pipeline 2026 refit will NOT fix. Fix offline==online and broaden Platt
  coverage first; those are cheaper than a destructive wipe and are prerequisites for the backtest number to
  mean anything post-reset.
- The reset is also NOT cleanly achievable with the current tooling: `refit_sport` preserves stale pre-ABS
  matchup/team blocks (Finding 2) and, run bare, demotes the one prop that has a real 2026 fit (Finding 3).
- ABS (2026 strike-zone change) is a legitimate reason to distrust pre-2026 K/BB calibration, and the warmup
  block is already null so the *structure* supports 2026-only. But confirm 2026 data volume (Open Q1) and
  remember that matchup_mult + team markets are outside the props refit and stay pre-ABS unless separately
  re-fit.
