# Sportsbook Application Improvement Study

The application has useful statistical components, a substantial historical corpus, and foundations for reproducible research. Its best improvement path is to make the complete prediction and decision process measurable, then develop models of player opportunity and conditional outcome distributions. More elaborate team-strength features alone are a lower priority. Better execution prices, correct settlement probabilities, and faithful deployment can improve decisions even when the underlying sports forecast changes little.

Predictive accuracy and reliable betting decisions are the primary objectives of this study. Application performance, reliability, architecture, and usability are assessed according to how much they support those objectives. The code baseline is `bd18fd0`, examined on September 8, 2026. This is a separate study from the [independent correctness review](INDEPENDENT_REVIEW_2026-09-08.md); its production fixes remain outstanding.

Three new empirical results guide the recommendations. First, MLB batter-hit probabilities have an avoidable discontinuity caused by rounding expected at-bats; two opportunity-mixture alternatives improve an isolated component in both seasons at the 0.5 line. Second, inexpensive alternative NFL probability models produce no convincing replacement for the existing baseline and remain behind the cleaned closing market. Third, matching historical offers across DraftKings and FanDuel reveals frequent price improvements that the predominantly DraftKings-oriented serving path does not fully exploit.

No profitable new betting strategy is established here. The evidence does support a concrete development program with several opportunities beyond those covered by the original review.

## 1. Coverage and priorities

The source inventory contains 181 root Python modules, including 87 test modules. The remaining modules contain 67,827 lines, including application code, ingestion, backtests, and research utilities. The local inventory contains 116 Parquet files, including the `_valid` files that hold much of the MLB corpus. These counts describe scope, not code quality or test coverage.

| Area | Current foundation | Most valuable improvement |
|---|---|---|
| MLB props | Real-line calibration, methods A–E, xBA inputs, per-line methods, online recalibration | Model playing opportunity without rounding; distinguish rate uncertainty from workload uncertainty; validate the entire served distribution |
| MLB team markets | Additive runs, starter and bullpen inputs, market blending | Correct baseline first; test replacement components and coherent score distributions against a market-only benchmark |
| NFL team markets | EPA, projected QB, injuries, rest, dedicated margin model | Restore fit/serve agreement and data availability; test probability distributions and incremental information beyond the contemporaneous market |
| NFL props | Player-week, snaps, PBP, roster, NGS and historical odds; incomplete production calibration | Build an opportunity-first model for a small set of props; implement participation-aware outcome construction |
| NBA | Generic team/prop pipeline and historical odds; ESPN-oriented statistics paths | Establish durable statistics and role/minutes forecasts before importing MLB calibration conclusions |
| Pricing and selection | EV, fractional Kelly, caps, rule-based selection, copula parlays | Preserve each executable offer; account for pushes, actual ticket prices, open exposure and estimation uncertainty |
| Application and operations | Streamlit, SQL, Parquet mirrors, caches, ledger, candidate staging | Isolate prediction from maintenance, version all inputs, surface degraded predictions, and test complete user paths |

NFL is the active sport, but an existing, well-scoped MLB defect can be cheaper to address than a new NFL feature. NBA deserves a separate predictive benchmark before substantial model development. Generic NHL references in shared modules do not establish a validated NHL product; it is outside the substantive sport-specific modeling conclusions here.

Recommendations use three evidence levels: **measured** means an experiment or deterministic reproduction supports the statement; **observed** means the behavior follows from inspected code or local data; **proposed** means a plausible improvement still needs a comparative experiment. A measured component improvement is not automatically an improvement to production or betting returns.

## 2. Improvements established by new evidence

### 2.1 MLB opportunity modeling: remove the rounded-at-bat discontinuity

The distributional hit model computes `n = round(expected_ab * exposure_mult)` and then evaluates a binomial survival probability. In [props.py:452](../props.py), changing expected at-bats from 3.49 to 3.51, with an unchanged 0.25 hit probability per at-bat, changes the probability of at least one hit from **57.81% to 68.36%**. A 0.02 change in expected opportunity produces a **10.55 percentage-point** probability jump. The same mechanism appears in the historical distributional helper because it calls the shared primitive.

This is a model specification problem. Expected at-bats is a mean of a random count, not a count that should become certain after rounding. A small change in batting-order exposure can cross a rounding boundary, while a larger change elsewhere can do nothing. This can distort feature tests as well as individual predictions: a useful exposure adjustment may appear ineffective until it suddenly moves a whole at-bat.

Two replacements were screened. An **adjacent-count mixture** puts probability on the integers immediately below and above expected at-bats, preserving its mean. An **empirical opportunity mixture** averages the binomial probability over the player's distribution of prior at-bat counts. The latter is a starting point, not a complete model of a confirmed starter's future workload.

| Season / hit line | Observations | Rounded count Brier | Adjacent-count mixture | Empirical AB mixture |
|---|---:|---:|---:|---:|
| 2024 / 0.5 | 27,349 | 0.242029 | **0.240806** | 0.240880 |
| 2025 / 0.5 | 27,978 | 0.240686 | 0.239469 | **0.239405** |
| 2024 / 1.5 | 4,731 | 0.219787 | 0.219732 | **0.219394** |
| 2025 / 1.5 | 4,587 | 0.222896 | 0.223070 | **0.222733** |

These are exploratory component results. They use exact player/game IDs, recorded regular-season outcomes, DraftKings closing thresholds captured within 60 minutes before the listed start, and at least 20 prior AB-positive games in the same season. Same-day games do not enter one another's feature history. The rate is historical H/AB; the experiment does **not** reproduce the shipped xBA blend, contextual adjustments, eligibility rules or final recalibration. There is no ROI result, confidence interval or untouched confirmation set for this MLB screen, and observations at different lines can share an outcome.

**Recommendation:** prioritize a full-pipeline opportunity-mixture challenger. Keep xBA and all other inputs fixed initially, recalibrate each challenger on training data, and compare on identical eligible offers. Use confirmed batting order, starting status and opposing pitching context to model opportunity where timestamps permit. Require smooth probability changes when expected exposure changes continuously. The consistent 0.5-line improvement justifies this experiment; the mixed 1.5-line result argues against choosing the adjacent-count version universally.

### 2.2 Offer preservation and DraftKings/FanDuel execution

[odds_client.py:1107](../odds_client.py) returns one selected line per player/prop, normally anchored on DraftKings. A synthetic event with a DraftKings 1.5 line and FanDuel 0.5 and 1.5 lines retains only 1.5 in the parsed output. [props.py:1930](../props.py) then selects direction and measures executable EV at the DraftKings price. This limits both line shopping and market coverage.

Historical team offers provide a separate, real-data measurement of the opportunity. The comparison below matches event, side, market, line and snapshot label; both captures precede the listed start, their timestamps differ by at most 60 seconds, and duplicate quote grains are excluded. It uses the stored calendar files through 2026.

| Sport / snapshot | Matched side-line offers | FanDuel better | Share with better FD price | Mean decimal-odds gain from choosing the better book |
|---|---:|---:|---:|---:|
| MLB / early 4h | 23,438 | 9,634 | 41.1% | 0.01259 |
| NFL / early 4h | 3,738 | 1,544 | 41.3% | 0.02622 |
| NBA / early 4h | 17,770 | 7,099 | 39.9% | 0.02446 |
| MLB / closing | 22,952 | 9,489 | 41.3% | 0.01341 |
| NFL / closing | 3,672 | 1,564 | 42.6% | 0.02407 |
| NBA / closing | 17,107 | 7,044 | 41.2% | 0.02511 |

These are offers, not independent games or recommended bets. The study does not verify historical account availability, accepted stakes or the ability to execute both quotes. Snapshot labels do not guarantee precise target lead times. Decimal-price improvement cannot be read directly as ROI improvement; it must be multiplied by the relevant win probability. Nevertheless, choosing a higher payout for the same settlement event increases expected return for any positive win probability.

For a purely illustrative 54% forecast, moving from −110 to +100 changes estimated ROI from 3.09% to 8.00%. No model improvement is required. That example is larger than the average stored price difference and should not be used as a revenue forecast.

**Recommendation:** introduce an offer-level representation keyed by sport, event, player, market, side, line, book, quote time and settlement rules. Compute probability once per settlement event and evaluate every permitted executable offer against it. Preserve all raw lines through ingestion. Anchor research probabilities to an appropriate reference market, while selecting execution only from DraftKings and FanDuel under the current handoff. Older memory mentions bet365; resolve that configuration discrepancy before any expansion, rather than inheriting book lists from comments.

### 2.3 Single-bet selection: separate constraints from statistical claims

The selector reuses parlay conflict logic and rejects pairs with heuristic correlation at or below −0.20. For example, an NBA game-total UNDER and a player-points OVER are rejected by [bet_selector.py:241](../bet_selector.py). For separately settled positive-EV bets, negative covariance can reduce portfolio variance. It does not negate the expected return of either bet.

A feasible illustrative Bernoulli pair with win probabilities 0.55 and correlation −0.40 has a 40% lower variance of the sum than the independent pair. This is mathematical evidence against treating negative correlation itself as a defect in a portfolio of singles. It does not establish that the application's numerical correlation estimates are accurate. Explicit owner preferences and genuinely infeasible combinations remain valid constraints.

The selector's opening claim that greedy selection is correct is also too strong. A reproduction using the **unmodified NBA rules** assigns a total UNDER estimated EV of 10%, and two opposing-team player-points OVER bets 9% each. The first bet conflicts with both others; the latter two are mutually allowed. With two available slots and equal $1 stakes, greedy selection returns only the 10-cent expectation, while the allowed pair has an 18-cent expectation. This establishes suboptimality, not the size of its effect on real slates.

**Recommendation:** retain the simple selector as an explicitly labeled heuristic. First report exclusions and compare it with exhaustive selection on small recorded candidate pools. Then consider a constrained optimizer that uses expected dollar profit, stake limits and covariance. Do not maximize summed ROI percentages when stakes differ, and do not silently remove owner-requested rules.

### 2.4 Parlays: identify ticket value from the ticket price

[parlay.py:536](../parlay.py) estimates same-game parlay EV by multiplying the individual decimal prices and substituting the independent probability for the modeled joint probability. This assumes the bookmaker's correlation adjustment cancels the model's correlation benefit exactly. There is no general justification for that equality, especially when the model and bookmaker disagree about dependence or margins.

For a hypothetical two-leg ticket with modeled joint probability 0.35, decimal odds of 2.50 imply −12.5% EV, while odds of 3.00 imply +5.0%. With two 55% individual forecasts at −110, the current surrogate returns +10.25% irrespective of which combined ticket price is available. Both hypothetical payouts are consistent with the same individual leg prices. The missing ticket quote therefore prevents identifying executable SGP value.

The copula has another improvement opportunity. Independent cross-game legs still go through 5,000 Monte Carlo draws. For three independent 55% legs, the exact joint probability is 16.6375%; 30 seeds produced estimates from 15.78% to 17.74%. The theoretical standard error at 5,000 samples is about 0.527 percentage points. That simulation error is material to a value gate, and reranking many combinations can favor numerical noise.

**Recommendation:** use exact multiplication whenever the chosen dependence model is independence. For same-game tickets, distinguish an estimated joint probability from executable value and require the actual ticket quote before displaying the latter. Fit and validate dependence separately: heuristic outcome/rank correlations cannot simply be assumed to equal latent Gaussian correlations. For dependent simulation, consider scrambled Sobol draws with independent replications and convergence checks, not a single apparently precise number. Numerical improvement does not validate an incorrect dependence model.[^1]

### 2.5 Exposure limits must include outstanding bets

[app.py:490](../app.py) sizes the current selected batch from the ledger balance, applies a correlation haircut within that batch, and caps that batch's stake. It does not include existing pending bets in the calculation. The ledger accrues settled profit, so repeated submissions can each satisfy a 25% batch cap while total unresolved exposure exceeds 25%.

**Recommendation:** define bankroll equity, spendable cash and unresolved exposure separately. Size new tickets against the entire open portfolio, including previous submissions, sports and books. Preserve existing ledger reconciliation and audit history. Add a deterministic test in which a second batch arrives while the first remains pending. Kelly fractions and drawdown controls belong after trustworthy probabilities and complete exposure accounting; theoretical risk-constrained Kelly methods still depend on the assumed return distribution.[^2]

## 3. A stronger evaluation methodology

### 3.1 Evaluate the decision that could actually have been made

The earlier review found outcome-dependent QB identity, post-kickoff closing captures, dropped rest inputs, night-game date mismatches, an incorrect totals sign, a wrong-game MLB fallback and incomplete candidate diffs. Those are prerequisites for a reliable baseline. Fitting a more complex model on the same flawed observation construction would create a more expensive ambiguity.

Every evaluation should specify a decision time: for example, four hours before kickoff, one hour before first pitch, or after a confirmed lineup. Each input needs its event time and, where relevant, its publication/availability time. A statistic from an earlier game may be available; an injury designation published later that day may not be. Weather measurements from the completed game are not early forecasts.

Use a canonical game spine and retain separate fields for sport season, local game date, UTC start and quote capture. Grade by positive game/player identity; contradictory known IDs must fail closed. Require quote sides to belong to the same market and line. Preserve the probability forecast, offered price, actual execution price and settlement rule as distinct objects.

Historical provider archives can retain data errors, and historical odds are available only from when the provider added the relevant markets. The provider documentation itself supports treating archive provenance as evidence to inspect, not automatic certification.[^3]

### 3.2 Make the entire pipeline chronological

Adopt expanding calendar folds with a training period, a later calibration/selection period, and a still-later evaluation period. Keep all rows for a game together, including alternate lines, both sides, players and books. Build preprocessing, priors, transformations, model weights, probability calibration and selection thresholds from permitted earlier data. Scikit-learn's time-series documentation explains the future-to-past problem, but its generic equal-row split is not a substitute for game- and calendar-aware splitting here.[^4]

A one-day embargo is useful only when it matches the availability or overlap problem. It is not a universal fix. For NFL, group by game and week; for MLB, keep doubleheaders and daily data corrections explicitly scoped. Evaluate week 1, playoffs, missing-feature games and rookies separately rather than silently removing them from a model advertised for all games.

The 2023–2025 seasons have already influenced model choice and prior conclusions. Repartitioning them is helpful for diagnosis but does not make them untouched. Freeze a prospective protocol before the next genuinely unseen observations, and keep any currently unused period unused until its role is declared.

### 3.3 Use metrics that answer different questions

| Question | Primary evidence | What it does not establish |
|---|---|---|
| Is the point forecast better? | Paired MAE/RMSE on identical games | Accurate tails, prices or profitability |
| Are probabilities better? | Brier and log loss; reliability by relevant subgroup | A positive edge at executable odds |
| Is the full distribution useful? | CRPS or appropriate discrete log score, interval coverage, tail diagnostics | A profitable selection rule |
| Does a signal add beyond the market? | Comparison with market-only and market-plus-signal models at the same decision time | Ability to obtain or retain the quoted price |
| Is the strategy actionable? | Real offered/accepted prices, settlement, turnover, drawdown and forward results | Permanent future profitability |

Proper scoring rules reward honest probabilistic forecasts in expectation. They should be the main predictive evaluation tools; raw hit rate can improve simply by choosing more favorites. A Brier score of 0.25 is the score of a constant 50% binary forecast, not a universal noise floor.[^5]

Assess uncertainty from paired differences. Cluster by game and use week/day blocks where dependence warrants it; retain seasonal results and missingness coverage. A hundred alternate-line rows from one contest do not supply a hundred independent outcomes. Compare calibration within decision-relevant populations, but freeze subgroup definitions before treating them as evidence for action.

### 3.4 Control the research process, not just the final p-value

The repository already has an experiment registry concept and `overfit_stats.py`. Extend those instead of building another disconnected research tracker. Record hypothesis, eligible population, feature availability, code/data hashes, complete candidate set, selection rule, metric, stopping rule and negative results.

`overfit_stats.deflated_roi` maps a t-statistic minus an approximate expected maximum into `deflated_prob`, described as approximately the probability the true edge is positive. That is a heuristic search penalty, not an established Bayesian posterior probability. Correlated candidates, repeated holdout inspection and game-level dependence make a literal confidence interpretation especially questionable. Rename the output accordingly or validate a formal procedure with simulation.

Backtest-overfitting research motivates recording all tried variants and testing the selection procedure. CSCV/PBO can be a useful supplementary diagnostic; balanced block combinations that train on future blocks do not replace a historical forward evaluation.[^6] Prefer a blocked permutation or bootstrap of the complete selection process when making a formal search-adjusted claim, with assumptions documented.

For scale, an illustrative independent-return calculation with unit standard deviation, a two-sided 5% test and 80% power needs approximately `(1.96 + 0.84)^2 / 0.02^2 = 19,600` bets to detect a 2% mean return. Detecting 1% needs about 78,400. Actual requirements depend on price distribution, correlation and selection. Small, promising cells should remain hypotheses; lack of significance is not evidence of an irreducible ceiling.

## 4. NFL predictive development

### 4.1 New chronological probability screen

The new screen trains on 2023 and evaluates 2024, then trains on 2023–2024 and evaluates 2025. It uses projected QB identity and includes rest, with the existing injury features. Five fixed specifications were compared without a hyperparameter search. One tied game is omitted from binary scoring, leaving 511 observations.

| Method | 2024 Brier | 2025 Brier | Pooled Brier | Pooled log loss |
|---|---:|---:|---:|---:|
| OLS margin + normal probability | 0.221880 | 0.226838 | 0.224354 | 0.638646 |
| Same, with 0.65 shrink toward 50% | 0.225740 | 0.227892 | 0.226814 | 0.645139 |
| OLS + empirical training residuals | **0.219563** | 0.227049 | **0.223299** | **0.636557** |
| Standardized ridge + normal probability | 0.222269 | 0.226771 | 0.224515 | 0.638944 |
| Regularized direct logistic win model | 0.225478 | **0.225915** | 0.225696 | 0.641927 |

The empirical residual approach improves pooled Brier by approximately 0.001055. A season-stratified week-block bootstrap gives a 95% interval of **−0.003049 to +0.000853** for challenger minus baseline Brier. It worsens slightly in 2025. This is a lead for distribution research, not a confirmed winner. The other fixed specifications also fail to establish an improvement; this does not reject their entire model families.

On 470 matched games whose stored DraftKings closing captures are before kickoff and within 60 minutes, baseline Brier is **0.227115**, empirical-residual Brier **0.226367**, and market Brier **0.209675**. The market advantage remains substantial. These closing comparisons assess predictive information at the close; they do not measure an early-price strategy.

Limitations remain: historical periods have been reused, injury publication times are not certified, the injury feature still has the previously identified QB overlap, and residual distributions are estimated from training residuals rather than an independent calibration period. The bootstrap does not solve model-selection reuse or every form of temporal dependence. The study does not reproduce the full Streamlit serving behavior, and does not recommend removing production shrink solely from this table.

### 4.2 Model information the market has not already absorbed

The most useful next team-market baseline is the **market available at the decision time**, not an isolated sports model compared only with the final close. Test a regularized correction such as `logit(p_final) = logit(p_market_at_t) + delta(features_at_t)`. Include a market-only model and constrain the correction enough that it can shrink to zero.

Candidate residual information includes an explicitly timestamped QB change, updated expected participation, reliable changes in player workload, weather forecast revisions, and disagreement between independently sourced prices. The important question is whether a feature adds information after the contemporaneous price is known. A large model-market disagreement should be a diagnostic feature, not proof that the market is wrong.

Use a small linear or generalized additive correction first. A shallow boosted tree can then test interactions on the same frozen folds. Tabular-learning research provides a reason to include tree ensembles as baselines, not evidence they will beat this sportsbook market.[^7] Deep neural models and a GPU purchase have no present justification from the size of the three-season odds-labeled game sample.

### 4.3 Separate the mean from the distribution

A single normal margin distribution ignores integer scoring, pushes, key margins and potentially changing variance. A better mean does not guarantee better spread or moneyline probabilities. Test three increasingly demanding alternatives: a regularized empirical residual distribution; a simple conditional scale model; and a discrete distribution over margins or jointly over team scores.

For conditional variance, start with a few defensible inputs such as the contemporaneous total, favorite strength, projected pace, outdoor conditions and QB uncertainty. Estimate scale from past out-of-sample residuals where feasible. A distributional boosting method such as NGBoost is a later candidate because it can learn multiple distribution parameters; its general benchmark results are not NFL evidence.[^8]

Model score distributions should generate moneyline, spread and total probabilities consistently, including ties and pushes. Evaluate them at real posted lines, using appropriate discrete probabilities. Do not hard-code a key-number bump selected from the same three seasons used to judge the improvement.

### 4.4 NFL props should model opportunity before efficiency

The committed NFL props calibration has only three entries; passing yards and rushing yards are the acknowledged placeholder pair. That is a product-readiness gap, not a small tuning task. Existing player-week and snap data create a plausible foundation for a real model, but do not by themselves establish usable pregame forecasts.

Use a staged generative design:

1. Forecast team opportunities: offensive plays, attempts, dropbacks and rushes, conditional on information available before the game.
2. Forecast a player's share of opportunities using recent role, prior role, team membership and timestamped availability.
3. Forecast efficiency per opportunity and integrate over the uncertain workload.

For receptions, model targets and catch probability separately. For rushing yards, forecast carries and a yardage distribution that permits losses; a plain nonnegative count distribution on yards is misspecified. For passing yards, forecast volume and efficiency with residual uncertainty. For touchdowns, distinguish a count model from the probability of scoring at least once. Shared team constraints should prevent individual projected target shares from collectively exceeding the modeled opportunity pool.

Start with **receptions and rushing attempts**, provided the offer corpus supports them, because their opportunity mechanisms are more directly observed than many yardage tails. Compare player mean, recent mean, role-adjusted count model, and market-only baselines. Use hierarchical shrinkage by player/position/team and allow an explicit role-change component, rather than applying one recency window to every player.

The outcome table must distinguish an active player with zero targets, an active player with zero receptions, a nonparticipant whose wager is void, and an unresolved identity or result. The independent review found that most missing player-week rows in the apparent 0.5-reception edge belonged to players with positive snaps and no targets. A reusable model must retain those valid zeros without converting all missing rows into losses. Settlement eligibility is book-, market- and rule-version-specific.[^9]

### 4.5 Data availability is part of the model

The local 2026 depth-chart file has **505,422 rows but only `team` and `gsis_id` columns**. It lacks snapshot time, position and rank, so it cannot supply a time-correct depth order. The current validator treats enrichment layers largely as informational and can report large row counts without establishing that consumers can use the data.

Require model-specific schemas and coverage, not just file existence. A margin model dependent on injuries and QB projections needs a clear policy for missing layers: use a separately validated reduced model, or mark the estimate unavailable. Treating missing injury information as no injury is not the same statistical assumption.

The upstream nflverse availability page still states that its injury source ended after 2024, while the local mirror contains 6,068 rows labeled 2025 injuries. This discrepancy does **not** prove the local file is invalid or empty. It means the source/version and availability history need reconciliation. The same page confirms the depth-chart change from week keys to timestamped snapshots.[^10]

## 5. MLB predictive development beyond the opportunity fix

### 5.1 Estimate rates and uncertainty with partial pooling

The next statistical increment after opportunity modeling is a hierarchical rate model. Estimate a player's latent rate from that player's observations plus an appropriate population prior, with the amount of shrinkage determined by evidence. Early-season players and recent role changes should not receive the same certainty as established full-season players.

Empirical-Bayes batting research directly studied forecasting later performance from earlier observations and found advantages over simply carrying forward a raw batting average. Its results concern a historical batting-average problem, not contemporary sportsbook profitability; they justify a candidate family, not a promoted betting model.[^11]

For batter hits, compare a beta-binomial or related hierarchical model against the current xBA-driven approach while preserving the same exposure model. The posterior uncertainty of the hit rate and the randomness of the next game's hits are different quantities. Report both where useful. Do not substitute the player's outcome variance for uncertainty about the model's estimated probability.

### 5.2 Match distributions to the statistic

Total bases is a compound outcome of singles, doubles, triples and home runs, with many zero outcomes and variable opportunities. A negative binomial can be a useful approximation, but it does not encode those mechanisms. A multinomial plate-appearance model plus an opportunity distribution is a plausible challenger. RBI additionally depends on runners and batting order; a team-context opportunity model is more defensible than multiplying an otherwise unchanged average by another context factor.

Pitcher strikeouts should couple strikeout rate to batters faced and managerial removal. Expected innings alone misses uncertainty from pitch counts, recent workload, opener roles, return from injury and game state. Earned runs and outs share that stopping mechanism: poor performance can shorten exposure. Compare a joint workload/rate model with the existing negative-binomial and residual methods before introducing a full game simulator.

For team markets, preserve the additive-run model's useful starter/bullpen decomposition. Test changing one component at a time. A lineup-based offense estimate should **replace** the corresponding team-offense estimate rather than be added on top of it; the prior rejected additive lineup experiment already provides evidence about double counting.

### 5.3 Revisit rejected experiments only when the hypothesis changes

The memory records negative results for multiple recency windows, weather effects on props, CSW blending, opponent adjustments, lineup additions and other features. Those are valuable results. They should not be repeated with the same data and a more favorable selection rule.

Legitimate reasons to reopen an idea include a corrected exposure model, repaired game joins, new pregame information, a replacement rather than additive feature, a different target such as workload or tail probability, or a materially larger future sample. The improvement study should preserve each original negative result alongside the changed hypothesis.

For example, the at-bat rounding issue provides a specific reason to revisit exposure adjustments after the probability model changes. It does not justify turning every previously rejected matchup multiplier back on. Similarly, the totals sign correction changes the NFL totals baseline; it does not establish that all weather features are useful.

### 5.4 Pitch physics and simulation are later-stage options

The local Statcast mirror has contact/outcome fields but lacks the velocity, spin, movement and release fields needed for a pitch-physics model. Those fields are documented in the public Statcast CSV format.[^12] A useful initial study would model whiff or another clearly defined pitch outcome using physics known from **previous** pitches, then test whether the resulting pitcher summary improves a future workload or rate forecast.

FanGraphs' Stuff+ primer describes a model based on physical pitch characteristics and separates it from location-aware and combined pitching models. That supports distinguishing the mechanisms instead of labeling any contact-quality metric “Stuff+.”[^13] A custom model would require an expanded schema, versioned training corpus, season/pitcher-aware evaluation and protection against using the target game's pitch measurements as predictors.

A plate-appearance simulator could eventually unify team runs, player outcomes and same-game dependence. Its value would be coherence and conditional distributions; simulation cannot create predictive information absent from its inputs. Begin with a small validated distributional model. Preserve extra innings, walk-off truncation, home-team opportunity, substitution and relief-pitcher changes when extending it. Defer pitch-by-pitch simulation and GPU optimization until a simpler model demonstrates a useful limitation.

## 6. NBA and shared modeling

NBA's clearest modeling opportunity is playing time and role. The current reliability filter removes low-participation and interrupted histories and requires a rebuilt streak. This can produce a stable conditional sample while leaving the next game's participation uncertainty unmodeled. A forecast for a healthy established starter and a returning reserve should have different opportunity distributions, not just different sample sizes.

Develop a minutes distribution, role/usage model and per-minute production model with team context. Evaluate role transitions separately: starting-lineup changes, teammates absent, traded players and restricted returns. Those cases need timestamped information. Do not use the completed game's minutes, starting designation or teammate participation as if known pregame.

The `nba_api` project supplies a maintained interface and endpoint documentation for NBA data, but an adapter is not a durable warehouse or availability guarantee. Its release notes also document endpoints becoming unusable, reinforcing the need for schema tests, caching and source-specific fallbacks.[^14] Build sport-specific game/player facts behind a shared contract, rather than forcing NBA possessions or NFL snaps into MLB at-bat assumptions.

Across sports, keep a small common core for offers, identities, probability distributions, calibration, settlement and evaluation. Sport-specific features and participation rules should remain explicit. Copying calibration files or assuming a method that wins for MLB hits also wins for NFL passing yards would bypass the central empirical question.

## 7. Calibration, market information and uncertainty

Calibration should be evaluated as part of the final composition. The MLB team path intentionally combines probability shrinkage with market blending; whether that composition is useful is an empirical question. Its effective weight on the raw model can be much smaller than either individual setting suggests. Store and report the final effective formula, and compare it with the simpler market-only forecast.

Fit calibrators using predictions that were not fitted on the same outcomes. Scikit-learn's calibration guidance explicitly warns about fitting a calibrator from optimistic training predictions.[^15] A small calibrator comparison can include no extra calibration, logit-Platt, beta calibration and sufficiently supported isotonic regression. The repository's `sigmoid(a*logit(p)+b)` already includes the identity map at `a=1, b=0`; lack of an identity map is therefore **not** a valid criticism of this implementation. Beta calibration is a candidate for asymmetric distortion, not an automatic upgrade.[^16]

Per-line calibration can improve a frequently observed threshold while creating incoherent probabilities across thresholds. Require nonincreasing `P(X > line)` across increasing lines, nonnegative push mass and a normalized underlying distribution. Check these properties after every overlay, including online updates. Avoid reporting a very precise probability when its estimate is unsupported in that bucket or feature regime.

For a one-unit stake at decimal odds `d`, with unconditional win/loss/push probabilities, expected profit is `p_win*(d-1) - p_loss`. A push contributes zero profit. Break-even decimal odds are `1 + p_loss/p_win`, assuming no other settlement costs. The familiar `p*d-1` applies when `p` is a binary probability for a wager without pushes, or when the treatment of conditioning is explicitly consistent. This matters for integer totals, spreads, player counts and NFL ties.

Separate three kinds of uncertainty: random game outcomes, uncertainty about estimated parameters, and uncertainty about the information state such as who will play. Disagreement among A–E methods is not automatically a valid confidence interval because the methods share data and assumptions. A high outcome variance also does not imply low EV. Validate any uncertainty-based abstention or sizing rule against both predictive scores and coverage.

Conformal methods can help monitor interval coverage, with adaptations for distribution shift. Coverage of an outcome interval is not a calibrated probability for a particular betting side and does not establish a profitable price. Treat conformal tools as supplementary diagnostics rather than a shortcut from “90% interval” to a 90% safe bet.[^17]

## 8. Data, execution timing and source strategy

Create a compact model-input manifest for every run: game IDs, data cutoffs, source versions, required fields, coverage, missingness and hashes. Historical facts, derived features and model predictions have different invalidation rules. Cache a prediction only with the model and calibration fingerprint that produced it. A corrected source value must invalidate relevant features even when row count and maximum date are unchanged.

For live odds, distinguish provider snapshot time, bookmaker update time, retrieval time and analysis completion time. The current relative staleness trim can retain missing timestamps and considers freshness against peers; it does not establish that an entire group is fresh in absolute time. A predicted edge on an unavailable or stale quote is not actionable.

The transient-line hypothesis requires a separate observation design. Log permitted books and reference prices at a defined cadence, preserve all observations, record candidate creation and quote rechecks, and measure whether the apparent advantage survives realistic delay. Use the existing three stored windows to estimate coverage and coarse movement first. They cannot reveal the duration of sub-hour price discrepancies. A paid high-frequency logger is a future implementation and budget decision, not an activity authorized by this study.

Closing-line value should retain exact-line matching and price movement as separate measurements. A closing reference is useful evidence, but CLV alone is not a guarantee of positive expectation, particularly when the closing reference is noisy, stale or from the execution book itself. Report availability and unmatched cases rather than manufacturing a close at a different line.

Weather research needs a true forecast vintage. Open-Meteo distinguishes stitched historical forecasts from individual model runs. Its Single Runs documentation also states that initialization time precedes public availability, and describes different archive start dates and some hindcast coverage. A retrospectively generated hindcast is not evidence that the same forecast was published before a historical bet. Verify the archive's vintage, model version and release lag before using it in a trading simulation.[^18]

Prefer existing stored raw data and free official/community sources before buying additional coverage. Restoring discarded FanDuel lines from preserved payloads, where available, may be more valuable than purchasing another backfill. Source availability and historical depth must be checked per field and per season; a library's broad feature list does not certify the local corpus.

## 9. Application performance and architecture

### 9.1 Measured local opportunities

| Component | Existing approach | Prototype / alternative | Measurement and limit |
|---|---|---|---|
| NFL EPA at 58 historical dates | Repeatedly scan 35,082 plays | Accumulate daily sums/counts and snapshot before each date | 0.680 s → 0.0277 s, 24.6× for this component; maximum numerical difference 0 |
| Read 2025 NFL PBP | All columns | Five required EPA columns | Arrow buffers 21.12 MB → 3.42 MB; median local reads 8.21 ms → 4.93 ms |
| Independent parlay probability | 5,000 Monte Carlo samples per call | Exact product | Removes simulation error entirely under the independence assumption |

The EPA prototype includes construction of the daily index but excludes data loading and prior-season shrink assembly. The existing application already caches repeated aggregate requests; the gain applies to constructing many previously unseen cutoffs, not every repeat live request. The Parquet comparison is a small three-read local measurement with operating-system cache effects. Neither result estimates Azure or full Streamlit latency.

Column projection and row-group filtering are supported by Arrow; use them before adopting another analytics engine solely for speed.[^19] A read-only SQL-style analytical layer such as DuckDB could later be compared on actual research workloads, but another datastore is not needed to obtain the demonstrated gains.

### 9.2 Remove maintenance from interactive prediction

`props.analyze_player_props_value` calls `maybe_auto_refit`, and [recalibration.py:2360](../recalibration.py) calls maintenance synchronously when its gate opens. Historical resolution, ingestion and model updates therefore share the user's prediction path. The current phase-3 thread pool improves parallel event processing but does not make this lifecycle reproducible or bound maintenance latency.

Introduce a pure prediction service that accepts a frozen input bundle and returns forecasts. Give maintenance a separate bounded job interface, progress state and retry policy. A simple scheduled task or separate worker is sufficient for a single-user application; a durable job queue is warranted only if operational requirements demand it. A daemon thread alone is not a durable scheduler on a restarting host.

Freeze one model/calibration version per slate so maintenance cannot change probabilities halfway through an analysis. Keep Streamlit calls in the script thread; the official threading guidance describes why background workers should return results instead of directly accessing session UI state.[^20]

### 9.3 Measure the actual bottleneck

Instrument phase timings, SQL query counts and pool wait, rows transferred, cache hit rate, provider duration, and maintenance duration. Benchmark one game and a full slate in both cold and warm states. Record median and tail latency over repeated runs. The report cannot assign current production p95 latency because the deployed application and live SQL were not profiled.

Then consolidate repeated player histories across prop markets, batch identity and outcome reads, and precompute sufficient statistics for stable full-season models. Avoid replacing a distribution with a mean-only cache: methods A/C and context-dependent filters need more information. Choose concurrency by measured database capacity; increasing 16 or 20 workers can simply move waiting into the database pool. SQLAlchemy's performance guidance distinguishes database execution, fetch and Python materialization costs, which require different fixes.[^21]

For Streamlit, cache immutable data by data/model version and use explicit freshness for mutable sources. Share connection engines rather than live transaction state. Streamlit distinguishes copied cached data from shared cached resources; the latter require thread-safe use.[^22]

### 9.4 Improve boundaries incrementally

The 1,037-line prop analyzer and large backtest/refit modules make it difficult to prove that research, serving and grading use the same transformation. Extract interfaces according to statistical responsibility: input assembly, projection, predictive distribution, calibration, offer evaluation, selection and settlement. Share those functions across live and historical paths.

Keep the current SQL warehouse and mirror architecture unless measurements justify a change. Introduce versioned, typed records at boundaries rather than rewriting the application. Candidate promotion should validate the full nested configuration, produce a complete diff and preserve rollback artifacts. Record which exact model generated every forward prediction and wager recommendation.

## 10. Product reliability and usability

The primary screen should make the decision assessable: offered book/line/price, quote age, model probability, break-even price, expected return and a short statement of input quality. Show confirmed versus projected participation and whether the full or reduced model is running. Put implementation details and detailed source diagnostics in an expandable view.

Use explicit states such as “estimated,” “waiting for lineup,” “stale price,” “insufficient model coverage,” and “ready at this quoted price.” A missing calibration or failed injury layer should not look identical to a successful complete prediction. Existing telemetry provides a foundation, but process-local counters alone are insufficient to reconstruct a problem after a restart.

Separate a forecast from a recommendation and a recommendation from an executed wager. Preserve the actual executed price and timestamp. Show the effect of a changed price on EV before recording a ticket, without automatically requesting a paid refresh. Explain why a candidate was excluded so selection rules can be reviewed empirically.

“Safe Mode” should show its modeled hit probability and price tradeoff without implying certainty. A high hit probability can still be poor value. Alternate-line probabilities should come from the same coherent distribution as standard lines, and same-game combined value should remain unpriced until a real ticket quote is supplied.

The performance page should show prospective results by model version, decision horizon, sport and market, with sample counts, uncertainty and unresolved/void rates. Include a market-only benchmark. Retain the research history separately so a favorable historical experiment does not appear to be the application's forward record.

## 11. Verification and experiment plan

This study's six offline measurement commands completed and produced separate JSON results: inventory, deterministic decision examples, performance, historical quote comparison, NFL probability screen and MLB opportunity screen. The experiment helper blocks provider requests and writes only study outputs. Production code, calibrations and stored sports facts were not changed.

A focused existing suite ran 164 tests: **161 passed and 3 failed**. Two telemetry tests passed when rerun with the mirror disabled because installed mirror data had bypassed their mocked SQL failures. The remaining `test_distributional.SelectLineMethodsTests.test_deep_bucket_adopts_d` still expects two buckets although the implementation now defines `[0.5, 1.5, None]`. This is evidence of test/contract drift, not evidence that the two-bucket behavior should be restored. The default suite was not green, and no full-application or deployed-environment certification is claimed.

Build integration fixtures around complete decisions: a night NFL game, a QB change, missing injury data, an MLB doubleheader, a returning player, a valid zero, a void, an integer-line push, a changed executable price, and a repeated stake submission. Run the same fixture through research and serving. Streamlit's AppTest supports exercising the app's input/output path without a browser, with all external providers replaced by fixtures.[^23]

| Experiment | Fixed comparison | Adoption condition | Rejection or stop condition |
|---|---|---|---|
| Baseline repair | Same eligible games before/after each correctness fix | Identity/time/serving invariants pass; changed metrics are explained | Any unresolved wrong-game join or future input |
| MLB opportunity distribution | Rounded D+xBA vs adjacent and role-conditioned opportunity mixtures, each recalibrated | Stable proper-score gain at supported real lines; smooth/monotone probabilities; acceptable coverage | Gain disappears in complete serving composition or depends on postgame role filtering |
| NFL props pilot | Player mean/recent mean vs opportunity-plus-rate vs market-only | Forward probability improvement and stable participation/zero handling | Advantage comes from omitted zeros, unknown starters or unusable quotes |
| Market residual model | Same-time market-only vs market plus bounded model correction | Incremental out-of-time score gain; executable-price evaluation survives delay | Correction adds no stable information beyond market |
| NFL conditional distribution | Fixed normal vs residual/conditional scale/discrete alternatives | Better distribution scores and real-line calibration, including pushes | Tail/line gains fail another period or require extensive retuning |
| Offer-level execution | Same selections priced DK-only vs best permitted contemporaneous offer | No line/rule/timestamp mismatches; measurable attainable price improvement | Improvement relies on unmatched lines or unavailable offers |
| Selection and sizing | Current heuristic vs exact small-pool optimizer, then open-portfolio policy | Feasible choices improve declared utility; drawdown/exposure constraints hold | Gains arise only from optimistic probability estimates or changed stake accounting |
| Performance | Fixed cached fixtures, one game/full slate, cold/warm | Identical forecast/selection outputs, lower measured latency or resource use | Faster output changes inputs, freshness or probabilities |

Adoption thresholds should be declared before each full experiment and scaled to economic relevance and sample size. A universal 0.002 Brier hurdle can reject a cheap, stable improvement while accepting an expensive one without showing betting utility. Correctness invariants should not depend on finding positive historical ROI.

## 12. Prioritized implementation roadmap

Effort ranges below are planning estimates in focused engineering days, including focused verification. They exclude waiting for future observations, source-access problems and major data reconstruction. They are not delivery promises or instructions to purchase data.

| Priority | Work package | Why it is ordered here | Estimated effort |
|---|---|---|---:|
| 1 | Repair review defects and create a frozen end-to-end baseline | Every predictive comparison depends on trustworthy identities, timestamps and serving features | 3–7 days |
| 2 | MLB at-bat opportunity challenger with full D+xBA recalibration | New, directly reproduced specification defect with encouraging component evidence | 2–4 days |
| 3 | Offer preservation and DK/FD exact-line pricing | Measured price opportunity; benefits decisions independently of new sports signal | 3–6 days |
| 4 | NFL participation-aware outcome table and two-prop opportunity pilot | Largest underdeveloped predictive area with useful local input layers | 5–10 days |
| 5 | Same-time market benchmark and bounded residual model | Tests whether new information adds beyond prices already available | 2–5 days |
| 6 | Correct parlay value labeling, exact independent probabilities and open-exposure limits | Removes avoidable numerical and decision-accounting weaknesses | 2–4 days |
| 7 | Time-aware model manifests, candidate diffs and prospective experiment registry | Makes improvements reproducible and prevents validation drift | 2–5 days, partly shared with priority 1 |
| 8 | NFL conditional/discrete distribution experiment | Modest residual-distribution lead; better target for tail research than another broad feature sweep | 3–6 days |
| 9 | Pure prediction boundary, bounded maintenance and targeted batching | Improves consistency and quote timeliness; profile before broad optimization | 3–6 days |
| 10 | MLB hierarchical rates and pitcher workload pilot | Mechanistically meaningful new information/uncertainty treatment | 4–8 days |
| 11 | NBA statistics/role-minutes foundation | Needed for credible NBA improvements; secondary to the active NFL program | 5–10 days |
| 12 | Pitch physics, boosted distribution models and full simulation | Higher cost, less direct local evidence of incremental market information | Scope only after simpler candidates pass |

Begin with baseline repair, the MLB opportunity experiment and the NFL props outcome foundation. Offer preservation and parlay/exposure correctness are worthwhile application changes alongside that research. Reserve most modeling effort for opportunity, availability and full distributions; retain team-feature sophistication as a bounded hypothesis rather than the assumed route to predictive improvement.

## Sources and supporting artifacts

The external references below support methods, data semantics and implementation options. None establishes profitability for this application. Documentation was accessed September 8, 2026; current documentation can differ from the installed versions and historical source behavior.

1. SciPy. [Sobol quasi-Monte Carlo reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html). Current documentation; balance requirements and replicated scrambling.
2. Busseti, Ryu and Boyd. [Risk-Constrained Kelly Gambling](https://web.stanford.edu/~boyd/papers/kelly.html). *Journal of Investing*, 2016. Growth/drawdown tradeoffs under an assumed return distribution.
3. The Odds API. [API v4 documentation](https://the-odds-api.com/liveapi/guides/v4/). Historical snapshot behavior, errors, coverage and quota semantics.
4. Scikit-learn. [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Chronological splits and assumptions.
5. Gneiting and Raftery. [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). *JASA*, 2007.
6. Bailey, Borwein, López de Prado and Zhu. [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf). Authors' manuscript; backtest selection and CSCV.
7. Grinsztajn, Oyallon and Varoquaux. [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815). 2022. General tabular benchmark, not sports evidence.
8. Duan et al. [NGBoost: Natural Gradient Boosting for Probabilistic Prediction](https://proceedings.mlr.press/v119/duan20a.html). ICML/PMLR, 2020.
9. DraftKings / Massachusetts Gaming Commission. [DraftKings House Rules, May 14, 2024](https://massgaming.com/wp-content/uploads/DraftKings-House-Rules-5.14.24.pdf). Football player-prop participation provisions, PDF page 52. Historical example; apply the correct book/date/jurisdiction rules to each ticket.
10. nflverse / nflreadr. [Data Update and Availability Schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html). Depth-chart schema transition and injury-source statement; the latter conflicts with local 2025 file presence and needs reconciliation.
11. Brown, Lawrence D. [In-season prediction of batting averages: A field test of empirical Bayes and Bayes methodologies](https://arxiv.org/abs/0803.3697). *Annals of Applied Statistics*, 2008; study of 2005 batting records.
12. MLB Baseball Savant. [Statcast Search CSV Documentation](https://baseballsavant.mlb.com/csv-docs). Pitch physics and outcome-field definitions.
13. McGrattan, Owen. [Stuff+, Location+, and Pitching+ Primer](https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/). FanGraphs, March 10, 2023; explanation by a model maintainer.
14. `nba_api` maintainers. [Project documentation](https://github.com/swar/nba_api/blob/master/README.md) and [release notes](https://github.com/swar/nba_api/releases). NBA access and endpoint changes.
15. Scikit-learn. [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html). Calibration data independence and diagnostic methods.
16. Kull, Silva Filho and Flach. [Beta calibration](https://proceedings.mlr.press/v54/kull17a.html). AISTATS/PMLR, 2017.
17. Gibbs and Candès. [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170). 2021. Online prediction-set coverage under changing distributions.
18. Open-Meteo. [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) and [Single Runs API](https://open-meteo.com/en/docs/single-runs-api). Archive structure, initialization/publication distinction and coverage limitations.
19. Apache Arrow. [Reading and Writing the Apache Parquet Format](https://arrow.apache.org/docs/python/parquet.html). Column projection and row-group filtering.
20. Streamlit. [Multithreading](https://docs.streamlit.io/develop/concepts/design/multithreading). Worker/script-thread boundaries.
21. SQLAlchemy. [Performance FAQ](https://docs.sqlalchemy.org/en/20/faq/performance.html). Query, fetch and application profiling.
22. Streamlit. [Caching overview](https://docs.streamlit.io/develop/concepts/architecture/caching). Data copies, shared resources and freshness.
23. Streamlit. [App testing](https://docs.streamlit.io/develop/api-reference/app-testing). Complete app-path tests with controlled inputs.

Supporting local artifacts:

- [Standalone reading copy](APPLICATION_IMPROVEMENT_STUDY_2026-09-08.html) and [roadmap spreadsheet](improvement_study_20260908_tables/08_roadmap.csv). Every report table is also exported as CSV with source/context columns.
- [Original independent review](INDEPENDENT_REVIEW_2026-09-08.md): reproduced baseline defects and limitations.
- [Study measurement helper](improvement_study_20260908.py): six independent offline commands, with assumptions recorded in each output.
- [Reproduction instructions](IMPROVEMENT_STUDY_RUNBOOK_2026-09-08.md): commands, scope, known test failures and artifact checks. The inventory includes SHA-256 hashes of root Python source and local Parquet inputs.
- [MLB opportunity results](improvement_study_20260908_mlb.json), [NFL probability results](improvement_study_20260908_nfl.json), [quote comparison](improvement_study_20260908_quotes.json), [decision examples](improvement_study_20260908_decisions.json), [component performance](improvement_study_20260908_performance.json), and [source/data inventory](improvement_study_20260908_inventory.json).

[^1]: [Sobol quasi-Monte Carlo reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html). Publication details: source 1 above.
[^2]: [Risk-Constrained Kelly Gambling](https://web.stanford.edu/~boyd/papers/kelly.html). Publication details: source 2 above.
[^3]: [API v4 documentation](https://the-odds-api.com/liveapi/guides/v4/). Publication details: source 3 above.
[^4]: [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Publication details: source 4 above.
[^5]: [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). Publication details: source 5 above.
[^6]: [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf). Publication details: source 6 above.
[^7]: [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815). Publication details: source 7 above.
[^8]: [NGBoost: Natural Gradient Boosting for Probabilistic Prediction](https://proceedings.mlr.press/v119/duan20a.html). Publication details: source 8 above.
[^9]: [DraftKings House Rules, May 14, 2024](https://massgaming.com/wp-content/uploads/DraftKings-House-Rules-5.14.24.pdf). Publication details: source 9 above.
[^10]: [Data Update and Availability Schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html). Publication details: source 10 above.
[^11]: [In-season prediction of batting averages: A field test of empirical Bayes and Bayes methodologies](https://arxiv.org/abs/0803.3697). Publication details: source 11 above.
[^12]: [Statcast Search CSV Documentation](https://baseballsavant.mlb.com/csv-docs). Publication details: source 12 above.
[^13]: [Stuff+, Location+, and Pitching+ Primer](https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/). Publication details: source 13 above.
[^14]: [Project documentation](https://github.com/swar/nba_api/blob/master/README.md). Publication details: source 14 above.
[^15]: [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html). Publication details: source 15 above.
[^16]: [Beta calibration](https://proceedings.mlr.press/v54/kull17a.html). Publication details: source 16 above.
[^17]: [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170). Publication details: source 17 above.
[^18]: [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api). Publication details: source 18 above.
[^19]: [Reading and Writing the Apache Parquet Format](https://arrow.apache.org/docs/python/parquet.html). Publication details: source 19 above.
[^20]: [Multithreading](https://docs.streamlit.io/develop/concepts/design/multithreading). Publication details: source 20 above.
[^21]: [Performance FAQ](https://docs.sqlalchemy.org/en/20/faq/performance.html). Publication details: source 21 above.
[^22]: [Caching overview](https://docs.streamlit.io/develop/concepts/architecture/caching). Publication details: source 22 above.
[^23]: [App testing](https://docs.streamlit.io/develop/api-reference/app-testing). Publication details: source 23 above.
