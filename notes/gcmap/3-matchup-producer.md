# SUBSYSTEM
MLB per-game matchup_features producer (mlb_starters.build_matchup_features) and its downstream consumers (team totals/spreads/ML in analysis.py, batter/pitcher props in props.py), plus the ESPN pace-factor helpers. This is the shared per-game feature object built once in app.py and passed to every market and every prop.

## FUNCTIONS
- **build_matchup_features** (mlb_starters.py:867-1013)
  THE producer. Assembles the per-game matchup_features dict from probable starters + team offense splits + bullpen + Savant expected-runs factors. Returns None-tolerant nested dict. This is where a GameContext would naturally be produced.
- **get_pitcher_quality** (mlb_starters.py:695-768)
  Season pitching quality for a starter id: {'name','throws','era','ip','avg_ip','bf','k_pct','bb_pct','xera','xwoba','xba','run_suppression','run_suppression_basis'}. run_suppression = clamp(LEAGUE_AVG['era']/xera_or_era, 0.5..2.0). File-cached per pitcher-season (1h TTL).
- **get_team_offense_splits** (mlb_starters.py:771-801)
  Team hitting by opposing-pitcher hand: {'vL':{'ops','k_pct'},'vR':{'ops','k_pct'}}. Season-level from StatsAPI statSplits. Cached per team-season (1h).
- **get_team_bullpen_quality** (mlb_starters.py:804-831)
  Reliever run-suppression: {'rp_era','bullpen_suppression'} where suppression=clamp(LEAGUE_AVG['era']/rp_era). Cached per team-season (24h).
- **get_expected_runs_team_factors** (mlb_starters.py:194-312)
  Leakage-safe (as_of cutoff = as_of-1 day) Savant league-relative xwOBA inputs: {'league_xwoba','league_bullpen_xwoba','offense_vs_hand':{'L':{ABBR:xwoba},'R':{...}},'bullpen_xwoba':{ABBR:xwoba}}. The ONLY as-of-safe piece. Cached per season+cutoff-date (24h).
- **expected_runs_from_factors** (mlb_starters.py:350-375)
  Converts (base_runs, offense_factor, staff_suppression, offense_weight, pitching_weight) into a scalar expected-runs MEAN, clamped 0.5..12.0. Already produces per-team run means — the seed of a run-total model.
- **poisson_margin_probability** (mlb_starters.py:393-424)
  ANALYTIC joint two-team run distribution: builds independent Poisson pmfs over 0..30 runs for home/away expected runs and sums P(home+spread>away). No Monte-Carlo. Currently used for spread cover only.
- **negative_binomial_margin_probability** (mlb_starters.py:427-472)
  Over-dispersed analytic joint run distribution (variance=mean+dispersion*mean^2); falls back to Poisson at dispersion=0. Same 0..30 pmf convolution pattern.
- **get_probable_starters** (mlb_starters.py:510-538)
  {norm_team_name:{'pitcher_id','name','team_id'}} for a date. Cached per date (1h).
- **get_confirmed_lineup / lineup_player_context** (mlb_starters.py:541-611)
  Announced batting orders per game + per-player {batting_order,...}. NOTE: this is the ONLY place the full set of a team's batters is available — needed for any bottom-up 'sum of batter hits' reconciliation, but props.py never aggregates it.
- **get_team_index / _match_team_id** (mlb_starters.py:475-507)
  {norm_name:{'id','name','abbr'}} for all 30 teams; tolerant name->StatsAPI-id/abbr matcher. Cached per season (7d).
- **_mlb_matchup_features / _matchup_features_for** (backtest.py:997-1041)
  Backtest builder that reproduces app.py's live features; caches team_index per season in module-global _MLB_TEAM_INDEX (backtest.py:994). NOT as-of safe for MLB (pulls season-final stats).
- **analyze_totals_value** (analysis.py:341-532)
  Consumer. Computes projected_total (a GAME TOTAL, analysis.py:400-402) from ESPN recency scoring, then applies starter_total_shift = -(run_scale*excess)-(bullpen_w*bullpen_excess) (analysis.py:408-424). Normal model P(over) via total_std. This game total is decoupled from the spread analyzer's per-team run means.
- **analyze_spreads_value / _mlb_expected_runs_projection** (analysis.py:535-721 / 136-192)
  Consumer. _mlb_expected_runs_projection returns {'home_runs','away_runs','margin','spread_share','margin_share'} from expected_runs factors; analyze_spreads_value feeds home_runs/away_runs into poisson_margin_probability (analysis.py:690,707) for cover prob. Spread-only (gated by live_markets['spreads']).
- **_predict_margin** (analysis.py:77-133)
  Consumer (ML+spreads). Shifts pred_margin by _starter_adjustment('spreads')*starter_edge (analysis.py:126-128). Baseline Normal margin from ESPN recent-game margins.
- **_mlb_prop_matchup_mult** (props.py:93-162)
  Consumer (props). Bounded [0.7,1.4] per-player multiplier. Pitcher props use matchup_features[own side].opp_offense_vs_hand; batter props use matchup_features[opp side].starter (opposing starter's xBA/k_pct via log5). Uses player_context {base_projection,expected_exposure}. Entirely per-player, NO team/game aggregation.
- **analyze_player_props_value** (props.py:864-... (call site props.py:1199-1205; base_proj props.py:1122; avg_stat props.py:1254))
  Consumer. Loops player-by-player: base_proj (recency+shrinkage+xBA blend) * combined_mult -> avg_stat; over_rate via empirical / method D binomial (_distributional_over_rate props.py:471) / B calibration / E negbin. No cross-player or team-run linkage exists.
- **get_team_pace_factor** (espn_client.py:210-256)
  ESPN core-stats 'paceFactor' (possessions/game proxy). NBA/NHL only; MLB has no pace. Returns None when unavailable.
- **cached_pace_factor / _team_pace_lookup** (backtest.py:2207-2244)
  File-caches pace per team-season (key pace/{sport}/{league}/{team_id}/{season}, 7d TTL); _team_pace_lookup builds {team:pace}+league_avg. Backtest-side, not MLB-relevant.

## DATA_FLOW
ENTRY (live, app.py:2608-2620): for each MLB event, game_date is the US-Eastern local date derived from commence_time; app.py calls mlb_starters.build_matchup_features(home, away, game_date, int(game_date[:4])) ONCE per game (app.py:2619) and passes the SAME returned object into analyze_moneyline_value/analyze_spreads_value/analyze_totals_value (app.py:2708-2712) AND analyze_player_props_value(matchup_features=...) (app.py:2747). confirmed_lineup and probable_starters are fetched separately (app.py:2624,2632) and NOT folded into matchup_features.

EXACT matchup_features SHAPE (build_matchup_features return, mlb_starters.py:884-1013):
{
  'home': None | {'starter': <pitcher_quality dict>, 'opp_offense_vs_hand': {'ops':float,'k_pct':float} | None, 'bullpen': {'rp_era':float,'bullpen_suppression':float}},
  'away': None | {same shape},
  'starter_edge': None | float in ~[-1,1]   (tanh(_eff(home)-_eff(away)); only set when BOTH sides have resolved starter quality),
  'expected_runs': {                          (only added when both starters resolve, mlb_starters.py:1004-1011)
      'complete': bool,                       (True iff all four factors non-None)
      'home_offense_factor': float|None,      (away offense vs home starter hand? NO — home team offense vs AWAY starter hand, ~1.0-centered, clamp 0.5..2.0)
      'away_offense_factor': float|None,
      'home_staff_suppression': float|None,   (innings-weighted starter+bullpen xwOBA suppression, 0.5..2.0)
      'away_staff_suppression': float|None }
}
SIDE SEMANTICS (critical): matchup_features[side]['starter'] is THAT side's own probable starter; matchup_features[side]['opp_offense_vs_hand'] is the OPPOSING lineup's OPS/K% vs that starter's hand; matchup_features[side]['bullpen'] is that side's own bullpen. So pitcher props read matchup_features[own_side]; batter props read matchup_features[opp_side]['starter'] (the pitcher they face). upcoming_is_home for a prop is resolved per player from ESPN team id, not from this dict.

pitcher_quality dict (get_pitcher_quality, mlb_starters.py:733-766): {'name','throws'('L'|'R'),'era','ip','avg_ip','bf','k_pct','bb_pct','xera','xwoba','xba','run_suppression'(clamp LEAGUE_AVG.era/basis,0.5..2.0),'run_suppression_basis'}.

TRANSFORM per market:
- Totals (analysis.py:400-424): projected_total = mean(off_sum, def_sum) from ESPN recent_games; then += -(run_scale*sum(starter.run_suppression-1)) -(bullpen_w*sum(bullpen.bullpen_suppression-1)). total_std from weighted historical totals; P(over)=1-Phi((line-projected_total)/total_std). This projected_total IS a game-total scalar but has no per-team split and no link to expected_runs.
- Spreads (analysis.py:571-577,689-710): _mlb_expected_runs_projection -> {home_runs,away_runs,margin,spread_share,margin_share} via expected_runs_from_factors(base_runs,offense_factor,staff_suppression,weights); pred_margin blended by margin_share; cover prob via poisson_margin_probability(home_runs,away_runs,spread). expected_runs config loaded from load_expected_runs_challenger; gated by enabled + live_markets['spreads'].
- ML (analysis.py:126-128,230-237): pred_margin += _starter_adjustment('spreads')*starter_edge; P(home win)=Phi(pred_margin/pred_std).
- Batter/pitcher props (props.py:93-162, called 1199-1205): raw ratio -> mult=1+weight*(raw-1), clamp[0.7,1.4]; batter_hits raw = log5-blended starter+bullpen hit rate vs base_proj (uses starter.xba, avg_ip->starter_share, _MLB_LEAGUE['ba']=0.243). avg_stat=base_proj*combined_mult (props.py:1254). Each player independent.

EXIT: candidate dicts per market/prop with model probabilities, edges, ROI (analysis.py:501-530 totals; 650-682 spreads; props return list). Nothing carries a shared run-total or joint distribution across markets/players.

## GAMECONTEXT_RELEVANCE
CURRENTLY: there is NO single shared per-game run/pace/total object. Three independent, uncoordinated run estimates coexist: (1) analyze_totals_value.projected_total — a GAME TOTAL scalar (analysis.py:402) built from ESPN recency scoring + a starter/bullpen run-suppression shift, with a Normal spread (total_std); (2) analyze_spreads_value expected_runs {home_runs,away_runs,margin} — PER-TEAM expected run MEANS (mlb_starters.expected_runs_from_factors) fed to an ANALYTIC joint Poisson/NegBin run distribution (poisson_margin_probability / negative_binomial_margin_probability, 0..30-run pmf convolution) — but spread-cover-only and never used by totals or props; (3) per-batter _mlb_prop_matchup_mult multipliers keyed off the opposing starter's xBA via log5 — no team hit sum, no team run total, no link to (1) or (2).

ALREADY PRESENT and reusable for 3.1: (a) per-team expected-run MEANS via expected_runs_from_factors + the leakage-safe as-of Savant factor feed (get_expected_runs_team_factors); (b) a working stdlib ANALYTIC joint distribution over both teams' integer run totals (Poisson and NegBin pmf convolution over 0..30) — exactly the analytic moment-matched joint object 3.1 wants, currently thrown away after computing one spread-cover scalar; (c) LEAGUE_AVG (era/ops/k_pct/ba) baselines; (d) per-team OPS/hand splits and bullpen suppression for pace/run inputs; (e) get_confirmed_lineup + lineup_player_context giving the full ordered batter set and slot exposure.

MISSING for a per-game GameContext joint model: (1) no object that binds a game total + per-team run split + a batter-hit distribution into one shared, cached structure; (2) no team-hits layer at all (only team RUNS exist, and only in the spread path) and no mapping team_runs<->team_hits; (3) props are computed per-player in a loop (props.py:1173-1360) with ZERO cross-player aggregation, so 'sum of batter hits' does not exist anywhere and there is no seam that sums the 9 batters or reconciles them to a team total; (4) the totals game-total (ESPN-based) and the spreads per-team runs (Savant-based) are computed from different data and never reconciled to each other; (5) batter props see only the opposing-starter xBA multiplier, never the shared game run/pace environment; (6) 'pace' in this codebase is an ESPN NBA/NHL possessions proxy (get_team_pace_factor) — there is NO MLB pace/PA-per-team concept; a GameContext PA/pace term would be new (closest proxy today is expected_exposure = weighted recent AB/PA per player in props.py:1187, and avg_ip->starter_share for staff workload).

## INTEGRATION_SEAMS
- PRODUCER (primary): mlb_starters.build_matchup_features(home_team, away_team, date, season, team_index=None) at mlb_starters.py:867 — attach the new per-game GameContext here (e.g. result['game_context']=...); it already fetches both starters, both offense-vs-hand splits, both bullpens, and calls get_expected_runs_team_factors(season, date) (mlb_starters.py:960) so per-team run means + as-of Savant factors are already in scope. Return dict assembled at mlb_starters.py:884-1013.
- PER-TEAM RUN MEAN source: mlb_starters.expected_runs_from_factors(base_runs, offense_factor, staff_suppression, offense_weight=1.0, pitching_weight=1.0)->float at mlb_starters.py:350 (already produces the run mean per team, clamp 0.5..12.0).
- ANALYTIC JOINT DISTRIBUTION source: mlb_starters.poisson_margin_probability(home_runs, away_runs, home_spread, max_runs=30) at mlb_starters.py:393 and negative_binomial_margin_probability(home_runs, away_runs, home_spread, dispersion, max_runs=30) at mlb_starters.py:427 — the inner probabilities(expected) closures build the per-team 0..30 pmf that a GameContext should expose (currently discarded after the margin sum).
- APP wiring (single build point, shared object): app.py:2619 matchup_features = mlb_starters.build_matchup_features(home, away, game_date, int(game_date[:4])); passed to team markets at app.py:2708-2712 and to props at app.py:2747 (matchup_features=matchup_features). game_date is US-Eastern (app.py:2610-2616). This is the natural per-game cache/produce point.
- BACKTEST parity: backtest._mlb_matchup_features(home, away, date, sport_key) at backtest.py:997 and _matchup_features_for at backtest.py:1033 must produce the identical GameContext or the odds backtest grades a different model; team_index cached in _MLB_TEAM_INDEX (backtest.py:994).
- TOTALS consumer: analyze_totals_value(game_odds, home_team_stats, away_team_stats, threshold_pct, sport_key, matchup_features) at analysis.py:341; projected_total assembled analysis.py:400-402 and shifted analysis.py:408-424 — the seam to REPLACE the ESPN-based game total with a GameContext-derived total mean+distribution.
- SPREADS consumer: analyze_spreads_value at analysis.py:535 with _mlb_expected_runs_projection(sport_key, matchup_features)->{'home_runs','away_runs','margin','spread_share','margin_share'} at analysis.py:136; poisson call at analysis.py:690,707 — seam to source per-team runs from the shared GameContext.
- ML consumer: _predict_margin(game_odds, home_team_stats, away_team_stats, sport_key, matchup_features=None) at analysis.py:77; starter_edge shift at analysis.py:126-128.
- BATTER-PROP consumer: props._mlb_prop_matchup_mult(prop_key, upcoming_is_home, matchup_features, weight, player_context=None) at props.py:93 (called props.py:1200); player_context={'base_projection','expected_exposure','batting_order'} built props.py:1192-1198; final avg_stat=base_proj*combined_mult at props.py:1254; distributional binomial _distributional_over_rate(prop_key, line, values, at_bats, weights, rate_mult, exposure_mult, player_name, commence_iso, cfg, xstats_strength, teams=None) at props.py:471 — the seam where a per-batter hit distribution would be reconciled to a team-hits/GameContext total. NOTE: analyze_player_props_value loops players independently (no aggregation seam exists; one must be ADDED, likely in app.py's props block app.py:2714-2751 or a new pre-pass over confirmed_lineup).
- FULL-LINEUP source for bottom-up sum: mlb_starters.get_confirmed_lineup(home_team, away_team, date) at mlb_starters.py:541 and lineup_player_context at mlb_starters.py:603 (fetched app.py:2624) — the only source of the complete ordered batter set; slot exposure via load_lineup_adjustment/_lineup_exposure_mult (props.py:165).
- CALIBRATABLE WEIGHTS gate: pricing_common._starter_adjustment(sport_key, key, prop_key=None) at pricing_common.py:136 — keys 'moneyline','spreads','run_scale','bullpen','props'(per-prop); FAILS CLOSED to 0.0. Any GameContext-driven adjustment must be gated by a fitted weight here or it won't turn on in prod.

## RISKS
- LEAKAGE: MLB matchup_features are NOT as-of-safe except get_expected_runs_team_factors (which cuts at as_of-1 day, mlb_starters.py:203). get_pitcher_quality/get_team_offense_splits/get_team_bullpen_quality pull SEASON-level stats (season-final when replayed historically), so in backtest they leak future info. The NFL path IS leakage-safe (nfl_epa as_of_date). A GameContext built on run_suppression/opp_offense_vs_hand/bullpen inherits this leak; only the expected_runs xwOBA factors are clean. Build the GameContext run means on the as-of factor path to keep backtest ROI/Brier honest.
- STDLIB-ONLY: no numpy/scipy/pandas. The existing analytic joint (poisson/negbin over 0..30, mlb_starters.py:410-424) is pure Python nested loops (O(max_runs^2) per game). A per-game GameContext that convolves team-run then team-hit then per-batter distributions must stay analytic/moment-matched and bounded — no per-Streamlit-rerun Monte-Carlo (explicit 3.1 constraint).
- PER-RERUN COST + NO GAME-LEVEL CACHE: build_matchup_features is called once PER EVENT PER RERUN in app.py (app.py:2619) and is NOT wrapped in st.cache_data (only ESPN/odds fetch helpers are @st.cache_data, app.py:713-882). Sub-calls hit the file cache (deploy/cache, _read_cache/_write_cache, CACHE_MAX_AGE=3600) but the dict assembly + tanh + any new joint-distribution math re-runs every rerun. A GameContext with an analytic joint pmf per game must be memoized (per home/away/date, or via the file cache) to avoid recomputing the full distribution on every Streamlit interaction.
- CACHE GRANULARITY MISMATCH: existing file cache is per-team-season / per-pitcher-season / per-date (probables, expected_runs by season+cutoff), NOT per-game. A GameContext is inherently per-game (home,away,date) — a new cache key/granularity is needed; watch TTLs (probables 1h, offense/pitcher 1h, bullpen/savant/expected_runs 24h, team_index 7d) so a stale sub-input doesn't silently freeze a game context.
- FAIL-OPEN / DEGRADE CONTRACT: every consumer treats matchup_features and its sub-keys as optionally None and degrades to the team-only/empirical model (analysis.py:409, props.py:107,116,127; _starter_adjustment fails closed to 0.0). A GameContext must preserve None-tolerance end-to-end and must not turn on any adjustment unless a fitted weight exists in load_starter_adjustment/expected_runs_challenger.
- THREE UNRECONCILED RUN ESTIMATES today (ESPN totals projected_total vs Savant per-team expected_runs vs per-batter xBA mult). If a GameContext replaces one but not the others, the displayed projected_total, spread margin, and prop projections can move in opposite directions — the totals code already warns about this coupling (analysis.py:426-429). 3.1 must decide which becomes the single source and keep display/probability consistent.
- NO EXISTING BATTER AGGREGATION: props are computed player-by-player with no team-hit sum and only for players who have book lines (not the full 9). Bottom-up reconciliation (sum batter hits <-> team hits <-> team runs) requires a NEW aggregation pass and the full confirmed lineup (get_confirmed_lineup), plus handling of batters without book lines. This is net-new coupling between the props loop and the team-run layer that does not exist anywhere today.
- NAME/ID MATCHING FRAGILITY: team resolution via _match_team_id substring/nickname fallback (mlb_starters.py:495) and Savant->StatsAPI abbr mapping with explicit coverage-gap warnings (mlb_starters.py:269-281). A GameContext keyed by team must survive these gaps (fail-open per team) or it silently disables for renamed/unmapped teams.
- MLB HAS NO PACE INPUT: get_team_pace_factor (espn_client.py:210) is an ESPN NBA/NHL possessions proxy; MLB returns nothing useful. A GameContext 'pace'/PA-per-team term is net-new — the only per-team volume proxies today are expected_exposure (weighted recent AB/PA, props.py:1187) and avg_ip->starter_share (props.py:135). Do not assume get_team_pace_factor for MLB.
