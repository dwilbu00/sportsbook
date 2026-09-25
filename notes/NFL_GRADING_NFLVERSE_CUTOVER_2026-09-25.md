# NFL prop grading → nflverse cutover (SCOPE)

**Date:** 2026-09-25 · **Status:** scoped, not built · **Owner:** Doug

## Why

NFL **player-prop** grading is the last path still on ESPN per-player gamelogs
(`recalibration._load_player_gamelog` → `espn_cache.cached_athlete_id/cached_gamelog`).
That path is:
- the per-player, network-bound loop we just had to memoize + cap (the 113s→0.1s fix), and
- the source of the two recent grading bugs — the "only YDS+TD cached" gap (`61a7510`) and
  name→athlete-id ambiguity.

**Neither bug is possible against nflverse**: the weekly table carries every stat column,
keyed by a hard `player_id` (gsis) with a clean `player_display_name` — no per-player fetch,
no label collisions, no name→id search. MLB already made this move (statsapi medallion; "ESPN
fully removed for MLB"). NFL is the holdout; this mirrors that decision.

## The key finding: the hard part already exists

The live, dep-free, credit-free nflverse fetch stack is **already in production** (serves the
prop models + the bonus eligibility gate on Streamlit Cloud):

- **Player box scores** — `nfl_opportunity_serving._load(season)`: one `pd.read_parquet` of
  `https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet`
  (bulk = ALL players/weeks), 6h TTL cache, self-healing, serves stale on transient failure,
  normalizes `player_display_name` → `player_norm` via `nfl_props_scan._norm`. **One fetch grades
  the whole slate** — replaces the entire per-player ESPN loop.
- **Week mapping + (optional) scores** — `nfl_schedule.fetch_games()` pulls nflverse `games.csv`
  live (raw.githubusercontent, cached); `resolve_event(home, away, commence)` → `game_id` =
  `{season}_{wk}_{AWAY}_{HOME}`; `team_scores_index` gives final scores.

So grading just needs to (1) find `(season, week)` for the graded game and (2) read the actual
stat column for `(player_norm, week)`. All cloud-safe, all already fetched.

## Changes

1. **`nfl_opportunity_serving._load`** — add `rushing_tds`, `receiving_tds`, `special_teams_tds`
   to `_KEEP` + `_NUM_COLS` (today only `passing_tds` is kept). Already guarded
   (`[c for c in _KEEP if c in df.columns]`, missing numerics → 0.0), so it's safe for older
   seasons and harmless to the serving/gate callers that ignore the new cols.

2. **`_resolve_nfl_actual(prop_key, player, game_date, commence, home_team, away_team)`** (new;
   in `recalibration.py` or a small `nfl_grading.py`):
   - `(season, week)` from `nfl_schedule.resolve_event(home, away, commence)` → split `game_id`;
     fallback to a `game_date → week` lookup built from `games.csv` (gameday column) if the
     prediction row lacks team names.
   - `df = nfl_opportunity_serving._load(season)`; `norm = nfl_props_scan._norm(player)`.
   - `sub = df[(df.player_norm == norm) & (df.week == int(week))]`; **empty → return None**
     (pending — either a DNP or nflverse hasn't posted the week yet).
   - map `prop_key` → column (below); return `float(actual)`.

3. **prop_key → nflverse column** (all present in `nfl_ingest._PLAYER_WEEK_COLS`):

   | prop_key | nflverse column |
   |---|---|
   | player_pass_yds | passing_yards |
   | player_rush_yds | rushing_yards |
   | player_reception_yds | receiving_yards |
   | player_receptions | receptions |
   | player_pass_attempts | attempts |
   | player_rush_attempts | carries |
   | player_pass_completions | completions |
   | player_pass_tds | passing_tds |
   | player_anytime_td | rushing_tds + receiving_tds + special_teams_tds |

   Note: anytime-TD gains `special_teams_tds` (return TDs) vs. the current ESPN rush+rec sum —
   DK/FD "anytime TD scorer" DOES pay return TDs, so this is a small **correctness** gain, not
   just a source swap. Confirm against a settled return-TD example.

4. **Wire `resolve_one_prop` (NFL branch)** — try `_resolve_nfl_actual` FIRST; fall back to the
   existing ESPN gamelog path only on a nflverse miss (mirrors MLB's warehouse-first pattern).
   Keep ESPN as the belt-and-suspenders fallback through phases 1–3.

5. **Finality** — presence of the `(player_norm, season, week)` row in nflverse **is** official
   final (drop the `_espn_row_final` heuristic on the nflverse path). A true DNP = no row →
   stays pending → the existing `STALE_DNP_HOURS` void still applies.

## Freshness

`stats_player_week` + `games.csv` refresh via nflverse automation hours after games complete
(not days); `_load` caches 6h. Grading is calibration/CLV, not live bet settlement, so a
same-day/next-morning refresh is fine. **Verify the live cadence once** before flipping primary.

## Edge cases (all fail-open to "pending", never a wrong grade)

- **Same-normalized-name in one week** (rare): disambiguate by `team` (player_week has it; the
  prediction row's event gives the two teams). Note + guard; low risk.
- **Playoffs**: game_id weeks 19–22 — `games.csv` covers them.
- **Bye week / inactive**: no row → pending → correct.

## Testing

- **Unit (mock `_load`)**: reads the right column; anytime_td sums all three TD types; returns
  None for missing player/week; week mapping from a stub games index.
- **Parity harness (the confidence gate)**: re-resolve the 165 already-ESPN-graded NFL rows via
  nflverse and assert actuals MATCH (diff report). Owner runs it (fetches nflverse). Only flip
  nflverse-primary after parity is clean — mirrors the MLB parity approach.

## Rollout / sequencing

1. Add TD cols + `_resolve_nfl_actual` + map + unit tests (additive; ESPN still primary).
2. **Owner parity run** → confirm 165/165 match.
3. Flip nflverse-primary; ESPN stays fallback.
4. *(Optional, later)* move NFL **team** markets to `games.csv` scores too → drop ESPN
   (`game_results`) for NFL entirely (mirror MLB's ESPN teardown). Then the per-player ESPN
   pre-warm / `_GAMELOG_MEMO` / `MAX_RESOLVE_PER_LAUNCH` become NFL-moot and can be simplified.

## Effort / risk

Small-to-medium; reuses the whole live fetch/cache/normalize stack. ~1 function + a column map
+ 3 added columns + a week helper + tests + a parity run. Risk low: every failure mode degrades
to "pending", and ESPN remains the fallback until parity proves the swap.
