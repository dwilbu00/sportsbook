# Workstream 9 — batter_strikeouts missing from forward tracking (prediction_log)

**Status:** DESIGN ONLY (read-only investigation complete). Root cause found + fix spec below.
**Date:** 2026-08-07. Repo: `SPORTSBOOK_ODDS/deploy`, branch `main`.
**Owner flag:** "batter_strikeouts ARE analyzed from time to time but ZERO rows in forward tracking; they SHOULD appear."

---

## 0. TL;DR — the surprising root cause

**There is NO drop in the logging / forward-capture code.** The forward-log path
(`props.py:1646-1667` → `recalibration.log_prediction` / `log_prediction_rows`) is
functionally correct and **empirically logs batter_strikeouts when the market is present**
(I proved this with a live in-process call — see §3). There is **no prop allowlist, filter,
or gate that excludes batter_strikeouts from logging.**

**The real cause is upstream data availability: The Odds API never returns
`batter_strikeouts` market outcomes for these MLB events**, so the market never enters
`prop_data["props"]`, so the (correct) logging loop at `props.py:953` never iterates it.
The owner's "I select it and run analysis" is not the same as "the vendor returned lines";
selecting a market that the vendor doesn't serve produces **zero** offers → zero candidates
→ zero logs, silently.

**Decision-grade evidence (read-only prod SQL + raw cache, §2):**
- `odds_snapshot.markets LIKE '%batter_strikeouts%'` → **113 snapshots requested it**
  (2026-07-29 … 2026-08-08).
- `odds_line` for those 113 snapshots: `batter_hits` 3876, `pitcher_earned_runs` 390,
  `pitcher_outs` 386, `pitcher_strikeouts` 378, **`batter_strikeouts` 0**. 112/113 snapshots
  produced ≥1 line — so parsing worked; batter_strikeouts was simply absent from the response.
- Raw odds JSON disk cache (`cache/*.json`): **101 files contain `"key":"batter_hits"`,
  ZERO files mention `batter_strikeouts` anywhere** — not even as a bookmaker market.
- The absence is **market-specific, not positional truncation**: snapshots with markets like
  `...,pitcher_outs,batter_strikeouts,pitcher_earned_runs` returned `pitcher_earned_runs`
  (which sits *after* batter_strikeouts) but not batter_strikeouts.
- `prediction_log` GROUP BY prop_key (baseball_mlb): batter_hits 3267, pitcher_strikeouts 253,
  pitcher_earned_runs 223, pitcher_outs 221, **batter_strikeouts 0.**

**Conclusion:** to "get analyzed batter_strikeouts logged" the fix must be at the
**request/vendor layer** (make the API return the market), OR the market should be dropped
as dead weight (audit's "candidate to drop"). No change to the logging code is required or
would help. Recommended: a small, reversible request-side fix + a UI guardrail so a
selected-but-empty market can never again masquerade as "analyzed."

---

## 1. Full path trace (every hop verified against current line numbers)

Prediction/logging is 100% driven by what the owner analyzes interactively (the
"forward-capture timer" in `app.py:2305` just re-runs the same user-selected slate; it has
no independent prop list). The chain:

1. **Market menu** — `app.py:793` `PLAYER_PROPS_BY_SPORT["baseball_mlb"]` **includes
   `batter_strikeouts`** (line 796). Multiselect at `app.py:2216-2223` (no `default=`, so the
   user must pick each session). → batter_strikeouts IS selectable. ✓
2. **Request** — `app.py:2722-2726` `get_event_odds(..., markets=",".join(selected_props),
   bookmakers=None)`. `get_event_odds` (`odds_client.py:201`) uses **`regions="us"` default**
   (app passes no `regions`), builds params at `:227-235`, and returns `resp.json()` at `:269`.
   The **same markets string** is archived by `warehouse.capture_event_odds` (`:276`) → that's
   why `odds_snapshot.markets` shows batter_strikeouts was requested. ✓
3. **Parse** — `odds_client.parse_player_props` (`:1007`). `batter_strikeouts` is in
   `PROP_LABELS` (`app.py:834`; also `odds_client.PROP_LABELS`), so the market-key filter at
   `:1034` (`if market_key not in PROP_LABELS: continue`) does **not** drop it. The two-sided
   requirement at `:1060-1062` (`if "Over" not in sides or "Under" not in sides: continue`) is
   never even reached because **the response carries no batter_strikeouts outcomes at all** →
   `result["props"]` gets no `batter_strikeouts` key. ✓ (No parse-level bug specific to it.)
4. **History** — `app.py:2974-2995` builds `player_histories` by iterating
   `prop_data["props"]`; since there's no batter_strikeouts key, no history is even requested.
   (Note: history *would* succeed if asked — `espn_client.PROP_STAT_MAP["batter_strikeouts"]
   = ["K","SO"]` at `espn_client.py:962`, and the batter gamelog stores `SO`
   (`gamelog_store.py:89` `_BATTER_STATS = ("AB","H","SO",...)`), so `get_player_stat_history`
   falls back K→SO and returns `found=True` — verified live in §3.)
5. **Analyze** — `props.analyze_player_props_value` (`props.py:862`) loops
   `for prop_key, players in prop_data.get("props", {}).items()` at `:953`. With no
   batter_strikeouts key, the loop never processes it.
6. **Log (the alleged "drop point")** — inside that loop, `props.py:1646-1667`:
   `if log_game_date and sport_key and not lineup_out:` → `log_prediction(..., write=False)`
   → appended to `prediction_rows`; then `log_prediction_rows(prediction_rows)` at
   `props.py:1716`. `recalibration.log_prediction` is at `:537` (range guard `0<=raw_prob<=1`
   at `:564`), `log_prediction_rows` at `:623`. **This path has no prop-key filter and works
   for batter_strikeouts** — it is simply never reached.

### Recommendation-side gates (correctly NOT logging gates — ruled out)
- `SUPPRESS_UNDER_MAX_LINE={batter_hits:0.5}` (`props.py:379`) — batter_hits only; only sets
  `is_value=False`, never skips the log.
- `lineup_adjustment.props.batter_strikeouts=0.0` — feeds `_mlb_lineup_exposure_mult`
  (`props.py:183/1204`); scales the projection only, never skips the log.
- `player_start_status(prop_key,...)` (`mlb_starters.py:861`) — only `pitcher_`-prefixed keys
  take the pitcher branch (`:886`); batter_strikeouts is correctly a **batter** prop gated on
  the confirmed lineup, identical to batter_hits. `lineup_out` (`props.py:1638-1641`) fires
  only on a confident "out" for a benched batter — same as batter_hits, not a differentiator.
- Safe/Alt-lines mode (`props.py:1370`) `continue`s before logging for **all** props (so it
  can't explain batter_hits=3267 vs batter_strikeouts=0). Also
  `PLAYER_PROP_ALTS_BY_SPORT["baseball_mlb"]` (`app.py:812-815`) has no batter_strikeouts alt.
- Calibration exists: `calibration/baseball_mlb.json` props.batter_strikeouts = method A,
  `opp_defense_strength=1.0` — so a returned line would be calibrated and logged.

---

## 2. Read-only evidence gathered (reproducible; all COUNT/GROUP BY only, no writes)

Run from `deploy/` after `db_store.promote_secrets_from_toml()`:

```
prediction_log  GROUP BY prop_key (baseball_mlb):
  batter_hits 3267 | pitcher_strikeouts 253 | pitcher_earned_runs 223 | pitcher_outs 221 | batter_strikeouts 0

odds_line GROUP BY prop_key (bet_type='player_prop'):
  batter_hits 11364 | pitcher_earned_runs 992 | pitcher_strikeouts 952 | pitcher_outs 556 | (batter_strikeouts absent)

odds_snapshot WHERE markets LIKE '%batter_strikeouts%'  →  113 (2026-07-29 … 2026-08-08)
  of those, snapshots with >=1 line: 112
  their odds_line prop_keys: batter_hits 3876 | pitcher_earned_runs 390 | pitcher_outs 386 | pitcher_strikeouts 378 | batter_strikeouts 0

cache/*.json: grep -l '"key":"batter_hits"' → 101 files ;  grep -l 'batter_strikeouts' → 0 files
```

Interpretation: requested broadly for ~11 days, every other requested market returned lines,
`batter_strikeouts` returned nothing every time. This is the vendor silently ignoring an
un-served market in a multi-market event-odds request (the request does not hard-error — a
`markets=('batter_strikeouts',)`-only snapshot row exists, i.e. a response came back empty of
that market).

---

## 3. Empirical proof the logging path is NOT the drop

In-process (read-only; patched `props.log_prediction_rows` to capture instead of write), fed a
synthetic one-event slate with a REAL batter's history (`get_player_stat_history` SQL path):

- `batter_strikeouts` for a real batter: `found=True, stat_label='SO', values=[1,0,1,0,0,...]`.
- `analyze_player_props_value(prop_data{batter_strikeouts}, ...)` → **1 candidate, 1 logged
  row** (`prop_key=batter_strikeouts, line=0.5, raw_prob=0.2, direction=UNDER`), identical
  shape to the batter_hits control.

→ When the market is present, batter_strikeouts is captured for forward tracking exactly like
the other props. The zero-row fact is entirely due to the market being absent upstream.

---

## 4. Fix spec

Because the capture code is correct, "so analyzed batter_strikeouts get logged" can only be
achieved by making the vendor return the market (Path A) or, if it is genuinely unobtainable,
retiring the dead market honestly (Path B). Plus a guardrail (Path C) that should ship
regardless, so a selected-but-empty market can never silently look "analyzed" again.

### Prerequisite (owner action, NOT me — vendor call is out of scope for this design pass)
One-off diagnostic: request the MLB event-odds endpoint for a single live event with
`markets=batter_strikeouts` alone (and separately with `regions=us,us2` and `regions=us_dfs`).
Confirm whether The Odds API serves `batter_strikeouts` for MLB and in which region/books.
This decides Path A vs Path B. (Empirically, the `us` region + all-US-books it currently uses
returns nothing.)

### Path A — make the vendor return it, then the existing path logs it automatically (PREFERRED if available)
Smallest reversible change: **thread a `regions` parameter into the MLB props fetch** so
batter_strikeouts (likely a DFS-style / secondary-region market) is included.
- `get_event_odds` already accepts `regions` (`odds_client.py:201`), and the cache key already
  includes it (`:220`) — no signature change needed there.
- Edit the props fetch call at `app.py:2724-2726` to pass e.g. `regions="us,us2"` (or a
  config-driven value). Keep team-market fetches on `us` unless separately validated.
- **Cost caveat (must surface to owner):** Odds API cost = (#markets × #regions). Adding a
  region multiplies props credit cost. Gate behind a config flag (e.g.
  `config.json` `"prop_regions": "us"` default) so it is opt-in and reversible by config, not
  code. Update `app.py`'s cost estimator (`:2400-2406`) to multiply by region count so the
  credit preview stays honest.
- **No change to parse / analyze / log** — once outcomes are present, `parse_player_props`
  (two-sided), `analyze_player_props_value`, and `log_prediction_rows` handle it unchanged
  (proven §3). The warehouse (`_emit_prop_lines`, `warehouse.py:284`) also captures it for
  free (same iteration over `parsed["props"]`).
- Leakage: none — this is a request-side / display-side change; no training data, no as-of
  primitives touched.
- Reversible: flip the config flag / revert the one call-site edit.

### Path B — retire the dead market (if Path A confirms it's unavailable)
Per audit synthesis §2 ("candidate to drop") and open question ("Is batter_strikeouts still an
offered market?"):
- Remove `"batter_strikeouts"` from `PLAYER_PROPS_BY_SPORT["baseball_mlb"]` (`app.py:796`) so
  it is no longer an empty, credit-wasting selectable market.
- Leave the `calibration/baseball_mlb.json` props.batter_strikeouts block in place (do NOT
  delete — `save_calibration` preserves it; harmless dormant), OR annotate it with a
  `"_note"` like the existing `lineup_adjustment.props` disabled note. Deleting risks the
  hysteresis/no-recovery issues flagged elsewhere in the audit; annotation is safer.
- Reversible: re-add the one list entry.
- This is the smallest change but explicitly does NOT satisfy "get them logged" — choose only
  if the vendor truly doesn't serve the market.

### Path C — guardrail (SHIP REGARDLESS; independent of A/B)
Root of the owner's confusion: a selected market that returns zero offers produces zero rows
with **no user-visible signal**. Add a non-fatal notice.
- After parsing in `app.py` (`:2806-2810` builds `parsed_props`), compute the set of selected
  prop keys that produced **no** entry across all analyzed events
  (`selected_props` minus the union of `parsed.get("props").keys()`), and surface a
  `st.caption`/`warnings.append` like: "No lines returned for: Batter Strikeouts (books did
  not post this market)." Reuses the existing `warnings` list already threaded through the UI.
- Purely display-side; no leakage; trivially reversible.

---

## 5. Tests

### Existing tests that pin this area (must stay green)
- `test_prediction_log.py` — `log_prediction` / `log_prediction_rows` upsert + identity
  (`:47-144`). Pins the writer.
- `test_modeling.py` — batter_strikeouts projection weighting + `player_start_status` batter
  branch (`:70-90`, `:206-263`); `starter_adjustment`/`lineup_adjustment` prop-weight
  independence (`:42-60`, `:123-147`). Pins that batter_strikeouts is a first-class batter prop.
- `test_odds_source.py` — odds parsing.
- `test_warehouse.py` — snapshot/line capture.
- `test_market_prediction_log.py` — team-market forward log (parallel structure).

### New tests to add
1. **Regression lock on the capture path (Path independence — add now):** in
   `test_modeling.py` or `test_prediction_log.py`, feed `analyze_player_props_value` a
   `prop_data` whose `props` includes `batter_strikeouts` with a two-sided offer + a found
   history, patch `props.log_prediction_rows` to capture, assert exactly one row with
   `prop_key="batter_strikeouts"` is produced (mirrors §3). This pins "no future refactor may
   add a logging-side prop filter that silently excludes batter_strikeouts."
2. **Parse coverage (Path A/general):** in `test_odds_source.py`, a `parse_player_props`
   fixture with a two-sided `batter_strikeouts` market → asserts it appears in
   `result["props"]["batter_strikeouts"]` (guards the `PROP_LABELS` gate at
   `odds_client.py:1034`).
3. **regions threading (only if Path A):** assert `get_event_odds` forwards a non-default
   `regions` into params + cache key (extend `test_odds_source.py`).
4. **Empty-market guardrail (Path C):** unit test the "selected minus parsed" diff helper
   returns the correct missing-label set.

---

## 6. Reversibility, leakage, and open questions

- **Reversibility:** Path A = config flag + one call-site arg; Path B = one list entry;
  Path C = display-only. All git-revertible; none touch calibration JSON or training data.
- **Leakage:** none in any path (request-side / display-side only).
- **Open questions (require a vendor call, out of scope here):**
  1. Does The Odds API serve `batter_strikeouts` for MLB at all, and in which region
     (`us` vs `us2` vs `us_dfs`) / from which books? Decides Path A vs B. Current `us` region
     empirically returns nothing across 113 requests / 11 days.
  2. If served only via an extra region, is the added props credit cost acceptable to the
     owner (cost = markets × regions)?
  3. Is capturing batter_strikeouts actually wanted (owner does not bet it), or is Path B
     (retire) the better call? The audit already nominated it to drop.

## 7. The exact "drop point" (as demanded)
There is **no logging-code drop**. The earliest point at which batter_strikeouts disappears is
the **vendor response consumed at `odds_client.py:269` (`data = resp.json()`)**, which contains
no batter_strikeouts outcomes → `parse_player_props` (`odds_client.py:1007`, market-key gate
`:1034` passed but zero outcomes) yields no `batter_strikeouts` key → the analysis loop
`props.py:953` never iterates it → the fully-functional forward-log call at
`props.py:1646-1667` / `props.py:1716` is never reached. Fix at the request/region layer
(`app.py:2724-2726` + `odds_client.get_event_odds` `regions`), not the logging layer.
