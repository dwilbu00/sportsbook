# Workstream 5 — Reversible `TRAIN_SEASON_MIN` config floor (read-side date filter, NO deletes)

**Date:** 2026-08-07
**Status:** DESIGN ONLY (read-only investigation; no code/DB/sweep/vendor changes made).
**Goal:** Add a single, reversible, default-OFF training-season floor that a 2026-basis
"reset" can lean on, implemented purely as read-side date filters (WHERE-clause / list
comprehension). No row is ever deleted. Default (unset) == today's behavior, byte-for-byte.

This is Step 5 of the synthesis plan (`00-SYNTHESIS.md` §3.5). It establishes the lever the
owner asked for **without** the destructive reset that every investigator + verdict
contraindicates. Because the durable corpus is already ~100% MLB-2026, the floor excludes
~0 rows today (at most the 11 stray 2025 `mlb_pitcher_gamelog` rows, which never reach the
prop calibration path). Its real value is *forward-proofing* (auto-excludes 2027+ or any
backfilled pre-2026 data) and *intent documentation*.

---

## 0. Verified code map (all line numbers re-checked against current source 2026-08-07)

Every hook the task names, confirmed by reading the file this session:

| Stage | File:line (verified) | Season-scoped today? | Hook action |
|---|---|---|---|
| DB tier — prop lines | `db_store.py:1104` sig; `date_from`/`date_to` WHERE at `:1137-1140` | supports it, **never fed** | none (already supports `date_from`) |
| DB tier — team lines (sibling) | `db_store.py:1037` sig; WHERE `:1069-1072` | supports it | none (reference impl; test at `test_db_store.py:782`) |
| Warehouse assembler | `warehouse.py:950` `load_prop_lines(sport_key, dates=None)`; calls `_db.player_prop_lines(sport_key, dates=dates)` `:964` | **NO `date_from` kwarg** | **Edit 2**: add `date_from`/`date_to` passthrough |
| Harvest union | `book_line_calibration.py:258` `harvest_real_line_book_lines`; warehouse call `:282`, store `:285`, pred-log `:286` | **NO** | **Edit 3**: resolve floor internally, push to SQL + post-filter all 3 sources |
| Harvest — local JSON store | `book_line_calibration.py:142` `harvest_book_lines_from_store` (rows carry `game_date` `:173`) | **NO** | covered by Edit 3 post-filter (no change to this fn) |
| Harvest — pred-log backstop | `book_line_calibration.py:189` `harvest_book_lines_from_prediction_log`; reads `recalibration._read_log(where={resolved:True})` `:200`; sets `game_date` `:217/221` | **NO** | covered by Edit 3 post-filter (no change to this fn) |
| Gamelog pull / prior-games | `book_line_calibration.py:373-375` `cached_gamelog(... no season_year)`; `prior_games = gamelog[idx+1:]` `:450`; `<10` gate `:451` | de-facto current season | **Edit 4**: floor-trim `prior_games` (mirrors synthetic sweep's `cross_season=strict`) |
| Statcast as-of window | `refit_calibration.py:859-866` `years = {game_date[:4] for enriched}` → `sh.load_days(f"{y}-03-01", f"{y}-11-30")` | **auto-follows enriched obs years** | **none** — transitively floored once obs are floored (Edit 3). Verified. |
| Online-Platt loop read | `recalibration.py:2154` `rows=_read_log()`; resolved filter loop `:2159-2168`; append `:2178` | **NO date floor** | **Edit 6**: skip resolved rows below floor |
| Online-Platt seed | `recalibration.py:2330` `seed_from_book_line_cache` → `harvest_real_line_book_lines(...)` | **NO** | covered by Edit 3 (transitive) |

**All ~8 `harvest_real_line_book_lines` call sites** (`refit_calibration.py:802,1098,1307,1433,1654,1873,2184`
[= `refit_sport_real_lines` + `diagnose_distributional/negbin/center/roi/features/conditional`]
and `recalibration.py:2330` [seed]) inherit the floor from a **single** internal resolve in
Edit 3 — zero call-site edits. That is deliberate: flooring the no-write diagnose lenses too
means Step 6 (re-adjudicate gates via `diagnose_*`) measures exactly what a floored refit would fit.

Config plumbing: `config.json` is loaded independently by several modules (`app.py:880`,
backfill scripts). `pricing_common.py` is the leaf-safe shared layer (imports only
`calibration_loader` + `odds_client`; imports **none** of recalibration/book_line/refit/warehouse/props —
verified) and is **already imported by `book_line_calibration.py:42`**. It is the correct home
for the knob helper. `recalibration.py` does not import it today → use a lazy import (matches
that module's existing cycle-avoidance style).

Env-var precedent: `espn_cache.py:152` reads `ODI_GAMELOG_TTL_HOURS`. Mirror it with
`ODI_TRAIN_SEASON_MIN` for per-run offline override without editing committed config.

No prior `train_season_min` / `TRAIN_SEASON_MIN` / `ODI_TRAIN_SEASON` usage exists anywhere
(grep clean) — clean slate.

---

## 1. The single knob

One value, an **inclusive season-year floor** (int, e.g. `2026`), default **None = OFF (no filter)**.

Resolution order (first hit wins):
1. Env `ODI_TRAIN_SEASON_MIN` (offline per-run override; mirrors `ODI_GAMELOG_TTL_HOURS`).
2. `config.json` key `"train_season_min"`.
3. `None` → no filter (today's behavior).

A season year (not an arbitrary date) is chosen because: it matches `--season` / `fit_season`
semantics; a `YYYY-01-01` `date_from` derived from it never lands on a real MLB game (season
starts in March), so the UTC-stored-`game_date` vs ET-derived-`game_date` boundary fuzz in the
warehouse path (`warehouse.load_prop_lines` derives ET `game_date` *after* the SQL query at
`:980`) is a non-issue — Jan-1 has no games either side. String comparison of 4-digit year
prefixes (`game_date[:4] >= "2026"`) is lexicographically correct and matches existing patterns
(`recalibration.py:1156,1600` compare `game_date >= today` as strings).

### Edit 1 — `pricing_common.py` (new helper, best-effort, uncached)

```python
import json, os  # add if not already imported at module top

def train_season_min():
    """Optional INCLUSIVE training-season floor (year int) or None (default OFF).

    Read-side only: threads into the real-line harvest, prior-game trim and the
    online-Platt resolved-row read so a refit/fit can be pinned to >= this season
    WITHOUT deleting any data. Env ODI_TRAIN_SEASON_MIN overrides config.json
    'train_season_min'. Best-effort: any parse / IO error -> None (no filter)."""
    raw = os.environ.get("ODI_TRAIN_SEASON_MIN")
    if raw is None:
        try:
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "config.json")
            with open(cfg_path) as f:
                raw = json.load(f).get("train_season_min")
        except Exception:
            raw = None
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None

def train_date_from():
    """`YYYY-01-01` for the season floor, or None. Feeds the SQL date_from filter."""
    y = train_season_min()
    return f"{y:04d}-01-01" if y else None
```

Uncached is deliberate: these run only in refit / harvest / seed / gated-online-refit paths
(never the per-prediction hot path), so a cheap best-effort file read per refit is fine and keeps
tests deterministic across env changes.

### Documentation (optional, recommended)
Add `"train_season_min": null` to `config.json.example` (git-tracked). `config.json` itself is
git-ignored — the owner sets the value locally / in deploy env.

---

## 2. Hook-by-hook design

### Edit 2 — `warehouse.py:950` `load_prop_lines`: forward `date_from`/`date_to`
```python
def load_prop_lines(sport_key, dates=None, date_from=None, date_to=None):
    ...
    rows = _db.player_prop_lines(sport_key, dates=dates,
                                 date_from=date_from, date_to=date_to)
```
`db_store.player_prop_lines` already honors `date_from`/`date_to` (`:1104,:1137-1140`) — this is
pure passthrough. Backward compatible (new kwargs default None). Pushes the filter to SQL so the
warehouse path pulls fewer rows (efficiency + intent), even though the corpus is 2026-only today.

### Edit 3 — `book_line_calibration.py:258` `harvest_real_line_book_lines`: resolve + apply floor (the 3 harvest hooks, one place)
- Extend the existing import: `from pricing_common import et_local_date, train_date_from`
  (currently `:42` imports only `et_local_date`).
- At the top of the function: `date_from = train_date_from()`.
- Warehouse primary (`:282`): `warehouse.load_prop_lines(sport_key, date_from=date_from)`.
- After `primary` and `pred_lines` are assigned (`:282-286`), **before** the dedup loop, floor
  both source lists (so the printed `n_primary`/`n_pred` reflect the floored basis):
```python
if date_from:
    primary    = [r for r in primary    if (r.get("game_date") or "") >= date_from]
    pred_lines = [r for r in pred_lines if (r.get("game_date") or "") >= date_from]
```
This single post-filter covers the **local-JSON store** and **pred-log backstop** hooks (both emit
`game_date`), and is redundant-but-harmless over the already-SQL-floored warehouse rows. The
`harvest_book_lines_from_store` (`:142`) and `harvest_book_lines_from_prediction_log` (`:189`)
functions themselves are **unchanged**.
- Because `seed_from_book_line_cache` (`recalibration.py:2330`) and all six `diagnose_*` lenses call
  this function, they are floored transitively — no edits there.

### Edit 4 — `book_line_calibration.py:339` `join_book_lines_to_actuals`: floor-trim `prior_games` (gamelog hook)
- At the function top (once): `season_min = train_season_min()`.
- At `:450`, before the `<10` gate:
```python
prior_games = gamelog[idx + 1:]
if season_min:
    prior_games = [g for g in prior_games
                   if (g.get("game_date") or "")[:4] >= str(season_min)]
if len(prior_games) < 10:
    ...
```
Rationale: the real-line path currently does **not** season-trim `prior_games` (unlike the
synthetic sweep, which enforces `cross_season="strict"` via `_filter_to_current_season`,
`backtest.py:2494`). When the floor is ON this makes the real-line projection basis consistent
with the synthetic sweep (both season-strict); when OFF (None) it is a strict no-op. Placed before
the `<10` gate so the eligibility count reflects the floored basis. Default None → byte-identical.

### Edit 5 — Statcast as-of window: NO CODE CHANGE (verified transitive)
`refit_calibration.py:859-866` derives `years` from the *enriched* obs' `game_date[:4]` and loads
`sh.load_days` per year. Once obs are floored (Edit 3 + Edit 4), `years` cannot contain a
below-floor year, so the raw-Statcast load auto-spans only floored seasons. `statcast_asof` is
season-bucketed independently. Confirmed by reading the block — no separate knob needed. (An
explicit `if not season_min or int(y) >= season_min` guard is *possible* but omitted to keep the
change minimal; it would be pure redundancy.)

### Edit 6 — `recalibration.py:2127` `refit_sport`: floor the online-Platt resolved-row read
- Lazily resolve inside the function (before the read at `:2154`):
  `from pricing_common import train_season_min` → `season_min = train_season_min()`.
- In the row loop (`:2159-2168`), immediately after the `if not r.get("resolved"): continue` guard:
```python
if season_min and (r.get("game_date") or "")[:4] < str(season_min):
    continue
```
Default None → no filter. Note: this function runs in the **live app** via
`maybe_auto_refit → maintain_sport → refit_sport` (`:2278,:2240,:2268`), so setting the floor
intentionally also pins live Platt fitting to `>= floor` — that is the desired uniform semantics,
and fully reversible by unsetting. The refit-trigger gate `_count_resolved_since` (`:2219`) is
**left untouched** (it is a cadence trigger, not a training read; flooring it is moot on a
2026-only corpus and would complicate the trigger for no benefit). The seed path
(`:2330`) is already floored via Edit 3.

---

## 3. Reversibility & leakage analysis

- **Reversible:** every hook is a WHERE-clause or list-comprehension read filter keyed off one
  value. Lower it, or unset env + config → the full corpus reappears on the next run. Nothing is
  archived or restored because nothing is destroyed. (Contrast `sql/clear_tables.sql`, which
  TRUNCATEs everything — explicitly NOT used here.)
- **Leakage-safe:** the floor only ever *removes* older-than-floor rows. It cannot introduce future
  data and cannot make the backtest look better (it shrinks, never shifts forward, the training
  set). The as-of primitives (`02-leakage-asof-correctness.md`) are untouched. Edit 4 makes the
  real-line prior-game basis *more* strict (season-trimmed), matching the synthetic sweep — the
  safe direction.
- **Behavior-preserving by default:** with `train_season_min()==None`, Edits 2/3/4/6 short-circuit
  (`date_from`/`season_min` falsy) → byte-identical to today; Edit 5 is a no-op by construction.
  On the current 2026-only corpus, even `train_season_min=2026` excludes 0 rows on the prop paths.

---

## 4. Tests

### Existing tests that PIN this area (must stay green, unchanged)
- `test_realline_calibration.py:467` (`test_prediction_log_harvest_filters`), `:475`
  (`test_union_dedups_store_preferred`), `:494` (`WarehouseHarvestTests`, patches
  `warehouse.load_prop_lines` — a MagicMock swallows the new `date_from` kwarg, so unaffected).
- `test_db_store.py:883-953` (`player_prop_lines` + `load_prop_lines` shape/closing-pick/ET-date).
- `test_db_store.py:782` (`test_excludes_props_and_filters_dates`) — the reference `date_from`/
  `date_to` test on the sibling `team_market_lines`; mirror it for `player_prop_lines`.
- `test_prediction_log.py:168` (patches both `harvest_book_lines_from_*`).
- `test_feature_diag.py`, `test_distributional.py`, `test_negbin.py`, `test_roi_diag.py` (all patch
  `harvest_real_line_book_lines` wholesale → the internal floor resolve is bypassed → unaffected).
- `test_recalibration_durability.py` (online-Platt refit loop durability).

### New tests to ADD
1. **`test_pricing_common` (new or existing suite):** `train_season_min` / `train_date_from` —
   env `ODI_TRAIN_SEASON_MIN` override; config.json fallback; None default; junk value (`"abc"`)
   → None; `train_date_from()` == `"2026-01-01"` when floor 2026, None when unset.
2. **`test_db_store.py`:** `player_prop_lines(date_from=..., date_to=...)` range filter, mirroring
   the `team_market_lines` test at `:782` (in-range returns rows; out-of-range returns `[]`).
3. **`test_realline_calibration.py`:** with the floor set (patch `pricing_common.train_season_min`
   → 2026, or set env), feed store+pred rows spanning 2025 and 2026; assert only `>=2026-01-01`
   rows survive and `warehouse.load_prop_lines` was called with `date_from="2026-01-01"`. Plus an
   explicit "floor None → identical to un-floored output" no-op assertion.
4. **`test_realline_calibration.py` (or `book_line`):** `join_book_lines_to_actuals` prior-game
   trim — a gamelog spanning 2025+2026 with floor 2026 drops the 2025 prior games (and may trip
   the `<10` gate); floor None leaves `prior_games` unchanged.
5. **`test_recalibration_durability.py` (or new):** `recalibration.refit_sport` online loop drops
   resolved rows below the floor when set; includes all rows when None.

---

## 5. Risks / open questions
- **Live coupling (Edit 6):** setting the floor changes live online-Platt fitting, not just offline
  refits. Intended (uniform semantics), reversible, and a no-op on today's corpus — but flag it so
  the owner knows the knob is not purely offline.
- **Edit 4 runtime-vs-fit nuance:** production keeps cross-season prior games but *warmup-blends*
  them (`apply_calibration_with_warmup`), whereas Edit 4 *drops* below-floor priors (matching the
  synthetic sweep's `cross_season=strict`, not runtime warmup). This is a pre-existing
  synthetic-vs-runtime asymmetry, not introduced here; Edit 4 only aligns real-line with the
  synthetic sweep. No-op today (2026-only) and when floor=None. If the owner prefers the real-line
  path to keep priors like runtime, Edit 4 can be dropped without affecting the other hooks — the
  floor still filters *obs* via Edit 3.
- **Near-no-op today:** on the current 2026-only corpus the floor removes 0 prop-path rows. Its
  value is forward-proofing + documented intent, not an immediate accuracy change (matches
  `08-pipeline-map-and-reset-design.md` §3 caveat).
- **Local-store path only fires with an explicit `--store-label`** (`harvest_real_line_book_lines`
  prefers the SQL warehouse when `db_store.enabled()` and no label). Edit 3's post-filter still
  covers it for the offline `--store-label` case.

## 6. Rollout
Ship Edits 1-4 + 6 (Edit 5 = no change) with the floor **unset** (default OFF) so the change is a
verified no-op in prod. The owner flips it on later (`config.json` `train_season_min: 2026` or env)
only when a floored refit is actually run under Step 9 of the synthesis plan — at which point it
becomes a live, reversible lever with no data destroyed.
