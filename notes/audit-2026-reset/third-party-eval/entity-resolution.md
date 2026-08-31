# Third-Party Audit Eval — Cluster: Event-level Entity Resolution

**Scope:** Audit **P2** ("Resolve Player Identity at the Event Level") + the entire
`mlb_player_identification_strategy.txt` deep-dive.
**Date:** 2026-08-10  •  **Evaluator:** entity-resolution cluster agent
**Method:** every audit claim treated as a hypothesis; confirmed/refuted against code with file:line.

---

## 0. TL;DR verdict

The SFBB MLBAM-id hybrid migration already satisfies the **core** of the identity
strategy — MLBAM id is identity, team is evidence/tiebreak, normalize+fuzzy is
candidate-gen only, and the id-binder refuses to guess among namesakes. Those items
are genuinely **already-done**, not aspirational.

What is genuinely **NEW and valuable** and NOT done:

1. **Fail-CLOSED at the pipeline level (P2d).** The id-*binder* fails closed (returns
   `None` on a namesake collision), but the *pipeline* fails **open**: an ambiguous
   name falls back to a `name:<norm>` key and STILL produces a prediction/bet, and
   the stat-fetch then falls through to ESPN `search_athlete`, which returns
   `candidates[0]` (ESPN's first result) when its team hint doesn't disambiguate — a
   real wrong-bind tail. The audit's "ambiguous identity → NO model output" is NOT
   implemented. **Highest-value keeper.**
2. **Game/team context is not threaded into the projection's stat-fetch resolvers.**
   The odds-warehouse enrichment passes `teams=(home,away)` to the SFBB resolver, but
   the *projection* stat-fetch calls the authoritative SFBB resolver with **no team
   hint** (batter path `_mlb_espn_id`, pitcher path `find_player_id(name, season)`),
   leaning on ESPN's numeric-team-id tiebreak instead. Cheap, concrete inconsistency.

**Partial / defer:** gamePk *game-roster* candidate universe (approximated today by
the two-team hint; the gamePk machinery exists but is wired only into grading, not
resolution); market/role as a resolver tiebreak (only used to reject wrong-role
gamelogs, not to break a namesake tie).

**Reject as already-done or over-engineering:** event-scoped identity KEY; team-as-
evidence; normalize/fuzzy-as-candidate-gen; a dedicated `provider_alias` table with
validity windows (the static SFBB map + per-row `player_mlb_id` denormalization +
`athlete_id_cache` already cover the realized value; a validity-windowed alias store
is machinery this single-provider, DK-only app doesn't need yet).

No existing workstream (WS1–WS9) covers identity. Keepers → **new WS10 "Event-level
entity-resolution hardening"** (small, reversible, fail-closed-aligned). Provider-
alias table → backlog.

---

## 1. How the app maps an odds-feed name → canonical player (verified)

### 1.1 The identity key — MLBAM-id-first, name-fallback; team NOT in the key
`db_store.player_key(row)` (**db_store.py:108-120**):

```python
def player_key(row):
    mid = row.get("player_mlb_id")
    if mid:
        return f"mlb:{mid}"
    return f"name:{normalize_name(row.get('player') or '')}"
```

`prediction_log` UNIQUE = `(sport_key, event_key, prop_key, player_key, line)`
(**db_store.py:167-169**), where `event_key = event_id or game_date`
(**db_store.py:456**). So the durable identity is **already event-scoped** and keyed
on the hybrid id — team never enters the key. `normalize_name` (**db_store.py:98-105**)
NFKD-folds accents → ASCII, lowercases, keeps alnum+space.

→ **Audit P2a ("lookup identity = odds_event_id + prop_key + normalized name") is
ALREADY-DONE**, in fact stronger (id-first).

### 1.2 The resolver — global name index, team as tiebreak only
`player_id_map.mlb_id_for_name(name, teams=None)` (**player_id_map.py:581-585**) →
`_unique_id` (**player_id_map.py:537-578**):

- Candidate gen = `_rows_for_name(name)` (**506-518**): exact `normalize_name`
  lookup in the global `by_name` index, then a **suffix-stripped fallback** ("Jazz
  Chisholm Jr." → "Jazz Chisholm") that only fires on an exact miss.
- `by_name` is built over the **entire** SFBB player map (~3,800 active rows) plus a
  `dk_name` alias index (**_build_player_index, 413-436**). It is **GLOBAL**, not
  game-scoped.
- Returns the single distinct `mlb_id`, or `None` when 0 or >1 ids survive. Team is
  used **only** to break a genuine >1-id tie, narrowing to candidates whose SFBB
  `team` ∈ the hinted teams — and **only when every hint canonicalizes** (else fail
  open, so a namesake on an unresolved team can't be wrongly excluded)
  (**560-578**). A UNIQUE match is **never** filtered by team (the map's team is a
  possibly-stale single snapshot). `dk_name` is deliberately NOT a tiebreak.

→ **"Team is evidence, not identity" (strategy §2, §7, §15) is ALREADY-DONE** and the
reasoning in the docstring (**548-556**) is exactly the audit's — including the
just-traded-star trap the audit warns about.
→ **"Fuzzy/normalize is candidate-gen, not authority" (strategy §9, §10) is
ALREADY-DONE**: resolution is exact-normalized-or-suffix-stripped; there is **no
similarity-score authority** anywhere. A 92%-match never binds.

### 1.3 Where the id is actually resolved (two parallel paths)
- **Odds-warehouse / prediction enrichment** — `warehouse._enrich_ids`
  (**warehouse.py:336-360**): at capture time, prop lines get
  `player_mlb_id = mlb_id_for_name(player, teams=(home, away))`, team lines get
  `team_code`, the snapshot gets home/away codes. This path **is** event-scoped via
  the two-team hint. `db_store.capture_odds_snapshot` (**db_store.py:900-943**)
  persists `player_mlb_id`/`team_code` per line.
- **Projection stat-fetch** — `espn_client.get_player_stat_history`
  (**espn_client.py:967-1041**) → `gamelog_store.get_athlete_id`
  (**gamelog_store.py:496-561**) → `_mlb_espn_id(name)` (**475-493**).

The `app.py` analyze loop DOES pass game context to the projection —
`get_player_stat_history(..., team_ids=event_team_ids)` (**app.py:2825-2829**), where
`event_team_ids` = the two teams' ESPN ids — but those numeric ESPN ids are used only
inside ESPN `search_athlete._pick` (**espn_client.py:596-610**), NOT inside the
authoritative SFBB resolver (see §3).

---

## 2. Candidate universe — GLOBAL, not gamePk roster (P2b: PARTIAL)

- SFBB resolver: global name index (§1.2).
- statsapi fallback: `find_player_id` (**mlb_starters.py:1456-1475**) → SFBB first,
  then `_player_index(season)` (**1388-1413**) = `sports/1/players?season=` — the
  **season-wide** roster, unique-exact only (no team tiebreak in the fallback; it
  simply requires `len(matches)==1`).
- **The gamePk machinery EXISTS but is wired only into grading, not resolution:**
  `get_schedule_index(date)` (**mlb_starters.py:1342-1386**) returns
  `{gamePk: {home, away, status, ...}}` per date and is used by the outcome resolver
  (**§ around 1270-1340, 1613-1720**) to hard-map a prediction to its gamePk for
  grading. It is **not** consulted when resolving a name → id at prediction time.

→ The recommended "restrict candidates to the **gamePk game roster**" (P2b, strategy
§3-4, §14) is **ABSENT**. The app **approximates** it with the two-team (home/away)
hint in the warehouse path — a weaker filter (two full rosters, and only among SFBB
namesakes, using SFBB's stale team column) but a real one. The stronger version would
use the actual game boxscore/lineup as the authority.

**Trap for whoever builds it:** the two-team hint already covers the dominant
same-game-namesake case cheaply and offline. A gamePk-roster upgrade adds a live
StatsAPI boxscore/roster fetch on the resolution hot path (or a cache thereof) for a
tail that the two-team hint mostly already breaks. Value is real but incremental;
sequence it *after* the fail-closed gate (§4), which is where the actual damage lives.

---

## 3. Game/team context NOT threaded into the projection's stat-fetch resolvers (NEW)

This is a concrete, cheap inconsistency the audit implies but doesn't name directly.

- **Batter path:** `gamelog_store.get_athlete_id(sport, league, name, team_ids=...)`
  (**gamelog_store.py:496-534**) receives `team_ids`, but for baseball it first calls
  `_mlb_espn_id(name)` (**530-534**) which calls
  `player_id_map.espn_id_for_name(name)` / `mlb_id_for_name(name)` **with no `teams=`
  argument** (**485-490**). So the authoritative SFBB tiebreak can't fire; on a
  genuine namesake it returns `None` and drops to ESPN `search_athlete`, whose
  no-team-match fallback returns `candidates[0]` (**espn_client.py:610**).
- **Pitcher path:** `get_pitcher_gamelog(player_name, season)` calls
  `find_player_id(player_name, season)` **without `teams=`** (**mlb_starters.py:1533**),
  even though `find_player_id` accepts and forwards a `teams` hint (1456-1471).

So the two-team context that `_enrich_ids` uses for the warehouse `player_mlb_id` is
**not** used when fetching the player's STATS for the projection. The disambiguation
there is delegated to ESPN's numeric team-id preference — less reliable than the SFBB
map and subject to the `candidates[0]` fallback.

**Fix shape (S):** thread the game's home/away **team names** (not ESPN numeric ids)
down to `_mlb_espn_id` and `get_pitcher_gamelog`, forwarding to
`mlb_id_for_name/espn_id_for_name(..., teams=(home,away))`. `app.py` has the names at
the analyze level. This makes the authoritative map break the tie before ESPN's lossy
first-result path is ever reached. Fully fail-open (a None hint just reproduces
today's behavior).

---

## 4. Fail-closed on ambiguous identity (P2d / strategy §12) — PARTIAL, top keeper

**The id-binder fails closed:**
- `_unique_id` returns `None` on >1 surviving id (**player_id_map.py:560-578**).
- `find_player_id` returns `None` unless unique (**mlb_starters.py:1472-1474**).
- `get_pitcher_gamelog` returns `[]` on a non-unique / wrong-role name — docstring:
  "never a wrong-player bind" (**mlb_starters.py:1515-1517**), and it refuses to bind
  a pitcher prop to a batter (**1537-1538**).

**But the pipeline fails OPEN:**
- Ambiguous name → `player_mlb_id = None` → `player_key = name:<norm>` → the
  prediction/bet is **still produced** under the name key.
- The projection's stat-fetch then drops to ESPN `search_athlete`, which returns
  `candidates[0]` when the team hint fails to match (**espn_client.py:596-610**) — a
  wrong-bind that silently contaminates the projection → edge → recommendation →
  outcome tracking → calibration, precisely the "false positive is far more damaging"
  cascade the audit (P2, strategy §12) calls out.

Frequency is low (concentrated on genuine active namesakes — multiple "Luis Garcia")
but damage is high and silent. This is the single most defensible NEW idea in the
cluster and it aligns with the stated philosophy ("fail-closed identity").

**Fix shape (M):** an explicit ambiguity gate. When candidate gen yields a genuine
>1-id namesake collision that the (now team-hinted, §3) resolver cannot break, and
the market is MLB, **skip the prop** (no prediction, no bet) rather than fall through
to the ESPN first-result path — surfaced as a counted "skipped: ambiguous identity"
diagnostic, not a silent drop. Keep the SFBB unique-match fast path untouched (zero
cost for the 99% case). Must be MLB-gated and fail-open for sports without an SFBB map
(NBA/NFL), or it would wrongly start skipping there.

**Trap:** don't over-fire. A name absent from the SFBB map entirely (unmapped callup)
is NOT the same as an ambiguous collision — the former should keep degrading to ESPN
search (it's the only source), the latter should skip. Gate on "≥2 distinct ids for
this normalized name" specifically, not on "no SFBB id."

---

## 5. Provider-alias table + persist resolutions (strategy §6/§7/§8) — PARTIAL → defer

- **No `provider_alias` table exists.** `sql/schema.sql` tables: prediction_log,
  wagers, market_prediction_log, recalibration_*, odds_snapshot, odds_line,
  mlb_batter_gamelog, mlb_pitcher_gamelog, nba/nfl_gamelog, gamelog_fetch_meta,
  athlete_id_cache, statcast_player_asof, player_id_map, team_id_map, id_map_meta,
  bankroll_ledger, app_settings. No `valid_from/valid_to/confidence/
  resolution_method` columns anywhere (grep confirms; the only "alias" hit is a
  comment about the curated CLE→Guardians team nickname).
- `player_id_map` is a **static SFBB CSV cross-map** (name_norm/dk_name → mlb/espn
  ids), refreshed on a TTL (**player_id_map.py:320-334, 356-386**). It is NOT a
  learned per-provider-name resolution store.
- **`athlete_id_cache` (gamelog_store.py:149-162) is a de-facto resolution-persistence
  cache**: `(sport, league, player_name_lower, team_key) → athlete_id` with TTL +
  negative caching (**get_athlete_id 496-561, seed_athlete_id 564+**). It PARTIALLY
  satisfies strategy §8 "persist successful resolutions," but: it's keyed on ESPN
  numeric `team_key` (not names), it caches the LOSSY `search_athlete` result too (not
  only deterministic resolutions), and it lacks confidence/method/validity windows.
  It's an ESPN-id cache, not an MLBAM-canonical alias with provenance.

**Verdict — DEFER (lean reject as designed):** a full validity-windowed
`provider_alias` table is more machinery than this app needs. The realized value the
audit wants (learn provider-specific names, reuse safe resolutions, debuggable
provenance) is *mostly already captured* by (a) the SFBB `dk_name` alias index +
suffix-strip normalization, (b) per-row `player_mlb_id` denormalization on
prediction_log/wagers/odds_line, and (c) `athlete_id_cache`. There is a **single
provider** here (The Odds API) and DK-only betting — the multi-provider aliasing the
table is designed for doesn't apply. The strategy §7 caveat ("don't make
provider+name globally unique") is *already honored*, because the resolver never binds
an ambiguous name at all.
**Trigger to revisit:** a concrete logged namesake mis-resolution the SFBB map+team
hint cannot fix, OR onboarding a second odds provider with systematically different
name spellings. Then a narrow `player_alias(provider, name_norm, team_code,
valid_from, valid_to, mlbam_id, method)` becomes worth it.

---

## 6. Market/role as a resolver tiebreak (strategy §11) — PARTIAL → adapt (small) / defer

Role is used to **reject a wrong-role gamelog**, not to **break a namesake tie**:
- `get_pitcher_gamelog` refuses a batter for a pitcher prop (**mlb_starters.py:1537-1538**);
  `_resolve_is_pitcher` (**1444-1453**) makes statsapi the authority.

But when two same-name active players differ in role (a pitcher "Luis Garcia" vs a
batter "Luis Garcia"), the prop_key's role could deterministically break the tie
*inside* the resolver — today it can't, because `_unique_id` just returns `None`.
**Adapt (S):** as a late tiebreak in the resolver, when the market implies a role
(pitcher_* vs batter_*), narrow surviving candidates to that role (via SFBB
POS/ALLPOS or the statsapi is_pitcher index) before failing closed. Low frequency,
low cost; can ride along with WS10 or be deferred.

---

## 7. Cross-cluster note (strategy §13, P1 overlap)

Strategy §13 ("gamePk + MLBAM as the player-game key") and much of
`mlb_statsapi_architecture_summary.txt` are the **data-layer / gamelog** cluster (P1),
not entity resolution. Flagging only that the pitcher gamelog already carries a
`_gamePk` (**mlb_starters.py:1564**) used as a doubleheader sort tiebreak, and the
batter gamelog store is still ESPN-athlete/synthetic-key shaped — evaluate under the
P1 cluster, not here. The identity resolver and that data layer share the gamePk
concept, so if P1 builds a gamePk→boxscore roster cache, §2's roster-universe upgrade
becomes nearly free (reuse the same cache) — worth coordinating.

---

## 8. Per-item verdict table

| # | Audit item | current_state | evidence | verdict | integration | effort |
|---|---|---|---|---|---|---|
| ER1 | P2a event-scoped identity KEY | already-done | db_store.py:167-169, 456; player_key 108-120 | reject (done, stronger) | — | S |
| ER2 | Team=evidence / fuzzy=candidate-gen / id-bind fails closed on ambiguity (P2c, §2,§7,§9,§10) | already-done | player_id_map.py:537-578, 498-518; normalize_name db_store.py:98-105 | reject (the SFBB win) | — | S |
| ER3 | P2b gamePk game-roster candidate universe | partial (two-team hint; gamePk unwired to resolution) | warehouse.py:350-354; mlb_starters.py:1342-1386, 1388-1413 | adapt→defer | WS10 (after P1 roster cache) | M |
| ER4 | P2d fail-CLOSED at pipeline (skip on ambiguity) | partial (id-bind closed; pipeline open → ESPN candidates[0]) | player_id_map.py:560-578; espn_client.py:596-610; player_key name-fallback | **adapt (top keeper)** | **WS10** | M |
| ER5 | Thread game/team context into projection stat-fetch resolvers | absent (no team hint on SFBB path) | gamelog_store.py:485-490, 530-534; mlb_starters.py:1533 | **adapt (cheap, high value)** | **WS10** | S |
| ER6 | provider_alias table w/ validity windows + persist resolutions (§6,§7,§8) | partial (athlete_id_cache; no alias table) | schema.sql (no alias cols); gamelog_store.py:149-162 | defer (lean reject) | backlog | M/L |
| ER7 | Market/role as resolver tiebreak (§11) | partial (rejects wrong-role, not a tiebreak) | mlb_starters.py:1537-1538, 1444-1453 | adapt (small)/defer | WS10 | S |

---

## 9. Recommended integration

Propose **new WS10 — "Event-level entity-resolution hardening"**, small and
reversible, honoring fail-closed identity:
- **ER5 (S)** first — thread home/away **names** into `_mlb_espn_id` and
  `get_pitcher_gamelog` → `mlb_id_for_name/espn_id_for_name(..., teams=)`. Makes the
  authoritative map break ties before ESPN's lossy path. Fail-open.
- **ER4 (M)** — explicit MLB-gated ambiguity gate: on a genuine ≥2-id namesake
  collision the team-hinted resolver can't break, **skip** the prop (counted
  diagnostic), don't fall to ESPN `candidates[0]`. Distinguish "ambiguous" from
  "unmapped" (the latter keeps degrading). Add a forward-tracking counter.
- **ER7 (S)** — optional role tiebreak inside the resolver (pitcher_* / batter_*).
- **ER3 (M)** — gamePk-roster candidate universe: **defer** until/if the P1 gamelog
  cluster builds a gamePk→boxscore roster cache; then layer it as a tiebreak refining
  the SFBB unique-match. Don't add a live boxscore fetch to the hot path standalone.
- **ER6** — backlog; revisit on a real namesake incident or a 2nd odds provider.

**Philosophy check:** none of the keepers violate the stated principles. ER4 *is* the
"fail-closed identity" principle finally enforced end-to-end. All keepers are
fail-open for non-MLB sports (no SFBB map) and reversible pure-code changes; none
touches `calibration/baseball_mlb.json` or the Brier gate. Skipping ambiguous props
slightly reduces slate size — an intentional, philosophy-aligned trade (one lost
opportunity ≪ one poisoned calibration observation).

---

## Verifier verdict (adversarial re-check, 2026-08-10)

Re-opened every cited line against the live code (db_store, player_id_map,
gamelog_store, espn_client, mlb_starters, warehouse, app, props, recalibration,
sql/schema.sql). **All 7 findings CONFIRMED; no verdict changed.** No false
"already-done" and no over-eager "adopt" found. Refinements below.

- **ER1 — CONFIRMED (reject/done).** player_key id-first at db_store.py:108-120;
  UNIQUE (sport_key, event_key, prop_key, player_key, line) at 168-169; event_key =
  event_id or game_date, DERIVED on every write at 452-458 (self-heals, can't drift).
  Stronger than the audit's suggested key. Agree.
- **ER2 — CONFIRMED (reject/done), strengthened.** _unique_id returns None on >1 id,
  team only as a canonicalized tiebreak, UNIQUE match never team-filtered
  (player_id_map.py:537-578); candidate gen = exact-norm + suffix-strip only
  (498-518). Extra check I ran: `grep -niE "difflib|SequenceMatcher|fuzz|similarity|
  ratio\("` across all identity modules → **zero hits**. The memo's strong claim ("no
  similarity-score authority anywhere") is literally true. Agree.
- **ER3 — CONFIRMED (partial → defer).** Two-team hint at warehouse.py:350-354;
  get_schedule_index at mlb_starters.py:1342-1382 is called ONLY at 1660 & 1725 —
  both inside the outcome-resolver / is_confirmed_dnp grading path, never at
  resolution time. "gamePk unwired to resolution" verified. Agree — defer, reuse a
  P1 gamePk→roster cache. (Minor: the finding JSON labels this verdict "adapt" while
  the memo body says "defer"; treat as defer/coordinate — substance is identical.)
- **ER4 — CONFIRMED (top keeper).** Pipeline-open verified end-to-end: SQL projection
  path get_player_stat_history (espn_client.py:1008-1013) → gamelog_store.get_athlete_id
  → _mlb_espn_id(name) with **no team hint** (gamelog_store.py:531) → on ambiguity
  returns None → search_athlete → candidates[0] fallback (espn_client.py:610).
  **Calibration-reachability refinement (sharpens, doesn't weaken the finding):** the
  *statsapi hard-ID* grading path DOES fail closed on ambiguity — resolve_player_game_stat
  → find_player_id(name) with no teams → None (mlb_starters.py:1619-1621). But grading
  then falls back to ESPN via cached_athlete_id (recalibration.py:1462), which has the
  same candidates[0] leak. So the "→ outcome tracking → calibration" cascade IS
  reachable (via the ESPN fallback in both projection and grading), but the *primary*
  and larger damage is the poisoned LIVE projection→edge→recommendation (the money
  problem) — the statsapi hard-ID path already blocks the cleaner calibration route.
  Note also: ER4 must fire only AFTER ER5's team-hint (else it would skip cases ESPN's
  numeric-team-id preference would resolve); once ER5 lands, ER4's residual is
  genuinely-ambiguous-even-with-team, where ESPN can't do better anyway. Sequencing
  (ER5→ER4) is right. Agree — adapt, WS10.
- **ER5 — CONFIRMED (absent, do-first), strengthened.** Batter path _mlb_espn_id calls
  espn_id_for_name/mlb_id_for_name with no teams (gamelog_store.py:485-490); pitcher
  path get_pitcher_gamelog→find_player_id(name, season) with no teams
  (mlb_starters.py:1533). ESPN *numeric* team_ids reach only search_athlete
  (app.py:2819-2829). **Plumbing already partly proven:** props.py:514 and
  props.py:1146-1148 ALREADY call find_player_id(..., teams=(home,away)) for the
  Statcast xBA feature — so the home/away NAMES are demonstrably in scope at the
  projection layer, and the core gamelog resolvers are simply inconsistent with the
  feature path. Cheap, fail-open, do-first. Impl note: espn_id_for_name(name)
  currently takes NO teams param (player_id_map.py:588) — add one alongside the wiring.
  Agree.
- **ER6 — CONFIRMED (defer/lean-reject).** No provider_alias table / no
  valid_from/valid_to/confidence/resolution_method columns anywhere in schema.sql
  (only "alias" hit = the CLE→Guardians nickname comment, line 609). athlete_id_cache
  is keyed on ESPN-numeric team_key and caches the lossy search_athlete result too
  (gamelog_store.py:149-162, 496-561) — a real de-facto resolution cache but not an
  MLBAM-canonical alias with provenance. Single-provider + DK-only makes the
  validity-windowed table over-engineering today; philosophy-aligned to defer. Agree.
- **ER7 — CONFIRMED (partial, small/defer).** Role rejects a wrong-role gamelog
  (get_pitcher_gamelog mlb_starters.py:1537-1538; _resolve_is_pitcher 1444-1453) but
  is never a namesake tiebreak — _unique_id has no role knowledge and just returns
  None. Lowest-value keeper; would require threading prop_key/role into _unique_id.
  Fine to ride WS10 or defer. Agree.

**Net:** the finder's assessment is accurate and well-calibrated to the philosophy.
Keepers → WS10 in the order ER5 (S) → ER4 (M) → ER7 (S optional); ER3 deferred to a
P1 roster cache; ER6 to backlog. Only substantive addition is the ER4 damage-mechanism
refinement (live-recommendation damage is primary; calibration route is partially
blocked by the statsapi hard-ID grader but leaks through the ESPN fallback).
