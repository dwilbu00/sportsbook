"""R2 backtest data layer: fetch DraftKings + Pinnacle closing prop lines from the
per-book warehouse, pair them at the SAME snapshot, and build book-aware,
point-preserving offers for the r2_edge/r2_sharp pricing core — plus a bulk
realized-outcome index for game_pk-exact grading.

Built to the red-team spec (wf_46cca99d). The non-obvious guards it enforces:

  * TWO book-scoped reads, never bookmaker=None — the bulk readers' SELECT omits
    the bookmaker column (db_store.py:1506-1514), so a book can't be recovered after
    the fact. We read 'draftkings' and 'pinnacle' separately and tag each row.
  * DK and Pinnacle MUST come from the SAME snapshot_id (one odds_snapshot holds all
    books at one capture instant). We pick, per event, the LATEST snapshot with
    captured_at <= commence_time that carries BOTH books, and assemble both from it —
    so the fair and the price are contemporaneous by construction. No cross-snapshot
    fill, no abs-nearest (which can select a post-first-pitch in-play snapshot).
  * Pinnacle offers keep the per-POINT grain (0.5 and 1.5 both survive) so the
    cross-line projector can price DK's line.
  * Realized grading is game_pk-EXACT off FROZEN bulk facts (get_calib_gamelogs_bulk,
    zero network) — never a date match (doubleheader/namesake cross-grade) and never
    resolve_actual (live boxscore fetch = non-reproducible).

Pure assembly (select_prop_legs / _assemble_from_snapshot / _parse_ts) is separated
from the DB-touching fetch so it unit-tests without a warehouse. The driver
(r2_backtest.py) consumes PropLeg records + the outcome index.
"""
from collections import Counter, defaultdict, namedtuple
from datetime import datetime, timezone

# One bettable DK prop leg paired with the sharp book's offers at the SAME snapshot.
#   pinnacle_offers = [{point, over_price, under_price}, ...] (per-point, both sides
#   where posted) — fed straight to r2_edge.prop_leg_edges.
PropLeg = namedtuple("PropLeg",
                     "event_id game_date commence_time captured_at snapshot_id "
                     "game_pk player player_mlb_id prop_key dk_point "
                     "dk_over_price dk_under_price pinnacle_offers ref_prop")
PropLeg.__new__.__defaults__ = (None,)      # ref_prop defaults to same-prop pricing

_DK = "draftkings"
_PIN = "pinnacle"

# Cross-market sharp-reference synonyms: {DK prop: (Pinnacle prop, {valid DK points})}.
# batter_hits Over/Under 0.5 == batter_total_bases Over/Under 0.5 EXACTLY — a batter
# accrues a total base ONLY via a hit and any hit gives >=1 TB, so TB>=1 <=> H>=1.
# ONLY the 0.5 line maps: at 1.5, P(TB>=2) != P(H>=2) (a double is 2 TB from one hit).
# Pinnacle books ZERO batter_hits but books TB densely (incl. at 0.5), so this makes
# TB the sharp reference for DK's largest prop, hits 0.5. Grading still uses the DK
# market's own actual (H) — the synonym is a PRICING reference only.
_SYNONYM = {"batter_hits": ("batter_total_bases", frozenset({0.5}))}


def _parse_ts(s):
    """Parse an ISO String(40) warehouse timestamp (e.g. '2024-06-26T23:05:38Z') to
    an aware UTC datetime, or None. Mirrors mlb_warehouse._parse_utc but dependency-
    free so the pure assembly path unit-tests without importing the warehouse."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _player_key(row):
    """Group DK and Pinnacle legs for the same player: prefer the MLBAM id (both
    books enrich off the same SFBB map, so ids match even when name strings differ);
    fall back to a normalized name only when the id is absent."""
    pid = row.get("player_mlb_id")
    if pid:
        return ("id", str(pid))
    name = (row.get("player") or "").strip().lower()
    return ("nm", name)


def _pin_offers(pin_by_point):
    """Per-point two-sided Pinnacle offers list from {point: {side: price}}."""
    return [{"point": pt, "over_price": s.get("OVER"), "under_price": s.get("UNDER")}
            for pt, s in pin_by_point.items()]


def _assemble_from_snapshot(snap_rows, stats):
    """Build PropLeg records from the rows of ONE (event, close) snapshot that
    carries both books. Groups by PLAYER (so a DK prop can reference a DIFFERENT
    Pinnacle prop via _SYNONYM — hits priced off TB), then per prop: Pinnacle keeps
    its per-point two-sided offers, DK yields one leg per posted point. Drops+counts
    legs with no two-sided Pinnacle reference or an out-of-range synonym line."""
    players = defaultdict(lambda: {
        "dk": defaultdict(lambda: defaultdict(dict)),   # prop -> point -> {side: price}
        "pin": defaultdict(lambda: defaultdict(dict)),
        "meta": None})
    for r in snap_rows:
        pl = players[_player_key(r)]
        # Prefer a meta row that carries the MLBAM id + game_pk for grading.
        if pl["meta"] is None or (r.get("player_mlb_id") and not pl["meta"].get("player_mlb_id")):
            pl["meta"] = r
        prop, side = r.get("prop_key"), (r.get("direction") or "").upper()
        pt, price = r.get("point"), r.get("price")
        if not prop or pt is None or price is None or side not in ("OVER", "UNDER"):
            continue
        book = "dk" if r.get("book") == _DK else "pin"
        pl[book][prop][pt][side] = price

    out = []
    for _pkey, pl in players.items():
        meta = pl["meta"]
        for dk_prop, dk_by_point in pl["dk"].items():
            # Sharp reference: same prop by default, or a cross-market synonym
            # (hits -> TB) restricted to the lines where the identity holds.
            ref_prop, valid_pts = _SYNONYM.get(dk_prop, (dk_prop, None))
            pin_offers = _pin_offers(pl["pin"].get(ref_prop, {}))
            has_two_sided = any(o["over_price"] is not None
                                and o["under_price"] is not None for o in pin_offers)
            for dk_pt, sides in dk_by_point.items():
                if valid_pts is not None and dk_pt not in valid_pts:
                    stats["leg_dropped_synonym_bad_point"] += 1
                    continue                      # e.g. hits 1.5 has no TB identity
                if not has_two_sided:
                    stats["leg_dropped_no_pinnacle_twosided"] += 1
                    continue
                out.append(PropLeg(
                    event_id=meta.get("event_id"), game_date=meta.get("game_date"),
                    commence_time=meta.get("commence_time"),
                    captured_at=meta.get("captured_at"),
                    snapshot_id=meta.get("snapshot_id"), game_pk=meta.get("game_pk"),
                    player=meta.get("player"), player_mlb_id=meta.get("player_mlb_id"),
                    prop_key=dk_prop, dk_point=dk_pt,
                    dk_over_price=sides.get("OVER"), dk_under_price=sides.get("UNDER"),
                    pinnacle_offers=pin_offers,
                    ref_prop=(ref_prop if ref_prop != dk_prop else None)))
                stats["leg_built"] += 1
    return out


def select_prop_legs(rows):
    """PURE core: from combined DK+Pinnacle prop rows (each tagged row['book']),
    pick each event's CLOSE snapshot (latest captured_at <= commence_time carrying
    BOTH books) and assemble PropLeg records from it.

    Returns (legs, stats). stats counts dropped events/legs for coverage reporting.
    Enforces the temporal + same-snapshot guards: post-commence / undated snapshots
    are excluded, and DK and Pinnacle are only ever paired within one snapshot_id."""
    stats = Counter()
    by_event = defaultdict(list)
    for r in rows:
        by_event[r.get("event_id")].append(r)

    legs = []
    for eid, ev_rows in by_event.items():
        stats["events_seen"] += 1
        # Per snapshot_id: latest capture instant + which books + its rows.
        snaps = {}
        for r in ev_rows:
            cap = _parse_ts(r.get("captured_at"))
            com = _parse_ts(r.get("commence_time"))
            if cap is None or com is None or cap > com:   # drop undated / in-play
                continue
            sid = r.get("snapshot_id")
            s = snaps.setdefault(sid, {"cap": cap, "books": set(), "rows": []})
            s["books"].add(r.get("book"))
            s["rows"].append(r)
        both = [(s["cap"], sid, s) for sid, s in snaps.items()
                if {_DK, _PIN} <= s["books"]]
        if not both:
            stats["events_dropped_no_both_book_close"] += 1
            continue
        both.sort(key=lambda x: x[0])
        _cap, _sid, close = both[-1]                       # latest pre-commence close
        legs.extend(_assemble_from_snapshot(close["rows"], stats))
    return legs, stats


# ── DB-touching orchestration ────────────────────────────────────────────────

def _fetch_book(sport, season, prop_keys, bookmaker):
    """One book-scoped, season-scoped bulk read, tagged with its book."""
    import db_store
    rows = db_store.player_prop_lines(
        sport, date_from=f"{season}-01-01", date_to=f"{season}-12-31",
        prop_keys=list(prop_keys), bookmaker=bookmaker)
    for r in rows:
        r["book"] = bookmaker
    return rows


def load_prop_legs(sport, seasons, prop_keys):
    """Fetch + pair DK/Pinnacle prop legs for each season. Returns
    (legs_by_season={season: [PropLeg]}, stats_by_season={season: Counter})."""
    import db_store
    db_store.promote_secrets_from_toml()
    legs_by_season, stats_by_season = {}, {}
    for s in seasons:
        dk = _fetch_book(sport, s, prop_keys, _DK)
        pin = _fetch_book(sport, s, prop_keys, _PIN)
        legs, stats = select_prop_legs(dk + pin)
        # Assert the same-snapshot invariant held (defensive; select_prop_legs
        # already only pairs within one snapshot_id).
        legs_by_season[s] = legs
        stats_by_season[s] = stats
    return legs_by_season, stats_by_season


# ── Realized-outcome index (frozen facts, game_pk-exact) ──────────────────────

def _prop_outcome_spec(prop_key):
    """(role, stat_column, xform) for a prop, derived from mlb_warehouse's own
    _ACTUAL_STAT_SPEC so it can never drift from the grading source of truth."""
    import mlb_warehouse as wh
    spec = wh._ACTUAL_STAT_SPEC.get(prop_key)
    if not spec:
        return None
    table, col, xform = spec
    role = "pitcher" if table is wh.mlb_pitcher_game else "batter"
    return role, col, xform


def build_outcome_index(seasons, prop_keys):
    """{role: {(athlete_id_str, game_pk_int): gamelog_rec}} across seasons, from the
    FROZEN bulk facts (get_calib_gamelogs_bulk, zero network). ~one query per
    (role, season)."""
    import mlb_warehouse as wh
    roles = {spec[0] for pk in prop_keys
             if (spec := _prop_outcome_spec(pk)) is not None}
    idx = {}
    for role in roles:
        role_idx = {}
        for s in seasons:
            bulk = wh.get_calib_gamelogs_bulk(role, int(s))
            for aid, games in bulk.items():
                for g in games:
                    gpk = g.get("game_pk")
                    if gpk is None:
                        continue
                    role_idx[(str(aid), int(gpk))] = g
        idx[role] = role_idx
    return idx


def outcome_value(idx, prop_key, player_mlb_id, game_pk):
    """Realized stat for (player, game) via game_pk-exact lookup, or None (DNP / not
    ingested / unmapped). NEVER coalesces a miss to 0 (that would grade a scratch as
    an OVER loss)."""
    spec = _prop_outcome_spec(prop_key)
    if spec is None or player_mlb_id is None or game_pk is None:
        return None
    role, col, xform = spec
    try:
        key = (str(player_mlb_id), int(game_pk))
    except (TypeError, ValueError):
        return None
    rec = idx.get(role, {}).get(key)
    if rec is None:
        return None
    val = rec.get(col)
    if val is None:
        return None
    if xform == "ip_to_outs":
        import mlb_warehouse as wh
        return wh._ip_to_outs(float(val))
    return float(val)
