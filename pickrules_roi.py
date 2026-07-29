"""Pick-rules ROI evaluation lens.

Grade the RECOMMENDED SLATE (the value picks that survive `bet_selector`'s pick
rules) on real resolved outcomes, and compare it against the raw is_value POOL
(no rules). The delta answers *"do the pick rules improve real-world ROI?"*.

Deliberately a THIRD ROI notion, distinct from:
  - `backtest.py` edge-vs-close ROI (stakes only when model_p-market_p >=
    threshold, at the de-vigged closing price), and
  - `wagers` stake-weighted realized ROI (what the user actually bet).
It reuses the flat-1-unit-per-pick convention of `recalibration.summarize_*`
(win=+dec-1, loss=-1, push=0.0; push excluded from hit/Brier denominators,
included in the ROI denominator; direction-aware; dedup resolved-over-unresolved
by identity).

The slate is RE-DERIVED, not read from a stored flag. The cross-market rules
(ER/K, total-over+prop-under, anti-correlation) only fire on markets present in
the analyzed pool, so a per-session "recommended" flag would be an artifact of
*which markets the user happened to analyze together* — an earned-runs OVER would
be on-slate in a hits+ER run but off-slate in a hits+ER+K run. Instead we group
every logged is_value forecast by (sport, ET game-date) — `prediction_log`
accumulates across runs — and run the real rule engine once over that whole pool.

Structural guarantees (see the plan): neither the pool nor the slate can grade
both sides of a line or both teams — `prediction_log` stores one directional row
per (sport, event, prop, player, line) and `market_prediction_log` one favored
side per (event, bet_type). The slate additionally drops softer cross-market
contradictions (kept in the pool as the comparison baseline).

Team-based rules (Rule-of-3, opposing-team L3, batting-order) need `team` /
`batting_order`, which are logged from the prop candidate going forward; rows
predating those columns simply don't fire those rules (the rule predicates
exempt team=None / fail open on batting_order=None) and are reported in the
`rules_skipped` note rather than silently replayed.
"""

import argparse

from odds_client import american_to_decimal
from pricing_common import _expected_roi, et_local_date
import recalibration
from recalibration import (
    market_prediction_identity,
    prediction_identity,
    summarize_market_prediction_rows,
    summarize_prediction_rows,
)

# Rules that replay from stored fields regardless of the team/batting_order
# backfill state (fields they need are always present in the logs).
_ALWAYS_REPLAYED = [
    "is_value", "under-0.5 suppression", "L1 hard conflicts",
    "L2 anti-correlation", "ER/K same-pitcher", "total-over + prop-under",
]
# Rules that need team / batting_order (logged going forward only).
_TEAM_DEPENDENT = ["Rule-of-3", "opposing-team L3", "batting-order gate"]


# ── candidate reconstruction from log rows ─────────────────────────────────

def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prop_pool_entry(row):
    """A `(sel_key, bet_type, side, cand)` pool entry for `bet_selector` built
    from a prediction_log row. `final_prob`/`raw_prob` are ALWAYS P(OVER), so the
    pick's hit probability is direction-adjusted; EV/edge are derived from the
    logged DK price. sel_key = ("prop",) + prediction_identity (maps back 1:1)."""
    direction = (row.get("direction") or "OVER").upper()
    over_prob = row.get("final_prob")
    if over_prob is None:
        over_prob = row.get("raw_prob")
    try:
        over_prob = float(over_prob) if over_prob is not None else None
    except (TypeError, ValueError):
        over_prob = None
    p_hit = None
    if over_prob is not None:
        p_hit = over_prob if direction == "OVER" else 1.0 - over_prob
    price = _int_or_none(row.get("price"))
    ev = _expected_roi(p_hit, price) if (p_hit is not None and price) else None
    edge = None
    if p_hit is not None and price:
        edge = p_hit - 1.0 / american_to_decimal(price)  # vig-inclusive approx
    cand = {
        "event_id": row.get("event_id"),
        "team": row.get("team"),
        "player": row.get("player"),
        "prop": row.get("prop_key"),
        "direction": direction,
        "batting_order": row.get("batting_order"),
        "over_rate": over_prob * 100.0 if over_prob is not None else None,
        "expected_roi_pct": ev * 100.0 if ev is not None else None,
        "edge_pct": edge * 100.0 if edge is not None else None,
    }
    return (("prop",) + prediction_identity(row), "player_prop", None, cand)


def _market_pool_entry(row):
    """A pool entry from a market_prediction_log row. `model_prob` is already the
    PICKED side's probability; totals carry side over/under (uppercased to match
    bet_selector). sel_key = ("mkt",) + market_prediction_identity."""
    bet_type = row.get("bet_type")
    side = None
    if bet_type == "total":
        side = "OVER" if (row.get("side") or "").lower() == "over" else "UNDER"
    try:
        p_hit = float(row.get("model_prob"))
    except (TypeError, ValueError):
        p_hit = None
    price = _int_or_none(row.get("price"))
    ev = _expected_roi(p_hit, price) if (p_hit is not None and price) else None
    ev_pct = ev * 100.0 if ev is not None else None
    cand = {
        "event_id": row.get("event_id"),
        "team": row.get("team"),
        "expected_roi_pct": ev_pct,
        "edge_pct": ev_pct,  # only a tiebreaker under metric="ev"
    }
    if bet_type == "total":
        key = "over" if side == "OVER" else "under"
        cand[f"{key}_expected_roi_pct"] = ev_pct
        cand[f"{key}_edge_pct"] = ev_pct
    return (("mkt",) + market_prediction_identity(row), bet_type, side, cand)


# ── slate re-derivation ─────────────────────────────────────────────────────

def _dedup_value_rows(rows, sport_key, identity_fn):
    """Return {identity: row} for is_value rows of `sport_key`, preferring the
    resolved row (then the latest ts) per identity — mirrors summarize_*'s dedup
    so the re-derived slate uses the same representative row summarize grades."""
    best = {}
    for row in rows:
        if sport_key and row.get("sport_key") != sport_key:
            continue
        if not row.get("is_value"):
            continue
        ident = identity_fn(row)
        cur = best.get(ident)
        if cur is None:
            best[ident] = row
            continue
        cur_res, new_res = bool(cur.get("resolved")), bool(row.get("resolved"))
        if new_res and not cur_res:
            best[ident] = row
        elif new_res == cur_res and (row.get("ts") or "") >= (cur.get("ts") or ""):
            best[ident] = row
    return best


def _slate_date(row):
    """ET game-date grouping key (a slate = one day's games). ET on read; falls
    back to the stored game_date when commence_time is absent."""
    return et_local_date(row.get("commence_time")) or (row.get("game_date") or "")


def rederive_slate(pred_rows, market_rows, sport_key):
    """Re-derive the recommended slate over every logged is_value forecast.

    Groups deduped is_value picks by (sport, ET game-date) — so cross-market
    rules see everything analyzed for that date, not one session — and runs
    `bet_selector.select_top_bets(pool, sport, n=len(pool), metric="ev")` per
    date (n=len(pool) removes the top-N cap so the result isolates the rule
    FILTERS). Returns (slate_pred_idents, slate_market_idents, rules_applied,
    rules_skipped).
    """
    import bet_selector

    pred_by_ident = _dedup_value_rows(pred_rows, sport_key, prediction_identity)
    mkt_by_ident = _dedup_value_rows(market_rows, sport_key,
                                     market_prediction_identity)

    # Group both stores by ET game-date.
    by_date = {}
    for ident, row in pred_by_ident.items():
        by_date.setdefault(_slate_date(row), {"prop": [], "mkt": []})["prop"].append(row)
    for ident, row in mkt_by_ident.items():
        by_date.setdefault(_slate_date(row), {"prop": [], "mkt": []})["mkt"].append(row)

    slate_pred, slate_mkt = set(), set()
    for _date, groups in by_date.items():
        pool = ([_prop_pool_entry(r) for r in groups["prop"]]
                + [_market_pool_entry(r) for r in groups["mkt"]])
        chosen = bet_selector.select_top_bets(pool, sport_key, n=len(pool),
                                              metric="ev")
        for sel_key in chosen:
            if sel_key[0] == "prop":
                slate_pred.add(sel_key[1:])
            else:
                slate_mkt.add(sel_key[1:])

    # Honest reporting: team-based rules only fully replay once team/order backfill.
    without_team = sum(1 for r in pred_by_ident.values() if not r.get("team"))
    rules_applied = list(_ALWAYS_REPLAYED)
    rules_skipped = []
    if without_team:
        rules_skipped = [
            f"{r} (needs team/batting_order; {without_team} of "
            f"{len(pred_by_ident)} prop picks predate those columns)"
            for r in _TEAM_DEPENDENT
        ]
    else:
        rules_applied += _TEAM_DEPENDENT
    return slate_pred, slate_mkt, rules_applied, rules_skipped


# ── pool-vs-slate grading ───────────────────────────────────────────────────

def _combine(prop_summary, market_summary):
    """Combine prop + team summaries: ROI priced-count-weighted, hit-rate
    graded-count-weighted (the denominators differ, so never average the two
    percentages)."""
    roi_num, roi_den = 0.0, 0
    for summ, roi_key in ((prop_summary, "realized_roi"), (market_summary, "roi")):
        roi, priced = summ.get(roi_key), summ.get("priced_resolved") or 0
        if roi is not None and priced:
            roi_num += roi * priced
            roi_den += priced
    hit_num, hit_den = 0.0, 0
    for summ, hr_key in ((prop_summary, "direction_hit_rate"),
                         (market_summary, "hit_rate")):
        hr, graded = summ.get(hr_key), summ.get("graded") or 0
        if hr is not None and graded:
            hit_num += hr * graded
            hit_den += graded

    def _sum(key):
        return (prop_summary.get(key) or 0) + (market_summary.get(key) or 0)

    return {
        "total": _sum("total"),
        "resolved": _sum("resolved"),
        "graded": _sum("graded"),
        "priced_resolved": _sum("priced_resolved"),
        "roi": roi_num / roi_den if roi_den else None,
        "hit_rate": hit_num / hit_den if hit_den else None,
    }


def _rows_for_idents(rows, sport_key, identity_fn, idents):
    """All is_value rows of `sport_key` whose identity is in `idents` (keeps both
    a resolved row and its unresolved re-log sibling — summarize dedups to the
    resolved one, so no graded outcome is lost)."""
    out = []
    for row in rows:
        if sport_key and row.get("sport_key") != sport_key:
            continue
        if not row.get("is_value"):
            continue
        if identity_fn(row) in idents:
            out.append(row)
    return out


def _value_rows(rows, sport_key):
    return [r for r in rows
            if r.get("is_value") and (not sport_key or r.get("sport_key") == sport_key)]


def slate_vs_pool(pred_rows, market_rows, sport_key):
    """Grade the is_value POOL and the re-derived rule SLATE and return the delta.

    Returns a dict with per-section (props / markets / combined) {pool, slate}
    summaries, the combined delta, n_dropped picks, and the replayed/skipped rule
    notes. Both sides use `recalibration.summarize_*` (flat-1u convention)."""
    slate_pred, slate_mkt, rules_applied, rules_skipped = rederive_slate(
        pred_rows, market_rows, sport_key)

    pool_pred = _value_rows(pred_rows, sport_key)
    pool_mkt = _value_rows(market_rows, sport_key)
    slate_pred_rows = _rows_for_idents(pred_rows, sport_key,
                                       prediction_identity, slate_pred)
    slate_mkt_rows = _rows_for_idents(market_rows, sport_key,
                                      market_prediction_identity, slate_mkt)

    props = {"pool": summarize_prediction_rows(pool_pred, sport_key),
             "slate": summarize_prediction_rows(slate_pred_rows, sport_key)}
    markets = {"pool": summarize_market_prediction_rows(pool_mkt, sport_key),
               "slate": summarize_market_prediction_rows(slate_mkt_rows, sport_key)}
    combined = {"pool": _combine(props["pool"], markets["pool"]),
                "slate": _combine(props["slate"], markets["slate"])}

    def _delta(key):
        s, p = combined["slate"].get(key), combined["pool"].get(key)
        return (s - p) if (s is not None and p is not None) else None

    return {
        "sport_key": sport_key,
        "props": props,
        "markets": markets,
        "combined": combined,
        "delta": {"roi": _delta("roi"), "hit_rate": _delta("hit_rate")},
        "n_dropped": (combined["pool"]["total"] - combined["slate"]["total"]),
        "rules_applied": rules_applied,
        "rules_skipped": rules_skipped,
    }


# ── CLI report ──────────────────────────────────────────────────────────────

def _pl_units(summary, roi_key):
    roi, priced = summary.get(roi_key), summary.get("priced_resolved") or 0
    return roi * priced if (roi is not None and priced) else 0.0


def _fmt_hit(x):
    return f"{x * 100:.1f}%" if x is not None else "   n/a"


def _fmt_brier(x):
    return f"{x:.3f}" if x is not None else "  n/a"


def _fmt_roi(x):
    return f"{x * 100:+.1f}" if x is not None else "  n/a"


def _section_rows(label, pool, slate, hit_key, brier_key, roi_key):
    hdr = (f"  {label}\n"
           f"    {'':<12} {'picks':>6} {'graded':>7} {'hit%':>7} "
           f"{'Brier':>7} {'priced':>7} {'ROI%':>7} {'P/L(u)':>8}")
    lines = [hdr, "    " + "-" * (len(hdr.splitlines()[-1]) - 4)]
    for name, summ in (("Value pool", pool), ("Rule slate", slate)):
        lines.append(
            f"    {name:<12} {summ.get('total', 0):>6} "
            f"{summ.get('graded', 0):>7} {_fmt_hit(summ.get(hit_key)):>7} "
            f"{_fmt_brier(summ.get(brier_key)):>7} "
            f"{summ.get('priced_resolved', 0):>7} "
            f"{_fmt_roi(summ.get(roi_key)):>7} "
            f"{_pl_units(summ, roi_key):>+8.2f}")
    d_picks = slate.get("total", 0) - pool.get("total", 0)
    d_roi = (slate.get(roi_key) - pool.get(roi_key)
             if slate.get(roi_key) is not None and pool.get(roi_key) is not None
             else None)
    d_pl = _pl_units(slate, roi_key) - _pl_units(pool, roi_key)
    lines.append(
        f"    {'Δ slate-pool':<12} {d_picks:>+6} {'':>7} {'':>7} {'':>7} "
        f"{'':>7} {_fmt_roi(d_roi):>7} {d_pl:>+8.2f}")
    return "\n".join(lines)


def print_report(result):
    sport = result["sport_key"]
    print(f"\n=== Pick-rules ROI lens — {sport} ===")
    print("Grades the RECOMMENDED SLATE (bet_selector rules) vs the raw is_value")
    print("POOL on real outcomes, flat 1u/pick. Distinct from backtest edge-vs-")
    print("close ROI and the wagers ledger.\n")
    print("Replayed rules: " + ", ".join(result["rules_applied"]))
    if result["rules_skipped"]:
        print("Partial/skipped:")
        for note in result["rules_skipped"]:
            print(f"  - {note}")
    print()
    print(_section_rows("Props", result["props"]["pool"], result["props"]["slate"],
                        "direction_hit_rate", "probability_brier", "realized_roi"))
    print()
    print(_section_rows("Team markets", result["markets"]["pool"],
                        result["markets"]["slate"], "hit_rate", "brier", "roi"))
    print()
    # Combined block uses the unified keys _combine emits.
    print(_section_rows("Combined", result["combined"]["pool"],
                        result["combined"]["slate"], "hit_rate", None, "roi"))
    print()
    d_roi, pool_roi = result["delta"]["roi"], result["combined"]["pool"].get("roi")
    slate_roi = result["combined"]["slate"].get("roi")
    if d_roi is None:
        print("Summary: not enough resolved priced picks to compare ROI yet.")
    else:
        print(f"Summary: rule slate ROI {_fmt_roi(slate_roi)}% vs pool "
              f"{_fmt_roi(pool_roi)}% (Δ {d_roi * 100:+.1f}pp) over "
              f"{result['combined']['slate'].get('priced_resolved', 0)} priced "
              f"slate picks; the pick rules dropped {result['n_dropped']} picks.")


def _within_days(row, cutoff):
    if cutoff is None:
        return True
    return _slate_date(row) >= cutoff


def main():
    p = argparse.ArgumentParser(
        description="Pick-rules ROI lens: grade the recommended slate vs the "
                    "is_value pool on real outcomes (flat 1u/pick).")
    p.add_argument("--sport", choices=list(recalibration.SPORT_ESPN_MAP.keys()),
                   required=True)
    p.add_argument("--days", type=int, default=None,
                   help="Only include games within the last N ET days.")
    p.add_argument("--json", action="store_true",
                   help="Emit the raw result dict as JSON instead of the table.")
    args = p.parse_args()

    # Target the SQL backend when SQL_* secrets are configured (mirrors refit).
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass

    cutoff = None
    if args.days is not None:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=args.days)).strftime("%Y-%m-%d")

    pred_rows = [r for r in recalibration.read_prediction_log()
                 if _within_days(r, cutoff)]
    market_rows = [r for r in recalibration.read_market_prediction_log()
                   if _within_days(r, cutoff)]
    result = slate_vs_pool(pred_rows, market_rows, args.sport)

    if args.json:
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
