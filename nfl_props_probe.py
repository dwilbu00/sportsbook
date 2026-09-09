"""nfl_props_probe.py — opportunity-model PROBE for NFL player props (receptions POC).

The MLB opportunity-first arc taught us to PROBE before building: cheap go/no-go on
whether the opportunity structure adds real signal, vs the incumbent AND vs the market.

Receptions is the cleanest NFL opportunity structure and a direct analog of MLB's
method-D: receptions = targets (trials) × catch-rate (success) → a binomial, so
P(receptions >= k) = stats.hits_at_least(k, expected_targets, catch_rate). This probes
two questions on the real-line corpus (quote-time filtered, participation-safe, per
season):

  1. PREDICTION — does the opportunity binomial beat the incumbent (method A empirical
     over-rate) on OOS Brier? (keep-if-better)
  2. EDGE — since NFL props are RECREATIONAL, does betting the model's edge (model
     P(over) vs de-vigged market) beat the DK/FD close (ROI + realized-minus-implied),
     replicated across seasons? (the actual prize)

Leakage-safe: features use only the player's PRIOR weeks in the same season. Diagnostic
only — writes nothing. Reuses nfl_props_scan's loaders (quote-time + participation-safe
grading) so this and the scan can't drift.
"""
import argparse
import math
from collections import defaultdict

import nfl_schedule
import nfl_data
import nfl_props_scan as scan
from stats import hits_at_least
from odds_client import (american_to_decimal, american_to_implied_prob,
                         devig_two_way)

SPORT = "americanfootball_nfl"
PROP = "player_receptions"

HALF_LIFE = 4          # weeks — recency weight on prior-week targets/catch-rate
MIN_PRIOR = 3          # need >= this many prior-week rows to project
CATCH_BOUNDS = (0.20, 0.95)


def _recency_weights(n, half_life):
    if n <= 0:
        return []
    if not half_life or half_life <= 0:
        return [1.0] * n
    # index 0 = most recent (prior_weeks are ordered most-recent-first below)
    return [0.5 ** (i / half_life) for i in range(n)]


def _player_target_series(seasons):
    """{(norm_name, season): [(week:int, targets, receptions), ...] most-recent-first}
    from player_week (rows exist only for games the player recorded stats in)."""
    pw = nfl_data.player_week([str(s) for s in seasons])
    if pw is None:
        return {}
    cols = set(pw.columns)
    namecol = "player_display_name" if "player_display_name" in cols else "player_name"
    have_t = "targets" in cols
    acc = defaultdict(list)
    for r in pw.to_dict("records"):
        try:
            wk = int(r.get("week"))
        except (TypeError, ValueError):
            continue
        nm = scan._norm(r.get(namecol))
        if not nm:
            continue
        tg = r.get("targets") if have_t else None
        rc = r.get("receptions")
        acc[(nm, str(r.get("season")))].append((wk, tg, rc))
    for k in acc:
        acc[k].sort(key=lambda x: -x[0])       # most-recent-first
    return acc


def build_corpus(seasons, snapshot="closing", book="draftkings"):
    """List of receptions obs with as-of opportunity features + market + outcome."""
    props = scan._load_props(seasons, snapshot, book)
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    pw_stat = scan._player_week_index(seasons)          # (name,season,week)->stat
    snaps = scan._snap_presence([str(s) for s in seasons])
    series = _player_target_series(seasons)

    obs = []
    n_props = n_nogame = n_void = n_played0 = n_thin = n_notargets = 0
    for (eid, player, pk), d in props.items():
        if pk != PROP:
            continue
        n_props += 1
        gid, _ = nfl_schedule.resolve_event(d["home"], d["away"], d["commence"],
                                            index=idx)
        if gid is None:
            n_nogame += 1
            continue
        parts = gid.split("_")
        season, week = parts[0], int(parts[1])
        nm = scan._norm(player)
        # ── actual (participation-safe) ──
        actual = (pw_stat.get((nm, season, str(week))) or {}).get("receptions")
        if actual is None:
            if snaps.get((nm, season, str(week))):
                actual = 0.0
                n_played0 += 1
            else:
                n_void += 1
                continue
        # ── as-of prior-week features (same season, strictly earlier weeks) ──
        prior = [(wk, tg, rc) for (wk, tg, rc) in series.get((nm, season), [])
                 if wk < week]
        if len(prior) < MIN_PRIOR:
            n_thin += 1
            continue
        w = _recency_weights(len(prior), HALF_LIFE)
        tw = sum(wi * (tg or 0.0) for (_wk, tg, _rc), wi in zip(prior, w))
        rw = sum(wi * (rc or 0.0) for (_wk, _tg, rc), wi in zip(prior, w))
        wsum = sum(w)
        exp_targets = tw / wsum if wsum else 0.0
        if exp_targets <= 0:
            n_notargets += 1
            continue
        catch = rw / tw if tw > 0 else 0.0
        catch = max(CATCH_BOUNDS[0], min(CATCH_BOUNDS[1], catch))
        # incumbent A: empirical over-rate at the line over prior weeks (rec history)
        rec_hist = [rc for (_wk, _tg, rc) in prior if rc is not None]
        line = d.get("OVER", d.get("UNDER", (None, None)))[0]
        if line is None:
            continue
        emp_over = (sum(1 for rc in rec_hist if rc > line) / len(rec_hist)
                    if rec_hist else 0.5)
        over = d.get("OVER")
        under = d.get("UNDER")
        obs.append({
            "season": season, "week": week, "player": player, "line": line,
            "over_price": over[1] if over else None,
            "under_price": under[1] if under else None,
            "actual": float(actual),
            "exp_targets": exp_targets, "catch": catch, "emp_over": emp_over,
            "n_prior": len(prior),
        })
    meta = {"n_props": n_props, "n_obs": len(obs), "n_nogame": n_nogame,
            "n_void": n_void, "n_played0": n_played0, "n_thin": n_thin,
            "n_notargets": n_notargets,
            "seasons": sorted({g.split('_')[0] for g in idx})}
    return obs, meta


def _p_over_opportunity(line, exp_targets, catch):
    """Binomial P(receptions >= k), k=int(line)+1, with the adjacent-count mixture
    over ⌊m⌋/⌈m⌉ expected targets (smooth in exp_targets)."""
    k = int(line) + 1
    m = max(0.0, exp_targets)
    lo = int(m)
    w_hi = m - lo
    surv = lambda nn: hits_at_least(k, nn, catch) if nn >= 1 else 0.0
    return (1.0 - w_hi) * surv(lo) + w_hi * surv(lo + 1)


def _brier(sub, fn):
    return sum((fn(o) - o["y"]) ** 2 for o in sub) / len(sub) if sub else None


def probe(seasons, snapshot="closing", book="draftkings"):
    obs, meta = build_corpus(seasons, snapshot, book)
    print("=" * 92)
    print(f"  NFL RECEPTIONS opportunity PROBE (DK {snapshot}, {meta['seasons']})")
    print(f"  built {meta['n_obs']:,} obs  (played0={meta['n_played0']}, "
          f"void={meta['n_void']}, no-game={meta['n_nogame']}, thin<{MIN_PRIOR}="
          f"{meta['n_thin']})")
    print("=" * 92)
    # drop pushes; attach outcome + both model probs
    rows = []
    for o in obs:
        if abs(o["actual"] - o["line"]) < 1e-9:
            continue
        o["y"] = 1 if o["actual"] > o["line"] else 0
        o["p_opp"] = max(0.0, min(1.0, _p_over_opportunity(
            o["line"], o["exp_targets"], o["catch"])))
        o["p_inc"] = max(0.0, min(1.0, o["emp_over"]))
        if o["over_price"] is not None and o["under_price"] is not None:
            try:
                o["mkt_over"] = devig_two_way(
                    american_to_implied_prob(o["over_price"]),
                    american_to_implied_prob(o["under_price"]))[0]
            except Exception:
                o["mkt_over"] = None
        else:
            o["mkt_over"] = None
        rows.append(o)
    if len(rows) < 200:
        print(f"  Only {len(rows)} usable obs (<200) — too thin.")
        return
    rows.sort(key=lambda r: (r["season"], r["week"]))

    # ── 1) PREDICTION: opportunity binomial vs incumbent A, OOS + per season ──
    split = len(rows) // 2
    tr, te = rows[:split], rows[split:]
    print("\n  1) PREDICTION (Brier, lower better) — opportunity binomial vs incumbent A:")
    print(f"     single split (train {len(tr)} / test {len(te)}):  "
          f"incumbent A={_brier(te, lambda o: o['p_inc']):.4f}   "
          f"opportunity={_brier(te, lambda o: o['p_opp']):.4f}   "
          f"market={_brier([o for o in te if o['mkt_over'] is not None], lambda o: o['mkt_over']) or float('nan'):.4f}")
    print("     per-season (all obs):")
    for s in meta["seasons"]:
        ss = [o for o in rows if o["season"] == s]
        if len(ss) < 50:
            continue
        mk = [o for o in ss if o["mkt_over"] is not None]
        print(f"       {s}: n={len(ss):>5}  A={_brier(ss, lambda o: o['p_inc']):.4f}  "
              f"opp={_brier(ss, lambda o: o['p_opp']):.4f}  "
              f"mkt={(_brier(mk, lambda o: o['mkt_over']) if mk else float('nan')):.4f}")

    # ── 2) EDGE: model P(over) vs de-vigged market → ROI by model-edge bucket ──
    priced = [o for o in rows if o["mkt_over"] is not None
              and o["over_price"] is not None and o["under_price"] is not None]
    print(f"\n  2) EDGE — opportunity model vs market ({len(priced)} priced obs).")
    print("     model_edge = p_opp − mkt_over; bet the model's side at the DK price.")
    print("     realized−implied is price-robust; ROI is at DK price (vig in).")

    def _edge_report(sub, label):
        if not sub:
            print(f"     {label:<22} (empty)")
            return
        # bet OVER when p_opp>mkt_over by the threshold, UNDER when below
        roi = []
        realized_side = []
        implied_side = []
        for o in sub:
            if o["p_opp"] >= o["mkt_over"]:        # model likes OVER
                win = o["y"] == 1
                px = o["over_price"]
                realized_side.append(o["y"])
                implied_side.append(o["mkt_over"])
            else:                                  # model likes UNDER
                win = o["y"] == 0
                px = o["under_price"]
                realized_side.append(1 - o["y"])
                implied_side.append(1 - o["mkt_over"])
            d = american_to_decimal(px)
            roi.append((d - 1.0) if win else -1.0)
        n = len(roi)
        edge = sum(realized_side) / n - sum(implied_side) / n
        print(f"     {label:<22} n={n:>5}  win={sum(1 for r in roi if r>0)/n*100:4.1f}%  "
              f"ROI={sum(roi)/n*100:+6.2f}%  realized−implied={edge*100:+5.2f}%")

    # bucket by |model_edge|
    for lo, hi, lbl in [(0.0, 0.03, "|edge| 0-3%"), (0.03, 0.06, "|edge| 3-6%"),
                        (0.06, 0.10, "|edge| 6-10%"), (0.10, 1.0, "|edge| >=10%")]:
        _edge_report([o for o in priced if lo <= abs(o["p_opp"] - o["mkt_over"]) < hi],
                     lbl)
    print("\n     top-edge (|edge|>=6%) by season (replication):")
    big = [o for o in priced if abs(o["p_opp"] - o["mkt_over"]) >= 0.06]
    for s in meta["seasons"]:
        _edge_report([o for o in big if o["season"] == s], f"  {s}")
    print("=" * 92)
    print("  READ: opportunity beats A on Brier ⇒ better prediction (keep). Positive")
    print("  realized−implied + ROI in the high-edge bucket, replicated ⇒ a real edge")
    print("  (validate next at true DK/FD execution). Flat/negative ⇒ market efficient.")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--snapshot", default="closing", choices=["closing", "early_4h"])
    ap.add_argument("--book", default="draftkings")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    probe(seasons, snapshot=args.snapshot, book=args.book)


if __name__ == "__main__":
    main()
