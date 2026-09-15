"""nfl_bonus_optimizer.py — surface the +EV plays for the CURRENTLY ACTIVE bonuses (Stage A).

Consumes the active promos (bonus.LIVE_BONUSES or a config) + the current DK/FD prop board and
returns concrete +EV plays, honest by construction:

  * LEG PROBABILITY = the book's DE-VIGGED price (calibrated; our model is over-confident in the
    high-P tail — proven in nfl_bonus_backtest). Our MODEL is used ONLY as the eligibility gate:
    a leg is allowed only for a TRUSTWORTHY prop above its opportunity threshold.
  * CROSS-GAME parlays / singles: legs independent -> joint = product of marginals -> full boosted
    EV + fractional-Kelly stake (respecting the bonus's max/min wager). Fully priced from leg odds.
  * SGP (same-game): joint uses the CORRELATION copula (nfl_sgp_correlation). Because a book prices
    an SGP as ONE combined number (its own generic correlation haircut), we can't take the product;
    so for each recommended same-side +corr STACK we report our copula joint P and the REQUIRED
    combined price for +EV under the boost -> compare to the book's SGP builder price live.

Offline (this CLI): reads the local ladder extract for a historical --season/--week to demonstrate
and sanity-check the output. Live: swap the line source to the Odds API board (same leg dict).
Diagnostic — spends nothing.
"""
import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np

import nfl_schedule
import nfl_props_scan as scan
import nfl_props_accuracy as acc
import nfl_ladder_clv as clv
import bonus as bonuslib
import nfl_sgp_correlation as sgp
import nfl_opportunity_serving as srv
from odds_client import american_to_decimal, american_to_implied_prob, devig_two_way

TRUSTWORTHY = sgp.TRUSTWORTHY               # {prop: opp_threshold}
_CUR_SLATE_WEEK = 99                        # sentinel: "all completed weeks are prior" (upcoming)
PROP_ABBR = sgp.PROP_ABBR
MAX_PARLAY = 3       # cap ticket size we enumerate
TOP_N = 12           # top legs (by P) per book to consider
TOP_K = 8            # plays to surface per bonus


def collect_week_legs(train_seasons, season, week, books):
    """{book: [leg]} for one slate. leg carries the FAVORITE side + that side's price + de-vig P
    + team/prop for SGP categorization. (Offline source = local ladder extract, DK/FD closing.)"""
    tmap = sgp._team_map([season])
    idx = nfl_schedule.game_index([str(season)])
    out = {b: [] for b in books}
    for prop, tmin in TRUSTWORTHY.items():
        cfg = acc.PROPS[prop]
        hl, mp, k = acc.SWEPT.get(prop, (4, 3, 12))
        acc.HALF_LIFE, acc.MIN_PRIOR, acc.SHRINK_K = hl, mp, k
        tr = acc._obs_from_series(acc._series(train_seasons), cfg)
        if len(tr) < 200:
            continue
        comp = acc.fit_components(tr, cfg)
        feat = {(o["name"], o["season"], o["week"]): o
                for o in acc._obs_from_series(acc._series([season]), cfg)}
        lines, meta = clv._load_local(prop, [season])
        for (eid, player), bybook in lines.items():
            m = meta[(eid, player)]
            gid, _ = nfl_schedule.resolve_event(m["home"], m["away"], m["commence"], index=idx)
            if gid is None:
                continue
            ps = gid.split("_")
            if int(ps[1]) != week:
                continue
            o = feat.get((scan._norm(player), ps[0], int(ps[1])))
            if o is None or o.get("exp_vol", o.get("mean_base", 0.0)) < tmin:
                continue
            tm, opp, pos = tmap.get((scan._norm(player), ps[0], week), (None, None, None))
            for book in books:
                d = bybook.get(book, {}).get("closing")
                if not d:
                    continue
                over_q, under_q = d.get("OVER"), d.get("UNDER")
                if not (over_q and under_q):
                    continue
                line = over_q[0]
                if line is None:
                    continue
                try:
                    fair = devig_two_way(american_to_implied_prob(over_q[1]),
                                         american_to_implied_prob(under_q[1]))[0]
                except Exception:
                    continue
                fav_over = fair >= 0.5
                out[book].append({
                    "gid": gid, "prop": prop, "player": player, "team": tm, "opp": opp, "pos": pos,
                    "line": line, "side": "OVER" if fav_over else "UNDER",
                    "P": fair if fav_over else 1.0 - fair, "fair_over": fav_over,
                    "t_over": sgp._ppf(1.0 - fair),
                    "odds": (over_q if fav_over else under_q)[1]})
    return out


def _season_of(commence_time):
    """NFL season for an ISO commence time (season spans Sep→Feb → Jan/Feb belong to prior yr)."""
    try:
        y, m = int(commence_time[:4]), int(commence_time[5:7])
        return y if m >= 8 else y - 1
    except Exception:
        return None


def legs_from_board(parsed_boards, book, season=None, week=_CUR_SLATE_WEEK):
    """LIVE leg source: map the app's parsed prop board(s) → optimizer leg dicts, applying the
    opportunity gate (§serving feed) and using MARKET de-vig P + the book's own price.

    parsed_boards: list of `odds_client.parse_player_props` results (one per event).
    book: 'draftkings' | 'fanduel' (its bonus uses its price; leg P is book-agnostic de-vig).
    Returns the same leg dict shape as the offline `collect_week_legs`.
    """
    pxkey = "dk" if book == "draftkings" else "fd"
    legs = []
    for board in parsed_boards:
        gid = board.get("game_id")
        home, away = board.get("home_team"), board.get("away_team")
        seas = season or _season_of(board.get("commence_time") or "")
        if seas is None:
            continue
        for prop in TRUSTWORTHY:
            for player, p in board.get("props", {}).get(prop, {}).items():
                pn = scan._norm(player)
                if not srv.passes_gate(pn, seas, week, prop):
                    continue
                fair = p.get("over_implied")
                line = p.get("line")
                if fair is None or line is None:
                    continue
                fav_over = fair >= 0.5
                price = p.get(f"{pxkey}_{'over' if fav_over else 'under'}_price")
                if price is None:                      # this book doesn't post the fav side
                    continue
                team = srv.player_team(pn, seas)
                opp = away if team == home else (home if team == away else None)
                legs.append({
                    "gid": gid, "prop": prop, "player": player, "team": team, "opp": opp,
                    "line": line, "side": "OVER" if fav_over else "UNDER",
                    "P": fair if fav_over else 1.0 - fair, "fair_over": fav_over,
                    "t_over": sgp._ppf(1.0 - fair), "odds": price})
    return legs


def _leg_ok(lg, bonus):
    return american_to_decimal(lg["odds"]) >= bonuslib.american_to_dec(bonus.min_odds_leg) - 1e-9


def _label(lg):
    return f"{lg['player'][:18]:<18} {PROP_ABBR[lg['prop']]}{lg['side'][0]} {lg['line']:>4.1f} @{lg['odds']:>+5.0f}"


def _legkey(lg):
    return (lg["gid"], lg["player"], lg["prop"], lg["side"])


def _diversify(plays, k):
    """Greedy non-overlapping portfolio: highest-EV first, skip any play reusing a leg already
    taken. Turns a pile of permutations into distinct tickets you'd actually field."""
    used, out = set(), []
    for item in plays:
        combo = item[-1]
        keys = {_legkey(l) for l in combo}
        if keys & used:
            continue
        used |= keys
        out.append(item)
        if len(out) >= k:
            break
    return out


def cross_game_plays(legs, bonus, bankroll):
    """Concrete +EV cross-game parlays/singles ranked by boosted EV (independence joint)."""
    elig = sorted([l for l in legs if _leg_ok(l, bonus)], key=lambda x: -x["P"])[:TOP_N]
    sizes = range(1, MAX_PARLAY + 1) if bonus.bet_type in ("single", "any") else range(2, MAX_PARLAY + 1)
    sizes = [s for s in sizes if s >= max(1, bonus.min_legs)]
    plays = []
    for K in sizes:
        for combo in combinations(elig, K):
            if len({l["gid"] for l in combo}) != K:        # cross-game only
                continue
            r = bonuslib.evaluate([(l["P"], l["odds"]) for l in combo], bonus, bankroll=bankroll)
            if r["qualifies"] and r["boosted_ev_pct"] > 0:
                plays.append((r["boosted_ev_pct"], r, combo))
    plays.sort(key=lambda x: -x[0])
    return plays


def sgp_stacks(legs, bonus, rho, bankroll):
    """Per-game same-side +corr STACKS for an SGP bonus: copula joint P + REQUIRED combined price.
    (The book prices the SGP as one number; compare its builder price to 'need >=' live.)"""
    bygame = defaultdict(list)
    for l in legs:
        if _leg_ok(l, bonus):
            bygame[l["gid"]].append(l)
    rng = np.random.default_rng(0)
    out = []
    need = max(2, bonus.min_legs)
    for gid, gl in bygame.items():
        if len(gl) < need:
            continue
        gl = sorted(gl, key=lambda x: -x["P"])[:TOP_N]
        best = None
        for combo in combinations(gl, need):
            # prefer positively-correlated same-side content; skip strong script-conflict stacks
            rhos = [sgp._rho_val(rho.get(sgp.category(a, b))) for a, b in combinations(combo, 2)]
            if min(rhos) <= -0.30:            # avoid same-team RB-committee / pass-rush conflicts
                continue
            jp = sgp.joint_prob(list(combo), rho, [l["fair_over"] for l in combo], rng)
            score = jp * (1.0 + max(rhos))    # favor higher joint + a +corr pair
            if best is None or score > best[0]:
                best = (score, jp, combo, max(rhos))
        if best is None:
            continue
        _s, jp, combo, mx = best
        # required combined decimal for +EV under the boost: jp*(D-1)*(1+b) > (1-jp)
        b = bonus.boost_pct
        req_dec = 1.0 + (1.0 - jp) / (jp * (1.0 + b)) if jp > 0 else float("inf")
        out.append((jp, mx, combo, bonuslib.dec_to_american(req_dec)))
    out.sort(key=lambda x: -x[0])
    return out


def load_rho():
    """Frozen SGP correlations (calibration/nfl_sgp_correlations.json); {} if absent."""
    return sgp.load_frozen() or {}


def evaluate_slate(legs_by_book, bonuses, rho, bankroll):
    """Structured optimizer output for a slate — the shared core for the CLI and the app.
    Returns [{book, label, bet_type, n_legs, cross:[(ev,r,combo)], sgp:[(jp,mx,combo,need)]}]."""
    out = []
    for base in bonuses:
        for book, legs in legs_by_book.items():
            bonus = bonuslib.Bonus(**{**base.__dict__, "book": book})
            cross = [] if bonus.bet_type == "sgp" else _diversify(
                cross_game_plays(legs, bonus, bankroll), TOP_K)
            stacks = (sgp_stacks(legs, bonus, rho, bankroll)[:TOP_K]
                      if bonus.bet_type in ("sgp", "parlay", "any") else [])
            out.append({"book": book, "label": base.label, "bet_type": bonus.bet_type,
                        "boost_pct": bonus.boost_pct, "max_wager": bonus.max_wager,
                        "n_legs": len(legs), "cross": cross, "sgp": stacks})
    return out


def run(season, week, books, bankroll):
    train = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
    legs_by_book = collect_week_legs(train, str(season), week, books)
    rho = load_rho() or sgp.fit_correlations(sgp.collect_game_legs(train, ["2023", "2024"]))

    print("=" * 100)
    print(f"  NFL BONUS OPTIMIZER — +EV plays for active promos, {season} week {week}")
    print("  Leg P = book de-vig (calibrated); model = eligibility gate only (trustworthy high-opp).")
    print(f"  bankroll=${bankroll:.0f}, fractional-Kelly stakes (capped by each bonus's max wager).")
    print("=" * 100)
    for base in bonuslib.LIVE_BONUSES:
        for book in books:
            bonus = bonuslib.Bonus(**{**base.__dict__, "book": book})
            legs = legs_by_book.get(book, [])
            print(f"\n  [{book}] {base.label}   ({len(legs)} eligible legs)")
            # cross-game parlays / singles (fully priced) — NOT for SGP-only boosts (must be same-game)
            if bonus.bet_type == "sgp":
                print("    CROSS-GAME / SINGLE: n/a (this boost applies to SAME-GAME parlays only).")
            else:
                plays = cross_game_plays(legs, bonus, bankroll)
                if plays:
                    print(f"    CROSS-GAME / SINGLE (+EV, priced; non-overlapping portfolio):")
                    for ev, r, combo in _diversify(plays, TOP_K):
                        tag = "single" if len(combo) == 1 else f"{len(combo)}-leg"
                        print(f"      EV {ev:>+6.1f}%  stake ${r['kelly_stake']:>5.2f}  {tag} "
                              f"@{r['combined_american']:>+6.0f}  P={r['joint_P']*100:>4.1f}%")
                        for l in combo:
                            print(f"          - {_label(l)}")
                else:
                    print("    CROSS-GAME / SINGLE: none clear +EV at qualifying prices.")
            # SGP stacks (need the book's combined price live)
            if bonus.bet_type in ("sgp", "parlay", "any"):
                stacks = sgp_stacks(legs, bonus, rho, bankroll)
                if stacks:
                    print(f"    SGP STACKS (build in-app; +EV iff book SGP price >= 'need'):")
                    for jp, mx, combo, need_am in stacks[:TOP_K]:
                        gtag = combo[0]["gid"]
                        print(f"      joint P={jp*100:>4.1f}%  maxCorr={mx:+.2f}  need >= {need_am:>+6.0f}"
                              f"  [{gtag}]")
                        for l in combo:
                            print(f"          - {_label(l)} ({l['team']})")
    print("\n  READ: cross-game plays are ready to bet (independent legs, full EV). SGP stacks are")
    print("  correlation-vetted suggestions — the book quotes ONE SGP number; bet only if its price")
    print("  meets 'need'. All leg probabilities are de-vigged market (calibrated), not our model.")
    print("=" * 100)


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2024")
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--book", default="both", choices=["draftkings", "fanduel", "both"])
    ap.add_argument("--bankroll", type=float, default=1000.0)
    args = ap.parse_args()
    books = ["draftkings", "fanduel"] if args.book == "both" else [args.book]
    run(args.season, args.week, books, args.bankroll)


if __name__ == "__main__":
    main()
