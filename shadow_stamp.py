"""shadow_stamp.py — Commit C Phase 3 read-only shadow comparison.

Resolves recent REAL MLB prop players through BOTH identity id-cores — the current
SFBB-only stamp (``ODI_MLB_STAMP_RESOLVER`` OFF) and the game-context/role-verified
resolver (gate ON) — and reports how the stamped MLBAM id WOULD CHANGE if the gate
is flipped. This must be run BEFORE the flip because grading reads the stamp with no
read-side kill switch, so a flip (and the later legacy re-stamp) changes grading
immediately.

NON-DESTRUCTIVE: it calls the two id-cores DIRECTLY (the exact code the envelope
selects under each gate state), so there is NO game_pk derivation, NO player_alias
write, and NO prediction/grading mutation. Inputs come from the odds_line warehouse
(the same real slates the calibration refit harvests).

Categories per unique (player, role, matchup, season):
  agree     old == new (both non-null)   — no change on the flip
  gain      old null  -> new non-null     — the resolver pins what SFBB-only couldn't
  loss      old non-null -> new null      — the resolver REFUSES where SFBB guessed
                                            (fail-closed; review these)
  changed   old != new (both non-null)    — namesake / drift correction (review these)
  both_null neither resolves              — the residual unresolved rate

CAVEAT: historical rows carry no posted lineup / probables, so the NEW path exercises
only tier 2 (statsapi season roster unique-exact) + tier 3 (role-checked SFBB). Tier 1
(today's posted game) is a LIVE-only gain not measurable here — so real-world 'gain'
and namesake-safety are AT LEAST what this reports.
"""

import argparse
from collections import defaultdict
from datetime import date, timedelta

import db_store


def _season_of(row):
    for k in ("game_date", "commence_time"):
        v = row.get(k)
        if v:
            try:
                return int(str(v)[:4])
            except (TypeError, ValueError):
                pass
    return None


def _old_id(name, home, away):
    """The gate-OFF id-core: SFBB bare name (globally unique) else two-team hint."""
    import player_id_map
    mid = player_id_map.mlb_id_for_name(name, teams=None)
    if not mid:
        mid = player_id_map.mlb_id_for_name(name, teams=[home, away])
    return str(mid) if mid else None


def _new_id(name, prop_key, home, away, season):
    """The gate-ON id-core: the game-context/role-verified resolver (no lineup /
    probables for a historical row -> tiers 2+3 only)."""
    import mlb_starters
    found = mlb_starters.resolve_mlbam_id(
        name, season, prop_key=prop_key, teams=[home, away],
        confirmed_lineup=None, probable_starters=None)
    return str(found[0]) if found else None


def shadow(sport_key="baseball_mlb", limit=None, days=None):
    """Return (summary_counts, changed_samples, loss_samples, n_unique)."""
    import warehouse
    rows = warehouse.load_prop_lines(sport_key)
    if days is not None:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        rows = [r for r in rows if (r.get("game_date") or "") >= cutoff]

    # Dedup to unique RESOLUTION inputs — resolution depends only on
    # (name, role, home, away, season); many lines share one.
    seen = {}
    for r in rows:
        name = r.get("player")
        home = r.get("home_team")
        away = r.get("away_team")
        prop_key = r.get("prop_key")
        season = _season_of(r)
        if not (name and season):
            continue
        role = "P" if str(prop_key or "").startswith("pitcher_") else "B"
        key = (name, role, home, away, season)
        seen.setdefault(key, (name, prop_key, home, away, season))

    items = list(seen.values())
    if limit is not None:
        items = items[:limit]

    cats = defaultdict(int)
    changed, loss = [], []
    for (name, prop_key, home, away, season) in items:
        old = _old_id(name, home, away)
        new = _new_id(name, prop_key, home, away, season)
        matchup = f"{away} @ {home}"
        if old is None and new is None:
            cats["both_null"] += 1
        elif old is None:
            cats["gain"] += 1
        elif new is None:
            cats["loss"] += 1
            loss.append((name, prop_key, matchup, season, old))
        elif old == new:
            cats["agree"] += 1
        else:
            cats["changed"] += 1
            changed.append((name, prop_key, matchup, season, old, new))
    return cats, changed, loss, len(items)


def _print_report(cats, changed, loss, n_unique, n_samples):
    print(f"\n=== Commit C stamp-resolver SHADOW (unique resolution inputs: "
          f"{n_unique:,}) ===")
    if not n_unique:
        print("  No warehouse prop rows to compare (SQL off, empty, or filtered).")
        return
    order = ["agree", "changed", "gain", "loss", "both_null"]
    for cat in order:
        n = cats.get(cat, 0)
        print(f"  {cat:9s} {n:6,d}  ({100.0 * n / n_unique:5.1f}%)")
    # The two categories that CHANGE grading on the flip — eyeball for correctness.
    if changed:
        print(f"\n  CHANGED (namesake/drift corrections) — first {n_samples}:")
        for name, pk, matchup, season, old, new in changed[:n_samples]:
            print(f"    {name} [{pk}] {matchup} {season}: {old} -> {new}")
    if loss:
        print(f"\n  LOSS (resolver refuses where SFBB guessed) — first {n_samples}:")
        for name, pk, matchup, season, old in loss[:n_samples]:
            print(f"    {name} [{pk}] {matchup} {season}: {old} -> None")
    print("\n  changed + loss are the only rows whose stamp/grading move on a flip; "
          "gain is pure upside; agree/both_null are unchanged.")


def _main():
    from cli_encoding import configure_stdio
    configure_stdio()
    ap = argparse.ArgumentParser(
        description="Commit C Phase 3: read-only shadow of the stamp-resolver flip.")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap unique inputs resolved (default: all).")
    ap.add_argument("--days", type=int, default=None,
                    help="Only rows with game_date within the last N days.")
    ap.add_argument("--samples", type=int, default=25,
                    help="How many changed/loss rows to print (default 25).")
    args = ap.parse_args()

    db_store.promote_secrets_from_toml()
    if not db_store.enabled():
        raise SystemExit("SQL is not configured (SQL_* secrets) — nothing to compare.")
    cats, changed, loss, n = shadow(args.sport, limit=args.limit, days=args.days)
    _print_report(cats, changed, loss, n, args.samples)


if __name__ == "__main__":
    _main()
