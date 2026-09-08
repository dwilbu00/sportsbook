"""test_nfl_epa.py — leakage + integrity guards for the NFL EPA layer.

No Azure parity net exists for NFL, so these are the safety rail: the as-of
aggregation must NEVER count a play from the target game or later (leakage would
silently inflate backtest edge), and the team-abbreviation map must stay in lock-
step with the mirror + spine (a drifted abbr = silently dropped team = wrong
ratings). Runs off the committed mirror parquets; skips cleanly if absent.

Run:  python test_nfl_epa.py            (standalone, prints PASS/FAIL)
      pytest test_nfl_epa.py            (also works)
"""
import nfl_epa

SEASON = 2024
AS_OF = "2024-11-01"     # a mid-season Friday; games exist on both sides


def _plays_or_skip():
    plays = nfl_epa.load_plays(SEASON)
    if not plays:
        print(f"[SKIP] no pbp mirror for {SEASON} — nothing to test")
        return None
    return plays


def test_leakage_asof():
    """team_epa(as_of) must count exactly the plays strictly before as_of — no
    play from on/after the cutoff may contribute to any team's rating."""
    plays = _plays_or_skip()
    if plays is None:
        return True
    ratings = nfl_epa.team_epa(SEASON, as_of_date=AS_OF, prior_shrink=False)
    # ground truth computed independently from the raw play list
    off_n = {}
    for p in plays:
        if (p["game_date"] or "") >= AS_OF:
            continue
        off_n[p["posteam"]] = off_n.get(p["posteam"], 0) + 1
    ok = True
    for t, exp in off_n.items():
        got = ratings.get(t, {}).get("off_plays", 0)
        if got != exp:
            print(f"  [FAIL] {t}: off_plays as_of={got} but expected {exp}")
            ok = False
    # and the converse: NO play on/after the cutoff leaked in (total play count)
    counted = sum(v.get("off_plays", 0) for v in ratings.values())
    expected_total = sum(off_n.values())
    if counted != expected_total:
        print(f"  [FAIL] total off_plays as_of={counted} expected {expected_total} "
              f"(leak of {counted - expected_total} plays)")
        ok = False
    # sanity: the cutoff actually excluded a non-trivial chunk (else the test is vacuous)
    full = sum(1 for p in plays)
    if expected_total >= full:
        print(f"  [FAIL] as_of excluded nothing ({expected_total}>={full}) — test vacuous")
        ok = False
    print(f"  [{'PASS' if ok else 'FAIL'}] leakage as_of {AS_OF}: "
          f"{expected_total}/{full} plays before cutoff, ratings match")
    return ok


def test_abbr_parity():
    """The canonical abbr map must exactly cover the teams present in the pbp
    mirror (a drifted/extra/missing abbr = a silently mis-rated team)."""
    plays = _plays_or_skip()
    if plays is None:
        return True
    canon = set(nfl_epa.TEAM_ABBR_TO_NAME)
    seen = set()
    for p in plays:
        seen.add(p["posteam"])
        seen.add(p["defteam"])
    seen.discard(None)
    extra = seen - canon      # in pbp but not mapped → would KeyError / drop
    missing = canon - seen    # mapped but never seen this season (informational)
    ok = not extra
    print(f"  [{'PASS' if ok else 'FAIL'}] abbr parity: pbp teams ⊆ canonical "
          f"(unmapped in pbp: {sorted(extra) or 'none'}; "
          f"mapped-but-absent: {sorted(missing) or 'none'})")
    return ok


def test_spine_parity():
    """Canonical abbr map must equal the spine's team set (both directions) — the
    join map that resolves odds→game_id shares this alphabet."""
    try:
        import nfl_schedule
        games = nfl_schedule.load_games([str(SEASON)])
    except Exception as exc:
        print(f"  [SKIP] spine unavailable: {type(exc).__name__}")
        return True
    if not games:
        print("  [SKIP] no spine games")
        return True
    canon = set(nfl_epa.TEAM_ABBR_TO_NAME)
    spine = set()
    for g in games:
        spine.add(g.get("home_team"))
        spine.add(g.get("away_team"))
    spine.discard(None)
    extra, missing = spine - canon, canon - spine
    ok = not extra and not missing
    print(f"  [{'PASS' if ok else 'FAIL'}] spine parity: "
          f"spine-not-canon: {sorted(extra) or 'none'}; "
          f"canon-not-spine: {sorted(missing) or 'none'}")
    return ok


def test_mirror_is_source():
    """load_plays should be served by the mirror (dep-free), not a live download,
    when the parquet is present."""
    m = nfl_epa._plays_from_mirror(SEASON)
    ok = m is not None and len(m) > 1000
    print(f"  [{'PASS' if ok else 'SKIP'}] mirror is the source: "
          f"{len(m) if m else 0} plays from parquet")
    return ok or m is None


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    print("=== nfl_epa leakage + integrity tests ===")
    results = [
        test_mirror_is_source(),
        test_leakage_asof(),
        test_abbr_parity(),
        test_spine_parity(),
    ]
    ok = all(results)
    print(f"=== {'ALL PASS' if ok else 'FAIL — see above'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
