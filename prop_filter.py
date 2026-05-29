"""
Player-prop reliability filter (streak-based).

Two notions of "valid" sit on top of each other:

  ── A game is **valid** only if it sits inside a *consecutive run* of
     ≥ `min_streak` played games where every game is itself valid (not
     low-minutes, not a pre-layoff game) AND no layoff (≥ missed_thresh
     team games missed) interrupts the run.

  ── Predictions for the upcoming game are **paused** until the player's
     *current open run* of consecutive valid games reaches `min_streak`.
     Games accumulated during the rebuild become valid retroactively once
     the run hits `min_streak`; until then they are not used.

The default `min_streak` is `max(STREAK_FLOOR, (half_life or 0) + 1)`. With
NBA `half_life=7` that's 8 — i.e., the player needs 8 consecutive, healthy,
no-layoff games before we trust their projection again.

A "break" in the run is any of:
  • A low-minutes game (MIN < max(MIN_FLOOR, MIN_FRACTION × median MIN)).
  • The game played immediately BEFORE a layoff (treated as the
    likely injury / suspension trigger).
  • A layoff itself (≥ missed_thresh team games missed; sport-specific,
    with an adaptive widening for naturally low-cadence players such as
    MLB starting pitchers).
"""
import bisect

# Sport-specific "missed games" gap that qualifies as a layoff.
MISSED_GAMES_THRESHOLDS = {
    "basketball_nba": 3,
    "baseball_mlb": 3,
    "americanfootball_nfl": 2,
    "icehockey_nhl": 3,
}
DEFAULT_MISSED_GAMES_THRESHOLD = 3

# Number of games BEFORE a layoff to invalidate (likely injury/suspension game).
PRE_LAYOFF_N = 1

# Low-minutes filter knobs.
MIN_PLAYED_FLOOR = 10.0
MIN_FRACTION = 0.5

# Absolute floor on the streak threshold when half_life is None / 0.
STREAK_FLOOR = 5


def _missed_games_threshold(sport_key):
    return MISSED_GAMES_THRESHOLDS.get(sport_key, DEFAULT_MISSED_GAMES_THRESHOLD)


def _missed_count_between(sched_dates_sorted, prev_date, curr_date):
    """Number of team-schedule dates strictly between prev_date and curr_date."""
    lo = bisect.bisect_right(sched_dates_sorted, prev_date)
    hi = bisect.bisect_left(sched_dates_sorted, curr_date)
    return max(0, hi - lo)


def _median(sorted_vals):
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    mid = n // 2
    return sorted_vals[mid] if n % 2 else 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _extract_schedule_dates(team_schedule):
    if not team_schedule:
        return []
    out = []
    for s in team_schedule:
        d = s.get("date") or s.get("gameDate") or s.get("game_date")
        if d:
            out.append(d)
    out.sort()
    return out


def resolve_min_streak(half_life, override=None):
    """Compute the streak threshold from a variant's half_life."""
    if override is not None:
        return max(1, int(override))
    return max(STREAK_FLOOR, (int(half_life) if half_life else 0) + 1)


def filter_player_gamelog(gamelog, team_schedule, sport_key, half_life=None,
                          pre_layoff_n=PRE_LAYOFF_N,
                          min_floor=MIN_PLAYED_FLOOR, min_fraction=MIN_FRACTION,
                          min_streak=None):
    """
    Apply the streak-based reliability filter.

    Returns:
        {
          "eligible_games": list of game dicts (newest-first, like input),
          "skip_prediction": bool,
          "skip_reason": str | None,
          "current_streak": int,        # length of the player's open valid run
          "min_streak": int,            # threshold required to be valid
          "n_excluded_low_min": int,
          "n_excluded_pre_layoff": int,
          "n_excluded_short_streak": int,
          "median_min": float,
          "min_threshold": float | None,
        }

    gamelog: list of dicts (any ordering) with 'game_date' (ISO str). For
        the MIN-based filter to engage, dicts should also have 'MIN'.
    team_schedule: iterable of dicts with 'date' / 'gameDate' / 'game_date'.
        None disables layoff detection (only low-min remains, plus the
        sport-independent streak threshold).
    half_life: int or None. Used to derive min_streak when min_streak is None.
    min_streak: optional explicit override; otherwise computed from half_life.
    """
    threshold = resolve_min_streak(half_life, override=min_streak)

    empty_result = {
        "eligible_games": [],
        "skip_prediction": True,
        "skip_reason": "empty_gamelog",
        "current_streak": 0,
        "min_streak": threshold,
        "n_excluded_low_min": 0,
        "n_excluded_pre_layoff": 0,
        "n_excluded_post_layoff": 0,
        "n_excluded_short_streak": 0,
        "median_min": 0.0,
        "min_threshold": None,
    }
    if not gamelog:
        return empty_result

    # Sort chronologically (oldest first) for layoff / streak detection.
    chrono = sorted(gamelog, key=lambda g: g.get("game_date") or "")
    n = len(chrono)

    # ── Low-min threshold from median of played minutes (MIN > 0). ──
    played_mins = sorted(
        (g.get("MIN") or 0.0) for g in chrono if (g.get("MIN") or 0.0) > 0
    )
    median_min = _median(played_mins)
    min_threshold = (max(min_floor, min_fraction * median_min)
                     if median_min > 0 else None)

    sched_dates = _extract_schedule_dates(team_schedule)
    base_missed_thresh = _missed_games_threshold(sport_key)

    # ── Adaptive missed-games threshold (handles naturally low-cadence
    # players such as MLB starting pitchers). The threshold widens to at
    # least 2× the player's own median gap (in team games), floored at the
    # sport-level default.
    missed_thresh = base_missed_thresh
    layoff_event_indices = set()
    if sched_dates and n >= 2:
        gaps = []
        prev_d = None
        for g in chrono:
            d = g.get("game_date")
            if prev_d and d:
                gaps.append(_missed_count_between(sched_dates, prev_d, d) + 1)
            prev_d = d
        if gaps:
            median_gap = _median(sorted(gaps))
            if median_gap > 1:
                missed_thresh = max(base_missed_thresh, int(2 * median_gap))
        # Detect layoff events (indices in `chrono` of the first game played
        # AFTER a layoff: the player missed ≥ missed_thresh team games).
        prev_date = None
        for i, g in enumerate(chrono):
            d = g.get("game_date")
            if prev_date and d:
                if _missed_count_between(sched_dates, prev_date, d) >= missed_thresh:
                    layoff_event_indices.add(i)
            prev_date = d

    # ── Per-game flags ──
    # Two distinct notions:
    #
    #   is_invalid:    low-minutes OR pre-layoff. These BREAK the streak
    #                  counter — a streak can't span an invalid game.
    #
    #   is_post_layoff: the 1st played game after a layoff. This does NOT
    #                  break the streak (it actually starts a new one), and
    #                  it COUNTS toward the streak-length requirement. But
    #                  it is permanently excluded from the projection model
    #                  per user policy — the 1st-back game is unreliable,
    #                  even though we let it qualify the player to resume
    #                  predictions.
    is_low_min = [False] * n
    is_pre_layoff = [False] * n
    is_post_layoff = [i in layoff_event_indices for i in range(n)]
    for L in layoff_event_indices:
        for k in range(1, pre_layoff_n + 1):
            if L - k >= 0:
                is_pre_layoff[L - k] = True
    if min_threshold is not None:
        for i, g in enumerate(chrono):
            if (g.get("MIN") or 0.0) < min_threshold:
                is_low_min[i] = True

    is_invalid = [is_low_min[i] or is_pre_layoff[i] for i in range(n)]

    # ── Compute streak length each game belongs to ──
    # A streak is a maximal run of consecutive valid games with no layoff
    # break in the middle. Invalid games have streak length 0.
    streak_len = [0] * n
    i = 0
    while i < n:
        if is_invalid[i]:
            i += 1
            continue
        j = i + 1
        while j < n:
            if is_invalid[j]:
                break
            if j in layoff_event_indices:
                break  # layoff just before j → new streak begins at j
            j += 1
        length = j - i
        for k in range(i, j):
            streak_len[k] = length
        i = j

    # ── Eligibility ──
    # A game is eligible iff:
    #   - streak length ≥ threshold (run was long enough to trust), AND
    #   - the game itself is not low-min / pre-layoff (is_invalid), AND
    #   - the game is not a 1st-game-back (is_post_layoff) — those count
    #     toward the streak length but are permanently dropped from the model.
    n_low_min = sum(is_low_min)
    n_pre_layoff = sum(is_pre_layoff)
    n_post_layoff = sum(is_post_layoff)
    n_short_streak = 0
    eligible_chrono = []
    for i, g in enumerate(chrono):
        if is_invalid[i]:
            continue
        if streak_len[i] < threshold:
            n_short_streak += 1
            continue
        if is_post_layoff[i]:
            continue  # 1st-back: counts toward streak length, not used
        eligible_chrono.append(g)

    # ── Current open streak (the run containing the last played game). ──
    last_idx = n - 1
    if is_invalid[last_idx]:
        current_streak = 0
    else:
        current_streak = streak_len[last_idx]

    # ── Skip prediction policy ──
    skip_reason = None
    last_game = chrono[last_idx]
    last_min = last_game.get("MIN") or 0.0
    if is_low_min[last_idx]:
        skip_reason = (f"last_game_low_min "
                       f"(MIN={last_min:.0f} < {min_threshold:.0f})")
    elif is_pre_layoff[last_idx]:
        skip_reason = "last_game_pre_layoff_invalid"
    elif current_streak < threshold:
        skip_reason = (f"current_streak_too_short "
                       f"({current_streak} < {threshold})")

    return {
        "eligible_games": list(reversed(eligible_chrono)),  # newest-first
        "skip_prediction": skip_reason is not None,
        "skip_reason": skip_reason,
        "current_streak": current_streak,
        "min_streak": threshold,
        "n_excluded_low_min": n_low_min,
        "n_excluded_pre_layoff": n_pre_layoff,
        "n_excluded_post_layoff": n_post_layoff,
        "n_excluded_short_streak": n_short_streak,
        "median_min": median_min,
        "min_threshold": min_threshold,
    }
