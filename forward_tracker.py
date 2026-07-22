"""Maintain resolved prediction logs (grade outcomes + gated recalibration).

Outcome grading uses ESPN box scores + MLB statsapi (see
recalibration.resolve_pending_outcomes) and spends no odds-API credits.
Recalibration (Platt) is gated by calibration age and the number of newly
resolved observations. This CLI bypasses the live app's hourly maintenance gate,
so it's the right tool to drain a backlog on demand.

Resolution speed is limited by the per-player ESPN/statsapi fetches, NOT the
storage backend — so the default per-run cap keeps the LIVE app responsive, while
this CLI can pass a high --max-resolve to clear a backlog off the hot path.

Examples:
    python forward_tracker.py --resolve                     # default 80/sport
    python forward_tracker.py --resolve --sport mlb --max-resolve 2000
"""

import argparse

from recalibration import (
    MAX_RESOLVE_PER_LAUNCH,
    maintain_sport,
    prediction_log_storage,
)


SPORT_ALIASES = {
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}


def resolve_and_refit(sport_key=None, max_resolve=MAX_RESOLVE_PER_LAUNCH):
    sports = [sport_key] if sport_key else list(SPORT_ALIASES.values())
    return {key: maintain_sport(key, max_resolve=max_resolve) for key in sports}


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve past outcomes and run gated recalibration")
    parser.add_argument("--sport", choices=list(SPORT_ALIASES), default=None,
                        help="Limit work to one sport (default: all logged sports)")
    parser.add_argument("--max-resolve", type=int, default=MAX_RESOLVE_PER_LAUNCH,
                        help=f"Cap on successful resolutions per sport this run "
                             f"(default {MAX_RESOLVE_PER_LAUNCH}). Use a high value "
                             f"to drain a backlog offline.")
    args = parser.parse_args()
    if not args.resolve:
        parser.error("choose --resolve")

    # Target the SQL backend when the SQL_* secrets are configured (mirrors the
    # app's boot promotion; outside Streamlit these aren't in the env yet). Falls
    # back to Blob/local when SQL isn't configured or db_store is unavailable.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass

    sport_key = SPORT_ALIASES.get(args.sport) if args.sport else None
    print(f"Prediction log storage: {prediction_log_storage()}")
    for key, result in resolve_and_refit(sport_key, args.max_resolve).items():
        print(f"{key}: resolved={result['newly_resolved']}, "
              f"refit={result['refit']}")


if __name__ == "__main__":
    _main()
