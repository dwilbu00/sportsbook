"""Maintain resolved prediction logs (grade outcomes + gated recalibration).

Outcome grading uses ESPN box scores (see recalibration.resolve_pending_outcomes)
and spends no odds-API credits. Recalibration (Platt) is gated by calibration age
and the number of newly resolved observations.

Example:
    python forward_tracker.py --resolve
"""

import argparse

from recalibration import maintain_sport, prediction_log_storage


SPORT_ALIASES = {
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}


def resolve_and_refit(sport_key=None):
    sports = [sport_key] if sport_key else list(SPORT_ALIASES.values())
    return {key: maintain_sport(key) for key in sports}


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve past outcomes and run gated recalibration")
    parser.add_argument("--sport", choices=list(SPORT_ALIASES), default=None,
                        help="Limit work to one sport (default: all logged sports)")
    args = parser.parse_args()
    if not args.resolve:
        parser.error("choose --resolve")

    sport_key = SPORT_ALIASES.get(args.sport) if args.sport else None
    print(f"Prediction log storage: {prediction_log_storage()}")
    for key, result in resolve_and_refit(sport_key).items():
        print(f"{key}: resolved={result['newly_resolved']}, "
              f"refit={result['refit']}")


if __name__ == "__main__":
    _main()
