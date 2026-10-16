#!/usr/bin/env bash
# Stage the OOS-winning recal map (Platt or isotonic) for every NBA prop into the committed
# seed calibration/recalibration_basketball_nba.json (2026-10-16 overconfidence response).
# Read-only on DB/mirror; the only write is the seed file. Review with `git diff`.
cd "$(dirname "$0")/../.."
for p in player_points player_rebounds player_assists player_threes player_steals \
         player_blocks player_turnovers player_points_rebounds player_points_assists \
         player_points_rebounds_assists; do
  echo "################ $p"
  python refit_calibration.py --sport nba --recalibrate --recal-prop "$p" --save-recal
done
