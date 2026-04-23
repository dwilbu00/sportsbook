"""
Analysis engine for comparing sportsbook odds against historical data.
Identifies value bets where book implied probability < historical probability.
"""

from odds_client import PROP_LABELS


def analyze_moneyline_value(game_odds, home_team_stats, away_team_stats, threshold_pct=5.0):
    """
    Compare moneyline implied probabilities against historical win rates.

    Parameters:
        game_odds (dict): Parsed game odds from odds_client.parse_game_odds()
        home_team_stats (dict): Home team stats with 'season' and 'recent' keys
        away_team_stats (dict): Away team stats with 'season' and 'recent' keys
        threshold_pct (float): Minimum edge (percentage points) to flag as value

    Returns:
        list: Value candidates with edge details
    """
    candidates = []
    threshold = threshold_pct / 100.0

    home_team = game_odds["home_team"]
    away_team = game_odds["away_team"]

    for team_name, stats in [(home_team, home_team_stats), (away_team, away_team_stats)]:
        ml_odds = game_odds["moneyline"].get(team_name, [])
        if not ml_odds:
            continue

        # Average implied probability across all books
        avg_implied = sum(o["implied_prob"] for o in ml_odds) / len(ml_odds)

        # Historical probability: weighted blend of season and recent form
        season_wp = stats["season"]["win_pct"]
        recent_wp = stats["recent"]["win_pct"]
        # Weight recent form more heavily (60/40)
        hist_prob = (0.4 * season_wp) + (0.6 * recent_wp)

        edge = hist_prob - avg_implied

        best_odds = max(ml_odds, key=lambda o: o["implied_prob"] if o["price"] > 0 else -o["price"])
        worst_book_prob = min(o["implied_prob"] for o in ml_odds)
        best_edge = hist_prob - worst_book_prob

        result = {
            "type": "moneyline",
            "team": team_name,
            "opponent": away_team if team_name == home_team else home_team,
            "home_away": "HOME" if team_name == home_team else "AWAY",
            "book_implied_prob": round(avg_implied * 100, 2),
            "season_win_pct": round(season_wp * 100, 2),
            "recent_win_pct": round(recent_wp * 100, 2),
            "hist_prob": round(hist_prob * 100, 2),
            "edge_pct": round(edge * 100, 2),
            "best_edge_pct": round(best_edge * 100, 2),
            "best_book": min(ml_odds, key=lambda o: o["implied_prob"])["book"],
            "best_price": min(ml_odds, key=lambda o: o["implied_prob"])["price"],
            "is_value": edge >= threshold,
        }
        candidates.append(result)

    return candidates


def analyze_totals_value(game_odds, home_team_stats, away_team_stats, threshold_pct=5.0):
    """
    Compare over/under lines against historical scoring averages.

    Parameters:
        game_odds (dict): Parsed game odds
        home_team_stats (dict): Home team stats
        away_team_stats (dict): Away team stats
        threshold_pct (float): Minimum edge to flag

    Returns:
        list: Value candidates for over/under
    """
    candidates = []
    threshold = threshold_pct / 100.0

    over_odds = game_odds["totals"].get("Over", [])
    under_odds = game_odds["totals"].get("Under", [])

    if not over_odds:
        return candidates

    # Get the consensus line (most common total)
    lines = [o["line"] for o in over_odds]
    consensus_line = max(set(lines), key=lines.count) if lines else 0

    # Historical average total from recent games
    home_avg_scored = home_team_stats["recent"]["avg_scored"]
    home_avg_allowed = home_team_stats["recent"]["avg_allowed"]
    away_avg_scored = away_team_stats["recent"]["avg_scored"]
    away_avg_allowed = away_team_stats["recent"]["avg_allowed"]

    # Projected total: average of (home_scored + away_scored) and (home_allowed + away_allowed)
    projected_from_offense = home_avg_scored + away_avg_scored
    projected_from_defense = home_avg_allowed + away_avg_allowed
    projected_total = (projected_from_offense + projected_from_defense) / 2

    # Determine over/under probability from historical games
    home_recent_games = home_team_stats.get("recent_games", [])
    away_recent_games = away_team_stats.get("recent_games", [])

    # Count how often recent games went over the line
    over_count = 0
    total_counted = 0
    for games in [home_recent_games, away_recent_games]:
        for g in games:
            if g["total_score"] > consensus_line:
                over_count += 1
            total_counted += 1

    over_hit_rate = over_count / total_counted if total_counted > 0 else 0.5

    diff = projected_total - consensus_line

    candidates.append({
        "type": "total_over",
        "matchup": f"{game_odds['away_team']} @ {game_odds['home_team']}",
        "line": consensus_line,
        "projected_total": round(projected_total, 2),
        "diff_from_line": round(diff, 2),
        "over_hit_rate": round(over_hit_rate * 100, 2),
        "home_avg_scored": round(home_avg_scored, 2),
        "away_avg_scored": round(away_avg_scored, 2),
        "is_over_value": diff > 0 and over_hit_rate > 0.5 + threshold,
        "is_under_value": diff < 0 and (1 - over_hit_rate) > 0.5 + threshold,
    })

    return candidates


def analyze_spreads_value(game_odds, home_team_stats, away_team_stats, threshold_pct=5.0):
    """
    Compare spread lines against historical scoring margins.

    For each team, calculates:
    - Average margin of victory/defeat from recent games
    - Historical cover rate against the given spread
    - Flags value where historical cover rate significantly exceeds implied ~50%

    Parameters:
        game_odds (dict): Parsed game odds from odds_client.parse_game_odds()
        home_team_stats (dict): Home team stats with 'recent' and 'recent_games' keys
        away_team_stats (dict): Away team stats with 'recent' and 'recent_games' keys
        threshold_pct (float): Minimum edge to flag as value

    Returns:
        list: Value candidates for spread bets
    """
    candidates = []
    threshold = threshold_pct / 100.0

    home_team = game_odds["home_team"]
    away_team = game_odds["away_team"]

    for team_name, stats, is_home in [
        (home_team, home_team_stats, True),
        (away_team, away_team_stats, False),
    ]:
        spread_odds = game_odds["spreads"].get(team_name, [])
        if not spread_odds:
            continue

        # Get the consensus spread (most common line)
        spreads = [o["spread"] for o in spread_odds]
        consensus_spread = max(set(spreads), key=spreads.count) if spreads else 0

        # Calculate average margin from recent games
        recent_games = stats.get("recent_games", [])
        margins = []
        cover_count = 0
        for game in recent_games:
            if game["home_team"] == team_name:
                margin = game["home_score"] - game["away_score"]
            elif game["away_team"] == team_name:
                margin = game["away_score"] - game["home_score"]
            else:
                continue
            margins.append(margin)
            # A team "covers" if their margin > the negative of their spread
            # e.g., favored by 5 (spread = -5): need to win by more than 5
            # e.g., underdog by 5 (spread = +5): can lose by less than 5 or win
            if margin + consensus_spread > 0:
                cover_count += 1

        if not margins:
            continue

        avg_margin = sum(margins) / len(margins)
        cover_rate = cover_count / len(margins)

        # Book implies ~50% cover rate (vig aside). Value = historical cover rate - 0.50
        edge = cover_rate - 0.50

        candidates.append({
            "type": "spread",
            "team": team_name,
            "opponent": away_team if is_home else home_team,
            "home_away": "HOME" if is_home else "AWAY",
            "spread": consensus_spread,
            "avg_margin": round(avg_margin, 2),
            "cover_rate": round(cover_rate * 100, 2),
            "games_sampled": len(margins),
            "edge_pct": round(edge * 100, 2),
            "is_value": edge >= threshold,
        })

    return candidates


def format_moneyline_report(candidates):
    """Format moneyline value candidates into a readable report."""
    value_bets = [c for c in candidates if c["is_value"]]
    non_value = [c for c in candidates if not c["is_value"]]

    lines = []
    if value_bets:
        lines.append("=" * 80)
        lines.append("  VALUE BETS FOUND (Historical Prob > Book Implied Prob + Threshold)")
        lines.append("=" * 80)
        for c in sorted(value_bets, key=lambda x: x["edge_pct"], reverse=True):
            lines.append("")
            lines.append(f"  >>> {c['team']} ({c['home_away']}) vs {c['opponent']}")
            lines.append(f"      Book Implied:   {c['book_implied_prob']}%")
            lines.append(f"      Season Win%:    {c['season_win_pct']}%")
            lines.append(f"      Recent Win%:    {c['recent_win_pct']}% (last N games)")
            lines.append(f"      Blended Hist:   {c['hist_prob']}%")
            lines.append(f"      EDGE:           +{c['edge_pct']}% (best: +{c['best_edge_pct']}% at {c['best_book']})")
            lines.append(f"      Best Price:     {c['best_price']:+d}")
    else:
        lines.append("\n  No moneyline value bets found above threshold.")

    if non_value:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  Other Matchups (no edge above threshold)")
        lines.append("-" * 80)
        for c in non_value:
            edge_str = f"+{c['edge_pct']}%" if c['edge_pct'] > 0 else f"{c['edge_pct']}%"
            lines.append(f"  {c['team']:30s} | Implied: {c['book_implied_prob']:5.2f}% | Hist: {c['hist_prob']:5.2f}% | Edge: {edge_str}")

    return "\n".join(lines)


def format_spreads_report(candidates):
    """Format spread value candidates into a readable report."""
    value_bets = [c for c in candidates if c["is_value"]]
    non_value = [c for c in candidates if not c["is_value"]]

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  SPREAD ANALYSIS (Historical Cover Rate vs Implied ~50%)")
    lines.append("=" * 80)

    if value_bets:
        for c in sorted(value_bets, key=lambda x: x["edge_pct"], reverse=True):
            lines.append("")
            lines.append(f"  >>> {c['team']} ({c['home_away']}) vs {c['opponent']}")
            lines.append(f"      Spread:         {c['spread']:+.2f}")
            lines.append(f"      Avg Margin:     {c['avg_margin']:+.2f} (last {c['games_sampled']} games)")
            lines.append(f"      Cover Rate:     {c['cover_rate']}%")
            lines.append(f"      EDGE:           +{c['edge_pct']}% over implied 50%")
    else:
        lines.append("\n  No spread value bets found above threshold.")

    if non_value:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  Other Spreads (no edge above threshold)")
        lines.append("-" * 80)
        for c in non_value:
            edge_str = f"+{c['edge_pct']}%" if c['edge_pct'] > 0 else f"{c['edge_pct']}%"
            lines.append(f"  {c['team']:30s} | Spread: {c['spread']:+5.2f} | Cover: {c['cover_rate']:5.2f}% | Edge: {edge_str}")

    return "\n".join(lines)


def analyze_player_props_value(prop_data, player_histories, threshold_pct=5.0):
    """
    Compare player prop lines against historical stat values from ESPN.

    Parameters:
        prop_data (dict): Parsed player props from odds_client.parse_player_props()
        player_histories (dict): {player_name: {prop_key: stat_history_dict}}
        threshold_pct (float): Minimum edge to flag as value

    Returns:
        list: Value candidates for player props
    """
    candidates = []
    threshold = threshold_pct / 100.0

    matchup = f"{prop_data['away_team']} @ {prop_data['home_team']}"

    for prop_key, players in prop_data.get("props", {}).items():
        for player_name, odds_info in players.items():
            line = odds_info["line"]
            over_implied = odds_info["over_implied"]
            under_implied = odds_info["under_implied"]
            over_price = odds_info["over_price"]
            under_price = odds_info["under_price"]

            history = player_histories.get(player_name, {}).get(prop_key)
            if not history or not history.get("found") or not history.get("values"):
                candidates.append({
                    "type": "player_prop",
                    "matchup": matchup,
                    "player": player_name,
                    "prop": prop_key,
                    "prop_label": PROP_LABELS.get(prop_key, prop_key),
                    "line": line,
                    "over_price": over_price,
                    "under_price": under_price,
                    "over_implied": round(over_implied * 100, 2),
                    "under_implied": round(under_implied * 100, 2),
                    "avg_stat": None,
                    "over_rate": None,
                    "games_sampled": 0,
                    "edge_pct": 0,
                    "direction": None,
                    "is_value": False,
                    "no_history": True,
                })
                continue

            values = history["values"]
            avg_stat = sum(values) / len(values)
            over_count = sum(1 for v in values if v > line)
            over_rate = over_count / len(values) if values else 0.5

            # Compare historical over rate vs book implied over probability
            over_edge = over_rate - over_implied
            under_rate = 1 - over_rate
            under_edge = under_rate - under_implied

            if over_edge > under_edge:
                direction = "OVER"
                edge = over_edge
                best_price = over_price
            else:
                direction = "UNDER"
                edge = under_edge
                best_price = under_price

            candidates.append({
                "type": "player_prop",
                "matchup": matchup,
                "player": player_name,
                "prop": prop_key,
                "prop_label": PROP_LABELS.get(prop_key, prop_key),
                "line": line,
                "over_price": over_price,
                "under_price": under_price,
                "over_implied": round(over_implied * 100, 2),
                "under_implied": round(under_implied * 100, 2),
                "avg_stat": round(avg_stat, 2),
                "over_rate": round(over_rate * 100, 2),
                "games_sampled": len(values),
                "edge_pct": round(edge * 100, 2),
                "direction": direction,
                "best_price": best_price,
                "is_value": edge >= threshold,
                "no_history": False,
            })

    return candidates


def format_props_report(candidates):
    """Format player props value candidates into a readable report."""
    value_bets = [c for c in candidates if c["is_value"]]
    no_history = [c for c in candidates if c.get("no_history")]
    non_value = [c for c in candidates if not c["is_value"] and not c.get("no_history")]

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  PLAYER PROPS ANALYSIS")
    lines.append("=" * 80)

    if value_bets:
        lines.append("")
        lines.append("  VALUE PROPS FOUND:")
        for c in sorted(value_bets, key=lambda x: x["edge_pct"], reverse=True):
            lines.append("")
            lines.append(f"  >>> {c['player']} — {c['prop_label']} {c['direction']} {c['line']}")
            lines.append(f"      Matchup:        {c['matchup']}")
            lines.append(f"      Line:           {c['line']}  |  Over: {c['over_price']:+d}  |  Under: {c['under_price']:+d}")
            lines.append(f"      Avg Stat:       {c['avg_stat']} (last {c['games_sampled']} games)")
            lines.append(f"      Over Rate:      {c['over_rate']}% historical")
            lines.append(f"      Book Implied:   Over {c['over_implied']}% / Under {c['under_implied']}%")
            lines.append(f"      EDGE:           +{c['edge_pct']}% on {c['direction']} ({c['best_price']:+d})")
    else:
        lines.append("\n  No player prop value bets found above threshold.")

    if non_value:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  Other Props (no edge above threshold)")
        lines.append("-" * 80)
        for c in non_value:
            edge_str = f"+{c['edge_pct']}%" if c['edge_pct'] > 0 else f"{c['edge_pct']}%"
            dir_str = c['direction'] or "?"
            lines.append(f"  {c['player']:25s} | {c['prop_label']:18s} | Line: {c['line']:5} | Avg: {c['avg_stat']:5} | {dir_str}: {edge_str}")

    if no_history:
        lines.append("")
        lines.append(f"  ({len(no_history)} prop(s) skipped — no ESPN history found)")

    return "\n".join(lines)


def format_totals_report(candidates):
    """Format totals value candidates into a readable report."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  OVER/UNDER ANALYSIS")
    lines.append("=" * 80)

    for c in candidates:
        flag = ""
        if c["is_over_value"]:
            flag = " <<< OVER VALUE"
        elif c["is_under_value"]:
            flag = " <<< UNDER VALUE"

        lines.append(f"")
        lines.append(f"  {c['matchup']}")
        lines.append(f"      Line:              {c['line']}")
        lines.append(f"      Projected Total:   {c['projected_total']} (diff: {c['diff_from_line']:+.2f}){flag}")
        lines.append(f"      Over Hit Rate:     {c['over_hit_rate']}% (recent games)")
        lines.append(f"      Avg Scored (Home): {c['home_avg_scored']}  |  (Away): {c['away_avg_scored']}")

    return "\n".join(lines)


def _normalize_legs(all_ml, all_spreads, all_totals, all_props):
    """
    Convert all analysis results into a uniform leg format for parlay building.
    Only include legs with positive edge.
    
    Each leg dict has:
        game_key: str (e.g., "Team A @ Team B" or matchup)
        team: str or None
        bet_type: str (moneyline, spread, total_over, total_under, player_prop_over, player_prop_under)
        label: str (human readable description)
        player: str or None
        prop_key: str or None (e.g., "player_points")
        edge_pct: float
        odds_price: int or None (American odds)
        hist_prob: float (0-1, historical probability)
        implied_prob: float (0-1, book implied probability)
    """
    legs = []
    
    for c in all_ml:
        if c["edge_pct"] <= 0:
            continue
        game_key = f"{c['opponent']} @ {c['team']}" if c["home_away"] == "HOME" else f"{c['team']} @ {c['opponent']}"
        legs.append({
            "game_key": game_key,
            "team": c["team"],
            "bet_type": "moneyline",
            "label": f"{c['team']} ML ({c['home_away']})",
            "player": None,
            "prop_key": None,
            "edge_pct": c["edge_pct"],
            "odds_price": c.get("best_price"),
            "hist_prob": c["hist_prob"] / 100.0,
            "implied_prob": c["book_implied_prob"] / 100.0,
        })
    
    for c in all_spreads:
        if c["edge_pct"] <= 0 or c.get("games_sampled", 0) < 5:
            continue
        game_key = f"{c['opponent']} @ {c['team']}" if c["home_away"] == "HOME" else f"{c['team']} @ {c['opponent']}"
        legs.append({
            "game_key": game_key,
            "team": c["team"],
            "bet_type": "spread",
            "label": f"{c['team']} {c['spread']:+.2f}",
            "player": None,
            "prop_key": None,
            "edge_pct": c["edge_pct"],
            "odds_price": None,
            "hist_prob": c["cover_rate"] / 100.0,
            "implied_prob": 0.50,
        })
    
    for c in all_totals:
        if c.get("is_over_value"):
            legs.append({
                "game_key": c["matchup"],
                "team": None,
                "bet_type": "total_over",
                "label": f"OVER {c['line']} ({c['matchup']})",
                "player": None,
                "prop_key": None,
                "edge_pct": c["over_hit_rate"] - 50.0,
                "odds_price": None,
                "hist_prob": c["over_hit_rate"] / 100.0,
                "implied_prob": 0.50,
            })
        if c.get("is_under_value"):
            legs.append({
                "game_key": c["matchup"],
                "team": None,
                "bet_type": "total_under",
                "label": f"UNDER {c['line']} ({c['matchup']})",
                "player": None,
                "prop_key": None,
                "edge_pct": (100.0 - c["over_hit_rate"]) - 50.0,
                "odds_price": None,
                "hist_prob": (100.0 - c["over_hit_rate"]) / 100.0,
                "implied_prob": 0.50,
            })
    
    for c in all_props:
        if c.get("no_history") or c["edge_pct"] <= 0 or c.get("games_sampled", 0) < 5:
            continue
        direction = c.get("direction", "OVER")
        bt = f"player_prop_{direction.lower()}"
        price = c.get("best_price", c.get("over_price") if direction == "OVER" else c.get("under_price"))
        
        if direction == "OVER":
            hp = (c["over_rate"] / 100.0) if c.get("over_rate") is not None else 0.5
            ip = (c["over_implied"] / 100.0) if c.get("over_implied") is not None else 0.5
        else:
            hp = (1.0 - c["over_rate"] / 100.0) if c.get("over_rate") is not None else 0.5
            ip = (c["under_implied"] / 100.0) if c.get("under_implied") is not None else 0.5
        
        legs.append({
            "game_key": c["matchup"],
            "team": None,
            "bet_type": bt,
            "label": f"{c['player']} {c['prop_label']} {direction} {c['line']}",
            "player": c["player"],
            "prop_key": c.get("prop"),
            "edge_pct": c["edge_pct"],
            "odds_price": price,
            "hist_prob": hp,
            "implied_prob": ip,
        })
    
    return legs


def _has_hard_conflict(leg_a, leg_b):
    """
    Check if two legs have a hard conflict (mutually exclusive or contradictory).
    These combos should NEVER appear in a parlay together.
    """
    same_game = leg_a["game_key"] == leg_b["game_key"]
    
    if not same_game:
        return False
    
    ta = leg_a["bet_type"]
    tb = leg_b["bet_type"]
    
    # Opposite moneylines in same game
    if ta == "moneyline" and tb == "moneyline":
        return leg_a["team"] != leg_b["team"]  # different teams = conflict
    
    # Opposite spreads in same game
    if ta == "spread" and tb == "spread":
        return leg_a["team"] != leg_b["team"]
    
    # Over + Under on same game total
    if {ta, tb} == {"total_over", "total_under"}:
        return True
    
    # Same player, same prop, opposite direction
    if "player_prop" in ta and "player_prop" in tb:
        if (leg_a["player"] == leg_b["player"] 
            and leg_a["prop_key"] == leg_b["prop_key"]
            and ta != tb):
            return True
    
    return False


def _correlation_penalty(leg_a, leg_b, sport_key):
    """
    Return a correlation penalty (negative = bad combo, positive = good synergy).
    Used to score parlay quality beyond just raw edge.
    
    Returns a float:
        negative values = legs work against each other
        0 = neutral
        positive values = legs complement each other (positively correlated)
    """
    same_game = leg_a["game_key"] == leg_b["game_key"]
    ta = leg_a["bet_type"]
    tb = leg_b["bet_type"]
    
    # Cross-game parlays are preferred (less priced in by books)
    if not same_game:
        # Small bonus for cross-game diversification
        return 0.5
    
    # ── Same-game correlation rules ──
    
    # NBA-specific
    if sport_key == "basketball_nba":
        # Two player prop overs from same team = negative (usage cap)
        if "player_prop_over" in ta and "player_prop_over" in tb:
            # Same team check: if both players are in the same matchup, 
            # they might be on same team. We can't tell for sure from matchup alone,
            # but penalize same-game multi-prop overs
            return -2.0
        
        # Game total under + player points over = negative
        if (ta == "total_under" and "player_prop_over" in tb and 
            leg_b.get("prop_key") == "player_points"):
            return -3.0
        if (tb == "total_under" and "player_prop_over" in ta and 
            leg_a.get("prop_key") == "player_points"):
            return -3.0
        
        # Game total over + player points over = positive
        if (ta == "total_over" and "player_prop_over" in tb and 
            leg_b.get("prop_key") == "player_points"):
            return 1.5
        if (tb == "total_over" and "player_prop_over" in ta and 
            leg_a.get("prop_key") == "player_points"):
            return 1.5
        
        # Team ML + player prop over for same team = positive
        if ta == "moneyline" and "player_prop_over" in tb:
            return 1.0
        if tb == "moneyline" and "player_prop_over" in ta:
            return 1.0
    
    # NFL-specific
    elif sport_key == "americanfootball_nfl":
        # QB passing yards over + team ML = strong positive
        if (ta == "moneyline" and "player_prop_over" in tb and 
            leg_b.get("prop_key") == "player_pass_yds"):
            return 2.0
        if (tb == "moneyline" and "player_prop_over" in ta and 
            leg_a.get("prop_key") == "player_pass_yds"):
            return 2.0
        
        # RB rushing yards over + game under = negative
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "player_rush_yds" 
            and tb == "total_under"):
            return -2.0
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "player_rush_yds" 
            and ta == "total_under"):
            return -2.0
        
        # QB passing yards over + game over = positive
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "player_pass_yds" 
            and tb == "total_over"):
            return 1.5
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "player_pass_yds" 
            and ta == "total_over"):
            return 1.5
        
        # Multiple player prop overs same game = slight negative (usage)
        if "player_prop_over" in ta and "player_prop_over" in tb:
            return -1.0
    
    # MLB-specific
    elif sport_key == "baseball_mlb":
        # Pitcher K's over + game under = positive (dominant pitching)
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "pitcher_strikeouts" 
            and tb == "total_under"):
            return 2.0
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "pitcher_strikeouts" 
            and ta == "total_under"):
            return 2.0
        
        # Pitcher K's over + game over = negative
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "pitcher_strikeouts" 
            and tb == "total_over"):
            return -2.0
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "pitcher_strikeouts" 
            and ta == "total_over"):
            return -2.0
        
        # Batter hits over + team ML = positive
        if (ta == "moneyline" and "player_prop_over" in tb and 
            leg_b.get("prop_key") == "batter_hits"):
            return 1.5
        if (tb == "moneyline" and "player_prop_over" in ta and 
            leg_a.get("prop_key") == "batter_hits"):
            return 1.5
    
    # Default same-game slight penalty (less diversification)
    return -0.5


def _same_team_prop_count(legs):
    """Count how many player prop overs are from the same game (proxy for same team)."""
    game_prop_counts = {}
    for leg in legs:
        if "player_prop_over" in leg["bet_type"]:
            gk = leg["game_key"]
            game_prop_counts[gk] = game_prop_counts.get(gk, 0) + 1
    return max(game_prop_counts.values()) if game_prop_counts else 0


def _score_parlay(legs, sport_key, mode="value"):
    """
    Score a parlay combination. Higher is better.
    
    Modes:
        value: Prioritizes edge (higher edge = better)
        safe: Prioritizes probability of hitting (higher hist_prob = better)
    """
    # Pairwise correlation scoring
    correlation_score = 0.0
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            correlation_score += _correlation_penalty(legs[i], legs[j], sport_key)
    
    # Penalty for too many same-game player prop overs (usage cap)
    same_team_count = _same_team_prop_count(legs)
    usage_penalty = 0
    if same_team_count > 2:
        usage_penalty = -5.0 * (same_team_count - 2)
    
    # Combined probabilities
    combined_hist = 1.0
    combined_implied = 1.0
    for leg in legs:
        combined_hist *= leg["hist_prob"]
        combined_implied *= leg["implied_prob"]
    
    if mode == "safe":
        # Prioritize highest combined probability of hitting
        # Scale hist_prob heavily so it dominates the score
        prob_score = combined_hist * 1000
        # Still consider edge but weighted much less
        total_edge = sum(leg["edge_pct"] for leg in legs) * 0.1
        return prob_score + total_edge + correlation_score + usage_penalty
    else:
        # Prioritize edge value
        total_edge = sum(leg["edge_pct"] for leg in legs)
        parlay_edge = (combined_hist - combined_implied) * 100
        return total_edge + correlation_score + usage_penalty + parlay_edge


def generate_parlays(all_ml, all_spreads, all_totals, all_props, sport_key, mode="value"):
    """
    Generate the top recommended 3, 4, and 5 leg parlays.
    
    Parameters:
        all_ml: Moneyline analysis results
        all_spreads: Spread analysis results
        all_totals: Totals analysis results
        all_props: Player prop analysis results
        sport_key: Sport key (e.g., 'basketball_nba')
        mode: 'value' (prioritize edge) or 'safe' (prioritize hit probability)
    
    Returns:
        dict: {3: parlay_dict, 4: parlay_dict, 5: parlay_dict}
    """
    from itertools import combinations
    
    legs = _normalize_legs(all_ml, all_spreads, all_totals, all_props)
    
    if len(legs) < 3:
        return {}
    
    # Sort and take top candidates to limit combinatorics
    if mode == "safe":
        legs.sort(key=lambda x: x["hist_prob"], reverse=True)
    else:
        legs.sort(key=lambda x: x["edge_pct"], reverse=True)
    candidates = legs[:25]  # Cap at 25 to keep combos manageable
    
    results = {}
    
    for size in [3, 4, 5]:
        if len(candidates) < size:
            continue
        
        best_parlay = None
        best_score = float("-inf")
        
        for combo in combinations(candidates, size):
            combo_list = list(combo)
            
            # Check for hard conflicts
            has_conflict = False
            for i in range(len(combo_list)):
                for j in range(i + 1, len(combo_list)):
                    if _has_hard_conflict(combo_list[i], combo_list[j]):
                        has_conflict = True
                        break
                if has_conflict:
                    break
            
            if has_conflict:
                continue
            
            score = _score_parlay(combo_list, sport_key, mode)
            
            if score > best_score:
                best_score = score
                best_parlay = combo_list
        
        if best_parlay:
            combined_hist = 1.0
            combined_implied = 1.0
            combined_edge = 0.0
            for leg in best_parlay:
                combined_hist *= leg["hist_prob"]
                combined_implied *= leg["implied_prob"]
                combined_edge += leg["edge_pct"]
            
            results[size] = {
                "legs": best_parlay,
                "score": best_score,
                "combined_edge": round(combined_edge, 2),
                "combined_hist_prob": round(combined_hist * 100, 2),
                "combined_implied_prob": round(combined_implied * 100, 2),
                "parlay_edge_pct": round((combined_hist - combined_implied) * 100, 2),
            }
    
    return results
