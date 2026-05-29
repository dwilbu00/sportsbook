# 🎯 Sportsbook Value Finder

Compare sportsbook odds against historical ESPN statistics to find value betting opportunities — with a model that's careful about what it claims.

## Features
- **Moneyline, Spreads, Over/Under** analysis for NBA, NFL, and MLB
- **Player Props** analysis (Points, Assists, Rebounds, TDs, Hits, Strikeouts, etc.)
- **Safe Mode** — picks lower alt-lines the model believes will hit with high confidence
- **Value & Safe Parlay Builder** that accounts for how legs move together (no double-counting correlated outcomes)
- **Parallel processing** for fast multi-game analysis
- **Credit-aware caching** to minimize API usage
- **Self-learning calibration** — the model gets more accurate the more you use it

## How the model thinks (in plain English)

### 1. It only learns from "normal" games
A star player having a 10-minute foul-out night or returning from injury is going to ruin a prediction. So the model **ignores** games that look unreliable:
- Games where the player barely played (well below their usual minutes)
- The game right before a multi-game absence (likely they got hurt mid-game)
- Games during and right after a break in their schedule (injury, suspension, etc.)
- The first game they played after returning from a break (still ramping up)

On top of that, predictions are **paused** for a player until they've put together a solid streak of normal games (about 8 for NBA, 6 for MLB, 4 for NFL). No streak = no prediction. This eliminates the "I just bet $50 on a player who played 10 minutes" disasters.

### 2. It blends with last season early in a new season
Early in a season we don't have enough current-year games to be confident. So the model **blends current-season data with last season's** until you have ~10 current-season games for the player. The weight shifts toward the current season as the year progresses, so by mid-season the prior year doesn't matter anymore.

### 3. It corrects for its own overconfidence (and keeps improving)
A raw model that says "80% likely" often only hits 65% of the time in the real world. So we apply a **calibration correction** that's fit on actual past bets vs. actual outcomes. When the app says "75%", it really means ~75%.

What makes this special: **the calibration updates automatically** as you use the app. Every prediction gets logged, every game's outcome eventually gets resolved against ESPN's stats, and every so often the calibration re-fits itself with the new data. The longer you use it, the sharper it gets.

### 4. Safe Mode picks bets it can actually back up
"Safe Mode" answers a different question: instead of "is this over/under a good bet?" it asks "what's an alt-line I'm 90% sure this player will clear?" Then it suggests something like "Steph Curry **8+ points**" when the book line is 27.5.

To avoid lying about confidence, Safe Mode has guards:
- The historical hit rate at that suggested line must be within 5 points of the claimed confidence (so a "90% pick" really hit 85%+ historically).
- It refuses to make a "high-confidence" claim when the bet collapses to "did the player do the thing at all?" (e.g., 1+ assist). Those low bars look safe but are actually noisy.
- The suggested line has to be at least 50% of the book line (no useless "Wemby 2+ points" suggestions).

### 5. Parlays know that some legs aren't independent
Naive parlay math assumes each leg is a coin flip independent of the others. That's wrong: if LeBron has a huge game, both his points OVER and his teammate's rebounds OVER are more likely to hit together. We use a **correlation model** so the joint probability isn't artificially inflated.

## What this gets you
- **NBA points and rebounds** in Safe Mode hit their target within 1–3 points (e.g., 90% target → 91% actual).
- **NBA assists** stays calibrated at 85–90% target after rejecting "1+" floor bets.
- **MLB pitcher strikeouts** uses a fallback data source when ESPN's main gamelog is empty.
- Bad bets ("ignore this player, they just got back from injury") get filtered out automatically rather than tricking you.

## Setup

### Streamlit Cloud
1. Fork/clone this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and deploy the app
3. In the app settings, add your secret:
   ```toml
   ODDS_API_KEY = "your_api_key_here"
   ```
4. Get a free API key at [the-odds-api.com](https://the-odds-api.com/#get-access) (500 credits/month)

### Local
1. `pip install -r requirements.txt`
2. Create `.streamlit/secrets.toml`:
   ```toml
   ODDS_API_KEY = "your_api_key_here"
   ```
3. `streamlit run app.py`
