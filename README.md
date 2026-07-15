# 🎯 Sportsbook Value Finder

Sportsbook Value Finder is a probability engine for finding mispriced sportsbook lines. It compares live odds against a layered forecasting model built from ESPN history, sportsbook market data, MLB Statcast-style expected metrics, NFL EPA, calibration files, and outcome feedback loops.

Use the live application here: **https://sbvaluefinder.streamlit.app/**

> **Source of truth:** this `deploy` directory is the standalone Git repository
> deployed to Streamlit Cloud. The similarly named scripts in its parent
> `SPORTSBOOK_ODDS` directory are a legacy working copy; do not calibrate or
> release from that copy.

This is **not** a simple “last 5 games average” app. The model now uses a full stack of statistical tools to estimate outcomes, correct its own bias, reject bad samples, and understand when legs in a parlay are mathematically connected.

```text
Live odds + ESPN history + sport-specific advanced metrics
        │
        ▼
Reliability filters → weighted projections → matchup adjustments
        │
        ▼
Residual calibration + Platt recalibration + market blending
        │
        ▼
Value bets, safe alt-lines, and correlation-aware parlays
```

## What it analyzes

- **Moneyline, spread, and total markets** for NBA, NFL, and MLB
- **Player props** across points, rebounds, assists, passing yards, rushing yards, touchdowns, pitcher strikeouts, pitcher outs, earned runs, batter hits, batter strikeouts, and more
- **Alternate lines** for safer threshold-style props such as `8+ points` or `4+ strikeouts`
- **Value parlays and safe parlays** using a joint-probability model instead of naïvely multiplying every leg together
- **Live sportsbook prices** converted into implied probabilities, de-vigged from same-book/same-line pairs while retaining the best executable side prices
- **Market Comparison** context showing DraftKings' line and price beside the median of at least three complete peer-book offers, without changing model edge or rankings

## The mathematical stack

| Layer | What it does | Why it matters |
|---|---|---|
| Implied probability conversion | Converts American odds into decimal odds and book-implied probabilities | Puts every sportsbook line on the same probability scale |
| De-vigging | Removes two-sided sportsbook hold from moneyline, spread, and total markets | Compares the model against a fairer market baseline instead of the book’s padded number |
| Exponential recency weighting | Applies sport-specific half-lives so newer games matter more than older games | Captures current form without throwing away useful history |
| Venue weighting | Up-weights past games played in the same home/away context | Accounts for home court, road splits, and venue-sensitive performance |
| Weighted means, rates, variance, and quantiles | Computes projections, hit rates, volatility, and lower-bound alt-lines from weighted samples | Keeps all downstream probabilities tied to the same evidence base |
| Bayesian shrinkage | Pulls noisy player projections back toward the broader season mean with pseudo-observations | Prevents small hot streaks from becoming overconfident predictions |
| Normal probability models | Uses standard normal CDF / inverse CDF to turn projected margins and stat distributions into probabilities | Converts “projected by 3.2” into “covers 58.7% of the time” |
| Empirical ECDF residuals | Uses observed residual distributions rather than assuming everything is perfectly normal | Handles skewed and fat-tailed player-stat behavior |
| Brier-optimized probability shrink | Pulls team-market probabilities toward 50/50 when backtests show overconfidence | Makes confident outputs harder to earn |
| Model-market blending | Blends the internal model with de-vigged market probability using fitted weights | Respects the wisdom of the market while still surfacing edges |
| Platt scaling | Fits `sigmoid(a * logit(p_raw) + b)` on resolved predictions | Recalibrates raw model probabilities so “70%” means closer to 70% in practice |
| Gaussian copula parlays | Estimates joint parlay hit probability with correlated Bernoulli legs | Avoids pretending related legs are independent |
| Cholesky simulation | Builds positive-definite correlation matrices and Monte Carlo samples joint outcomes | Lets the parlay engine price interaction risk instead of ignoring it |
| Chronological holdout backtests | Fits and scores using time-ordered historical observations | Reduces leakage and makes validation closer to live use |

## How predictions are built

### 1. Odds become fair probabilities

Every line starts as a sportsbook price. The app converts American odds into implied probability, then de-vigs two-sided markets when both sides are present.

Example:

```text
Book line → implied probability → de-vigged market probability → model comparison
```

That means the edge calculation is not just “the model likes it.” It is:

```text
edge = calibrated_model_probability - market_implied_probability
```

### 2. Recent performance is weighted, not blindly averaged

The model uses exponential decay by sport. A game `half_life` games ago contributes about half as much as the most recent game.

- NBA team form defaults around a 10-game half-life
- NBA player props can use tighter prop-specific half-lives such as 7 games
- NFL uses shorter half-lives because each game carries more signal
- MLB uses tuned baseball-specific decay for team and prop markets

The result is a smoother projection than “last 5” but more responsive than full-season averages.

### 3. Bad samples are removed before the model trusts them

The player-prop engine rejects games that distort a player’s true baseline:

- Low-minute games below a median-based threshold
- The game immediately before a multi-game absence, which may indicate an in-game injury
- First game back after a layoff, when ramp-up minutes are unreliable
- Short current streaks after a break, injury, suspension, or role disruption

Predictions are paused until the player has rebuilt a clean streak of valid games. The goal is to avoid recommending a player who technically has history but is not currently in a stable usage pattern.

### 4. Team markets use a shared margin distribution

Moneyline and spread calculations now come from the same home-perspective margin model:

```text
margin ~ Normal(predicted_margin, predicted_std)

P(home wins)   = P(margin > 0)
P(home covers) = P(margin > -spread)
```

That keeps the model internally coherent. At a zero spread, the home moneyline probability and home spread-cover probability line up instead of contradicting each other.

### 5. Totals combine offense, defense, starters, bullpens, and market calibration

Totals start from weighted team scoring and allowed scoring, then apply sport-specific matchup signals where available. For MLB, the model can adjust totals based on probable starters, expected pitcher quality, and bullpen run suppression.

After that, fitted probability shrink and optional market blending correct overconfidence before a total is labeled as value.

## Sport-specific advanced modeling

### MLB: starter, bullpen, Statcast, and handedness features

The MLB layer is no longer just historical scores. It uses MLB Stats API and Baseball Savant-style expected metrics to build matchup features:

- Probable starting pitchers for each side
- Starter handedness
- ERA fallback plus **xERA / xwOBA** when available
- Pitcher xBA for batter-hit matchups, with strikeouts included in its at-bat denominator
- Strikeout and walk rates
- Average innings per start to weight starter vs. bullpen influence
- Team bullpen run-suppression index
- Opposing lineup quality vs. left- or right-handed pitching
- Bounded starter edge using a `tanh` transform
- Calibrated starter weights for moneyline, spreads, total run scaling, and bullpen effects
- Log5-style batter hit/K rates weighted by the starter's expected innings share
- Confirmed batting-order exposure for batter-hit props, using expected at-bats by lineup slot

The model separates feature generation from feature strength. Raw starter, bullpen, and lineup features are built first; calibration files decide how much those features are allowed to move a prediction. The announced-lineup adjustment runs only when a complete nine-player order is available. A 2024 chronological fit selected a 0.75 exposure blend for batter hits (holdout MAE improved 0.262%, with both later rolling folds improving); the equivalent batter-strikeout exposure signal failed forward validation and remains disabled.

#### MLB expected-runs spread ensemble

`backtest_starters.py --season 2024 --test-runs` fits separate home and away
run expectations from leakage-safe offense-vs-hand, probable-starter workload,
and bullpen run-prevention features. Moneyline probability uses the modern MLB
Pythagorean exponent of **1.83**; home `-1.5` probability comes from the same
expected runs through an independent-Poisson score distribution.

The initial July 1 through October 5, 2024 holdout improved moneyline Brier
score (`0.2489` current → `0.2463` challenger), winner accuracy (`52.12%` →
`56.64%`), margin RMSE (`4.861` → `4.402`), and home `-1.5` Brier score
(`0.2410` → `0.2226`). A second independent 2023 holdout still improved margin
RMSE (`4.887` → `4.434`) and home `-1.5` Brier (`0.2405` → `0.2289`), but it
narrowly missed the moneyline gate (`0.2472` current vs. `0.2473` challenger).
Adding all of 2023 to the pre-2024 fit strengthened the 2024 results to `0.2460`
moneyline Brier, `4.392` margin RMSE, and `0.2212` home `-1.5` Brier.

The same chronological runs separately test three additions:

- Home-team/park residual factors fitted only before the holdout. They improve
  the 2024 holdout only when a full prior season is available; the single-season
  2023 and 2024 fits fail.
- Actual bullpen relief-pitch workload over the prior three days (decayed by
  recency). Its holdout gains are below the minimum practical thresholds.
- A fitted negative-binomial score distribution. It models score variance much
  better than Poisson, but does not improve run-line Brier and log loss by the
  required `0.001` in any holdout.

These tests use the free MLB Stats API and Baseball Savant cache, not Odds API
credits.

A final predeclared test used July 1 through October 5, 2025 as a previously
untouched holdout. Raw Pythagorean moneyline probability (`0.2486` Brier) and a
train-fitted current/Pythagorean ensemble (`0.2473`) both lost to the current
moneyline model (`0.2456`), so expected runs will not replace or blend into MLB
moneylines. The market-specific result was different: the expected-runs
ensemble improved margin RMSE from `4.968` to `4.583` and home `-1.5` Brier from
`0.2426` to `0.2311`. Its challenger shares—fitted before the holdout—were
`0.90` for margin and `0.70` for the run line.

Actual-venue park factors estimated only from completed 2023–2024 outcomes also
failed the final gate: they slightly improved score NLL and total RMSE but made
home `-1.5` Brier slightly worse than unadjusted expected runs. Moneyline and
park candidates therefore remain rejected.

The validated ensemble is now active for **MLB spreads only**. It blends 70% of
the independent-Poisson expected-runs cover probability with 30% of the current
calibrated spread probability, and displays a margin blended 90% toward the
expected-runs margin. Its live factors use the same league-relative Baseball
Savant expected-wOBA basis as the historical fit: starter expected wOBA,
team expected wOBA split by opposing pitcher hand, and reliever expected wOBA.
The small team aggregates are cached daily. The ensemble activates only when
both probable starters, both offense-vs-hand splits, and both staff-quality
inputs are available; otherwise the existing spread model runs unchanged. MLB
moneylines and totals are not modified, and park factors and negative-binomial
scoring remain disabled.

### NFL: EPA-based team strength

The NFL layer uses **Expected Points Added per play** from nflverse play-by-play data. EPA is more predictive than raw yards or final scores because it measures the value of each play in game context.

The NFL feature engine includes:

- Offensive EPA/play
- Defensive EPA/play
- Net EPA edge between teams
- Prior-season shrinkage early in the season until current plays stabilize
- Leakage-safe as-of-date ratings for backtests
- OLS-fitted margin weights from historical games
- Probability shrink for moneyline, spread, and total markets

In other words: the NFL side is using play-level efficiency, not just scoreboard outcomes.

### NBA: calibrated prop distributions and opponent-defense adjustment

NBA props combine usage-stability filters with calibrated residual distributions:

- Points, rebounds, and assists have prop-specific calibration files
- Opponent defensive environment can scale projections
- Residual Gaussian and ECDF methods are selected by Brier score
- Early-season output blends current-season calibration with a prior-season warmup distribution
- Platt recalibration is available for props with enough resolved prediction history

## Player props are calibrated twice

Player-prop probability starts with a weighted historical hit rate, but the app does not stop there.

### Residual calibration

The calibration file stores how far actual outcomes usually land from the projected stat:

```text
residual = actual_stat - projected_stat
```

Depending on the prop, the runtime model can use:

- **Method A:** empirical weighted hit rate
- **Method B:** pooled Gaussian residual model
- **Method C:** empirical residual CDF

The best method is selected per prop using chronological holdout scoring.

### Warmup blending

Early in a season, current-year samples are thin. The model blends current-season calibration with prior-season warmup calibration:

```text
w = min(current_season_games / warmup_games, 1)
p = w * p_current + (1 - w) * p_warmup
```

As a player accumulates current-season games, the prior-season influence naturally fades out.

### Self-updating Platt recalibration

Every published prop prediction can be logged, resolved later against ESPN outcomes, and used to refit a Platt sigmoid:

```text
p_calibrated = sigmoid(a * logit(p_raw) + b)
```

The fit uses cross-entropy loss, Newton-Raphson optimization, mild L2 regularization, and minimum-sample guards. A mapping is enabled only if it improves both Brier score and log loss in two expanding-window chronological folds; repeated logs for the same player/game/line are de-duplicated. This gives the app a feedback loop without letting in-sample fit quality masquerade as validation.

The Model Guide also reports forward-log status by sport and prop: resolved and pending counts, model-side hit rate, probability Brier score, realized ROI, and exact-line closing-line value (CLV). New rows retain the event ID, start time, raw and final probabilities, opening price/book, and one pregame closing snapshot. Positive CLV means the model captured a better price than the final snapshot at the exact same player, prop, line, and side. Legacy rows remain usable through backward-compatible fallbacks.

## Safe Mode: conservative alt-line math

Safe Mode answers a different question than standard value betting.

Instead of asking:

```text
Is Over 27.5 a good price?
```

it asks:

```text
What alternate threshold is this player likely to clear with high confidence?
```

The model computes a lower confidence bound:

```text
safe_threshold ≈ projected_mean - z * weighted_std
```

Then it tightens or rejects the threshold using empirical hit-rate guards:

- The same residual, warmup, and Platt calibration stack used by standard props is applied at the suggested threshold
- The historical hit rate at the suggested threshold must be within 5 percentage points of the target confidence
- Low-information `1+` style props are rejected at high confidence targets
- Suggested thresholds must remain realistic relative to the main book line
- The exact alternate-line price is fetched before the pick can be recommended
- The calibrated model probability must beat that exact price's implied probability by at least 5 percentage points

This is why Safe Mode can suggest something like `Curry 8+ points` while refusing fake-safe lines that only look good because the threshold collapsed too far.

## Built-in model guide and performance report

Open **Model Guide & Performance** from the app sidebar to see the production definitions and the evidence behind them in one place:

- Model probability, book implied probability, edge, and expected ROI formulas
- Per-prop chronological model-selection accuracy and Brier score loaded from the committed calibration files
- Team-market backtest/calibration status, with unavailable metrics labeled instead of guessed
- Safe Mode's full confidence-and-price decision pipeline
- MLB xStats holdout decisions and observation counts

The MLB prop-matchup test is deliberately conservative. In the leakage-safe 2024 fit, starter K rate improved batter-strikeout MAE on the main holdout and both rolling folds, enabling a `0.5` weight. Starter xBA slightly improved the aggregate batter-hit holdout but regressed in one rolling fold, so batter hits remain at `0.0`. Pitcher strikeouts, outs, and earned runs also remain at `0.0` until their exact live signals pass the same gates.

## Parlays use joint probability instead of naïve multiplication

Most simple parlay calculators multiply leg probabilities:

```text
P(parlay) = P(leg1) * P(leg2) * P(leg3)
```

That is only valid if every leg is independent. Sports bets are often not independent.

Examples:

- NBA player points over and game total over are positively related
- Pitcher strikeouts over and game total under are positively related
- Player points over and game total under can fight each other
- Multiple prop overs in the same game can be constrained by usage and possessions

This app builds a sport-aware correlation matrix, repairs it if needed so it can be simulated, then estimates joint probability using a Gaussian copula Monte Carlo engine.

The parlay output includes both:

- Independent combined probability
- Correlation-adjusted combined probability

That makes the parlay builder much harder to fool with same-game correlation traps.

## Backtesting and calibration workflow

The repository includes scripts for testing and refitting the model:

- `backtest.py` — team-market projection, player-prop sweeps, Safe Mode tests, and odds-history evaluation
- `backtest_market_consensus.py` — budget-guarded, same-snapshot comparison of de-vigged peer-book consensus against DraftKings
- `backtest_props.py` — chronological MLB starter xBA/K and batting-order exposure fits with holdout and rolling-fold gates
- `book_line_calibration.py` — joins cached book lines to actual player outcomes
- `refit_calibration.py` — writes per-sport prop calibration files
- `recalibration.py` — resolves logged predictions and refits Platt scaling
- `forward_tracker.py` — captures one-shot exact-line closing prices and runs outcome maintenance
- `backtest_starters.py` — fits MLB starter/bullpen weights and validates the spread-only expected-runs/Pythagorean ensemble
- `backtest_nfl_epa.py` — fits NFL EPA margin weights
- `savant_history.py` — leakage-safe historical Statcast feature cache for MLB backtests

Validation focuses on metrics that punish overconfidence:

- Brier score
- Log loss
- Hit rate
- MAE / RMSE for point projections
- Chronological holdout performance
- Out-of-sample safe-line hit rates

## Why the process is hard to fake

The app has several built-in defenses against impressive-looking but fragile predictions:

- It rejects unstable player histories before projecting
- It shrinks probabilities when backtests show overconfidence
- It blends with the market when the market is empirically stronger
- It keeps moneyline and spread probabilities mathematically coherent
- It uses residual distributions instead of assuming every prop is normally distributed
- It avoids leakage in historical backtests by using only information available before the game
- It resolves real predictions and recalibrates from actual outcomes
- It penalizes correlated parlays instead of multiplying probabilities blindly

## Setup

### Streamlit Cloud

1. Fork or clone this repo to your GitHub account.
2. Deploy the app at [share.streamlit.io](https://share.streamlit.io).
3. Add your Odds API key in Streamlit secrets:

   ```toml
   ODDS_API_KEY = "your_api_key_here"
   PREDICTION_LOG_BLOB_URL = "https://ACCOUNT.blob.core.windows.net/CONTAINER/prediction_log.jsonl?SAS_TOKEN"
   ```

4. Get an API key from [the-odds-api.com](https://the-odds-api.com/#get-access).
5. For durable forward tracking, create a private Azure Block Blob SAS URL with
   **read, create, and write** permissions. Add the same URL to Streamlit Cloud
   secrets.

Without `PREDICTION_LOG_BLOB_URL`, the app falls back to `cache/predictions/`.
That local container storage is not durable across Streamlit Cloud restarts or
redeployments.

### Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local secrets, create `.streamlit/secrets.toml`:

```toml
ODDS_API_KEY = "your_api_key_here"
PREDICTION_LOG_BLOB_URL = "https://ACCOUNT.blob.core.windows.net/CONTAINER/prediction_log.jsonl?SAS_TOKEN"
```

### Event-timed forward tracking in Streamlit

After a player-prop analysis logs an upcoming event, the active Streamlit
session arms a one-shot fragment for five minutes before that event starts. The
fragment makes no periodic Odds API requests: it wakes once at the target time,
requests the required prop markets, and records the exact player/prop/line/side
snapshot. A checked event is not retried automatically, even when an exact line
has disappeared, so multiple app sessions do not intentionally repeat the same
credit spend.

The browser session must remain active until the target time. The sidebar's
**Capture eligible closing odds now** button provides a manual retry for an
analyzed event beginning within ten minutes. Outcome resolution remains part of
normal app analysis and uses ESPN rather than The Odds API.

Useful manual commands:

```powershell
python forward_tracker.py --capture-closing --dry-run
python forward_tracker.py --capture-closing
python forward_tracker.py --resolve
```

## Practical note

This project produces calibrated probability estimates, not guarantees. Odds move, data feeds can be incomplete, players get hurt, books shade markets, and variance is real. The goal is to make the decision process mathematically honest, transparent, and harder to fool.
