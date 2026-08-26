# Advanced MLB Prediction Infrastructure: Azure & Python Pipelines
This architectural guide outlines the deployment of production-grade Major League Baseball (MLB) machine learning infrastructure utilizing **Statcast telemetry** data managed inside an **Azure Data Lakehouse / Synapse** architecture and executed via **Python**.

---

## 1. Architectural Strategy & Pipeline Overview

Predicting MLB outcomes requires two distinct computational paradigms due to the different mathematical nature of team-level and player-level variances:

1. **Team Wins, Moneylines, & Run Lines (Spreads):** Modeled using aggregated, time-decayed underlying physics via Gradient Boosted Decision Trees (**XGBoost / LightGBM**). Rather than predicting binary wins, the system targets **Run Differential**, which is subsequently passed into a **Monte Carlo Simulation** engine to map spread probabilities.
2. **Player Props (Strikeouts, Total Bases, Walks):** Modeled using **Empirical Bayes Regression**. Raw sample sizes over short time horizons suffer from high volatility. Metrics are algorithmically regressed toward structural league baselines using verified **Stabilization Constants ($K$)** until specific thresholds are met.

---

## 2. Team Wins & Spreads: Gradient Boosting and Simulation Pipeline

### Step A: SQL/PySpark Feature Extraction (Azure Synapse / Databricks)
To feed the team model, we construct rolling, time-decayed tracking arrays for both offensive and defensive groupings. The pipeline completely ignores surface-level wins and losses, focusing entirely on **Expected Weighted On-Base Average (xwOBA)** and physical contact quality.

```python
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def calculate_time_decay_features(df, lambda_decay=0.05):
    """
    Applies an exponential time-decay function to recent team performance arrays.
    
    Parameters:
        df (pd.DataFrame): Contains daily metrics with columns:
                           'days_ago', 'xwoba', 'hard_hit_rate', 'barrel_rate'
        lambda_decay (float): Hyperparameter controlling the rate of recency discount.
    """
    # Calculate exponential weights based on how many days ago the match occurred
    df['weight'] = np.exp(-lambda_decay * df['days_ago'])
    
    # Derivation of weighted statistical metrics
    weighted_xwoba = np.sum(df['xwoba'] * df['weight']) / np.sum(df['weight'])
    weighted_barrel = np.sum(df['barrel_rate'] * df['weight']) / np.sum(df['weight'])
    
    return {
        'moving_xwoba': weighted_xwoba,
        'moving_barrel_rate': weighted_barrel
    }
```

### Step B: Training Target Framework (XGBoost)
The model targets the final **Run Differential** ($Score_{Home} - Score_{Away}$) rather than a classification vector. This allows the output to adapt dynamically to standard moneylines as well as the industry-standard **-1.5 / +1.5 Run Line** spreads.

```python
# Feature Vector Configuration
features = [
    'away_offensive_moving_xwoba', 'away_offensive_moving_barrel',
    'home_pitching_moving_xwoba_allowed', 'home_pitching_moving_stuff_plus',
    'home_offensive_moving_xwoba', 'home_offensive_moving_barrel',
    'away_pitching_moving_xwoba_allowed', 'away_pitching_moving_stuff_plus',
    'park_factor_total', 'temperature_f', 'wind_speed_mph', 'air_density_kg_m3'
]

# X represents the training matrix; y represents historical match run differentials
def train_team_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
```

### Step C: Monte Carlo Run Line Engine
Once the machine learning architecture outputs an expected score margin (e.g., Home Team expected to win by $+0.65$ runs) alongside the historical standard error ($\sigma$) of the model, a Monte Carlo simulation loops the matchup $10,000	imes$ to map precise probabilities.

```python
def run_monte_carlo_simulation(expected_margin, std_error, simulations=10000):
    """
    Simulates match run differentials to extract moneyline and run-line probabilities.
    """
    # Generate distribution based on predicted margin and residual standard deviation
    simulated_margins = np.random.normal(expected_margin, std_error, simulations)
    
    # Calculate Probabilities
    home_win_ml = np.mean(simulated_margins > 0)
    home_cover_minus_1_5 = np.mean(simulated_margins > 1.5)
    away_cover_plus_1_5 = np.mean(simulated_margins < 1.5)
    
    return {
        'home_moneyline_prob': home_win_ml,
        'home_spread_minus_1_5_prob': home_cover_minus_1_5,
        'away_spread_plus_1_5_prob': away_cover_plus_1_5
    }
```

---

## 3. Player Props: Empirical Bayes & Stabilization Math

Short-term stat lines in baseball are notoriously noisy. To counter this, advanced prop engines employ an **Empirical Bayes Estimation** framework to blend player-specific Statcast measurements with an adjusted league baseline.

The math underlying the calculation is:

$$	ext{Projected Metric} = \left( rac{N}{N + K} 	imes 	ext{Player Metric} ight) + \left( rac{K}{N + K} 	imes 	ext{League Baseline} ight)$$

Where:
* $N$ = Total sample size of individual opportunities (Plate Appearances or Batters Faced).
* $K$ = The unique **Stabilization Constant** representing the exact sample size where signal splits identically with random noise ($50/50$).

### Proven Mathematical Stabilization Constants ($K$)

| Market Archetype | Core Statcast Tracking Metric | Stabilization Point ($K$) | Target Prop Application |
| :--- | :--- | :--- | :--- |
| **Batter Context** | Exit Velocity (EV) | **40 Batted Ball Events (BBE)** | Total Bases, Hit Props |
| **Batter Context** | Launch Angle (LA) | **80 Batted Ball Events (BBE)** | Home Run Props, Out Props |
| **Batter Context** | Barrel Rate (%) | **50 Batted Ball Events (BBE)** | Total Bases, Over/Under HRs |
| **Pitcher Context** | Strikeout Rate (K%) | **70 Batters Faced (BF)** | Over/Under Pitcher Strikeouts |
| **Pitcher Context** | Walk Rate (BB%) | **170 Batters Faced (BF)** | Over/Under Pitcher Walks Issued |

### Stabilization Script Implementation

```python
def calculate_stabilized_prop_projection(player_df, league_baseline, metric_col, K_constant):
    """
    Regresses small-sample player telemetry back to an adaptive league baseline.
    """
    # Track count of sample sizes (N)
    N = len(player_df)
    
    if N == 0:
        return league_baseline
        
    player_sample_mean = player_df[metric_col].mean()
    
    # Execute Empirical Bayes Blend
    stabilized_value = ((N / (N + K_constant)) * player_sample_mean) + ((K_constant / (N + K_constant)) * league_baseline)
    return stabilized_value
```

---

## 4. Pitch-by-Pitch "Stuff+" Micro-Modeling

Because you possess pitch-by-pitch tracking telemetry natively inside your Azure environment, you do not need to wait for a pitcher's box-score results to stabilize over $70$ batters. You can train a **Stuff+ Classification Model** at the single-pitch level using pure physics.

### Step 1: Feature Matrix
Train an `XGBoostClassifier` using specific physical attributes of the pitch, with a binary target of **Whiff (1)** versus **Contact (0)**:
* `release_speed` (Raw velocity in MPH)
* `release_spin_rate` (Spin revolutions per minute)
* `p_vertical_break` / `p_horizontal_break` (Statcast movement vectors in inches)
* `release_extension` (Extension distance toward home plate)
* `plate_x` / `plate_z` (Location coordinates inside or outside the strike zone)

```python
from xgboost import XGBClassifier

def train_stuff_plus_model(pitch_level_df):
    """
    Classifies the mathematical probability of a pitch generating a swing-and-miss.
    """
    features_pitch = ['release_speed', 'release_spin_rate', 'p_vertical_break', 'p_horizontal_break', 'release_extension']
    
    X = pitch_level_df[features_pitch]
    y = pitch_level_df['is_whiff'] # Binary mapping: 1 for whiff, 0 for contact/foul/foul tip
    
    stuff_model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    stuff_model.fit(X, y)
    return stuff_model
```

### Step 2: Prop Integration
An index score can be calculated for every pitcher based on their expected whiff rates from their most recent $50$ pitches. If a pitcher's physical **Stuff+ Index** is increasing but their recorded game-level strikeout totals remain low due to sequencing luck, they present a high-edge structural **Over** opportunity on their next Strikeout prop line.

---

## 5. Daily Environmental & Boundary Vectors

To achieve a production-grade predictive canvas, final calculation passes must scale their outputs through localized environmental adjustments before exporting odds:

1. **Park Factors Matrix:** Apply specific run-scoring and extra-base matrices unique to the target venue. (e.g., scaling home run probability coefficients upward at Great American Ball Park and downward at Oracle Park).
2. **Air Density Adjustments:** Read daily weather forecasts from your warehouse pipeline. For every $10^\circ	ext{F}$ change in temperature, ball carry alters by roughly $3.3	ext{ feet}$. Adjust your expected barrel-to-home-run mapping values dynamically based on local air density vectors ($	ext{kg/m}^3$).
3. **The First-Five (F5) Isolation Split:** Bullpen usage introduces highly erratic structural noise. For advanced player prop and moneyline verification, run an isolated pipeline targeting exclusively the **First 5 Innings**. This explicitly pairs the starting pitcher's stabilized metrics directly against the opposing batting lineup's top rotations.
