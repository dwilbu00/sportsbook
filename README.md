# 🎯 Sportsbook Value Finder

Compare sportsbook odds against historical ESPN statistics to find value betting opportunities.

## Features
- **Moneyline, Spreads, Over/Under** analysis for NBA, NFL, and MLB
- **Player Props** analysis (Points, Assists, Rebounds, TDs, Hits, Strikeouts, etc.)
- **Parallel processing** for fast multi-game analysis
- **Value & Safe Parlay Builder** with sport-specific correlation logic
- **Credit-aware caching** to minimize API usage

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
