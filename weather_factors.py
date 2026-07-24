"""MLB park geometry + pre-game weather forecast for the projection (roadmap 1.3).

Temperature and wind move MLB offense (a wind blowing OUT to center adds fly-ball
carry / HR; hotter, thinner air carries farther). This module supplies the
per-game weather the projection multiplier (``props._weather_factor_mult``) needs:

  1. A static 30-park geo table (``MLB_PARK_GEO``) — latitude/longitude plus the
     compass bearing from home plate to center field, so a forecast wind
     *direction* can be projected onto the out/in-to-CF axis (the part that
     actually helps or hurts the batter).
  2. ``get_game_weather(home_team, commence_time_iso)`` — the ONLY networked
     piece: an Open-Meteo hourly forecast (FREE, no API key) at the game hour.
     props.py / backtest.py stay network-free (they receive the returned dict).

Why a forecast and not MLB's ``gameData.weather``: that statsapi field is EMPTY
pre-game (populates only near first pitch) — useless when predictions are made.
Open-Meteo gives a real pre-game forecast; combined with the CF bearing it
reproduces the "Out To CF / In From CF" signal at analysis time.

Modeling: applied by props as an ABSOLUTE, baseline-relative nudge (vs 70 F / no
wind), NOT a road-context delta like park factors — a player's ~15-game sample
spans random weather, so its mean already reflects typical conditions.

Fails open everywhere: an unknown park, a dome, or any network/parse error yields
a neutral result so a weather miss never blocks or distorts a recommendation.

``cf_bearing`` values are an APPROXIMATE static prior (well-known published park
orientations, lightly rounded), not a fitted quantity — refine as needed; the
adjustment is bounded and conservative, and a bearing within ~45 deg preserves
the wind sign. The Athletics (Sacramento) and Rays (displaced) have unsettled
home venues and are OMITTED (→ neutral weather), mirroring park_factors' neutral
treatment of them. Pure stdlib + requests (already a dependency).
"""

import hashlib
import json
import math
import os
import time

import requests

from park_factors import _park_key


# Canonical full team name → home-park geometry.
#   lat/lon    : ballpark coordinates (Open-Meteo forecast point)
#   cf_bearing : compass degrees home plate → center field (0=N, 90=E, 180=S)
#   roof       : "open" | "retractable" | "dome"
#                (retractable is treated as open — the day's roof state isn't known
#                 pre-game; "dome" → weather-neutral, no fetch)
MLB_PARK_GEO = {
    "Arizona Diamondbacks":  {"lat": 33.445, "lon": -112.067, "cf_bearing": 2,   "roof": "retractable"},
    "Atlanta Braves":        {"lat": 33.891, "lon": -84.468,  "cf_bearing": 51,  "roof": "open"},
    "Baltimore Orioles":     {"lat": 39.284, "lon": -76.622,  "cf_bearing": 30,  "roof": "open"},
    "Boston Red Sox":        {"lat": 42.346, "lon": -71.097,  "cf_bearing": 43,  "roof": "open"},
    "Chicago Cubs":          {"lat": 41.948, "lon": -87.656,  "cf_bearing": 34,  "roof": "open"},
    "Chicago White Sox":     {"lat": 41.830, "lon": -87.634,  "cf_bearing": 130, "roof": "open"},
    "Cincinnati Reds":       {"lat": 39.097, "lon": -84.507,  "cf_bearing": 120, "roof": "open"},
    "Cleveland Guardians":   {"lat": 41.496, "lon": -81.685,  "cf_bearing": 0,   "roof": "open"},
    "Colorado Rockies":      {"lat": 39.756, "lon": -104.994, "cf_bearing": 2,   "roof": "open"},
    "Detroit Tigers":        {"lat": 42.339, "lon": -83.049,  "cf_bearing": 150, "roof": "open"},
    "Houston Astros":        {"lat": 29.757, "lon": -95.355,  "cf_bearing": 20,  "roof": "retractable"},
    "Kansas City Royals":    {"lat": 39.051, "lon": -94.480,  "cf_bearing": 45,  "roof": "open"},
    "Los Angeles Angels":    {"lat": 33.800, "lon": -117.883, "cf_bearing": 45,  "roof": "open"},
    "Los Angeles Dodgers":   {"lat": 34.074, "lon": -118.240, "cf_bearing": 24,  "roof": "open"},
    "Miami Marlins":         {"lat": 25.778, "lon": -80.220,  "cf_bearing": 40,  "roof": "retractable"},
    "Milwaukee Brewers":     {"lat": 43.028, "lon": -87.971,  "cf_bearing": 128, "roof": "retractable"},
    "Minnesota Twins":       {"lat": 44.982, "lon": -93.278,  "cf_bearing": 88,  "roof": "open"},
    "New York Mets":         {"lat": 40.757, "lon": -73.846,  "cf_bearing": 27,  "roof": "open"},
    "New York Yankees":      {"lat": 40.829, "lon": -73.926,  "cf_bearing": 76,  "roof": "open"},
    "Philadelphia Phillies": {"lat": 39.906, "lon": -75.166,  "cf_bearing": 15,  "roof": "open"},
    "Pittsburgh Pirates":    {"lat": 40.447, "lon": -80.006,  "cf_bearing": 118, "roof": "open"},
    "San Diego Padres":      {"lat": 32.707, "lon": -117.157, "cf_bearing": 0,   "roof": "open"},
    "San Francisco Giants":  {"lat": 37.778, "lon": -122.389, "cf_bearing": 92,  "roof": "open"},
    "Seattle Mariners":      {"lat": 47.591, "lon": -122.332, "cf_bearing": 60,  "roof": "retractable"},
    "St. Louis Cardinals":   {"lat": 38.622, "lon": -90.193,  "cf_bearing": 62,  "roof": "open"},
    "Texas Rangers":         {"lat": 32.747, "lon": -97.083,  "cf_bearing": 135, "roof": "retractable"},
    "Toronto Blue Jays":     {"lat": 43.641, "lon": -79.389,  "cf_bearing": 0,   "roof": "retractable"},
    "Washington Nationals":  {"lat": 38.873, "lon": -77.007,  "cf_bearing": 33,  "roof": "open"},
}
# Built once: normalized team key → geo (odds-API / ESPN spellings resolve via
# park_factors._park_key + its alias map — the same normalization park factors use).
_NORMALIZED_GEO = {_park_key(name): geo for name, geo in MLB_PARK_GEO.items()}


# Open-Meteo forecast endpoint (free, no key).
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_HTTP_HEADERS = {"User-Agent": "SportsbookValueFinder/1.0"}
_FETCH_TIMEOUT = 15

# File cache — forecasts are slate-stable, so a short TTL avoids re-hitting the
# API on every Streamlit rerun without going stale within an analysis session.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "weather")
CACHE_TTL_SECONDS = 2 * 3600


def park_geo(team_name):
    """Home-park geometry for a team, or None if unknown (→ neutral weather)."""
    return _NORMALIZED_GEO.get(_park_key(team_name))


def _neutral(dome=False):
    return {"temp_f": None, "wind_mph": None, "wind_dir_deg": None,
            "wind_out_mph": None, "dome": dome}


def wind_out_component(wind_speed, wind_from_deg, cf_bearing):
    """Signed wind component along the home-plate→CF axis, in the wind's units.

    Positive = blowing OUT toward center (helps carry); negative = blowing IN.
    Open-Meteo reports the direction the wind blows FROM, so the blows-TO bearing
    is ``wind_from_deg + 180``. Returns None on any missing input.
    """
    if wind_speed is None or wind_from_deg is None or cf_bearing is None:
        return None
    wind_to = (wind_from_deg + 180.0) % 360.0
    return wind_speed * math.cos(math.radians(wind_to - cf_bearing))


def _cache_path(key):
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, digest + ".json")


def _read_cache(key, max_age=CACHE_TTL_SECONDS):
    try:
        with open(_cache_path(key), "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        if time.time() - blob.get("cached_at", 0) <= max_age:
            return blob.get("data")
    except (OSError, ValueError):
        pass
    return None


def _write_cache(key, data):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(key), "w", encoding="utf-8") as fh:
            json.dump({"cached_at": time.time(), "data": data}, fh)
    except OSError:
        pass


def _parse_iso_utc(iso_str):
    """Parse an ISO timestamp to an aware UTC datetime (trailing Z tolerated)."""
    from datetime import datetime, timezone
    s = str(iso_str).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _pick_nearest_hour(hourly, commence_dt):
    """Return (temp_f, wind_mph, wind_dir_deg) for the forecast hour nearest the
    game's first pitch, or (None, None, None) if the payload is unusable."""
    from datetime import timezone
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    winds = hourly.get("wind_speed_10m") or []
    dirs = hourly.get("wind_direction_10m") or []
    best_i, best_delta = None, None
    for i, t in enumerate(times):
        try:
            # Times are returned in the requested (UTC) timezone, no offset suffix.
            ft = _parse_iso_utc(t) if ("T" in str(t)) else None
        except ValueError:
            ft = None
        if ft is None:
            continue
        ft = ft.replace(tzinfo=timezone.utc)
        delta = abs((ft - commence_dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta, best_i = delta, i
    if best_i is None:
        return None, None, None

    def _at(seq):
        return seq[best_i] if best_i < len(seq) else None

    return _at(temps), _at(winds), _at(dirs)


def get_game_weather(home_team, commence_time_iso, use_cache=True):
    """Pre-game weather for a game, projected onto the park's out/in-to-CF axis.

    Returns {"temp_f","wind_mph","wind_dir_deg","wind_out_mph","dome"} (values may
    be None). Fails OPEN — an unknown park, a dome, a missing time, or any network
    / parse error yields a neutral dict (all-None, dome as applicable), never an
    exception. ``props._weather_factor_mult`` treats a neutral dict as no change.
    """
    geo = park_geo(home_team)
    if not geo:
        return _neutral()
    if geo.get("roof") == "dome":
        return _neutral(dome=True)
    if not commence_time_iso:
        return _neutral()
    try:
        commence_dt = _parse_iso_utc(commence_time_iso)
    except (ValueError, TypeError):
        return _neutral()

    cache_key = "%s|%s|%s" % (
        _park_key(home_team), geo["lat"], commence_dt.strftime("%Y%m%dT%H"))
    if use_cache:
        cached = _read_cache(cache_key)
        if cached is not None:
            return cached

    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": geo["lat"], "longitude": geo["lon"],
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "timezone": "UTC", "forecast_days": 3,
            },
            headers=_HTTP_HEADERS, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
    except Exception:
        return _neutral()

    temp_f, wind_mph, wind_dir_deg = _pick_nearest_hour(hourly, commence_dt)
    result = {
        "temp_f": temp_f,
        "wind_mph": wind_mph,
        "wind_dir_deg": wind_dir_deg,
        "wind_out_mph": wind_out_component(wind_mph, wind_dir_deg, geo["cf_bearing"]),
        "dome": False,
    }
    if use_cache and (temp_f is not None or wind_mph is not None):
        _write_cache(cache_key, result)
    return result
