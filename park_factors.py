"""Static MLB park factors for the player-prop projection (roadmap 1.2).

Ballpark environment shifts hits and runs materially (Coors/elevation up;
Oracle/Petco/T-Mobile down). The projection is built from a player's recent
game log, which already embeds the parks he played in, so callers must apply
these as a ROAD-CONTEXT DELTA (upcoming park factor / his recent-sample park
factor) — never as an absolute multiply, which would double-count his home park.
See ``props._park_factor_mult``.

Values are multipliers (1.00 = league-neutral), ≈3-season published averages
(the well-known Statcast / FanGraphs / ESPN park-factor indices, rescaled from
index-100 to a 1.00 base and lightly rounded). They are a static prior, not a
fitted quantity — refresh occasionally as parks/dimensions change. Two stat
kinds are carried:
  - "hits":  BABIP/hits environment  → applies to ``batter_hits``
  - "runs":  run-scoring environment → applies to ``pitcher_earned_runs``

CAVEATS (fail closed to 1.0 = neutral on anything unknown):
  - The Athletics (relocated to Sacramento, 2025) and the Rays (displaced from
    Tropicana Field for 2025) have unsettled home venues; both are set to a
    NEUTRAL prior rather than guess a temporary park. Revisit once a stable
    multi-season sample exists.
  - Any team/park not in the table returns 1.0.

Pure stdlib; no API calls.
"""


# Canonical full team names → per-kind park factors. Keyed by readable names;
# _NORMALIZED (built below) is what lookups actually use, so odds-API and ESPN
# spellings both resolve through _park_key + the alias map.
MLB_PARK_FACTORS = {
    # ── Hitter-friendly ──
    "Colorado Rockies":       {"hits": 1.14, "runs": 1.20},  # Coors / elevation
    "Boston Red Sox":         {"hits": 1.07, "runs": 1.06},  # Fenway (doubles)
    "Cincinnati Reds":        {"hits": 1.03, "runs": 1.08},  # Great American
    "Kansas City Royals":     {"hits": 1.04, "runs": 1.01},  # Kauffman (gaps)
    "Philadelphia Phillies":  {"hits": 1.02, "runs": 1.04},  # Citizens Bank
    "Arizona Diamondbacks":   {"hits": 1.03, "runs": 1.04},  # Chase Field
    "Texas Rangers":          {"hits": 1.02, "runs": 1.03},  # Globe Life
    "Toronto Blue Jays":      {"hits": 1.02, "runs": 1.02},  # Rogers Centre
    "Chicago Cubs":           {"hits": 1.02, "runs": 1.02},  # Wrigley (wind)
    "Los Angeles Angels":     {"hits": 1.02, "runs": 1.01},  # Angel Stadium
    "Atlanta Braves":         {"hits": 1.01, "runs": 1.02},  # Truist Park
    "Washington Nationals":   {"hits": 1.01, "runs": 1.02},  # Nationals Park
    "Baltimore Orioles":      {"hits": 1.01, "runs": 1.00},  # Camden Yards
    "New York Yankees":       {"hits": 1.00, "runs": 1.03},  # short RF porch
    # ── Roughly neutral ──
    "Houston Astros":         {"hits": 1.00, "runs": 1.00},  # Daikin Park
    "Minnesota Twins":        {"hits": 1.00, "runs": 1.00},  # Target Field
    "Milwaukee Brewers":      {"hits": 1.00, "runs": 1.00},  # American Family
    "Chicago White Sox":      {"hits": 0.99, "runs": 1.00},  # Rate Field
    "Athletics":              {"hits": 1.00, "runs": 1.00},  # relocated (neutral)
    "Tampa Bay Rays":         {"hits": 1.00, "runs": 1.00},  # displaced (neutral)
    # ── Pitcher-leaning ──
    "St. Louis Cardinals":    {"hits": 0.99, "runs": 0.99},  # Busch
    "Cleveland Guardians":    {"hits": 0.99, "runs": 0.99},  # Progressive
    "Pittsburgh Pirates":     {"hits": 0.99, "runs": 0.98},  # PNC
    "Los Angeles Dodgers":    {"hits": 0.98, "runs": 0.99},  # Dodger Stadium
    "New York Mets":          {"hits": 0.98, "runs": 0.97},  # Citi Field
    "Comerica Park Tigers":   {"hits": 0.97, "runs": 0.97},  # placeholder, overwritten below
    "Detroit Tigers":         {"hits": 0.97, "runs": 0.97},  # Comerica (deep OF)
    "Miami Marlins":          {"hits": 0.97, "runs": 0.96},  # loanDepot park
    "San Diego Padres":       {"hits": 0.96, "runs": 0.95},  # Petco
    "San Francisco Giants":   {"hits": 0.96, "runs": 0.94},  # Oracle (marine)
    "Seattle Mariners":       {"hits": 0.95, "runs": 0.94},  # T-Mobile
}
# Remove the placeholder key (kept only so the block reads cleanly).
MLB_PARK_FACTORS.pop("Comerica Park Tigers", None)


# Which prop keys get a non-neutral park effect, and which stat kind drives it.
# Any prop NOT listed here is treated as park-neutral (multiplier 1.0):
# strikeout rate, outs, etc. are minimally park-sensitive.
PROP_PARK_KIND = {
    "batter_hits": "hits",
    "pitcher_earned_runs": "runs",
}


# Normalized team spellings that should map onto a canonical key above.
_ALIASES = {
    "oaklandathletics": "athletics",
    "sacramentoathletics": "athletics",
    "lasvegasathletics": "athletics",
    "theathletics": "athletics",
    "clevelandindians": "clevelandguardians",
    "arizonadbacks": "arizonadiamondbacks",
}


def _park_key(team_name):
    """Normalize a team name (odds-API or ESPN spelling) to a lookup key.

    Lowercases and strips every non-alphanumeric character, then applies a small
    alias map for relocations/renames. Returns "" for empty/None input.
    """
    normalized = "".join(ch for ch in str(team_name or "").lower() if ch.isalnum())
    return _ALIASES.get(normalized, normalized)


# Built once: normalized key → factors.
_NORMALIZED = {_park_key(name): factors for name, factors in MLB_PARK_FACTORS.items()}


def park_factor(team_name, kind):
    """Park factor for the home team's park and stat `kind` ("hits"|"runs").

    Returns 1.0 (neutral) for an unknown team or kind — the caller then applies
    no adjustment. Never raises.
    """
    factors = _NORMALIZED.get(_park_key(team_name))
    if not factors:
        return 1.0
    value = factors.get(kind)
    return value if isinstance(value, (int, float)) else 1.0
