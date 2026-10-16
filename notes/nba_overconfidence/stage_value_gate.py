"""Stage the NBA value_gate suppress list as a CANDIDATE (2026-10-16 overconfidence response).

Every NBA prop loses to the de-vigged market out-of-sample (BSS < 0 for all 10 even after the
best recal map), so none may ever be flagged as a value bet. Projections + the honest
(recalibrated) P still display; bonus legs are book-priced and never read the gate.

Suppress-only on purpose: no ev_floor -> the gate shape stays the legacy edge threshold (moot
while every prop is suppressed), and NBA team markets are untouched (they're not in the list).
Any pre-existing value_gate fields are preserved. Review with
`python refit_calibration.py --sport nba --diff`, then `--promote`.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from calibration_loader import (_load_write_blob, load_calibration, save_value_gate,  # noqa: E402
                                set_candidate_mode)

SPORT = "basketball_nba"

set_candidate_mode(True)                         # write -> <sport>.candidate.json, seeded from live
props = sorted(load_calibration(SPORT))
assert len(props) == 10, props
gate = dict(_load_write_blob(SPORT).get("value_gate") or {})
gate["suppress"] = sorted(set(gate.get("suppress") or []) | set(props))
save_value_gate(SPORT, gate, meta={
    "source": "refit_calibration --recalibrate (2026-10-16; NBA method B + C props)",
    "basis": "all 10 NBA props: OOS BSS vs de-vigged market < 0 even after the winning "
             "recal map (raw ECE 0.07-0.12; |e|>5% fake edges lost 4-8% on held-out rows). "
             "Suppress until a model beats the market; recal seed keeps the displayed P honest."})
print("staged value_gate:", gate)
