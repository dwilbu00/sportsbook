"""betslip.py — DraftKings / FanDuel betslip deep links + QR codes.

Source of the raw links/ids: The Odds API includeLinks/includeSids (see odds_client) —
every outcome carries a ready-made `link` (used VERBATIM for singles) and a `sid`. For
PARLAYS we stack selections onto one slip:
  DK: {DK_BASE}/?outcomes=<sid1>,<sid2>,...     (sid = the outcome token; '#' url-encoded)
  FD: {FD_BASE}/addToBetslip?marketId=Mi&selectionId=Si per leg     [EXPERIMENTAL]

A link only PRE-FILLS the slip — the user taps Submit inside DK/FD. Nothing is auto-placed.
FanDuel multi-leg URL adds are unverified, so the UI always offers per-leg links as a
fallback next to the stacked one.
"""
import urllib.parse as _url

DK_BASE = "https://sportsbook.draftkings.com"
FD_BASE = "https://sportsbook.fanduel.com"


def dk_parlay_link(sids):
    """DK betslip link stacking every sid onto one slip. `sids` are the raw outcome
    tokens from The Odds API (e.g. '0ML84695463_1' or '0QA..#.._..Q20'). '#' and other
    reserved chars are percent-encoded; commas separate legs. None if no sids."""
    clean = [str(s) for s in (sids or []) if s]
    if not clean:
        return None
    return f"{DK_BASE}/?outcomes=" + _url.quote(",".join(clean), safe=",")


def _fd_pair_from_link(link):
    """(marketId, selectionId) parsed from an FD addToBetslip link, or None."""
    try:
        q = _url.parse_qs(_url.urlparse(str(link)).query)
        mid = (q.get("marketId") or [None])[0]
        sid = (q.get("selectionId") or [None])[0]
        return (mid, sid) if mid and sid else None
    except Exception:
        return None


def fd_parlay_link(fd_links):
    """Stack FD legs into one slip from their per-leg addToBetslip links. EXPERIMENTAL:
    FD multi-add via URL is unverified — keep per-leg links as the fallback. None if none
    of the links parse."""
    pairs = [p for p in (_fd_pair_from_link(x) for x in (fd_links or [])) if p]
    if not pairs:
        return None
    q = "&".join(f"marketId={_url.quote(m)}&selectionId={_url.quote(s)}" for m, s in pairs)
    return f"{FD_BASE}/addToBetslip?{q}"


def qr_png(url, scale=4):
    """A QR-code PNG (bytes) encoding `url`, or None (empty url / segno unavailable).
    Pure-Python via segno; safe to call from Streamlit."""
    if not url:
        return None
    try:
        import io
        import segno
        buf = io.BytesIO()
        segno.make(str(url), error="m").save(buf, kind="png", scale=scale, border=2)
        return buf.getvalue()
    except Exception:
        return None
