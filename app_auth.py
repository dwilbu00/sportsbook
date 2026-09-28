"""app_auth.py — signed "remember this browser" token for the F20 password gate.

The token is ``exp.HMAC_SHA256(app_password, exp)`` stored in a browser cookie. It NEVER
contains the password itself; verifying it requires the secret, so it can't be forged, and
rotating ``app_password`` invalidates every outstanding token at once (a global logout).
Pure functions (no Streamlit) so the auth logic is unit-tested; the cookie I/O lives in
app.py via extra-streamlit-components.
"""
import hashlib
import hmac
import time

COOKIE_NAME = "odi_auth"
DEFAULT_TTL_DAYS = 14


def _sig(secret, exp):
    return hmac.new(str(secret).encode("utf-8"), str(exp).encode("utf-8"),
                    hashlib.sha256).hexdigest()


def make_token(secret, ttl_days=DEFAULT_TTL_DAYS, now=None):
    """A signed token ``'exp.sig'`` valid for ``ttl_days``. None if no secret."""
    if not secret:
        return None
    now = time.time() if now is None else now
    exp = int(now + ttl_days * 86400)
    return f"{exp}.{_sig(secret, exp)}"


def token_valid(token, secret, now=None):
    """True iff ``token`` is a well-formed, correctly-signed, unexpired token for
    ``secret``. Constant-time signature check; never raises."""
    if not token or not secret:
        return False
    try:
        exp_str, sig = str(token).split(".", 1)
        exp = int(exp_str)
    except (ValueError, AttributeError):
        return False
    now = time.time() if now is None else now
    if exp < now:
        return False
    return hmac.compare_digest(_sig(secret, exp_str), sig)
