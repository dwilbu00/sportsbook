"""app_auth signed "remember this browser" token — the persistent-login security core.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_app_auth -v
"""
import unittest

import app_auth

SECRET = "hunter2"
T = 1_000_000.0          # fixed clock so expiry math is deterministic


class TokenTests(unittest.TestCase):
    def test_round_trip_valid(self):
        t = app_auth.make_token(SECRET, ttl_days=14, now=T)
        self.assertTrue(app_auth.token_valid(t, SECRET, now=T))

    def test_boundary_expiry(self):
        t = app_auth.make_token(SECRET, ttl_days=1, now=T)
        self.assertTrue(app_auth.token_valid(t, SECRET, now=T + 86400 - 10))   # still valid
        self.assertFalse(app_auth.token_valid(t, SECRET, now=T + 86400 + 10))  # expired

    def test_wrong_secret_invalid(self):
        t = app_auth.make_token(SECRET, ttl_days=14, now=T)
        self.assertFalse(app_auth.token_valid(t, "different", now=T))

    def test_tampered_signature_invalid(self):
        t = app_auth.make_token(SECRET, ttl_days=14, now=T)
        exp, sig = t.split(".", 1)
        self.assertFalse(app_auth.token_valid(f"{exp}.{'0' * len(sig)}", SECRET, now=T))

    def test_forged_expiry_invalid(self):
        # Extending the expiry without the secret can't produce a valid signature.
        t = app_auth.make_token(SECRET, ttl_days=1, now=T)
        exp, sig = t.split(".", 1)
        forged = f"{int(exp) + 30 * 86400}.{sig}"
        self.assertFalse(app_auth.token_valid(forged, SECRET, now=T))

    def test_malformed_invalid(self):
        for bad in ("", None, "no-dot", "abc.def", "123", ".", "123.", ".sig"):
            self.assertFalse(app_auth.token_valid(bad, SECRET, now=T))

    def test_no_secret(self):
        self.assertIsNone(app_auth.make_token("", 14))
        self.assertIsNone(app_auth.make_token(None, 14))
        self.assertFalse(app_auth.token_valid("anything", "", now=T))
        self.assertFalse(app_auth.token_valid("anything", None, now=T))


if __name__ == "__main__":
    unittest.main(verbosity=2)
