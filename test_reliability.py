"""Tests for cost/reliability fixes: stale-odds fallback (P1.5), the short
negative-cache TTL for failed ESPN lookups (P1.6), and surfacing otherwise
silently-swallowed ESPN failures (P2 batch 3)."""

import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import requests

import espn_cache
import espn_client
import odds_client


def _resp(status, body=None):
    r = requests.Response()
    r.status_code = status
    if body is not None:
        r._content = json.dumps(body).encode()
    return r


class QuotaErrorDetectionTests(unittest.TestCase):
    def test_429_is_quota(self):
        self.assertTrue(odds_client._is_quota_error(_resp(429)))

    def test_401_usage_credits_is_quota(self):
        self.assertTrue(odds_client._is_quota_error(
            _resp(401, {"error_code": "OUT_OF_USAGE_CREDITS",
                        "message": "Usage limit reached"})))

    def test_401_invalid_key_is_not_quota(self):
        self.assertFalse(odds_client._is_quota_error(
            _resp(401, {"error_code": "INVALID_KEY",
                        "message": "Invalid API key"})))

    def test_500_is_not_quota(self):
        self.assertFalse(odds_client._is_quota_error(_resp(500)))


class StaleCacheFallbackTests(unittest.TestCase):
    def test_serves_bounded_stale_on_quota_and_flags_it(self):
        with patch.object(odds_client, "_read_cache", return_value=(None, None)), \
             patch.object(odds_client, "_get_with_retry", return_value=_resp(429)), \
             patch.object(odds_client, "_read_cache_expired",
                          return_value=({"id": "e1", "bookmakers": []}, 100)):
            data = odds_client.get_event_odds(
                "key", "baseball_mlb", "e1", markets="h2h")
        self.assertTrue(data.get("_stale_cache"))
        self.assertEqual(data.get("_stale_age_seconds"), 100)

    def test_refuses_stale_beyond_max_age(self):
        too_old = odds_client.STALE_CACHE_MAX_AGE + 1
        with patch.object(odds_client, "_read_cache", return_value=(None, None)), \
             patch.object(odds_client, "_get_with_retry", return_value=_resp(429)), \
             patch.object(odds_client, "_read_cache_expired",
                          return_value=({"id": "e1"}, too_old)):
            with self.assertRaises(requests.HTTPError):
                odds_client.get_event_odds(
                    "key", "baseball_mlb", "e1", markets="h2h")

    def test_invalid_key_401_never_serves_stale(self):
        bad_key = _resp(401, {"error_code": "INVALID_KEY",
                              "message": "Invalid API key"})
        with patch.object(odds_client, "_read_cache", return_value=(None, None)), \
             patch.object(odds_client, "_get_with_retry", return_value=bad_key), \
             patch.object(odds_client, "_read_cache_expired",
                          return_value=({"id": "e1"}, 100)) as expired:
            with self.assertRaises(requests.HTTPError):
                odds_client.get_event_odds(
                    "key", "baseball_mlb", "e1", markets="h2h")
        expired.assert_not_called()


class RetryOnNetworkErrorTests(unittest.TestCase):
    """P2b — transient network faults (no HTTP response, no credit spent) are
    retried instead of failing the whole call on a momentary blip."""

    def test_retries_transient_error_then_succeeds(self):
        ok = _resp(200)
        calls = []

        def flaky(*a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise requests.exceptions.ConnectionError("reset")
            return ok

        with patch.object(odds_client.time, "sleep"), \
                patch("odds_client.requests.get", side_effect=flaky):
            resp = odds_client._get_with_retry(
                "http://x", {}, max_retries=5, backoff_base=1.0)
        self.assertIs(resp, ok)
        self.assertEqual(len(calls), 3)

    def test_reraises_after_exhausting_retries(self):
        with patch.object(odds_client.time, "sleep"), \
                patch("odds_client.requests.get",
                      side_effect=requests.exceptions.Timeout("t")):
            with self.assertRaises(requests.exceptions.Timeout):
                odds_client._get_with_retry(
                    "http://x", {}, max_retries=2, backoff_base=1.0)


class NegativeCacheTTLTests(unittest.TestCase):
    def _age_file(self, path, seconds):
        old = time.time() - seconds
        os.utime(path, (old, old))

    def test_failed_lookup_recovers_after_short_ttl(self):
        with tempfile.TemporaryDirectory() as d, \
                patch.object(espn_cache, "CACHE_DIR", d):
            calls = []

            def fake_search(*a, **k):
                calls.append(1)
                return None  # miss / transient failure -> None

            with patch.object(espn_cache, "search_athlete", side_effect=fake_search):
                self.assertIsNone(espn_cache.cached_athlete_id(
                    "basketball", "nba", "Ghost Player"))
                # Re-call immediately: within the negative TTL, no refetch.
                self.assertIsNone(espn_cache.cached_athlete_id(
                    "basketball", "nba", "Ghost Player"))
                self.assertEqual(len(calls), 1)
                # Age past the short negative TTL -> must refetch (recover).
                path = espn_cache._cache_key(
                    "athlete_id", "basketball", "nba", "ghost player")
                self._age_file(path, espn_cache.NEGATIVE_TTL_HOURS * 3600 + 10)
                self.assertIsNone(espn_cache.cached_athlete_id(
                    "basketball", "nba", "Ghost Player"))
                self.assertEqual(len(calls), 2)

    def test_successful_lookup_uses_long_ttl(self):
        with tempfile.TemporaryDirectory() as d, \
                patch.object(espn_cache, "CACHE_DIR", d):
            calls = []

            def fake_search(*a, **k):
                calls.append(1)
                return {"id": "999"}

            with patch.object(espn_cache, "search_athlete", side_effect=fake_search):
                self.assertEqual(espn_cache.cached_athlete_id(
                    "basketball", "nba", "Real Player"), "999")
                # Age past the negative TTL but far within the 30-day success
                # TTL -> still served from cache, no refetch.
                path = espn_cache._cache_key(
                    "athlete_id", "basketball", "nba", "real player")
                self._age_file(path, espn_cache.NEGATIVE_TTL_HOURS * 3600 + 10)
                self.assertEqual(espn_cache.cached_athlete_id(
                    "basketball", "nba", "Real Player"), "999")
                self.assertEqual(len(calls), 1)


class EspnFailureVisibilityTests(unittest.TestCase):
    """P2 batch 3 — ESPN calls still fail closed, but no longer silently: a
    network/parse fault is emitted to stderr so an outage is distinguishable
    from a player who genuinely has no data."""

    def test_gamelog_network_error_returns_empty_and_warns(self):
        stderr = io.StringIO()
        with patch("espn_client.requests.get",
                   side_effect=requests.exceptions.ConnectionError("reset")), \
                redirect_stderr(stderr):
            result = espn_client.get_athlete_gamelog("basketball", "nba", "123")
        self.assertEqual(result, [])
        self.assertIn("get_athlete_gamelog failed", stderr.getvalue())

    def test_search_athlete_failure_returns_none_and_warns(self):
        stderr = io.StringIO()
        with patch("espn_client.requests.get",
                   side_effect=requests.exceptions.Timeout("t")), \
                redirect_stderr(stderr):
            result = espn_client.search_athlete("basketball", "nba", "Ghost")
        self.assertIsNone(result)
        # Both the primary and fallback endpoints failed; each is surfaced.
        self.assertIn("search_athlete", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
