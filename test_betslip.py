"""betslip deep-link + QR builders. Pure functions (no network).

Run: PYTHONIOENCODING=utf-8 python -m unittest test_betslip -v
"""
import unittest
import urllib.parse

import betslip


class DkParlayTests(unittest.TestCase):
    def test_joins_and_encodes_hash(self):
        url = betslip.dk_parlay_link(["0ML84695463_1", "0QA#tok_Q20"])
        self.assertTrue(url.startswith("https://sportsbook.draftkings.com/?outcomes="))
        self.assertIn("0ML84695463_1", url)
        self.assertIn("%23", url)        # '#' -> %23 (never a URL fragment)
        self.assertIn(",", url)          # legs comma-joined

    def test_single(self):
        self.assertEqual(betslip.dk_parlay_link(["abc_1"]),
                         "https://sportsbook.draftkings.com/?outcomes=abc_1")

    def test_empty_is_none(self):
        self.assertIsNone(betslip.dk_parlay_link([]))
        self.assertIsNone(betslip.dk_parlay_link(None))
        self.assertIsNone(betslip.dk_parlay_link([None, ""]))

    def test_repeated_variant(self):
        url = betslip.dk_parlay_link_repeated(["a_1", "b#2"])
        self.assertEqual(url.count("outcomes="), 2)   # one param per leg
        self.assertIn("outcomes=a_1", url)
        self.assertIn("outcomes=b%232", url)          # '#' encoded per leg
        self.assertIsNone(betslip.dk_parlay_link_repeated([]))


class FdParlayTests(unittest.TestCase):
    L1 = "https://sportsbook.fanduel.com/addToBetslip?marketId=42.608420704&selectionId=50194"
    L2 = "https://sportsbook.fanduel.com/addToBetslip?marketId=42.610463352&selectionId=12302543"

    def test_pair_parse(self):
        self.assertEqual(betslip._fd_pair_from_link(self.L1), ("42.608420704", "50194"))
        self.assertIsNone(betslip._fd_pair_from_link("https://x.com/no-params"))

    def test_stack_two(self):
        url = betslip.fd_parlay_link([self.L1, self.L2])
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(parsed["marketId"], ["42.608420704", "42.610463352"])
        self.assertEqual(parsed["selectionId"], ["50194", "12302543"])

    def test_empty_or_garbage_is_none(self):
        self.assertIsNone(betslip.fd_parlay_link([]))
        self.assertIsNone(betslip.fd_parlay_link(["garbage", None]))


class QrTests(unittest.TestCase):
    def test_png_bytes(self):
        png = betslip.qr_png("https://sportsbook.draftkings.com/?outcomes=abc")
        self.assertIsInstance(png, (bytes, bytearray))
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")   # PNG magic number

    def test_empty_is_none(self):
        self.assertIsNone(betslip.qr_png(""))
        self.assertIsNone(betslip.qr_png(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
