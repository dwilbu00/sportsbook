"""nba_names — odds-feed NBA player names -> warehouse name_norm (replaces the ESPN fuzzy search
as the safety net for spelling differences). Covers every real 2022-26 mismatch, the namesake
safety rules, and the two consumers that grade/join by name.
"""
import unittest
from unittest.mock import patch

import nba_names as N

# the warehouse spellings the 28 real odds-feed mismatches must land on
WAREHOUSE = {
    "jimmy butler iii", "nic claxton", "bub carrington", "derrick jones jr", "herbert jones",
    "isaiah stewart", "ronald holland ii", "robert williams iii", "bruce brown",
    "vince williams jr", "moritz wagner", "cam thomas", "paul reed", "marvin bagley iii",
    "lonnie walker iv", "gg jackson", "kj martin", "kevin knox ii", "mo bamba",
    "dennis smith jr", "trey jemison iii", "marcus morris sr", "xavier tillman", "jay huff",
    "reggie bullock jr", "tj warren", "hansen yang", "matt hurt",
    # neighbours that must NOT be confused
    "jalen williams", "jaylin williams", "markieff morris", "kelan martin",
}
REAL_CASES = {
    "Jimmy Butler": "jimmy butler iii", "Nicolas Claxton": "nic claxton",
    "Carlton Carrington": "bub carrington", "Derrick Jones": "derrick jones jr",
    "Herb Jones": "herbert jones", "Isaiah Stewart II": "isaiah stewart",
    "Ron Holland": "ronald holland ii", "Robert Williams": "robert williams iii",
    "Bruce Brown Jr": "bruce brown", "Vincent Williams Jr": "vince williams jr",
    "Moe Wagner": "moritz wagner", "Cameron Thomas": "cam thomas", "Paul Reed Jr": "paul reed",
    "Marvin Bagley": "marvin bagley iii", "Lonnie Walker": "lonnie walker iv",
    "Gregory Jackson": "gg jackson", "Kenyon Martin Jr.": "kj martin",
    "Kevin Knox": "kevin knox ii", "Mohamed Bamba": "mo bamba", "Dennis Smith": "dennis smith jr",
    "Trey Jemison": "trey jemison iii", "Marcus Morris": "marcus morris sr",
    "Xavier Tillman, Sr.": "xavier tillman", "James Huff": "jay huff",
    "Reggie Bullock": "reggie bullock jr", "T. J. Warren": "tj warren",
    "Yang Hansen": "hansen yang", "Matthew Hurt": "matt hurt",
}


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.r = N.Resolver(WAREHOUSE)

    def test_every_real_mismatch(self):
        for odds_name, want in REAL_CASES.items():
            self.assertEqual(self.r.resolve(odds_name), want, odds_name)

    def test_exact_names_untouched(self):
        for nn in ("jalen williams", "jaylin williams", "markieff morris", "kelan martin"):
            self.assertEqual(self.r.resolve(nn.title()), nn)

    def test_prefix_is_not_a_given_name_match(self):
        r = N.Resolver({"jaylen smith"})
        self.assertIsNone(r.resolve("Jay Smith"))            # merely starts like it
        self.assertIsNone(N.Resolver({"jalen williams"}).resolve("Jaylin Williams"))

    def test_ambiguous_returns_none(self):
        # two warehouse players reduce to the same suffix-free key -> never guess
        r = N.Resolver({"gary trent jr", "gary trent sr"})
        self.assertIsNone(r.resolve("Gary Trent"))
        r2 = N.Resolver({"nicolas batum", "nicholas batum"})
        self.assertIsNone(r2.resolve("Nic Batum"))

    def test_absent_and_blank(self):
        self.assertIsNone(self.r.resolve("Nobody Here"))
        self.assertIsNone(self.r.resolve(""))
        self.assertIsNone(self.r.resolve(None))

    def test_key_rules(self):
        self.assertEqual(N._key("t j warren"), "tj warren")
        self.assertEqual(N._key("jimmy butler iii"), "jimmy butler")
        self.assertEqual(N._key("p j washington jr"), "pj washington")


class ConsumerTests(unittest.TestCase):
    def test_calibration_join_resolves_odds_spelling(self):
        import book_line_calibration as blc
        prior = [{"season": 2025, "game_date": f"2025-01-{d:02d}", "played": True, "MIN": 30.0,
                  "points": 20.0} for d in range(1, 13)]
        test_game = {"season": 2025, "game_date": "2025-01-20", "played": True, "MIN": 34.0,
                     "points": 31.0}
        bulk = {"jimmy butler iii": [test_game] + list(reversed(prior))}
        line = {"player": "Jimmy Butler", "prop_key": "player_points", "game_date": "2025-01-20",
                "line": 24.5}
        with patch("db_store.enabled", return_value=True), \
             patch("nba_warehouse.calib_gamelogs_bulk", return_value=bulk):
            out = blc._nba_grade_book_lines([line])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["actual"], 31.0)

    def test_grading_frame_resolves_odds_spelling(self):
        import pandas as pd
        import recalibration as rc
        df = pd.DataFrame([{"player_norm": "nic claxton", "game_date": "2025-01-20",
                            "played": True, "minutes": 30.0, "rebounds": 11.0}])
        self.assertEqual(rc._nba_canonical(df, "Nicolas Claxton"), "nic claxton")
        row = rc._nba_frame_match(df, rc._nba_canonical(df, "Nicolas Claxton"), "2025-01-20")
        self.assertEqual(row["rebounds"], 11.0)
        self.assertEqual(rc._nba_canonical(df, "Somebody Else"), "somebody else")  # plain fallback


if __name__ == "__main__":
    unittest.main()
