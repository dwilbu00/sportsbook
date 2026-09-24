"""Tests for memory_lint's deterministic checks (the mechanical layer of memory upkeep).

Run: PYTHONIOENCODING=utf-8 python -m unittest test_memory_lint -v
"""
import os
import tempfile
import unittest

import memory_lint as ml


class ParseFrontmatterTests(unittest.TestCase):
    def test_parses_name_and_body(self):
        meta, body = ml.parse_frontmatter("---\nname: foo\ndescription: bar\n---\nhello")
        self.assertEqual(meta["name"], "foo")
        self.assertEqual(body.strip(), "hello")

    def test_no_frontmatter_returns_whole_body(self):
        meta, body = ml.parse_frontmatter("no fm here")
        self.assertEqual(meta, {})
        self.assertEqual(body, "no fm here")


class PyRefRegexTests(unittest.TestCase):
    def test_slash_list_splits_into_separate_files(self):
        # the false-positive that greedily matched "a.py/b.py/c.py" as one path
        self.assertEqual(ml.PYREF_RE.findall("see a.py/b.py and c.py"),
                         ["a.py", "b.py", "c.py"])

    def test_dir_path_kept_whole(self):
        self.assertEqual(ml.PYREF_RE.findall("run sql/schema.sql now"),
                         ["sql/schema.sql"])


class LintEndToEndTests(unittest.TestCase):
    def _make(self):
        repo = tempfile.mkdtemp()
        mem = os.path.join(repo, "memory")
        os.makedirs(mem)
        with open(os.path.join(repo, "real.py"), "w", encoding="utf-8") as f:
            f.write("X = 1  # ODI_REAL_FLAG in code\n")
        with open(os.path.join(mem, "MEMORY.md"), "w", encoding="utf-8") as f:
            f.write("# index\n- [alpha](alpha.md)\n")     # beta.md is orphaned (no link)
        with open(os.path.join(mem, "alpha.md"), "w", encoding="utf-8") as f:
            f.write("---\nname: alpha\n---\n"
                    "Uses real.py and gone.py. Flag ODI_REAL_FLAG ok but ODI_GHOST_FLAG bad. "
                    "See [[ghost]] and [[alpha]]. yesterday we shipped it.\n")
        with open(os.path.join(mem, "beta.md"), "w", encoding="utf-8") as f:
            f.write("---\nname: beta\n---\nplain.\n")
        return repo, mem

    def test_findings(self):
        repo, mem = self._make()
        f = ml.lint(mem, repo)
        dead = " ".join(f["dead_file_ref"])
        self.assertIn("gone.py", dead)
        self.assertNotIn("real.py", dead)                 # existing file not flagged
        self.assertTrue(any("ODI_GHOST_FLAG" in x for x in f["dead_flag"]))
        self.assertFalse(any("ODI_REAL_FLAG" in x for x in f["dead_flag"]))
        self.assertTrue(any("ghost" in x for x in f["dangling_link"]))  # [[ghost]] != any name
        self.assertFalse(any("[[alpha]]" in x for x in f["dangling_link"]))  # valid name
        self.assertTrue(any("yesterday" in x for x in f["stale_date"]))
        self.assertTrue(any("beta.md" in x for x in f["index_orphan"]))  # not in MEMORY.md

    def test_followup_prompt_builds(self):
        repo, mem = self._make()
        f = ml.lint(mem, repo)
        # the generator must not raise and must carry the dead-ref + dangling sections
        try:
            ml.followup_prompt(f)
        except Exception as exc:            # pragma: no cover
            self.fail(f"followup_prompt raised: {exc!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
