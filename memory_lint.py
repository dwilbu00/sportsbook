#!/usr/bin/env python3
"""memory_lint.py — deterministic hygiene checks for the project memory files
(deploy/memory/*.md) + a generated follow-up prompt for the SEMANTIC pass a script
can't do (contradictions, superseded ideas, prune judgment).

The split is deliberate: a script is *better* than a human skim at the mechanical,
exhaustive, repeatable checks below; it is useless at judging meaning. So it prints a
ready-to-paste prompt that hands the meaning-work to Cal, pre-loaded with exactly what
the mechanical pass surfaced.

Deterministic checks:
  * dead repo refs  — a .py/.sql file or ODI_* flag named in memory that no longer exists
                      in the codebase (enforces "verify it still exists before recommending")
  * dangling links  — [[name]] with no matching memory (allowed by the rules — a marker —
                      but listed so you can create the target or fix a typo)
  * index drift     — memory .md files missing from / orphaned by MEMORY.md
  * stale dates     — relative date words the rules say to convert to absolute
  * bloat           — oversized files / mega-lines (e.g. ACTIVE.md)
  * duplicates      — sentences repeated near-verbatim across files (one-fact-one-place)

Run:  PYTHONIOENCODING=utf-8 python memory_lint.py
      python memory_lint.py --memory-dir memory --repo-dir .
Read-only; never edits. Exit 0 always (it's a report). Stdlib only.
"""
import argparse
import difflib
import os
import re
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
REL_DATE_RE = re.compile(
    r"\b(yesterday|today|tomorrow|last week|next week|this week|last month|next month|"
    r"this month|as of (?:now|today)|right now)\b", re.I)
WIKILINK_RE = re.compile(r"\[\[([A-Za-z0-9][A-Za-z0-9_-]*)\]\]")
# Path segments have no interior dots, so "a.py/b.py" yields two matches, not one span.
PYREF_RE = re.compile(r"((?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.(?:py|sql))")
FLAG_RE = re.compile(r"\bODI_[A-Z0-9_]+\b")
MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+\.md)\)")
_STOP_SENT = 60          # min chars for a sentence to be a dup candidate
_BIG_FILE = 200          # lines
_BIG_LINE = 2000         # chars (a single blockquote paragraph over this = fold it)


def parse_frontmatter(text):
    """(meta, body). Frontmatter = a leading --- ... --- block; simple key: value scan."""
    meta = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].splitlines():
                m = re.match(r"\s*([A-Za-z_]+):\s*(.*)", line)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
            body = text[end + 4:]
    return meta, body


def load_files(memory_dir):
    out = {}
    for fn in sorted(os.listdir(memory_dir)):
        if fn.endswith(".md"):
            with open(os.path.join(memory_dir, fn), encoding="utf-8") as f:
                out[fn] = f.read()
    return out


def repo_index(repo_dir):
    """(source_blob, existing_paths) for dead-ref checks: all .py/.sql text concatenated
    + the set of every file's rel-path and basename."""
    blob, paths = [], set()
    skip = {".git", "__pycache__", "memory", "node_modules", ".venv", "audits",
            ".streamlit", "warehouse_mirror_data"}
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), repo_dir).replace("\\", "/")
            paths.add(rel)
            paths.add(fn)
            if fn.endswith((".py", ".sql")):
                try:
                    with open(os.path.join(root, fn), encoding="utf-8",
                              errors="ignore") as f:
                        blob.append(f.read())
                except OSError:
                    pass
    return "\n".join(blob), paths


def _sentences(body):
    """Rough sentence split that also breaks the giant one-line blockquotes in ACTIVE.md."""
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", body):
        s = re.sub(r"\s+", " ", chunk).strip(" >-*#")
        if len(s) >= _STOP_SENT:
            yield s


def lint(memory_dir, repo_dir):
    files = load_files(memory_dir)
    blob, paths = repo_index(repo_dir)
    names = set()                                    # every memory `name:` slug
    bodies = {}
    for fn, text in files.items():
        meta, body = parse_frontmatter(text)
        bodies[fn] = body
        if meta.get("name"):
            names.add(meta["name"])

    f = defaultdict(list)                            # findings by category

    # index (MEMORY.md) drift
    index = files.get("MEMORY.md", "")
    linked = set(MD_LINK_RE.findall(index))
    for fn in files:
        if fn == "MEMORY.md":
            continue
        if fn not in linked:
            f["index_orphan"].append(f"{fn}: not linked from MEMORY.md")
    for tgt in linked:
        if tgt not in files:
            f["index_broken"].append(f"MEMORY.md -> {tgt} (file missing)")

    for fn, body in bodies.items():
        # dead repo references (.py/.sql)
        for ref in sorted(set(PYREF_RE.findall(body))):
            base = ref.split("/")[-1]
            if ref not in paths and base not in paths:
                f["dead_file_ref"].append(f"{fn}: `{ref}` — not found in repo")
        # dead ODI_* flags
        for flag in sorted(set(FLAG_RE.findall(body))):
            if flag not in blob:
                f["dead_flag"].append(f"{fn}: {flag} — not found in any .py")
        # dangling [[wikilinks]]
        for link in sorted(set(WIKILINK_RE.findall(body))):
            if link not in names:
                f["dangling_link"].append(f"{fn}: [[{link}]] — no memory with that name")
        # stale relative dates
        for m in set(x.lower() for x in REL_DATE_RE.findall(body)):
            f["stale_date"].append(f"{fn}: relative date '{m}' (make absolute)")
        # bloat
        lines = body.splitlines()
        longest = max((len(x) for x in lines), default=0)
        if len(lines) > _BIG_FILE or longest > _BIG_LINE:
            f["bloat"].append(
                f"{fn}: {len(lines)} lines, longest line {longest} chars"
                + (" — mega-line(s), consider splitting/folding" if longest > _BIG_LINE
                   else ""))

    # cross-file duplicate sentences (one-fact-one-place). Bucket by (length band, 16-char
    # prefix) so difflib only runs within small candidate groups — bounds the cost on
    # ACTIVE.md's mega-line blockquotes (a naive all-pairs pass is O(n^2) and hangs).
    buckets = defaultdict(list)
    for fn, body in bodies.items():
        for s in _sentences(body):
            key = re.sub(r"[^a-z0-9 ]", "", s.lower())
            buckets[(len(key) // 20, key[:16])].append((fn, s, key))
    dup_seen = set()
    for group in buckets.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            fn_i, s_i, k_i = group[i]
            for j in range(i + 1, len(group)):
                fn_j, _s_j, k_j = group[j]
                if fn_i == fn_j:
                    continue
                if difflib.SequenceMatcher(None, k_i, k_j).ratio() > 0.90:
                    sig = (tuple(sorted((fn_i, fn_j))), k_i[:40])
                    if sig not in dup_seen:
                        dup_seen.add(sig)
                        f["duplicate"].append(f"{fn_i} ~ {fn_j}: \"{s_i[:90]}...\"")
                    break
    return f


def _section(title, items, cap=40):
    if not items:
        return ""
    shown = items[:cap]
    more = f"\n    ... (+{len(items) - cap} more)" if len(items) > cap else ""
    return f"\n{title} ({len(items)}):\n" + "\n".join(f"  - {x}" for x in shown) + more


def report(f):
    print("=" * 78)
    print("  MEMORY LINT — deterministic findings")
    print("=" * 78)
    order = [("DEAD FILE REFERENCES", "dead_file_ref"),
             ("DEAD ODI_* FLAGS", "dead_flag"),
             ("INDEX: ORPHANED FILES", "index_orphan"),
             ("INDEX: BROKEN LINKS", "index_broken"),
             ("DANGLING [[WIKILINKS]]", "dangling_link"),
             ("STALE RELATIVE DATES", "stale_date"),
             ("BLOAT", "bloat"),
             ("DUPLICATE SENTENCES ACROSS FILES", "duplicate")]
    total = 0
    for title, key in order:
        s = _section(title, f.get(key, []))
        total += len(f.get(key, []))
        if s:
            print(s)
    if not total:
        print("\n  clean — no deterministic issues found.")
    print(f"\n  total findings: {total}")
    return total


def followup_prompt(f):
    """Emit a ready-to-paste prompt for Cal's semantic pass, seeded with what the
    deterministic pass found (so the meaning-work is targeted, not a blind re-read)."""
    def names(key):
        return sorted({x.split(":")[0] for x in f.get(key, [])})
    dead = f.get("dead_file_ref", []) + f.get("dead_flag", [])
    dup = f.get("duplicate", [])
    bloated = names("bloat")
    stale = names("stale_date")
    lines = [
        "Run a SEMANTIC maintenance pass on deploy/memory/*.md. `memory_lint.py` already",
        "handled the mechanical checks; do the judgment it can't, then show me a concrete",
        "edit plan (per file) BEFORE applying, and keep the discipline in CLAUDE.md /",
        "MEMORY.md (one fact one place; result supersedes idea; wishlist append/transfer-only).",
        "",
        "1. CONTRADICTIONS / SUPERSEDED: read across the files and find claims that conflict",
        "   or an idea a later RESULT should replace (e.g. \"test X\" vs \"tested X -> Y\").",
        "   Delete the idea, keep the result.",
    ]
    if dead:
        lines += ["",
                  "2. DEAD REFERENCES the linter flagged — confirm each is truly gone, then",
                  "   fix or remove the mention (don't recommend a file/flag that no longer exists):"]
        lines += [f"   - {x}" for x in dead[:30]]
    if dup:
        lines += ["",
                  "3. NEAR-DUPLICATE sentences across files — decide the ONE home for each fact,",
                  "   dedup the rest, and check the duplicates don't actually disagree:"]
        lines += [f"   - {x}" for x in dup[:20]]
    if bloated:
        lines += ["",
                  f"4. BLOAT: {', '.join(bloated)} — fold SHIPPED/COMPLETED in-flight items into",
                  "   their domain files and slim ACTIVE.md's mega-line blockquotes to what's live."]
    if stale:
        lines += ["",
                  f"5. STALE RELATIVE DATES in: {', '.join(stale)} — convert to absolute dates."]
    dangling = f.get("dangling_link", [])
    if dangling:
        lines += ["",
                  "6. DANGLING [[WIKILINKS]] — allowed as forward-markers, but many point to the",
                  "   OLD ~/.claude memory names, not the repo's files. Fix case mismatches",
                  "   ([[ACTIVE]] -> [[active]]) and de-link (plain text) or create the targets",
                  "   for legacy refs that will never exist as repo memories:"]
        lines += [f"   - {x}" for x in dangling[:15]]
        if len(dangling) > 15:
            lines += [f"   ... (+{len(dangling) - 15} more)"]
    lines += ["",
              "Propose the edits as a plan first; I'll approve before you write."]
    print("\n\n" + "=" * 78)
    print("  FOLLOW-UP PROMPT  —  copy everything between the lines to Cal")
    print("=" * 78)
    print("\n".join(lines))
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memory-dir", default=os.path.join(_HERE, "memory"))
    ap.add_argument("--repo-dir", default=_HERE)
    ap.add_argument("--no-prompt", action="store_true",
                    help="skip the generated follow-up prompt (findings only)")
    a = ap.parse_args()
    f = lint(a.memory_dir, a.repo_dir)
    report(f)
    if not a.no_prompt:
        followup_prompt(f)


if __name__ == "__main__":
    main()
