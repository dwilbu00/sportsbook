"""nba_names.py — resolve an odds-feed NBA player name to the warehouse's name_norm.

The stats.nba.com warehouse keys players by ``db_store.normalize_name``. Odds feeds spell ~2.6%
of NBA prop lines differently (28 distinct names over 2022-26), which used to be papered over by
ESPN's fuzzy athlete search. This replaces that safety net with deterministic, auditable rules,
tried in order — each later rule must match EXACTLY ONE warehouse name (never guess between
namesakes):

  1. exact name_norm
  2. explicit alias (verified by hand: different given names / nicknames)
  3. generational suffix + initials insensitive key   ("jimmy butler" = "jimmy butler iii",
                                                       "t j warren" = "tj warren")
  4. same tokens, different order                      ("yang hansen" = "hansen yang")
  5. same surname, equivalent given names from GIVEN_NAME_EQUIV ("nic" = "nicolas",
     "cam" = "cameron") — an explicit table, never a raw prefix ("jay" != "jaylen")

Build a Resolver from the set of warehouse names the caller is matching against (the player dim,
a season frame, the mirror corpus) so every lookup site resolves identically.
"""
from __future__ import annotations

import db_store

SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Equivalent given names (short form -> formal forms). Rule 5 bridges two first names only when
# they are listed here, so a name that merely STARTS like another ("jay"/"jaylen") never matches.
_GIVEN = {
    "nic": ("nicolas", "nicholas", "nick"), "nick": ("nicholas", "nicolas"),
    "cam": ("cameron",), "herb": ("herbert",), "matt": ("matthew",),
    "mo": ("mohamed", "mohammed", "mohammad", "maurice"), "ron": ("ronald",),
    "vince": ("vincent",), "mike": ("michael",), "chris": ("christopher",),
    "alex": ("alexander", "alexandre"), "ben": ("benjamin",), "will": ("william",),
    "bill": ("william",), "josh": ("joshua",), "jon": ("jonathan",), "dan": ("daniel",),
    "danny": ("daniel",), "tony": ("anthony",), "rob": ("robert",), "bob": ("robert",),
    "zach": ("zachary", "zachariah"), "sam": ("samuel",), "tim": ("timothy",),
    "tom": ("thomas",), "pat": ("patrick",), "jim": ("james",), "jimmy": ("james",),
    "greg": ("gregory",), "ed": ("edward",), "andy": ("andrew",), "drew": ("andrew",),
    "jake": ("jacob",), "joe": ("joseph",), "steve": ("steven", "stephen"),
    "dave": ("david",), "fred": ("frederick",), "max": ("maxwell", "maximilian"),
}
GIVEN_NAME_EQUIV = {}
for _short, _forms in _GIVEN.items():
    for _f in _forms:
        GIVEN_NAME_EQUIV.setdefault(_short, set()).add(_f)
        GIVEN_NAME_EQUIV.setdefault(_f, set()).add(_short)


def _given_equiv(a, b):
    return a == b or b in GIVEN_NAME_EQUIV.get(a, ())


# odds-feed name_norm -> warehouse name_norm, for names no general rule can bridge (verified
# against the stats.nba.com player dim, 2026-10).
ALIASES = {
    "carlton carrington": "bub carrington",
    "kenyon martin jr": "kj martin",
    "kenyon martin": "kj martin",
    "moe wagner": "moritz wagner",
    "gregory jackson": "gg jackson",
    "james huff": "jay huff",
}


def _key(nn):
    """Suffix-free, initials-collapsed match key: drop generational suffix tokens and merge runs
    of single-letter tokens ("t j warren" -> "tj warren")."""
    toks = [t for t in nn.split() if t not in SUFFIXES]
    out, run = [], ""
    for t in toks:
        if len(t) == 1:
            run += t
            continue
        if run:
            out.append(run)
            run = ""
        out.append(t)
    if run:
        out.append(run)
    return " ".join(out)


class Resolver:
    """Odds-feed name -> warehouse name_norm over a fixed set of known warehouse names."""

    def __init__(self, known_norms):
        self.known = {n for n in known_norms if n}
        self._by_key, self._by_tokens, self._by_rest = {}, {}, {}
        for n in self.known:
            k = _key(n)
            self._by_key.setdefault(k, set()).add(n)
            self._by_tokens.setdefault(tuple(sorted(k.split())), set()).add(n)
            toks = k.split()
            if len(toks) >= 2:
                self._by_rest.setdefault(" ".join(toks[1:]), []).append((toks[0], n))

    def resolve(self, name):
        """The warehouse name_norm for ``name``, or None when absent or ambiguous."""
        nn = db_store.normalize_name(name or "")
        if not nn:
            return None
        if nn in self.known:
            return nn
        alias = ALIASES.get(nn) or ALIASES.get(_key(nn))
        if alias in self.known:
            return alias
        k = _key(nn)
        hit = self._by_key.get(k)
        if hit and len(hit) == 1:
            return next(iter(hit))
        hit = self._by_tokens.get(tuple(sorted(k.split())))
        if hit and len(hit) == 1:
            return next(iter(hit))
        toks = k.split()
        if len(toks) >= 2:
            first = toks[0]
            cands = {n for f, n in self._by_rest.get(" ".join(toks[1:]), ())
                     if f != first and _given_equiv(first, f)}
            if len(cands) == 1:
                return next(iter(cands))
        return None


def resolve_in(name, known_norms):
    """One-shot resolve against an iterable of known name_norms (builds a Resolver)."""
    return Resolver(known_norms).resolve(name)
