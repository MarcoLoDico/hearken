"""Comparison-only text normalisation and word-sequence alignment.

Nothing here rewrites transcript text. Normalised forms are used only to
decide whether two hypotheses (or a hypothesis and a reference) say the same
thing, e.g. "10,000" vs "ten thousand", "colour" vs "color", "it's" vs "it is".

English uses Whisper's EnglishTextNormalizer (the de-facto standard for WER
reporting, also used by the Open ASR Leaderboard). Other languages fall back
to Whisper's BasicTextNormalizer behaviour (lower-case, strip punctuation).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

_PUNCT = re.compile(r"[^\w\s']|_", re.UNICODE)


def _basic(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = _PUNCT.sub(" ", s)
    return " ".join(s.replace("'", "").split())


@lru_cache(maxsize=1)
def _english():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
        return EnglishTextNormalizer()
    except ImportError:  # optional dependency
        return None


def normalize(text: str, language: str = "en") -> str:
    if language == "en":
        n = _english()
        if n is not None:
            return " ".join(n(text).split())
    return _basic(text)


@lru_cache(maxsize=65536)
def _word_key(text: str, language: str) -> tuple[str, ...]:
    return tuple(normalize(text, language).split())


@dataclass
class DiffRegion:
    a0: int  # word index range in sequence A: [a0, a1)
    a1: int
    b0: int  # word index range in sequence B: [b0, b1)
    b1: int


def diff_words(a: list[str], b: list[str], language: str = "en") -> list[DiffRegion]:
    """Minimal-edit alignment of two word sequences; return regions that differ.

    Regions whose normalised text is identical when compared as a phrase
    (spelling/number-format differences) are dropped.
    """
    from rapidfuzz.distance import Levenshtein

    ta, oa = _flatten(a, language)
    tb, ob = _flatten(b, language)
    regions: list[DiffRegion] = []
    for op in Levenshtein.opcodes(ta, tb):
        if op.tag == "equal":
            continue
        a0, a1 = _owner_range(oa, op.src_start, op.src_end, len(a))
        b0, b1 = _owner_range(ob, op.dest_start, op.dest_end, len(b))
        if regions and a0 <= regions[-1].a1 and b0 <= regions[-1].b1:
            regions[-1].a1, regions[-1].b1 = max(a1, regions[-1].a1), max(b1, regions[-1].b1)
        else:
            regions.append(DiffRegion(a0, a1, b0, b1))
    out = []
    for r in regions:
        # Re-check with one word of context on each side so number phrases
        # ("twenty five" vs "25") are normalised as a unit.
        ctx_a = " ".join(a[max(0, r.a0 - 1):r.a1 + 1])
        ctx_b = " ".join(b[max(0, r.b0 - 1):r.b1 + 1])
        if normalize(ctx_a, language).replace(" ", "") == normalize(ctx_b, language).replace(" ", ""):
            continue
        out.append(r)
    return out


def _flatten(words: list[str], language: str) -> tuple[list[str], list[int]]:
    toks, owner = [], []
    for i, w in enumerate(words):
        for t in _word_key(w, language):
            toks.append(t)
            owner.append(i)
    return toks, owner


def _owner_range(owner: list[int], s: int, e: int, n_words: int) -> tuple[int, int]:
    """Map a token range back to the word range that produced it."""
    if e > s:
        return owner[s], owner[e - 1] + 1
    # Empty range (pure insertion on the other side): position between words.
    pos = owner[s] if s < len(owner) else n_words
    return pos, pos
