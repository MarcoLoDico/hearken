"""Vocabulary files (names, jargon, domain terms).

Plain UTF-8 text, one term per line. Blank lines and lines starting with '#'
are ignored. Terms are passed verbatim to the engines (spelling and casing
matter: write "Kullback-Leibler", "LaTeX", "Navier-Stokes").
"""

from __future__ import annotations

from pathlib import Path

from .errors import ConfigError

MAX_TERMS = 300  # LLM context and trie size both degrade beyond a few hundred


def load(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    p = Path(path)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read vocabulary file {p}: {e}") from e
    terms = []
    for line in lines:
        t = line.strip()
        if t and not t.startswith("#"):
            terms.append(t)
    terms = list(dict.fromkeys(terms))
    if len(terms) > MAX_TERMS:
        raise ConfigError(f"vocabulary has {len(terms)} terms; keep it under {MAX_TERMS} (most specific terms first)")
    return terms
