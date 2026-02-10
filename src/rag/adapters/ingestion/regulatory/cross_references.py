"""CFR cross-reference detection, rewriting, and citation sorting utilities.

These are the shared building blocks used by both the first-generation
normalizer (``cfr.py``) and the second-generation pipeline
(``ecfr_parser`` + ``normalizer``).
"""

from __future__ import annotations

import re

# --- Regex toolkit ----------------------------------------------------------
# Each pattern targets a different surface form of a CFR citation.

# Matches any wikilink ``[[...]]`` regardless of content.
_WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")

# Matches a wikilink whose body is a CFR citation, e.g. ``[[§ 50.36]]``.
_WIKILINK_CITATION_RE = re.compile(
    r"\[\[\s*(?:(?P<title>\d+)\s*CFR\s*)?§\s*(?P<section>\d+\.\d+[A-Za-z0-9-]*(?:\([A-Za-z0-9]+\))*)\s*\]\]"
)

# Matches a bare (non-wikilink) CFR reference, e.g. ``§ 50.36`` or ``10 CFR § 50.36``.
_CFR_REFERENCE_RE = re.compile(
    r"(?:(?P<title>\d+)\s*CFR\s*)?§+\s*(?P<section>\d+\.\d+[A-Za-z0-9-]*(?:\([A-Za-z0-9]+\))*)"
)


# --- Citation helpers -------------------------------------------------------


def _strip_subsection_suffix(section: str) -> str:
    """Remove parenthesized subsection markers: ``50.36(a)(1)`` → ``50.36``."""
    return re.sub(r"\([A-Za-z0-9]+\)", "", section).strip()


def section_sort_key(section: str) -> tuple[int, int, str]:
    """Return a sort key that orders section numbers naturally.

    ``50.2`` < ``50.10`` < ``50.10a`` — numeric parts sort numerically, and
    alphabetic suffixes sort lexicographically within the same number.
    """
    major_raw, _, rest = section.partition(".")
    try:
        major = int(major_raw)
    except ValueError:
        major = 10**9

    m = re.match(r"(?P<num>\d+)(?P<suffix>[A-Za-z0-9-]*)", rest)
    if m is None:
        return (major, 10**9, rest)

    return (major, int(m.group("num")), m.group("suffix"))


def _canonical_citation(title: str | None, section: str) -> str:
    """Build the canonical form ``<title> CFR §<section>`` used as a stable key."""
    title_num = title or "10"
    canonical_section = _strip_subsection_suffix(section)
    return f"{title_num} CFR §{canonical_section}"


def _split_outside_wikilinks(text: str) -> list[tuple[str, bool]]:
    """Split *text* into ``(segment, is_wikilink)`` pairs.

    This lets callers rewrite only the non-wikilink portions, avoiding
    double-wrapping citations that are already wikilinked.
    """
    parts: list[tuple[str, bool]] = []
    cursor = 0
    for match in _WIKILINK_RE.finditer(text):
        if match.start() > cursor:
            parts.append((text[cursor : match.start()], False))
        parts.append((match.group(0), True))
        cursor = match.end()
    if cursor < len(text):
        parts.append((text[cursor:], False))
    if not parts:
        return [(text, False)]
    return parts


# --- Public API -------------------------------------------------------------


def _citation_sort_key(citation: str) -> tuple[int, int, str]:
    """Sort key for a full citation string like ``10 CFR §50.36``."""
    return section_sort_key(citation.split("§", 1)[1].strip())


def extract_cross_references(text: str) -> list[str]:
    """Return canonical CFR citation keys referenced in *text*."""
    refs: set[str] = set()

    for match in _WIKILINK_CITATION_RE.finditer(text):
        refs.add(_canonical_citation(match.group("title"), match.group("section")))

    scrubbed = _WIKILINK_RE.sub(" ", text)
    for match in _CFR_REFERENCE_RE.finditer(scrubbed):
        refs.add(_canonical_citation(match.group("title"), match.group("section")))

    return sorted(refs, key=_citation_sort_key)


def rewrite_cross_references_to_wikilinks(text: str) -> str:
    """Rewrite CFR references like ``§ 50.36`` to deterministic wikilinks."""

    def rewrite_segment(segment: str) -> str:
        def repl(match: re.Match[str]) -> str:
            citation = _canonical_citation(match.group("title"), match.group("section"))
            return f"[[{citation}]]"

        return _CFR_REFERENCE_RE.sub(repl, segment)

    out: list[str] = []
    for segment, is_wikilink in _split_outside_wikilinks(text):
        out.append(segment if is_wikilink else rewrite_segment(segment))
    return "".join(out)


__all__ = [
    "extract_cross_references",
    "rewrite_cross_references_to_wikilinks",
    "section_sort_key",
]
