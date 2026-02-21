"""Domain model for extracted citation spans.

A ``CitationSpan`` represents a single citation found in document text,
carrying the raw match, a stable canonical key for dedup/linking, its
location in the normalized text, and a deterministic confidence score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches a CFR *section* reference: optional title, "CFR", optional §, section number
# with optional subsections.  Captures (title, section_with_subsections).
_CFR_SECTION_NORM_RE = re.compile(
    r"(\d{1,2})\s+CFR\s+§*\s*(\d+\.\d+\w*(?:\([^)]*\))*)",
)


def normalize_citation_key(raw: str) -> str:
    """Canonicalize a CFR citation string to the human-readable form.

    Maps common variants (missing §, double §§, collapsed whitespace, title
    corruption) to the canonical ``{title} CFR §{section}`` format that
    regulatory chunk metadata already uses.

    Non-section citations (Part refs, Appendix refs, non-CFR strings) are
    returned stripped and whitespace-collapsed but otherwise unchanged.
    """
    text = raw.strip()
    text = re.sub(r"\s+", " ", text)

    m = _CFR_SECTION_NORM_RE.match(text)
    if m:
        title = m.group(1) if m.group(1) != "0" else "10"
        section = m.group(2)
        return f"{title} CFR §{section}"

    return text


@dataclass(frozen=True, slots=True)
class CitationSpan:
    """A single citation extracted from document text."""

    kind: str  # "cfr" | "cfrpart" | "cfrapp" | "docket" | "adams" | "nureg" | "ris" | "gl" | "in"
    raw: str  # exact matched text
    key: str  # canonical key (stable, for dedup/linking)
    start: int  # span start in normalized text
    end: int  # span end in normalized text
    confidence: float  # 0.0-1.0 (deterministic scoring)
    source_field: str  # "title" | "content" | "metadata"
    context: str | None = None  # short context window for debugging/UI
    attrs: dict[str, object] = field(default_factory=dict)  # parsed structure
