"""Domain model for extracted citation spans.

A ``CitationSpan`` represents a single citation found in document text,
carrying the raw match, a stable canonical key for dedup/linking, its
location in the normalized text, and a deterministic confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
