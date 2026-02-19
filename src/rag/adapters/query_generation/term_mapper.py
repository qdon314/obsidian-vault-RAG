"""Term-to-regulation mapping for case-derived query generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TermMatch:
    """A dictionary term found in case content."""

    term: str
    citations: list[str]
    frequency: int


@dataclass(frozen=True, slots=True)
class TermMapper:
    """Loads a term-to-regulation dictionary and matches terms in text."""

    _terms: dict[str, list[str]]

    @classmethod
    def from_json(cls, path: Path) -> TermMapper:
        """Load and validate a term dictionary from a JSON file.

        Expected format: {"term": ["10 CFR XX.YY", ...], ...}
        Raises ValueError on malformed entries.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"Expected JSON object, got {type(raw).__name__}"
            raise ValueError(msg)
        for key, citations in raw.items():
            if not isinstance(key, str):
                msg = f"Term key must be str, got {type(key).__name__}"
                raise ValueError(msg)
            if not isinstance(citations, list) or len(citations) == 0:
                msg = f"Term '{key}' must map to a non-empty list"
                raise ValueError(msg)
            for c in citations:
                if not isinstance(c, str):
                    msg = f"Citation for '{key}' must be str, got {type(c).__name__}"
                    raise ValueError(msg)
        return cls(_terms=raw)

    def lookup(self, term: str) -> list[str]:
        """Return citations for a term (case-insensitive), or empty list."""
        key = term.lower()
        for k, v in self._terms.items():
            if k.lower() == key:
                return list(v)
        return []

    def scan_content(self, content: str) -> list[TermMatch]:
        """Find all dictionary terms in content.

        Returns matches with frequency >= 2, sorted by descending frequency.
        Matching is case-insensitive with word boundaries.
        """
        content_lower = content.lower()
        matches: list[TermMatch] = []
        for term, citations in self._terms.items():
            pattern = re.compile(r"\b" + re.escape(term.lower()) + r"\b")
            count = len(pattern.findall(content_lower))
            if count >= 2:
                matches.append(
                    TermMatch(term=term, citations=list(citations), frequency=count)
                )
        matches.sort(key=lambda m: (-m.frequency, m.term))
        return matches
