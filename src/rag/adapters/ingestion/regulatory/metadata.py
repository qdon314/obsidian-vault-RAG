"""Chunk-level metadata enrichment for regulatory documents.

After the structural chunker splits a normalized markdown file into chunks,
each chunk carries generic metadata.  This module stamps regulatory-specific
fields onto ``Chunk.metadata``:

* ``citation_key`` -- the section-level citation (e.g. ``10 CFR §50.36``).
* ``citation`` -- a chunk-specific citation that appends subsection markers
  derived from the chunk's heading path (e.g. ``10 CFR §50.36(c)(2)(ii)``).
* ``cross_references`` -- list of other CFR sections referenced by this chunk.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# Matches parenthesized subsection markers: ``(a)``, ``(1)``, ``(iv)``, etc.
_SUBSECTION_TOKEN_RE = re.compile(r"\(([A-Za-z0-9ivxlcdmIVXLCDM]+)\)")


def _normalize_cross_references(value: Any) -> list[str]:
    """Coerce *value* (string, list, or other) into a deduplicated list of citation strings."""
    if isinstance(value, str):
        refs = [item.strip() for item in value.split(",") if item.strip()]
        return list(dict.fromkeys(refs))

    if isinstance(value, list):
        refs = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return list(dict.fromkeys(refs))

    return []


def _extract_subsection_tokens(section_path: str | None, section_heading: str | None) -> list[str]:
    """Extract ordered subsection markers from the chunk's heading hierarchy.

    Given a *section_path* like ``"(c) > (2) > (ii)"``, returns ``["c", "2", "ii"]``.
    Falls back to *section_heading* if no path is available.
    """
    segments: list[str] = []
    if section_path:
        segments = [segment.strip() for segment in section_path.split(" > ") if segment.strip()]
    elif section_heading:
        segments = [section_heading]

    tokens: list[str] = []
    for segment in segments:
        tokens.extend(_SUBSECTION_TOKEN_RE.findall(segment))

    return tokens


def _build_specific_citation(
    citation_key: str, section_path: str | None, section_heading: str | None
) -> str:
    """Append subsection markers to *citation_key* to form a chunk-specific citation.

    If *citation_key* already contains some markers (e.g. from the frontmatter),
    only the *additional* markers from the heading path are appended to avoid
    duplication.
    """
    heading_tokens = _extract_subsection_tokens(section_path, section_heading)
    if not heading_tokens:
        return citation_key

    base_tokens = _SUBSECTION_TOKEN_RE.findall(citation_key)
    if base_tokens and heading_tokens[: len(base_tokens)] == base_tokens:
        heading_tokens = heading_tokens[len(base_tokens) :]

    if not heading_tokens:
        return citation_key

    suffix = "".join(f"({token})" for token in heading_tokens)
    return f"{citation_key}{suffix}"


def enrich_regulatory_chunk_metadata(
    metadata: Mapping[str, Any],
    *,
    section_heading: str | None,
    section_path: str | None,
) -> dict[str, Any]:
    """Add regulatory citation metadata when a chunk belongs to the regulatory corpus.

    Returns *metadata* unchanged if ``corpus != "regulatory"`` or if no
    ``citation_key`` is present.  Otherwise enriches the dict with
    ``citation``, ``citation_key``, and ``cross_references``.
    """
    out = dict(metadata)

    if str(out.get("corpus", "")).lower() != "regulatory":
        return out

    citation_key_value = out.get("citation_key")
    if not isinstance(citation_key_value, str) or not citation_key_value.strip():
        frontmatter = out.get("frontmatter")
        if isinstance(frontmatter, dict):
            fm_key = frontmatter.get("citation_key")
            if isinstance(fm_key, str) and fm_key.strip():
                citation_key_value = fm_key

    if not isinstance(citation_key_value, str) or not citation_key_value.strip():
        return out

    citation_key = citation_key_value.strip()
    out["citation_key"] = citation_key
    out["citation"] = _build_specific_citation(citation_key, section_path, section_heading)

    cross_references = _normalize_cross_references(out.get("cross_references"))
    if not cross_references:
        frontmatter = out.get("frontmatter")
        if isinstance(frontmatter, dict):
            cross_references = _normalize_cross_references(frontmatter.get("cross_references"))

    if cross_references:
        out["cross_references"] = cross_references

    return out
