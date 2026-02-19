"""Generate evaluation queries from NRC case markdown files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.adapters.ingestion.loaders.obsidian_markdown_loader import (
    split_obsidian_frontmatter,
)
from rag.adapters.query_generation.term_mapper import TermMapper

_STRATEGY1_TEMPLATES = [
    "What are the requirements of {citation}?",
    "What does {citation} require?",
    "Summarize the key provisions of {citation}.",
]

_STRATEGY2_TEMPLATES = [
    "What are the regulatory requirements for {term}?",
    "What regulations govern {term} at nuclear power plants?",
    "What does the NRC require regarding {term}?",
]

_METADATA_FILTER: dict[str, Any] = {
    "filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}
}


def _normalize_citation(citation: str) -> str:
    """Normalize a citation string for case-insensitive comparison.

    Strips the section symbol and lowercases.
    """
    return citation.replace("§", "").replace("  ", " ").strip().lower()


@dataclass(frozen=True, slots=True)
class CaseQueryGenerator:
    """Generates evaluation queries from NRC case markdown files.

    Uses two strategies:
    1. Direct citation queries from frontmatter cross_references
    2. Term mapping queries from TermMapper scan of body content
    """

    term_mapper: TermMapper
    max_queries_per_case: int = 50
    _dc_counter: list[int] = field(
        init=False, repr=False, compare=False, hash=False, default_factory=lambda: [0]
    )
    _tm_counter: list[int] = field(
        init=False, repr=False, compare=False, hash=False, default_factory=lambda: [0]
    )

    def generate(self, case_file: Path) -> list[dict[str, Any]]:
        """Generate queries from a single case markdown file.

        Returns a list of query dicts ready for JSONL serialization.
        """
        text = case_file.read_text(encoding="utf-8")
        frontmatter, body = split_obsidian_frontmatter(text)

        accession = frontmatter.get("accession_number", "")
        doc_type = frontmatter.get("document_type", "")
        cross_refs: list[str] = frontmatter.get("cross_references", [])

        queries: list[dict[str, Any]] = []

        # Strategy 1: Direct citation queries
        seen_citations: set[str] = set()
        for i, citation in enumerate(cross_refs):
            norm = _normalize_citation(citation)
            if norm in seen_citations:
                continue
            seen_citations.add(norm)

            self._dc_counter[0] += 1
            qid = f"case-dc-{self._dc_counter[0]:03d}"
            template = _STRATEGY1_TEMPLATES[i % len(_STRATEGY1_TEMPLATES)]

            queries.append(
                {
                    "qid": qid,
                    "query": template.format(citation=citation),
                    "difficulty": "easy",
                    "query_type": "factual",
                    "requires_synthesis": False,
                    "is_unanswerable": False,
                    "expected_answer": None,
                    "unanswerable_reason": None,
                    "relevant_citations": [citation],
                    "tags": ["case-derived", "citation-direct"],
                    "source_case": accession,
                    "case_document_type": doc_type,
                    "metadata": dict(_METADATA_FILTER),
                }
            )

        # Strategy 2: Term mapping queries
        cross_ref_normalized = {_normalize_citation(c) for c in cross_refs}
        matches = self.term_mapper.scan_content(body)
        top_matches = matches[:5]

        for i, match in enumerate(top_matches):
            self._tm_counter[0] += 1
            qid = f"case-tm-{self._tm_counter[0]:03d}"
            template = _STRATEGY2_TEMPLATES[i % len(_STRATEGY2_TEMPLATES)]

            # Get citation labels from anchors
            term_map = self.term_mapper.term_map
            anchor_refs = [a.ref for a in match.anchors]
            citation_labels = [
                term_map.refs[ref].label for ref in anchor_refs if ref in term_map.refs
            ]

            # Check overlap: do any of this term's mapped citations appear
            # in the frontmatter cross_references?
            has_overlap = any(
                _normalize_citation(c) in cross_ref_normalized for c in citation_labels
            )

            # Build tags with term type
            tags = ["case-derived", "term-mapping", f"{match.term_type.value}-term"]
            if has_overlap:
                tags.append("overlaps-citation")

            entry: dict[str, Any] = {
                "qid": qid,
                "query": template.format(term=match.term),
                "difficulty": "medium",
                "query_type": "interpretive",
                "requires_synthesis": True,
                "is_unanswerable": False,
                "expected_answer": None,
                "unanswerable_reason": None,
                "relevant_citations": citation_labels,
                "anchor_refs": anchor_refs,
                "tags": tags,
                "source_case": accession,
                "technical_term": match.term,
                "term_type": match.term_type.value,
                "metadata": dict(_METADATA_FILTER),
            }
            if has_overlap:
                entry["overlaps_direct_citation"] = True

            queries.append(entry)

        return queries[: self.max_queries_per_case]
