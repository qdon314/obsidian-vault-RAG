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

_SCNARIO_TEMPLATES: dict[str, list[str]] = {
    "inspection": [
        "A {reactor_type} plant identified issues with {term} during routine inspection. What regulations apply to {term}?",
        "An inspector found deficiencies related to {term} at a nuclear facility. What are the applicable regulatory requirements?",
        "During a plant walkdown, problems were identified with {term}. What regulatory standards govern this area?",
    ],
    "enforcement": [
        "What regulatory requirements could be the basis for an enforcement action involving {term} at a nuclear facility?",
        "A nuclear plant received a notice of violation related to {term}. What are the underlying regulatory requirements?",
        "An enforcement case was opened regarding {term}. What regulations are most relevant?",
    ],
    "vendor_part21": [
        "A vendor discovers a defect affecting {term}. What are the reporting and notification requirements?",
        "A nuclear component supplier identified an issue with {term}. What regulatory obligations apply?",
        "A Part 21 report was filed regarding {term}. What are the applicable regulatory provisions?",
    ],
    "operations": [
        "What are the regulatory requirements when {term} is found inoperable during plant operations?",
        "An operator discovers that {term} is not functioning as designed. What regulations govern this situation?",
        "During normal operations, a degraded condition is identified affecting {term}. What requirements apply?",
    ],
    "licensing": [
        "What regulatory provisions govern {term} in the context of a nuclear facility license application?",
        "A license amendment is being sought related to {term}. What are the key regulatory requirements?",
        "What does the regulatory framework require regarding {term} for nuclear plant licensing?",
    ],
}

_GENERIC_SCENARIO_TEMPLATES = [
    "What are the regulatory requirements related to {term} at a nuclear power plant?",
    "A nuclear facility identified an issue involving {term}. What regulations apply?",
    "What does the NRC regulatory framework require regarding {term}?",
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

    Uses three strategies:
    1. Direct citation queries from frontmatter cross_references
    2. Term mapping queries from TermMapper scan of body content
    3. Scenario-based queries from term matches + case category
    """

    term_mapper: TermMapper
    max_queries_per_case: int = 50
    _dc_counter: list[int] = field(
        init=False, repr=False, compare=False, hash=False, default_factory=lambda: [0]
    )
    _tm_counter: list[int] = field(
        init=False, repr=False, compare=False, hash=False, default_factory=lambda: [0]
    )
    _sc_counter: list[int] = field(
        init=False, repr=False, compare=False, hash=False, default_factory=lambda: [0]
    )

    def generate(self, case_file: Path) -> list[dict[str, Any]]:
        """Generate queries from a single case markdown file.

        Returns a list of query dicts ready for JSONL serialization.
        """
        text = case_file.read_text(encoding="utf-8")
        frontmatter, body = split_obsidian_frontmatter(text)

        queries: list[dict[str, Any]] = []
        queries.extend(self._citation_direct(frontmatter))
        queries.extend(self._term_mapping(body, frontmatter))
        queries.extend(self._scenario(body, frontmatter))
        return queries[: self.max_queries_per_case]

    def _citation_direct(self, frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
        """Strategy 1: Generate queries from frontmatter cross_references."""
        accession = frontmatter.get("accession_number", "")
        doc_type = frontmatter.get("document_type", "")
        cross_refs: list[str] = frontmatter.get("cross_references", [])

        queries: list[dict[str, Any]] = []
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
        return queries

    def _term_mapping(self, body: str, frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
        """Strategy 2: Generate queries from TermMapper scan of body content."""
        accession = frontmatter.get("accession_number", "")
        cross_refs: list[str] = frontmatter.get("cross_references", [])
        cross_ref_normalized = {_normalize_citation(c) for c in cross_refs}
        matches = self.term_mapper.scan_content(body)
        top_matches = matches[:5]

        queries: list[dict[str, Any]] = []
        for i, match in enumerate(top_matches):
            self._tm_counter[0] += 1
            qid = f"case-tm-{self._tm_counter[0]:03d}"
            template = _STRATEGY2_TEMPLATES[i % len(_STRATEGY2_TEMPLATES)]

            term_map = self.term_mapper.term_map
            anchor_refs = [a.ref for a in match.anchors]
            citation_labels = [
                term_map.refs[ref].label for ref in anchor_refs if ref in term_map.refs
            ]

            has_overlap = any(
                _normalize_citation(c) in cross_ref_normalized for c in citation_labels
            )

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
        return queries

    def _scenario(self, body: str, frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
        """Strategy 3: Generate scenario-based queries from term matches + category."""
        accession = frontmatter.get("accession_number", "")
        category = frontmatter.get("case_category", "unknown")
        reactor_type = frontmatter.get("reactor_type", "nuclear")
        cross_refs: list[str] = frontmatter.get("cross_references", [])

        templates = _SCNARIO_TEMPLATES.get(category, _GENERIC_SCENARIO_TEMPLATES)
        matches = self.term_mapper.scan_content(body)
        top_matches = matches[:5]

        queries: list[dict[str, Any]] = []
        for i, match in enumerate(top_matches):
            term_map = self.term_mapper.term_map
            anchor_refs = [a.ref for a in match.anchors]
            citation_labels = [
                term_map.refs[ref].label for ref in anchor_refs if ref in term_map.refs
            ]

            # Also include any cross_references from frontmatter as relevant citations
            all_citations = list(citation_labels)
            for cr in cross_refs:
                if cr not in all_citations:
                    all_citations.append(cr)

            # Skip terms with no resolvable citations
            if not citation_labels:
                continue

            self._sc_counter[0] += 1
            qid = f"case-sc-{self._sc_counter[0]:03d}"
            template = templates[i % len(templates)]

            queries.append(
                {
                    "qid": qid,
                    "query": template.format(term=match.term, reactor_type=reactor_type),
                    "difficulty": "hard",
                    "query_type": "scenario",
                    "requires_synthesis": True,
                    "is_unanswerable": False,
                    "expected_answer": None,
                    "unanswerable_reason": None,
                    "relevant_citations": citation_labels,
                    "anchor_refs": anchor_refs,
                    "tags": [
                        "case-derived",
                        "scenario-based",
                        f"{match.term_type.value}-term",
                    ],
                    "source_case": accession,
                    "technical_term": match.term,
                    "term_type": match.term_type.value,
                    "metadata": dict(_METADATA_FILTER),
                }
            )
        return queries
