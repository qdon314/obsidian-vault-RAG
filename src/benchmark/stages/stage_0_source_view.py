"""Stage 0: Corpus normalization — build benchmark-friendly source spans.

Reads ``ParsedSection`` objects from the eCFR parser and the existing chunk
index, producing ``BenchmarkSourceSpan`` records.  This is a plain builder
function (not a swappable port) because there is one sensible implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmark.domain.models import BenchmarkSourceSpan
from rag.adapters.ingestion.regulatory.ecfr_parser import ParsedParagraph, ParsedSection
from rag.domain.models import Chunk


def _build_citation(section_number: str, paragraph: ParsedParagraph) -> str:
    """Build a citation string like ``10 CFR 50.46(b)(1)``."""
    base = f"10 CFR {section_number}"
    if not paragraph.subsection_tokens:
        return base
    suffix = "".join(f"({t})" for t in paragraph.subsection_tokens)
    return f"{base}{suffix}"


def _build_citation_key(section_number: str, paragraph: ParsedParagraph) -> str:
    """Build a stable citation key like ``10_cfr_50.46_b_1``."""
    parts = ["10_cfr", section_number.replace(".", "_")]
    parts.extend(paragraph.subsection_tokens)
    return "_".join(parts)


def _find_overlapping_chunks(
    chunks: Sequence[Chunk],
    para_start: int,
    para_end: int,
) -> tuple[str, ...]:
    """Return chunk IDs whose char range overlaps [para_start, para_end)."""
    result: list[str] = []
    for chunk in chunks:
        if chunk.start_char is None or chunk.end_char is None:
            continue
        # Overlap: chunk starts before para ends AND chunk ends after para starts
        if chunk.start_char < para_end and chunk.end_char > para_start:
            result.append(chunk.chunk_id)
    return tuple(result)


def build_source_spans(
    *,
    sections: Sequence[ParsedSection],
    doc_id: str,
    chunk_index: Sequence[Chunk],
    corpus_snapshot_id: str,
    effective_date: str,
) -> list[BenchmarkSourceSpan]:
    """Build ``BenchmarkSourceSpan`` records from parsed eCFR sections.

    Each ``(section, paragraph)`` pair produces one span.  Character offsets
    are computed cumulatively within each section.

    Args:
        sections: Parsed eCFR sections from ``parse_ecfr_xml()``.
        doc_id: The document ID for the source document.
        chunk_index: Existing chunks for overlap resolution.
        corpus_snapshot_id: Snapshot hash from ``compute_snapshot_id()``.
        effective_date: ISO date string for the corpus effective date.
    """
    spans: list[BenchmarkSourceSpan] = []
    for section in sections:
        char_offset = 0
        for para in section.paragraphs:
            para_start = char_offset
            para_end = char_offset + len(para.text)
            overlapping = _find_overlapping_chunks(chunk_index, para_start, para_end)

            spans.append(
                BenchmarkSourceSpan(
                    source_doc_id=doc_id,
                    citation=_build_citation(section.section_number, para),
                    citation_key=_build_citation_key(section.section_number, para),
                    section_title=section.title,
                    text=para.text,
                    char_start=para_start,
                    char_end=para_end,
                    chunk_ids_overlapping_span=overlapping,
                    parent_section_id=section.section_number,
                    effective_date=effective_date,
                    corpus_snapshot_id=corpus_snapshot_id,
                )
            )
            char_offset = para_end

    return spans
