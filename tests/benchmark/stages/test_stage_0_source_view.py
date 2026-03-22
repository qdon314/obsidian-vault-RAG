# tests/benchmark/stages/test_stage_0_source_view.py
"""Tests for Stage 0 corpus normalization."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmark.stages.stage_0_source_view import build_source_spans

from rag.adapters.ingestion.regulatory.ecfr_parser import (
    CrossRef,
    ParsedParagraph,
    ParsedSection,
)
from rag.domain.models import Chunk


def _make_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    start_char: int,
    end_char: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        chunk_index=0,
        start_char=start_char,
        end_char=end_char,
    )


class TestBuildSourceSpans:
    def test_basic_span_creation(self) -> None:
        """A single section with one paragraph produces one span."""
        section = ParsedSection(
            section_number="50.46",
            title="Acceptance criteria for ECCS",
            part_number="50",
            paragraphs=(
                ParsedParagraph(
                    text="(b)(1) Peak cladding temperature shall not exceed 2200°F.",
                    level=2,
                    prefix="1",
                    subsection_tokens=("b", "1"),
                ),
            ),
        )
        chunks = [
            _make_chunk("c1", "doc_50", "Peak cladding temperature", 0, 100),
        ]
        spans = build_source_spans(
            sections=[section],
            doc_id="doc_50",
            chunk_index=chunks,
            corpus_snapshot_id="snap1",
            effective_date="2026-01-01",
        )
        assert len(spans) == 1
        span = spans[0]
        assert span.source_doc_id == "doc_50"
        assert span.parent_section_id == "50.46"
        assert span.section_title == "Acceptance criteria for ECCS"
        assert span.corpus_snapshot_id == "snap1"
        assert span.effective_date == "2026-01-01"
        assert "50.46" in span.citation
        assert span.citation_key  # non-empty

    def test_citation_includes_subsection(self) -> None:
        """Citation includes the subsection chain."""
        section = ParsedSection(
            section_number="50.46",
            title="ECCS criteria",
            part_number="50",
            paragraphs=(
                ParsedParagraph(
                    text="(b)(1) Limit text.",
                    level=2,
                    prefix="1",
                    subsection_tokens=("b", "1"),
                ),
            ),
        )
        spans = build_source_spans(
            sections=[section],
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert "(b)(1)" in spans[0].citation

    def test_paragraph_without_subsection(self) -> None:
        """A paragraph with no subsection chain still produces a span."""
        section = ParsedSection(
            section_number="50.46",
            title="ECCS criteria",
            part_number="50",
            paragraphs=(
                ParsedParagraph(text="General intro text.", level=0, prefix=None),
            ),
        )
        spans = build_source_spans(
            sections=[section],
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert len(spans) == 1
        assert spans[0].citation == "10 CFR 50.46"

    def test_chunk_overlap_resolution(self) -> None:
        """Spans include chunk IDs whose char ranges overlap the paragraph."""
        section = ParsedSection(
            section_number="50.46",
            title="ECCS",
            part_number="50",
            paragraphs=(
                ParsedParagraph(
                    text="(a) First paragraph text.",
                    level=1,
                    prefix="a",
                    subsection_tokens=("a",),
                ),
            ),
        )
        # Paragraph is the first one, char_start=0.  Chunk c1 overlaps, c2 doesn't.
        chunks = [
            _make_chunk("c1", "d1", "First paragraph", 0, 50),
            _make_chunk("c2", "d1", "Later text", 200, 300),
        ]
        spans = build_source_spans(
            sections=[section],
            doc_id="d1",
            chunk_index=chunks,
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert "c1" in spans[0].chunk_ids_overlapping_span
        assert "c2" not in spans[0].chunk_ids_overlapping_span

    def test_multiple_paragraphs_multiple_spans(self) -> None:
        """Each paragraph becomes its own span."""
        section = ParsedSection(
            section_number="50.46",
            title="ECCS",
            part_number="50",
            paragraphs=(
                ParsedParagraph(text="(a) Para A.", level=1, prefix="a", subsection_tokens=("a",)),
                ParsedParagraph(text="(b) Para B.", level=1, prefix="b", subsection_tokens=("b",)),
            ),
        )
        spans = build_source_spans(
            sections=[section],
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert len(spans) == 2
        assert spans[0].citation != spans[1].citation

    def test_multiple_sections(self) -> None:
        """Spans from multiple sections are all returned."""
        sections = [
            ParsedSection(
                section_number="50.46",
                title="ECCS",
                part_number="50",
                paragraphs=(
                    ParsedParagraph(text="(a) A.", level=1, prefix="a", subsection_tokens=("a",)),
                ),
            ),
            ParsedSection(
                section_number="50.47",
                title="Emergency plans",
                part_number="50",
                paragraphs=(
                    ParsedParagraph(text="(a) B.", level=1, prefix="a", subsection_tokens=("a",)),
                ),
            ),
        ]
        spans = build_source_spans(
            sections=sections,
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert len(spans) == 2
        assert spans[0].parent_section_id == "50.46"
        assert spans[1].parent_section_id == "50.47"

    def test_empty_sections(self) -> None:
        """An empty section list produces no spans."""
        spans = build_source_spans(
            sections=[],
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert spans == []
