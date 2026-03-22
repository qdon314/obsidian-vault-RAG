# tests/benchmark/adapters/extraction/test_rules_extractor.py
"""Tests for Stage 1a deterministic rules extractor."""

from __future__ import annotations

from benchmark.adapters.extraction.rules_extractor import RulesExtractor
from benchmark.domain.enums import UnitKind
from benchmark.domain.models import BenchmarkSourceSpan


def _span(
    section_id: str = "50.46",
    subsection: tuple[str, ...] = ("b", "1"),
    text: str = "Peak cladding temperature.",
    cross_refs: tuple[str, ...] = (),
    citation: str | None = None,
) -> BenchmarkSourceSpan:
    suffix = "".join(f"({t})" for t in subsection) if subsection else ""
    cit = citation or f"10 CFR {section_id}{suffix}"
    key = "_".join(["10_cfr", section_id.replace(".", "_"), *subsection])
    return BenchmarkSourceSpan(
        source_doc_id="doc_1",
        citation=cit,
        citation_key=key,
        section_title="Title",
        text=text,
        char_start=0,
        char_end=len(text),
        chunk_ids_overlapping_span=(),
        parent_section_id=section_id,
        effective_date="2026-01-01",
        corpus_snapshot_id="snap1",
        metadata={"subsection_tokens": subsection, "cross_references": cross_refs},
    )


class TestRulesExtractor:
    def test_single_span_produces_unit(self) -> None:
        extractor = RulesExtractor()
        spans = [_span()]
        units = extractor.extract(spans)
        assert len(units) == 1
        unit = units[0]
        assert unit.parent_section_id == "50.46"
        assert unit.subsection_chain == ("b", "1")
        assert unit.corpus_snapshot_id == "snap1"

    def test_unit_id_from_subsection_chain(self) -> None:
        extractor = RulesExtractor()
        spans = [_span(section_id="50.46", subsection=("b", "1"))]
        units = extractor.extract(spans)
        assert units[0].unit_id == "50.46_b_1"

    def test_unit_id_no_subsection(self) -> None:
        extractor = RulesExtractor()
        spans = [_span(section_id="50.46", subsection=())]
        units = extractor.extract(spans)
        assert units[0].unit_id == "50.46"

    def test_cross_reference_detection(self) -> None:
        extractor = RulesExtractor()
        spans = [_span(cross_refs=("10 CFR §50.55a",))]
        units = extractor.extract(spans)
        assert units[0].cross_references == ("10 CFR §50.55a",)
        assert units[0].kind == UnitKind.CROSS_REFERENCE

    def test_default_kind_is_obligation(self) -> None:
        extractor = RulesExtractor()
        spans = [_span(cross_refs=())]
        units = extractor.extract(spans)
        assert units[0].kind == UnitKind.OBLIGATION

    def test_multiple_spans_same_section_grouped(self) -> None:
        """Spans with same section_id + subsection_chain are grouped into one unit."""
        extractor = RulesExtractor()
        spans = [
            _span(section_id="50.46", subsection=("b", "1"), text="First."),
            _span(section_id="50.46", subsection=("b", "1"), text="Second."),
        ]
        units = extractor.extract(spans)
        assert len(units) == 1
        assert len(units[0].spans) == 2

    def test_different_subsections_different_units(self) -> None:
        extractor = RulesExtractor()
        spans = [
            _span(section_id="50.46", subsection=("b", "1")),
            _span(section_id="50.46", subsection=("b", "2")),
        ]
        units = extractor.extract(spans)
        assert len(units) == 2
        assert {u.unit_id for u in units} == {"50.46_b_1", "50.46_b_2"}

    def test_satisfies_unit_extractor_protocol(self) -> None:
        """RulesExtractor structurally satisfies the UnitExtractor protocol."""
        from benchmark.ports.unit_extractor import UnitExtractor

        extractor: UnitExtractor = RulesExtractor()
        assert hasattr(extractor, "extract")

    def test_empty_input(self) -> None:
        extractor = RulesExtractor()
        assert extractor.extract([]) == []
