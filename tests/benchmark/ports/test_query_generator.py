"""Tests for the QueryGenerator port protocol."""

from __future__ import annotations

from benchmark.domain.enums import QueryClass, UnitKind
from benchmark.domain.models import (
    BenchmarkSourceSpan,
    EvidenceSet,
    QueryCandidate,
    RegulatoryUnit,
)
from benchmark.ports.query_generator import QueryGenerator


def _make_span() -> BenchmarkSourceSpan:
    return BenchmarkSourceSpan(
        source_doc_id="doc_1",
        citation="10 CFR 50.46(b)(1)",
        citation_key="10_cfr_50.46_b_1",
        section_title="ECCS criteria",
        text="Peak cladding temperature.",
        char_start=0,
        char_end=26,
        chunk_ids_overlapping_span=("c1",),
        parent_section_id="50.46",
        effective_date="2026-01-01",
        corpus_snapshot_id="snap1",
    )


def _make_unit() -> RegulatoryUnit:
    return RegulatoryUnit(
        unit_id="50.46_b_1",
        kind=UnitKind.OBLIGATION,
        spans=(_make_span(),),
        citation="10 CFR 50.46(b)(1)",
        subsection_chain=("b", "1"),
        parent_section_id="50.46",
        corpus_snapshot_id="snap1",
    )


class _MockQueryGenerator:
    """Mock satisfying the QueryGenerator protocol."""

    def generate(
        self,
        unit: RegulatoryUnit,
        evidence: EvidenceSet,
        query_class: QueryClass,
    ) -> list[QueryCandidate]:
        return [
            QueryCandidate(
                candidate_id=f"qc_{unit.unit_id}_{query_class.value}_0",
                unit_id=unit.unit_id,
                query="What does 10 CFR 50.46(b)(1) require?",
                query_class=query_class,
                source_citations=(unit.citation,),
                evidence_span_ids=tuple(e.span_id for e in evidence.critical),
            )
        ]


class TestQueryGeneratorProtocol:
    def test_isinstance_check(self) -> None:
        gen = _MockQueryGenerator()
        assert isinstance(gen, QueryGenerator)

    def test_generate_returns_candidates(self) -> None:
        gen = _MockQueryGenerator()
        unit = _make_unit()
        evidence = EvidenceSet(unit_id=unit.unit_id)
        result = gen.generate(unit, evidence, QueryClass.CITATION_LOOKUP)
        assert len(result) == 1
        assert result[0].query_class == QueryClass.CITATION_LOOKUP
        assert result[0].unit_id == unit.unit_id
