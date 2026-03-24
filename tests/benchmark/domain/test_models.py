# tests/benchmark/domain/test_models.py
"""Tests for benchmark domain models."""

from __future__ import annotations

import dataclasses

from benchmark.domain.enums import EvidenceTier, QueryClass, UnitKind
from benchmark.domain.models import (
    BenchmarkSourceSpan,
    EvidenceEntry,
    EvidenceSet,
    QueryCandidate,
    RegulatoryUnit,
    StageConfig,
    ValidatedQuery,
    ValidationResult,
)


class TestBenchmarkSourceSpan:
    def test_frozen(self) -> None:
        span = BenchmarkSourceSpan(
            source_doc_id="doc_1",
            citation="10 CFR 50.46(b)(1)",
            citation_key="10_cfr_50.46_b_1",
            section_title="Acceptance criteria for ECCS",
            text="Peak cladding temperature shall not exceed 2200°F.",
            char_start=0,
            char_end=51,
            chunk_ids_overlapping_span=("chunk_17",),
            parent_section_id="50.46",
            effective_date="2026-01-01",
            corpus_snapshot_id="abc123",
        )
        assert dataclasses.is_dataclass(span)
        with_error = False
        try:
            span.text = "mutated"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            with_error = True
        assert with_error

    def test_defaults(self) -> None:
        span = BenchmarkSourceSpan(
            source_doc_id="doc_1",
            citation="10 CFR 50.46(b)(1)",
            citation_key="10_cfr_50.46_b_1",
            section_title="Acceptance criteria",
            text="Some text.",
            char_start=0,
            char_end=10,
            chunk_ids_overlapping_span=(),
            parent_section_id="50.46",
            effective_date="2026-01-01",
            corpus_snapshot_id="abc123",
        )
        assert span.metadata == {}


class TestRegulatoryUnit:
    def test_frozen(self) -> None:
        span = BenchmarkSourceSpan(
            source_doc_id="doc_1",
            citation="10 CFR 50.46(b)(1)",
            citation_key="10_cfr_50.46_b_1",
            section_title="ECCS criteria",
            text="Temperature limit.",
            char_start=0,
            char_end=18,
            chunk_ids_overlapping_span=("c1",),
            parent_section_id="50.46",
            effective_date="2026-01-01",
            corpus_snapshot_id="snap1",
        )
        unit = RegulatoryUnit(
            unit_id="50.46_b_1_peak_cladding_temp",
            kind=UnitKind.THRESHOLD,
            spans=(span,),
            citation="10 CFR 50.46(b)(1)",
            subsection_chain=("b", "1"),
            parent_section_id="50.46",
            corpus_snapshot_id="snap1",
        )
        assert dataclasses.is_dataclass(unit)
        assert unit.unit_id == "50.46_b_1_peak_cladding_temp"

    def test_defaults(self) -> None:
        span = BenchmarkSourceSpan(
            source_doc_id="doc_1",
            citation="10 CFR 50.46(b)(1)",
            citation_key="key",
            section_title="Title",
            text="Text.",
            char_start=0,
            char_end=5,
            chunk_ids_overlapping_span=(),
            parent_section_id="50.46",
            effective_date="2026-01-01",
            corpus_snapshot_id="snap1",
        )
        unit = RegulatoryUnit(
            unit_id="50.46_b_1",
            kind=UnitKind.OBLIGATION,
            spans=(span,),
            citation="10 CFR 50.46(b)(1)",
            subsection_chain=("b", "1"),
            parent_section_id="50.46",
            corpus_snapshot_id="snap1",
        )
        assert unit.cross_references == ()
        assert unit.canonical_statement is None
        assert unit.entities == ()
        assert unit.conditions == ()
        assert unit.metadata == {}


class TestEvidenceEntry:
    def test_frozen(self) -> None:
        entry = EvidenceEntry(
            span_id="50.46_b_1_0",
            citation="10 CFR 50.46(b)(1)",
            text="Peak cladding temperature shall not exceed 2200°F.",
            char_start=0,
            char_end=51,
            chunk_ids=("chunk_17", "chunk_18"),
            tier=EvidenceTier.CRITICAL,
        )
        assert dataclasses.is_dataclass(entry)
        with_error = False
        try:
            entry.tier = EvidenceTier.SUPPORTING  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            with_error = True
        assert with_error

    def test_fields(self) -> None:
        entry = EvidenceEntry(
            span_id="50.46_b_1_0",
            citation="10 CFR 50.46(b)(1)",
            text="Some text.",
            char_start=100,
            char_end=110,
            chunk_ids=("c1",),
            tier=EvidenceTier.SUPPORTING,
        )
        assert entry.span_id == "50.46_b_1_0"
        assert entry.tier == EvidenceTier.SUPPORTING
        assert entry.chunk_ids == ("c1",)


class TestEvidenceSet:
    def test_frozen(self) -> None:
        es = EvidenceSet(unit_id="50.46_b_1")
        assert dataclasses.is_dataclass(es)
        with_error = False
        try:
            es.unit_id = "changed"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            with_error = True
        assert with_error

    def test_defaults_empty_tuples(self) -> None:
        es = EvidenceSet(unit_id="50.46_b_1")
        assert es.critical == ()
        assert es.supporting == ()
        assert es.contextual == ()

    def test_with_entries(self) -> None:
        critical = EvidenceEntry(
            span_id="u1_0",
            citation="10 CFR 50.46(b)(1)",
            text="Temperature limit.",
            char_start=0,
            char_end=18,
            chunk_ids=("c1",),
            tier=EvidenceTier.CRITICAL,
        )
        contextual = EvidenceEntry(
            span_id="u1_1",
            citation="10 CFR 50.46(b)(2)",
            text="Related provision.",
            char_start=18,
            char_end=36,
            chunk_ids=("c2",),
            tier=EvidenceTier.CONTEXTUAL,
        )
        es = EvidenceSet(
            unit_id="50.46_b_1",
            critical=(critical,),
            contextual=(contextual,),
        )
        assert len(es.critical) == 1
        assert len(es.supporting) == 0
        assert len(es.contextual) == 1
        assert es.critical[0].tier == EvidenceTier.CRITICAL


class TestStageConfig:
    def test_defaults(self) -> None:
        cfg = StageConfig(model="gpt-4o")
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 4096
        assert cfg.max_retries == 3
        assert cfg.timeout_s == 60.0


class TestQueryCandidate:
    def test_frozen(self) -> None:
        qc = QueryCandidate(
            candidate_id="qc_50.46_b_1_citation_lookup_0",
            unit_id="50.46_b_1",
            query="What does 10 CFR 50.46(b)(1) require?",
            query_class=QueryClass.CITATION_LOOKUP,
            source_citations=("10 CFR 50.46(b)(1)",),
            evidence_span_ids=("50.46_b_1_0",),
        )
        assert dataclasses.is_dataclass(qc)
        with_error = False
        try:
            qc.query = "mutated"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            with_error = True
        assert with_error

    def test_defaults(self) -> None:
        qc = QueryCandidate(
            candidate_id="qc_1",
            unit_id="u1",
            query="Test query?",
            query_class=QueryClass.CITATION_LOOKUP,
            source_citations=("10 CFR 50.46",),
            evidence_span_ids=("u1_0",),
        )
        assert qc.difficulty == "easy"
        assert qc.corpus_snapshot_id == ""
        assert qc.metadata == {}

    def test_all_query_classes_accepted(self) -> None:
        for qclass in QueryClass:
            qc = QueryCandidate(
                candidate_id=f"qc_{qclass.value}",
                unit_id="u1",
                query="Test?",
                query_class=qclass,
                source_citations=(),
                evidence_span_ids=(),
            )
            assert qc.query_class == qclass


class TestValidationResult:
    def test_frozen(self) -> None:
        vr = ValidationResult(
            candidate_id="qc_1",
            is_valid=True,
            flags=(),
        )
        assert dataclasses.is_dataclass(vr)
        with_error = False
        try:
            vr.is_valid = False  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            with_error = True
        assert with_error

    def test_defaults(self) -> None:
        vr = ValidationResult(
            candidate_id="qc_1",
            is_valid=False,
            flags=("too_short",),
        )
        assert vr.scores == {}

    def test_with_flags_and_scores(self) -> None:
        vr = ValidationResult(
            candidate_id="qc_1",
            is_valid=False,
            flags=("no_citation", "too_short"),
            scores={"length_score": 0.3},
        )
        assert len(vr.flags) == 2
        assert vr.scores["length_score"] == 0.3


class TestValidatedQuery:
    def test_frozen(self) -> None:
        vq = ValidatedQuery(
            candidate_id="qc_1",
            unit_id="u1",
            query="What does 10 CFR 50.46(b)(1) require?",
            query_class=QueryClass.CITATION_LOOKUP,
            source_citations=("10 CFR 50.46(b)(1)",),
            evidence_span_ids=("u1_0",),
            difficulty="easy",
            corpus_snapshot_id="snap1",
        )
        assert dataclasses.is_dataclass(vq)
        with_error = False
        try:
            vq.query = "mutated"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            with_error = True
        assert with_error

    def test_defaults(self) -> None:
        vq = ValidatedQuery(
            candidate_id="qc_1",
            unit_id="u1",
            query="Test?",
            query_class=QueryClass.CITATION_LOOKUP,
            source_citations=(),
            evidence_span_ids=(),
            difficulty="easy",
            corpus_snapshot_id="snap1",
        )
        assert vq.validation_scores == {}
        assert vq.metadata == {}

    def test_shares_fields_with_query_candidate(self) -> None:
        """Common fields between QueryCandidate and ValidatedQuery
        should have the same names for easy conversion."""
        qc_fields = {f.name for f in dataclasses.fields(QueryCandidate)}
        vq_fields = {f.name for f in dataclasses.fields(ValidatedQuery)}
        common = {
            "candidate_id",
            "unit_id",
            "query",
            "query_class",
            "source_citations",
            "evidence_span_ids",
            "difficulty",
            "corpus_snapshot_id",
            "metadata",
        }
        assert common <= qc_fields
        assert common <= vq_fields
