# tests/benchmark/domain/test_models.py
"""Tests for benchmark domain models."""

from __future__ import annotations

import dataclasses

from benchmark.domain.enums import EvidenceTier, UnitKind
from benchmark.domain.models import (
    BenchmarkSourceSpan,
    EvidenceEntry,
    EvidenceSet,
    RegulatoryUnit,
    StageConfig,
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
