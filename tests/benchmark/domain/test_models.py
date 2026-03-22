# tests/benchmark/domain/test_models.py
"""Tests for benchmark domain models."""

from __future__ import annotations

import dataclasses

from benchmark.domain.enums import UnitKind
from benchmark.domain.models import BenchmarkSourceSpan, RegulatoryUnit, StageConfig


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


class TestStageConfig:
    def test_defaults(self) -> None:
        cfg = StageConfig(model="gpt-4o")
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 4096
        assert cfg.max_retries == 3
        assert cfg.timeout_s == 60.0
