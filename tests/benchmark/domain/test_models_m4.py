"""Tests for M4 domain model additions.

Covers: HardNegativeResult, BenchmarkRecord, BenchmarkDataset.
"""

from __future__ import annotations

import dataclasses
import json

from benchmark.domain.enums import EvidenceTier, QueryClass
from benchmark.domain.models import (
    BenchmarkDataset,
    BenchmarkRecord,
    EvidenceEntry,
    HardNegativeResult,
)

# -- Helpers ------------------------------------------------------------------


def _make_evidence_entry(
    span_id: str = "span_0",
    tier: EvidenceTier = EvidenceTier.CRITICAL,
) -> EvidenceEntry:
    return EvidenceEntry(
        span_id=span_id,
        citation="10 CFR 50.46(b)(1)",
        text="Peak cladding temperature shall not exceed 2200F.",
        char_start=0,
        char_end=50,
        chunk_ids=("chunk_1", "chunk_2"),
        tier=tier,
    )


def _make_hard_negative_result(
    candidate_id: str = "qc_test_0",
) -> HardNegativeResult:
    return HardNegativeResult(
        candidate_id=candidate_id,
        hard_negative_chunk_ids=("chunk_99", "chunk_100"),
        retriever_config={
            "model": "text-embedding-3-small",
            "index_version": "ecfr_2026-01-01",
            "top_k": 20,
        },
    )


def _make_benchmark_record(
    qid: str = "qc_test_0",
) -> BenchmarkRecord:
    return BenchmarkRecord(
        qid=qid,
        query="What is the peak cladding temperature under 10 CFR 50.46(b)(1)?",
        query_class=QueryClass.CITATION_LOOKUP,
        difficulty="easy",
        source_unit_ids=("50.46_b_1_peak_cladding_temp",),
        source_citations=("10 CFR 50.46(b)(1)",),
        critical_evidence=(_make_evidence_entry(),),
        supporting_evidence=(),
        contextual_evidence=(),
        hard_negative_chunk_ids=("chunk_99", "chunk_100"),
        hard_negatives_retriever_config={
            "model": "text-embedding-3-small",
            "index_version": "ecfr_2026-01-01",
            "top_k": 20,
        },
        corpus_snapshot_id="abc123",
        valid_as_of="2026-03-24",
    )


# -- HardNegativeResult tests ------------------------------------------------


class TestHardNegativeResult:
    def test_frozen(self) -> None:
        hnr = _make_hard_negative_result()
        assert dataclasses.is_dataclass(hnr)
        assert getattr(hnr, "__dataclass_fields__", None) is not None

    def test_insufficient_defaults_false(self) -> None:
        hnr = _make_hard_negative_result()
        assert hnr.insufficient is False

    def test_insufficient_explicit(self) -> None:
        hnr = HardNegativeResult(
            candidate_id="qc_test_1",
            hard_negative_chunk_ids=("chunk_50",),
            retriever_config={"model": "test", "top_k": 10},
            insufficient=True,
        )
        assert hnr.insufficient is True
        assert len(hnr.hard_negative_chunk_ids) == 1

    def test_json_round_trip(self) -> None:
        hnr = _make_hard_negative_result()
        d = dataclasses.asdict(hnr)
        serialized = json.dumps(d, default=str)
        restored = json.loads(serialized)
        assert restored["candidate_id"] == hnr.candidate_id
        assert restored["hard_negative_chunk_ids"] == list(hnr.hard_negative_chunk_ids)
        assert restored["retriever_config"] == hnr.retriever_config
        assert restored["insufficient"] is False


# -- BenchmarkRecord tests ----------------------------------------------------


class TestBenchmarkRecord:
    def test_frozen(self) -> None:
        rec = _make_benchmark_record()
        assert dataclasses.is_dataclass(rec)

    def test_qid_preserves_candidate_id(self) -> None:
        rec = _make_benchmark_record(qid="qc_custom_id")
        assert rec.qid == "qc_custom_id"

    def test_optional_defaults(self) -> None:
        rec = _make_benchmark_record()
        assert rec.is_unanswerable is False
        assert rec.unanswerable_reason is None
        assert rec.robustness_parent_qid is None
        assert rec.validation_scores == {}
        assert rec.metadata == {}

    def test_unanswerable_record(self) -> None:
        rec = BenchmarkRecord(
            qid="qc_unanswerable_0",
            query="What does 10 CFR 50.48(f) require?",
            query_class=QueryClass.UNANSWERABLE,
            difficulty="easy",
            source_unit_ids=(),
            source_citations=(),
            critical_evidence=(),
            supporting_evidence=(),
            contextual_evidence=(),
            hard_negative_chunk_ids=(),
            hard_negatives_retriever_config={},
            corpus_snapshot_id="abc123",
            valid_as_of="2026-03-24",
            is_unanswerable=True,
            unanswerable_reason="fabricated_citation",
        )
        assert rec.is_unanswerable is True
        assert rec.unanswerable_reason == "fabricated_citation"
        assert rec.critical_evidence == ()

    def test_json_round_trip(self) -> None:
        rec = _make_benchmark_record()
        d = dataclasses.asdict(rec)
        serialized = json.dumps(d, default=str)
        restored = json.loads(serialized)
        assert restored["qid"] == rec.qid
        assert restored["query_class"] == "citation_lookup"
        assert len(restored["critical_evidence"]) == 1
        assert restored["critical_evidence"][0]["span_id"] == "span_0"

    def test_evidence_tiers_populated(self) -> None:
        critical = _make_evidence_entry("s1", EvidenceTier.CRITICAL)
        supporting = _make_evidence_entry("s2", EvidenceTier.SUPPORTING)
        contextual = _make_evidence_entry("s3", EvidenceTier.CONTEXTUAL)
        rec = BenchmarkRecord(
            qid="qc_tiered",
            query="Test query for 10 CFR 50.46(b)(1)?",
            query_class=QueryClass.CITATION_LOOKUP,
            difficulty="medium",
            source_unit_ids=("unit_1",),
            source_citations=("10 CFR 50.46(b)(1)",),
            critical_evidence=(critical,),
            supporting_evidence=(supporting,),
            contextual_evidence=(contextual,),
            hard_negative_chunk_ids=(),
            hard_negatives_retriever_config={},
            corpus_snapshot_id="abc123",
            valid_as_of="2026-03-24",
        )
        assert len(rec.critical_evidence) == 1
        assert len(rec.supporting_evidence) == 1
        assert len(rec.contextual_evidence) == 1


# -- BenchmarkDataset tests ---------------------------------------------------


class TestBenchmarkDataset:
    def test_frozen(self) -> None:
        ds = BenchmarkDataset(
            schema_version="1.0",
            records=(),
            corpus_snapshot_id="abc123",
            created_at="2026-03-24T12:00:00Z",
        )
        assert dataclasses.is_dataclass(ds)

    def test_records_is_tuple(self) -> None:
        rec = _make_benchmark_record()
        ds = BenchmarkDataset(
            schema_version="1.0",
            records=(rec,),
            corpus_snapshot_id="abc123",
            created_at="2026-03-24T12:00:00Z",
        )
        assert isinstance(ds.records, tuple)
        assert len(ds.records) == 1

    def test_metadata_defaults_empty(self) -> None:
        ds = BenchmarkDataset(
            schema_version="1.0",
            records=(),
            corpus_snapshot_id="abc123",
            created_at="2026-03-24T12:00:00Z",
        )
        assert ds.metadata == {}

    def test_json_round_trip(self) -> None:
        rec = _make_benchmark_record()
        ds = BenchmarkDataset(
            schema_version="1.0",
            records=(rec,),
            corpus_snapshot_id="abc123",
            created_at="2026-03-24T12:00:00Z",
            metadata={"pipeline_version": "m4"},
        )
        d = dataclasses.asdict(ds)
        serialized = json.dumps(d, default=str)
        restored = json.loads(serialized)
        assert restored["schema_version"] == "1.0"
        assert len(restored["records"]) == 1
        assert restored["records"][0]["qid"] == rec.qid
        assert restored["metadata"]["pipeline_version"] == "m4"
