"""Tests for the EvalQuery exporter adapter."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.adapters.export.eval_query_exporter import (
    _QUERY_CLASS_TO_QUERY_TYPE,
    EvalQueryExporter,
)
from benchmark.domain.enums import EvidenceTier, QueryClass
from benchmark.domain.models import (
    BenchmarkDataset,
    BenchmarkRecord,
    EvidenceEntry,
)
from benchmark.ports.exporter import BenchmarkExporter
from rag.eval.schema import EvalQuery

# -- Helpers ------------------------------------------------------------------


def _make_evidence_entry(
    span_id: str = "span_0",
    chunk_ids: tuple[str, ...] = ("chunk_1",),
    tier: EvidenceTier = EvidenceTier.CRITICAL,
) -> EvidenceEntry:
    return EvidenceEntry(
        span_id=span_id,
        citation="10 CFR 50.46(b)(1)",
        text="Peak cladding temperature.",
        char_start=0,
        char_end=26,
        chunk_ids=chunk_ids,
        tier=tier,
    )


def _make_record(
    qid: str = "qc_test_0",
    query_class: QueryClass = QueryClass.CITATION_LOOKUP,
    is_unanswerable: bool = False,
    unanswerable_reason: str | None = None,
    critical_chunk_ids: tuple[str, ...] = ("chunk_1",),
    supporting_chunk_ids: tuple[str, ...] = (),
    contextual_chunk_ids: tuple[str, ...] = (),
) -> BenchmarkRecord:
    critical = (
        (_make_evidence_entry(span_id="s1", chunk_ids=critical_chunk_ids),)
        if critical_chunk_ids
        else ()
    )
    supporting = (
        (_make_evidence_entry(span_id="s2", chunk_ids=supporting_chunk_ids, tier=EvidenceTier.SUPPORTING),)
        if supporting_chunk_ids
        else ()
    )
    contextual = (
        (_make_evidence_entry(span_id="s3", chunk_ids=contextual_chunk_ids, tier=EvidenceTier.CONTEXTUAL),)
        if contextual_chunk_ids
        else ()
    )
    return BenchmarkRecord(
        qid=qid,
        query="What is the peak cladding temperature under 10 CFR 50.46(b)(1)?",
        query_class=query_class,
        difficulty="easy",
        source_unit_ids=("50.46_b_1",),
        source_citations=("10 CFR 50.46(b)(1)",),
        critical_evidence=critical,
        supporting_evidence=supporting,
        contextual_evidence=contextual,
        hard_negative_chunk_ids=("chunk_99",),
        hard_negatives_retriever_config={"model": "test", "top_k": 20},
        corpus_snapshot_id="snap1",
        valid_as_of="2026-03-24",
        is_unanswerable=is_unanswerable,
        unanswerable_reason=unanswerable_reason,
    )


def _make_dataset(*records: BenchmarkRecord) -> BenchmarkDataset:
    return BenchmarkDataset(
        schema_version="1.0",
        records=tuple(records),
        corpus_snapshot_id="snap1",
        created_at="2026-03-24T00:00:00Z",
    )


# -- Protocol conformance -----------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_benchmark_exporter_protocol(self, tmp_path: Path) -> None:
        exporter = EvalQueryExporter(tmp_path / "out.jsonl")
        assert isinstance(exporter, BenchmarkExporter)


# -- Export output ------------------------------------------------------------


class TestExport:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        exporter = EvalQueryExporter(tmp_path / "out.jsonl")
        exporter.export(_make_dataset(_make_record()))

        assert (tmp_path / "out.jsonl").exists()

    def test_one_line_per_record(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        exporter = EvalQueryExporter(path)
        exporter.export(_make_dataset(_make_record("r1"), _make_record("r2")))

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "out.jsonl"
        exporter = EvalQueryExporter(path)
        exporter.export(_make_dataset(_make_record()))

        assert path.exists()

    def test_empty_dataset_creates_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        exporter = EvalQueryExporter(path)
        exporter.export(_make_dataset())

        assert path.exists()
        assert path.read_text().strip() == ""


# -- Field mapping ------------------------------------------------------------


class TestFieldMapping:
    def _load_first(self, tmp_path: Path, record: BenchmarkRecord) -> dict:
        path = tmp_path / "out.jsonl"
        EvalQueryExporter(path).export(_make_dataset(record))
        return json.loads(path.read_text().splitlines()[0])

    def test_qid_mapped(self, tmp_path: Path) -> None:
        row = self._load_first(tmp_path, _make_record(qid="my_qid"))
        assert row["qid"] == "my_qid"

    def test_query_mapped(self, tmp_path: Path) -> None:
        row = self._load_first(tmp_path, _make_record())
        assert "10 CFR 50.46" in row["query"]

    def test_relevant_chunk_ids_is_union_of_all_tiers(self, tmp_path: Path) -> None:
        record = _make_record(
            critical_chunk_ids=("c1",),
            supporting_chunk_ids=("s1",),
            contextual_chunk_ids=("ctx1",),
        )
        row = self._load_first(tmp_path, record)
        # Sets are serialized as lists by DataClassJsonMixin
        relevant = set(row["relevant_chunk_ids"])
        assert relevant == {"c1", "s1", "ctx1"}

    def test_critical_chunk_ids_correct(self, tmp_path: Path) -> None:
        record = _make_record(critical_chunk_ids=("c1", "c2"), supporting_chunk_ids=())
        row = self._load_first(tmp_path, record)
        assert set(row["critical_chunk_ids"]) == {"c1", "c2"}

    def test_supporting_chunk_ids_correct(self, tmp_path: Path) -> None:
        record = _make_record(supporting_chunk_ids=("s1",), contextual_chunk_ids=())
        row = self._load_first(tmp_path, record)
        assert set(row["supporting_chunk_ids"]) == {"s1"}

    def test_context_chunk_ids_correct(self, tmp_path: Path) -> None:
        record = _make_record(contextual_chunk_ids=("ctx1",))
        row = self._load_first(tmp_path, record)
        assert set(row["context_chunk_ids"]) == {"ctx1"}

    def test_source_citations_mapped_to_relevant_citations(self, tmp_path: Path) -> None:
        row = self._load_first(tmp_path, _make_record())
        assert "10 CFR 50.46(b)(1)" in row["relevant_citations"]

    def test_unanswerable_fields_mapped(self, tmp_path: Path) -> None:
        record = _make_record(is_unanswerable=True, unanswerable_reason="fabricated_citation")
        row = self._load_first(tmp_path, record)
        assert row["is_unanswerable"] is True
        assert row["unanswerable_reason"] == "fabricated_citation"

    def test_is_unanswerable_false_by_default(self, tmp_path: Path) -> None:
        row = self._load_first(tmp_path, _make_record())
        assert row["is_unanswerable"] is False


# -- Query type mapping -------------------------------------------------------


class TestQueryTypeMapping:
    def _get_query_type(self, tmp_path: Path, qc: QueryClass) -> str:
        record = _make_record(query_class=qc)
        path = tmp_path / f"out_{qc.value}.jsonl"
        EvalQueryExporter(path).export(_make_dataset(record))
        return json.loads(path.read_text())["query_type"]

    def test_citation_lookup_maps_to_factual(self, tmp_path: Path) -> None:
        assert self._get_query_type(tmp_path, QueryClass.CITATION_LOOKUP) == "factual"

    def test_rule_explanation_maps_to_procedural(self, tmp_path: Path) -> None:
        assert self._get_query_type(tmp_path, QueryClass.RULE_EXPLANATION) == "procedural"

    def test_cross_reference_maps_to_multi_hop(self, tmp_path: Path) -> None:
        assert self._get_query_type(tmp_path, QueryClass.CROSS_REFERENCE) == "multi_hop"

    def test_unanswerable_maps_to_factual(self, tmp_path: Path) -> None:
        assert self._get_query_type(tmp_path, QueryClass.UNANSWERABLE) == "factual"

    def test_all_query_classes_have_mapping(self) -> None:
        for qc in QueryClass:
            assert qc in _QUERY_CLASS_TO_QUERY_TYPE, f"Missing mapping for {qc}"


# -- Round-trip ---------------------------------------------------------------


class TestRoundTrip:
    def test_output_loadable_as_eval_query(self, tmp_path: Path) -> None:
        record = _make_record()
        path = tmp_path / "out.jsonl"
        EvalQueryExporter(path).export(_make_dataset(record))

        line = path.read_text().strip()
        eq = EvalQuery.from_json(line)

        assert eq.qid == record.qid
        assert eq.query == record.query
        assert not eq.is_unanswerable

    def test_unanswerable_round_trip(self, tmp_path: Path) -> None:
        record = _make_record(
            is_unanswerable=True,
            unanswerable_reason="near_miss",
            critical_chunk_ids=(),
        )
        path = tmp_path / "out.jsonl"
        EvalQueryExporter(path).export(_make_dataset(record))

        eq = EvalQuery.from_json(path.read_text().strip())

        assert eq.is_unanswerable is True
        assert eq.unanswerable_reason == "near_miss"
