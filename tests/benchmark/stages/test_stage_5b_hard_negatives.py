"""Tests for Stage 5b hard negative mining."""

from __future__ import annotations

from unittest.mock import MagicMock

from benchmark.domain.enums import EvidenceTier, QueryClass
from benchmark.domain.models import (
    EvidenceEntry,
    EvidenceSet,
    ValidatedQuery,
)
from benchmark.stages.stage_5b_hard_negatives import mine_hard_negatives
from rag.domain.models import Candidate, Chunk

# -- Helpers ------------------------------------------------------------------

_RETRIEVER_CONFIG = {"model": "text-embedding-3-small", "index_version": "v1", "top_k": 20}


def _make_validated_query(
    candidate_id: str = "qc_0",
    unit_id: str = "50.46_b_1",
    query: str = "What is the peak cladding temperature under 10 CFR 50.46(b)(1)?",
) -> ValidatedQuery:
    return ValidatedQuery(
        candidate_id=candidate_id,
        unit_id=unit_id,
        query=query,
        query_class=QueryClass.CITATION_LOOKUP,
        source_citations=("10 CFR 50.46(b)(1)",),
        evidence_span_ids=("span_0",),
        difficulty="easy",
        corpus_snapshot_id="snap1",
    )


def _make_evidence(
    unit_id: str = "50.46_b_1",
    chunk_ids: tuple[str, ...] = ("chunk_1", "chunk_2"),
) -> EvidenceSet:
    entry = EvidenceEntry(
        span_id="span_0",
        citation="10 CFR 50.46(b)(1)",
        text="Peak cladding temperature.",
        char_start=0,
        char_end=26,
        chunk_ids=chunk_ids,
        tier=EvidenceTier.CRITICAL,
    )
    return EvidenceSet(unit_id=unit_id, critical=(entry,))


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        text="",
        chunk_index=0,
        metadata={},
    )


def _make_candidate(chunk_id: str) -> Candidate:
    return Candidate(chunk=_make_chunk(chunk_id), score=0.9)


def _mock_retriever(chunk_ids: list[str]) -> MagicMock:
    mock = MagicMock()
    mock.retrieve.return_value = [_make_candidate(cid) for cid in chunk_ids]
    return mock


# -- Core behavior ------------------------------------------------------------


class TestMineHardNegatives:
    def test_returns_one_result_per_query(self) -> None:
        queries = [_make_validated_query("qc_0"), _make_validated_query("qc_1")]
        evidence = {"50.46_b_1": _make_evidence()}
        retriever = _mock_retriever(["chunk_10", "chunk_11", "chunk_12"])

        results = mine_hard_negatives(
            queries, evidence, retriever, retriever_config=_RETRIEVER_CONFIG
        )

        assert len(results) == 2

    def test_result_candidate_ids_match_queries(self) -> None:
        queries = [_make_validated_query("qc_0"), _make_validated_query("qc_1")]
        evidence = {"50.46_b_1": _make_evidence()}
        retriever = _mock_retriever(["chunk_10"])

        results = mine_hard_negatives(
            queries, evidence, retriever, retriever_config=_RETRIEVER_CONFIG
        )

        assert results[0].candidate_id == "qc_0"
        assert results[1].candidate_id == "qc_1"

    def test_evidence_chunk_ids_excluded(self) -> None:
        # Evidence contains chunk_1 and chunk_2
        # Retriever returns chunk_1, chunk_2, chunk_3 → only chunk_3 is a hard negative
        queries = [_make_validated_query()]
        evidence = {"50.46_b_1": _make_evidence(chunk_ids=("chunk_1", "chunk_2"))}
        retriever = _mock_retriever(["chunk_1", "chunk_2", "chunk_3"])

        results = mine_hard_negatives(
            queries, evidence, retriever, retriever_config=_RETRIEVER_CONFIG
        )

        assert results[0].hard_negative_chunk_ids == ("chunk_3",)

    def test_all_tiers_excluded_from_hard_negatives(self) -> None:
        # Evidence with critical + supporting + contextual tiers
        critical = EvidenceEntry(
            span_id="s1",
            citation="c",
            text="t",
            char_start=0,
            char_end=1,
            chunk_ids=("critical_chunk",),
            tier=EvidenceTier.CRITICAL,
        )
        supporting = EvidenceEntry(
            span_id="s2",
            citation="c",
            text="t",
            char_start=0,
            char_end=1,
            chunk_ids=("supporting_chunk",),
            tier=EvidenceTier.SUPPORTING,
        )
        contextual = EvidenceEntry(
            span_id="s3",
            citation="c",
            text="t",
            char_start=0,
            char_end=1,
            chunk_ids=("contextual_chunk",),
            tier=EvidenceTier.CONTEXTUAL,
        )
        evidence = {
            "50.46_b_1": EvidenceSet(
                unit_id="50.46_b_1",
                critical=(critical,),
                supporting=(supporting,),
                contextual=(contextual,),
            )
        }
        queries = [_make_validated_query()]
        retriever = _mock_retriever(
            ["critical_chunk", "supporting_chunk", "contextual_chunk", "new_chunk"]
        )

        results = mine_hard_negatives(
            queries, evidence, retriever, retriever_config=_RETRIEVER_CONFIG
        )

        assert results[0].hard_negative_chunk_ids == ("new_chunk",)

    def test_retriever_config_passed_through(self) -> None:
        config = {"model": "test-model", "index_version": "v2", "top_k": 5}
        queries = [_make_validated_query()]
        evidence = {"50.46_b_1": _make_evidence()}
        retriever = _mock_retriever(["chunk_99"])

        results = mine_hard_negatives(queries, evidence, retriever, retriever_config=config)

        assert results[0].retriever_config == config


# -- Insufficient flag --------------------------------------------------------


class TestInsufficientFlag:
    def test_insufficient_false_when_enough_hard_negatives(self) -> None:
        queries = [_make_validated_query()]
        evidence = {"50.46_b_1": _make_evidence()}
        retriever = _mock_retriever(["chunk_10", "chunk_11", "chunk_12"])

        results = mine_hard_negatives(
            queries,
            evidence,
            retriever,
            min_hard_negatives=2,
            retriever_config=_RETRIEVER_CONFIG,
        )

        assert results[0].insufficient is False

    def test_insufficient_true_when_below_threshold(self) -> None:
        queries = [_make_validated_query()]
        # Evidence contains chunk_1; retriever returns chunk_1 + chunk_10
        # → only 1 hard negative (chunk_10), below min=2
        evidence = {"50.46_b_1": _make_evidence(chunk_ids=("chunk_1",))}
        retriever = _mock_retriever(["chunk_1", "chunk_10"])

        results = mine_hard_negatives(
            queries,
            evidence,
            retriever,
            min_hard_negatives=2,
            retriever_config=_RETRIEVER_CONFIG,
        )

        assert results[0].insufficient is True

    def test_insufficient_true_on_empty_retriever_result(self) -> None:
        queries = [_make_validated_query()]
        evidence = {"50.46_b_1": _make_evidence()}
        retriever = _mock_retriever([])  # Returns nothing

        results = mine_hard_negatives(
            queries, evidence, retriever, retriever_config=_RETRIEVER_CONFIG
        )

        assert results[0].hard_negative_chunk_ids == ()
        assert results[0].insufficient is True


# -- Edge cases ---------------------------------------------------------------


class TestEdgeCases:
    def test_no_evidence_set_for_unit(self) -> None:
        """Query's unit_id has no evidence set — all retrieved chunks are hard negatives."""
        queries = [_make_validated_query(unit_id="unit_with_no_evidence")]
        evidence: dict[str, EvidenceSet] = {}
        retriever = _mock_retriever(["chunk_10", "chunk_11"])

        results = mine_hard_negatives(
            queries, evidence, retriever, retriever_config=_RETRIEVER_CONFIG
        )

        assert set(results[0].hard_negative_chunk_ids) == {"chunk_10", "chunk_11"}

    def test_retriever_called_with_correct_query_text(self) -> None:
        query_text = "What is the peak cladding temperature under 10 CFR 50.46(b)(1)?"
        queries = [_make_validated_query(query=query_text)]
        evidence = {"50.46_b_1": _make_evidence()}
        retriever = _mock_retriever([])

        mine_hard_negatives(queries, evidence, retriever, retriever_config=_RETRIEVER_CONFIG)

        retriever.retrieve.assert_called_once_with(query_text, top_k=20)

    def test_custom_top_k_passed_to_retriever(self) -> None:
        queries = [_make_validated_query()]
        evidence = {"50.46_b_1": _make_evidence()}
        retriever = _mock_retriever([])

        mine_hard_negatives(
            queries, evidence, retriever, top_k=5, retriever_config=_RETRIEVER_CONFIG
        )

        retriever.retrieve.assert_called_once_with(queries[0].query, top_k=5)
