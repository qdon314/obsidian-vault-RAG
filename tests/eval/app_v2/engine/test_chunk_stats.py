from eval.app_v2.engine.derived.chunk_stats import build_chunk_stats
from eval.app_v2.engine.domain.enums import (
    DiagnosticCode,
    GenerationStatus,
    PackingStatus,
    RerankStatus,
    RetrievalStatus,
    Severity,
)
from eval.app_v2.engine.domain.models import AnalyzedQuery, QueryDiagnostic, QueryRecord


def _diag(qid: str = "q1") -> QueryDiagnostic:
    return QueryDiagnostic(
        qid=qid,
        diagnostic_code=DiagnosticCode.GROUNDED_ANSWER,
        severity=Severity.OK,
        retrieval_status=RetrievalStatus.HIT,
        rerank_status=RerankStatus.ABSENT,
        packing_status=PackingStatus.ABSENT,
        generation_status=GenerationStatus.GROUNDED,
        root_cause_summary="",
        suggested_next_check=None,
        evidence_present=True,
        trace_available=False,
    )


def _aq(
    qid: str,
    relevant: list[str],
    retrieved: list[str],
    reranked: list[str] | None = None,
    packed: list[str] | None = None,
) -> AnalyzedQuery:
    record = QueryRecord(
        qid=qid,
        query="q",
        query_type=None,
        difficulty=None,
        is_unanswerable=False,
        requires_synthesis=False,
        tags=(),
        relevant_chunk_ids=frozenset(relevant),
        retrieved_chunk_ids=tuple(retrieved),
        reranked_chunk_ids=tuple(reranked) if reranked is not None else None,
        packed_chunk_ids=tuple(packed) if packed is not None else None,
        per_query_recall_at_k={10: 1.0},
        per_query_precision_at_k={10: 1.0},
        per_query_ndcg_at_k={10: 1.0},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None,
        answer_metrics=None,
        groundedness=None,
        latency_ms=None,
        trace_id=None,
        trace=None,
    )
    return AnalyzedQuery(record=record, diagnostic=_diag(qid))


def test_fully_missed_chunk_has_miss_rate_1():
    queries = [_aq("q1", relevant=["c1"], retrieved=["c2"])]
    stats = build_chunk_stats(queries)
    c1 = next(s for s in stats if s.chunk_id == "c1")
    assert c1.miss_rate == 1.0
    assert c1.queries_where_relevant == 1
    assert c1.queries_where_retrieved == 0


def test_always_retrieved_chunk_has_miss_rate_0():
    queries = [
        _aq("q1", relevant=["c1"], retrieved=["c1"]),
        _aq("q2", relevant=["c1"], retrieved=["c1"]),
    ]
    stats = build_chunk_stats(queries)
    c1 = next(s for s in stats if s.chunk_id == "c1")
    assert c1.miss_rate == 0.0


def test_rerank_drop_detected():
    queries = [_aq("q1", relevant=["c1"], retrieved=["c1", "c2"], reranked=["c2"])]
    stats = build_chunk_stats(queries)
    c1 = next(s for s in stats if s.chunk_id == "c1")
    assert c1.rerank_drop_rate > 0.0


def test_sorted_by_miss_rate_descending():
    queries = [
        _aq("q1", relevant=["bad"], retrieved=[]),  # miss_rate = 1.0
        _aq("q2", relevant=["good"], retrieved=["good"]),  # miss_rate = 0.0
    ]
    stats = build_chunk_stats(queries)
    assert stats[0].chunk_id == "bad"
    assert stats[-1].chunk_id == "good"


def test_no_queries_returns_empty():
    assert build_chunk_stats([]) == ()


def test_every_stat_has_nonzero_presence():
    queries = [_aq("q1", relevant=["c1"], retrieved=["c2"])]
    stats = build_chunk_stats(queries)
    for s in stats:
        assert s.queries_where_relevant > 0 or s.queries_where_retrieved > 0
