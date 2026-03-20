from eval.app_v2.engine.derived.contributors import contributor_queries_for_code
from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import QueryRecord


def _r(qid, relevant, retrieved):
    return QueryRecord(
        qid=qid,
        query="q",
        query_type=None,
        difficulty=None,
        is_unanswerable=False,
        requires_synthesis=False,
        tags=(),
        relevant_chunk_ids=frozenset(relevant),
        retrieved_chunk_ids=tuple(retrieved),
        reranked_chunk_ids=None,
        packed_chunk_ids=None,
        per_query_recall_at_k={10: 0.0},
        per_query_precision_at_k={10: 0.0},
        per_query_ndcg_at_k={10: 0.0},
        per_query_hit_rate_at_k={10: 0.0},
        answer_text=None,
        answer_metrics=None,
        groundedness=None,
        latency_ms=None,
        trace_id=None,
        trace=None,
    )


def test_contributor_queries_for_retrieval_miss():
    analyzed = analyze_queries(
        [
            _r("q1", ["c1"], ["c2"]),  # miss
            _r("q2", ["c2"], ["c2"]),  # hit
            _r("q3", ["c3"], ["c4"]),  # miss
        ]
    )
    contributors = contributor_queries_for_code(analyzed, DiagnosticCode.RETRIEVAL_MISS, limit=10)
    assert len(contributors) == 2
    qids = {a.record.qid for a in contributors}
    assert qids == {"q1", "q3"}
