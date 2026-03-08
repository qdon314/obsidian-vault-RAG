# tests/eval/app_v2/engine/test_health.py
from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.derived.health import build_health
from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import QueryRecord, RunHealthSummary


def _records():
    def r(qid, relevant, retrieved):
        return QueryRecord(
            qid=qid, query="q", query_type=None, difficulty=None,
            is_unanswerable=False, requires_synthesis=False, tags=(),
            relevant_chunk_ids=frozenset(relevant),
            retrieved_chunk_ids=tuple(retrieved),
            reranked_chunk_ids=None, packed_chunk_ids=None,
            per_query_recall_at_k={10: len(set(relevant) & set(retrieved)) / max(len(relevant), 1)},
            per_query_precision_at_k={10: 0.5},
            per_query_ndcg_at_k={10: 0.7},
            per_query_hit_rate_at_k={10: 1.0},
            answer_text=None, answer_metrics=None, groundedness=None,
            latency_ms=100, trace_id=None, trace=None,
        )
    return [r("q1", ["c1"], ["c1"]), r("q2", ["c2"], ["c3"])]


def test_build_health_returns_summary():
    analyzed = analyze_queries(_records())
    health = build_health(analyzed, recall_at_10=0.5, ndcg_at_10=0.7)
    assert isinstance(health, RunHealthSummary)
    assert health.severity_counts[Severity.MODERATE] >= 1
    assert health.dominant_failure_mode == DiagnosticCode.RETRIEVAL_MISS
