# tests/eval/app_v2/engine/test_health.py
from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.derived.health import build_health
from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import QueryRecord, RunHealthSummary
from rag.eval.models import EvalAggregates, RetrievalSummary


def _make_aggregates(
    recall_10=0.5,
    ndcg_10=0.7,
    mrr=0.6,
    map_=0.55,
    hit_rate_10=0.8,
    precision_10=0.3,
    critical_recall=None,
    weighted_recall=None,
    answer_quality=None,
    latency_ms=None,
) -> EvalAggregates:
    overall = RetrievalSummary(
        num_queries=2,
        avg_retrieved=10.0,
        recall_at_k={10: recall_10},
        ndcg_at_k={10: ndcg_10},
        hit_rate_at_k={10: hit_rate_10},
        precision_at_k={10: precision_10},
        critical_recall_at_k={10: critical_recall} if critical_recall is not None else {},
        weighted_recall_at_k={10: weighted_recall} if weighted_recall is not None else {},
        mrr=mrr,
        map=map_,
    )
    # EvalAggregates accepts None for answer_quality/latency_ms; the `or {}` here
    # mirrors build_health's own guard so the helper produces the same effect.
    return EvalAggregates(
        overall=overall,
        answer_quality=answer_quality if answer_quality is not None else {},
        latency_ms=latency_ms if latency_ms is not None else {},
    )


def _records():
    def r(qid, relevant, retrieved):
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
            per_query_recall_at_k={10: len(set(relevant) & set(retrieved)) / max(len(relevant), 1)},
            per_query_precision_at_k={10: 0.5},
            per_query_ndcg_at_k={10: 0.7},
            per_query_hit_rate_at_k={10: 1.0},
            answer_text=None,
            answer_metrics=None,
            groundedness=None,
            latency_ms=100,
            trace_id=None,
            trace=None,
        )

    return [r("q1", ["c1"], ["c1"]), r("q2", ["c2"], ["c3"])]


def test_build_health_returns_summary():
    analyzed = analyze_queries(_records())
    aggs = _make_aggregates()
    health = build_health(analyzed, aggs)
    assert isinstance(health, RunHealthSummary)
    assert health.severity_counts[Severity.MODERATE] >= 1
    assert health.dominant_failure_mode == DiagnosticCode.RETRIEVAL_MISS


def test_build_health_full_aggregates():
    analyzed = analyze_queries(_records())
    aggs = _make_aggregates(
        recall_10=0.6,
        ndcg_10=0.5,
        mrr=0.65,
        map_=0.55,
        hit_rate_10=0.9,
        precision_10=0.4,
        critical_recall=0.7,
        weighted_recall=0.55,
        answer_quality={
            "avg_quality_score": 0.72,
            "median_quality_score": 0.75,
            "avg_correctness_0_5": 3.5,
            "avg_hallucination_severity_0_5": 1.2,
            "avg_citation_coverage": 0.85,
            "evidence_bounded_rate": 0.68,
            "hallucinated_on_unanswerable_rate": 0.05,
        },
        latency_ms={"avg": 450.0, "p50": 400.0, "p95": 900.0},
    )
    health = build_health(analyzed, aggs)

    assert health.headline_recall_at_10 == 0.6
    assert health.headline_ndcg_at_10 == 0.5
    assert health.headline_mrr == 0.65
    assert health.headline_map == 0.55
    assert health.headline_hit_rate_at_10 == 0.9
    assert health.headline_precision_at_10 == 0.4
    assert health.headline_critical_recall_at_10 == 0.7
    assert health.headline_weighted_recall_at_10 == 0.55

    assert health.avg_quality_score == 0.72
    assert health.median_quality_score == 0.75
    assert health.avg_correctness == 3.5
    assert health.avg_hallucination_severity == 1.2
    assert health.avg_citation_coverage == 0.85
    assert health.evidence_bounded_rate == 0.68
    assert health.hallucinated_on_unanswerable_rate == 0.05

    assert health.avg_latency_ms == 450.0
    assert health.p50_latency_ms == 400.0
    assert health.p95_latency_ms == 900.0


def test_build_health_retrieval_only():
    """For a retrieval-only run, all answer-quality and latency fields should be None."""
    analyzed = analyze_queries(_records())
    aggs = _make_aggregates()  # no answer_quality, no latency
    health = build_health(analyzed, aggs)

    assert health.avg_correctness is None
    assert health.avg_hallucination_severity is None
    assert health.avg_citation_coverage is None
    assert health.evidence_bounded_rate is None
    assert health.hallucinated_on_unanswerable_rate is None
    assert health.median_quality_score is None
    assert health.p50_latency_ms is None
    assert health.p95_latency_ms is None


def test_build_health_no_tiered_metrics():
    """When critical_recall_at_k is empty, headline_critical_recall_at_10 should be None."""
    analyzed = analyze_queries(_records())
    aggs = _make_aggregates()  # critical_recall=None → empty dict
    health = build_health(analyzed, aggs)

    assert health.headline_critical_recall_at_10 is None
    assert health.headline_weighted_recall_at_10 is None


def test_build_health_avg_quality_falls_back_to_per_query():
    """If aggregates.answer_quality has no avg_quality_score, fall back to per-query average."""
    from rag.eval.answer_metrics import AnswerQualityMetrics

    def r_with_quality(qid, qs):
        am = AnswerQualityMetrics(quality_score=qs)
        return QueryRecord(
            qid=qid,
            query="q",
            query_type=None,
            difficulty=None,
            is_unanswerable=False,
            requires_synthesis=False,
            tags=(),
            relevant_chunk_ids=frozenset(["c1"]),
            retrieved_chunk_ids=("c1",),
            reranked_chunk_ids=None,
            packed_chunk_ids=None,
            per_query_recall_at_k={10: 1.0},
            per_query_precision_at_k={10: 1.0},
            per_query_ndcg_at_k={10: 1.0},
            per_query_hit_rate_at_k={10: 1.0},
            answer_text="yes",
            answer_metrics=am,
            groundedness=None,
            latency_ms=None,
            trace_id=None,
            trace=None,
        )

    records = [r_with_quality("q1", 0.6), r_with_quality("q2", 0.8)]
    analyzed = analyze_queries(records)
    aggs = _make_aggregates(answer_quality={})  # empty — no avg_quality_score
    health = build_health(analyzed, aggs)

    assert health.avg_quality_score is not None
    assert abs(health.avg_quality_score - 0.7) < 1e-9

def test_build_health_empty_run_gives_none_extended_metrics():
    """When aggregates has num_queries=0, headline MRR/MAP/hit_rate/precision are None."""
    from rag.eval.models import RetrievalSummary
    empty_aggs = EvalAggregates(overall=RetrievalSummary(num_queries=0, avg_retrieved=0.0))
    health = build_health([], empty_aggs)

    assert health.headline_mrr is None
    assert health.headline_map is None
    assert health.headline_hit_rate_at_10 is None
    assert health.headline_precision_at_10 is None
