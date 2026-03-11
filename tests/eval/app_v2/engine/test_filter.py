# tests/eval/app_v2/engine/test_filter.py
from rag.eval.answer_metrics import AnswerQualityMetrics

from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import QueryRecord
from eval.app_v2.engine.services.filter import apply_facet_filters


def _r(qid, qtype="factual", difficulty="easy", recall_10=1.0, quality_score=None, latency_ms=None):
    am = AnswerQualityMetrics(quality_score=quality_score) if quality_score is not None else None
    return QueryRecord(
        qid=qid,
        query="q",
        query_type=qtype,
        difficulty=difficulty,
        is_unanswerable=False,
        requires_synthesis=False,
        tags=(),
        relevant_chunk_ids=frozenset(["c1"]),
        retrieved_chunk_ids=("c1",) if recall_10 > 0 else (),
        reranked_chunk_ids=None,
        packed_chunk_ids=None,
        per_query_recall_at_k={10: recall_10},
        per_query_precision_at_k={10: 1.0},
        per_query_ndcg_at_k={10: recall_10},
        per_query_hit_rate_at_k={10: 1.0 if recall_10 > 0 else 0.0},
        answer_text=None,
        answer_metrics=am,
        groundedness=None,
        latency_ms=latency_ms,
        trace_id=None,
        trace=None,
    )


def test_filter_by_query_type():
    analyzed = analyze_queries(
        [
            _r("q1", "factual", "easy"),
            _r("q2", "conceptual", "hard"),
            _r("q3", "factual", "hard"),
        ]
    )
    filtered = apply_facet_filters(analyzed, {"query_type": "factual"})
    assert len(filtered) == 2
    assert all(a.record.query_type == "factual" for a in filtered)


def test_no_filter_returns_all():
    analyzed = analyze_queries([_r("q1"), _r("q2", "conceptual")])
    filtered = apply_facet_filters(analyzed, {})
    assert len(filtered) == 2


def test_none_value_is_no_filter():
    analyzed = analyze_queries([_r("q1"), _r("q2", "conceptual")])
    filtered = apply_facet_filters(analyzed, {"query_type": None})
    assert len(filtered) == 2


# ── New: numeric range filter tests ──────────────────────────────────────────


def test_numeric_range_filter_within_bounds():
    """Only queries with recall@10 ≤ 0.5 should pass the filter."""
    analyzed = analyze_queries(
        [
            _r("q1", recall_10=0.2),
            _r("q2", recall_10=0.5),
            _r("q3", recall_10=0.8),
            _r("q4", recall_10=1.0),
        ]
    )
    filtered = apply_facet_filters(analyzed, {"recall_at_10": (0.0, 0.5)})
    assert len(filtered) == 2
    assert all(a.record.per_query_recall_at_k[10] <= 0.5 for a in filtered)


def test_numeric_range_full_range_no_filter():
    """Setting the filter to the full range should return all queries."""
    analyzed = analyze_queries(
        [
            _r("q1", recall_10=0.0),
            _r("q2", recall_10=0.5),
            _r("q3", recall_10=1.0),
        ]
    )
    # None == no filter (full range)
    filtered = apply_facet_filters(analyzed, {"recall_at_10": None})
    assert len(filtered) == 3


def test_numeric_range_null_passthrough():
    """Queries where the metric is None (e.g. no answer_metrics) should pass through."""
    # q1 has no quality score (answer_metrics=None)
    # q2 has a low quality score
    analyzed = analyze_queries(
        [
            _r("q1", quality_score=None),
            _r("q2", quality_score=0.3),
            _r("q3", quality_score=0.9),
        ]
    )
    # Filter to quality_score ≤ 0.5
    filtered = apply_facet_filters(analyzed, {"quality_score": (0.0, 0.5)})
    qids = {a.record.qid for a in filtered}
    # q1 (None) passes through, q2 (0.3) within range, q3 (0.9) excluded
    assert "q1" in qids
    assert "q2" in qids
    assert "q3" not in qids


def test_numeric_range_latency():
    """Latency range filter should work on latency_ms field."""
    analyzed = analyze_queries(
        [
            _r("q1", latency_ms=100),
            _r("q2", latency_ms=500),
            _r("q3", latency_ms=2000),
        ]
    )
    filtered = apply_facet_filters(analyzed, {"latency_ms": (0.0, 600.0)})
    assert len(filtered) == 2
    assert all(a.record.latency_ms is not None and a.record.latency_ms <= 600 for a in filtered)


def test_diagnostic_code_filter():
    """Filtering by diagnostic code should return only queries with that code."""
    analyzed = analyze_queries(
        [
            _r("q1", recall_10=0.0),  # will be RETRIEVAL_MISS
            _r("q2", recall_10=1.0),  # will be GROUNDED_ANSWER or NO_CLEAR_FAILURE
            _r("q3", recall_10=0.5),  # will be RETRIEVAL_PARTIAL
        ]
    )
    # Find the actual code for q1 and filter on it
    q1_code = next(a.diagnostic.diagnostic_code for a in analyzed if a.record.qid == "q1")
    filtered = apply_facet_filters(analyzed, {"diagnostic_code": str(q1_code)})
    assert len(filtered) >= 1, f"Expected at least one match for code {q1_code!r}"
    assert all(str(a.diagnostic.diagnostic_code) == str(q1_code) for a in filtered)


def test_conjunctive_numeric_and_enum():
    """Combining a recall range filter with a difficulty enum filter should AND them."""
    analyzed = analyze_queries(
        [
            _r("q1", difficulty="hard", recall_10=0.2),
            _r("q2", difficulty="hard", recall_10=0.9),
            _r("q3", difficulty="easy", recall_10=0.2),
            _r("q4", difficulty="easy", recall_10=0.9),
        ]
    )
    filtered = apply_facet_filters(
        analyzed,
        {
            "difficulty": "hard",
            "recall_at_10": (0.0, 0.5),
        },
    )
    assert len(filtered) == 1
    assert filtered[0].record.qid == "q1"
