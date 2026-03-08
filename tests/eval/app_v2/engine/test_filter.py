# tests/eval/app_v2/engine/test_filter.py
from eval.app_v2.engine.services.filter import apply_facet_filters
from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.domain.models import QueryRecord


def _r(qid, qtype, difficulty):
    return QueryRecord(
        qid=qid, query="q", query_type=qtype, difficulty=difficulty,
        is_unanswerable=False, requires_synthesis=False, tags=(),
        relevant_chunk_ids=frozenset(["c1"]),
        retrieved_chunk_ids=("c1",),
        reranked_chunk_ids=None, packed_chunk_ids=None,
        per_query_recall_at_k={10: 1.0},
        per_query_precision_at_k={10: 1.0},
        per_query_ndcg_at_k={10: 1.0},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None, answer_metrics=None, groundedness=None,
        latency_ms=None, trace_id=None, trace=None,
    )


def test_filter_by_query_type():
    analyzed = analyze_queries([
        _r("q1", "factual", "easy"),
        _r("q2", "conceptual", "hard"),
        _r("q3", "factual", "hard"),
    ])
    filtered = apply_facet_filters(analyzed, {"query_type": "factual"})
    assert len(filtered) == 2
    assert all(a.record.query_type == "factual" for a in filtered)


def test_no_filter_returns_all():
    analyzed = analyze_queries([_r("q1", "factual", "easy"), _r("q2", "conceptual", "hard")])
    filtered = apply_facet_filters(analyzed, {})
    assert len(filtered) == 2


def test_none_value_is_no_filter():
    analyzed = analyze_queries([_r("q1", "factual", "easy"), _r("q2", "conceptual", "hard")])
    filtered = apply_facet_filters(analyzed, {"query_type": None})
    assert len(filtered) == 2
