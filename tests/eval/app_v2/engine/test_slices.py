# tests/eval/app_v2/engine/test_slices.py
from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.derived.slices import build_slice_table
from eval.app_v2.engine.domain.models import QueryRecord, SliceMetricTable


def _r(qid, qtype, recall):
    return QueryRecord(
        qid=qid,
        query="q",
        query_type=qtype,
        difficulty=None,
        is_unanswerable=False,
        requires_synthesis=False,
        tags=(),
        relevant_chunk_ids=frozenset(["c1"]),
        retrieved_chunk_ids=("c1",),
        reranked_chunk_ids=None,
        packed_chunk_ids=None,
        per_query_recall_at_k={10: recall},
        per_query_precision_at_k={10: recall},
        per_query_ndcg_at_k={10: recall},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None,
        answer_metrics=None,
        groundedness=None,
        latency_ms=None,
        trace_id=None,
        trace=None,
    )


def test_build_slice_table_groups_by_query_type():
    analyzed = analyze_queries(
        [_r("q1", "factual", 1.0), _r("q2", "factual", 0.5), _r("q3", "conceptual", 0.0)]
    )
    table = build_slice_table(analyzed, group_by=["query_type"])
    assert isinstance(table, SliceMetricTable)
    keys = [dict(r.key.parts)["query_type"] for r in table.rows]
    assert "factual" in keys
    assert "conceptual" in keys


def test_build_slice_table_multi_group():
    analyzed = analyze_queries([_r("q1", "factual", 1.0), _r("q2", "conceptual", 0.0)])
    table = build_slice_table(analyzed, group_by=["query_type", "difficulty"])
    assert isinstance(table, SliceMetricTable)
