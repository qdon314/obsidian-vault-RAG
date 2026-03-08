# tests/eval/app_v2/engine/test_diagnostics.py
from eval.app_v2.engine.derived.diagnostics import analyze_queries, build_query_diagnostic
from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import QueryRecord


def _record(qid="q1", relevant=frozenset(["c1"]), retrieved=("c1",)):
    return QueryRecord(
        qid=qid, query="q", query_type=None, difficulty=None,
        is_unanswerable=False, requires_synthesis=False, tags=(),
        relevant_chunk_ids=relevant,
        retrieved_chunk_ids=retrieved,
        reranked_chunk_ids=None, packed_chunk_ids=None,
        per_query_recall_at_k={10: 1.0},
        per_query_precision_at_k={10: 1.0},
        per_query_ndcg_at_k={10: 1.0},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None, answer_metrics=None, groundedness=None,
        latency_ms=None, trace_id=None, trace=None,
    )


def test_build_query_diagnostic_returns_diagnostic():
    from eval.app_v2.engine.domain.models import QueryDiagnostic
    diag = build_query_diagnostic(_record())
    assert isinstance(diag, QueryDiagnostic)
    assert diag.qid == "q1"


def test_analyze_queries_returns_analyzed_queries():
    from eval.app_v2.engine.domain.models import AnalyzedQuery
    records = [_record("q1"), _record("q2", retrieved=("c2",))]
    analyzed = analyze_queries(records)
    assert len(analyzed) == 2
    assert all(isinstance(a, AnalyzedQuery) for a in analyzed)
    codes = {a.diagnostic.diagnostic_code for a in analyzed}
    assert DiagnosticCode.RETRIEVAL_MISS in codes
