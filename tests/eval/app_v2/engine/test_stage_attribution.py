# tests/eval/app_v2/engine/test_stage_attribution.py
from eval.app_v2.engine.derived.stage_attribution import classify_query
from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import QueryRecord


def _record(**kwargs) -> QueryRecord:
    defaults = {
        "qid": "q1",
        "query": "test",
        "query_type": None,
        "difficulty": None,
        "is_unanswerable": False,
        "requires_synthesis": False,
        "tags": (),
        "relevant_chunk_ids": frozenset(["c1"]),
        "retrieved_chunk_ids": ("c1",),
        "reranked_chunk_ids": None,
        "packed_chunk_ids": None,
        "per_query_recall_at_k": {10: 1.0},
        "per_query_precision_at_k": {10: 1.0},
        "per_query_ndcg_at_k": {10: 1.0},
        "per_query_hit_rate_at_k": {10: 1.0},
        "answer_text": None,
        "answer_metrics": None,
        "groundedness": None,
        "latency_ms": None,
        "trace_id": None,
        "trace": None,
    }
    defaults.update(kwargs)
    return QueryRecord(**defaults)


def test_retrieval_miss():
    r = _record(relevant_chunk_ids=frozenset(["c1"]), retrieved_chunk_ids=("c2", "c3"))
    code, severity = classify_query(r)
    assert code == DiagnosticCode.RETRIEVAL_MISS
    assert severity == Severity.MODERATE


def test_grounded_answer():
    r = _record(relevant_chunk_ids=frozenset(["c1"]), retrieved_chunk_ids=("c1",))
    code, severity = classify_query(r)
    assert code == DiagnosticCode.GROUNDED_ANSWER
    assert severity == Severity.OK


def test_retrieval_partial():
    r = _record(
        relevant_chunk_ids=frozenset(["c1", "c2"]),
        retrieved_chunk_ids=("c1",),
        per_query_recall_at_k={10: 0.5},
    )
    code, severity = classify_query(r)
    assert code == DiagnosticCode.RETRIEVAL_PARTIAL
    assert severity == Severity.MINOR


def test_no_relevant_chunks_is_data_insufficient():
    r = _record(relevant_chunk_ids=frozenset())
    code, _ = classify_query(r)
    assert code == DiagnosticCode.DATA_INSUFFICIENT
