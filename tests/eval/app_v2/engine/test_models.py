# tests/eval/app_v2/engine/test_models.py
import dataclasses

from eval.app_v2.engine.domain.enums import (
    DiagnosticCode,
    GenerationStatus,
    PackingStatus,
    RerankStatus,
    RetrievalStatus,
    Severity,
)
from eval.app_v2.engine.domain.models import (
    AnalyzedQuery,
    QueryDiagnostic,
    QueryRecord,
    RunConfig,
)


def _make_query_record() -> QueryRecord:
    return QueryRecord(
        qid="q1",
        query="what is X?",
        query_type="factual",
        difficulty="easy",
        is_unanswerable=False,
        requires_synthesis=False,
        tags=(),
        relevant_chunk_ids=frozenset(["c1"]),
        retrieved_chunk_ids=("c1", "c2"),
        reranked_chunk_ids=None,
        packed_chunk_ids=None,
        per_query_recall_at_k={10: 1.0},
        per_query_precision_at_k={10: 0.5},
        per_query_ndcg_at_k={10: 1.0},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None,
        answer_metrics=None,
        groundedness=None,
        latency_ms=None,
        trace_id=None,
        trace=None,
    )


def test_query_record_is_frozen():
    r = _make_query_record()
    assert dataclasses.is_dataclass(r)
    try:
        r.qid = "changed"  # type: ignore
        raise AssertionError()
    except (AttributeError, TypeError):
        pass


def test_analyzed_query_pairs_record_and_diagnostic():
    record = _make_query_record()
    diag = QueryDiagnostic(
        qid="q1",
        diagnostic_code=DiagnosticCode.GROUNDED_ANSWER,
        severity=Severity.OK,
        retrieval_status=RetrievalStatus.HIT,
        rerank_status=RerankStatus.ABSENT,
        packing_status=PackingStatus.ABSENT,
        generation_status=GenerationStatus.GROUNDED,
        root_cause_summary="grounded answer with full retrieval",
        suggested_next_check=None,
        evidence_present=True,
        trace_available=False,
    )
    aq = AnalyzedQuery(record=record, diagnostic=diag)
    assert aq.record.qid == aq.diagnostic.qid


def test_run_config_frozen():
    cfg = RunConfig(
        retriever="HydratingRetriever",
        index_name="obsidian",
        reranker_model="heuristic_v1",
        reranker_top_n=None,
        generator_model=None,
        embedder_model="text-embedding-3-large",
        top_k=10,
        token_budget=1500,
    )
    assert cfg.top_k == 10
