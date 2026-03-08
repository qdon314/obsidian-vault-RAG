# eval/app_v2/engine/derived/diagnostics.py
from __future__ import annotations

from collections.abc import Sequence

from eval.app_v2.engine.derived.stage_attribution import classify_query, derive_stage_statuses
from eval.app_v2.engine.domain.models import AnalyzedQuery, QueryDiagnostic, QueryRecord


def _prose(code) -> tuple[str, str | None]:
    """Return (root_cause_summary, suggested_next_check) for a DiagnosticCode."""
    from eval.app_v2.engine.domain.enums import DiagnosticCode
    mapping = {
        DiagnosticCode.RETRIEVAL_MISS:               ("No relevant chunks retrieved", "Check embedder / index coverage"),
        DiagnosticCode.RETRIEVAL_PARTIAL:             ("Some relevant chunks missed at retrieval", "Increase top_k or check embedding quality"),
        DiagnosticCode.RERANK_DROPPED_RELEVANT:       ("Reranker dropped relevant chunks", "Inspect reranker scores for this query"),
        DiagnosticCode.RERANK_DEGRADED_RANK:          ("Reranker degraded rank of relevant chunks", "Review reranker model or heuristic weights"),
        DiagnosticCode.PACKING_OMITTED_RELEVANT:      ("Packing omitted relevant chunks within token budget", "Increase token budget or check packing order"),
        DiagnosticCode.PACKING_TRUNCATED_RELEVANT:    ("Token budget forced truncation of relevant content", "Increase token budget"),
        DiagnosticCode.UNSUPPORTED_ANSWER:            ("Generated answer not grounded in retrieved context", "Inspect citations and groundedness judge"),
        DiagnosticCode.GROUNDED_ANSWER:               ("Answer is grounded and retrieval succeeded", None),
        DiagnosticCode.BAD_ABSTAIN_ON_ANSWERABLE:     ("Model abstained despite evidence present", "Review generator prompt / abstain threshold"),
        DiagnosticCode.FAILED_ABSTAIN_ON_UNANSWERABLE:("Model answered an unanswerable question", "Review abstain instructions in prompt"),
        DiagnosticCode.TRACE_MISSING:                 ("Trace unavailable for this query", "Re-run with tracing enabled"),
        DiagnosticCode.DATA_INSUFFICIENT:             ("No relevant chunks defined; cannot diagnose", "Check query dataset annotations"),
        DiagnosticCode.NO_CLEAR_FAILURE:              ("No clear failure mode detected", None),
    }
    summary, suggestion = mapping.get(code, ("Unknown diagnostic", None))
    return summary, suggestion


def build_query_diagnostic(record: QueryRecord) -> QueryDiagnostic:
    code, severity = classify_query(record)
    ret_status, rrk_status, pck_status, gen_status = derive_stage_statuses(record, code)
    summary, suggestion = _prose(code)
    return QueryDiagnostic(
        qid=record.qid,
        diagnostic_code=code,
        severity=severity,
        retrieval_status=ret_status,
        rerank_status=rrk_status,
        packing_status=pck_status,
        generation_status=gen_status,
        root_cause_summary=summary,
        suggested_next_check=suggestion,
        evidence_present=bool(record.relevant_chunk_ids & frozenset(record.retrieved_chunk_ids)),
        trace_available=record.trace is not None,
    )


def analyze_queries(records: Sequence[QueryRecord]) -> tuple[AnalyzedQuery, ...]:
    return tuple(
        AnalyzedQuery(record=r, diagnostic=build_query_diagnostic(r))
        for r in records
    )
