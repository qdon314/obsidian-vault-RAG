# eval/app_v2/engine/derived/stage_attribution.py
"""
Classify a QueryRecord into a (DiagnosticCode, Severity) pair.

Decision order (see design doc):
1. Data sufficiency
2. Unanswerable behavior
3. Retrieval
4. Rerank
5. Packing
6. Generation
7. Fallback
"""
from __future__ import annotations

from eval.app_v2.engine.domain.enums import (
    DiagnosticCode,
    GenerationStatus,
    PackingStatus,
    RerankStatus,
    RetrievalStatus,
    Severity,
)
from eval.app_v2.engine.domain.models import QueryRecord

# Severity per code — single source of truth
_SEVERITY: dict[DiagnosticCode, Severity] = {
    DiagnosticCode.DATA_INSUFFICIENT:              Severity.MODERATE,
    DiagnosticCode.TRACE_MISSING:                  Severity.MINOR,
    DiagnosticCode.FAILED_ABSTAIN_ON_UNANSWERABLE: Severity.CRITICAL,
    DiagnosticCode.BAD_ABSTAIN_ON_ANSWERABLE:      Severity.MODERATE,
    DiagnosticCode.RETRIEVAL_MISS:                 Severity.MODERATE,
    DiagnosticCode.RETRIEVAL_PARTIAL:              Severity.MINOR,
    DiagnosticCode.RERANK_DROPPED_RELEVANT:        Severity.MODERATE,
    DiagnosticCode.RERANK_DEGRADED_RANK:           Severity.MINOR,
    DiagnosticCode.PACKING_OMITTED_RELEVANT:       Severity.MODERATE,
    DiagnosticCode.PACKING_TRUNCATED_RELEVANT:     Severity.MODERATE,
    DiagnosticCode.UNSUPPORTED_ANSWER:             Severity.CRITICAL,
    DiagnosticCode.GROUNDED_ANSWER:                Severity.OK,
    DiagnosticCode.NO_CLEAR_FAILURE:               Severity.OK,
}


def classify_query(record: QueryRecord) -> tuple[DiagnosticCode, Severity]:
    """Return (DiagnosticCode, Severity) for a single QueryRecord."""
    relevant = record.relevant_chunk_ids
    retrieved = frozenset(record.retrieved_chunk_ids)

    # 1. Data sufficiency
    if not relevant:
        code = DiagnosticCode.DATA_INSUFFICIENT
        return code, _SEVERITY[code]

    hits = relevant & retrieved

    # 2. Unanswerable behavior (requires generation data)
    if record.is_unanswerable and record.answer_text is not None:
        # answered when it should have abstained
        code = DiagnosticCode.FAILED_ABSTAIN_ON_UNANSWERABLE
        return code, _SEVERITY[code]

    # 3. Retrieval
    if not hits:
        code = DiagnosticCode.RETRIEVAL_MISS
        return code, _SEVERITY[code]

    if len(hits) < len(relevant):
        code = DiagnosticCode.RETRIEVAL_PARTIAL
        return code, _SEVERITY[code]

    # 4. Rerank — check if relevant chunk was dropped
    if record.reranked_chunk_ids is not None:
        reranked_set = frozenset(record.reranked_chunk_ids)
        if hits and not (hits & reranked_set):
            code = DiagnosticCode.RERANK_DROPPED_RELEVANT
            return code, _SEVERITY[code]

    # 5. Packing — check if relevant survived rerank but lost in packing
    if record.packed_chunk_ids is not None:
        packed_set = frozenset(record.packed_chunk_ids)
        reranked_set = frozenset(record.reranked_chunk_ids) if record.reranked_chunk_ids else retrieved
        survived_rerank = hits & reranked_set
        if survived_rerank and not (survived_rerank & packed_set):
            code = DiagnosticCode.PACKING_OMITTED_RELEVANT
            return code, _SEVERITY[code]

    # 6. Generation — if groundedness says unsupported
    if record.groundedness is not None:
        gnd = record.groundedness
        # GroundednessJudgeResult.unsupported_claims is a list
        if hasattr(gnd, "unsupported_claims") and gnd.unsupported_claims:
            code = DiagnosticCode.UNSUPPORTED_ANSWER
            return code, _SEVERITY[code]

    # 6b. Abstain check on answerable query
    if not record.is_unanswerable and record.answer_text is None and record.trace is not None:
        code = DiagnosticCode.BAD_ABSTAIN_ON_ANSWERABLE
        return code, _SEVERITY[code]

    # 7. Fallback — full retrieval, no failure detected
    if hits == relevant:
        code = DiagnosticCode.GROUNDED_ANSWER
        return code, _SEVERITY[code]

    code = DiagnosticCode.NO_CLEAR_FAILURE
    return code, _SEVERITY[code]


def derive_stage_statuses(
    record: QueryRecord, code: DiagnosticCode
) -> tuple[RetrievalStatus, RerankStatus, PackingStatus, GenerationStatus]:
    """Map DiagnosticCode back to per-stage status enums."""
    relevant = record.relevant_chunk_ids
    retrieved = frozenset(record.retrieved_chunk_ids)
    hits = relevant & retrieved

    if not relevant:
        return RetrievalStatus.UNKNOWN, RerankStatus.UNKNOWN, PackingStatus.UNKNOWN, GenerationStatus.UNKNOWN

    if not hits:
        ret = RetrievalStatus.MISS
    elif len(hits) < len(relevant):
        ret = RetrievalStatus.PARTIAL
    else:
        ret = RetrievalStatus.HIT

    if record.reranked_chunk_ids is None:
        rrk = RerankStatus.ABSENT
    elif hits and not (hits & frozenset(record.reranked_chunk_ids)):
        rrk = RerankStatus.DEGRADED
    else:
        rrk = RerankStatus.NEUTRAL

    if record.packed_chunk_ids is None:
        pck = PackingStatus.ABSENT
    elif hits and not (hits & frozenset(record.packed_chunk_ids)):
        pck = PackingStatus.OMITTED
    else:
        pck = PackingStatus.COMPLETE

    if record.answer_text is None:
        gen = GenerationStatus.ABSENT
    elif code == DiagnosticCode.UNSUPPORTED_ANSWER:
        gen = GenerationStatus.UNSUPPORTED
    elif code == DiagnosticCode.FAILED_ABSTAIN_ON_UNANSWERABLE:
        gen = GenerationStatus.FAILED_TO_ABSTAIN
    else:
        gen = GenerationStatus.GROUNDED

    return ret, rrk, pck, gen
