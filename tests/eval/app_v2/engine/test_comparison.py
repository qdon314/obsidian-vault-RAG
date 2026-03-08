from eval.app_v2.engine.domain.enums import (
    ComparisonClassification, DeltaDirection, DiagnosticCode, Severity,
    RetrievalStatus, RerankStatus, PackingStatus, GenerationStatus,
)
from eval.app_v2.engine.domain.models import QueryDiagnostic
from eval.app_v2.engine.services.comparison import (
    compare_diagnostics,
    classify_compared_query,
    RETRIEVAL_DELTA_THRESHOLD,
)


def _diag(code, severity, retrieval=RetrievalStatus.HIT):
    return QueryDiagnostic(
        qid="q1",
        diagnostic_code=code,
        severity=severity,
        retrieval_status=retrieval,
        rerank_status=RerankStatus.ABSENT,
        packing_status=PackingStatus.ABSENT,
        generation_status=GenerationStatus.GROUNDED,
        root_cause_summary="",
        suggested_next_check=None,
        evidence_present=True,
        trace_available=False,
    )


def test_improved_retrieval_delta():
    from eval.app_v2.engine.domain.models import QueryDeltaSummary
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.RETRIEVAL_MISS, Severity.MODERATE),
        diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        recall_before=0.0,
        recall_after=1.0,
    )
    assert delta.retrieval == DeltaDirection.IMPROVED
    assert delta.severity == DeltaDirection.IMPROVED


def test_regressed_classification():
    from eval.app_v2.engine.domain.models import QueryDeltaSummary
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        diag_after=_diag(DiagnosticCode.RETRIEVAL_MISS, Severity.MODERATE),
        recall_before=1.0,
        recall_after=0.0,
    )
    result = classify_compared_query(delta, diag_after=_diag(DiagnosticCode.RETRIEVAL_MISS, Severity.MODERATE))
    assert result == ComparisonClassification.REGRESSED


def test_unchanged_within_threshold():
    from eval.app_v2.engine.domain.models import QueryDeltaSummary
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        recall_before=0.80,
        recall_after=0.82,  # < RETRIEVAL_DELTA_THRESHOLD
    )
    result = classify_compared_query(delta, diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK))
    assert result == ComparisonClassification.UNCHANGED
