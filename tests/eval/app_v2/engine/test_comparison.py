from eval.app_v2.engine.domain.enums import (
    ComparisonClassification,
    DeltaDirection,
    DiagnosticCode,
    GenerationStatus,
    PackingStatus,
    RerankStatus,
    RetrievalStatus,
    Severity,
)
from eval.app_v2.engine.domain.models import QueryDiagnostic
from eval.app_v2.engine.services.comparison import (
    classify_compared_query,
    compare_diagnostics,
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
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.RETRIEVAL_MISS, Severity.MODERATE),
        diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        recall_before=0.0,
        recall_after=1.0,
    )
    assert delta.retrieval == DeltaDirection.IMPROVED
    assert delta.severity == DeltaDirection.IMPROVED


def test_regressed_classification():
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        diag_after=_diag(DiagnosticCode.RETRIEVAL_MISS, Severity.MODERATE),
        recall_before=1.0,
        recall_after=0.0,
    )
    result = classify_compared_query(delta, diag_after=_diag(DiagnosticCode.RETRIEVAL_MISS, Severity.MODERATE))
    assert result == ComparisonClassification.REGRESSED


def test_unchanged_within_threshold():
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        recall_before=0.80,
        recall_after=0.82,  # < RETRIEVAL_DELTA_THRESHOLD
    )
    result = classify_compared_query(delta, diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK))
    assert result == ComparisonClassification.UNCHANGED


# ── Quality / hallucination / correctness / completeness ──────────────────────

def test_quality_score_improved():
    delta = compare_diagnostics(
        diag_before=None, diag_after=None,
        quality_score_before=0.50,
        quality_score_after=0.80,  # +0.30 > QUALITY_SCORE_DELTA_THRESHOLD
    )
    assert delta.quality == DeltaDirection.IMPROVED


def test_quality_score_regressed():
    delta = compare_diagnostics(
        diag_before=None, diag_after=None,
        quality_score_before=0.80,
        quality_score_after=0.50,
    )
    assert delta.quality == DeltaDirection.REGRESSED


def test_quality_score_unchanged_within_threshold():
    delta = compare_diagnostics(
        diag_before=None, diag_after=None,
        quality_score_before=0.70,
        quality_score_after=0.72,  # +0.02 < QUALITY_SCORE_DELTA_THRESHOLD
    )
    assert delta.quality == DeltaDirection.UNCHANGED


def test_quality_score_insufficient_when_missing():
    delta = compare_diagnostics(diag_before=None, diag_after=None)
    assert delta.quality == DeltaDirection.INSUFFICIENT


def test_hallucination_severity_improved_when_lower():
    # hallucination_severity is 0..5 where lower is better
    delta = compare_diagnostics(
        diag_before=None, diag_after=None,
        hallucination_severity_before=3.0,
        hallucination_severity_after=1.0,  # -2.0 > SCORE_DELTA_THRESHOLD → IMPROVED
    )
    assert delta.hallucination == DeltaDirection.IMPROVED


def test_hallucination_severity_regressed_when_higher():
    delta = compare_diagnostics(
        diag_before=None, diag_after=None,
        hallucination_severity_before=1.0,
        hallucination_severity_after=3.0,
    )
    assert delta.hallucination == DeltaDirection.REGRESSED


def test_hallucination_severity_unchanged_within_threshold():
    delta = compare_diagnostics(
        diag_before=None, diag_after=None,
        hallucination_severity_before=2.0,
        hallucination_severity_after=2.3,  # +0.3 < SCORE_DELTA_THRESHOLD
    )
    assert delta.hallucination == DeltaDirection.UNCHANGED


def test_correctness_improved():
    delta = compare_diagnostics(
        diag_before=None, diag_after=None,
        correctness_before=2.0,
        correctness_after=4.0,
    )
    assert delta.correctness == DeltaDirection.IMPROVED


def test_completeness_regressed():
    delta = compare_diagnostics(
        diag_before=None, diag_after=None,
        completeness_before=4.0,
        completeness_after=2.0,
    )
    assert delta.completeness == DeltaDirection.REGRESSED


def test_hallucination_regression_contributes_to_mixed_classification():
    """Retrieval unchanged + hallucination regressed → MIXED."""
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        recall_before=0.80,
        recall_after=0.80,
        hallucination_severity_before=0.0,
        hallucination_severity_after=3.0,
        quality_score_before=0.85,
        quality_score_after=0.50,
    )
    result = classify_compared_query(delta, diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK))
    assert result == ComparisonClassification.REGRESSED


def test_quality_improvement_alone_classifies_improved():
    """No retrieval data, only quality improved → IMPROVED."""
    delta = compare_diagnostics(
        diag_before=None, diag_after=None,
        quality_score_before=0.40,
        quality_score_after=0.75,
        correctness_before=2.0,
        correctness_after=4.5,
    )
    result = classify_compared_query(delta)
    assert result == ComparisonClassification.IMPROVED
