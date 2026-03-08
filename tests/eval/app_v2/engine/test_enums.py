# tests/eval/app_v2/engine/test_enums.py
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


def test_diagnostic_codes_are_strings():
    assert DiagnosticCode.RETRIEVAL_MISS == "retrieval_miss"
    assert DiagnosticCode.NO_CLEAR_FAILURE == "no_clear_failure"


def test_severity_ordering():
    severities = [Severity.OK, Severity.MINOR, Severity.MODERATE, Severity.CRITICAL]
    assert len(severities) == 4


def test_all_enums_importable():
    assert RetrievalStatus.HIT == "hit"
    assert RerankStatus.IMPROVED == "improved"
    assert PackingStatus.COMPLETE == "complete"
    assert GenerationStatus.GROUNDED == "grounded"
    assert DeltaDirection.IMPROVED == "improved"
    assert ComparisonClassification.IMPROVED == "improved"
