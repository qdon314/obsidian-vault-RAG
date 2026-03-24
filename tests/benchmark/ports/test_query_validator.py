"""Tests for the QueryValidator port protocol."""

from __future__ import annotations

from benchmark.domain.enums import QueryClass
from benchmark.domain.models import (
    EvidenceSet,
    QueryCandidate,
    ValidatedQuery,
    ValidationResult,
)
from benchmark.ports.query_validator import QueryValidator


class _MockQueryValidator:
    """Mock satisfying the QueryValidator protocol."""

    def validate(self, candidate: QueryCandidate) -> ValidationResult:
        return ValidationResult(
            candidate_id=candidate.candidate_id,
            is_valid=True,
            flags=(),
        )

    def refine_evidence(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> EvidenceSet:
        return evidence


class TestQueryValidatorProtocol:
    def test_isinstance_check(self) -> None:
        val = _MockQueryValidator()
        assert isinstance(val, QueryValidator)

    def test_validate_returns_result(self) -> None:
        val = _MockQueryValidator()
        candidate = QueryCandidate(
            candidate_id="qc_1",
            unit_id="u1",
            query="What does 10 CFR 50.46(b)(1) require?",
            query_class=QueryClass.CITATION_LOOKUP,
            source_citations=("10 CFR 50.46(b)(1)",),
            evidence_span_ids=("u1_0",),
        )
        result = val.validate(candidate)
        assert result.is_valid is True
        assert result.candidate_id == "qc_1"

    def test_refine_evidence_returns_evidence_set(self) -> None:
        val = _MockQueryValidator()
        vq = ValidatedQuery(
            candidate_id="qc_1",
            unit_id="u1",
            query="Test?",
            query_class=QueryClass.CITATION_LOOKUP,
            source_citations=(),
            evidence_span_ids=(),
            difficulty="easy",
            corpus_snapshot_id="snap1",
        )
        evidence = EvidenceSet(unit_id="u1")
        result = val.refine_evidence(vq, evidence)
        assert result.unit_id == "u1"
