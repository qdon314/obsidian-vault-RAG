"""Port protocol for Stage 5a query validation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from benchmark.domain.models import (
    EvidenceSet,
    QueryCandidate,
    ValidatedQuery,
    ValidationResult,
)


@runtime_checkable
class QueryValidator(Protocol):
    """Validate query candidates and optionally refine evidence.

    ``validate`` checks a candidate and returns a ``ValidationResult``.
    ``refine_evidence`` narrows an ``EvidenceSet`` for a specific validated
    query — the M3 deterministic adapter returns evidence unchanged, but
    the M4 LLM adapter will implement per-query refinement.
    """

    def validate(self, candidate: QueryCandidate) -> ValidationResult: ...

    def refine_evidence(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> EvidenceSet: ...
