"""Port protocol for Stage 3 query generation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from benchmark.domain.enums import QueryClass
from benchmark.domain.models import EvidenceSet, QueryCandidate, RegulatoryUnit


@runtime_checkable
class QueryGenerator(Protocol):
    """Generate query candidates from a regulatory unit.

    The ``evidence`` parameter is included alongside the unit because
    generators need to reference critical spans to construct answerable
    questions.  This is an intentional deviation from the design doc's
    ``generate(unit, query_class)`` signature.
    """

    def generate(
        self,
        unit: RegulatoryUnit,
        evidence: EvidenceSet,
        query_class: QueryClass,
    ) -> list[QueryCandidate]: ...
