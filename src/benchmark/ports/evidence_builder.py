"""Port protocol for Stage 2 evidence set construction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from benchmark.domain.models import EvidenceSet, RegulatoryUnit


@runtime_checkable
class EvidenceBuilder(Protocol):
    """Build unit-relative tiered evidence for a regulatory unit."""

    def build(self, unit: RegulatoryUnit) -> EvidenceSet: ...
