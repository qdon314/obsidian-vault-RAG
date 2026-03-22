"""Port protocol for Stage 1 regulatory unit extraction."""

from __future__ import annotations

from typing import Protocol

from benchmark.domain.models import BenchmarkSourceSpan, RegulatoryUnit


class UnitExtractor(Protocol):
    """Extract regulatory units from benchmark source spans."""

    def extract(self, spans: list[BenchmarkSourceSpan]) -> list[RegulatoryUnit]: ...
