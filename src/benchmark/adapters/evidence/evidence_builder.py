"""Default (deterministic) evidence builder.

Assigns all unit spans to the critical tier without LLM involvement.
Used as the baseline/fallback when no LLM client is wired.
"""

from __future__ import annotations

from benchmark.domain.enums import EvidenceTier
from benchmark.domain.models import EvidenceEntry, EvidenceSet, RegulatoryUnit


class DefaultEvidenceBuilder:
    """Deterministic evidence builder: all unit spans are marked critical."""

    def build(self, unit: RegulatoryUnit) -> EvidenceSet:
        entries = tuple(
            EvidenceEntry(
                span_id=f"{unit.unit_id}_{i}",
                citation=span.citation,
                text=span.text,
                char_start=span.char_start,
                char_end=span.char_end,
                chunk_ids=span.chunk_ids_overlapping_span,
                tier=EvidenceTier.CRITICAL,
            )
            for i, span in enumerate(unit.spans)
        )
        return EvidenceSet(unit_id=unit.unit_id, critical=entries)
