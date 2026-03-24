"""Port protocol for Stage 6 gold answer synthesis."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from benchmark.domain.models import EvidenceSet, GoldAnswer, ValidatedQuery


@runtime_checkable
class GoldAnswerSynthesizer(Protocol):
    """Synthesise a gold answer and answer-core rubric for a validated query.

    Accepts the query and its (refined) evidence set so the synthesiser can
    ground the gold answer in the correct critical and supporting spans.
    """

    def synthesize(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> GoldAnswer: ...
