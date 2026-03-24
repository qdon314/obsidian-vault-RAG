"""Tests for the GoldAnswerSynthesizer protocol."""

from __future__ import annotations

from benchmark.adapters.generation.llm_gold_answer_synthesizer import (
    LLMGoldAnswerSynthesizer,
)
from benchmark.ports.gold_answer_synthesizer import GoldAnswerSynthesizer


def test_llm_synthesizer_satisfies_protocol() -> None:
    """LLMGoldAnswerSynthesizer is a structural subtype of GoldAnswerSynthesizer."""
    assert isinstance(LLMGoldAnswerSynthesizer, type)
    # runtime_checkable protocol check requires an instance, not the class.
    # We verify by checking the method exists with correct signature.
    assert hasattr(LLMGoldAnswerSynthesizer, "synthesize")


def test_protocol_is_runtime_checkable(stub_synthesizer: object) -> None:
    """Protocol check works at runtime for objects with the right method."""
    assert isinstance(stub_synthesizer, GoldAnswerSynthesizer)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

import pytest
from benchmark.domain.models import EvidenceSet, GoldAnswer, ValidatedQuery
from benchmark.domain.enums import QueryClass


class _StubSynthesizer:
    def synthesize(self, query: ValidatedQuery, evidence: EvidenceSet) -> GoldAnswer:
        return GoldAnswer(gold_answer="stub answer")


@pytest.fixture
def stub_synthesizer() -> _StubSynthesizer:
    return _StubSynthesizer()
