"""Tests for LLMGoldAnswerSynthesizer adapter.

All tests use a stub LLMClient — no real API calls are made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from benchmark.adapters.generation.llm_gold_answer_synthesizer import (
    LLMGoldAnswerSynthesizer,
)
from benchmark.domain.enums import EvidenceTier, QueryClass
from benchmark.domain.models import (
    EvidenceEntry,
    EvidenceSet,
    GoldAnswer,
    StageConfig,
    ValidatedQuery,
)
from benchmark.ports.llm_client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubLLMClient:
    """Stub LLMClient returning a fixed response."""

    response_text: str
    raise_on_call: bool = False

    def complete(self, prompt: str, config: StageConfig) -> LLMResponse:
        if self.raise_on_call:
            raise RuntimeError("LLM call failed")
        return LLMResponse(text=self.response_text, model="stub-model")


def _make_client(data: dict[str, Any]) -> _StubLLMClient:
    return _StubLLMClient(response_text=json.dumps(data))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> StageConfig:
    return StageConfig(model="stub-model")


@pytest.fixture
def validated_query() -> ValidatedQuery:
    return ValidatedQuery(
        candidate_id="qc_50.46_b_1_citation_lookup_0",
        unit_id="50.46_b_1",
        query="What is the peak cladding temperature limit?",
        query_class=QueryClass.CITATION_LOOKUP,
        source_citations=("10 CFR 50.46(b)(1)",),
        evidence_span_ids=("span_1",),
        difficulty="easy",
        corpus_snapshot_id="abc123",
    )


@pytest.fixture
def evidence_set() -> EvidenceSet:
    entry = EvidenceEntry(
        span_id="span_1",
        citation="10 CFR 50.46(b)(1)",
        text="The peak cladding temperature shall not exceed 2200°F.",
        char_start=0,
        char_end=55,
        chunk_ids=("chunk_1",),
        tier=EvidenceTier.CRITICAL,
    )
    return EvidenceSet(unit_id="50.46_b_1", critical=(entry,))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMGoldAnswerSynthesizerSuccess:
    def test_returns_gold_answer_from_valid_json(
        self,
        config: StageConfig,
        validated_query: ValidatedQuery,
        evidence_set: EvidenceSet,
    ) -> None:
        client = _make_client(
            {
                "gold_answer": "The limit is 2200°F.",
                "acceptable_answer_variants": ["2200 degrees Fahrenheit"],
                "required_points": ["Contains 2200°F"],
                "forbidden_errors": ["Wrong threshold"],
            }
        )
        synthesizer = LLMGoldAnswerSynthesizer(client, config)
        result = synthesizer.synthesize(validated_query, evidence_set)

        assert result.gold_answer == "The limit is 2200°F."
        assert result.acceptable_answer_variants == ("2200 degrees Fahrenheit",)
        assert result.required_points == ("Contains 2200°F",)
        assert result.forbidden_errors == ("Wrong threshold",)

    def test_handles_empty_optional_lists(
        self,
        config: StageConfig,
        validated_query: ValidatedQuery,
        evidence_set: EvidenceSet,
    ) -> None:
        client = _make_client({"gold_answer": "2200°F."})
        synthesizer = LLMGoldAnswerSynthesizer(client, config)
        result = synthesizer.synthesize(validated_query, evidence_set)

        assert result.gold_answer == "2200°F."
        assert result.acceptable_answer_variants == ()
        assert result.required_points == ()
        assert result.forbidden_errors == ()

    def test_strips_markdown_fences(
        self,
        config: StageConfig,
        validated_query: ValidatedQuery,
        evidence_set: EvidenceSet,
    ) -> None:
        raw = "```json\n" + json.dumps({"gold_answer": "2200°F."}) + "\n```"
        client = _StubLLMClient(response_text=raw)
        synthesizer = LLMGoldAnswerSynthesizer(client, config)
        result = synthesizer.synthesize(validated_query, evidence_set)
        assert result.gold_answer == "2200°F."

    def test_prompt_includes_query_and_evidence(
        self,
        config: StageConfig,
        validated_query: ValidatedQuery,
        evidence_set: EvidenceSet,
    ) -> None:
        captured: list[str] = []

        class _CapturingClient:
            def complete(self, prompt: str, cfg: StageConfig) -> LLMResponse:
                captured.append(prompt)
                return LLMResponse(
                    text=json.dumps({"gold_answer": "2200°F."}),
                    model="stub",
                )

        synthesizer = LLMGoldAnswerSynthesizer(_CapturingClient(), config)
        synthesizer.synthesize(validated_query, evidence_set)

        assert len(captured) == 1
        prompt = captured[0]
        assert "peak cladding temperature" in prompt
        assert "10 CFR 50.46(b)(1)" in prompt
        assert "2200" in prompt  # evidence text


class TestLLMGoldAnswerSynthesizerFailures:
    def test_returns_empty_on_llm_error(
        self,
        config: StageConfig,
        validated_query: ValidatedQuery,
        evidence_set: EvidenceSet,
    ) -> None:
        client = _StubLLMClient(response_text="", raise_on_call=True)
        synthesizer = LLMGoldAnswerSynthesizer(client, config)
        result = synthesizer.synthesize(validated_query, evidence_set)
        assert result == GoldAnswer(gold_answer="")

    def test_returns_empty_on_invalid_json(
        self,
        config: StageConfig,
        validated_query: ValidatedQuery,
        evidence_set: EvidenceSet,
    ) -> None:
        client = _StubLLMClient(response_text="not json at all")
        synthesizer = LLMGoldAnswerSynthesizer(client, config)
        result = synthesizer.synthesize(validated_query, evidence_set)
        assert result == GoldAnswer(gold_answer="")

    def test_returns_empty_when_gold_answer_key_missing(
        self,
        config: StageConfig,
        validated_query: ValidatedQuery,
        evidence_set: EvidenceSet,
    ) -> None:
        client = _make_client({"required_points": ["something"]})
        synthesizer = LLMGoldAnswerSynthesizer(client, config)
        result = synthesizer.synthesize(validated_query, evidence_set)
        assert result == GoldAnswer(gold_answer="")

    def test_returns_empty_when_gold_answer_is_empty_string(
        self,
        config: StageConfig,
        validated_query: ValidatedQuery,
        evidence_set: EvidenceSet,
    ) -> None:
        client = _make_client({"gold_answer": "   "})
        synthesizer = LLMGoldAnswerSynthesizer(client, config)
        result = synthesizer.synthesize(validated_query, evidence_set)
        assert result == GoldAnswer(gold_answer="")

    def test_empty_evidence_set_still_returns_answer(
        self,
        config: StageConfig,
        validated_query: ValidatedQuery,
    ) -> None:
        """Empty evidence is forwarded as '(none)' — adapter doesn't raise."""
        client = _make_client({"gold_answer": "2200°F."})
        empty_evidence = EvidenceSet(unit_id="50.46_b_1")
        synthesizer = LLMGoldAnswerSynthesizer(client, config)
        result = synthesizer.synthesize(validated_query, empty_evidence)
        assert result.gold_answer == "2200°F."
