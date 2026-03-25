"""Tests for the contamination probe.

All LLM calls are stubbed — no real API calls.
"""

from __future__ import annotations

import json

import pytest

from benchmark.domain.enums import EvidenceTier, QueryClass
from benchmark.domain.models import (
    BenchmarkRecord,
    EvidenceEntry,
    StageConfig,
)
from benchmark.ports.llm_client import LLMResponse
from benchmark.stages.contamination_probe import run_contamination_probe

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

MODEL_ID = "gpt-4o-2025-01-01"


class _StubLLMClient:
    """Returns canned responses: first call = generation, second = judge."""

    def __init__(
        self, gen_text: str, judge_data: dict | None, *, raise_on: int | None = None
    ) -> None:
        self._gen_text = gen_text
        self._judge_data = judge_data
        self._raise_on = raise_on
        self._call_count = 0

    def complete(self, prompt: str, config: StageConfig) -> LLMResponse:
        self._call_count += 1
        if self._raise_on == self._call_count:
            raise RuntimeError("stubbed failure")
        if self._call_count == 1:
            return LLMResponse(text=self._gen_text, model="stub")
        return LLMResponse(
            text=json.dumps(self._judge_data) if self._judge_data is not None else "bad json",
            model="stub",
        )


def _make_record(
    *,
    qid: str = "q1",
    gold_answer: str | None = "The limit is 2200°F.",
    contamination_probes: dict | None = None,
) -> BenchmarkRecord:
    entry = EvidenceEntry(
        span_id="s1",
        citation="10 CFR 50.46(b)(1)",
        text="text",
        char_start=0,
        char_end=4,
        chunk_ids=("c1",),
        tier=EvidenceTier.CRITICAL,
    )
    return BenchmarkRecord(
        qid=qid,
        query="What is the peak cladding temperature limit?",
        query_class=QueryClass.CITATION_LOOKUP,
        difficulty="easy",
        source_unit_ids=("50.46_b_1",),
        source_citations=("10 CFR 50.46(b)(1)",),
        critical_evidence=(entry,),
        supporting_evidence=(),
        contextual_evidence=(),
        hard_negative_chunk_ids=(),
        hard_negatives_retriever_config={},
        corpus_snapshot_id="abc",
        valid_as_of="2026-03-24",
        gold_answer=gold_answer,
        contamination_probes=contamination_probes or {},
    )


@pytest.fixture
def config() -> StageConfig:
    return StageConfig(model="stub-model")


# ---------------------------------------------------------------------------
# Tests: core contamination logic
# ---------------------------------------------------------------------------


class TestRunContaminationProbe:
    def test_high_score_flags_as_contaminated(self, config: StageConfig) -> None:
        """correctness=4.5 → score_0_1=0.9 → contaminated=True."""
        client = _StubLLMClient(
            gen_text="The temperature limit is 2200°F.",
            judge_data={"correctness": 4.5, "completeness": 4.0, "relevance": 4.0},
        )
        records = [_make_record()]
        result = run_contamination_probe(records, client, config, MODEL_ID)

        assert len(result) == 1
        assert result[0].contamination_probes[MODEL_ID] is True

    def test_low_score_not_contaminated(self, config: StageConfig) -> None:
        """correctness=2.0 → score_0_1=0.4 → contaminated=False."""
        client = _StubLLMClient(
            gen_text="I don't know.",
            judge_data={"correctness": 2.0},
        )
        records = [_make_record()]
        result = run_contamination_probe(records, client, config, MODEL_ID)

        assert result[0].contamination_probes[MODEL_ID] is False

    def test_threshold_boundary_at_exactly_0_7(self, config: StageConfig) -> None:
        """correctness=3.5 → score_0_1=0.70 → contaminated=True (>= threshold)."""
        client = _StubLLMClient(
            gen_text="Answer.",
            judge_data={"correctness": 3.5},
        )
        records = [_make_record()]
        result = run_contamination_probe(
            records, client, config, MODEL_ID, contamination_threshold=0.7
        )
        assert result[0].contamination_probes[MODEL_ID] is True

    def test_custom_threshold(self, config: StageConfig) -> None:
        """score_0_1=0.9 is not contaminated when threshold=0.95."""
        client = _StubLLMClient(
            gen_text="Answer.",
            judge_data={"correctness": 4.5},
        )
        records = [_make_record()]
        result = run_contamination_probe(
            records, client, config, MODEL_ID, contamination_threshold=0.95
        )
        assert result[0].contamination_probes[MODEL_ID] is False

    def test_multiple_records_processed(self, config: StageConfig) -> None:
        records = [_make_record(qid="q1"), _make_record(qid="q2")]

        # Need a fresh client per call since it uses call count.
        class _MultiClient:
            def complete(self, prompt: str, cfg: StageConfig) -> LLMResponse:
                return LLMResponse(text=json.dumps({"correctness": 1.0}), model="s")

        result = run_contamination_probe(records, _MultiClient(), config, MODEL_ID)
        assert len(result) == 2
        assert MODEL_ID in result[0].contamination_probes
        assert MODEL_ID in result[1].contamination_probes


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestContaminationEdgeCases:
    def test_skips_record_without_gold_answer(self, config: StageConfig) -> None:
        client = _StubLLMClient(gen_text="", judge_data=None)
        records = [_make_record(gold_answer=None)]
        result = run_contamination_probe(records, client, config, MODEL_ID)

        assert result[0].contamination_probes == {}
        assert client._call_count == 0  # no LLM calls made

    def test_idempotent_skips_already_probed(self, config: StageConfig) -> None:
        existing = {MODEL_ID: False}
        client = _StubLLMClient(gen_text="", judge_data=None)
        records = [_make_record(contamination_probes=existing)]
        result = run_contamination_probe(records, client, config, MODEL_ID)

        assert result[0].contamination_probes[MODEL_ID] is False
        assert client._call_count == 0

    def test_new_model_probe_preserves_existing(self, config: StageConfig) -> None:
        """Adding probe for model-b keeps existing model-a probe."""
        existing = {"model-a": True}
        client = _StubLLMClient(gen_text="answer", judge_data={"correctness": 1.0})
        records = [_make_record(contamination_probes=existing)]
        result = run_contamination_probe(records, client, config, "model-b")

        assert result[0].contamination_probes["model-a"] is True
        assert "model-b" in result[0].contamination_probes

    def test_generation_failure_treated_as_not_contaminated(self, config: StageConfig) -> None:
        client = _StubLLMClient(gen_text="", judge_data=None, raise_on=1)
        records = [_make_record()]
        result = run_contamination_probe(records, client, config, MODEL_ID)
        assert result[0].contamination_probes[MODEL_ID] is False

    def test_judge_failure_treated_as_not_contaminated(self, config: StageConfig) -> None:
        client = _StubLLMClient(gen_text="answer", judge_data=None, raise_on=2)
        records = [_make_record()]
        result = run_contamination_probe(records, client, config, MODEL_ID)
        assert result[0].contamination_probes[MODEL_ID] is False

    def test_invalid_judge_json_treated_as_not_contaminated(self, config: StageConfig) -> None:
        class _BadJsonClient:
            def complete(self, prompt: str, cfg: StageConfig) -> LLMResponse:
                return LLMResponse(text="this is not json", model="s")

        records = [_make_record()]
        result = run_contamination_probe(records, _BadJsonClient(), config, MODEL_ID)
        assert result[0].contamination_probes[MODEL_ID] is False

    def test_none_correctness_treated_as_zero(self, config: StageConfig) -> None:
        """GoldJudgeResult with correctness=None → score=0.0 → not contaminated."""
        client = _StubLLMClient(
            gen_text="answer",
            judge_data={"completeness": 4.0},  # no correctness key
        )
        records = [_make_record()]
        result = run_contamination_probe(records, client, config, MODEL_ID)
        assert result[0].contamination_probes[MODEL_ID] is False

    def test_original_fields_preserved(self, config: StageConfig) -> None:
        """All original BenchmarkRecord fields survive the replace()."""
        client = _StubLLMClient(gen_text="answer", judge_data={"correctness": 1.0})
        original = _make_record(qid="original_qid")
        result = run_contamination_probe([original], client, config, MODEL_ID)

        assert result[0].qid == "original_qid"
        assert result[0].query == original.query
        assert result[0].critical_evidence == original.critical_evidence
