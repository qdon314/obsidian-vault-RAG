"""M5 runner tests: Stage 6 (gold synthesis) and Stage 5c (contamination probe)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.domain.enums import EvidenceTier, QueryClass, UnitKind
from benchmark.domain.models import (
    BenchmarkRecord,
    BenchmarkSourceSpan,
    EvidenceEntry,
    EvidenceSet,
    GoldAnswer,
    QueryCandidate,
    RegulatoryUnit,
    StageConfig,
    ValidatedQuery,
    ValidationResult,
)
from benchmark.pipeline.runner import PipelineConfig, PipelineResult, PipelineRunner
from benchmark.ports.llm_client import LLMResponse

# ── Helpers ──────────────────────────────────────────────────────────


def _make_span(snapshot_id: str = "snap1") -> BenchmarkSourceSpan:
    text = "Peak cladding temperature limit."
    return BenchmarkSourceSpan(
        source_doc_id="doc_1",
        citation="10 CFR 50.46(b)(1)",
        citation_key="10_cfr_50.46_b_1",
        section_title="ECCS criteria",
        text=text,
        char_start=0,
        char_end=len(text),
        chunk_ids_overlapping_span=("c1",),
        parent_section_id="50.46",
        effective_date="2026-01-01",
        corpus_snapshot_id=snapshot_id,
    )


def _make_unit(snapshot_id: str = "snap1") -> RegulatoryUnit:
    return RegulatoryUnit(
        unit_id="50.46_b_1",
        kind=UnitKind.OBLIGATION,
        spans=(_make_span(snapshot_id=snapshot_id),),
        citation="10 CFR 50.46(b)(1)",
        subsection_chain=("b", "1"),
        parent_section_id="50.46",
        corpus_snapshot_id=snapshot_id,
        canonical_statement="peak cladding temp",
    )


def _make_evidence() -> EvidenceSet:
    return EvidenceSet(
        unit_id="50.46_b_1",
        critical=(
            EvidenceEntry(
                span_id="50.46_b_1_0",
                citation="10 CFR 50.46(b)(1)",
                text="Peak cladding temperature limit.",
                char_start=0,
                char_end=31,
                chunk_ids=("c1",),
                tier=EvidenceTier.CRITICAL,
            ),
        ),
    )


def _make_candidate() -> QueryCandidate:
    return QueryCandidate(
        candidate_id="qc_50.46_b_1_citation_lookup_0",
        unit_id="50.46_b_1",
        query="What does 10 CFR 50.46(b)(1) require?",
        query_class=QueryClass.CITATION_LOOKUP,
        source_citations=("10 CFR 50.46(b)(1)",),
        evidence_span_ids=("50.46_b_1_0",),
        corpus_snapshot_id="snap1",
    )


# ── Mock adapters ────────────────────────────────────────────────────


class _MockUnitExtractor:
    def extract(self, spans: list[BenchmarkSourceSpan]) -> list[RegulatoryUnit]:
        return [_make_unit(snapshot_id=spans[0].corpus_snapshot_id)]


class _MockEvidenceBuilder:
    def build(self, unit: RegulatoryUnit) -> EvidenceSet:
        return _make_evidence()


class _MockQueryGenerator:
    def generate(
        self,
        unit: RegulatoryUnit,
        evidence: EvidenceSet,
        query_class: QueryClass,
    ) -> list[QueryCandidate]:
        return [_make_candidate()]


class _MockQueryValidator:
    def validate(self, candidate: QueryCandidate) -> ValidationResult:
        return ValidationResult(
            candidate_id=candidate.candidate_id,
            is_valid=True,
            flags=(),
        )

    def refine_evidence(self, query: object, evidence: EvidenceSet) -> EvidenceSet:
        return evidence


def _mock_classifier(units: list[RegulatoryUnit]) -> list[RegulatoryUnit]:
    return units


class _StubGoldAnswerSynthesizer:
    """Returns a fixed GoldAnswer for every query."""

    def __init__(self, *, gold_answer: str = "The limit is 2200°F.") -> None:
        self._gold_answer = gold_answer

    def synthesize(self, query: ValidatedQuery, evidence: EvidenceSet) -> GoldAnswer:
        return GoldAnswer(
            gold_answer=self._gold_answer,
            required_points=("2200°F",),
        )


class _StubEmptyGoldAnswerSynthesizer:
    """Simulates a failed synthesis — returns empty gold_answer."""

    def synthesize(self, query: ValidatedQuery, evidence: EvidenceSet) -> GoldAnswer:
        return GoldAnswer(gold_answer="")


class _StubLLMClient:
    """Stub LLMClient that returns configurable canned responses."""

    def __init__(
        self,
        generation_response: str = "It is limited to 2200°F.",
        judge_response: str = '{"correctness": 4, "completeness": 4, "reasoning": "good"}',
    ) -> None:
        self._generation_response = generation_response
        self._judge_response = judge_response
        self._call_count = 0

    def complete(self, prompt: str, config: StageConfig) -> LLMResponse:
        self._call_count += 1
        # Alternate: odd calls are generation, even are judge.
        text = self._generation_response if self._call_count % 2 == 1 else self._judge_response
        return LLMResponse(text=text, model=config.model or "stub")


class _StubContaminatedLLMClient:
    """Returns a high-score judge response — marks queries contaminated."""

    def __init__(self) -> None:
        self._call_count = 0

    def complete(self, prompt: str, config: StageConfig) -> LLMResponse:
        self._call_count += 1
        # First call: generation; second: judge with score >= 0.7.
        if self._call_count % 2 == 1:
            text = "The answer is 2200°F as stated in 10 CFR 50.46."
        else:
            text = '{"correctness": 5, "completeness": 5, "reasoning": "perfect"}'
        return LLMResponse(text=text, model=config.model or "stub")


# ── Builder ──────────────────────────────────────────────────────────


def _build_runner(
    tmp_path: Path,
    *,
    gold_synthesizer: object | None = None,
    llm_client: object | None = None,
    contamination_model_id: str | None = None,
    resume_from: str | None = None,
) -> PipelineRunner:
    config = PipelineConfig(
        run_id="m5_run",
        output_dir=str(tmp_path),
        resume_from=resume_from,
    )
    return PipelineRunner(
        config,
        corpus_spans_builder=lambda: [_make_span()],
        unit_extractor=_MockUnitExtractor(),
        llm_classifier=_mock_classifier,
        evidence_builder=_MockEvidenceBuilder(),
        query_generator=_MockQueryGenerator(),
        query_validator=_MockQueryValidator(),
        gold_answer_synthesizer=gold_synthesizer,  # type: ignore[arg-type]
        llm_client=llm_client,  # type: ignore[arg-type]
        contamination_model_id=contamination_model_id,
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestM5BackwardCompat:
    """M5 constructor additions must not break M4-only runners."""

    def test_no_m5_args_produces_m4_stage_order(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        result = runner.run()

        # Stage 6 and 5c should be absent — skipped via continue.
        assert "stage_6" not in result.stages_completed
        assert "stage_5c" not in result.stages_completed

    def test_m5_defaults_do_not_break_export(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        result = runner.run()

        assert result.total_exported == 1

    def test_total_contaminated_zero_without_probe(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        result = runner.run()

        assert result.total_contaminated == 0


class TestStage6GoldSynthesis:
    def test_stage_6_in_stages_completed_when_synthesizer_provided(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, gold_synthesizer=_StubGoldAnswerSynthesizer())
        result = runner.run()

        assert "stage_6" in result.stages_completed

    def test_stage_6_checkpoint_written(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, gold_synthesizer=_StubGoldAnswerSynthesizer())
        runner.run()

        assert (tmp_path / "m5_run" / "stage_6_gold_answers.jsonl").exists()

    def test_gold_answer_populated_in_checkpoint(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, gold_synthesizer=_StubGoldAnswerSynthesizer())
        runner.run()

        path = tmp_path / "m5_run" / "stage_6_gold_answers.jsonl"
        record = json.loads(path.read_text().strip().splitlines()[0])
        assert record["gold_answer"] == "The limit is 2200°F."
        assert "2200°F" in record["required_points"]

    def test_empty_gold_answer_record_retained_without_gold(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, gold_synthesizer=_StubEmptyGoldAnswerSynthesizer())
        runner.run()

        path = tmp_path / "m5_run" / "stage_6_gold_answers.jsonl"
        record = json.loads(path.read_text().strip().splitlines()[0])
        # Record is still present but gold_answer field is null/absent.
        assert record.get("gold_answer") is None or record.get("gold_answer") == ""

    def test_benchmark_records_used_for_export_when_stage6_ran(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, gold_synthesizer=_StubGoldAnswerSynthesizer())
        result = runner.run()

        assert result.total_exported == 1
        # benchmark_records.jsonl should contain gold_answer data.
        path = tmp_path / "m5_run" / "benchmark_records.jsonl"
        record = json.loads(path.read_text().strip().splitlines()[0])
        assert record["gold_answer"] == "The limit is 2200°F."

    def test_stage_6_skipped_without_synthesizer(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        result = runner.run()

        assert "stage_6" not in result.stages_completed
        assert not (tmp_path / "m5_run" / "stage_6_gold_answers.jsonl").exists()


class TestStage5cContaminationProbe:
    def test_stage_5c_in_stages_completed_when_enabled(self, tmp_path: Path) -> None:
        runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubLLMClient(),
            contamination_model_id="gpt-4o",
        )
        result = runner.run()

        assert "stage_5c" in result.stages_completed

    def test_stage_5c_checkpoint_written(self, tmp_path: Path) -> None:
        runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubLLMClient(),
            contamination_model_id="gpt-4o",
        )
        runner.run()

        assert (tmp_path / "m5_run" / "stage_5c_probed_records.jsonl").exists()

    def test_non_contaminated_record_flagged_false(self, tmp_path: Path) -> None:
        # Stub returns correctness=4 → score 0.8 → contaminated=True.
        # Use a low-score stub instead.
        low_score_client = _StubLLMClient(
            judge_response='{"correctness": 2, "completeness": 2, "reasoning": "poor"}'
        )
        runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=low_score_client,
            contamination_model_id="gpt-4o",
        )
        runner.run()

        path = tmp_path / "m5_run" / "stage_5c_probed_records.jsonl"
        record = json.loads(path.read_text().strip().splitlines()[0])
        assert record["contamination_probes"]["gpt-4o"] is False

    def test_contaminated_record_flagged_true(self, tmp_path: Path) -> None:
        runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubContaminatedLLMClient(),
            contamination_model_id="gpt-4o",
        )
        runner.run()

        path = tmp_path / "m5_run" / "stage_5c_probed_records.jsonl"
        record = json.loads(path.read_text().strip().splitlines()[0])
        assert record["contamination_probes"]["gpt-4o"] is True

    def test_total_contaminated_count_in_result(self, tmp_path: Path) -> None:
        runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubContaminatedLLMClient(),
            contamination_model_id="gpt-4o",
        )
        result = runner.run()

        assert result.total_contaminated == 1

    def test_stage_5c_skipped_without_model_id(self, tmp_path: Path) -> None:
        runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubLLMClient(),
            contamination_model_id=None,
        )
        result = runner.run()

        assert "stage_5c" not in result.stages_completed

    def test_stage_5c_skipped_without_llm_client(self, tmp_path: Path) -> None:
        runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=None,
            contamination_model_id="gpt-4o",
        )
        result = runner.run()

        assert "stage_5c" not in result.stages_completed

    def test_full_m5_stage_order(self, tmp_path: Path) -> None:
        runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubLLMClient(),
            contamination_model_id="gpt-4o",
        )
        result = runner.run()

        assert result.stages_completed == (
            "stage_0",
            "stage_1a",
            "stage_1b",
            "stage_2",
            "stage_3",
            "stage_5a",
            "stage_5b",
            "stage_6",
            "stage_5c",
            "export",
        )


class TestM5ResumeFromStage6:
    def test_resume_from_stage_6_reruns_synthesis(self, tmp_path: Path) -> None:
        # Full run first to create checkpoints.
        _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubLLMClient(),
            contamination_model_id="gpt-4o",
        ).run()

        resume_runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubLLMClient(),
            contamination_model_id="gpt-4o",
            resume_from="stage_6",
        )
        result = resume_runner.run()

        assert result.stages_completed == ("stage_6", "stage_5c", "export")

    def test_resume_from_stage_5c_skips_synthesis(self, tmp_path: Path) -> None:
        # Full run first.
        _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubLLMClient(),
            contamination_model_id="gpt-4o",
        ).run()

        resume_runner = _build_runner(
            tmp_path,
            gold_synthesizer=_StubGoldAnswerSynthesizer(),
            llm_client=_StubLLMClient(),
            contamination_model_id="gpt-4o",
            resume_from="stage_5c",
        )
        result = resume_runner.run()

        assert result.stages_completed == ("stage_5c", "export")


class TestM5PipelineResultM5Field:
    def test_total_contaminated_field_has_default(self) -> None:
        result = PipelineResult(
            run_id="r1",
            stages_completed=("stage_0",),
            output_dir="/tmp/test",
            total_candidates=1,
            total_validated=1,
            total_flagged=0,
        )
        assert result.total_contaminated == 0
