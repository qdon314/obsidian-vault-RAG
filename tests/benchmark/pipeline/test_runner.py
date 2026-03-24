"""Tests for the benchmark pipeline runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.domain.enums import EvidenceTier, QueryClass, UnitKind
from benchmark.domain.models import (
    BenchmarkSourceSpan,
    EvidenceEntry,
    EvidenceSet,
    QueryCandidate,
    RegulatoryUnit,
    ValidationResult,
)
from benchmark.pipeline.runner import PipelineConfig, PipelineResult, PipelineRunner

# ── Helpers ──────────────────────────────────────────────────────────


def _make_span(
    *,
    text: str = "Temperature limit.",
    snapshot_id: str = "snap1",
) -> BenchmarkSourceSpan:
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
                text="Temperature limit.",
                char_start=0,
                char_end=18,
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
    def __init__(self, *, is_valid: bool = True) -> None:
        self._is_valid = is_valid

    def validate(self, candidate: QueryCandidate) -> ValidationResult:
        flags = () if self._is_valid else ("test_flag",)
        return ValidationResult(
            candidate_id=candidate.candidate_id,
            is_valid=self._is_valid,
            flags=flags,
        )

    def refine_evidence(self, query: object, evidence: EvidenceSet) -> EvidenceSet:
        return evidence


def _mock_classifier(units: list[RegulatoryUnit]) -> list[RegulatoryUnit]:
    return units  # identity — no LLM needed


def _build_runner(
    tmp_path: Path,
    *,
    resume_from: str | None = None,
    snapshot_id: str = "",
    include_classifier: bool = True,
    include_evidence: bool = True,
    include_generator: bool = True,
    include_validator: bool = True,
    validator_valid: bool = True,
) -> PipelineRunner:
    config = PipelineConfig(
        run_id="test_run",
        output_dir=str(tmp_path),
        resume_from=resume_from,
        corpus_snapshot_id=snapshot_id,
    )
    return PipelineRunner(
        config,
        corpus_spans_builder=lambda: [_make_span()],
        unit_extractor=_MockUnitExtractor(),
        llm_classifier=_mock_classifier if include_classifier else None,
        evidence_builder=_MockEvidenceBuilder() if include_evidence else None,
        query_generator=_MockQueryGenerator() if include_generator else None,
        query_validator=(
            _MockQueryValidator(is_valid=validator_valid) if include_validator else None
        ),
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestFullRun:
    def test_executes_all_stages(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        result = runner.run()

        assert result.run_id == "test_run"
        assert result.stages_completed == (
            "stage_0",
            "stage_1a",
            "stage_1b",
            "stage_2",
            "stage_3",
            "stage_5a",
            "stage_5b",
            "export",
        )

    def test_writes_all_checkpoints(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        runner.run()

        run_dir = tmp_path / "test_run"
        assert (run_dir / "run_config.json").exists()
        assert (run_dir / "stage_0_spans.jsonl").exists()
        assert (run_dir / "stage_1a_units.jsonl").exists()
        assert (run_dir / "stage_1b_classified.jsonl").exists()
        assert (run_dir / "stage_2_evidence.jsonl").exists()
        assert (run_dir / "stage_3_candidates.jsonl").exists()
        assert (run_dir / "stage_5a_validated.jsonl").exists()

    def test_checkpoint_files_are_valid_jsonl(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        runner.run()

        candidates_path = tmp_path / "test_run" / "stage_3_candidates.jsonl"
        lines = candidates_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        parsed = json.loads(lines[0])
        assert parsed["candidate_id"] == "qc_50.46_b_1_citation_lookup_0"
        assert parsed["query_class"] == "citation_lookup"

    def test_run_config_written(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        runner.run()

        config_path = tmp_path / "test_run" / "run_config.json"
        config = json.loads(config_path.read_text())
        assert config["run_id"] == "test_run"

    def test_result_counts(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        result = runner.run()

        assert result.total_candidates == 1
        assert result.total_validated == 1
        assert result.total_flagged == 0

    def test_result_counts_with_flagged(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, validator_valid=False)
        result = runner.run()

        assert result.total_flagged == 1


class TestResume:
    def test_resume_from_stage_5a(self, tmp_path: Path) -> None:
        # First: full run to create checkpoints
        runner = _build_runner(tmp_path)
        runner.run()

        # Resume from stage_5a — should run stage_5a, stage_5b, export
        resume_runner = _build_runner(
            tmp_path,
            resume_from="stage_5a",
            include_classifier=False,
            include_evidence=False,
            include_generator=False,
        )
        result = resume_runner.run()

        assert result.stages_completed == ("stage_5a", "stage_5b", "export")
        assert result.total_validated == 1

    def test_resume_from_stage_3(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path)
        runner.run()

        resume_runner = _build_runner(
            tmp_path,
            resume_from="stage_3",
            include_classifier=False,
            include_evidence=False,
        )
        result = resume_runner.run()

        assert result.stages_completed == ("stage_3", "stage_5a", "stage_5b", "export")

    def test_resume_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        # No prior run — checkpoint files don't exist
        runner = _build_runner(tmp_path, resume_from="stage_5a")

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            runner.run()


class TestMissingPort:
    def test_missing_classifier_raises(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, include_classifier=False)
        with pytest.raises(ValueError, match="LLM classifier"):
            runner.run()

    def test_missing_evidence_builder_raises(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, include_evidence=False)
        with pytest.raises(ValueError, match="EvidenceBuilder"):
            runner.run()

    def test_missing_query_generator_produces_no_candidates(self, tmp_path: Path) -> None:
        # No generator registered — Stage 3 logs a warning and produces 0 candidates.
        runner = _build_runner(tmp_path, include_generator=False)
        result = runner.run()

        assert result.total_candidates == 0
        assert result.total_validated == 0

    def test_missing_query_validator_raises(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, include_validator=False)
        with pytest.raises(ValueError, match="QueryValidator"):
            runner.run()


class TestSnapshotVerification:
    def test_matching_snapshot_passes(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, snapshot_id="snap1")
        result = runner.run()
        assert "stage_0" in result.stages_completed

    def test_mismatched_snapshot_raises(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, snapshot_id="wrong_snap")
        with pytest.raises(ValueError, match="Corpus snapshot mismatch"):
            runner.run()

    def test_empty_snapshot_skips_verification(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, snapshot_id="")
        result = runner.run()
        assert "stage_0" in result.stages_completed


class TestInvalidResumeStage:
    def test_unknown_stage_raises(self, tmp_path: Path) -> None:
        runner = _build_runner(tmp_path, resume_from="stage_99")
        with pytest.raises(ValueError, match="Unknown stage"):
            runner.run()


class TestPipelineConfig:
    def test_frozen(self) -> None:
        config = PipelineConfig(run_id="r1", output_dir="/tmp/test")
        with pytest.raises(AttributeError):
            config.run_id = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        config = PipelineConfig(run_id="r1", output_dir="/tmp/test")
        assert config.resume_from is None
        assert config.corpus_snapshot_id == ""


class TestPipelineResult:
    def test_frozen(self) -> None:
        result = PipelineResult(
            run_id="r1",
            stages_completed=("stage_0",),
            output_dir="/tmp/test",
            total_candidates=10,
            total_validated=10,
            total_flagged=2,
        )
        with pytest.raises(AttributeError):
            result.run_id = "changed"  # type: ignore[misc]

    def test_m4_fields_have_defaults(self) -> None:
        result = PipelineResult(
            run_id="r1",
            stages_completed=("stage_0",),
            output_dir="/tmp/test",
            total_candidates=10,
            total_validated=10,
            total_flagged=2,
        )
        assert result.total_hard_negatives == 0
        assert result.total_exported == 0


# ── M4 runner tests ──────────────────────────────────────────────────


def _make_unanswerable_candidate() -> QueryCandidate:
    """A candidate with is_unanswerable metadata set."""
    return QueryCandidate(
        candidate_id="qc_50.46_b_1_unanswerable_near_miss_0",
        unit_id="50.46_b_1",
        query="What does subsection (z) of 50.46 require?",
        query_class=QueryClass.UNANSWERABLE,
        source_citations=(),
        evidence_span_ids=(),
        corpus_snapshot_id="snap1",
        metadata={
            "is_unanswerable": True,
            "unanswerable_reason": "subsection does not exist",
            "unanswerable_strategy": "near_miss",
        },
    )


class _MockMultiClassGenerator:
    """Returns different candidates per query_class."""

    def generate(
        self,
        unit: RegulatoryUnit,
        evidence: EvidenceSet,
        query_class: QueryClass,
    ) -> list[QueryCandidate]:
        if query_class == QueryClass.CITATION_LOOKUP:
            return [_make_candidate()]
        if query_class == QueryClass.UNANSWERABLE:
            return [_make_unanswerable_candidate()]
        return []


class _MockQueryValidatorPermissive:
    """Passes all candidates with no evidence check."""

    def validate(self, candidate: QueryCandidate) -> ValidationResult:
        return ValidationResult(
            candidate_id=candidate.candidate_id,
            is_valid=True,
            flags=(),
        )

    def refine_evidence(self, query: object, evidence: EvidenceSet) -> EvidenceSet:
        return evidence


def _build_runner_m4(
    tmp_path: Path,
    *,
    query_classes: tuple[QueryClass, ...] = (QueryClass.CITATION_LOOKUP,),
    multi_class: bool = False,
    resume_from: str | None = None,
    valid_as_of: str = "2026-03-24",
) -> PipelineRunner:
    config = PipelineConfig(
        run_id="m4_run",
        output_dir=str(tmp_path),
        resume_from=resume_from,
    )
    if multi_class:
        generator_map = {
            QueryClass.CITATION_LOOKUP: _MockMultiClassGenerator(),
            QueryClass.UNANSWERABLE: _MockMultiClassGenerator(),
        }
        return PipelineRunner(
            config,
            corpus_spans_builder=lambda: [_make_span()],
            unit_extractor=_MockUnitExtractor(),
            llm_classifier=_mock_classifier,
            evidence_builder=_MockEvidenceBuilder(),
            query_generators=generator_map,
            query_validator=_MockQueryValidatorPermissive(),
            query_classes=query_classes,
            valid_as_of=valid_as_of,
        )
    return PipelineRunner(
        config,
        corpus_spans_builder=lambda: [_make_span()],
        unit_extractor=_MockUnitExtractor(),
        llm_classifier=_mock_classifier,
        evidence_builder=_MockEvidenceBuilder(),
        query_generator=_MockMultiClassGenerator(),
        query_validator=_MockQueryValidatorPermissive(),
        query_classes=query_classes,
        valid_as_of=valid_as_of,
    )


class TestM4Checkpoints:
    def test_stage_5a_queries_checkpoint_written(self, tmp_path: Path) -> None:
        runner = _build_runner_m4(tmp_path)
        runner.run()

        assert (tmp_path / "m4_run" / "stage_5a_queries.jsonl").exists()

    def test_stage_5a_refined_evidence_checkpoint_written(self, tmp_path: Path) -> None:
        runner = _build_runner_m4(tmp_path)
        runner.run()

        assert (tmp_path / "m4_run" / "stage_5a_refined_evidence.jsonl").exists()

    def test_benchmark_records_checkpoint_written(self, tmp_path: Path) -> None:
        runner = _build_runner_m4(tmp_path)
        runner.run()

        assert (tmp_path / "m4_run" / "benchmark_records.jsonl").exists()

    def test_benchmark_records_contain_valid_as_of(self, tmp_path: Path) -> None:
        runner = _build_runner_m4(tmp_path, valid_as_of="2026-03-24")
        runner.run()

        path = tmp_path / "m4_run" / "benchmark_records.jsonl"
        record = json.loads(path.read_text().strip().splitlines()[0])
        assert record["valid_as_of"] == "2026-03-24"

    def test_benchmark_records_qid_matches_candidate_id(self, tmp_path: Path) -> None:
        runner = _build_runner_m4(tmp_path)
        runner.run()

        path = tmp_path / "m4_run" / "benchmark_records.jsonl"
        record = json.loads(path.read_text().strip().splitlines()[0])
        assert record["qid"] == "qc_50.46_b_1_citation_lookup_0"

    def test_total_exported_in_result(self, tmp_path: Path) -> None:
        runner = _build_runner_m4(tmp_path)
        result = runner.run()

        assert result.total_exported == 1


class TestM4MultiClassGeneration:
    def test_multi_class_produces_both_types(self, tmp_path: Path) -> None:
        runner = _build_runner_m4(
            tmp_path,
            query_classes=(QueryClass.CITATION_LOOKUP, QueryClass.UNANSWERABLE),
            multi_class=True,
        )
        result = runner.run()

        assert result.total_candidates == 2

    def test_multi_class_benchmark_records_count(self, tmp_path: Path) -> None:
        runner = _build_runner_m4(
            tmp_path,
            query_classes=(QueryClass.CITATION_LOOKUP, QueryClass.UNANSWERABLE),
            multi_class=True,
        )
        runner.run()

        path = tmp_path / "m4_run" / "benchmark_records.jsonl"
        lines = [ln for ln in path.read_text().strip().splitlines() if ln]
        assert len(lines) == 2

    def test_unanswerable_record_has_is_unanswerable_true(self, tmp_path: Path) -> None:
        runner = _build_runner_m4(
            tmp_path,
            query_classes=(QueryClass.CITATION_LOOKUP, QueryClass.UNANSWERABLE),
            multi_class=True,
        )
        runner.run()

        path = tmp_path / "m4_run" / "benchmark_records.jsonl"
        records = [json.loads(ln) for ln in path.read_text().strip().splitlines()]
        unanswerable = [r for r in records if r["is_unanswerable"]]
        assert len(unanswerable) == 1
        assert unanswerable[0]["unanswerable_reason"] == "subsection does not exist"


class TestM4BackwardCompat:
    def test_singular_query_generator_still_works(self, tmp_path: Path) -> None:
        # query_generator= (singular) should be wrapped automatically.
        config = PipelineConfig(run_id="compat_run", output_dir=str(tmp_path))
        runner = PipelineRunner(
            config,
            corpus_spans_builder=lambda: [_make_span()],
            unit_extractor=_MockUnitExtractor(),
            llm_classifier=_mock_classifier,
            evidence_builder=_MockEvidenceBuilder(),
            query_generator=_MockQueryGenerator(),
            query_validator=_MockQueryValidatorPermissive(),
        )
        result = runner.run()
        assert result.total_candidates == 1

    def test_both_generator_params_raises(self, tmp_path: Path) -> None:
        config = PipelineConfig(run_id="err_run", output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Cannot pass both"):
            PipelineRunner(
                config,
                corpus_spans_builder=lambda: [_make_span()],
                unit_extractor=_MockUnitExtractor(),
                query_generator=_MockQueryGenerator(),
                query_generators={QueryClass.CITATION_LOOKUP: _MockQueryGenerator()},
            )

    def test_stage_5b_skipped_without_retriever(self, tmp_path: Path) -> None:
        # No retriever — stage_5b runs but produces no hard negatives.
        runner = _build_runner_m4(tmp_path)
        result = runner.run()

        assert result.total_hard_negatives == 0
        assert "stage_5b" in result.stages_completed


class TestM4Resume:
    def test_resume_from_stage_5b(self, tmp_path: Path) -> None:
        # Full run first to create checkpoints.
        _build_runner_m4(tmp_path).run()

        resume_runner = _build_runner_m4(tmp_path, resume_from="stage_5b")
        result = resume_runner.run()

        assert result.stages_completed == ("stage_5b", "export")

    def test_resume_from_export(self, tmp_path: Path) -> None:
        _build_runner_m4(tmp_path).run()

        resume_runner = _build_runner_m4(tmp_path, resume_from="export")
        result = resume_runner.run()

        assert result.stages_completed == ("export",)
