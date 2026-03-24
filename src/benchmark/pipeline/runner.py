"""Benchmark pipeline runner with JSONL checkpoint/resume.

Orchestrates stages 0 -> 1a -> 1b -> 2 -> 3 -> 5a -> 5b -> 6 -> 5c -> export,
writing a JSONL checkpoint after each stage.  Supports ``--resume-from`` to skip
completed stages and read from prior checkpoint files.

M4 additions:
- Stage 5a now calls ``refine_evidence()`` after validation.
- Stage 5b hard negative mining (optional, skipped when no retriever).
- Export stage assembles ``BenchmarkRecord`` objects and calls the exporter.
- Multi-class generation: ``query_generators`` dict replaces single ``query_generator``.

M5 additions:
- Stage 6 gold answer synthesis (optional, skipped when no gold_answer_synthesizer).
- Stage 5c contamination probe (optional, skipped when no contamination_model_id).
- ``PipelineResult.total_contaminated`` count.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.domain.enums import EvidenceTier, QueryClass, UnitKind
from benchmark.domain.models import (
    BenchmarkDataset,
    BenchmarkRecord,
    BenchmarkSourceSpan,
    EvidenceEntry,
    EvidenceSet,
    GoldAnswer,
    HardNegativeResult,
    QueryCandidate,
    RegulatoryUnit,
    StageConfig,
    ValidatedQuery,
    ValidationResult,
)
from benchmark.ports.evidence_builder import EvidenceBuilder
from benchmark.ports.exporter import BenchmarkExporter
from benchmark.ports.gold_answer_synthesizer import GoldAnswerSynthesizer
from benchmark.ports.llm_client import LLMClient
from benchmark.ports.query_generator import QueryGenerator
from benchmark.ports.query_validator import QueryValidator
from benchmark.ports.unit_extractor import UnitExtractor
from benchmark.stages.stage_5b_hard_negatives import mine_hard_negatives
from benchmark.stages.stage_5c_contamination import run_contamination_probe
from rag.ports.retriever import Retriever

logger = logging.getLogger(__name__)

# Stage ordering — used for resume logic.
_STAGE_ORDER = (
    "stage_0",
    "stage_1a",
    "stage_1b",
    "stage_2",
    "stage_3",
    "stage_5a",
    "stage_5b",
    "stage_6",   # M5: gold answer synthesis (must precede stage_5c)
    "stage_5c",  # M5: contamination probe
    "export",
)

# Maps each stage to the checkpoint file it reads on resume.
_RESUME_INPUT_FILES: dict[str, str] = {
    "stage_0": "",  # no input file — runs from scratch
    "stage_1a": "stage_0_spans.jsonl",
    "stage_1b": "stage_1a_units.jsonl",
    "stage_2": "stage_1b_classified.jsonl",
    "stage_3": "stage_2_evidence.jsonl",
    "stage_5a": "stage_3_candidates.jsonl",
    "stage_5b": "stage_5a_queries.jsonl",
    "stage_6": "stage_5a_queries.jsonl",          # reads validated queries
    "stage_5c": "stage_6_gold_answers.jsonl",     # reads records with gold answers
    "export": "stage_5a_queries.jsonl",
}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for a pipeline run."""

    run_id: str
    output_dir: str
    resume_from: str | None = None
    corpus_snapshot_id: str = ""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Summary of a completed pipeline run."""

    run_id: str
    stages_completed: tuple[str, ...]
    output_dir: str
    total_candidates: int
    total_validated: int
    total_flagged: int
    total_hard_negatives: int = 0
    total_exported: int = 0
    total_contaminated: int = 0  # M5: records flagged as contaminated


# Type alias for the Stage 1b classifier callable.
LLMClassifier = Callable[[list[RegulatoryUnit]], list[RegulatoryUnit]]


class PipelineRunner:
    """Orchestrate benchmark pipeline stages with checkpoint/resume.

    Each stage writes a JSONL checkpoint.  On resume, the runner reads
    from the checkpoint file for the resume stage and proceeds.

    M4 constructor additions:
    - ``query_generators``: dict mapping ``QueryClass`` to ``QueryGenerator``.
      Replaces the singular ``query_generator`` param (backward compatible).
    - ``retriever``: optional live RAG retriever for Stage 5b hard negative
      mining.  Stage 5b is skipped when ``None``.
    - ``retriever_config``: metadata dict recorded on every ``HardNegativeResult``
      for staleness detection.
    - ``exporter``: optional ``BenchmarkExporter`` for the terminal export stage.
      Export stage is skipped when ``None`` (benchmark_records.jsonl still written).
    - ``query_classes``: which classes to generate in Stage 3.
    - ``valid_as_of``: ISO date recorded on ``BenchmarkRecord`` exports.

    M5 constructor additions:
    - ``gold_answer_synthesizer``: optional ``GoldAnswerSynthesizer`` for Stage 6.
      Stage 6 is skipped when ``None``.
    - ``llm_client``: optional ``LLMClient`` used by Stage 5c contamination probe.
    - ``contamination_model_id``: model identifier for the contamination probe.
      Stage 5c is skipped when ``None``.
    - ``contamination_stage_config``: ``StageConfig`` passed to Stage 5c LLM calls.
      Defaults to ``StageConfig(model=contamination_model_id)`` when not provided.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        corpus_spans_builder: Callable[[], list[BenchmarkSourceSpan]],
        unit_extractor: UnitExtractor,
        llm_classifier: LLMClassifier | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        # Backward compat: singular form wraps as {CITATION_LOOKUP: generator}.
        query_generator: QueryGenerator | None = None,
        # M4: preferred plural form.
        query_generators: dict[QueryClass, QueryGenerator] | None = None,
        query_validator: QueryValidator | None = None,
        # M4 additions.
        retriever: Retriever | None = None,
        retriever_config: dict[str, Any] | None = None,
        exporter: BenchmarkExporter | None = None,
        query_classes: tuple[QueryClass, ...] = (QueryClass.CITATION_LOOKUP,),
        valid_as_of: str = "",
        # M5 additions.
        gold_answer_synthesizer: GoldAnswerSynthesizer | None = None,
        llm_client: LLMClient | None = None,
        contamination_model_id: str | None = None,
        contamination_stage_config: StageConfig | None = None,
    ) -> None:
        if query_generator is not None and query_generators is not None:
            msg = "Cannot pass both query_generator and query_generators"
            raise ValueError(msg)
        if query_generator is not None:
            query_generators = {QueryClass.CITATION_LOOKUP: query_generator}

        self._config = config
        self._corpus_spans_builder = corpus_spans_builder
        self._unit_extractor = unit_extractor
        self._llm_classifier = llm_classifier
        self._evidence_builder = evidence_builder
        self._query_generators: dict[QueryClass, QueryGenerator] = query_generators or {}
        self._query_validator = query_validator
        self._retriever = retriever
        self._retriever_config = retriever_config or {}
        self._exporter = exporter
        self._query_classes = query_classes
        self._valid_as_of = valid_as_of
        # M5
        self._gold_answer_synthesizer = gold_answer_synthesizer
        self._llm_client = llm_client
        self._contamination_model_id = contamination_model_id
        self._contamination_stage_config = contamination_stage_config or (
            StageConfig(model=contamination_model_id) if contamination_model_id else None
        )
        self._output_path = Path(config.output_dir) / config.run_id

    def run(self) -> PipelineResult:
        """Execute the pipeline, respecting resume_from config."""
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._write_run_config()

        start_idx = self._resolve_start_index()
        stages_completed: list[str] = []

        # Intermediate state passed between stages.
        spans: list[BenchmarkSourceSpan] = []
        units: list[RegulatoryUnit] = []
        classified: list[RegulatoryUnit] = []
        evidence_sets: list[EvidenceSet] = []
        candidates: list[QueryCandidate] = []
        results: list[ValidationResult] = []
        # M4 intermediate state.
        validated_queries: list[ValidatedQuery] = []
        refined_evidence_by_cid: dict[str, EvidenceSet] = {}
        hard_negatives: list[HardNegativeResult] = []
        # M5 intermediate state.
        benchmark_records: list[BenchmarkRecord] = []

        for idx in range(start_idx, len(_STAGE_ORDER)):
            stage = _STAGE_ORDER[idx]

            if stage == "stage_0":
                spans = self._run_stage_0()
            elif stage == "stage_1a":
                if not spans:
                    spans = self._read_checkpoint_spans()
                units = self._run_stage_1a(spans)
            elif stage == "stage_1b":
                if not units:
                    units = self._read_checkpoint_units("stage_1a_units.jsonl")
                classified = self._run_stage_1b(units)
            elif stage == "stage_2":
                if not classified:
                    classified = self._read_checkpoint_units(
                        "stage_1b_classified.jsonl"
                    )
                evidence_sets = self._run_stage_2(classified)
            elif stage == "stage_3":
                if not evidence_sets:
                    evidence_sets = self._read_checkpoint_evidence()
                candidates = self._run_stage_3(evidence_sets)
            elif stage == "stage_5a":
                if not candidates:
                    candidates = self._read_checkpoint_candidates()
                if not evidence_sets:
                    evidence_sets = self._read_checkpoint_evidence()
                results, validated_queries, refined_evidence_by_cid = (
                    self._run_stage_5a(candidates, evidence_sets)
                )
            elif stage == "stage_5b":
                if not validated_queries:
                    validated_queries = self._read_checkpoint_validated_queries()
                if not evidence_sets:
                    evidence_sets = self._read_checkpoint_evidence()
                hard_negatives = self._run_stage_5b(validated_queries, evidence_sets)
            elif stage == "stage_6":
                if self._gold_answer_synthesizer is None:
                    logger.info("Stage 6: skipping (no gold_answer_synthesizer)")
                    continue  # do not add to stages_completed
                if not validated_queries:
                    validated_queries = self._read_checkpoint_validated_queries()
                if not refined_evidence_by_cid:
                    refined_evidence_by_cid = self._read_checkpoint_refined_evidence()
                if not hard_negatives:
                    hard_negatives = self._read_checkpoint_hard_negatives()
                benchmark_records = self._run_stage_6(
                    validated_queries, refined_evidence_by_cid, hard_negatives
                )
            elif stage == "stage_5c":
                if self._contamination_model_id is None or self._llm_client is None:
                    logger.info("Stage 5c: skipping (no contamination_model_id or llm_client)")
                    continue  # do not add to stages_completed
                if not benchmark_records:
                    benchmark_records = self._read_checkpoint_benchmark_records()
                benchmark_records = self._run_stage_5c(benchmark_records)
            elif stage == "export":
                # Prefer M5 benchmark_records (have gold + contamination data).
                # On resume, try reading M5 checkpoints before falling back to
                # assembling from validated queries (M4-only path).
                if not benchmark_records:
                    benchmark_records = self._read_checkpoint_benchmark_records()
                if benchmark_records:
                    self._run_export_from_records(benchmark_records)
                else:
                    if not validated_queries:
                        validated_queries = self._read_checkpoint_validated_queries()
                    if not refined_evidence_by_cid:
                        refined_evidence_by_cid = (
                            self._read_checkpoint_refined_evidence()
                        )
                    if not hard_negatives:
                        hard_negatives = self._read_checkpoint_hard_negatives()
                    benchmark_records = self._run_export(
                        validated_queries, refined_evidence_by_cid, hard_negatives
                    )

            stages_completed.append(stage)

        flagged = sum(1 for r in results if not r.is_valid)
        hn_count = sum(
            len(hn.hard_negative_chunk_ids) for hn in hard_negatives
        )

        # Count exported records from checkpoint.
        exported_count = 0
        records_path = self._output_path / "benchmark_records.jsonl"
        if records_path.exists():
            exported_count = sum(1 for line in records_path.read_text().splitlines() if line.strip())

        # Count contaminated records (M5).
        contaminated_count = 0
        if self._contamination_model_id:
            mid = self._contamination_model_id
            contaminated_count = sum(
                1 for r in benchmark_records
                if r.contamination_probes.get(mid) is True
            )

        return PipelineResult(
            run_id=self._config.run_id,
            stages_completed=tuple(stages_completed),
            output_dir=str(self._output_path),
            total_candidates=len(candidates),
            total_validated=len(results),
            total_flagged=flagged,
            total_hard_negatives=hn_count,
            total_exported=exported_count,
            total_contaminated=contaminated_count,
        )

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_stage_0(self) -> list[BenchmarkSourceSpan]:
        logger.info("Running Stage 0: corpus normalization")
        spans = self._corpus_spans_builder()

        if self._config.corpus_snapshot_id:
            snapshot_ids = {s.corpus_snapshot_id for s in spans}
            if snapshot_ids != {self._config.corpus_snapshot_id}:
                msg = (
                    f"Corpus snapshot mismatch: expected "
                    f"'{self._config.corpus_snapshot_id}', "
                    f"got {snapshot_ids}"
                )
                raise ValueError(msg)

        self._write_checkpoint("stage_0_spans.jsonl", spans)
        return spans

    def _run_stage_1a(
        self, spans: list[BenchmarkSourceSpan]
    ) -> list[RegulatoryUnit]:
        logger.info("Running Stage 1a: structural segmentation")
        units = self._unit_extractor.extract(spans)
        self._write_checkpoint("stage_1a_units.jsonl", units)
        return units

    def _run_stage_1b(
        self, units: list[RegulatoryUnit]
    ) -> list[RegulatoryUnit]:
        if self._llm_classifier is None:
            msg = "Stage 1b requires an LLM classifier but none was provided"
            raise ValueError(msg)

        logger.info("Running Stage 1b: LLM classification")
        classified = self._llm_classifier(units)
        self._write_checkpoint("stage_1b_classified.jsonl", classified)
        return classified

    def _run_stage_2(
        self, units: list[RegulatoryUnit]
    ) -> list[EvidenceSet]:
        if self._evidence_builder is None:
            msg = "Stage 2 requires an EvidenceBuilder but none was provided"
            raise ValueError(msg)

        logger.info("Running Stage 2: evidence tier assignment")
        evidence_sets = [self._evidence_builder.build(u) for u in units]
        self._write_checkpoint("stage_2_evidence.jsonl", evidence_sets)
        return evidence_sets

    def _run_stage_3(
        self, evidence_sets: list[EvidenceSet]
    ) -> list[QueryCandidate]:
        logger.info("Running Stage 3: query generation (%s classes)", len(self._query_classes))

        # Build a lookup from unit_id to evidence set.
        evidence_by_unit = {es.unit_id: es for es in evidence_sets}

        # Read the classified units for generation context.
        classified = self._read_checkpoint_units("stage_1b_classified.jsonl")

        all_candidates: list[QueryCandidate] = []
        for unit in classified:
            evidence = evidence_by_unit.get(unit.unit_id)
            if evidence is None:
                logger.warning(
                    "No evidence set for unit %s; skipping", unit.unit_id
                )
                continue
            for query_class in self._query_classes:
                generator = self._query_generators.get(query_class)
                if generator is None:
                    logger.warning(
                        "No generator registered for %s; skipping",
                        query_class.value,
                    )
                    continue
                candidates = generator.generate(unit, evidence, query_class)
                all_candidates.extend(candidates)

        self._write_checkpoint("stage_3_candidates.jsonl", all_candidates)
        return all_candidates

    def _run_stage_5a(
        self,
        candidates: list[QueryCandidate],
        evidence_sets: list[EvidenceSet],
    ) -> tuple[
        list[ValidationResult],
        list[ValidatedQuery],
        dict[str, EvidenceSet],
    ]:
        if self._query_validator is None:
            msg = (
                "Stage 5a requires a QueryValidator but none was provided"
            )
            raise ValueError(msg)

        logger.info("Running Stage 5a: validation + evidence refinement")
        evidence_by_unit = {es.unit_id: es for es in evidence_sets}

        results: list[ValidationResult] = []
        validated_queries: list[ValidatedQuery] = []
        refined_evidence_by_cid: dict[str, EvidenceSet] = {}

        for candidate in candidates:
            result = self._query_validator.validate(candidate)
            results.append(result)

            if not result.is_valid:
                continue

            # Build ValidatedQuery from passing candidate + scores.
            vq = ValidatedQuery(
                candidate_id=candidate.candidate_id,
                unit_id=candidate.unit_id,
                query=candidate.query,
                query_class=candidate.query_class,
                source_citations=candidate.source_citations,
                evidence_span_ids=candidate.evidence_span_ids,
                difficulty=candidate.difficulty,
                corpus_snapshot_id=candidate.corpus_snapshot_id,
                validation_scores=result.scores,
                metadata=candidate.metadata,
            )
            validated_queries.append(vq)

            # Refine evidence for this specific query.
            unit_evidence = evidence_by_unit.get(candidate.unit_id)
            if unit_evidence is not None:
                refined = self._query_validator.refine_evidence(
                    vq, unit_evidence
                )
            else:
                refined = EvidenceSet(unit_id=candidate.unit_id)
            refined_evidence_by_cid[candidate.candidate_id] = refined

        self._write_checkpoint("stage_5a_validated.jsonl", results)
        self._write_checkpoint("stage_5a_queries.jsonl", validated_queries)
        self._write_refined_evidence_checkpoint(refined_evidence_by_cid)
        return results, validated_queries, refined_evidence_by_cid

    def _run_stage_5b(
        self,
        validated_queries: list[ValidatedQuery],
        evidence_sets: list[EvidenceSet],
    ) -> list[HardNegativeResult]:
        if self._retriever is None:
            logger.info(
                "Stage 5b: no retriever provided — skipping hard negative mining"
            )
            return []

        logger.info("Running Stage 5b: hard negative mining (%d queries)", len(validated_queries))
        evidence_by_unit = {es.unit_id: es for es in evidence_sets}

        hard_negatives = mine_hard_negatives(
            validated_queries,
            evidence_by_unit,
            self._retriever,
            retriever_config=self._retriever_config,
        )
        self._write_checkpoint("stage_5b_hard_negatives.jsonl", hard_negatives)
        return hard_negatives

    def _run_stage_6(
        self,
        validated_queries: list[ValidatedQuery],
        refined_evidence_by_cid: dict[str, EvidenceSet],
        hard_negatives: list[HardNegativeResult],
    ) -> list[BenchmarkRecord]:
        """Synthesise gold answers and assemble BenchmarkRecord objects.

        Caller guarantees ``self._gold_answer_synthesizer`` is not None.
        """
        assert self._gold_answer_synthesizer is not None
        logger.info(
            "Running Stage 6: gold answer synthesis (%d queries)",
            len(validated_queries),
        )
        hn_by_cid = {hn.candidate_id: hn for hn in hard_negatives}

        records: list[BenchmarkRecord] = []
        for vq in validated_queries:
            refined = refined_evidence_by_cid.get(
                vq.candidate_id, EvidenceSet(unit_id=vq.unit_id)
            )
            hn = hn_by_cid.get(vq.candidate_id)

            record = BenchmarkRecord(
                qid=vq.candidate_id,
                query=vq.query,
                query_class=vq.query_class,
                difficulty=vq.difficulty,
                source_unit_ids=(vq.unit_id,),
                source_citations=vq.source_citations,
                critical_evidence=refined.critical,
                supporting_evidence=refined.supporting,
                contextual_evidence=refined.contextual,
                hard_negative_chunk_ids=(
                    hn.hard_negative_chunk_ids if hn is not None else ()
                ),
                hard_negatives_retriever_config=(
                    hn.retriever_config if hn is not None else {}
                ),
                corpus_snapshot_id=vq.corpus_snapshot_id,
                valid_as_of=self._valid_as_of,
                is_unanswerable=bool(vq.metadata.get("is_unanswerable", False)),
                unanswerable_reason=vq.metadata.get("unanswerable_reason"),
                validation_scores=vq.validation_scores,
                metadata=vq.metadata,
            )

            gold = self._gold_answer_synthesizer.synthesize(vq, refined)
            if gold.gold_answer:
                record = dataclasses.replace(
                    record,
                    gold_answer=gold.gold_answer,
                    acceptable_answer_variants=gold.acceptable_answer_variants,
                    required_points=gold.required_points,
                    forbidden_errors=gold.forbidden_errors,
                )
            else:
                logger.warning(
                    "Empty gold answer for query %s; record retained without gold",
                    vq.candidate_id,
                )

            records.append(record)

        self._write_checkpoint("stage_6_gold_answers.jsonl", records)
        return records

    def _run_stage_5c(
        self,
        records: list[BenchmarkRecord],
    ) -> list[BenchmarkRecord]:
        """Run contamination probe for each record with a gold answer.

        Caller guarantees ``self._contamination_model_id`` and
        ``self._llm_client`` are not None.
        """
        assert self._contamination_model_id is not None
        assert self._llm_client is not None
        assert self._contamination_stage_config is not None  # set in __init__
        logger.info(
            "Running Stage 5c: contamination probe (%d records, model=%s)",
            len(records),
            self._contamination_model_id,
        )

        probed = run_contamination_probe(
            records,
            self._llm_client,
            self._contamination_stage_config,
            self._contamination_model_id,
        )
        self._write_checkpoint("stage_5c_probed_records.jsonl", probed)
        return probed

    def _run_export_from_records(
        self, records: list[BenchmarkRecord]
    ) -> None:
        """Export pre-assembled BenchmarkRecord objects (M5 path)."""
        logger.info(
            "Running export stage: writing %d benchmark records", len(records)
        )
        self._write_checkpoint("benchmark_records.jsonl", records)

        if self._exporter is not None:
            snapshot_id = records[0].corpus_snapshot_id if records else ""
            dataset = BenchmarkDataset(
                schema_version="1.0",
                records=tuple(records),
                corpus_snapshot_id=snapshot_id,
                created_at=self._valid_as_of or "",
            )
            self._exporter.export(dataset)
            logger.info("Export complete: %d records", len(records))
        else:
            logger.info(
                "No exporter provided — benchmark_records.jsonl written "
                "but no EvalQuery export performed"
            )

    def _run_export(
        self,
        validated_queries: list[ValidatedQuery],
        refined_evidence_by_cid: dict[str, EvidenceSet],
        hard_negatives: list[HardNegativeResult],
    ) -> list[BenchmarkRecord]:
        logger.info(
            "Running export stage: assembling %d benchmark records",
            len(validated_queries),
        )
        hn_by_cid = {hn.candidate_id: hn for hn in hard_negatives}

        records: list[BenchmarkRecord] = []
        for vq in validated_queries:
            refined = refined_evidence_by_cid.get(
                vq.candidate_id, EvidenceSet(unit_id=vq.unit_id)
            )
            hn = hn_by_cid.get(vq.candidate_id)

            record = BenchmarkRecord(
                qid=vq.candidate_id,
                query=vq.query,
                query_class=vq.query_class,
                difficulty=vq.difficulty,
                source_unit_ids=(vq.unit_id,),
                source_citations=vq.source_citations,
                critical_evidence=refined.critical,
                supporting_evidence=refined.supporting,
                contextual_evidence=refined.contextual,
                hard_negative_chunk_ids=(
                    hn.hard_negative_chunk_ids if hn is not None else ()
                ),
                hard_negatives_retriever_config=(
                    hn.retriever_config if hn is not None else {}
                ),
                corpus_snapshot_id=vq.corpus_snapshot_id,
                valid_as_of=self._valid_as_of,
                is_unanswerable=bool(
                    vq.metadata.get("is_unanswerable", False)
                ),
                unanswerable_reason=vq.metadata.get("unanswerable_reason"),
                validation_scores=vq.validation_scores,
                metadata=vq.metadata,
            )
            records.append(record)

        self._write_checkpoint("benchmark_records.jsonl", records)

        if self._exporter is not None:
            snapshot_id = (
                records[0].corpus_snapshot_id if records else ""
            )
            dataset = BenchmarkDataset(
                schema_version="1.0",
                records=tuple(records),
                corpus_snapshot_id=snapshot_id,
                created_at=self._valid_as_of or "",
            )
            self._exporter.export(dataset)
            logger.info("Export complete: %d records", len(records))
        else:
            logger.info(
                "No exporter provided — benchmark_records.jsonl written "
                "but no EvalQuery export performed"
            )
        return records

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _write_checkpoint(
        self, filename: str, records: list[Any]
    ) -> None:
        """Write records to a JSONL checkpoint file."""
        path = self._output_path / filename
        with path.open("w") as f:
            for record in records:
                line = json.dumps(
                    dataclasses.asdict(record), default=str
                )
                f.write(line + "\n")
        logger.info("Wrote %d records to %s", len(records), path)

    def _write_refined_evidence_checkpoint(
        self, refined_evidence_by_cid: dict[str, EvidenceSet]
    ) -> None:
        """Write refined evidence keyed by candidate_id to JSONL."""
        path = self._output_path / "stage_5a_refined_evidence.jsonl"
        with path.open("w") as f:
            for cid, evidence in refined_evidence_by_cid.items():
                row = {"candidate_id": cid, **dataclasses.asdict(evidence)}
                f.write(json.dumps(row, default=str) + "\n")
        logger.info(
            "Wrote %d refined evidence sets to %s",
            len(refined_evidence_by_cid),
            path,
        )

    def _read_checkpoint_spans(self) -> list[BenchmarkSourceSpan]:
        path = self._output_path / "stage_0_spans.jsonl"
        return [_dict_to_span(d) for d in self._read_jsonl(path)]

    def _read_checkpoint_units(
        self, filename: str
    ) -> list[RegulatoryUnit]:
        path = self._output_path / filename
        return [_dict_to_unit(d) for d in self._read_jsonl(path)]

    def _read_checkpoint_evidence(self) -> list[EvidenceSet]:
        path = self._output_path / "stage_2_evidence.jsonl"
        return [
            _dict_to_evidence_set(d) for d in self._read_jsonl(path)
        ]

    def _read_checkpoint_candidates(self) -> list[QueryCandidate]:
        path = self._output_path / "stage_3_candidates.jsonl"
        return [
            _dict_to_candidate(d) for d in self._read_jsonl(path)
        ]

    def _read_checkpoint_validated_queries(self) -> list[ValidatedQuery]:
        path = self._output_path / "stage_5a_queries.jsonl"
        return [
            _dict_to_validated_query(d) for d in self._read_jsonl(path)
        ]

    def _read_checkpoint_refined_evidence(self) -> dict[str, EvidenceSet]:
        path = self._output_path / "stage_5a_refined_evidence.jsonl"
        result: dict[str, EvidenceSet] = {}
        for d in self._read_jsonl(path):
            cid = d["candidate_id"]
            result[cid] = _dict_to_evidence_set(d)
        return result

    def _read_checkpoint_hard_negatives(self) -> list[HardNegativeResult]:
        path = self._output_path / "stage_5b_hard_negatives.jsonl"
        if not path.exists():
            return []
        return [
            _dict_to_hard_negative(d) for d in self._read_jsonl(path)
        ]

    def _read_checkpoint_benchmark_records(self) -> list[BenchmarkRecord]:
        """Read the latest available BenchmarkRecord checkpoint.

        Checks stage_5c output first (has contamination data), then
        stage_6 output (has gold answers), then falls back to returning
        an empty list so the caller can use the M4 assembly path.
        """
        for filename in (
            "stage_5c_probed_records.jsonl",
            "stage_6_gold_answers.jsonl",
        ):
            path = self._output_path / filename
            if path.exists():
                logger.info("Reading benchmark records from %s", path)
                return [
                    _dict_to_benchmark_record(d)
                    for d in self._read_jsonl(path)
                ]
        return []

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file into a list of dicts."""
        if not path.exists():
            msg = (
                f"Checkpoint file not found: {path}. "
                f"Cannot resume from this stage."
            )
            raise FileNotFoundError(msg)
        records: list[dict[str, Any]] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_run_config(self) -> None:
        """Write pipeline config to run_config.json."""
        path = self._output_path / "run_config.json"
        with path.open("w") as f:
            json.dump(dataclasses.asdict(self._config), f, indent=2)

    def _resolve_start_index(self) -> int:
        """Resolve the stage index to start from."""
        if self._config.resume_from is None:
            return 0
        if self._config.resume_from not in _STAGE_ORDER:
            msg = (
                f"Unknown stage '{self._config.resume_from}'. "
                f"Valid stages: {', '.join(_STAGE_ORDER)}"
            )
            raise ValueError(msg)
        return _STAGE_ORDER.index(self._config.resume_from)


# ------------------------------------------------------------------
# JSONL deserialization helpers
# ------------------------------------------------------------------


def _dict_to_span(d: dict[str, Any]) -> BenchmarkSourceSpan:
    return BenchmarkSourceSpan(
        source_doc_id=d["source_doc_id"],
        citation=d["citation"],
        citation_key=d["citation_key"],
        section_title=d["section_title"],
        text=d["text"],
        char_start=d["char_start"],
        char_end=d["char_end"],
        chunk_ids_overlapping_span=tuple(
            d["chunk_ids_overlapping_span"]
        ),
        parent_section_id=d["parent_section_id"],
        effective_date=d["effective_date"],
        corpus_snapshot_id=d["corpus_snapshot_id"],
        metadata=d.get("metadata", {}),
    )


def _dict_to_unit(d: dict[str, Any]) -> RegulatoryUnit:
    return RegulatoryUnit(
        unit_id=d["unit_id"],
        kind=UnitKind(d["kind"]),
        spans=tuple(_dict_to_span(s) for s in d["spans"]),
        citation=d["citation"],
        subsection_chain=tuple(d["subsection_chain"]),
        parent_section_id=d["parent_section_id"],
        corpus_snapshot_id=d["corpus_snapshot_id"],
        canonical_statement=d.get("canonical_statement"),
        entities=tuple(d.get("entities", ())),
        value=d.get("value"),
        conditions=tuple(d.get("conditions", ())),
        cross_references=tuple(d.get("cross_references", ())),
        metadata=d.get("metadata", {}),
    )


def _dict_to_evidence_entry(d: dict[str, Any]) -> EvidenceEntry:
    return EvidenceEntry(
        span_id=d["span_id"],
        citation=d["citation"],
        text=d["text"],
        char_start=d["char_start"],
        char_end=d["char_end"],
        chunk_ids=tuple(d["chunk_ids"]),
        tier=EvidenceTier(d["tier"]),
    )


def _dict_to_evidence_set(d: dict[str, Any]) -> EvidenceSet:
    return EvidenceSet(
        unit_id=d["unit_id"],
        critical=tuple(
            _dict_to_evidence_entry(e) for e in d.get("critical", ())
        ),
        supporting=tuple(
            _dict_to_evidence_entry(e) for e in d.get("supporting", ())
        ),
        contextual=tuple(
            _dict_to_evidence_entry(e) for e in d.get("contextual", ())
        ),
    )


def _dict_to_candidate(d: dict[str, Any]) -> QueryCandidate:
    return QueryCandidate(
        candidate_id=d["candidate_id"],
        unit_id=d["unit_id"],
        query=d["query"],
        query_class=QueryClass(d["query_class"]),
        source_citations=tuple(d["source_citations"]),
        evidence_span_ids=tuple(d["evidence_span_ids"]),
        difficulty=d.get("difficulty", "easy"),
        corpus_snapshot_id=d.get("corpus_snapshot_id", ""),
        metadata=d.get("metadata", {}),
    )


def _dict_to_validated_query(d: dict[str, Any]) -> ValidatedQuery:
    return ValidatedQuery(
        candidate_id=d["candidate_id"],
        unit_id=d["unit_id"],
        query=d["query"],
        query_class=QueryClass(d["query_class"]),
        source_citations=tuple(d["source_citations"]),
        evidence_span_ids=tuple(d["evidence_span_ids"]),
        difficulty=d.get("difficulty", "easy"),
        corpus_snapshot_id=d.get("corpus_snapshot_id", ""),
        validation_scores=d.get("validation_scores", {}),
        metadata=d.get("metadata", {}),
    )


def _dict_to_hard_negative(d: dict[str, Any]) -> HardNegativeResult:
    return HardNegativeResult(
        candidate_id=d["candidate_id"],
        hard_negative_chunk_ids=tuple(d["hard_negative_chunk_ids"]),
        retriever_config=d.get("retriever_config", {}),
        insufficient=d.get("insufficient", False),
    )


def _dict_to_benchmark_record(d: dict[str, Any]) -> BenchmarkRecord:
    """Deserialise a BenchmarkRecord from a JSONL dict.

    Safe for pre-M5 checkpoint files: all five new fields fall back to
    their domain defaults when absent.
    """
    return BenchmarkRecord(
        qid=d["qid"],
        query=d["query"],
        query_class=QueryClass(d["query_class"]),
        difficulty=d.get("difficulty", "easy"),
        source_unit_ids=tuple(d.get("source_unit_ids", ())),
        source_citations=tuple(d.get("source_citations", ())),
        critical_evidence=tuple(
            _dict_to_evidence_entry(e) for e in d.get("critical_evidence", ())
        ),
        supporting_evidence=tuple(
            _dict_to_evidence_entry(e) for e in d.get("supporting_evidence", ())
        ),
        contextual_evidence=tuple(
            _dict_to_evidence_entry(e) for e in d.get("contextual_evidence", ())
        ),
        hard_negative_chunk_ids=tuple(d.get("hard_negative_chunk_ids", ())),
        hard_negatives_retriever_config=d.get("hard_negatives_retriever_config", {}),
        corpus_snapshot_id=d.get("corpus_snapshot_id", ""),
        valid_as_of=d.get("valid_as_of", ""),
        is_unanswerable=d.get("is_unanswerable", False),
        unanswerable_reason=d.get("unanswerable_reason"),
        validation_scores=d.get("validation_scores", {}),
        metadata=d.get("metadata", {}),
        # M5 fields — safe defaults for pre-M5 checkpoint files.
        gold_answer=d.get("gold_answer"),
        acceptable_answer_variants=tuple(d.get("acceptable_answer_variants", ())),
        required_points=tuple(d.get("required_points", ())),
        forbidden_errors=tuple(d.get("forbidden_errors", ())),
        contamination_probes=dict(d.get("contamination_probes", {})),
    )
