"""Benchmark pipeline runner with JSONL checkpoint/resume.

Orchestrates stages 0 → 1a → 1b → 2 → 3 → 5a, writing a JSONL
checkpoint after each stage.  Supports ``--resume-from`` to skip
completed stages and read from prior checkpoint files.
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
    BenchmarkSourceSpan,
    EvidenceEntry,
    EvidenceSet,
    QueryCandidate,
    RegulatoryUnit,
    ValidationResult,
)
from benchmark.ports.evidence_builder import EvidenceBuilder
from benchmark.ports.query_generator import QueryGenerator
from benchmark.ports.query_validator import QueryValidator
from benchmark.ports.unit_extractor import UnitExtractor

logger = logging.getLogger(__name__)

# Stage ordering — used for resume logic.
_STAGE_ORDER = (
    "stage_0",
    "stage_1a",
    "stage_1b",
    "stage_2",
    "stage_3",
    "stage_5a",
)

# Maps each stage to the checkpoint file it reads on resume.
_RESUME_INPUT_FILES: dict[str, str] = {
    "stage_0": "",  # no input file — runs from scratch
    "stage_1a": "stage_0_spans.jsonl",
    "stage_1b": "stage_1a_units.jsonl",
    "stage_2": "stage_1b_classified.jsonl",
    "stage_3": "stage_2_evidence.jsonl",
    "stage_5a": "stage_3_candidates.jsonl",
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


# Type alias for the Stage 1b classifier callable.
# LLMExtractor.classify() has signature: (list[RegulatoryUnit]) -> list[RegulatoryUnit]
LLMClassifier = Callable[[list[RegulatoryUnit]], list[RegulatoryUnit]]


class PipelineRunner:
    """Orchestrate benchmark pipeline stages with checkpoint/resume.

    Each stage writes a JSONL checkpoint.  On resume, the runner reads
    from the checkpoint file for the resume stage and proceeds.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        corpus_spans_builder: Callable[[], list[BenchmarkSourceSpan]],
        unit_extractor: UnitExtractor,
        llm_classifier: LLMClassifier | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        query_generator: QueryGenerator | None = None,
        query_validator: QueryValidator | None = None,
    ) -> None:
        self._config = config
        self._corpus_spans_builder = corpus_spans_builder
        self._unit_extractor = unit_extractor
        self._llm_classifier = llm_classifier
        self._evidence_builder = evidence_builder
        self._query_generator = query_generator
        self._query_validator = query_validator
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
                results = self._run_stage_5a(candidates)

            stages_completed.append(stage)

        flagged = sum(1 for r in results if not r.is_valid)

        return PipelineResult(
            run_id=self._config.run_id,
            stages_completed=tuple(stages_completed),
            output_dir=str(self._output_path),
            total_candidates=len(candidates),
            total_validated=len(results),
            total_flagged=flagged,
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
        if self._query_generator is None:
            msg = (
                "Stage 3 requires a QueryGenerator but none was provided"
            )
            raise ValueError(msg)

        logger.info("Running Stage 3: query generation")
        # Build a lookup from unit_id to evidence set.
        evidence_by_unit = {es.unit_id: es for es in evidence_sets}

        # We need the original units to generate queries.  Read from
        # the Stage 1b checkpoint (classified units).
        classified = self._read_checkpoint_units("stage_1b_classified.jsonl")

        all_candidates: list[QueryCandidate] = []
        for unit in classified:
            evidence = evidence_by_unit.get(unit.unit_id)
            if evidence is None:
                logger.warning(
                    "No evidence set for unit %s; skipping", unit.unit_id
                )
                continue
            candidates = self._query_generator.generate(
                unit, evidence, QueryClass.CITATION_LOOKUP
            )
            all_candidates.extend(candidates)

        self._write_checkpoint("stage_3_candidates.jsonl", all_candidates)
        return all_candidates

    def _run_stage_5a(
        self, candidates: list[QueryCandidate]
    ) -> list[ValidationResult]:
        if self._query_validator is None:
            msg = (
                "Stage 5a requires a QueryValidator but none was provided"
            )
            raise ValueError(msg)

        logger.info("Running Stage 5a: deterministic validation")
        results = [
            self._query_validator.validate(c) for c in candidates
        ]
        self._write_checkpoint("stage_5a_validated.jsonl", results)
        return results

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

    def _read_checkpoint_spans(self) -> list[BenchmarkSourceSpan]:
        """Read BenchmarkSourceSpan records from checkpoint."""
        path = self._output_path / "stage_0_spans.jsonl"
        return [
            _dict_to_span(d) for d in self._read_jsonl(path)
        ]

    def _read_checkpoint_units(
        self, filename: str
    ) -> list[RegulatoryUnit]:
        """Read RegulatoryUnit records from checkpoint."""
        path = self._output_path / filename
        return [
            _dict_to_unit(d) for d in self._read_jsonl(path)
        ]

    def _read_checkpoint_evidence(self) -> list[EvidenceSet]:
        """Read EvidenceSet records from checkpoint."""
        path = self._output_path / "stage_2_evidence.jsonl"
        return [
            _dict_to_evidence_set(d) for d in self._read_jsonl(path)
        ]

    def _read_checkpoint_candidates(self) -> list[QueryCandidate]:
        """Read QueryCandidate records from checkpoint."""
        path = self._output_path / "stage_3_candidates.jsonl"
        return [
            _dict_to_candidate(d) for d in self._read_jsonl(path)
        ]

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
