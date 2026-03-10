# eval/app_v2/engine/domain/models.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from eval.app_v2.engine.domain.enums import (
    ComparisonClassification,
    DeltaDirection,
    DiagnosticCode,
    GenerationStatus,
    PackingStatus,
    RerankStatus,
    RetrievalStatus,
    Severity,
)
from eval.app_v2.engine.domain.warnings import BundleWarning

# Re-use existing eval domain types rather than reimplementing
from rag.eval.answer_metrics import AnswerQualityMetrics as AnswerMetrics
from rag.eval.judges import GroundednessJudgeResult as GroundednessOutcome
from rag.eval.models import EvalAggregates
from rag.eval.verdict import Verdict


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Normalized subset of EvalRunMeta used for config-change detection."""
    retriever: str | None
    index_name: str | None
    reranker_model: str | None
    reranker_top_n: int | None
    generator_model: str | None
    embedder_model: str | None
    top_k: int
    token_budget: int


@dataclass(frozen=True, slots=True)
class QueryTrace:
    """Normalized trace for a single query, joined from traces.jsonl."""
    trace_id: str
    reranked_chunk_ids: tuple[str, ...] | None
    packed_chunk_ids: tuple[str, ...] | None
    raw_data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class QueryRecord:
    qid: str
    query: str
    query_type: str | None
    difficulty: str | None
    is_unanswerable: bool
    requires_synthesis: bool
    tags: tuple[str, ...]

    # Retrieval
    relevant_chunk_ids: frozenset[str]
    retrieved_chunk_ids: tuple[str, ...]
    reranked_chunk_ids: tuple[str, ...] | None
    packed_chunk_ids: tuple[str, ...] | None

    # Per-query metrics
    per_query_recall_at_k: Mapping[int, float]
    per_query_precision_at_k: Mapping[int, float]
    per_query_ndcg_at_k: Mapping[int, float]
    per_query_hit_rate_at_k: Mapping[int, float]

    # Generation
    answer_text: str | None
    answer_metrics: AnswerMetrics | None
    groundedness: GroundednessOutcome | None
    latency_ms: int | None

    # Trace
    trace_id: str | None
    trace: QueryTrace | None


@dataclass(frozen=True, slots=True)
class QueryDiagnostic:
    qid: str
    diagnostic_code: DiagnosticCode
    severity: Severity
    retrieval_status: RetrievalStatus
    rerank_status: RerankStatus
    packing_status: PackingStatus
    generation_status: GenerationStatus
    root_cause_summary: str
    suggested_next_check: str | None
    evidence_present: bool
    trace_available: bool


@dataclass(frozen=True, slots=True)
class AnalyzedQuery:
    record: QueryRecord
    diagnostic: QueryDiagnostic


@dataclass(frozen=True, slots=True)
class SliceKey:
    parts: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SliceMetricRow:
    key: SliceKey
    size: int
    metrics: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class SliceMetricTable:
    group_by: tuple[str, ...]
    rows: tuple[SliceMetricRow, ...]


@dataclass(frozen=True, slots=True)
class RunHealthSummary:
    headline_recall_at_10: float
    headline_ndcg_at_10: float
    avg_quality_score: float | None
    avg_latency_ms: float | None
    severity_counts: Mapping[Severity, int]
    diagnostic_counts: Mapping[DiagnosticCode, int]
    dominant_failure_mode: DiagnosticCode | None
    dominant_failure_summary: str | None
    worst_slice: SliceKey | None
    verdict_status: Literal["SHIP", "BLOCK"] | None


@dataclass(frozen=True, slots=True)
class VerdictSummary:
    """Thin wrapper around Verdict for display in RunBundle."""
    decision: Literal["SHIP", "BLOCK"]
    failed_check_names: tuple[str, ...]
    raw: Verdict


@dataclass(frozen=True, slots=True)
class RunBundle:
    run_id: str
    display_name: str
    timestamp: datetime
    config: RunConfig
    aggregates: EvalAggregates
    queries: tuple[AnalyzedQuery, ...]
    health: RunHealthSummary
    verdict: VerdictSummary | None
    warnings: tuple[BundleWarning, ...]
    raw_artifacts: Mapping[str, object]


# ── Comparison models ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class QueryDeltaSummary:
    retrieval: DeltaDirection
    groundedness: DeltaDirection
    latency: DeltaDirection
    severity: DeltaDirection
    quality: DeltaDirection = DeltaDirection.INSUFFICIENT
    hallucination: DeltaDirection = DeltaDirection.INSUFFICIENT
    correctness: DeltaDirection = DeltaDirection.INSUFFICIENT
    completeness: DeltaDirection = DeltaDirection.INSUFFICIENT


@dataclass(frozen=True, slots=True)
class ComparedQuery:
    qid: str
    query: str
    retrieval_delta: float | None
    ndcg_delta: float | None
    latency_delta_ms: float | None
    quality_delta: float | None
    correctness_delta: float | None
    completeness_delta: float | None
    hallucination_severity_delta: float | None
    diagnostic_before: QueryDiagnostic | None
    diagnostic_after: QueryDiagnostic | None
    delta_summary: QueryDeltaSummary
    classification: ComparisonClassification


@dataclass(frozen=True, slots=True)
class ComparisonBundle:
    run_a: RunBundle
    run_b: RunBundle
    aggregate_deltas: Mapping[str, float | None]
    slice_deltas: SliceMetricTable | None
    compared_queries: tuple[ComparedQuery, ...]


# ── Trend models ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ConfigFieldChange:
    field_name: str
    before: object
    after: object


@dataclass(frozen=True, slots=True)
class ConfigChangeEvent:
    from_run_id: str
    to_run_id: str
    timestamp: datetime
    changes: tuple[ConfigFieldChange, ...]
    annotation: str | None = None


@dataclass(frozen=True, slots=True)
class TrendBundle:
    runs: tuple[RunBundle, ...]
    timestamps: tuple[datetime, ...]
    metric_series: Mapping[str, tuple[float | None, ...]]
    diagnostic_rate_series: Mapping[DiagnosticCode, tuple[float | None, ...]]
    verdict_series: tuple[str | None, ...]
    config_change_events: tuple[ConfigChangeEvent, ...]
