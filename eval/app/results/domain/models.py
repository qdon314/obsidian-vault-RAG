"""
Domain models for evaluation results analysis.

These models provide rich representations for loaded evaluation runs,
comparisons between runs, and trend analysis across multiple runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag.eval.models import EvalAggregates, EvalResult, EvalRunMeta


@dataclass(frozen=True)
class RunSummary:
    """Lightweight run metadata for listing and selection.

    This contains just enough information to display runs in a selector
    without loading the full results.
    """

    run_id: str
    timestamp: datetime
    display_name: str
    run_dir: Path

    # Quick stats from metrics.json
    num_queries: int
    run_generation: bool
    generator_model: str | None
    embedder_model: str | None
    reranker_name: str | None

    # Top-level metrics (for quick filtering/sorting)
    overall_recall_at_10: float | None
    overall_ndcg_at_10: float | None
    avg_quality_score: float | None
    avg_latency_ms: float | None


@dataclass(frozen=True, slots=True)
class QueryTrace:
    """A single query trace from traces.jsonl.

    Contains detailed pipeline execution information for a single query.
    """

    trace_id: str
    query: str
    created_at: datetime | None

    # Retrieval stage
    top_k: int
    retrieved_candidates: tuple[dict[str, Any], ...]  # Full candidate data with chunks

    # Reranking stage
    reranked_candidates: tuple[dict[str, Any], ...] | None
    keep_k: int | None
    reranker: str | None

    # Context building stage
    token_budget: int | None
    packed_chunk_ids: tuple[str, ...] | None

    # Generation stage
    model: str | None
    answer_text: str | None
    citations: tuple[dict[str, Any], ...] | None

    # Performance
    latency_ms: int | None

    # Raw data for full access
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """Fully loaded evaluation run with all data.

    Contains the complete EvalRun object from the harness plus
    computed summary for quick access.
    """

    summary: RunSummary
    meta: EvalRunMeta
    aggregates: EvalAggregates
    results: tuple[EvalResult, ...]

    # Optional: traces indexed by trace_id
    traces: dict[str, QueryTrace] = field(default_factory=dict)

    # Raw metrics.json for full access
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def get_results_by_type(self, query_type: str) -> list[EvalResult]:
        """Filter results by query type."""
        return [r for r in self.results if r.query_type and r.query_type.value == query_type]

    def get_results_by_difficulty(self, difficulty: str) -> list[EvalResult]:
        """Filter results by difficulty level."""
        return [r for r in self.results if r.difficulty and r.difficulty.value == difficulty]

    def get_passing_results(self, threshold: float = 0.5) -> list[EvalResult]:
        """Filter results where recall@10 >= threshold."""
        passing = []
        for r in self.results:
            retrieved = set(r.retrieval_result.retrieved_chunk_ids[:10])
            relevant = r.retrieval_result.relevant_chunk_ids
            if relevant:
                recall = len(retrieved & relevant) / len(relevant)
                if recall >= threshold:
                    passing.append(r)
        return passing

    def get_failing_results(self, threshold: float = 0.5) -> list[EvalResult]:
        """Filter results where recall@10 < threshold."""
        failing = []
        for r in self.results:
            retrieved = set(r.retrieval_result.retrieved_chunk_ids[:10])
            relevant = r.retrieval_result.relevant_chunk_ids
            if relevant:
                recall = len(retrieved & relevant) / len(relevant)
                if recall < threshold:
                    failing.append(r)
            else:
                # No relevant chunks defined - can't determine pass/fail
                pass
        return failing


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Comparison between two evaluation runs.

    Contains metric deltas and lists of improved/regressed queries.
    """

    run_a: LoadedRun
    run_b: LoadedRun

    # Metric deltas (b - a, positive means improvement)
    recall_delta: dict[int, float] = field(default_factory=dict)
    precision_delta: dict[int, float] = field(default_factory=dict)
    ndcg_delta: dict[int, float] = field(default_factory=dict)
    mrr_delta: float = 0.0
    map_delta: float = 0.0

    # Answer quality deltas
    quality_delta: float | None = None
    latency_delta: float | None = None

    # Per-query changes
    improved_queries: tuple[str, ...] = ()
    regressed_queries: tuple[str, ...] = ()
    unchanged_queries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrendAnalysis:
    """Multi-run trending analysis over time.

    Contains time series data for key metrics across multiple runs.
    """

    runs: tuple[LoadedRun, ...]  # Sorted by timestamp

    # Time series data (indexed same as runs)
    timestamps: tuple[datetime, ...]

    # Retrieval metrics series (keyed by k)
    recall_series: dict[int, tuple[float, ...]] = field(default_factory=dict)
    precision_series: dict[int, tuple[float, ...]] = field(default_factory=dict)
    ndcg_series: dict[int, tuple[float, ...]] = field(default_factory=dict)

    # Global metrics series
    mrr_series: tuple[float, ...] = ()
    map_series: tuple[float, ...] = ()

    # Answer quality series (may have None values)
    quality_series: tuple[float | None, ...] = ()
    latency_series: tuple[float | None, ...] = ()

    @property
    def num_runs(self) -> int:
        """Number of runs in the analysis."""
        return len(self.runs)

    @property
    def date_range(self) -> tuple[datetime, datetime] | None:
        """Start and end timestamps of the analysis."""
        if not self.timestamps:
            return None
        return (self.timestamps[0], self.timestamps[-1])
