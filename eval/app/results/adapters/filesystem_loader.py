"""
Filesystem-based run loader for evaluation results.

Loads runs from the standard eval/runs/ directory structure:
    eval/runs/
        run_YYYY_MM_DDTHH-MM/
            metrics.json
            results.jsonl
            traces.jsonl (optional)
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.app.results.domain.models import LoadedRun, QueryTrace, RunSummary
from rag.eval.models import (
    EvalAggregates,
    EvalResult,
    EvalRunMeta,
)

logger = logging.getLogger(__name__)

# Pattern for parsing run directory names: run_YYYY_MM_DDTHH-MM
RUN_DIR_PATTERN = re.compile(r"run_(\d{4}_\d{2}_\d{2}T\d{2}-\d{2})")



def _parse_timestamp_from_dirname(dirname: str) -> datetime | None:
    """Parse timestamp from run directory name."""
    match = RUN_DIR_PATTERN.match(dirname)
    if not match:
        return None
    ts_str = match.group(1)
    try:
        return datetime.strptime(ts_str, "%Y_%m_%dT%H-%M").replace(tzinfo=UTC)
    except ValueError:
        return None


@dataclass
class FilesystemRunLoader:
    """Loads evaluation runs from filesystem directories.

    Implements the RunLoader protocol for loading runs from the standard
    eval/runs/ directory structure.
    """

    runs_dir: Path
    _cache: dict[str, LoadedRun] = field(default_factory=dict)

    def discover_runs(self) -> list[RunSummary]:
        """Discover all runs in the runs directory."""
        if not self.runs_dir.exists():
            logger.warning(f"Runs directory does not exist: {self.runs_dir}")
            return []

        summaries = []
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if not run_dir.name.startswith("run_"):
                continue

            metrics_file = run_dir / "metrics.json"
            if not metrics_file.exists():
                logger.debug(f"Skipping {run_dir.name}: no metrics.json")
                continue

            try:
                summary = self._load_summary_from_dir(run_dir)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to load summary from {run_dir}: {e}")
                continue

        # Sort by timestamp descending (most recent first)
        summaries.sort(key=lambda s: s.timestamp, reverse=True)
        return summaries

    def load_run(self, run_id: str) -> LoadedRun:
        """Load complete run data by ID."""
        if run_id in self._cache:
            return self._cache[run_id]

        run_dir = self._find_run_dir(run_id)
        loaded = self._load_run_from_dir(run_dir)
        self._cache[run_id] = loaded
        return loaded

    def load_summary(self, run_id: str) -> RunSummary:
        """Load only the summary for a run."""
        run_dir = self._find_run_dir(run_id)
        return self._load_summary_from_dir(run_dir)

    def _find_run_dir(self, run_id: str) -> Path:
        """Find run directory by ID."""
        # First try direct match
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir() and run_dir.name == run_id:
                return run_dir

        # Try matching by run_id from metrics.json
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with metrics_file.open() as f:
                        data = json.load(f)
                    if data.get("meta", {}).get("run_id") == run_id:
                        return run_dir
                except Exception:
                    continue

        raise FileNotFoundError(f"Run not found: {run_id}")

    def _load_summary_from_dir(self, run_dir: Path) -> RunSummary:
        """Load RunSummary from a run directory."""
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"No metrics.json in {run_dir}")
        with metrics_file.open() as f:
            data = json.load(f)

        meta = data.get("meta", {})
        overall = data.get("overall", {})
        answer_quality = data.get("answer_quality", {})
        latency = data.get("latency_ms", {})

        # Parse timestamp from directory name or meta
        timestamp = _parse_timestamp_from_dirname(run_dir.name)
        if timestamp is None and "started_at" in meta:
            try:
                timestamp = datetime.fromisoformat(meta["started_at"])
            except ValueError:
                timestamp = datetime.now(UTC)
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Use directory name as run_id for easier lookup
        run_id = run_dir.name

        # Build display name: include run_name if present
        run_name = meta.get("run_name")
        display_name = f"{run_dir.name} [{run_name}]" if run_name else run_dir.name

        return RunSummary(
            run_id=run_id,
            timestamp=timestamp,
            display_name=display_name,
            run_dir=run_dir,
            num_queries=int(overall.get("num_queries", 0)),
            run_generation=meta.get("run_generation", False),
            generator_model=meta.get("generator_model"),
            embedder_model=meta.get("embedder_model"),
            reranker_name=meta.get("reranker_name"),
            overall_recall_at_10=overall.get("recall@10"),
            overall_ndcg_at_10=overall.get("ndcg@10"),
            avg_quality_score=answer_quality.get("avg_quality_score"),
            avg_latency_ms=latency.get("avg"),
        )

    def _load_run_from_dir(self, run_dir: Path) -> LoadedRun:
        """Load complete run from directory."""
        # Load metrics.json
        metrics_file = run_dir / "metrics.json"
        with metrics_file.open() as f:
            metrics_data = json.load(f)

        # Parse meta
        meta = self._parse_meta(metrics_data.get("meta", {}))

        # Parse aggregates
        aggregates = self._parse_aggregates(metrics_data)

        # Load results.jsonl
        results = self._load_results(run_dir)

        # Load traces.jsonl (optional)
        traces = self._load_traces(run_dir)

        # Build summary
        summary = self._load_summary_from_dir(run_dir)

        return LoadedRun(
            summary=summary,
            meta=meta,
            aggregates=aggregates,
            results=tuple(results),
            traces=traces,
            raw_metrics=metrics_data,
        )

    def _parse_meta(self, data: dict[str, Any]) -> EvalRunMeta:
        """Parse EvalRunMeta from metrics.json meta section."""
        return EvalRunMeta.from_dict(data)

    def _parse_aggregates(self, data: dict[str, Any]) -> EvalAggregates:
        """Parse EvalAggregates from metrics.json (flattened format)."""
        return EvalAggregates.from_flat_dict(data)

    def _load_results(self, run_dir: Path) -> list[EvalResult]:
        """Load results from results.jsonl."""
        results_file = run_dir / "results.jsonl"
        if not results_file.exists():
            logger.warning(f"No results.jsonl in {run_dir}")
            return []

        results = []
        with results_file.open() as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    result = self._parse_eval_result(data)
                    results.append(result)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse result at line {line_num} in {results_file}: {e}"
                    )
                    continue

        return results

    def _parse_eval_result(self, data: dict[str, Any]) -> EvalResult:
        """Parse EvalResult from JSONL row (results.jsonl format)."""
        return EvalResult.from_results_dict(data)

    def _load_traces(self, run_dir: Path) -> dict[str, QueryTrace]:
        """Load traces from traces.jsonl."""
        traces_file = run_dir / "traces.jsonl"
        if not traces_file.exists():
            logger.debug(f"No traces.jsonl in {run_dir}")
            return {}

        traces = {}
        with traces_file.open() as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    trace = self._parse_trace(data)
                    traces[trace.trace_id] = trace
                except Exception as e:
                    logger.warning(
                        f"Failed to parse trace at line {line_num} in {traces_file}: {e}"
                    )
                    continue

        return traces

    def _parse_trace(self, data: dict[str, Any]) -> QueryTrace:
        """Parse QueryTrace from JSONL row."""
        # Parse created_at timestamp
        created_at = None
        if data.get("created_at"):
            with contextlib.suppress(ValueError):
                created_at = datetime.fromisoformat(str(data["created_at"]))

        # Parse retrieved candidates
        retrieved_candidates = []
        for cand in data.get("retrieved", []):
            retrieved_candidates.append(cand)

        # Parse reranked candidates
        reranked_candidates = None
        if data.get("reranked"):
            reranked_candidates = tuple(data["reranked"])

        # Parse packed chunk IDs
        packed_chunk_ids = None
        if data.get("packed_chunk_ids"):
            packed_chunk_ids = tuple(data["packed_chunk_ids"])

        # Parse answer data
        answer_text = None
        citations = None
        answer_data = data.get("answer")
        if answer_data:
            answer_text = answer_data.get("text")
            if answer_data.get("citations"):
                citations = tuple(answer_data["citations"])

        return QueryTrace(
            trace_id=data.get("trace_id", "unknown"),
            query=data.get("query", ""),
            created_at=created_at,
            top_k=data.get("top_k", 10),
            retrieved_candidates=tuple(retrieved_candidates),
            reranked_candidates=reranked_candidates,
            keep_k=data.get("keep_k"),
            reranker=data.get("reranker"),
            token_budget=data.get("token_budget"),
            packed_chunk_ids=packed_chunk_ids,
            model=data.get("model"),
            answer_text=answer_text,
            citations=citations,
            latency_ms=data.get("latency_ms"),
            raw_data=data,
        )

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
