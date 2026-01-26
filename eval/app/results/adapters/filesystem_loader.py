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
from rag.domain.models import Answer, Citation
from rag.eval.answer_metrics import AnswerQualityMetrics
from rag.eval.models import (
    EvalAggregates,
    EvalResult,
    EvalRunMeta,
    RetrievalResult,
    RetrievalSummary,
)
from rag.eval.schema import Difficulty, QueryType

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

        return RunSummary(
            run_id=run_id,
            timestamp=timestamp,
            display_name=run_dir.name,
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
        started_at = datetime.now(UTC)
        if "started_at" in data:
            with contextlib.suppress(ValueError):
                started_at = datetime.fromisoformat(data["started_at"])

        return EvalRunMeta(
            run_id=data.get("run_id", "unknown"),
            started_at=started_at,
            queries_path=data.get("queries_path"),
            top_k=data.get("top_k", 10),
            keep_k=data.get("keep_k"),
            token_budget=data.get("token_budget", 1500),
            run_generation=data.get("run_generation", False),
            use_llm_judge=data.get("use_llm_judge", False),
            judge_model=data.get("judge_model"),
            generator_model=data.get("generator_model"),
            embedder_model=data.get("embedder_model"),
            reranker_name=data.get("reranker_name"),
            notes=data.get("notes"),
            gold_judge_version=data.get("gold_judge_version"),
            groundedness_judge_version=data.get("groundedness_judge_version"),
            extra=data.get("extra"),
        )

    def _parse_aggregates(self, data: dict[str, Any]) -> EvalAggregates:
        """Parse EvalAggregates from metrics.json."""
        overall = self._parse_retrieval_summary(data.get("overall", {}))

        by_type: dict[str, RetrievalSummary] = {}
        for type_name, type_data in data.get("by_type", {}).items():
            by_type[type_name] = self._parse_retrieval_summary(type_data)

        by_difficulty: dict[str, RetrievalSummary] = {}
        for diff_name, diff_data in data.get("by_difficulty", {}).items():
            by_difficulty[diff_name] = self._parse_retrieval_summary(diff_data)

        return EvalAggregates(
            overall=overall,
            by_type=by_type,
            by_difficulty=by_difficulty,
            answer_quality=data.get("answer_quality"),
            latency_ms=data.get("latency_ms"),
        )

    def _parse_retrieval_summary(self, data: dict[str, Any]) -> RetrievalSummary:
        """Parse RetrievalSummary from flat dict format."""
        return RetrievalSummary.from_dict(data)

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
        """Parse EvalResult from JSONL row."""
        # Parse retrieval result
        retrieval_data = data.get("retrieval", {})
        retrieval = RetrievalResult(
            qid=data["qid"],
            retrieved_chunk_ids=tuple(retrieval_data.get("retrieved_chunk_ids", [])),
            relevant_chunk_ids=set(retrieval_data.get("relevant_chunk_ids", [])),
        )

        # Parse answer if present
        answer = None
        answer_data = data.get("answer")
        if answer_data:
            citations = []
            for cit_data in answer_data.get("citations", []):
                citations.append(
                    Citation(
                        chunk_id=cit_data.get("chunk_id", ""),
                        doc_id=cit_data.get("doc_id", ""),
                        uri=cit_data.get("uri"),
                        quote=cit_data.get("quote"),
                        section_heading=cit_data.get("section_heading"),
                        section_path=cit_data.get("section_path"),
                        start_char=cit_data.get("start_char"),
                        end_char=cit_data.get("end_char"),
                        metadata=cit_data.get("metadata", {}),
                    )
                )
            answer = Answer(
                query=data.get("query", ""),
                text=answer_data.get("text", ""),
                abstained=answer_data.get("abstained", False),
                citations=tuple(citations),
                confidence=answer_data.get("confidence"),
                metadata=answer_data.get("metadata", {}),
            )

        # Parse answer metrics if present
        answer_metrics = None
        metrics_data = data.get("answer_metrics")
        if metrics_data:
            answer_metrics = AnswerQualityMetrics(
                semantic_similarity=metrics_data.get("semantic_similarity"),
                correctness=metrics_data.get("correctness"),
                completeness=metrics_data.get("completeness"),
                relevance=metrics_data.get("relevance"),
                hallucination_severity=metrics_data.get("hallucination_severity"),
                is_correct=metrics_data.get("is_correct"),
                is_abstained=metrics_data.get("is_abstained"),
                answerable_from_context=metrics_data.get("answerable_from_context"),
                evidence_bounded=metrics_data.get("evidence_bounded"),
                has_hallucination=metrics_data.get("has_hallucination"),
                supported_claims=metrics_data.get("supported_claims"),
                unsupported_claims=metrics_data.get("unsupported_claims"),
                citation_coverage=metrics_data.get("citation_coverage"),
                answer_length=metrics_data.get("answer_length"),
                num_citations=metrics_data.get("num_citations"),
                quality_score=metrics_data.get("quality_score"),
            )

        # Parse query type and difficulty
        query_type = None
        if data.get("query_type"):
            with contextlib.suppress(ValueError):
                query_type = QueryType(data["query_type"])

        difficulty = None
        if data.get("difficulty"):
            with contextlib.suppress(ValueError):
                difficulty = Difficulty(data["difficulty"])

        return EvalResult(
            qid=data["qid"],
            query=data.get("query", ""),
            retrieval_result=retrieval,
            answer=answer,
            answer_metrics=answer_metrics,
            query_type=query_type,
            difficulty=difficulty,
            is_unanswerable=data.get("is_unanswerable", False),
            latency_ms=data.get("latency_ms"),
            trace_id=data.get("trace_id"),
        )

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
        answer_abstained = None
        citations = None
        answer_data = data.get("answer")
        if answer_data:
            answer_text = answer_data.get("text")
            answer_abstained = answer_data.get("abstained")
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
            answer_abstained=answer_abstained,
            citations=citations,
            latency_ms=data.get("latency_ms"),
            raw_data=data,
        )

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
