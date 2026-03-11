# eval/app_v2/engine/loaders/bundle.py
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.derived.health import build_health
from eval.app_v2.engine.derived.slices import build_slice_table
from eval.app_v2.engine.domain.models import (
    QueryRecord,
    QueryTrace,
    RunBundle,
    RunConfig,
    VerdictSummary,
)
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.registry import DEFAULT_LOADERS
from rag.eval.models import EvalAggregates, EvalResult, EvalRunMeta

_RUN_DIR_PATTERN = re.compile(r"run_(\d{4}_\d{2}_\d{2}T\d{2}-\d{2})")


def _parse_timestamp(dirname: str) -> datetime:
    m = _RUN_DIR_PATTERN.match(dirname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y_%m_%dT%H-%M").replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.now(UTC)


def _normalize_config(meta: EvalRunMeta) -> RunConfig:
    return RunConfig(
        retriever=meta.extra.get("retriever_class") if meta.extra else None,
        index_name=meta.index_name,
        reranker_model=meta.reranker_name,
        reranker_top_n=meta.keep_k,
        generator_model=meta.generator_model,
        embedder_model=meta.embedder_model,
        top_k=meta.top_k,
        token_budget=meta.token_budget,
    )


def _build_query_record(
    result: EvalResult,
    traces: dict[str, QueryTrace],
) -> QueryRecord:
    rr = result.retrieval_result
    trace = traces.get(result.trace_id) if result.trace_id else None

    reranked = trace.reranked_chunk_ids if trace else None
    packed = trace.packed_chunk_ids if trace else None

    relevant = frozenset(rr.relevant_chunk_ids)
    retrieved = tuple(rr.retrieved_chunk_ids)
    retrieved_set = frozenset(retrieved[:10])
    hits_at_10 = relevant & retrieved_set
    recall_10 = len(hits_at_10) / len(relevant) if relevant else 0.0
    hit_rate_10 = 1.0 if hits_at_10 else 0.0

    return QueryRecord(
        qid=result.qid,
        query=result.query,
        query_type=result.query_type.value if result.query_type else None,
        difficulty=result.difficulty.value if result.difficulty else None,
        is_unanswerable=result.is_unanswerable,
        requires_synthesis=False,
        tags=(),
        relevant_chunk_ids=relevant,
        retrieved_chunk_ids=retrieved,
        reranked_chunk_ids=reranked,
        packed_chunk_ids=packed,
        per_query_recall_at_k={10: recall_10},
        per_query_precision_at_k={10: len(hits_at_10) / 10 if retrieved else 0.0},
        per_query_ndcg_at_k={
            10: recall_10
        },  # simplified; replace with ndcg from result if available
        per_query_hit_rate_at_k={10: hit_rate_10},
        answer_text=result.answer.text if result.answer else None,
        answer_metrics=result.answer_metrics,
        groundedness=result.groundedness_result,
        latency_ms=result.latency_ms,
        trace_id=result.trace_id,
        trace=trace,
    )


def build_bundle(run_dir: Path) -> RunBundle:
    run_dir = run_dir.resolve()
    all_warnings: list[BundleWarning] = []
    raw_artifacts: dict[str, object] = {}

    # Run all loaders
    artifacts = {}
    for loader in DEFAULT_LOADERS:
        artifact = loader.load(run_dir) if loader.can_load(run_dir) else None
        if artifact is None:
            continue
        artifacts[loader.artifact_name] = artifact
        all_warnings.extend(artifact.warnings)
        raw_artifacts[loader.artifact_name] = artifact.payload

    # Unpack
    metrics_artifact = artifacts.get("metrics.json")
    results_artifact = artifacts.get("results.jsonl")
    traces_artifact = artifacts.get("traces.jsonl")
    verdict_artifact = artifacts.get("verdict.json")

    aggregates: EvalAggregates | None = None
    meta: EvalRunMeta | None = None
    if metrics_artifact and metrics_artifact.payload:
        aggregates, meta = metrics_artifact.payload

    results: tuple[EvalResult, ...] = ()
    if results_artifact and results_artifact.payload:
        results = results_artifact.payload

    traces: dict[str, QueryTrace] = {}
    if traces_artifact and traces_artifact.payload:
        traces = traces_artifact.payload
    elif results:
        all_warnings.append(
            BundleWarning(
                code=BundleWarningCode.MISSING_TRACES,
                message="traces.jsonl not found; pipeline drill-down unavailable",
            )
        )

    verdict_summary: VerdictSummary | None = None
    if verdict_artifact and verdict_artifact.payload:
        verdict_summary = verdict_artifact.payload

    # Fall back to defaults if meta/aggregates missing
    if meta is None:
        meta = EvalRunMeta()
    if aggregates is None:
        from rag.eval.models import RetrievalSummary

        aggregates = EvalAggregates(overall=RetrievalSummary(num_queries=0, avg_retrieved=0.0))

    records = [_build_query_record(r, traces) for r in results]
    analyzed = analyze_queries(records)

    slice_table = build_slice_table(analyzed, group_by=["query_type", "difficulty"])
    worst_slice = slice_table.rows[0].key if slice_table.rows else None

    verdict_flag = verdict_summary.decision if verdict_summary else None
    health = build_health(
        analyzed, aggregates, verdict_status=verdict_flag, worst_slice=worst_slice
    )

    # Warn when MRR is zero yet at least one query retrieved something at rank 1.
    # MRR=0 while recall@1>0 is a logical contradiction (you found something first,
    # so reciprocal rank must be 1.0), which indicates an incomplete artifact.
    # We deliberately do NOT warn when all queries genuinely missed — that is a valid 0.
    recall_at_1 = aggregates.overall.recall_at_k.get(1, 0.0)
    if aggregates.overall.num_queries > 0 and aggregates.overall.mrr == 0.0 and recall_at_1 > 0.0:
        all_warnings.append(
            BundleWarning(
                code=BundleWarningCode.HEALTH_PARTIAL,
                message=(
                    "MRR is 0.0 but recall@1 > 0; metrics artifact may be incomplete "
                    f"(recall@1={recall_at_1:.3f})"
                ),
                artifact_name="metrics.json",
            )
        )

    run_id = meta.run_id or run_dir.name
    timestamp = _parse_timestamp(run_dir.name)
    config = _normalize_config(meta)

    return RunBundle(
        run_id=run_id,
        display_name=meta.run_name or run_dir.name,
        timestamp=timestamp,
        config=config,
        aggregates=aggregates,
        queries=analyzed,
        health=health,
        verdict=verdict_summary,
        warnings=tuple(all_warnings),
        raw_artifacts=raw_artifacts,
    )
