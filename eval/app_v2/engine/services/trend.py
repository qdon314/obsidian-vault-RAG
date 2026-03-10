# eval/app_v2/engine/services/trend.py
from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence

from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import (
    ConfigChangeEvent,
    ConfigFieldChange,
    RunBundle,
    TrendBundle,
)

_DEFAULT_TRACKED_FIELDS: frozenset[str] = frozenset({
    "retriever",
    "index_name",
    "reranker_model",
    "reranker_top_n",
    "generator_model",
    "embedder_model",
    "top_k",
    "token_budget",
})


def detect_config_change_events(
    runs: Sequence[RunBundle],
    tracked_fields: set[str] | None = None,
) -> tuple[ConfigChangeEvent, ...]:
    """Return a ConfigChangeEvent for each adjacent pair of runs where tracked config fields differ."""
    if len(runs) < 2:
        return ()
    fields = tracked_fields if tracked_fields is not None else _DEFAULT_TRACKED_FIELDS
    events: list[ConfigChangeEvent] = []
    for prev, curr in itertools.pairwise(runs):
        prev_cfg = dataclasses.asdict(prev.config)
        curr_cfg = dataclasses.asdict(curr.config)
        changes = tuple(
            ConfigFieldChange(field_name=f, before=prev_cfg.get(f), after=curr_cfg.get(f))
            for f in sorted(fields)
            if prev_cfg.get(f) != curr_cfg.get(f)
        )
        if changes:
            events.append(ConfigChangeEvent(
                from_run_id=prev.run_id,
                to_run_id=curr.run_id,
                timestamp=curr.timestamp,
                changes=changes,
            ))
    return tuple(events)


def build_trend_bundle(runs: Sequence[RunBundle]) -> TrendBundle:
    """Assemble a TrendBundle from a collection of RunBundles, sorted by timestamp."""
    sorted_runs = tuple(sorted(runs, key=lambda r: r.timestamp))
    timestamps = tuple(r.timestamp for r in sorted_runs)

    metric_series: dict[str, tuple[float | None, ...]] = {
        "recall@10": tuple(r.health.headline_recall_at_10 for r in sorted_runs),
        "ndcg@10": tuple(r.health.headline_ndcg_at_10 for r in sorted_runs),
        "avg_latency_ms": tuple(r.health.avg_latency_ms for r in sorted_runs),
        "avg_quality_score": tuple(r.health.avg_quality_score for r in sorted_runs),
    }

    diagnostic_rate_series: dict[DiagnosticCode, tuple[float | None, ...]] = {}
    for code in DiagnosticCode:
        rates: list[float | None] = []
        for run in sorted_runs:
            total = sum(run.health.severity_counts.values())
            count = run.health.diagnostic_counts.get(code, 0)
            rates.append(count / total if total > 0 else None)
        diagnostic_rate_series[code] = tuple(rates)

    return TrendBundle(
        runs=sorted_runs,
        timestamps=timestamps,
        metric_series=metric_series,
        diagnostic_rate_series=diagnostic_rate_series,
        verdict_series=tuple(r.health.verdict_status for r in sorted_runs),
        config_change_events=detect_config_change_events(sorted_runs),
    )
