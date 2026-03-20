# eval/app_v2/engine/derived/health.py
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import AnalyzedQuery, RunHealthSummary
from rag.eval.models import EvalAggregates


def build_health(
    analyzed: Sequence[AnalyzedQuery],
    aggregates: EvalAggregates,
    verdict_status: Literal["SHIP", "BLOCK"] | None = None,
    worst_slice=None,
) -> RunHealthSummary:
    severity_counter: Counter[Severity] = Counter()
    code_counter: Counter[DiagnosticCode] = Counter()
    latencies: list[float] = []
    quality_scores: list[float] = []

    for aq in analyzed:
        severity_counter[aq.diagnostic.severity] += 1
        code_counter[aq.diagnostic.diagnostic_code] += 1
        if aq.record.latency_ms is not None:
            latencies.append(float(aq.record.latency_ms))
        if aq.record.answer_metrics is not None:
            qs = aq.record.answer_metrics.quality_score
            if qs is not None:
                quality_scores.append(float(qs))

    # Dominant failure = most common non-OK code
    failure_codes = {
        c: n
        for c, n in code_counter.items()
        if c not in (DiagnosticCode.GROUNDED_ANSWER, DiagnosticCode.NO_CLEAR_FAILURE)
    }
    dominant = max(failure_codes, key=failure_codes.get, default=None) if failure_codes else None  # type: ignore[arg-type]

    # Pull values from EvalAggregates
    overall = aggregates.overall
    has_queries = overall.num_queries > 0

    # Core retrieval — use None when no queries were evaluated (metrics were never computed)
    recall_at_10 = overall.recall_at_k.get(10, 0.0)
    ndcg_at_10 = overall.ndcg_at_k.get(10, 0.0)
    mrr: float | None = overall.mrr if has_queries else None
    map_: float | None = overall.map if has_queries else None
    hit_rate_at_10: float | None = overall.hit_rate_at_k.get(10) if has_queries else None
    precision_at_10: float | None = overall.precision_at_k.get(10) if has_queries else None

    critical_recall_at_10: float | None = None
    if overall.critical_recall_at_k:
        critical_recall_at_10 = overall.critical_recall_at_k.get(10)

    weighted_recall_at_10: float | None = None
    if overall.weighted_recall_at_k:
        weighted_recall_at_10 = overall.weighted_recall_at_k.get(10)

    # Answer quality
    quality_dict = aggregates.answer_quality or {}
    avg_correctness: float | None = quality_dict.get("avg_correctness_0_5")
    avg_hallucination_severity: float | None = quality_dict.get("avg_hallucination_severity_0_5")
    avg_citation_coverage: float | None = quality_dict.get("avg_citation_coverage")
    evidence_bounded_rate: float | None = quality_dict.get("evidence_bounded_rate")
    hallucinated_on_unanswerable_rate: float | None = quality_dict.get(
        "hallucinated_on_unanswerable_rate"
    )
    median_quality_score: float | None = quality_dict.get("median_quality_score")
    avg_quality: float | None = quality_dict.get("avg_quality_score")
    # Fall back to per-query mean when the harness aggregate is absent.
    # NOTE: this fallback differs slightly from the harness-computed avg because it
    # averages over AnalyzedQuery records (which may exclude queries dropped during
    # loading), whereas the harness averages over every evaluated query.
    if avg_quality is None and quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)

    # Latency — p50/p95 come only from the harness aggregate (not recomputable here)
    lat_dict = aggregates.latency_ms or {}
    avg_latency: float | None = lat_dict.get("avg")
    if avg_latency is None and latencies:
        # Same caveat as quality fallback above — approximate, not identical to harness.
        avg_latency = sum(latencies) / len(latencies)
    p50_latency: float | None = lat_dict.get("p50")
    p95_latency: float | None = lat_dict.get("p95")

    # Compression efficiency — derived from savings_pct_avg (ratio = 1 - savings_pct)
    compression_dict = aggregates.compression or {}
    avg_compression_ratio: float | None = None
    if compression_dict:
        savings = compression_dict.get("savings_pct_avg")
        if savings is not None:
            avg_compression_ratio = 1.0 - savings

    return RunHealthSummary(
        headline_recall_at_10=recall_at_10,
        headline_ndcg_at_10=ndcg_at_10,
        avg_quality_score=avg_quality,
        avg_latency_ms=avg_latency,
        severity_counts=dict(severity_counter),
        diagnostic_counts=dict(code_counter),
        dominant_failure_mode=dominant,
        dominant_failure_summary=str(dominant) if dominant else None,
        worst_slice=worst_slice,
        verdict_status=verdict_status,
        # Extended retrieval
        headline_mrr=mrr,
        headline_map=map_,
        headline_hit_rate_at_10=hit_rate_at_10,
        headline_precision_at_10=precision_at_10,
        headline_critical_recall_at_10=critical_recall_at_10,
        headline_weighted_recall_at_10=weighted_recall_at_10,
        # Answer quality
        avg_correctness=avg_correctness,
        avg_hallucination_severity=avg_hallucination_severity,
        avg_citation_coverage=avg_citation_coverage,
        evidence_bounded_rate=evidence_bounded_rate,
        hallucinated_on_unanswerable_rate=hallucinated_on_unanswerable_rate,
        median_quality_score=median_quality_score,
        # Latency percentiles
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        # Compression
        avg_compression_ratio=avg_compression_ratio,
    )
