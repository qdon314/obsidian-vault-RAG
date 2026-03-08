# eval/app_v2/engine/services/comparison.py
from __future__ import annotations

from eval.app_v2.engine.domain.enums import (
    ComparisonClassification,
    DeltaDirection,
    Severity,
)
from eval.app_v2.engine.domain.models import (
    AnalyzedQuery,
    ComparedQuery,
    ComparisonBundle,
    QueryDeltaSummary,
    QueryDiagnostic,
    RunBundle,
)

# Materiality thresholds
RETRIEVAL_DELTA_THRESHOLD = 0.05   # recall or ndcg delta to count as material
LATENCY_DELTA_MS_THRESHOLD = 100.0
LATENCY_DELTA_PCT_THRESHOLD = 0.10

# Severity ordering for delta classification
_SEV_ORDER = {Severity.OK: 0, Severity.MINOR: 1, Severity.MODERATE: 2, Severity.CRITICAL: 3}


def _severity_direction(before: Severity, after: Severity) -> DeltaDirection:
    diff = _SEV_ORDER[after] - _SEV_ORDER[before]
    if diff < 0:
        return DeltaDirection.IMPROVED
    if diff > 0:
        return DeltaDirection.REGRESSED
    return DeltaDirection.UNCHANGED


def compare_diagnostics(
    *,
    diag_before: QueryDiagnostic | None,
    diag_after: QueryDiagnostic | None,
    recall_before: float | None = None,
    recall_after: float | None = None,
    ndcg_before: float | None = None,
    ndcg_after: float | None = None,
    latency_before: int | None = None,
    latency_after: int | None = None,
) -> QueryDeltaSummary:
    # Retrieval direction
    if recall_before is not None and recall_after is not None:
        delta = recall_after - recall_before
        if abs(delta) < RETRIEVAL_DELTA_THRESHOLD:
            ret_dir = DeltaDirection.UNCHANGED
        elif delta > 0:
            ret_dir = DeltaDirection.IMPROVED
        else:
            ret_dir = DeltaDirection.REGRESSED
    else:
        ret_dir = DeltaDirection.INSUFFICIENT

    # Groundedness direction (severity is a proxy if no direct groundedness delta)
    if diag_before and diag_after:
        sev_dir = _severity_direction(diag_before.severity, diag_after.severity)
        gnd_dir = sev_dir  # simplified: severity captures grounding degradation
    else:
        sev_dir = gnd_dir = DeltaDirection.INSUFFICIENT

    # Latency direction
    if latency_before is not None and latency_after is not None:
        lat_delta_ms = latency_after - latency_before
        lat_delta_pct = abs(lat_delta_ms) / max(latency_before, 1)
        if abs(lat_delta_ms) < LATENCY_DELTA_MS_THRESHOLD and lat_delta_pct < LATENCY_DELTA_PCT_THRESHOLD:
            lat_dir = DeltaDirection.UNCHANGED
        elif lat_delta_ms < 0:
            lat_dir = DeltaDirection.IMPROVED
        else:
            lat_dir = DeltaDirection.REGRESSED
    else:
        lat_dir = DeltaDirection.INSUFFICIENT

    return QueryDeltaSummary(
        retrieval=ret_dir,
        groundedness=gnd_dir,
        latency=lat_dir,
        severity=sev_dir,
    )


def classify_compared_query(
    delta: QueryDeltaSummary,
    *,
    diag_after: QueryDiagnostic | None = None,
) -> ComparisonClassification:
    dims = [delta.retrieval, delta.groundedness, delta.latency, delta.severity]
    material = [d for d in dims if d != DeltaDirection.INSUFFICIENT]

    if not material:
        return ComparisonClassification.INSUFFICIENT_DATA

    improvements = [d for d in material if d == DeltaDirection.IMPROVED]
    regressions  = [d for d in material if d == DeltaDirection.REGRESSED]

    # Severity override: CRITICAL regression dominates
    if diag_after and diag_after.severity == Severity.CRITICAL and delta.severity == DeltaDirection.REGRESSED:
        return ComparisonClassification.REGRESSED

    if not improvements and not regressions:
        return ComparisonClassification.UNCHANGED
    if improvements and not regressions:
        return ComparisonClassification.IMPROVED
    if regressions and not improvements:
        return ComparisonClassification.REGRESSED
    return ComparisonClassification.MIXED


def _index_queries(bundle: RunBundle) -> dict[str, AnalyzedQuery]:
    return {aq.record.qid: aq for aq in bundle.queries}


def build_comparison(run_a: RunBundle, run_b: RunBundle) -> ComparisonBundle:
    """Compare run_b against run_a (b = after, a = before)."""
    index_a = _index_queries(run_a)
    index_b = _index_queries(run_b)
    all_qids = sorted(set(index_a) | set(index_b))

    compared: list[ComparedQuery] = []
    for qid in all_qids:
        aq_a = index_a.get(qid)
        aq_b = index_b.get(qid)

        recall_a = aq_a.record.per_query_recall_at_k.get(10) if aq_a else None
        recall_b = aq_b.record.per_query_recall_at_k.get(10) if aq_b else None
        ndcg_a   = aq_a.record.per_query_ndcg_at_k.get(10)   if aq_a else None
        ndcg_b   = aq_b.record.per_query_ndcg_at_k.get(10)   if aq_b else None
        lat_a    = aq_a.record.latency_ms if aq_a else None
        lat_b    = aq_b.record.latency_ms if aq_b else None

        delta_summary = compare_diagnostics(
            diag_before=aq_a.diagnostic if aq_a else None,
            diag_after=aq_b.diagnostic if aq_b else None,
            recall_before=recall_a, recall_after=recall_b,
            ndcg_before=ndcg_a, ndcg_after=ndcg_b,
            latency_before=lat_a, latency_after=lat_b,
        )
        classification = classify_compared_query(
            delta_summary,
            diag_after=aq_b.diagnostic if aq_b else None,
        )

        query_text = (aq_b or aq_a).record.query if (aq_b or aq_a) else "" # type: ignore

        compared.append(ComparedQuery(
            qid=qid,
            query=query_text,
            retrieval_delta=recall_b - recall_a if (recall_a is not None and recall_b is not None) else None,
            ndcg_delta=ndcg_b - ndcg_a if (ndcg_a is not None and ndcg_b is not None) else None,
            latency_delta_ms=float(lat_b - lat_a) if (lat_a is not None and lat_b is not None) else None,
            quality_delta=None,  # extend when quality score is available on both sides
            diagnostic_before=aq_a.diagnostic if aq_a else None,
            diagnostic_after=aq_b.diagnostic if aq_b else None,
            delta_summary=delta_summary,
            classification=classification,
        ))

    # Aggregate deltas
    agg_a = run_a.aggregates.overall
    agg_b = run_b.aggregates.overall
    agg_deltas: dict[str, float | None] = {}
    for k in (5, 10):
        r_a = agg_a.recall_at_k.get(k)
        r_b = agg_b.recall_at_k.get(k)
        agg_deltas[f"recall@{k}"] = r_b - r_a if (r_a is not None and r_b is not None) else None
        n_a = agg_a.ndcg_at_k.get(k)
        n_b = agg_b.ndcg_at_k.get(k)
        agg_deltas[f"ndcg@{k}"] = n_b - n_a if (n_a is not None and n_b is not None) else None

    return ComparisonBundle(
        run_a=run_a,
        run_b=run_b,
        aggregate_deltas=agg_deltas,
        slice_deltas=None,  # extend in Phase 5
        compared_queries=tuple(compared),
    )
