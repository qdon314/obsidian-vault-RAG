# eval/app_v2/engine/derived/health.py
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import AnalyzedQuery, RunHealthSummary


def build_health(
    analyzed: Sequence[AnalyzedQuery],
    recall_at_10: float,
    ndcg_at_10: float,
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
        if aq.record.answer_metrics is not None and hasattr(aq.record.answer_metrics, "quality_score"):
            qs = aq.record.answer_metrics.quality_score
            if qs is not None:
                quality_scores.append(float(qs))

    # Dominant failure = most common non-OK code
    failure_codes = {
        c: n for c, n in code_counter.items()
        if c not in (DiagnosticCode.GROUNDED_ANSWER, DiagnosticCode.NO_CLEAR_FAILURE)
    }
    dominant = max(failure_codes, key=failure_codes.get, default=None) if failure_codes else None  # type: ignore[arg-type]

    return RunHealthSummary(
        headline_recall_at_10=recall_at_10,
        headline_ndcg_at_10=ndcg_at_10,
        avg_quality_score=sum(quality_scores) / len(quality_scores) if quality_scores else None,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else None,
        severity_counts=dict(severity_counter),
        diagnostic_counts=dict(code_counter),
        dominant_failure_mode=dominant,
        dominant_failure_summary=str(dominant) if dominant else None,
        worst_slice=worst_slice,
        verdict_status=verdict_status,
    )
