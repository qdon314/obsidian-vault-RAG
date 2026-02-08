"""Release-gating verdict computation for evaluation runs.

This module sits above raw eval metrics and turns them into:
- threshold pass/fail checks,
- regression flags against an optional baseline,
- a final SHIP/BLOCK decision with renderable reports.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from rag.eval.models import EvalRun
from rag.eval.reducers import OutcomeLabel
from rag.eval.verdict_thresholds import VerdictThresholds


class Decision(StrEnum):
    SHIP = "ship"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class ThresholdCheck:
    """A single pass/fail check against a threshold."""

    name: str
    passed: bool
    current: float
    threshold: float
    baseline: float | None = None


@dataclass(frozen=True, slots=True)
class RegressionFlag:
    """A metric regression beyond tolerance."""

    metric: str
    qid: str | None = None
    query: str | None = None
    baseline_value: float = 0.0
    current_value: float = 0.0
    delta: float = 0.0


@dataclass(frozen=True, slots=True)
class OutcomeBucket:
    """Count and rate for one OutcomeLabel."""

    label: OutcomeLabel
    count: int
    rate: float


@dataclass(frozen=True, slots=True)
class Verdict:
    decision: Decision
    summary: str
    checks: tuple[ThresholdCheck, ...]
    regressions: tuple[RegressionFlag, ...]
    outcome_distribution: tuple[OutcomeBucket, ...]
    current_run_id: str
    baseline_run_id: str | None
    dataset_name: str | None
    created_at: datetime


def _safe_value(value: float | None, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _format_pct(value: float) -> str:
    return f"{(value * 100):.1f}%"


def _compute_outcome_distribution(run: EvalRun) -> tuple[OutcomeBucket, ...]:
    # Only judged rows can have an outcome label; unlabeled rows are excluded from rates.
    labels = [r.outcome_label for r in run.results if r.outcome_label is not None]
    counts = Counter(labels)
    total = sum(counts.values())
    buckets: list[OutcomeBucket] = []

    # Emit a stable full taxonomy ordering, including zero-count buckets.
    for label in OutcomeLabel:
        count = int(counts.get(label, 0))
        rate = (count / total) if total else 0.0
        buckets.append(OutcomeBucket(label=label, count=count, rate=rate))

    return tuple(buckets)


def _outcome_rate(distribution: tuple[OutcomeBucket, ...], label: OutcomeLabel) -> float:
    for bucket in distribution:
        if bucket.label == label:
            return bucket.rate
    return 0.0


def compute_verdict(
    current: EvalRun,
    baseline: EvalRun | None,
    thresholds: VerdictThresholds,
) -> Verdict:
    overall = current.aggregates.overall
    answer_quality = current.aggregates.answer_quality or {}
    latency = current.aggregates.latency_ms or {}
    baseline_overall = baseline.aggregates.overall if baseline else None
    baseline_answer_quality = (baseline.aggregates.answer_quality or {}) if baseline else {}
    baseline_latency = (baseline.aggregates.latency_ms or {}) if baseline else {}

    outcome_distribution = _compute_outcome_distribution(current)
    # Behavioral blockers are inferred from outcome distribution, not aggregate metrics.
    unsafe_miss_rate = _outcome_rate(outcome_distribution, OutcomeLabel.UNSAFE_MISS)
    abstain_bad_rate = _outcome_rate(outcome_distribution, OutcomeLabel.ABSTAIN_BAD)

    # --- Build checks list: always include retrieval, conditionally include answer/latency ---
    checks: list[ThresholdCheck] = [
        # recall@10 check (always present)
        ThresholdCheck(
            name="recall@10 >= min_recall_at_10",
            passed=_safe_value(overall.recall_at_k.get(10)) >= thresholds.min_recall_at_10,
            current=_safe_value(overall.recall_at_k.get(10)),
            threshold=thresholds.min_recall_at_10,
            baseline=(
                _safe_value(baseline_overall.recall_at_k.get(10))
                if baseline_overall
                else None
            ),
        ),
        # ndcg@10 check (always present)
        ThresholdCheck(
            name="ndcg@10 >= min_ndcg_at_10",
            passed=_safe_value(overall.ndcg_at_k.get(10)) >= thresholds.min_ndcg_at_10,
            current=_safe_value(overall.ndcg_at_k.get(10)),
            threshold=thresholds.min_ndcg_at_10,
            baseline=(_safe_value(baseline_overall.ndcg_at_k.get(10)) if baseline_overall else None),
        ),
        # mrr check (always present)
        ThresholdCheck(
            name="mrr >= min_mrr",
            passed=_safe_value(overall.mrr) >= thresholds.min_mrr,
            current=_safe_value(overall.mrr),
            threshold=thresholds.min_mrr,
            baseline=_safe_value(baseline_overall.mrr) if baseline_overall else None,
        ),
    ]

    # Answer quality checks: only when judge data is present.
    halluc_val = answer_quality.get("avg_hallucination_severity_0_5")
    if halluc_val is not None:
        checks.append(
            ThresholdCheck(
                name="avg_hallucination_severity <= max_avg_hallucination_severity",
                passed=float(halluc_val) <= thresholds.max_avg_hallucination_severity,
                current=float(halluc_val),
                threshold=thresholds.max_avg_hallucination_severity,
                baseline=_safe_value(baseline_answer_quality.get("avg_hallucination_severity_0_5"))
                if baseline
                else None,
            )
        )

    eb_val = answer_quality.get("evidence_bounded_rate")
    if eb_val is not None:
        checks.append(
            ThresholdCheck(
                name="evidence_bounded_rate >= min_evidence_bounded_rate",
                passed=float(eb_val) >= thresholds.min_evidence_bounded_rate,
                current=float(eb_val),
                threshold=thresholds.min_evidence_bounded_rate,
                baseline=_safe_value(baseline_answer_quality.get("evidence_bounded_rate"))
                if baseline
                else None,
            )
        )

    # Latency check: only when pipeline latency was measured.
    p95_val = latency.get("p95")
    if p95_val is not None:
        checks.append(
            ThresholdCheck(
                name="latency_p95_ms <= max_latency_p95_ms",
                passed=float(p95_val) <= thresholds.max_latency_p95_ms,
                current=float(p95_val),
                threshold=thresholds.max_latency_p95_ms,
                baseline=_safe_value(baseline_latency.get("p95")) if baseline else None,
            )
        )

    # Behavioral outcome checks: only when outcome labels were computed.
    has_outcomes = any(r.outcome_label is not None for r in current.results)
    if has_outcomes:
        checks.append(
            ThresholdCheck(
                name="unsafe_miss_rate <= max_unsafe_miss_rate",
                passed=unsafe_miss_rate <= thresholds.max_unsafe_miss_rate,
                current=unsafe_miss_rate,
                threshold=thresholds.max_unsafe_miss_rate,
                baseline=None,
            )
        )
        checks.append(
            ThresholdCheck(
                name="abstain_bad_rate <= max_abstain_bad_rate",
                passed=abstain_bad_rate <= thresholds.max_abstain_bad_rate,
                current=abstain_bad_rate,
                threshold=thresholds.max_abstain_bad_rate,
                baseline=None,
            )
        )

    regressions: list[RegressionFlag] = []
    if baseline_overall is not None:
        # Regression checks are directional:
        # - lower recall/quality is worse
        # - higher latency is worse
        current_recall = _safe_value(overall.recall_at_k.get(10))
        baseline_recall = _safe_value(baseline_overall.recall_at_k.get(10))
        recall_drop = baseline_recall - current_recall
        if recall_drop > thresholds.max_recall_regression:
            regressions.append(
                RegressionFlag(
                    metric="recall@10",
                    baseline_value=baseline_recall,
                    current_value=current_recall,
                    delta=recall_drop,
                )
            )

        current_quality = _safe_value(answer_quality.get("avg_quality_score"))
        baseline_quality = _safe_value(baseline_answer_quality.get("avg_quality_score"))
        quality_drop = baseline_quality - current_quality
        if quality_drop > thresholds.max_quality_regression:
            regressions.append(
                RegressionFlag(
                    metric="avg_quality_score",
                    baseline_value=baseline_quality,
                    current_value=current_quality,
                    delta=quality_drop,
                )
            )

        current_p95 = _safe_value(latency.get("p95"))
        baseline_p95 = _safe_value(baseline_latency.get("p95"))
        latency_increase = current_p95 - baseline_p95
        if latency_increase > thresholds.max_latency_regression_ms:
            regressions.append(
                RegressionFlag(
                    metric="latency_p95_ms",
                    baseline_value=baseline_p95,
                    current_value=current_p95,
                    delta=latency_increase,
                )
            )

    # Gate blocks if either absolute checks fail or regression tolerances are exceeded.
    failed_checks = [check for check in checks if not check.passed]
    decision = Decision.BLOCK if failed_checks or regressions else Decision.SHIP
    if decision is Decision.SHIP:
        summary = f"All {len(checks)} threshold checks passed. No regressions detected."
    else:
        summary = (
            f"{len(failed_checks)} threshold checks failed and {len(regressions)} regressions "
            "exceeded tolerance."
        )

    return Verdict(
        decision=decision,
        summary=summary,
        checks=tuple(checks),
        regressions=tuple(regressions),
        outcome_distribution=outcome_distribution,
        current_run_id=current.meta.run_id,
        baseline_run_id=baseline.meta.run_id if baseline else None,
        dataset_name=current.meta.queries_path,
        created_at=datetime.now(UTC),
    )


def verdict_to_dict(verdict: Verdict) -> dict[str, object]:
    # Keep serialization schema explicit for stable machine consumption.
    return {
        "decision": verdict.decision.value,
        "summary": verdict.summary,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "current": c.current,
                "threshold": c.threshold,
                "baseline": c.baseline,
            }
            for c in verdict.checks
        ],
        "regressions": [
            {
                "metric": r.metric,
                "qid": r.qid,
                "query": r.query,
                "baseline_value": r.baseline_value,
                "current_value": r.current_value,
                "delta": r.delta,
            }
            for r in verdict.regressions
        ],
        "outcome_distribution": [
            {"label": o.label.value, "count": o.count, "rate": o.rate}
            for o in verdict.outcome_distribution
        ],
        "current_run_id": verdict.current_run_id,
        "baseline_run_id": verdict.baseline_run_id,
        "dataset_name": verdict.dataset_name,
        "created_at": verdict.created_at.isoformat(),
    }


def verdict_from_dict(payload: dict[str, Any]) -> Verdict:
    # Deserialize from the external JSON contract into typed domain objects.
    checks = tuple(
        ThresholdCheck(
            name=str(item["name"]),
            passed=bool(item["passed"]),
            current=float(item["current"]),
            threshold=float(item["threshold"]),
            baseline=float(item["baseline"]) if item.get("baseline") is not None else None,
        )
        for item in payload.get("checks", [])
    )
    regressions = tuple(
        RegressionFlag(
            metric=str(item["metric"]),
            qid=str(item["qid"]) if item.get("qid") is not None else None,
            query=str(item["query"]) if item.get("query") is not None else None,
            baseline_value=float(item.get("baseline_value", 0.0)),
            current_value=float(item.get("current_value", 0.0)),
            delta=float(item.get("delta", 0.0)),
        )
        for item in payload.get("regressions", [])
    )
    outcomes = tuple(
        OutcomeBucket(
            label=OutcomeLabel(str(item["label"])),
            count=int(item["count"]),
            rate=float(item["rate"]),
        )
        for item in payload.get("outcome_distribution", [])
    )
    return Verdict(
        decision=Decision(str(payload["decision"])),
        summary=str(payload["summary"]),
        checks=checks,
        regressions=regressions,
        outcome_distribution=outcomes,
        current_run_id=str(payload["current_run_id"]),
        baseline_run_id=(
            str(payload["baseline_run_id"]) if payload.get("baseline_run_id") is not None else None
        ),
        dataset_name=str(payload["dataset_name"]) if payload.get("dataset_name") is not None else None,
        created_at=datetime.fromisoformat(str(payload["created_at"])),
    )


def render_verdict_json(verdict: Verdict) -> str:
    return json.dumps(verdict_to_dict(verdict), indent=2)


def render_verdict_markdown(verdict: Verdict) -> str:
    # Report is intentionally deterministic so CI artifacts diff cleanly.
    lines = [f"## Eval Verdict: {verdict.decision.value.upper()}", ""]
    baseline = verdict.baseline_run_id or "none"
    dataset = verdict.dataset_name or "unknown"
    lines.append(f"**Run:** {verdict.current_run_id} | **Baseline:** {baseline}")
    lines.append(f"**Dataset:** {dataset}")
    lines.append("")
    lines.append("### Threshold Checks")
    lines.append("")
    lines.append("| Check | Result | Current | Threshold | Baseline |")
    lines.append("|---|---|---|---|---|")
    for check in verdict.checks:
        result = "PASS" if check.passed else "FAIL"
        baseline_value = f"{check.baseline:.4f}" if check.baseline is not None else "-"
        lines.append(
            f"| {check.name} | {result} | {check.current:.4f} | {check.threshold:.4f} | {baseline_value} |"
        )

    lines.append("")
    lines.append("### Outcome Distribution")
    lines.append("")
    lines.append("| Outcome | Count | Rate |")
    lines.append("|---|---|---|")
    for bucket in verdict.outcome_distribution:
        lines.append(f"| {bucket.label.value} | {bucket.count} | {_format_pct(bucket.rate)} |")

    lines.append("")
    lines.append("### Regressions")
    lines.append("")
    if not verdict.regressions:
        lines.append("No regressions beyond tolerance.")
    else:
        lines.append("| Metric | Baseline | Current | Delta |")
        lines.append("|---|---|---|---|")
        for regression in verdict.regressions:
            lines.append(
                f"| {regression.metric} | {regression.baseline_value:.4f} | "
                f"{regression.current_value:.4f} | {regression.delta:.4f} |"
            )

    lines.append("")
    lines.append("### Rationale")
    lines.append("")
    lines.append(verdict.summary)
    return "\n".join(lines) + "\n"
