"""Tests for the eval verdict release-gating layer (spec 03)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from rag.eval.models import (
    EvalAggregates,
    EvalResult,
    EvalRun,
    EvalRunMeta,
    RetrievalResult,
    RetrievalSummary,
)
from rag.eval.reducers import OutcomeLabel
from rag.eval.verdict import (
    Decision,
    compute_verdict,
    render_verdict_json,
    render_verdict_markdown,
    verdict_from_dict,
)
from rag.eval.verdict_thresholds import VerdictThresholds


def _make_result(qid: str, label: OutcomeLabel | None) -> EvalResult:
    """Build a minimal per-query result with optional outcome label."""
    return EvalResult(
        qid=qid,
        query=f"query {qid}",
        retrieval_result=RetrievalResult(
            qid=qid,
            retrieved_chunk_ids=("c1", "c2"),
            relevant_chunk_ids={"c1"},
        ),
        outcome_label=label,
    )


def _make_run(
    *,
    run_id: str,
    recall10: float,
    ndcg10: float,
    mrr: float,
    avg_quality: float = 0.8,
    avg_hallucination: float = 0.5,
    evidence_bounded_rate: float = 0.9,
    p95_latency: float = 2000.0,
    labels: list[OutcomeLabel] | None = None,
) -> EvalRun:
    """Build a synthetic EvalRun for threshold/regression scenarios."""
    results = tuple(
        _make_result(f"q{i + 1}", label)
        for i, label in enumerate(labels or [OutcomeLabel.SUCCESS_GROUNDED] * 5)
    )
    return EvalRun(
        meta=EvalRunMeta(
            run_id=run_id,
            started_at=datetime.now(UTC),
            queries_path="eval/datasets/curated_queries.jsonl",
        ),
        results=results,
        aggregates=EvalAggregates(
            overall=RetrievalSummary(
                num_queries=len(results),
                avg_retrieved=2.0,
                recall_at_k={10: recall10},
                ndcg_at_k={10: ndcg10},
                mrr=mrr,
                map=0.0,
            ),
            answer_quality={
                "avg_quality_score": avg_quality,
                "avg_hallucination_severity_0_5": avg_hallucination,
                "evidence_bounded_rate": evidence_bounded_rate,
            },
            latency_ms={"p95": p95_latency},
        ),
    )


def test_verdict_ship_when_all_pass() -> None:
    """All thresholds met and no regressions should produce SHIP."""
    current = _make_run(
        run_id="current",
        recall10=0.80,
        ndcg10=0.70,
        mrr=0.60,
    )
    baseline = _make_run(
        run_id="baseline",
        recall10=0.78,
        ndcg10=0.66,
        mrr=0.58,
        avg_quality=0.77,
        p95_latency=1900.0,
    )

    verdict = compute_verdict(current, baseline, VerdictThresholds())

    assert verdict.decision == Decision.SHIP
    assert all(check.passed for check in verdict.checks)
    assert verdict.regressions == ()


def test_verdict_block_on_low_recall() -> None:
    """Recall below minimum threshold should block release."""
    current = _make_run(run_id="current", recall10=0.20, ndcg10=0.70, mrr=0.60)
    verdict = compute_verdict(current, None, VerdictThresholds(min_recall_at_10=0.60))
    assert verdict.decision == Decision.BLOCK
    assert any(
        (check.name.startswith("recall@10")) and (not check.passed) for check in verdict.checks
    )


def test_verdict_block_on_regression() -> None:
    """Large baseline drop beyond tolerance should block release."""
    current = _make_run(run_id="current", recall10=0.70, ndcg10=0.70, mrr=0.60)
    baseline = _make_run(run_id="baseline", recall10=0.90, ndcg10=0.70, mrr=0.60)

    verdict = compute_verdict(
        current,
        baseline,
        VerdictThresholds(max_recall_regression=0.05),
    )

    assert verdict.decision == Decision.BLOCK
    assert any(r.metric == "recall@10" for r in verdict.regressions)


def test_verdict_block_on_high_unsafe_miss_rate() -> None:
    """Unsafe miss behavioral rate above threshold should block release."""
    labels = [OutcomeLabel.UNSAFE_MISS, OutcomeLabel.SUCCESS_GROUNDED]
    current = _make_run(run_id="current", recall10=0.80, ndcg10=0.70, mrr=0.60, labels=labels)

    verdict = compute_verdict(current, None, VerdictThresholds(max_unsafe_miss_rate=0.10))

    assert verdict.decision == Decision.BLOCK
    unsafe_check = next(c for c in verdict.checks if c.name.startswith("unsafe_miss_rate"))
    assert unsafe_check.current == 0.5
    assert unsafe_check.passed is False


def test_verdict_block_on_high_abstain_bad_rate() -> None:
    """Abstain-bad behavioral rate above threshold should block release."""
    labels = [OutcomeLabel.ABSTAIN_BAD, OutcomeLabel.SUCCESS_GROUNDED]
    current = _make_run(run_id="current", recall10=0.80, ndcg10=0.70, mrr=0.60, labels=labels)

    verdict = compute_verdict(current, None, VerdictThresholds(max_abstain_bad_rate=0.10))

    assert verdict.decision == Decision.BLOCK
    abstain_check = next(c for c in verdict.checks if c.name.startswith("abstain_bad_rate"))
    assert abstain_check.current == 0.5
    assert abstain_check.passed is False


def test_verdict_no_baseline_checks_absolute_only() -> None:
    """No baseline should skip regression checks and only enforce absolutes."""
    current = _make_run(run_id="current", recall10=0.80, ndcg10=0.70, mrr=0.60)
    verdict = compute_verdict(current, None, VerdictThresholds())

    assert verdict.baseline_run_id is None
    assert verdict.regressions == ()


def test_outcome_distribution_computed_correctly() -> None:
    """Outcome bucket counts/rates should match labeled result distribution."""
    labels = [
        OutcomeLabel.SUCCESS_GROUNDED,
        OutcomeLabel.SUCCESS_GROUNDED,
        OutcomeLabel.UNSAFE_MISS,
        OutcomeLabel.ABSTAIN_BAD,
    ]
    current = _make_run(run_id="current", recall10=0.80, ndcg10=0.70, mrr=0.60, labels=labels)

    verdict = compute_verdict(current, None, VerdictThresholds())
    dist = {bucket.label: bucket for bucket in verdict.outcome_distribution}

    assert dist[OutcomeLabel.SUCCESS_GROUNDED].count == 2
    assert dist[OutcomeLabel.UNSAFE_MISS].count == 1
    assert dist[OutcomeLabel.ABSTAIN_BAD].count == 1
    assert dist[OutcomeLabel.UNSAFE_MISS].rate == 0.25


def test_render_markdown_includes_all_sections() -> None:
    """Markdown report should include all major verdict sections."""
    current = _make_run(run_id="current", recall10=0.80, ndcg10=0.70, mrr=0.60)
    verdict = compute_verdict(current, None, VerdictThresholds())

    markdown = render_verdict_markdown(verdict)

    assert "### Threshold Checks" in markdown
    assert "### Outcome Distribution" in markdown
    assert "### Regressions" in markdown
    assert "### Rationale" in markdown


def test_render_json_roundtrips() -> None:
    """JSON representation should deserialize back into an equivalent Verdict."""
    current = _make_run(run_id="current", recall10=0.80, ndcg10=0.70, mrr=0.60)
    verdict = compute_verdict(current, None, VerdictThresholds())

    payload = json.loads(render_verdict_json(verdict))
    roundtripped = verdict_from_dict(payload)

    assert roundtripped.decision == verdict.decision
    assert roundtripped.current_run_id == verdict.current_run_id
    assert len(roundtripped.checks) == len(verdict.checks)


def test_verdict_skips_checks_for_absent_data() -> None:
    """Retrieval-only runs with no answer_quality/latency should omit those checks."""
    results = tuple(_make_result(f"q{i + 1}", None) for i in range(5))
    current = EvalRun(
        meta=EvalRunMeta(
            run_id="retrieval-only",
            started_at=datetime.now(UTC),
            queries_path="eval/datasets/curated_queries.jsonl",
        ),
        results=results,
        aggregates=EvalAggregates(
            overall=RetrievalSummary(
                num_queries=5,
                avg_retrieved=2.0,
                recall_at_k={10: 0.80},
                ndcg_at_k={10: 0.70},
                mrr=0.60,
                map=0.0,
            ),
            answer_quality=None,
            latency_ms=None,
        ),
    )
    verdict = compute_verdict(current, None, VerdictThresholds())

    assert verdict.decision == Decision.SHIP
    check_names = {c.name for c in verdict.checks}
    # Retrieval checks must be present
    assert "recall@10 >= min_recall_at_10" in check_names
    assert "ndcg@10 >= min_ndcg_at_10" in check_names
    assert "mrr >= min_mrr" in check_names
    # Answer quality and latency checks must be absent
    assert "avg_hallucination_severity <= max_avg_hallucination_severity" not in check_names
    assert "evidence_bounded_rate >= min_evidence_bounded_rate" not in check_names
    assert "latency_p95_ms <= max_latency_p95_ms" not in check_names


def test_eval_result_from_results_dict_parses_outcome_label() -> None:
    """Legacy results loader should preserve persisted outcome labels for gating."""
    payload = {
        "qid": "q1",
        "query": "test",
        "retrieval": {"retrieved_chunk_ids": ["c1"], "relevant_chunk_ids": ["c1"]},
        "outcome_label": OutcomeLabel.UNSAFE_MISS.value,
    }

    result = EvalResult.from_results_dict(payload)

    assert result.outcome_label == OutcomeLabel.UNSAFE_MISS
