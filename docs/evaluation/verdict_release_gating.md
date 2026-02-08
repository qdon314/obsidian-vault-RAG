# Verdict and Release Gating

This document describes how evaluation results are converted into a release decision (`SHIP` or `BLOCK`).

## Overview

The verdict layer adds a decision step on top of existing eval metrics:

- Absolute threshold checks (retrieval, quality, groundedness, latency)
- Behavioral safety checks from outcome labels
- Regression checks versus an optional baseline run

Implementation lives in:

- `src/rag/eval/verdict.py`
- `src/rag/eval/verdict_thresholds.py`
- `eval/scripts/verdict.py`

## Inputs and Outputs

The verdict script expects a run directory with:

- `metrics.json`
- `results.jsonl`

It writes:

- `verdict.md` (human-readable report)
- `verdict.json` (machine-readable payload)

## Run Locally

Generate verdict artifacts:

```bash
./scripts/py eval/scripts/verdict.py \
  --current eval/runs/latest \
  --baseline eval/runs/baseline \
  --output eval/verdicts
```

Fail the command when decision is `BLOCK`:

```bash
./scripts/py eval/scripts/verdict.py \
  --current eval/runs/latest \
  --baseline eval/runs/baseline \
  --output eval/verdicts \
  --fail-on-block
```

Convenience target:

```bash
make verdict
```

## Decision Criteria

Thresholds are configured in `settings.toml` under `[eval.verdict]`.

Current checks include:

- `recall@10 >= min_recall_at_10`
- `ndcg@10 >= min_ndcg_at_10`
- `mrr >= min_mrr`
- `avg_hallucination_severity <= max_avg_hallucination_severity`
- `evidence_bounded_rate >= min_evidence_bounded_rate`
- `latency_p95_ms <= max_latency_p95_ms`
- `unsafe_miss_rate <= max_unsafe_miss_rate`
- `abstain_bad_rate <= max_abstain_bad_rate`

Regression checks (when baseline is provided):

- recall drop must not exceed `max_recall_regression`
- quality score drop must not exceed `max_quality_regression`
- p95 latency increase must not exceed `max_latency_regression_ms`

Final decision rule:

- `BLOCK` if any threshold check fails or any regression exceeds tolerance
- `SHIP` otherwise

## Baseline Management

Baseline is a normal run directory kept at `eval/runs/baseline`.

Promote a run manually:

```bash
cp -r eval/runs/run_YYYY_MM_DDTHH-MM eval/runs/baseline
```

If no baseline is passed, verdict still runs and enforces absolute/behavioral checks only.

## CI Integration

CI includes an `eval-gate` job in `.github/workflows/ci.yml` that:

1. Runs evaluation on the curated query set
2. Runs verdict with `--fail-on-block`
3. Uploads `eval/verdicts/` as an artifact
