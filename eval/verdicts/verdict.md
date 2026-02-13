## Eval Verdict: BLOCK

**Run:** 2757c8f705ce490389d9dad991fe6514 | **Baseline:** fb11c3c9c6304008bbeeb223c0075b8b
**Dataset:** eval/datasets/generated_queries.jsonl

### Threshold Checks

| Check | Result | Current | Threshold | Baseline |
|---|---|---|---|---|
| recall@10 >= min_recall_at_10 | FAIL | 0.4600 | 0.6000 | 0.7400 |
| ndcg@10 >= min_ndcg_at_10 | FAIL | 0.3449 | 0.5000 | 0.6658 |
| mrr >= min_mrr | FAIL | 0.3327 | 0.4000 | 0.6697 |
| avg_hallucination_severity <= max_avg_hallucination_severity | PASS | 0.2889 | 2.5000 | 0.1778 |
| evidence_bounded_rate >= min_evidence_bounded_rate | PASS | 0.9200 | 0.7000 | 0.9200 |
| latency_p95_ms <= max_latency_p95_ms | PASS | 4600.2500 | 5000.0000 | 5613.2000 |
| unsafe_miss_rate <= max_unsafe_miss_rate | PASS | 0.0222 | 0.1000 | - |
| abstain_bad_rate <= max_abstain_bad_rate | PASS | 0.0222 | 0.1000 | - |

### Outcome Distribution

| Outcome | Count | Rate |
|---|---|---|
| success_grounded | 28 | 62.2% |
| success_ungrounded | 1 | 2.2% |
| safe_miss | 10 | 22.2% |
| unsafe_miss | 1 | 2.2% |
| abstain_ok | 4 | 8.9% |
| abstain_bad | 1 | 2.2% |

### Regressions

| Metric | Baseline | Current | Delta |
|---|---|---|---|
| recall@10 | 0.7400 | 0.4600 | 0.2800 |

### Rationale

3 threshold checks failed and 1 regressions exceeded tolerance.
