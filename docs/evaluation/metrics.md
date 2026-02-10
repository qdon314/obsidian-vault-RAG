# Metrics Reference

This document describes the retrieval and answer-quality metrics used by the evaluation harness.

## Retrieval Metrics

Implemented in `src/rag/eval/metrics.py`.

### Recall@k

Fraction of relevant chunk IDs retrieved in top-k.

### Precision@k

Fraction of top-k results that are relevant.

### Hit Rate@k

Binary success metric: at least one relevant hit in top-k.

### MRR

Mean reciprocal rank of first relevant result.

### MAP

Mean average precision across queries.

### NDCG@k

Rank-sensitive relevance score normalized by ideal ranking.

### Tiered Relevance Metrics

For datasets that use critical/supporting/context relevance tiers:

- `critical_recall@k`
- `critical_hit_rate@k`
- `weighted_recall@k`

## Answer Quality Metrics

Implemented in `src/rag/eval/answer_metrics.py`.

### Core fields

- `correctness` (0-5, higher is better)
- `completeness` (0-5, higher is better)
- `relevance` (0-5, higher is better)
- `hallucination_severity` (0-5, lower is better)
- `semantic_similarity` (0-1, optional)
- `citation_coverage` (0-1, derived when supported/unsupported claims exist)
- `quality_score` (0-1, reducer output)

### Groundedness-related fields

- `answerable_from_context` (bool)
- `evidence_bounded` (bool)
- `supported_claims` (int)
- `unsupported_claims` (int)
- `has_hallucination` (bool)

## Aggregates

`EvalAggregates` stores:

- `overall` (`RetrievalSummary`)
- `by_type` (`dict[str, RetrievalSummary]`)
- `by_difficulty` (`dict[str, RetrievalSummary]`)
- `answer_quality` (`dict[str, float] | None`)
- `latency_ms` (`dict[str, float] | None`)

## Programmatic Example

```python
from rag.eval.metrics import summarize

summary = summarize(results=retrieval_results, ks=(1, 3, 5, 10))
print(summary.recall_at_k[10])
print(summary.mrr)
```

## Selection Guide

- Coverage-first: optimize `recall@k`
- Ranking quality: optimize `ndcg@k` and `mrr`
- Safety/groundedness: monitor `hallucination_severity`, `evidence_bounded_rate`
- End-to-end quality: track `quality_score` and judge sub-scores
