# Metrics Reference

This document describes all evaluation metrics used in the system.

## Retrieval Metrics

Retrieval metrics measure how well the retriever finds relevant chunks given a query and ground-truth chunk IDs.

### Recall@k

**Definition**: Fraction of relevant chunks that appear in the top-k retrieved results.

```
Recall@k = |Retrieved ∩ Relevant| / |Relevant|
```

**Interpretation**:
- 1.0 = All relevant chunks were retrieved
- 0.0 = No relevant chunks were retrieved
- Higher is better

**Use case**: Primary metric when you want to ensure all relevant information is retrieved before reranking.

```python
from src.rag.eval.metrics import recall_at_k

score = recall_at_k(
    retrieved=["c1", "c2", "c3", "c4", "c5"],
    relevant={"c1", "c3", "c7"},
    k=5,
)
# Returns 0.667 (2 out of 3 relevant chunks found)
```

### Precision@k

**Definition**: Fraction of top-k retrieved results that are relevant.

```
Precision@k = |Retrieved[:k] ∩ Relevant| / k
```

**Interpretation**:
- 1.0 = All top-k results are relevant
- 0.0 = No top-k results are relevant
- Higher is better

**Use case**: Important when context window is limited and irrelevant chunks waste tokens.

```python
from src.rag.eval.metrics import precision_at_k

score = precision_at_k(
    retrieved=["c1", "c2", "c3", "c4", "c5"],
    relevant={"c1", "c3"},
    k=5,
)
# Returns 0.4 (2 relevant in top 5)
```

### Hit Rate@k (Success@k)

**Definition**: Binary indicator of whether at least one relevant chunk is in top-k.

```
Hit@k = 1 if |Retrieved[:k] ∩ Relevant| > 0 else 0
```

**Interpretation**:
- 1 = At least one relevant chunk found
- 0 = No relevant chunks found

**Use case**: Useful for simple question-answering where one relevant chunk is sufficient.

```python
from src.rag.eval.metrics import hit_rate_at_k

score = hit_rate_at_k(
    retrieved=["c1", "c2", "c3"],
    relevant={"c5", "c6"},
    k=3,
)
# Returns 0 (no relevant chunks in top 3)
```

### Mean Reciprocal Rank (MRR)

**Definition**: Inverse of the rank of the first relevant result.

```
MRR = 1 / rank_of_first_relevant
```

**Interpretation**:
- 1.0 = First result is relevant
- 0.5 = Second result is relevant
- 0.0 = No relevant results found
- Higher is better

**Use case**: Prioritizes systems that rank the first relevant result highly.

```python
from src.rag.eval.metrics import mrr

score = mrr(
    retrieved=["c1", "c2", "c3", "c4", "c5"],
    relevant={"c3", "c5"},
)
# Returns 0.333 (first relevant at position 3)
```

### Average Precision (AP)

**Definition**: Average of precision values at each relevant item's rank.

```
AP = (1/|Relevant|) × Σ Precision@rank_i for each relevant item i
```

**Interpretation**:
- 1.0 = All relevant items ranked at the top in order
- 0.0 = No relevant items found
- Rewards placing relevant items higher in the ranking

**Use case**: Better than precision@k for comparing systems when order matters.

```python
from src.rag.eval.metrics import average_precision

score = average_precision(
    retrieved=["c1", "c2", "c3", "c4", "c5"],
    relevant={"c1", "c3"},
)
# Returns 0.833 (precision@1=1.0, precision@3=0.667, avg=0.833)
```

### NDCG@k (Normalized Discounted Cumulative Gain)

**Definition**: Measures ranking quality with position-weighted relevance.

```
DCG@k = Σ rel_i / log2(i + 1) for i in 1..k
NDCG@k = DCG@k / IDCG@k
```

Where IDCG is the ideal DCG (if all relevant items were ranked first).

**Interpretation**:
- 1.0 = Perfect ranking (all relevant items at top)
- 0.0 = No relevant items found
- Penalizes relevant items ranked lower more heavily than precision

**Use case**: Best metric when rank order significantly impacts user experience.

```python
from src.rag.eval.metrics import ndcg_at_k

score = ndcg_at_k(
    retrieved=["c1", "c2", "c3", "c4", "c5"],
    relevant={"c1", "c4"},
    k=5,
)
# Relevant at positions 1 and 4; discounted by log2(rank+1)
```

## Answer Quality Metrics

Answer metrics evaluate the quality of generated answers against expected answers.

### Semantic Similarity

**Definition**: Cosine similarity between embeddings of generated and expected answers.

```python
similarity = cosine_similarity(
    embed(generated_answer),
    embed(expected_answer),
)
```

**Interpretation**:
- 1.0 = Semantically identical
- 0.0 = Completely unrelated
- ~0.7+ typically indicates good semantic overlap

**Use case**: Quick automated check without LLM judge.

### LLM-as-Judge Metrics

The system uses GPT-4o-mini as a judge to evaluate answer quality across multiple dimensions:

| Metric | Description | Scale |
|--------|-------------|-------|
| `correctness` | Factual accuracy compared to expected answer | 0.0-1.0 |
| `completeness` | How much of the expected information is covered | 0.0-1.0 |
| `relevance` | How well the answer addresses the question | 0.0-1.0 |
| `hallucination_score` | Degree of fabricated information | 0.0-1.0 (lower is better) |

**Binary Flags**:
- `is_correct`: True if correctness >= 0.8
- `has_hallucination`: True if hallucination_score >= 0.3
- `is_abstained`: True if the model refused to answer

### Citation Metrics

| Metric | Description |
|--------|-------------|
| `citation_count` | Number of citations in the answer |
| `answer_length` | Character count of generated answer |

## Aggregation

### RetrievalSummary

Aggregates per-query metrics across an evaluation dataset:

```python
from src.rag.eval.metrics import summarize

summary = summarize(
    results=retrieval_results,
    k_values=[1, 3, 5, 10],
)

print(f"Recall@5: {summary.recall_at_k[5]:.2%}")
print(f"MRR: {summary.mrr:.3f}")
print(f"MAP: {summary.map:.3f}")
```

**Aggregated Metrics**:
- `recall_at_k`: Dict mapping k to mean recall
- `precision_at_k`: Dict mapping k to mean precision
- `hit_rate_at_k`: Dict mapping k to mean hit rate
- `ndcg_at_k`: Dict mapping k to mean NDCG
- `mrr`: Mean reciprocal rank across all queries
- `map`: Mean average precision across all queries

### EvalAggregates

Full aggregation with breakdowns:

```python
aggregates = EvalAggregates(
    overall={
        "count": 100,
        "recall@10": 0.85,
        "mrr": 0.72,
        # ...
    },
    by_type={
        QueryType.factual: {"count": 40, "recall@10": 0.92, ...},
        QueryType.multi_hop: {"count": 10, "recall@10": 0.65, ...},
        # ...
    },
    by_difficulty={
        Difficulty.easy: {"count": 30, "recall@10": 0.95, ...},
        Difficulty.hard: {"count": 20, "recall@10": 0.70, ...},
        # ...
    },
    answer_quality={
        "mean_correctness": 0.85,
        "mean_completeness": 0.78,
        # ...
    },
    latency_ms={
        "p50": 250,
        "p95": 450,
        "p99": 680,
    },
)
```

## Metric Selection Guide

| Goal | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Maximize coverage | Recall@k | Hit Rate@k |
| Minimize noise | Precision@k | NDCG@k |
| Optimize ranking | NDCG@k, MRR | MAP |
| End-to-end quality | Correctness | Completeness, Relevance |
| Detect failures | Has Hallucination | Is Abstained |

## Computing Custom Metrics

All metric functions are standalone and composable:

```python
from src.rag.eval.metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    mrr,
    average_precision,
    ndcg_at_k,
)

# Custom aggregation example
def f1_at_k(retrieved, relevant, k):
    r = recall_at_k(retrieved, relevant, k)
    p = precision_at_k(retrieved, relevant, k)
    if r + p == 0:
        return 0.0
    return 2 * (r * p) / (r + p)
```

## See Also

- [Running Evaluations](running_evaluations.md) - How to run evaluations
- [Traces and Logging](traces_and_logging.md) - Debugging metric issues
