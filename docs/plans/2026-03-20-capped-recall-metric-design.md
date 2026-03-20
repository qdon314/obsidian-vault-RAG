# Capped Recall Metric Design

**Date:** 2026-03-20
**Status:** Approved
**Problem:** Denominator explosion in recall metrics for section-level citations

## Problem

The `case_generated_queries.jsonl` dataset (198 queries) uses section-level
regulatory citations as ground truth (e.g., `"10 CFR §50.54"`). During eval,
`_resolve_relevance_tiers()` resolves these via `citation_key_to_ids`, pulling
in every chunk for that regulation section — potentially 50–100+ chunks.

`recall_at_k` divides hits by `len(relevant)`, so with k=10 and 80 relevant
chunks the theoretical max is 12.5%. The metric becomes uninterpretable.

The queries were auto-generated from NRC case documents; the intent is "did
the system find content from the right regulation" — not "did it retrieve
every chunk." Retrieving 2–5 chunks from the correct section is a good result.

## Solution

Add a **capped recall** metric: `hits_in_topk / min(len(relevant), k)`.

When `len(relevant) <= k`, this equals standard recall. It only diverges when
the ground truth set exceeds k — exactly the denominator explosion case.

This is additive — existing recall stays for backward compatibility. A
follow-up will refine the dataset to subsection-level citations for
precision-oriented eval.

## Changes

### 1. `src/rag/eval/metrics.py`

New function next to `recall_at_k`:

```python
def capped_recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for cid in topk if cid in relevant)
    return hits / float(min(len(relevant), k))
```

Add aggregation loop in `summarize()` following the existing pattern.

### 2. `src/rag/eval/models.py`

Add to `RetrievalSummary`:

```python
capped_recall_at_k: dict[int, float] = field(default_factory=dict)
```

Wire into `to_flat_dict()` as `capped_recall@{k}` and `from_flat_dict()` via
`_parse_flattened_metrics(data, "capped_recall")`.

### 3. `eval/app_v2/engine/domain/models.py`

Add to `RunHealthSummary`:

```python
headline_capped_recall_at_10: float | None = None
```

### 4. `eval/app_v2/engine/derived/health.py`

Extract from aggregates:

```python
capped_recall_at_10: float | None = None
if overall.capped_recall_at_k:
    capped_recall_at_10 = overall.capped_recall_at_k.get(10)
```

Pass as `headline_capped_recall_at_10=capped_recall_at_10`.

### 5. `eval/app_v2/ui/widgets/metric_cards.py`

Replace Precision@10 in Row 2 slot d2 with Capped Recall@10:

```python
d2.metric(
    "Capped Recall@10",
    _FMT_PCT(health.headline_capped_recall_at_10),
    help="Recall with denominator capped at k — interpretable when ground truth sets exceed k",
)
```

### 6. `tests/eval/test_metrics.py` (new)

Unit tests for `capped_recall_at_k`:

- `len(relevant) <= k`: equals standard recall
- `len(relevant) > k`: denominator is k
- Empty relevant: returns 0.0

Integration test: `summarize()` populates `capped_recall_at_k` field.

## Not Changed

- Eval harness, citation resolution, eval schema — ground truth stays the same
- Existing metric functions — all preserved
- `recall_at_k` — still computed for backward compatibility

## Follow-up

Refine `case_generated_queries.jsonl` to use subsection-level citations
(e.g., `"10 CFR §50.54(f)"`) for a separate precision-oriented eval methodology.
