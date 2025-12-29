from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    qid: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: set[str]


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """
    Compute Recall@K for a single query.
    Args:
        retrieved: List of retrieved chunk IDs, ordered by relevance.
        relevant: Set of relevant chunk IDs.
        k: The cutoff rank.
    Returns:
        Recall@K value.
    """
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for cid in topk if cid in relevant)
    return hits / float(len(relevant))


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """
    Compute Mean Reciprocal Rank for a single query.
    Args:
        retrieved: List of retrieved chunk IDs, ordered by relevance.
        relevant: Set of relevant chunk IDs.
    Returns:
        MRR value."""
    for i, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / float(i)
    return 0.0


def summarize(results: Iterable[RetrievalResult], *, ks: Sequence[int] = (5, 10)) -> dict[str, float]:
    """
    Summarize retrieval results across multiple queries.
    Args:
        results: Iterable of RetrievalResult objects.
        ks: Sequence of K values for Recall@K computation.
    Returns:
        Dictionary with average Recall@K and MRR.
    """
    results = list(results)
    if not results:
        return {"num_queries": 0}

    out: dict[str, float] = {"num_queries": float(len(results))}
    out["avg_retrieved"] = sum(len(r.retrieved_chunk_ids) for r in results) / len(results)

    for k in ks:
        out[f"recall@{k}"] = (
            sum(recall_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in results) / len(results)
        )
    out["mrr"] = sum(mrr(r.retrieved_chunk_ids, r.relevant_chunk_ids) for r in results) / len(results)
    return out
