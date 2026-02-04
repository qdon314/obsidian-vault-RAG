from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np

from rag.eval.models import RetrievalResult, RetrievalSummary
from rag.ports import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def semantic_similarity(
    text1: str,
    text2: str,
    embedder: Embedder,
) -> float:
    """
    Compute semantic similarity between two text strings using embeddings.

    Args:
        text1: First text string.
        text2: Second text string.
        embedder: An Embedder instance to generate embeddings.
    Returns:
        Cosine similarity score between the two texts (0.0 to 1.0).
    """
    try:
        v1 = np.asarray(embedder.embed_texts([text1])[0], dtype=np.float32)
        v2 = np.asarray(embedder.embed_texts([text2])[0], dtype=np.float32)

        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0.0:
            return 0.0

        return float(np.dot(v1, v2) / denom)

    except Exception as e:
        logger.error("Error computing similarity: %s", e)
        return 0.0


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


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """
    Compute Precision@K for a single query.
    Args:
        retrieved: List of retrieved chunk IDs, ordered by relevance.
        relevant: Set of relevant chunk IDs.
        k: The cutoff rank.
    Returns:
        Precision@K value (fraction of top-k that are relevant).
    """
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for cid in topk if cid in relevant)
    return hits / len(topk)


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """
    Compute Hit Rate (Success@K) for a single query.
    Args:
        retrieved: List of retrieved chunk IDs, ordered by relevance.
        relevant: Set of relevant chunk IDs.
        k: The cutoff rank.
    Returns:
        1.0 if at least one relevant item in top-k, else 0.0.
    """
    topk = retrieved[:k]
    return 1.0 if any(cid in relevant for cid in topk) else 0.0


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """
    Compute Reciprocal Rank for a single query.
    Args:
        retrieved: List of retrieved chunk IDs, ordered by relevance.
        relevant: Set of relevant chunk IDs.
    Returns:
        Reciprocal rank of first relevant item (1/rank), or 0.0 if none found.
    """
    for i, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / float(i)
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """
    Compute Average Precision for a single query.
    AP is the average of precision values at each relevant item's rank.
    Args:
        retrieved: List of retrieved chunk IDs, ordered by relevance.
        relevant: Set of relevant chunk IDs.
    Returns:
        Average Precision value.
    """
    if not relevant:
        return 0.0
    precisions = []
    hits = 0
    for i, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(relevant) if precisions else 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K for a single query.

    NDCG measures ranking quality by rewarding relevant items appearing earlier
    in the results. It applies a logarithmic discount to relevance scores based
    on position: items ranked first contribute more than items ranked later.
    The score is normalized against the ideal ranking (all relevant items first)
    to produce a value between 0 and 1.

    Uses binary relevance (1 if relevant, 0 otherwise).
    Args:
        retrieved: List of retrieved chunk IDs, ordered by relevance.
        relevant: Set of relevant chunk IDs.
        k: The cutoff rank.
    Returns:
        NDCG@K value (0.0 to 1.0, higher is better).
    """
    import math

    topk = retrieved[:k]
    if not relevant or not topk:
        return 0.0

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, cid in enumerate(topk, start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: all relevant items ranked first
    ideal_length = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_length + 1))

    return dcg / idcg if idcg > 0 else 0.0


def summarize(
    results: Iterable[RetrievalResult], *, ks: Sequence[int] = (5, 10)
) -> RetrievalSummary:
    results = list(results)
    if not results:
        return RetrievalSummary(
            num_queries=0,
            avg_retrieved=0.0,
            recall_at_k=dict.fromkeys(ks, 0.0),
            precision_at_k=dict.fromkeys(ks, 0.0),
            hit_rate_at_k=dict.fromkeys(ks, 0.0),
            ndcg_at_k=dict.fromkeys(ks, 0.0),
            mrr=0.0,
            map=0.0,
        )

    n = len(results)
    avg_retrieved = sum(len(r.retrieved_chunk_ids) for r in results) / n

    recall_at_k_res = {}
    precision_at_k_res = {}
    hit_rate_at_k_res = {}
    ndcg_at_k_res = {}

    for k in ks:
        recall_at_k_res[k] = (
            sum(recall_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in results) / n
        )
        precision_at_k_res[k] = (
            sum(precision_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in results) / n
        )
        hit_rate_at_k_res[k] = (
            sum(hit_rate_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in results) / n
        )
        ndcg_at_k_res[k] = (
            sum(ndcg_at_k(r.retrieved_chunk_ids, r.relevant_chunk_ids, k) for r in results) / n
        )

    mrr_score = sum(mrr(r.retrieved_chunk_ids, r.relevant_chunk_ids) for r in results) / n
    map_score = (
        sum(average_precision(r.retrieved_chunk_ids, r.relevant_chunk_ids) for r in results) / n
    )

    return RetrievalSummary(
        num_queries=n,
        avg_retrieved=avg_retrieved,
        recall_at_k=recall_at_k_res,
        precision_at_k=precision_at_k_res,
        hit_rate_at_k=hit_rate_at_k_res,
        ndcg_at_k=ndcg_at_k_res,
        mrr=mrr_score,
        map=map_score,
    )
