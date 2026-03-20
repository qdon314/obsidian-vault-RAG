# eval/app_v2/engine/derived/chunk_stats.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from eval.app_v2.engine.domain.models import AnalyzedQuery


@dataclass(frozen=True, slots=True)
class ChunkStat:
    chunk_id: str
    queries_where_relevant: int
    queries_where_retrieved: int
    queries_where_reranked: int
    queries_where_packed: int
    miss_rate: float  # (queries_where_relevant - queries_where_retrieved) / queries_where_relevant
    rerank_drop_rate: (
        float  # (queries_where_retrieved - queries_where_reranked) / queries_where_retrieved
    )


def build_chunk_stats(queries: Sequence[AnalyzedQuery]) -> tuple[ChunkStat, ...]:
    """Compute per-chunk aggregate statistics across all queries.

    Useful for identifying chunks that are consistently missed, dropped at rerank,
    or otherwise problematic across the run.
    """
    relevant_counts: dict[str, int] = defaultdict(int)
    retrieved_counts: dict[str, int] = defaultdict(int)
    reranked_counts: dict[str, int] = defaultdict(int)
    packed_counts: dict[str, int] = defaultdict(int)
    has_rerank_trace: set[str] = set()  # chunks seen in at least one query with rerank data

    for aq in queries:
        r = aq.record
        for cid in r.relevant_chunk_ids:
            relevant_counts[cid] += 1
        for cid in r.retrieved_chunk_ids:
            retrieved_counts[cid] += 1
        if r.reranked_chunk_ids is not None:
            for cid in r.reranked_chunk_ids:
                reranked_counts[cid] += 1
            # All retrieved chunks in this query now have rerank-trace data
            for cid in r.retrieved_chunk_ids:
                has_rerank_trace.add(cid)
        if r.packed_chunk_ids is not None:
            for cid in r.packed_chunk_ids:
                packed_counts[cid] += 1

    all_chunks = set(relevant_counts) | set(retrieved_counts)
    stats: list[ChunkStat] = []
    for cid in all_chunks:
        n_rel = relevant_counts.get(cid, 0)
        n_ret = retrieved_counts.get(cid, 0)
        n_rrk = reranked_counts.get(cid, 0)
        n_pck = packed_counts.get(cid, 0)
        miss_rate = (n_rel - n_ret) / n_rel if n_rel > 0 else 0.0
        rerank_drop_rate = (
            (n_ret - n_rrk) / n_ret if (n_ret > 0 and cid in has_rerank_trace) else 0.0
        )
        stats.append(
            ChunkStat(
                chunk_id=cid,
                queries_where_relevant=n_rel,
                queries_where_retrieved=n_ret,
                queries_where_reranked=n_rrk,
                queries_where_packed=n_pck,
                miss_rate=miss_rate,
                rerank_drop_rate=rerank_drop_rate,
            )
        )

    return tuple(sorted(stats, key=lambda s: s.miss_rate, reverse=True))
