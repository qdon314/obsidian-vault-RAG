from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass

from rag.domain.filters import Where
from rag.domain.models import Candidate
from rag.ports import Retriever


@dataclass(frozen=True, slots=True)
class HybridRetriever:
    """Fuses two retrievers via Reciprocal Rank Fusion (RRF)."""

    primary: Retriever
    secondary: Retriever
    primary_weight: float = 0.7
    secondary_weight: float = 0.3
    rrf_k: int = 60

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        primary_results = self.primary.retrieve(
            query, top_k=top_k * 2, where=where, metadata=metadata
        )
        secondary_results = self.secondary.retrieve(
            query, top_k=top_k * 2, where=where, metadata=metadata
        )
        return self._rrf_fuse(primary_results, secondary_results, top_k)

    def _rrf_fuse(
        self,
        a: list[Candidate],
        b: list[Candidate],
        top_k: int,
    ) -> list[Candidate]:
        """RRF: score = sum(weight / (k + rank)) for each list containing the item."""
        scores: dict[str, float] = {}
        lookup: dict[str, Candidate] = {}

        for rank, cand in enumerate(a, start=1):
            cid = cand.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + self.primary_weight / (self.rrf_k + rank)
            lookup[cid] = cand

        for rank, cand in enumerate(b, start=1):
            cid = cand.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + self.secondary_weight / (self.rrf_k + rank)
            # Keep the primary's Candidate when a chunk appears in both lists
            lookup.setdefault(cid, cand)

        ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        return [dataclasses.replace(lookup[cid], score=scores[cid]) for cid in ranked_ids]
