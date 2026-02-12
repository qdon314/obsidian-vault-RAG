"""Retriever wrapper that hydrates thin chunks from a ChunkStore.

When the vector store returns candidates with empty text (thin Qdrant
payloads), this retriever batch-fetches full chunk content from a
separate :class:`~rag.ports.chunk_store.ChunkStore`.

Pattern follows :class:`~rag.adapters.embedding.sqlite_cache.CachedEmbedder`:
wrap any ``Retriever`` via composition and add transparent behaviour.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace

from rag.domain.filters import Where
from rag.domain.models import Candidate
from rag.ports.chunk_store import ChunkStore
from rag.ports.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HydratingRetriever:
    """Wraps any Retriever, hydrating empty-text chunks from a ChunkStore.

    Implements the ``Retriever`` protocol via structural subtyping so it
    can be used anywhere a plain retriever is accepted.

    Behaviour
    ---------
    1. Delegates to the wrapped retriever's ``retrieve()``.
    2. Identifies candidates whose ``chunk.text`` is empty.
    3. Batch-fetches full chunks from ``chunk_store``.
    4. Replaces stub chunks while preserving scores, order, and metadata.

    If all candidates already have text (fat payload / JSONL path), the
    chunk store is never called.
    """

    retriever: Retriever
    chunk_store: ChunkStore

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        candidates = self.retriever.retrieve(
            query,
            top_k=top_k,
            where=where,
            metadata=metadata,
        )

        # Identify candidates that need hydration
        needs_hydration = [(i, cand) for i, cand in enumerate(candidates) if not cand.chunk.text]
        if not needs_hydration:
            return candidates

        # Batch fetch full chunks
        chunk_ids = [cand.chunk.chunk_id for _, cand in needs_hydration]
        hydrated_chunks = self.chunk_store.get_chunks(chunk_ids, metadata=metadata)

        # Replace stub chunks, preserving scores and order
        result = list(candidates)
        for idx, cand in needs_hydration:
            full_chunk = hydrated_chunks.get(cand.chunk.chunk_id)
            if full_chunk is not None:
                result[idx] = replace(cand, chunk=full_chunk)
            else:
                logger.warning(
                    "Chunk %s not found in ChunkStore; keeping stub",
                    cand.chunk.chunk_id,
                )

        return result
