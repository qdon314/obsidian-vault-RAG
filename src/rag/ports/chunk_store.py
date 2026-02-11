"""Port for ID-based chunk content storage and retrieval.

Distinct from ChunkLoader (path-based, local filesystem). ChunkStore is
designed for remote backends (S3 + Postgres) where chunks are looked up
by stable chunk_id rather than loaded from a directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag.domain.models import Chunk


class ChunkStore(Protocol):
    """Stores and retrieves chunk content by chunk_id.

    Implementations handle persistence (S3 shards, Postgres index, etc.)
    and support batch operations for efficient hydration at query time.
    """

    def get_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, Chunk]:
        """Batch fetch chunks by ID.

        Returns a dict mapping chunk_id to Chunk.  Missing IDs are
        silently omitted from the result (caller should handle gaps).
        """
        ...

    def store_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Persist chunks.  Idempotent (upsert by chunk_id)."""
        ...

    def list_all_chunk_ids(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[str]:
        """List all stored chunk IDs.

        Used by the eval harness for citation resolution when
        VectorStore.all_chunks() is unavailable (e.g. Qdrant).
        """
        ...
