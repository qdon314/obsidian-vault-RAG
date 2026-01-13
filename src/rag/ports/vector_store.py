from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag.domain.filters import Where
from rag.domain.models import Candidate, Chunk

Vector = list[float]


class VectorStore(Protocol):
    """
    Stores (Chunk, Vector) pairs and supports similarity search.
    """

    def upsert(
        self,
        *,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        ...

    def search(
        self,
        *,
        query_vector: Vector,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        ...

    def count(self) -> int:
        ...
    
    def save(self) -> None:
        """Persist the store to disk, if applicable."""
        ...

    def load(self) -> None:
        """Load the store from disk, if applicable."""
        ...