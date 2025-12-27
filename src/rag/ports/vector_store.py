from __future__ import annotations

from typing import Mapping, Protocol, Sequence

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
        filters: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        ...

    def count(self) -> int:
        ...
