from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Mapping, Optional, Sequence

from rag.domain.models import Candidate, Chunk

Vector = list[float]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return sqrt(sum(x * x for x in a)) or 1.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


@dataclass(slots=True)
class InMemoryVectorStore:
    """
    Simple cosine-similarity vector store to validate architecture.

    Stores chunks + vectors in memory; supports naive linear scan search.
    """
    _chunks: list[Chunk] = field(default_factory=list)
    _vectors: list[Vector] = field(default_factory=list)

    def upsert(
        self,
        *,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        self._chunks.extend(list(chunks))
        self._vectors.extend([list(v) for v in vectors])

    def search(
        self,
        *,
        query_vector: Vector,
        top_k: int,
        filters: Optional[Mapping[str, object]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> list[Candidate]:
        # Very basic filter support: match chunk.metadata[key] == value
        def allowed(chunk: Chunk) -> bool:
            if not filters:
                return True
            for k, v in filters.items():
                if chunk.metadata.get(k) != v:
                    return False
            return True

        scored: list[Candidate] = []
        for chunk, vec in zip(self._chunks, self._vectors):
            if not allowed(chunk):
                continue
            score = _cosine(query_vector, vec)
            scored.append(Candidate(chunk=chunk, score=score))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)
