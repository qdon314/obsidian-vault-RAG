from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from rag.adapters.filters.qdrant_compiler import QdrantFilterCompiler
from rag.domain.filters import Where
from rag.domain.models import Candidate, Chunk
from rag.ports import VectorStore

Vector = list[float]


@dataclass(slots=True)
class QdrantVectorStore(VectorStore):
    """
    Qdrant-backed vector store for scalable similarity search.

    Supports both local (in-memory or disk) and remote Qdrant instances.

    Args:
        collection_name: Name of the Qdrant collection.
        vector_size: Dimension of vectors (must match embedder output).
        url: Qdrant server URL (for remote). If None, uses local mode.
        path: Path for local disk persistence. If None with no url, uses in-memory.
        api_key: API key for Qdrant Cloud.
        distance: Distance metric (COSINE, EUCLID, DOT).
    """

    collection_name: str
    vector_size: int
    url: str | None = None
    path: str | None = None
    api_key: str | None = None
    distance: Distance = Distance.COSINE
    _client: QdrantClient = field(init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.url:
            # Remote Qdrant server
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        elif self.path:
            # Local disk persistence
            self._client = QdrantClient(path=self.path)
        else:
            # In-memory mode
            self._client = QdrantClient(":memory:")

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        if self._initialized:
            return

        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

        self._initialized = True

    def upsert(
        self,
        *,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        self._ensure_collection()

        points = []
        for chunk, vector in zip(chunks, vectors, strict=False):
            # Use chunk_id as point ID (hash to UUID for Qdrant compatibility)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = chunk.to_dict()

            points.append(
                PointStruct(
                    id=point_id,
                    vector=list(vector),
                    payload=payload,
                )
            )

        # Batch upsert
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        *,
        query_vector: Vector,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        self._ensure_collection()
        compiler = QdrantFilterCompiler()

        query_filter = compiler.compile(where)

        response = self._client.query_points(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        candidates = []
        for hit in response.points:
            if hit.payload:
                chunk = Chunk.from_dict(hit.payload)
                candidates.append(Candidate(chunk=chunk, score=hit.score))

        return candidates

    def count(self) -> int:
        self._ensure_collection()
        info = self._client.get_collection(self.collection_name)
        return info.points_count or 0

    def save(self) -> None:
        """
        Persist to disk (only applicable for local disk mode).

        For in-memory mode, this is a no-op.
        For remote mode, data is already persisted on the server.
        """
        # Qdrant handles persistence automatically in disk mode
        pass

    def load(self) -> None:
        """
        Load from disk (only applicable for local disk mode).

        For in-memory mode, this is a no-op.
        For remote mode, data is already on the server.
        """
        # Qdrant loads automatically; just ensure collection exists
        self._ensure_collection()

    def delete_collection(self) -> None:
        """Delete the entire collection. Use with caution."""
        self._client.delete_collection(self.collection_name)
        self._initialized = False

    def clear(self) -> None:
        """Delete all points in the collection but keep the collection."""
        self._ensure_collection()
        # Delete and recreate collection
        self.delete_collection()
        self._ensure_collection()
