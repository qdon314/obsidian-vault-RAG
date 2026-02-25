from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.adapters.filters.qdrant_compiler import QdrantFilterCompiler
from rag.domain.filters import Where
from rag.domain.models import Candidate, Chunk

Vector = list[float]

# Fields that map directly to named Chunk attributes (not part of metadata).
# Used when reconstructing a Chunk from a thin Qdrant payload.
_CHUNK_FIRST_CLASS_FIELDS: frozenset[str] = frozenset(
    {
        "chunk_id",
        "doc_id",
        "text",
        "chunk_index",
        "start_char",
        "end_char",
        "section_heading",
        "section_path",
        "language",
        "metadata",
    }
)


def _thin_payload(chunk: Chunk) -> dict[str, object]:
    """Build a thin Qdrant payload: IDs + filterable metadata, no text."""
    payload: dict[str, object] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "section_heading": chunk.section_heading,
        "section_path": chunk.section_path,
        "language": chunk.language,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }
    # Spread chunk.metadata so filterable fields (tags, citation, etc.) are
    # available for Qdrant filter queries.
    payload.update(chunk.metadata)
    return payload


def _chunk_from_thin_payload(payload: dict[str, object]) -> Chunk:
    """Reconstruct a stub Chunk from a thin payload (empty text)."""
    metadata = {k: v for k, v in payload.items() if k not in _CHUNK_FIRST_CLASS_FIELDS}
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        doc_id=str(payload["doc_id"]),
        text="",
        chunk_index=int(payload.get("chunk_index", 0)),  # type: ignore[call-overload]
        start_char=payload.get("start_char"),  # type: ignore[arg-type]
        end_char=payload.get("end_char"),  # type: ignore[arg-type]
        section_heading=payload.get("section_heading"),  # type: ignore[arg-type]
        section_path=payload.get("section_path"),  # type: ignore[arg-type]
        language=payload.get("language"),  # type: ignore[arg-type]
        metadata=metadata,
    )


@dataclass(slots=True)
class QdrantVectorStore:
    """Qdrant-backed vector store for scalable similarity search.

    Supports both local (in-memory or disk) and remote Qdrant instances.

    Args:
        collection_name: Name of the Qdrant collection.
        vector_size: Dimension of vectors (must match embedder output).
        url: Qdrant server URL (for remote). If None, uses local mode.
        path: Path for local disk persistence. If None with no url, uses in-memory.
        api_key: API key for Qdrant Cloud.
        distance: Distance metric (COSINE, EUCLID, DOT).
        thin_payloads: When True, store only IDs + filterable metadata (no text).
            Requires a ChunkStore + HydratingRetriever for text hydration at
            query time.
    """

    collection_name: str
    vector_size: int
    url: str | None = None
    path: str | None = None
    api_key: str | None = None
    distance: Distance = Distance.COSINE
    thin_payloads: bool = False
    _client: QdrantClient = field(init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.url:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        elif self.path:
            self._client = QdrantClient(path=self.path)
        else:
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

        self._ensure_payload_indexes()
        self._initialized = True

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for commonly-filtered fields.

        Idempotent — Qdrant silently ignores duplicate index creation.
        No-op for local/in-memory mode (qdrant-client emits a UserWarning).
        """
        indexes: list[tuple[str, PayloadSchemaType]] = [
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("language", PayloadSchemaType.KEYWORD),
            ("chunk_index", PayloadSchemaType.INTEGER),
            ("tags", PayloadSchemaType.KEYWORD),
            ("citation", PayloadSchemaType.KEYWORD),
        ]
        for field_name, field_schema in indexes:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

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
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = _thin_payload(chunk) if self.thin_payloads else chunk.to_dict()

            points.append(
                PointStruct(
                    id=point_id,
                    vector=list(vector),
                    payload=payload,
                )
            )

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
            query=list(query_vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        candidates = []
        for hit in response.points:
            if hit.payload:
                # Detect thin vs fat payload by checking for 'text' key
                if "text" in hit.payload:
                    chunk = Chunk.from_dict(hit.payload)
                else:
                    chunk = _chunk_from_thin_payload(hit.payload)
                candidates.append(Candidate(chunk=chunk, score=hit.score))

        return candidates

    def count(self) -> int:
        self._ensure_collection()
        info = self._client.get_collection(self.collection_name)
        return info.points_count or 0

    def all_chunks(self) -> list[Chunk]:
        raise NotImplementedError(
            "QdrantVectorStore does not support all_chunks(); "
            "chunks are stored server-side and not available for bulk retrieval."
        )

    def save(self) -> None:
        """Persist to disk (only applicable for local disk mode).

        For in-memory mode, this is a no-op.
        For remote mode, data is already persisted on the server.
        """

    def load(self) -> None:
        """Load from disk (only applicable for local disk mode).

        For in-memory mode, this is a no-op.
        For remote mode, data is already on the server.
        """
        self._ensure_collection()

    def delete_collection(self) -> None:
        """Delete the entire collection. Use with caution."""
        self._client.delete_collection(self.collection_name)
        self._initialized = False

    def clear(self) -> None:
        """Delete all points in the collection but keep the collection."""
        self._ensure_collection()
        self.delete_collection()
        self._ensure_collection()
