"""Tests for Qdrant payload index creation.

Verifies that _ensure_collection() creates payload indexes on
commonly-filtered fields. In-memory mode silently accepts index
creation calls (no-op), so these tests verify the flow completes
without exceptions.
"""

from __future__ import annotations

import warnings

from rag.adapters.vectorstores.qdrant_store import QdrantVectorStore
from tests.conftest import make_chunk


class TestPayloadIndexes:
    """Payload index creation in _ensure_collection."""

    def test_ensure_collection_creates_indexes_without_error(self) -> None:
        """_ensure_collection completes without raising on index creation."""
        store = QdrantVectorStore(collection_name="test-idx", vector_size=4)
        # Suppress the UserWarning from in-memory qdrant-client
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            store._ensure_collection()

        # Collection should exist
        collections = store._client.get_collections().collections
        assert any(c.name == "test-idx" for c in collections)

    def test_indexes_are_idempotent(self) -> None:
        """Calling _ensure_collection twice does not error."""
        store = QdrantVectorStore(collection_name="test-idem", vector_size=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Reset _initialized to force re-run
            store._ensure_collection()
            object.__setattr__(store, "_initialized", False)
            store._ensure_collection()

        assert store._initialized

    def test_upsert_and_search_with_indexes(self) -> None:
        """Upsert + search works after index creation (no interference)."""
        store = QdrantVectorStore(collection_name="test-search", vector_size=4)
        chunk = make_chunk(
            chunk_id="c1",
            doc_id="d1",
            metadata={"tags": ["ml"], "citation": "10 CFR 50.1"},
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            store.upsert(chunks=[chunk], vectors=[[1.0, 0.0, 0.0, 0.0]])
            results = store.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=1)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"

    def test_filtered_search_after_index_creation(self) -> None:
        """Filtered search works after payload indexes are created."""
        from rag.domain.filters import Eq

        store = QdrantVectorStore(collection_name="test-filter", vector_size=4)
        chunk = make_chunk(chunk_id="c1", doc_id="d1", text="hello")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            store.upsert(chunks=[chunk], vectors=[[1.0, 0.0, 0.0, 0.0]])
            results = store.search(
                query_vector=[1.0, 0.0, 0.0, 0.0],
                top_k=5,
                where=Eq(field="doc_id", value="d1"),
            )

        assert len(results) == 1
        assert results[0].chunk.doc_id == "d1"
