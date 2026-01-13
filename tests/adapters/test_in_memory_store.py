"""Tests for InMemoryVectorStore adapter."""

from __future__ import annotations

import pytest

from rag.domain.filters import Eq
from rag.ports import VectorStore
from tests.conftest import make_chunk, make_vector


class TestInMemoryStoreUpsert:
    """Upsert operations."""

    def test_upsert_increases_count(self, in_memory_store: VectorStore):
        """Upserting chunks increases store count."""
        chunks = [make_chunk(chunk_id=f"chunk-{i}") for i in range(3)]
        vectors = [make_vector() for _ in range(3)]

        in_memory_store.upsert(chunks=chunks, vectors=vectors)

        assert in_memory_store.count() == 3

    def test_upsert_empty_lists(self, in_memory_store: VectorStore):
        """Upserting empty lists is valid and changes nothing."""
        in_memory_store.upsert(chunks=[], vectors=[])

        assert in_memory_store.count() == 0

    def test_upsert_mismatched_lengths_raises_error(self, in_memory_store: VectorStore):
        """Upsert with mismatched chunks/vectors raises ValueError."""
        chunks = [make_chunk()]
        vectors = [make_vector(), make_vector()]  # Too many vectors

        with pytest.raises(ValueError, match="same length"):
            in_memory_store.upsert(chunks=chunks, vectors=vectors)

    def test_multiple_upserts_accumulate(self, in_memory_store: VectorStore):
        """Multiple upsert calls accumulate chunks."""
        in_memory_store.upsert(chunks=[make_chunk(chunk_id="1")], vectors=[make_vector()])
        in_memory_store.upsert(chunks=[make_chunk(chunk_id="2")], vectors=[make_vector()])

        assert in_memory_store.count() == 2


class TestInMemoryStoreSearch:
    """Search operations."""

    def test_search_empty_store_returns_empty(self, in_memory_store: VectorStore):
        """Searching empty store returns empty results."""
        query_vec = make_vector()

        results = in_memory_store.search(query_vector=query_vec, top_k=5)

        assert results == []

    def test_search_returns_candidates(self, in_memory_store: VectorStore):
        """Search returns Candidate objects."""
        chunk = make_chunk()
        vector = make_vector()
        in_memory_store.upsert(chunks=[chunk], vectors=[vector])

        results = in_memory_store.search(query_vector=vector, top_k=1)

        assert len(results) == 1
        assert hasattr(results[0], "chunk")
        assert hasattr(results[0], "score")

    def test_search_returns_top_k_candidates(self, in_memory_store: VectorStore):
        """Search returns at most top_k results."""
        chunks = [make_chunk(chunk_id=f"chunk-{i}") for i in range(10)]
        vectors = [make_vector(value=float(i) / 10) for i in range(10)]
        in_memory_store.upsert(chunks=chunks, vectors=vectors)

        results = in_memory_store.search(query_vector=make_vector(value=0.5), top_k=3)

        assert len(results) == 3

    def test_search_fewer_than_top_k(self, in_memory_store: VectorStore):
        """Search returns all results if fewer than top_k exist."""
        chunks = [make_chunk(chunk_id=f"chunk-{i}") for i in range(2)]
        vectors = [make_vector() for _ in range(2)]
        in_memory_store.upsert(chunks=chunks, vectors=vectors)

        results = in_memory_store.search(query_vector=make_vector(), top_k=10)

        assert len(results) == 2


class TestInMemoryStoreSimilarity:
    """Cosine similarity ordering."""

    def test_search_orders_by_similarity_descending(self, in_memory_store: VectorStore):
        """Search results are ordered by cosine similarity (highest first)."""
        chunk_similar = make_chunk(chunk_id="similar", text="similar")
        chunk_dissimilar = make_chunk(chunk_id="dissimilar", text="dissimilar")

        # Vectors: one similar to query, one opposite
        vec_similar = [1.0] * 128
        vec_dissimilar = [-1.0] * 128

        in_memory_store.upsert(
            chunks=[chunk_dissimilar, chunk_similar],
            vectors=[vec_dissimilar, vec_similar],
        )

        query_vec = [1.0] * 128  # Similar to vec_similar
        results = in_memory_store.search(query_vector=query_vec, top_k=2)

        assert results[0].chunk.chunk_id == "similar"
        assert results[0].score > results[1].score

    def test_identical_vector_has_score_near_one(self, in_memory_store: VectorStore):
        """Identical vectors should have cosine similarity near 1.0."""
        chunk = make_chunk()
        vector = [0.5] * 128
        in_memory_store.upsert(chunks=[chunk], vectors=[vector])

        results = in_memory_store.search(query_vector=vector, top_k=1)

        assert results[0].score > 0.99  # Should be ~1.0


class TestInMemoryStoreFiltering:
    """Metadata filtering."""

    def test_filter_by_single_metadata_field(self, in_memory_store: VectorStore):
        """Search filters by chunk metadata."""
        chunk_python = make_chunk(chunk_id="py", metadata={"category": "python"})
        chunk_rust = make_chunk(chunk_id="rs", metadata={"category": "rust"})
        vectors = [make_vector(), make_vector()]

        in_memory_store.upsert(chunks=[chunk_python, chunk_rust], vectors=vectors)

        results = in_memory_store.search(
            query_vector=make_vector(),
            top_k=10,
            where=Eq(field="category", value="python"),
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "py"

    def test_filter_returns_empty_if_no_match(self, in_memory_store: VectorStore):
        """Filtering with no matches returns empty list."""
        chunk = make_chunk(metadata={"category": "python"})
        in_memory_store.upsert(chunks=[chunk], vectors=[make_vector()])

        results = in_memory_store.search(
            query_vector=make_vector(),
            top_k=10,
            where=Eq(field="category", value="nonexistent"),
        )

        assert results == []

    def test_no_filters_returns_all(self, in_memory_store: VectorStore):
        """Without filters, all chunks are considered."""
        chunks = [make_chunk(chunk_id=f"chunk-{i}") for i in range(3)]
        vectors = [make_vector() for _ in range(3)]
        in_memory_store.upsert(chunks=chunks, vectors=vectors)

        results = in_memory_store.search(query_vector=make_vector(), top_k=10)

        assert len(results) == 3


class TestInMemoryStorePersistence:
    """Save/load operations (no-op for in-memory)."""

    def test_save_is_noop(self, in_memory_store: VectorStore):
        """Save should not raise and has no effect."""
        in_memory_store.upsert(chunks=[make_chunk()], vectors=[make_vector()])
        in_memory_store.save()  # Should not raise

        assert in_memory_store.count() == 1

    def test_load_is_noop(self, in_memory_store: VectorStore):
        """Load should not raise and has no effect."""
        in_memory_store.load()  # Should not raise

        assert in_memory_store.count() == 0
