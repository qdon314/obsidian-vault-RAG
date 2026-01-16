"""Tests for JsonlVectorStore adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.adapters.vectorstores.jsonl_store import JsonlVectorStore
from rag.domain.filters import Eq
from tests.conftest import make_chunk, make_vector, write_chunks_jsonl


class TestJsonlVectorStoreUpsert:
    """Upsert operations."""

    def test_upsert_stores_chunks(self, tmp_path: Path):
        """Upserting chunks makes them searchable."""
        store_dir = tmp_path / "store"
        store = JsonlVectorStore(store_dir)

        chunks = [make_chunk(chunk_id=f"chunk-{i}") for i in range(3)]
        vectors = [make_vector() for _ in range(3)]

        store.upsert(chunks=chunks, vectors=vectors)

        assert store.count() == 3

    def test_upsert_empty_lists(self, tmp_path: Path):
        """Upserting empty lists is valid."""
        store_dir = tmp_path / "store"
        store = JsonlVectorStore(store_dir)

        store.upsert(chunks=[], vectors=[])

        assert store.count() == 0

    def test_upsert_mismatched_lengths_raises_error(self, tmp_path: Path):
        """Upsert with mismatched chunks/vectors raises ValueError."""
        store_dir = tmp_path / "store"
        store = JsonlVectorStore(store_dir)

        chunks = [make_chunk()]
        vectors = [make_vector(), make_vector()]

        with pytest.raises(ValueError, match="same length"):
            store.upsert(chunks=chunks, vectors=vectors)


class TestJsonlVectorStorePersistence:
    """Save and load operations."""

    def test_save_creates_file(self, tmp_path: Path):
        """Save creates JSONL file."""
        store_dir = tmp_path / "store"
        store = JsonlVectorStore(store_dir)

        chunks = [make_chunk(chunk_id="test-chunk", text="Test content")]
        vectors = [make_vector()]
        store.upsert(chunks=chunks, vectors=vectors)

        store.save()

        data_file = store_dir / "chunks.jsonl"
        assert data_file.exists()
        content = data_file.read_text()
        assert "test-chunk" in content

    def test_save_overwrites_existing(self, tmp_path: Path):
        """Save overwrites existing file."""
        store_dir = tmp_path / "store"
        store_dir.mkdir()
        data_file = store_dir / "chunks.jsonl"
        data_file.write_text("old content\n")

        store = JsonlVectorStore(store_dir)
        store.upsert(chunks=[make_chunk(chunk_id="new")], vectors=[make_vector()])
        store.save()

        content = data_file.read_text()
        assert "old content" not in content
        assert "new" in content

    def test_load_restores_data(self, tmp_path: Path):
        """Load restores saved data."""
        store_dir = tmp_path / "store"

        # Create and save
        store1 = JsonlVectorStore(store_dir)
        chunks = [make_chunk(chunk_id=f"chunk-{i}") for i in range(3)]
        vectors = [make_vector(value=float(i)) for i in range(3)]
        store1.upsert(chunks=chunks, vectors=vectors)
        store1.save()

        # Load into new instance
        store2 = JsonlVectorStore(store_dir)
        store2.load()

        assert store2.count() == 3

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Load from nonexistent file doesn't crash."""
        store_dir = tmp_path / "nonexistent"
        store = JsonlVectorStore(store_dir)

        store.load()  # Should not raise

        assert store.count() == 0

    def test_roundtrip_preserves_chunks(self, tmp_path: Path):
        """Save and load preserves chunk data."""
        store_dir = tmp_path / "store"

        original_chunk = make_chunk(
            chunk_id="preserved-chunk",
            doc_id="preserved-doc",
            text="This text should be preserved",
            chunk_index=5,
            metadata={"custom": "metadata"},
        )
        original_vector = [0.1, 0.2, 0.3] * 42 + [0.4, 0.5]  # 128 dims

        store1 = JsonlVectorStore(store_dir)
        store1.upsert(chunks=[original_chunk], vectors=[original_vector])
        store1.save()

        store2 = JsonlVectorStore(store_dir)
        store2.load()

        results = store2.search(query_vector=original_vector, top_k=1)

        assert len(results) == 1
        loaded_chunk = results[0].chunk
        assert loaded_chunk.chunk_id == "preserved-chunk"
        assert loaded_chunk.doc_id == "preserved-doc"
        assert loaded_chunk.text == "This text should be preserved"
        assert loaded_chunk.chunk_index == 5


class TestJsonlVectorStoreSearch:
    """Search operations."""

    def test_search_returns_candidates(self, tmp_path: Path):
        """Search returns Candidate objects."""
        store_dir = tmp_path / "store"
        store = JsonlVectorStore(store_dir)

        chunk = make_chunk()
        vector = make_vector()
        store.upsert(chunks=[chunk], vectors=[vector])

        results = store.search(query_vector=vector, top_k=1)

        assert len(results) == 1
        assert hasattr(results[0], "chunk")
        assert hasattr(results[0], "score")

    def test_search_orders_by_similarity(self, tmp_path: Path):
        """Search orders by cosine similarity."""
        store_dir = tmp_path / "store"
        store = JsonlVectorStore(store_dir)

        chunk_similar = make_chunk(chunk_id="similar")
        chunk_dissimilar = make_chunk(chunk_id="dissimilar")

        vec_similar = [1.0] * 128
        vec_dissimilar = [-1.0] * 128

        store.upsert(
            chunks=[chunk_dissimilar, chunk_similar],
            vectors=[vec_dissimilar, vec_similar],
        )

        query_vec = [1.0] * 128
        results = store.search(query_vector=query_vec, top_k=2)

        assert results[0].chunk.chunk_id == "similar"

    def test_search_respects_top_k(self, tmp_path: Path):
        """Search returns at most top_k results."""
        store_dir = tmp_path / "store"
        store = JsonlVectorStore(store_dir)

        chunks = [make_chunk(chunk_id=f"chunk-{i}") for i in range(10)]
        vectors = [make_vector() for _ in range(10)]
        store.upsert(chunks=chunks, vectors=vectors)

        results = store.search(query_vector=make_vector(), top_k=3)

        assert len(results) == 3


class TestJsonlVectorStoreFiltering:
    """Metadata filtering."""

    def test_filter_by_metadata(self, tmp_path: Path):
        """Search filters by chunk metadata."""
        store_dir = tmp_path / "store"
        store = JsonlVectorStore(store_dir)

        chunk_a = make_chunk(chunk_id="a", metadata={"lang": "python"})
        chunk_b = make_chunk(chunk_id="b", metadata={"lang": "rust"})
        vectors = [make_vector(), make_vector()]

        store.upsert(chunks=[chunk_a, chunk_b], vectors=vectors)

        results = store.search(
            query_vector=make_vector(),
            top_k=10,
            where=Eq(field="lang", value="python"),
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "a"


class TestJsonlVectorStoreFileFormat:
    """JSONL file format verification."""

    def test_file_is_valid_jsonl(self, tmp_path: Path):
        """Saved file is valid JSONL (one JSON object per line)."""
        store_dir = tmp_path / "store"
        store = JsonlVectorStore(store_dir)

        chunks = [make_chunk(chunk_id=f"chunk-{i}") for i in range(3)]
        vectors = [make_vector() for _ in range(3)]
        store.upsert(chunks=chunks, vectors=vectors)
        store.save()

        data_file = store_dir / "chunks.jsonl"
        lines = data_file.read_text().strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            obj = json.loads(line)  # Should not raise
            assert "chunk" in obj
            assert "vector" in obj

    def test_load_externally_created_jsonl(self, tmp_path: Path):
        """Load can read externally created JSONL files."""
        store_dir = tmp_path / "external"
        store_dir.mkdir()
        data_file = store_dir / "chunks.jsonl"

        # Create JSONL manually using helper
        chunks = [make_chunk(chunk_id="external-1"), make_chunk(chunk_id="external-2")]
        vectors = [make_vector(value=0.1), make_vector(value=0.2)]
        write_chunks_jsonl(data_file, chunks, vectors)

        store = JsonlVectorStore(store_dir)
        store.load()

        assert store.count() == 2
