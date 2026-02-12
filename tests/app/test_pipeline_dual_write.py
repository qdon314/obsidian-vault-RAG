"""Tests for pipeline dual-write with optional ChunkStore.

Verifies that index_document and index_documents write to the ChunkStore
when provided, and skip it when None.
"""

from __future__ import annotations

from collections.abc import Mapping

from rag.app.pipeline import index_document, index_documents
from rag.domain.models import Chunk, Document
from rag.ports import Chunker, Embedder, VectorStore
from tests.conftest import FakeChunkStore, make_document

# =============================================================================
# index_document tests
# =============================================================================


class TestIndexDocumentDualWrite:
    """index_document writes to ChunkStore when provided."""

    def test_chunk_store_receives_chunks(
        self,
        dummy_embedder: Embedder,
        fixed_chunker: Chunker,
        in_memory_store: VectorStore,
        fake_chunk_store: FakeChunkStore,
    ) -> None:
        """When chunk_store is given, store_chunks is called."""
        doc = make_document(doc_id="d1", text="Hello world")

        count = index_document(
            doc,
            chunker=fixed_chunker,
            embedder=dummy_embedder,
            store=in_memory_store,
            chunk_store=fake_chunk_store,
        )

        assert count >= 1
        assert len(fake_chunk_store.stored_chunks) >= 1
        assert fake_chunk_store.stored_chunks[0].doc_id == "d1"

    def test_no_chunk_store_is_noop(
        self,
        dummy_embedder: Embedder,
        fixed_chunker: Chunker,
        in_memory_store: VectorStore,
    ) -> None:
        """When chunk_store is None, only vector store is written to."""
        doc = make_document(doc_id="d1", text="Hello world")

        count = index_document(
            doc,
            chunker=fixed_chunker,
            embedder=dummy_embedder,
            store=in_memory_store,
            chunk_store=None,
        )

        assert count >= 1
        assert in_memory_store.count() >= 1

    def test_empty_doc_skips_both_stores(
        self,
        dummy_embedder: Embedder,
        in_memory_store: VectorStore,
        fake_chunk_store: FakeChunkStore,
    ) -> None:
        """Empty document produces no writes to either store."""
        doc = make_document(doc_id="d1", text="")

        # Chunker that returns nothing for empty docs
        class _EmptyChunker:
            def chunk(
                self, doc: Document, *, metadata: Mapping[str, object] | None = None
            ) -> list[Chunk]:
                return []

            def get_config(self) -> dict[str, object]:
                return {}

        count = index_document(
            doc,
            chunker=_EmptyChunker(),
            embedder=dummy_embedder,
            store=in_memory_store,
            chunk_store=fake_chunk_store,
        )

        assert count == 0
        assert in_memory_store.count() == 0
        assert len(fake_chunk_store.stored_chunks) == 0


# =============================================================================
# index_documents tests
# =============================================================================


class TestIndexDocumentsDualWrite:
    """index_documents writes to ChunkStore when provided."""

    def test_batch_writes_to_chunk_store(
        self,
        dummy_embedder: Embedder,
        fixed_chunker: Chunker,
        in_memory_store: VectorStore,
        fake_chunk_store: FakeChunkStore,
    ) -> None:
        """All chunks across documents are written to the ChunkStore."""
        docs = [
            make_document(doc_id="d1", text="First"),
            make_document(doc_id="d2", text="Second"),
            make_document(doc_id="d3", text="Third"),
        ]

        count = index_documents(
            docs,
            chunker=fixed_chunker,
            embedder=dummy_embedder,
            store=in_memory_store,
            chunk_store=fake_chunk_store,
        )

        assert count >= 3
        assert len(fake_chunk_store.stored_chunks) >= 3
        stored_doc_ids = {ch.doc_id for ch in fake_chunk_store.stored_chunks}
        assert stored_doc_ids >= {"d1", "d2", "d3"}

    def test_batch_without_chunk_store(
        self,
        dummy_embedder: Embedder,
        fixed_chunker: Chunker,
        in_memory_store: VectorStore,
    ) -> None:
        """Batch indexing without chunk_store only writes vector store."""
        docs = [make_document(doc_id="d1", text="One")]

        count = index_documents(
            docs,
            chunker=fixed_chunker,
            embedder=dummy_embedder,
            store=in_memory_store,
            chunk_store=None,
        )

        assert count >= 1
        assert in_memory_store.count() >= 1
