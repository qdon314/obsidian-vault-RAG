"""Tests for FixedChunker adapter."""

from __future__ import annotations

from rag.adapters.chunking.fixed import FixedChunker
from rag.ports import Chunker
from tests.conftest import SAMPLE_LONG_TEXT, make_document


class TestFixedChunkerBasics:
    """Basic chunking behavior."""

    def test_chunk_returns_list(self, fixed_chunker: Chunker):
        """chunk() should return a list of Chunk objects."""
        doc = make_document(text="Sample text here")
        chunks = fixed_chunker.chunk(doc)
        assert isinstance(chunks, list)

    def test_chunk_empty_document(self, fixed_chunker: Chunker):
        """Empty document text should return empty list."""
        doc = make_document(text="")
        chunks = fixed_chunker.chunk(doc)
        assert chunks == []

    def test_chunk_whitespace_only(self, fixed_chunker: Chunker):
        """Whitespace-only document should return empty list."""
        doc = make_document(text="   \n\t  ")
        chunks = fixed_chunker.chunk(doc)
        assert chunks == []

    def test_chunk_single_chunk(self):
        """Short text should produce single chunk."""
        chunker = FixedChunker(chunk_size=100, overlap=0)
        doc = make_document(text="Short text")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == "Short text"

    def test_chunk_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        chunker = FixedChunker(chunk_size=50, overlap=10)
        doc = make_document(text="a" * 200)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunk_preserves_doc_id(self, fixed_chunker: Chunker):
        """All chunks should reference the source document."""
        doc = make_document(doc_id="my-doc-id", text=SAMPLE_LONG_TEXT)
        chunks = fixed_chunker.chunk(doc)
        assert all(c.doc_id == "my-doc-id" for c in chunks)


class TestFixedChunkerOverlap:
    """Overlap behavior between chunks."""

    def test_overlap_creates_shared_content(self):
        """Overlapping chunks should share text at boundaries."""
        text = "0123456789" * 10  # 100 chars
        chunker = FixedChunker(chunk_size=30, overlap=10)
        doc = make_document(text=text)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 2
        # End of first chunk should overlap with start of second
        first_end = chunks[0].text[-10:]
        assert first_end in chunks[1].text

    def test_zero_overlap(self):
        """With zero overlap, chunks should not share content."""
        text = "AAAAAAAAAA" + "BBBBBBBBBB" + "CCCCCCCCCC"  # 30 chars
        chunker = FixedChunker(chunk_size=10, overlap=0)
        doc = make_document(text=text)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 3
        assert chunks[0].text == "AAAAAAAAAA"
        assert chunks[1].text == "BBBBBBBBBB"
        assert chunks[2].text == "CCCCCCCCCC"

    def test_overlap_larger_than_step_handled(self):
        """Large overlap relative to chunk size should still work."""
        chunker = FixedChunker(chunk_size=100, overlap=90)
        doc = make_document(text="x" * 500)
        chunks = chunker.chunk(doc)
        # Should produce many overlapping chunks
        assert len(chunks) > 10


class TestFixedChunkerMetadata:
    """Metadata handling in chunks."""

    def test_chunk_id_contains_doc_id(self, fixed_chunker: Chunker):
        """Chunk IDs should contain the document ID."""
        doc = make_document(doc_id="test-doc-xyz")
        chunks = fixed_chunker.chunk(doc)
        if chunks:
            assert "test-doc-xyz" in chunks[0].chunk_id

    def test_chunk_ids_are_unique(self):
        """All chunks from a document should have unique IDs."""
        chunker = FixedChunker(chunk_size=50, overlap=10)
        doc = make_document(text=SAMPLE_LONG_TEXT)
        chunks = chunker.chunk(doc)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_index_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        chunker = FixedChunker(chunk_size=50, overlap=10)
        doc = make_document(text=SAMPLE_LONG_TEXT)
        chunks = chunker.chunk(doc)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_propagation(self, fixed_chunker: Chunker):
        """Document metadata should propagate to chunks."""
        doc = make_document(
            text="Test text content here",
            metadata={"author": "Test Author", "source": "test"},
        )
        chunks = fixed_chunker.chunk(doc)

        assert chunks[0].metadata.get("author") == "Test Author"
        assert chunks[0].metadata.get("source") == "test"

    def test_chunking_metadata_added(self, fixed_chunker: Chunker):
        """Chunks should include chunking strategy metadata."""
        doc = make_document(text="Some content")
        chunks = fixed_chunker.chunk(doc)

        assert chunks[0].metadata.get("chunking_strategy") == "test_v1"
        assert chunks[0].metadata.get("chunk_size") == 100
        assert chunks[0].metadata.get("overlap") == 20


class TestFixedChunkerProvenance:
    """Character position tracking."""

    def test_start_end_char_set(self, fixed_chunker: Chunker):
        """Chunks should have start_char and end_char set."""
        doc = make_document(text="Test document content")
        chunks = fixed_chunker.chunk(doc)

        assert chunks[0].start_char is not None
        assert chunks[0].end_char is not None

    def test_first_chunk_starts_at_zero(self, fixed_chunker: Chunker):
        """First chunk should start at character 0."""
        doc = make_document(text="Test document content")
        chunks = fixed_chunker.chunk(doc)

        assert chunks[0].start_char == 0

    def test_char_positions_enable_extraction(self):
        """Character positions should allow extracting original text."""
        original_text = "Hello world, this is a test document."
        chunker = FixedChunker(chunk_size=15, overlap=0)
        doc = make_document(text=original_text)
        chunks = chunker.chunk(doc)

        # Verify we can extract the chunk text from the original
        for chunk in chunks:
            extracted = original_text[chunk.start_char : chunk.end_char]
            assert extracted.strip() == chunk.text.strip()
