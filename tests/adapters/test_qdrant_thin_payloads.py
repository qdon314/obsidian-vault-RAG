"""Tests for Qdrant thin payload helpers.

Verifies that _thin_payload and _chunk_from_thin_payload correctly
convert between Chunk objects and minimal Qdrant payloads.
"""

from __future__ import annotations

from rag.adapters.vectorstores.qdrant_store import _chunk_from_thin_payload, _thin_payload
from tests.conftest import make_chunk


class TestThinPayload:
    """Building a thin payload from a Chunk."""

    def test_excludes_text(self) -> None:
        """Thin payload does not contain the text field."""
        chunk = make_chunk(text="This should not appear in payload")
        payload = _thin_payload(chunk)

        assert "text" not in payload

    def test_includes_ids(self) -> None:
        """Thin payload includes chunk_id and doc_id."""
        chunk = make_chunk(chunk_id="c1", doc_id="d1")
        payload = _thin_payload(chunk)

        assert payload["chunk_id"] == "c1"
        assert payload["doc_id"] == "d1"

    def test_includes_structural_fields(self) -> None:
        """Thin payload includes structural fields (index, offsets, headings)."""
        chunk = make_chunk(
            chunk_index=3,
            start_char=100,
            end_char=200,
            section_heading="Introduction",
            section_path="# Docs / ## Introduction",
            language="en",
        )
        payload = _thin_payload(chunk)

        assert payload["chunk_index"] == 3
        assert payload["start_char"] == 100
        assert payload["end_char"] == 200
        assert payload["section_heading"] == "Introduction"
        assert payload["section_path"] == "# Docs / ## Introduction"
        assert payload["language"] == "en"

    def test_spreads_metadata(self) -> None:
        """Chunk metadata fields are spread into the payload for filtering."""
        chunk = make_chunk(metadata={"citation": "Smith 2024", "tags": ["ml", "ai"]})
        payload = _thin_payload(chunk)

        assert payload["citation"] == "Smith 2024"
        assert payload["tags"] == ["ml", "ai"]


class TestChunkFromThinPayload:
    """Reconstructing a stub Chunk from a thin payload."""

    def test_empty_text(self) -> None:
        """Reconstructed chunk has empty text."""
        payload = {"chunk_id": "c1", "doc_id": "d1", "chunk_index": 0}
        chunk = _chunk_from_thin_payload(payload)

        assert chunk.text == ""

    def test_preserves_ids(self) -> None:
        """Reconstructed chunk has correct chunk_id and doc_id."""
        payload = {"chunk_id": "c1", "doc_id": "d1", "chunk_index": 0}
        chunk = _chunk_from_thin_payload(payload)

        assert chunk.chunk_id == "c1"
        assert chunk.doc_id == "d1"

    def test_preserves_structural_fields(self) -> None:
        """Structural fields are correctly reconstructed."""
        payload = {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_index": 5,
            "start_char": 100,
            "end_char": 200,
            "section_heading": "Methods",
            "section_path": "# Paper / ## Methods",
            "language": "en",
        }
        chunk = _chunk_from_thin_payload(payload)

        assert chunk.chunk_index == 5
        assert chunk.start_char == 100
        assert chunk.end_char == 200
        assert chunk.section_heading == "Methods"
        assert chunk.section_path == "# Paper / ## Methods"
        assert chunk.language == "en"

    def test_extra_fields_become_metadata(self) -> None:
        """Fields not in Chunk's first-class fields go into metadata."""
        payload = {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_index": 0,
            "citation": "Smith 2024",
            "tags": ["ml"],
        }
        chunk = _chunk_from_thin_payload(payload)

        assert chunk.metadata["citation"] == "Smith 2024"
        assert chunk.metadata["tags"] == ["ml"]


class TestRoundTrip:
    """Thin payload → stub chunk preserves all non-text data."""

    def test_roundtrip_ids_and_metadata(self) -> None:
        """IDs and metadata survive thin payload roundtrip."""
        original = make_chunk(
            chunk_id="c42",
            doc_id="d7",
            chunk_index=2,
            start_char=50,
            end_char=150,
            section_heading="Results",
            metadata={"citation_key": "paper_x"},
        )

        payload = _thin_payload(original)
        stub = _chunk_from_thin_payload(payload)

        assert stub.chunk_id == original.chunk_id
        assert stub.doc_id == original.doc_id
        assert stub.chunk_index == original.chunk_index
        assert stub.start_char == original.start_char
        assert stub.end_char == original.end_char
        assert stub.section_heading == original.section_heading
        assert stub.metadata["citation_key"] == "paper_x"
        assert stub.text == ""  # text is stripped
