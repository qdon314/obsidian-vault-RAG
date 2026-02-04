"""Shared test fixtures and factory functions for adapter tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.context_building.simple_context_builder import SimpleContextBuilder
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.adapters.ingestion.loaders.text_loader import TextLoader
from rag.adapters.reranking.rerank_heuristic import HeuristicReranker
from rag.adapters.reranking.rerank_noop import NoOpReranker
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore
from rag.domain.models import Candidate, Chunk, Document
from rag.ports import (
    Chunker,
    ContextBuilder,
    Embedder,
    Reranker,
    VectorStore,
)

# =============================================================================
# Test Data Constants
# =============================================================================

SAMPLE_TEXT = """Machine learning is a subset of artificial intelligence.
It focuses on building systems that can learn from data.
Deep learning uses neural networks with many layers."""

SAMPLE_LONG_TEXT = SAMPLE_TEXT * 20  # ~3000 chars, for chunking tests

SAMPLE_VECTOR_DIM = 128
SAMPLE_VECTOR = [0.1] * SAMPLE_VECTOR_DIM


# =============================================================================
# Adapter Fixtures
# =============================================================================


@pytest.fixture
def dummy_embedder() -> Embedder:
    """DummyEmbedder with test-appropriate dimensions."""
    return DummyEmbedder(dim=SAMPLE_VECTOR_DIM, model="test-embedder")


@pytest.fixture
def fixed_chunker() -> Chunker:
    """FixedChunker with small test-friendly chunk size."""
    return FixedChunker(chunk_size=100, overlap=20, strategy_name="test_v1")


@pytest.fixture
def in_memory_store() -> VectorStore:
    """Fresh InMemoryVectorStore for each test."""
    return InMemoryVectorStore()


@pytest.fixture
def heuristic_reranker() -> Reranker:
    """HeuristicReranker with test defaults."""
    return HeuristicReranker(overlap_weight=0.15, diversify=True, max_per_doc=3)


@pytest.fixture
def noop_reranker() -> Reranker:
    """NoOpReranker instance."""
    return NoOpReranker()


@pytest.fixture
def simple_context_builder() -> ContextBuilder:
    """SimpleContextBuilder with test defaults."""
    return SimpleContextBuilder(max_chunks=5, dedupe=True)


@pytest.fixture
def text_loader() -> TextLoader:
    """TextLoader with default settings."""
    return TextLoader()


# =============================================================================
# Domain Model Factory Functions
# =============================================================================


def make_document(
    doc_id: str = "test-doc-001",
    text: str = SAMPLE_TEXT,
    source: str = "test",
    uri: str = "/test/doc.md",
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> Document:
    """Create a test Document with sensible defaults."""
    return Document(
        doc_id=doc_id,
        text=text,
        source=source,
        uri=uri,
        metadata=metadata or {},
        created_at=created_at or datetime.now(UTC),
    )


def make_chunk(
    chunk_id: str = "test-doc-001:chunk:0",
    doc_id: str = "test-doc-001",
    text: str = SAMPLE_TEXT,
    chunk_index: int = 0,
    start_char: int | None = 0,
    end_char: int | None = None,
    section_heading: str | None = None,
    section_path: str | None = None,
    language: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Create a test Chunk with sensible defaults."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char or (start_char + len(text) if start_char is not None else len(text)),
        section_heading=section_heading,
        section_path=section_path,
        language=language,
        metadata=metadata or {},
    )


def make_candidate(
    chunk: Chunk | None = None,
    score: float = 0.85,
    rerank_score: float | None = None,
    debug: dict[str, Any] | None = None,
) -> Candidate:
    """Create a test Candidate with sensible defaults."""
    return Candidate(
        chunk=chunk or make_chunk(),
        score=score,
        rerank_score=rerank_score,
        debug=debug or {},
    )


def make_vector(dim: int = SAMPLE_VECTOR_DIM, value: float = 0.1) -> list[float]:
    """Create a test vector with uniform values."""
    return [value] * dim


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_documents() -> list[Document]:
    """Generate sample documents for testing."""
    return [
        make_document(
            doc_id="doc-1",
            text="Python is a programming language. It is widely used for data science.",
            uri="/docs/python.md",
        ),
        make_document(
            doc_id="doc-2",
            text="Machine learning is a subset of artificial intelligence. It uses statistical methods.",
            uri="/docs/ml.md",
        ),
        make_document(
            doc_id="doc-3",
            text="Vector databases store embeddings efficiently. They support similarity search.",
            uri="/docs/vectors.md",
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Generate sample chunks for testing."""
    return [
        make_chunk(
            chunk_id="doc-1:chunk:0",
            doc_id="doc-1",
            text="Python is a programming language.",
            chunk_index=0,
            metadata={"uri": "/docs/python.md"},
        ),
        make_chunk(
            chunk_id="doc-1:chunk:1",
            doc_id="doc-1",
            text="It is widely used for data science.",
            chunk_index=1,
            metadata={"uri": "/docs/python.md"},
        ),
        make_chunk(
            chunk_id="doc-2:chunk:0",
            doc_id="doc-2",
            text="Machine learning is a subset of AI.",
            chunk_index=0,
            metadata={"uri": "/docs/ml.md"},
        ),
        make_chunk(
            chunk_id="doc-3:chunk:0",
            doc_id="doc-3",
            text="Vector databases store embeddings.",
            chunk_index=0,
            metadata={"uri": "/docs/vectors.md"},
        ),
    ]


@pytest.fixture
def sample_candidates(sample_chunks: list[Chunk]) -> list[Candidate]:
    """Generate sample candidates from sample chunks."""
    scores = [0.95, 0.85, 0.75, 0.65]
    return [
        make_candidate(chunk=chunk, score=score)
        for chunk, score in zip(sample_chunks, scores, strict=False)
    ]


# =============================================================================
# Filesystem Fixtures
# =============================================================================


@pytest.fixture
def sample_vault(tmp_path: Path) -> Path:
    """Create a minimal test vault structure."""
    vault = tmp_path / "test_vault"
    vault.mkdir()

    # Simple markdown notes
    (vault / "note1.md").write_text(
        """---
title: First Note
tags: [python, testing]
---
# First Note

This is the first note with some content about Python programming.
"""
    )

    (vault / "note2.md").write_text(
        """# Second Note

A simple note without frontmatter about machine learning.
"""
    )

    # Plain text file
    (vault / "readme.txt").write_text("Plain text file content for testing.")

    # Subdirectory with nested note
    subdir = vault / "subfolder"
    subdir.mkdir()
    (subdir / "nested.md").write_text(
        """# Nested Note

Content inside a subdirectory.
"""
    )

    # Hidden file (should be skipped by ingestor)
    (vault / ".hidden.md").write_text("Hidden content that should be ignored.")

    # Unsupported extension (should be skipped)
    (vault / "data.csv").write_text("col1,col2\nval1,val2")

    return vault


@pytest.fixture
def jsonl_store_path(tmp_path: Path) -> Path:
    """Path to a temporary JSONL vector store file."""
    return tmp_path / "chunks.jsonl"


@pytest.fixture
def query_log_path(tmp_path: Path) -> Path:
    """Path to a temporary query log file."""
    return tmp_path / "queries.jsonl"


@pytest.fixture
def temp_vault(tmp_path: Path) -> Path:
    """Create an empty temporary vault directory for testing."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


# =============================================================================
# Obsidian Note Content Constants
# =============================================================================

SIMPLE_NOTE_CONTENT = "# Simple Note\n\nJust some content."

NOTE_WITH_FRONTMATTER = """---
title: Test Note
tags: [python, test]
---
# Content
Body text here."""

NOTE_WITH_INLINE_TAGS = "Content with #inline and #tags here."

NOTE_WITH_WIKILINKS = "See [[Other Note]] and [[Target|alias]] for more."

MOC_NOTE_CONTENT = "# Topic MOC\n\nIndex of topics."

CODE_BLOCK_NOTE = """# Code Example
```python
# This [[wikilink]] should stay
def foo():
    pass
```
Outside code [[stripped]] link."""


# =============================================================================
# Helper Functions for Test Data Creation
# =============================================================================


def write_note(vault: Path, name: str, content: str) -> Path:
    """Write a markdown note to a vault directory."""
    note = vault / name
    note.write_text(content)
    return note


def write_chunks_jsonl(
    path: Path,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    """Write chunks and vectors to a JSONL file (vector store format)."""
    with path.open("w", encoding="utf-8") as f:
        for chunk, vec in zip(chunks, vectors, strict=True):
            row = {
                "chunk": {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "section_heading": chunk.section_heading,
                    "section_path": chunk.section_path,
                    "language": chunk.language,
                    "metadata": dict(chunk.metadata),
                },
                "vector": vec,
            }
            f.write(json.dumps(row) + "\n")
