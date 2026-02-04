"""Tests for index manifest validation."""

from __future__ import annotations

import pytest

from rag.app.manifest_validation import validate_index
from rag.domain.errors import IndexIncompatibleError
from rag.domain.index_manifest import IndexManifest


class _FakeEmbedder:
    """Minimal embedder satisfying the protocol for testing."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts, *, metadata=None):  # type: ignore[ANN001, ANN003]
        return [[0.0] * 128 for _ in texts]


class TestValidateIndex:
    def test_passes_when_model_matches(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=5,
            chunk_count=20,
            embedding={"model": "text-embedding-3-large"},
        )
        # Should not raise
        validate_index(m, _FakeEmbedder("text-embedding-3-large"))

    def test_raises_on_model_mismatch(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=5,
            chunk_count=20,
            embedding={"model": "text-embedding-3-large"},
        )
        with pytest.raises(IndexIncompatibleError, match="text-embedding-3-large"):
            validate_index(m, _FakeEmbedder("text-embedding-3-small"))

    def test_raises_on_failed_status(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=0,
            chunk_count=0,
            status="failed",
        )
        with pytest.raises(IndexIncompatibleError, match="failed"):
            validate_index(m, _FakeEmbedder("text-embedding-3-large"))

    def test_passes_when_manifest_has_no_model(self) -> None:
        """Old manifests without embedding.model should not block queries."""
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=5,
            chunk_count=20,
            embedding={"backend": "openai"},
        )
        # Should not raise (cannot validate, so allow)
        validate_index(m, _FakeEmbedder("text-embedding-3-large"))
