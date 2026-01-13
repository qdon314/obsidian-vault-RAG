"""Tests for DummyEmbedder adapter."""

from __future__ import annotations

from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.ports import Embedder


class TestDummyEmbedderBasics:
    """Basic embedding functionality."""

    def test_embed_single_text(self, dummy_embedder: Embedder):
        """Embeds a single text to correct dimensions."""
        vectors = dummy_embedder.embed_texts(["test text"])

        assert len(vectors) == 1
        assert len(vectors[0]) == 128

    def test_embed_multiple_texts(self):
        """Embeds multiple texts in batch."""
        embedder = DummyEmbedder(dim=64)
        texts = ["first", "second", "third"]

        vectors = embedder.embed_texts(texts)

        assert len(vectors) == 3
        assert all(len(v) == 64 for v in vectors)

    def test_embed_empty_list(self, dummy_embedder: Embedder):
        """Embedding empty list returns empty list."""
        vectors = dummy_embedder.embed_texts([])

        assert vectors == []

    def test_embed_empty_string(self, dummy_embedder: Embedder):
        """Empty string still produces a vector."""
        vectors = dummy_embedder.embed_texts([""])

        assert len(vectors) == 1
        assert len(vectors[0]) == 128


class TestDummyEmbedderDeterminism:
    """Deterministic behavior of embeddings."""

    def test_same_text_same_vector(self, dummy_embedder: Embedder):
        """Same text produces identical embedding."""
        text = "deterministic test"

        vec1 = dummy_embedder.embed_texts([text])[0]
        vec2 = dummy_embedder.embed_texts([text])[0]

        assert vec1 == vec2

    def test_different_texts_different_vectors(self, dummy_embedder: Embedder):
        """Different texts produce different embeddings."""
        vectors = dummy_embedder.embed_texts(["text A", "text B"])

        assert vectors[0] != vectors[1]

    def test_deterministic_across_instances(self):
        """Same text produces same vector across different embedder instances."""
        text = "consistent embedding"
        embedder1 = DummyEmbedder(dim=128)
        embedder2 = DummyEmbedder(dim=128)

        vec1 = embedder1.embed_texts([text])[0]
        vec2 = embedder2.embed_texts([text])[0]

        assert vec1 == vec2


class TestDummyEmbedderConfiguration:
    """Configuration and properties."""

    def test_custom_dimensionality(self):
        """Respects custom embedding dimension."""
        embedder = DummyEmbedder(dim=256)
        vectors = embedder.embed_texts(["test"])

        assert len(vectors[0]) == 256

    def test_model_name_property(self):
        """Exposes model name."""
        embedder = DummyEmbedder(model="custom-dummy-v2")

        assert embedder.model_name == "custom-dummy-v2"

    def test_default_model_name(self):
        """Default model name is set."""
        embedder = DummyEmbedder()

        assert embedder.model_name == "dummy-embedder-v1"


class TestDummyEmbedderVectorProperties:
    """Properties of generated vectors."""

    def test_values_in_expected_range(self, dummy_embedder: Embedder):
        """Embedding values are in [-1, 1] range."""
        vectors = dummy_embedder.embed_texts(["range test", "another test"])

        for vec in vectors:
            assert all(-1.0 <= v <= 1.0 for v in vec)

    def test_vectors_are_lists_of_floats(self, dummy_embedder: Embedder):
        """Vectors are lists of float values."""
        vectors = dummy_embedder.embed_texts(["test"])

        assert isinstance(vectors[0], list)
        assert all(isinstance(v, float) for v in vectors[0])

    def test_vectors_have_variation(self, dummy_embedder: Embedder):
        """Vectors have some variation in values (not all same)."""
        vectors = dummy_embedder.embed_texts(["test with variation"])

        # Check that not all values are identical
        unique_values = set(vectors[0])
        assert len(unique_values) > 1
