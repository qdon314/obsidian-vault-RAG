"""Tests for OpenAI client hygiene: single-instance client, timeouts, retries."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag.adapters.embedding.openai_embedder import OpenAIEmbedder
from rag.adapters.generation.openai_chat import OpenAIChatGenerator

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class TestEmbedderClientReuse:
    """Client is created once, not per embed_texts call."""

    @patch("rag.adapters.embedding.openai_embedder.OpenAI")
    def test_client_created_once(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Simulate API response
        item = MagicMock()
        item.embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = MagicMock(data=[item])

        embedder = OpenAIEmbedder(api_key="test-key")

        # OpenAI() called exactly once during __post_init__
        assert mock_openai_cls.call_count == 1

        embedder.embed_texts(["first call"])
        embedder.embed_texts(["second call"])

        # Still only one OpenAI() instantiation
        assert mock_openai_cls.call_count == 1
        assert mock_client.embeddings.create.call_count == 2


class TestEmbedderTimeout:
    """Timeout is passed to the OpenAI client."""

    @patch("rag.adapters.embedding.openai_embedder.OpenAI")
    def test_timeout_configured(self, mock_openai_cls: MagicMock) -> None:
        OpenAIEmbedder(api_key="test-key", timeout=15.0)

        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            timeout=15.0,
            max_retries=3,
        )

    @patch("rag.adapters.embedding.openai_embedder.OpenAI")
    def test_default_timeout(self, mock_openai_cls: MagicMock) -> None:
        OpenAIEmbedder(api_key="test-key")

        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            timeout=30.0,
            max_retries=3,
        )


class TestEmbedderRetry:
    """SDK retry is enabled with configurable max_retries."""

    @patch("rag.adapters.embedding.openai_embedder.OpenAI")
    def test_max_retries_configured(self, mock_openai_cls: MagicMock) -> None:
        OpenAIEmbedder(api_key="test-key", max_retries=5)

        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            timeout=30.0,
            max_retries=5,
        )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TestGeneratorClientReuse:
    """Client is created once, not per generate call."""

    @patch("rag.adapters.generation.openai_chat.OpenAI")
    def test_client_created_once(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Simulate API response
        choice = MagicMock()
        choice.message.content = "test answer"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])

        generator = OpenAIChatGenerator(api_key="test-key")

        assert mock_openai_cls.call_count == 1

        # Build minimal ContextPack
        context = MagicMock()
        context.rendered_context = "some context"
        context.citations = []

        generator.generate("q1", context)
        generator.generate("q2", context)

        assert mock_openai_cls.call_count == 1
        assert mock_client.chat.completions.create.call_count == 2


class TestGeneratorTimeout:
    """Timeout is passed to the OpenAI client."""

    @patch("rag.adapters.generation.openai_chat.OpenAI")
    def test_timeout_configured(self, mock_openai_cls: MagicMock) -> None:
        OpenAIChatGenerator(api_key="test-key", timeout=120.0)

        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            timeout=120.0,
            max_retries=3,
        )

    @patch("rag.adapters.generation.openai_chat.OpenAI")
    def test_default_timeout(self, mock_openai_cls: MagicMock) -> None:
        OpenAIChatGenerator(api_key="test-key")

        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            timeout=60.0,
            max_retries=3,
        )


class TestGeneratorRetry:
    """SDK retry is enabled with configurable max_retries."""

    @patch("rag.adapters.generation.openai_chat.OpenAI")
    def test_max_retries_configured(self, mock_openai_cls: MagicMock) -> None:
        OpenAIChatGenerator(api_key="test-key", max_retries=1)

        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            timeout=60.0,
            max_retries=1,
        )
