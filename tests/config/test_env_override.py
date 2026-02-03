"""Tests for environment variable override layer."""

from __future__ import annotations

from rag.config.env_override import apply_env_overrides


class TestApplyEnvOverrides:
    """Test that RAG_ env vars override parsed TOML dict."""

    def test_no_env_vars_returns_unchanged(self) -> None:
        raw = {"embeddings": {"backend": "openai"}}
        result = apply_env_overrides(raw, environ={})
        assert result == {"embeddings": {"backend": "openai"}}

    def test_flat_override(self) -> None:
        raw = {"embeddings": {"backend": "openai"}}
        env = {"RAG_EMBEDDINGS__BACKEND": "dummy"}
        result = apply_env_overrides(raw, environ=env)
        assert result["embeddings"]["backend"] == "dummy"

    def test_nested_override(self) -> None:
        raw = {"vectorstore": {"backend": "jsonl", "qdrant_url": None}}
        env = {"RAG_VECTORSTORE__QDRANT_URL": "http://qdrant:6333"}
        result = apply_env_overrides(raw, environ=env)
        assert result["vectorstore"]["qdrant_url"] == "http://qdrant:6333"

    def test_numeric_override(self) -> None:
        raw = {"retrieval": {"top_k": 8}}
        env = {"RAG_RETRIEVAL__TOP_K": "12"}
        result = apply_env_overrides(raw, environ=env)
        assert result["retrieval"]["top_k"] == 12

    def test_boolean_override(self) -> None:
        raw = {"rerank": {"enabled": True}}
        env = {"RAG_RERANK__ENABLED": "false"}
        result = apply_env_overrides(raw, environ=env)
        assert result["rerank"]["enabled"] is False

    def test_non_rag_env_vars_ignored(self) -> None:
        raw = {"embeddings": {"backend": "openai"}}
        env = {"HOME": "/home/user", "OPENAI_API_KEY": "sk-xxx"}
        result = apply_env_overrides(raw, environ=env)
        assert result == {"embeddings": {"backend": "openai"}}

    def test_creates_missing_section(self) -> None:
        raw: dict = {}
        env = {"RAG_VECTORSTORE__BACKEND": "qdrant"}
        result = apply_env_overrides(raw, environ=env)
        assert result["vectorstore"]["backend"] == "qdrant"
