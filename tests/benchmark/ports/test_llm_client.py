"""Tests for the LLMClient port protocol and LLMResponse model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from benchmark.domain.models import StageConfig
from benchmark.ports.llm_client import LLMClient, LLMResponse


class _StubLLMClient:
    """Minimal stub that structurally satisfies LLMClient."""

    def complete(self, prompt: str, config: StageConfig) -> LLMResponse:
        return LLMResponse(
            text="stub response",
            model=config.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


class TestLLMResponse:
    def test_frozen(self) -> None:
        resp = LLMResponse(text="hello", model="test", usage={})
        with pytest.raises(FrozenInstanceError):
            resp.text = "changed"  # type: ignore[misc]

    def test_default_usage(self) -> None:
        resp = LLMResponse(text="hello", model="test")
        assert resp.usage == {}

    def test_fields(self) -> None:
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        resp = LLMResponse(text="answer", model="gpt-4o", usage=usage)
        assert resp.text == "answer"
        assert resp.model == "gpt-4o"
        assert resp.usage == usage


class TestLLMClientProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        stub = _StubLLMClient()
        assert isinstance(stub, LLMClient)

    def test_stub_returns_llm_response(self) -> None:
        stub = _StubLLMClient()
        config = StageConfig(model="test-model")
        resp = stub.complete("hello", config)
        assert isinstance(resp, LLMResponse)
        assert resp.model == "test-model"
