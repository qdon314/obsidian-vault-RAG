"""Test that query_runner stores context_before/after in compression trace metadata."""
from __future__ import annotations

from unittest.mock import MagicMock

from rag.app.query_runner import run_query
from rag.domain.models import (
    Answer,
    CompressionResult,
    ContextPack,
    QueryTrace,
)


def _make_pack(text: str, tokens: int = 100) -> ContextPack:
    return ContextPack(
        query="q",
        chunks=(),
        rendered_context=text,
        citations=(),
        token_budget=1000,
        metadata={"tokens_used_est": tokens},
    )


def _make_mocks(original_text: str, compressed_text: str):
    """Return a dict of minimal mocks for run_query."""
    original_pack = _make_pack(original_text, tokens=100)
    compressed_pack = _make_pack(compressed_text, tokens=80)

    compression_result = CompressionResult(
        context_pack=compressed_pack,
        successful=True,
        tokens_before=100,
        tokens_after=80,
        savings_pct=0.2,
        latency_ms=50,
        adapter="scaledown",
    )

    mock_compressor = MagicMock()
    mock_compressor.compress.return_value = compression_result

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = []

    mock_context_builder = MagicMock()
    mock_context_builder.build.return_value = original_pack

    mock_generator = MagicMock()
    mock_generator.generate.return_value = Answer(query="q", text="answer")

    captured_traces: list[QueryTrace] = []
    mock_logger = MagicMock()
    mock_logger.log.side_effect = captured_traces.append

    return {
        "retriever": mock_retriever,
        "reranker": mock_reranker,
        "context_builder": mock_context_builder,
        "generator": mock_generator,
        "logger": mock_logger,
        "compressor": mock_compressor,
        "captured_traces": captured_traces,
    }


def test_compression_meta_includes_context_before_and_after():
    """run_query stores context_before and context_after in compression trace metadata."""
    original_text = "CONTEXT:\n[1]\noriginal chunk\n"
    compressed_text = "CONTEXT:\n[1]\ncompressed chunk\n"

    mocks = _make_mocks(original_text, compressed_text)

    run_query(
        "test query",
        retriever=mocks["retriever"],
        reranker=mocks["reranker"],
        context_builder=mocks["context_builder"],
        generator=mocks["generator"],
        logger=mocks["logger"],
        compressor=mocks["compressor"],
        top_k=5,
        keep_k=None,
        token_budget=1000,
    )

    assert len(mocks["captured_traces"]) == 1
    trace = mocks["captured_traces"][0]
    compression = trace.metadata.get("compression", {})

    assert compression.get("context_before") == original_text
    assert compression.get("context_after") == compressed_text
