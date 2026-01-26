"""Tests for JsonlQueryLogger adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rag.adapters.logging.jsonl_logger import JsonlQueryLogger
from rag.domain.models import Answer, Candidate, QueryTrace
from tests.conftest import make_candidate, make_chunk


def make_query_trace(
    trace_id: str = "test-trace-001",
    query: str = "test query",
    retrieved: list[Candidate] | None = None,
    answer: Answer | None = None,
) -> QueryTrace:
    """Create a test QueryTrace with minimal data."""
    return QueryTrace(
        trace_id=trace_id,
        query=query,
        created_at=datetime.now(UTC),
        top_k=10,
        retrieved=tuple(retrieved or []),
        reranked=(),
        keep_k=5,
        reranker="test",
        token_budget=4000,
        packed_chunk_ids=(),
        model="test-model",
        latency_ms=100,
        answer=answer,
    )


class TestJsonlQueryLoggerBasics:
    """Basic logging functionality."""

    def test_log_creates_file(self, tmp_path: Path):
        """Logging creates the log file."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        trace = make_query_trace()
        logger.log(trace)

        assert log_path.exists()

    def test_log_appends_to_file(self, tmp_path: Path):
        """Multiple logs append to the same file."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        logger.log(make_query_trace(trace_id="trace-1"))
        logger.log(make_query_trace(trace_id="trace-2"))
        logger.log(make_query_trace(trace_id="trace-3"))

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_log_creates_parent_directories(self, tmp_path: Path):
        """Logger creates parent directories if needed."""
        log_path = tmp_path / "nested" / "dir" / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        logger.log(make_query_trace())

        assert log_path.exists()


class TestJsonlQueryLoggerSerialization:
    """Trace serialization to JSON."""

    def test_log_is_valid_json(self, tmp_path: Path):
        """Each log line is valid JSON."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        logger.log(make_query_trace())

        line = log_path.read_text().strip()
        obj = json.loads(line)  # Should not raise
        assert isinstance(obj, dict)

    def test_log_includes_trace_fields(self, tmp_path: Path):
        """Log includes essential trace fields."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        trace = make_query_trace(
            trace_id="my-trace-id",
            query="What is Python?",
        )
        logger.log(trace)

        line = log_path.read_text().strip()
        obj = json.loads(line)

        assert obj["trace_id"] == "my-trace-id"
        assert obj["query"] == "What is Python?"
        assert "created_at" in obj
        assert obj["top_k"] == 10

    def test_log_serializes_candidates(self, tmp_path: Path):
        """Log correctly serializes Candidate objects."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        chunk = make_chunk(chunk_id="test-chunk", text="Chunk content")
        candidate = make_candidate(chunk=chunk, score=0.95)
        trace = make_query_trace(retrieved=[candidate])

        logger.log(trace)

        line = log_path.read_text().strip()
        obj = json.loads(line)

        assert len(obj["retrieved"]) == 1
        assert obj["retrieved"][0]["score"] == 0.95

    def test_log_serializes_answer(self, tmp_path: Path):
        """Log correctly serializes Answer objects."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        answer = Answer(
            query="test",
            text="This is the answer.",
        )
        trace = make_query_trace(answer=answer)

        logger.log(trace)

        line = log_path.read_text().strip()
        obj = json.loads(line)

        assert obj["answer"]["text"] == "This is the answer."


class TestJsonlQueryLoggerRedaction:
    """Text redaction functionality."""

    def test_redact_text_removes_content(self, tmp_path: Path):
        """Redaction replaces text fields with [REDACTED]."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path, redact_text=True)

        chunk = make_chunk(text="Sensitive content here")
        candidate = make_candidate(chunk=chunk)
        trace = make_query_trace(retrieved=[candidate])

        logger.log(trace)

        content = log_path.read_text()
        assert "Sensitive content" not in content
        assert "[REDACTED]" in content

    def test_redact_replaces_answer_field(self, tmp_path: Path):
        """Redaction replaces entire answer field."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path, redact_text=True)

        answer = Answer(query="test", text="Secret answer")
        trace = make_query_trace(answer=answer)

        logger.log(trace)

        line = log_path.read_text().strip()
        obj = json.loads(line)

        # The "answer" key is one of the redacted fields, so entire value is replaced
        assert "answer" in obj
        assert obj["answer"] == "[REDACTED]"

    def test_no_redaction_by_default(self, tmp_path: Path):
        """Without redact_text, content is preserved."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path, redact_text=False)

        chunk = make_chunk(text="Visible content")
        candidate = make_candidate(chunk=chunk)
        trace = make_query_trace(retrieved=[candidate])

        logger.log(trace)

        content = log_path.read_text()
        assert "Visible content" in content


class TestJsonlQueryLoggerHelpers:
    """Helper method functionality."""

    def test_new_trace_id_is_unique(self, tmp_path: Path):
        """new_trace_id generates unique IDs."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        ids = [logger.new_trace_id() for _ in range(100)]

        assert len(set(ids)) == 100  # All unique

    def test_new_trace_id_is_hex_string(self, tmp_path: Path):
        """new_trace_id returns hex string."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        trace_id = logger.new_trace_id()

        assert isinstance(trace_id, str)
        int(trace_id, 16)  # Should be valid hex

    def test_now_iso_returns_iso_format(self, tmp_path: Path):
        """now_iso returns ISO format timestamp."""
        log_path = tmp_path / "queries.jsonl"
        logger = JsonlQueryLogger(log_path)

        timestamp = logger.now_iso()

        assert isinstance(timestamp, str)
        # Should be parseable as ISO format
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
