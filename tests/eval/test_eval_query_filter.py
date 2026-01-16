"""Tests for EvalQuery.get_filter() method."""

from __future__ import annotations

import logging

import pytest

from rag.domain.filters import And, Eq, In
from rag.eval.schema import Difficulty, EvalQuery, QueryType


class TestEvalQueryGetFilter:
    """Tests for EvalQuery.get_filter() method."""

    def test_get_filter_simple_eq(self) -> None:
        """Simple Eq filter in metadata is deserialized correctly."""
        query = EvalQuery(
            qid="test_001",
            query="What is Python?",
            relevant_chunk_ids={"chunk1"},
            metadata={"filter": {"type": "Eq", "field": "doc_id", "value": "doc123"}},
        )
        result = query.get_filter()
        assert result == Eq(field="doc_id", value="doc123")

    def test_get_filter_in(self) -> None:
        """In filter in metadata is deserialized correctly."""
        query = EvalQuery(
            qid="test_002",
            query="Compare Python and Rust",
            relevant_chunk_ids={"chunk1", "chunk2"},
            metadata={
                "filter": {
                    "type": "In",
                    "field": "doc_id",
                    "values": ["python_doc", "rust_doc"],
                }
            },
        )
        result = query.get_filter()
        assert result == In(field="doc_id", values=["python_doc", "rust_doc"])

    def test_get_filter_complex_and(self) -> None:
        """Complex And filter in metadata is deserialized correctly."""
        query = EvalQuery(
            qid="test_003",
            query="Test query",
            relevant_chunk_ids={"chunk1"},
            metadata={
                "filter": {
                    "type": "And",
                    "clauses": [
                        {"type": "Eq", "field": "language", "value": "python"},
                        {"type": "In", "field": "tags", "values": ["ml", "ai"]},
                    ],
                }
            },
        )
        result = query.get_filter()
        expected = And(
            clauses=[
                Eq(field="language", value="python"),
                In(field="tags", values=["ml", "ai"]),
            ]
        )
        assert result == expected

    def test_get_filter_no_filter_returns_none(self) -> None:
        """Query without filter in metadata returns None."""
        query = EvalQuery(
            qid="test_004",
            query="Test query",
            relevant_chunk_ids={"chunk1"},
        )
        assert query.get_filter() is None

    def test_get_filter_empty_metadata_returns_none(self) -> None:
        """Query with empty metadata returns None."""
        query = EvalQuery(
            qid="test_005",
            query="Test query",
            relevant_chunk_ids={"chunk1"},
            metadata={},
        )
        assert query.get_filter() is None

    def test_get_filter_other_metadata_preserved(self) -> None:
        """Other metadata keys are not affected by get_filter()."""
        query = EvalQuery(
            qid="test_006",
            query="Test query",
            relevant_chunk_ids={"chunk1"},
            metadata={
                "filter": {"type": "Eq", "field": "doc_id", "value": "doc123"},
                "custom_key": "custom_value",
                "source_chunk_ids": ["chunk1", "chunk2"],
            },
        )
        result = query.get_filter()
        assert result == Eq(field="doc_id", value="doc123")
        assert query.metadata["custom_key"] == "custom_value"
        assert query.metadata["source_chunk_ids"] == ["chunk1", "chunk2"]

    def test_get_filter_invalid_filter_returns_none_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid filter data returns None and logs a warning."""
        query = EvalQuery(
            qid="test_007",
            query="Test query",
            relevant_chunk_ids={"chunk1"},
            metadata={"filter": {"invalid": "data"}},  # Missing 'type' field
        )
        with caplog.at_level(logging.WARNING):
            result = query.get_filter()

        assert result is None
        assert "Failed to deserialize filter for query test_007" in caplog.text

    def test_get_filter_unknown_type_returns_none_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown filter type returns None and logs a warning."""
        query = EvalQuery(
            qid="test_008",
            query="Test query",
            relevant_chunk_ids={"chunk1"},
            metadata={"filter": {"type": "UnknownFilter", "field": "x"}},
        )
        with caplog.at_level(logging.WARNING):
            result = query.get_filter()

        assert result is None
        assert "Unknown filter type" in caplog.text

    def test_get_filter_missing_required_field_returns_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Filter missing required field returns None and logs."""
        query = EvalQuery(
            qid="test_009",
            query="Test query",
            relevant_chunk_ids={"chunk1"},
            metadata={"filter": {"type": "Eq", "field": "doc_id"}},  # Missing 'value'
        )
        with caplog.at_level(logging.WARNING):
            result = query.get_filter()

        assert result is None
        assert "Failed to deserialize filter" in caplog.text


class TestEvalQueryRoundtrip:
    """Tests for EvalQuery serialization roundtrip with filters."""

    def test_roundtrip_with_filter(self) -> None:
        """EvalQuery with filter survives to_dict/from_dict roundtrip."""
        original = EvalQuery(
            qid="roundtrip_001",
            query="What is machine learning?",
            relevant_chunk_ids={"chunk1", "chunk2"},
            expected_answer="Machine learning is...",
            query_type=QueryType.DEFINITION,
            difficulty=Difficulty.EASY,
            metadata={
                "filter": {"type": "Eq", "field": "doc_id", "value": "ml_doc"},
                "custom": "value",
            },
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = EvalQuery.from_dict(data)

        # Verify filter works on restored query
        filter_result = restored.get_filter()
        assert filter_result == Eq(field="doc_id", value="ml_doc")

        # Verify other fields preserved
        assert restored.qid == original.qid
        assert restored.query == original.query
        assert restored.relevant_chunk_ids == original.relevant_chunk_ids
        assert restored.query_type == original.query_type
        assert restored.metadata["custom"] == "value"

    def test_roundtrip_without_filter(self) -> None:
        """EvalQuery without filter survives roundtrip."""
        original = EvalQuery(
            qid="roundtrip_002",
            query="Test query",
            relevant_chunk_ids={"chunk1"},
        )

        data = original.to_dict()
        restored = EvalQuery.from_dict(data)

        assert restored.get_filter() is None
        assert restored.qid == original.qid

    def test_roundtrip_complex_filter(self) -> None:
        """EvalQuery with complex nested filter survives roundtrip."""
        complex_filter = {
            "type": "And",
            "clauses": [
                {"type": "Eq", "field": "language", "value": "python"},
                {
                    "type": "Or",
                    "clauses": [
                        {"type": "In", "field": "tags", "values": ["ml", "ai"]},
                        {"type": "Prefix", "field": "uri", "prefix": "/docs/"},
                    ],
                },
            ],
        }

        original = EvalQuery(
            qid="roundtrip_003",
            query="Complex query",
            relevant_chunk_ids={"chunk1"},
            metadata={"filter": complex_filter},
        )

        data = original.to_dict()
        restored = EvalQuery.from_dict(data)

        filter_result = restored.get_filter()
        assert isinstance(filter_result, And)
        assert len(filter_result.clauses) == 2
