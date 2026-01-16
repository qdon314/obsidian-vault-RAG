"""Tests for filter serialization and deserialization."""

from __future__ import annotations

import pytest

from rag.domain.filter_serde import deserialize_filter, serialize_filter
from rag.domain.filters import And, Contains, Eq, In, Not, Or, Prefix, Range


class TestSerializeFilter:
    """Tests for serialize_filter function."""

    def test_serialize_none(self) -> None:
        """None input returns None output."""
        assert serialize_filter(None) is None

    def test_serialize_eq(self) -> None:
        """Eq filter serializes correctly."""
        f = Eq(field="doc_id", value="abc123")
        result = serialize_filter(f)
        assert result == {"type": "Eq", "field": "doc_id", "value": "abc123"}

    def test_serialize_in(self) -> None:
        """In filter serializes correctly with list conversion."""
        f = In(field="category", values=("a", "b", "c"))
        result = serialize_filter(f)
        assert result == {"type": "In", "field": "category", "values": ["a", "b", "c"]}

    def test_serialize_contains(self) -> None:
        """Contains filter serializes correctly."""
        f = Contains(field="tags", value="python")
        result = serialize_filter(f)
        assert result == {"type": "Contains", "field": "tags", "value": "python"}

    def test_serialize_prefix(self) -> None:
        """Prefix filter serializes correctly."""
        f = Prefix(field="uri", prefix="/docs/")
        result = serialize_filter(f)
        assert result == {"type": "Prefix", "field": "uri", "prefix": "/docs/"}

    def test_serialize_range_full(self) -> None:
        """Range filter with all bounds serializes correctly."""
        f = Range(field="score", gte=0.5, lte=1.0, gt=None, lt=None)
        result = serialize_filter(f)
        assert result == {
            "type": "Range",
            "field": "score",
            "gte": 0.5,
            "lte": 1.0,
            "gt": None,
            "lt": None,
        }

    def test_serialize_range_partial(self) -> None:
        """Range filter with partial bounds serializes correctly."""
        f = Range(field="index", gte=10)
        result = serialize_filter(f)
        
        assert result is not None
        assert result["type"] == "Range"
        assert result["field"] == "index"
        assert result["gte"] == 10
        assert result["lte"] is None

    def test_serialize_and(self) -> None:
        """And filter serializes with nested clauses."""
        f = And(
            clauses=[
                Eq(field="language", value="python"),
                In(field="tags", values=["ml", "ai"]),
            ]
        )
        result = serialize_filter(f)
        
        assert result is not None
        assert result["type"] == "And"
        assert len(result["clauses"]) == 2
        assert result["clauses"][0] == {"type": "Eq", "field": "language", "value": "python"}
        assert result["clauses"][1] == {"type": "In", "field": "tags", "values": ["ml", "ai"]}

    def test_serialize_or(self) -> None:
        """Or filter serializes with nested clauses."""
        f = Or(
            clauses=[
                Eq(field="language", value="python"),
                Eq(field="language", value="rust"),
            ]
        )
        result = serialize_filter(f)
        
        assert result is not None
        assert result["type"] == "Or"
        assert len(result["clauses"]) == 2

    def test_serialize_not(self) -> None:
        """Not filter serializes with nested clause."""
        f = Not(clause=Eq(field="draft", value=True))
        result = serialize_filter(f)
        assert result == {
            "type": "Not",
            "clause": {"type": "Eq", "field": "draft", "value": True},
        }

    def test_serialize_deeply_nested(self) -> None:
        """Deeply nested filter serializes correctly."""
        f = And(
            clauses=[
                Or(
                    clauses=[
                        Eq(field="a", value=1),
                        Not(clause=Eq(field="b", value=2)),
                    ]
                ),
                Range(field="c", gte=0, lt=100),
            ]
        )
        result = serialize_filter(f)
        
        assert result is not None
        assert result["type"] == "And"
        assert result["clauses"][0]["type"] == "Or"
        assert result["clauses"][0]["clauses"][1]["type"] == "Not"

    def test_serialize_empty_and_raises(self) -> None:
        """And with empty clauses raises ValueError."""
        f = And(clauses=[])
        with pytest.raises(ValueError, match="And filter must have at least one clause"):
            serialize_filter(f)

    def test_serialize_empty_or_raises(self) -> None:
        """Or with empty clauses raises ValueError."""
        f = Or(clauses=[])
        with pytest.raises(ValueError, match="Or filter must have at least one clause"):
            serialize_filter(f)

    def test_serialize_unknown_type_raises(self) -> None:
        """Unknown filter type raises TypeError."""

        class UnknownFilter:
            pass

        with pytest.raises(TypeError, match="Unknown filter type"):
            serialize_filter(UnknownFilter())  # type: ignore


class TestDeserializeFilter:
    """Tests for deserialize_filter function."""

    def test_deserialize_none(self) -> None:
        """None input returns None output."""
        assert deserialize_filter(None) is None

    def test_deserialize_eq(self) -> None:
        """Eq filter deserializes correctly."""
        data = {"type": "Eq", "field": "doc_id", "value": "abc123"}
        result = deserialize_filter(data)
        assert result == Eq(field="doc_id", value="abc123")

    def test_deserialize_in(self) -> None:
        """In filter deserializes correctly."""
        data = {"type": "In", "field": "category", "values": ["a", "b", "c"]}
        result = deserialize_filter(data)
        assert result == In(field="category", values=["a", "b", "c"])

    def test_deserialize_contains(self) -> None:
        """Contains filter deserializes correctly."""
        data = {"type": "Contains", "field": "tags", "value": "python"}
        result = deserialize_filter(data)
        assert result == Contains(field="tags", value="python")

    def test_deserialize_prefix(self) -> None:
        """Prefix filter deserializes correctly."""
        data = {"type": "Prefix", "field": "uri", "prefix": "/docs/"}
        result = deserialize_filter(data)
        assert result == Prefix(field="uri", prefix="/docs/")

    def test_deserialize_range(self) -> None:
        """Range filter deserializes correctly."""
        data = {"type": "Range", "field": "score", "gte": 0.5, "lt": 1.0}
        result = deserialize_filter(data)
        assert result == Range(field="score", gte=0.5, lt=1.0)

    def test_deserialize_and(self) -> None:
        """And filter deserializes with nested clauses."""
        data = {
            "type": "And",
            "clauses": [
                {"type": "Eq", "field": "language", "value": "python"},
                {"type": "In", "field": "tags", "values": ["ml", "ai"]},
            ],
        }
        result = deserialize_filter(data)
        expected = And(
            clauses=[
                Eq(field="language", value="python"),
                In(field="tags", values=["ml", "ai"]),
            ]
        )
        assert result == expected

    def test_deserialize_or(self) -> None:
        """Or filter deserializes with nested clauses."""
        data = {
            "type": "Or",
            "clauses": [
                {"type": "Eq", "field": "language", "value": "python"},
                {"type": "Eq", "field": "language", "value": "rust"},
            ],
        }
        result = deserialize_filter(data)
        assert isinstance(result, Or)
        assert len(result.clauses) == 2

    def test_deserialize_not(self) -> None:
        """Not filter deserializes with nested clause."""
        data = {
            "type": "Not",
            "clause": {"type": "Eq", "field": "draft", "value": True},
        }
        result = deserialize_filter(data)
        assert result == Not(clause=Eq(field="draft", value=True))

    def test_deserialize_missing_type_raises(self) -> None:
        """Missing 'type' field raises ValueError."""
        data = {"field": "doc_id", "value": "abc"}
        with pytest.raises(ValueError, match="must have 'type' field"):
            deserialize_filter(data)

    def test_deserialize_unknown_type_raises(self) -> None:
        """Unknown filter type raises ValueError."""
        data = {"type": "InvalidFilter", "field": "x"}
        with pytest.raises(ValueError, match="Unknown filter type"):
            deserialize_filter(data)

    def test_deserialize_and_empty_clauses_raises(self) -> None:
        """And with empty clauses raises ValueError."""
        data = {"type": "And", "clauses": []}
        with pytest.raises(ValueError, match="And filter must have at least one clause"):
            deserialize_filter(data)

    def test_deserialize_or_empty_clauses_raises(self) -> None:
        """Or with empty clauses raises ValueError."""
        data = {"type": "Or", "clauses": []}
        with pytest.raises(ValueError, match="Or filter must have at least one clause"):
            deserialize_filter(data)

    def test_deserialize_and_with_none_clauses_filters_them(self) -> None:
        """And with None in clauses filters them out."""
        data = {
            "type": "And",
            "clauses": [
                {"type": "Eq", "field": "x", "value": 1},
                None,
                {"type": "Eq", "field": "y", "value": 2},
            ],
        }
        result = deserialize_filter(data)
        assert isinstance(result, And)
        assert len(result.clauses) == 2

    def test_deserialize_and_all_none_clauses_raises(self) -> None:
        """And with all None clauses raises ValueError."""
        data = {"type": "And", "clauses": [None, None]}
        with pytest.raises(ValueError, match="clauses cannot all be None"):
            deserialize_filter(data)

    def test_deserialize_not_missing_clause_raises(self) -> None:
        """Not without clause raises ValueError."""
        data = {"type": "Not"}
        with pytest.raises(ValueError, match="Not filter must have a clause"):
            deserialize_filter(data)

    def test_deserialize_not_null_clause_raises(self) -> None:
        """Not with null clause raises ValueError."""
        data = {"type": "Not", "clause": None}
        with pytest.raises(ValueError, match="Not filter must have a clause"):
            deserialize_filter(data)


class TestRoundtrip:
    """Tests for serialize -> deserialize roundtrip."""

    def test_roundtrip_eq(self) -> None:
        """Eq survives roundtrip."""
        original = Eq(field="doc_id", value="abc123")
        serialized = serialize_filter(original)
        deserialized = deserialize_filter(serialized)
        assert deserialized == original

    def test_roundtrip_complex(self) -> None:
        """Complex nested filter survives roundtrip."""
        original = And(
            clauses=[
                Eq(field="language", value="python"),
                Or(
                    clauses=[
                        In(field="tags", values=["ml", "ai"]),
                        Range(field="chunk_index", gte=0, lt=10),
                    ]
                ),
                Not(clause=Eq(field="draft", value=True)),
            ]
        )
        serialized = serialize_filter(original)
        deserialized = deserialize_filter(serialized)
        assert deserialized == original

    def test_roundtrip_all_filter_types(self) -> None:
        """All filter types survive roundtrip."""
        filters = [
            Eq(field="f", value="v"),
            In(field="f", values=[1, 2, 3]),
            Contains(field="f", value="x"),
            Prefix(field="f", prefix="pre"),
            Range(field="f", gte=0, lte=100),
            And(clauses=[Eq(field="a", value=1), Eq(field="b", value=2)]),
            Or(clauses=[Eq(field="a", value=1), Eq(field="b", value=2)]),
            Not(clause=Eq(field="a", value=1)),
        ]
        for original in filters:
            serialized = serialize_filter(original)
            deserialized = deserialize_filter(serialized)
            assert deserialized == original, f"Roundtrip failed for {type(original).__name__}"
