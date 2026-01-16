"""
Serialization and deserialization for Filter objects.

Provides bidirectional conversion between typed Filter dataclasses
and JSON-safe dictionaries for storage in EvalQuery metadata.
"""

from __future__ import annotations

from typing import Any

from rag.domain.filters import (
    And,
    Contains,
    Eq,
    In,
    Not,
    Or,
    Prefix,
    Range,
    Where,
)


def serialize_filter(where: Where) -> dict[str, Any] | None:
    """
    Convert a Filter to a JSON-safe dictionary.

    Args:
        where: Filter object or None

    Returns:
        Dictionary representation or None if where is None

    Raises:
        TypeError: If filter type is unknown

    Example:
        >>> eq = Eq(field="doc_id", value="abc123")
        >>> serialize_filter(eq)
        {'type': 'Eq', 'field': 'doc_id', 'value': 'abc123'}
    """
    if where is None:
        return None

    if isinstance(where, Eq):
        return {"type": "Eq", "field": where.field, "value": where.value}

    if isinstance(where, In):
        return {"type": "In", "field": where.field, "values": list(where.values)}

    if isinstance(where, Contains):
        return {"type": "Contains", "field": where.field, "value": where.value}

    if isinstance(where, Prefix):
        return {"type": "Prefix", "field": where.field, "prefix": where.prefix}

    if isinstance(where, Range):
        return {
            "type": "Range",
            "field": where.field,
            "gte": where.gte,
            "lte": where.lte,
            "gt": where.gt,
            "lt": where.lt,
        }

    if isinstance(where, And):
        if not where.clauses:
            raise ValueError("And filter must have at least one clause")
        return {"type": "And", "clauses": [serialize_filter(c) for c in where.clauses]}

    if isinstance(where, Or):
        if not where.clauses:
            raise ValueError("Or filter must have at least one clause")
        return {"type": "Or", "clauses": [serialize_filter(c) for c in where.clauses]}

    if isinstance(where, Not):
        return {"type": "Not", "clause": serialize_filter(where.clause)}

    raise TypeError(f"Unknown filter type: {type(where)}")


def deserialize_filter(data: dict[str, Any] | None) -> Where:
    """
    Convert a dictionary to a Filter object.

    Args:
        data: Dictionary with 'type' key and filter-specific fields, or None

    Returns:
        Filter object or None if data is None

    Raises:
        ValueError: If filter structure is invalid or type is unknown
        KeyError: If required fields are missing

    Example:
        >>> data = {'type': 'Eq', 'field': 'doc_id', 'value': 'abc123'}
        >>> deserialize_filter(data)
        Eq(field='doc_id', value='abc123')

        >>> data = {'type': 'And', 'clauses': [
        ...     {'type': 'Eq', 'field': 'x', 'value': 1},
        ...     {'type': 'In', 'field': 'y', 'values': [2, 3]}
        ... ]}
        >>> deserialize_filter(data)
        And(clauses=[Eq(field='x', value=1), In(field='y', values=[2, 3])])
    """
    if data is None:
        return None

    filter_type = data.get("type")
    if not filter_type:
        raise ValueError("Filter dict must have 'type' field")

    if filter_type == "Eq":
        return Eq(field=data["field"], value=data["value"])

    if filter_type == "In":
        return In(field=data["field"], values=data["values"])

    if filter_type == "Contains":
        return Contains(field=data["field"], value=data["value"])

    if filter_type == "Prefix":
        return Prefix(field=data["field"], prefix=data["prefix"])

    if filter_type == "Range":
        return Range(
            field=data["field"],
            gte=data.get("gte"),
            lte=data.get("lte"),
            gt=data.get("gt"),
            lt=data.get("lt"),
        )

    if filter_type == "And":
        raw_clauses = data.get("clauses")
        if not raw_clauses:
            raise ValueError("And filter must have at least one clause")

        parsed = [deserialize_filter(c) for c in raw_clauses]  # list[Filter | None]
        clauses = [c for c in parsed if c is not None]         # list[Filter]

        if not clauses:
            raise ValueError("And filter clauses cannot all be None")

        return And(clauses=clauses)

    if filter_type == "Or":
        raw_clauses = data.get("clauses")
        if not raw_clauses:
            raise ValueError("Or filter must have at least one clause")

        parsed = [deserialize_filter(c) for c in raw_clauses]
        clauses = [c for c in parsed if c is not None]

        if not clauses:
            raise ValueError("Or filter clauses cannot all be None")

        return Or(clauses=clauses)

    if filter_type == "Not":
        clause_data = data.get("clause")
        if clause_data is None:
            raise ValueError("Not filter must have a clause")

        clause = deserialize_filter(clause_data)
        if clause is None:
            raise ValueError("Not filter clause cannot be None")

        return Not(clause=clause)

    raise ValueError(f"Unknown filter type: {filter_type}")
