"""Shared encoder/decoder utilities for dataclasses_json."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

from rag.utils.enum_parse import parse_enum

T = TypeVar("T", bound=Enum)


# --- Tuple/Set encoders (for JSON array serialization) ---


def tuple_encoder(val: tuple[str, ...]) -> list[str]:
    """Encode a tuple as a JSON list."""
    return list(val)


def tuple_decoder(val: list[str] | None) -> tuple[str, ...]:
    """Decode a JSON list to a tuple."""
    return tuple(val) if val else ()


def set_encoder(val: set[str]) -> list[str]:
    """Encode a set as a sorted JSON list."""
    return sorted(val)


def set_decoder(val: list[str] | None) -> set[str]:
    """Decode a JSON list to a set."""
    return set(val) if val else set()


# --- Datetime encoders ---


def datetime_encoder(dt: datetime) -> str:
    """Encode a datetime as an ISO string."""
    return dt.isoformat()


def datetime_decoder(val: Any) -> datetime:
    """Decode an ISO string to datetime, defaulting to now(UTC) on failure."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            pass
    return datetime.now(UTC)


# --- Enum encoders (factory functions) ---


def enum_encoder(val: Enum | None) -> str | None:
    """Encode an enum as its value string."""
    return val.value if val else None


def make_enum_decoder(enum_cls: type[T], default: T | None = None):
    """
    Create a decoder function for an enum type.

    Args:
        enum_cls: The enum class to decode to
        default: Default value if decoding fails (None if not specified)

    Returns:
        A decoder function suitable for dataclasses_json config
    """

    def decoder(val: str | None) -> T | None:
        result = parse_enum(enum_cls, val)
        return result if result is not None else default

    return decoder
