from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T", bound=Enum)


def parse_enum(enum_cls: type[T], raw: Any) -> T | None:
    if raw is None:
        return None
    try:
        return enum_cls(raw)
    except ValueError:
        return None
