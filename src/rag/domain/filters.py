from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Filter: ...


@dataclass(frozen=True, slots=True)
class Eq(Filter):
    field: str
    value: Any


@dataclass(frozen=True, slots=True)
class In(Filter):
    field: str
    values: Sequence[Any]


@dataclass(frozen=True, slots=True)
class Contains(Filter):
    field: str
    value: Any


@dataclass(frozen=True, slots=True)
class Prefix(Filter):
    field: str
    prefix: str


@dataclass(frozen=True, slots=True)
class Range(Filter):
    field: str
    gte: Any | None = None
    lte: Any | None = None
    gt: Any | None = None
    lt: Any | None = None


@dataclass(frozen=True, slots=True)
class And(Filter):
    clauses: Sequence[Filter]


@dataclass(frozen=True, slots=True)
class Or(Filter):
    clauses: Sequence[Filter]


@dataclass(frozen=True, slots=True)
class Not(Filter):
    clause: Filter


Where = Filter | None
