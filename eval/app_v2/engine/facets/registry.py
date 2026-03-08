# eval/app_v2/engine/facets/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from eval.app_v2.engine.domain.models import QueryRecord


@dataclass(frozen=True)
class FacetDef:
    key: str
    label: str
    value_type: Literal["enum", "bool", "numeric_bucket"]
    extract: Callable[[QueryRecord], Any]
    higher_is_better: bool = True


FACETS: list[FacetDef] = [
    FacetDef(
        key="query_type",
        label="Query Type",
        value_type="enum",
        extract=lambda r: r.query_type,
    ),
    FacetDef(
        key="difficulty",
        label="Difficulty",
        value_type="enum",
        extract=lambda r: r.difficulty,
    ),
    FacetDef(
        key="requires_synthesis",
        label="Requires Synthesis",
        value_type="bool",
        extract=lambda r: r.requires_synthesis,
    ),
    FacetDef(
        key="is_unanswerable",
        label="Unanswerable",
        value_type="bool",
        extract=lambda r: r.is_unanswerable,
        higher_is_better=False,
    ),
    # Severity is on the diagnostic, not the record — use a wrapper
    FacetDef(
        key="severity",
        label="Severity",
        value_type="enum",
        extract=lambda r: None,  # overridden by filter.py which receives AnalyzedQuery
    ),
]


def get_facet(key: str) -> FacetDef | None:
    return next((f for f in FACETS if f.key == key), None)
