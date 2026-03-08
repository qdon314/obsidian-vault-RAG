# eval/app_v2/engine/services/filter.py
"""
Facet-based filtering of AnalyzedQuery lists.
Filters are applied conjunctively (AND).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eval.app_v2.engine.domain.models import AnalyzedQuery
from eval.app_v2.engine.facets.registry import get_facet


def _matches(aq: AnalyzedQuery, key: str, value: Any) -> bool:
    if value is None:
        return True
    if key == "severity":
        return str(aq.diagnostic.severity) == str(value)
    facet = get_facet(key)
    if facet is None:
        return True
    actual = facet.extract(aq.record)
    if isinstance(value, bool):
        return actual == value
    return str(actual) == str(value)


def apply_facet_filters(
    queries: Sequence[AnalyzedQuery],
    selections: dict[str, Any],
) -> tuple[AnalyzedQuery, ...]:
    """Apply facet selections (AND-conjunctive). None values = no filter."""
    result = []
    for aq in queries:
        if all(_matches(aq, k, v) for k, v in selections.items()):
            result.append(aq)
    return tuple(result)
