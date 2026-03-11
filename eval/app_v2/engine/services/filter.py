# eval/app_v2/engine/services/filter.py
"""
Facet-based filtering of AnalyzedQuery lists.
Filters are applied conjunctively (AND).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from eval.app_v2.engine.domain.models import AnalyzedQuery
from eval.app_v2.engine.facets.registry import get_facet

logger = logging.getLogger(__name__)


def _matches(aq: AnalyzedQuery, key: str, value: Any) -> bool:
    if value is None:
        return True
    if key == "severity":
        return str(aq.diagnostic.severity) == str(value)
    if key == "diagnostic_code":
        return str(aq.diagnostic.diagnostic_code) == str(value)
    # Numeric range: value is a (lo, hi) tuple of numbers
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(x, (int, float)) for x in value)
    ):
        lo, hi = value
        facet = get_facet(key)
        if facet is None:
            return True
        actual = facet.extract(aq.record)
        if actual is None:
            # Queries where the metric is unavailable pass through (not filtered out)
            return True
        try:
            return float(lo) <= float(actual) <= float(hi)
        except (TypeError, ValueError):
            logger.warning(
                "Numeric range filter failed for facet %r: could not convert %r to float",
                key,
                actual,
            )
            return True
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
