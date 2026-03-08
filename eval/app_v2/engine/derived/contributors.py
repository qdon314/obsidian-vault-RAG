# eval/app_v2/engine/derived/contributors.py
from __future__ import annotations

from collections.abc import Sequence

from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import AnalyzedQuery

_SEV_ORDER = {Severity.OK: 0, Severity.MINOR: 1, Severity.MODERATE: 2, Severity.CRITICAL: 3}


def contributor_queries_for_code(
    analyzed: Sequence[AnalyzedQuery],
    code: DiagnosticCode,
    *,
    limit: int = 20,
) -> tuple[AnalyzedQuery, ...]:
    """Return queries matching a DiagnosticCode, sorted by severity descending."""
    matching = [aq for aq in analyzed if aq.diagnostic.diagnostic_code == code]
    matching.sort(key=lambda aq: _SEV_ORDER[aq.diagnostic.severity], reverse=True)
    return tuple(matching[:limit])


def worst_queries(
    analyzed: Sequence[AnalyzedQuery],
    *,
    limit: int = 10,
) -> tuple[AnalyzedQuery, ...]:
    """Return queries sorted by severity descending."""
    sorted_qs = sorted(analyzed, key=lambda aq: _SEV_ORDER[aq.diagnostic.severity], reverse=True)
    return tuple(sorted_qs[:limit])
