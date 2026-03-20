# eval/app_v2/engine/services/forensics.py
"""
Navigation and selection over already-derived diagnostics.
Does NOT construct new diagnoses.
"""

from __future__ import annotations

from eval.app_v2.engine.derived.contributors import worst_queries
from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import AnalyzedQuery, RunBundle, SliceKey


def get_query(bundle: RunBundle, qid: str) -> AnalyzedQuery | None:
    for aq in bundle.queries:
        if aq.record.qid == qid:
            return aq
    return None


def list_queries_by_code(bundle: RunBundle, code: DiagnosticCode) -> tuple[AnalyzedQuery, ...]:
    return tuple(aq for aq in bundle.queries if aq.diagnostic.diagnostic_code == code)


def list_queries_by_slice(bundle: RunBundle, slice_key: SliceKey) -> tuple[AnalyzedQuery, ...]:
    """Return queries whose record fields match all parts of a SliceKey."""

    def matches(aq: AnalyzedQuery) -> bool:
        for field, value in slice_key.parts:
            if value == "__none__":
                if getattr(aq.record, field, None) is not None:
                    return False
            elif str(getattr(aq.record, field, None)) != value:
                return False
        return True

    return tuple(aq for aq in bundle.queries if matches(aq))


def get_worst_queries(bundle: RunBundle, *, limit: int = 10) -> tuple[AnalyzedQuery, ...]:
    return worst_queries(bundle.queries, limit=limit)


def contributor_queries_for_failure_mode(
    bundle: RunBundle, code: DiagnosticCode, *, limit: int = 20
) -> tuple[AnalyzedQuery, ...]:
    from eval.app_v2.engine.derived.contributors import contributor_queries_for_code

    return contributor_queries_for_code(bundle.queries, code, limit=limit)
