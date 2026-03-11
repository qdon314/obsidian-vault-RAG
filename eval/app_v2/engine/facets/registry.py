# eval/app_v2/engine/facets/registry.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from eval.app_v2.engine.domain.models import QueryRecord


@dataclass(frozen=True)
class FacetDef:
    key: str
    label: str
    value_type: Literal["enum", "bool", "numeric_range"]
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
    # Diagnostic code — also overridden in filter.py
    FacetDef(
        key="diagnostic_code",
        label="Diagnostic Code",
        value_type="enum",
        extract=lambda r: None,  # overridden by filter.py which receives AnalyzedQuery
    ),
    # ── Numeric range facets ─────────────────────────────────────────────────
    FacetDef(
        key="recall_at_10",
        label="Recall@10",
        value_type="numeric_range",
        extract=lambda r: r.per_query_recall_at_k.get(10),
        higher_is_better=True,
    ),
    FacetDef(
        key="ndcg_at_10",
        label="NDCG@10",
        value_type="numeric_range",
        extract=lambda r: r.per_query_ndcg_at_k.get(10),
        higher_is_better=True,
    ),
    FacetDef(
        key="quality_score",
        label="Quality Score",
        value_type="numeric_range",
        extract=lambda r: r.answer_metrics.quality_score if r.answer_metrics is not None else None,
        higher_is_better=True,
    ),
    FacetDef(
        key="hallucination_severity",
        label="Hallucination Severity",
        value_type="numeric_range",
        extract=lambda r: (
            r.answer_metrics.hallucination_severity if r.answer_metrics is not None else None
        ),
        higher_is_better=False,
    ),
    FacetDef(
        key="latency_ms",
        label="Latency (ms)",
        value_type="numeric_range",
        extract=lambda r: float(r.latency_ms) if r.latency_ms is not None else None,
        higher_is_better=False,
    ),
]


def get_facet(key: str) -> FacetDef | None:
    return next((f for f in FACETS if f.key == key), None)
