# eval/app_v2/engine/derived/slices.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from eval.app_v2.engine.domain.models import (
    AnalyzedQuery,
    SliceKey,
    SliceMetricRow,
    SliceMetricTable,
)


def _get_field(aq: AnalyzedQuery, field: str) -> str:
    """Extract a grouping field value from record or diagnostic."""
    val = getattr(aq.record, field, None)
    if val is None:
        return "__none__"
    return str(val)


def build_slice_table(
    queries: Sequence[AnalyzedQuery],
    group_by: Sequence[str],
) -> SliceMetricTable:
    groups: dict[tuple, list[AnalyzedQuery]] = defaultdict(list)
    for aq in queries:
        key_parts = tuple(_get_field(aq, f) for f in group_by)
        groups[key_parts].append(aq)

    rows: list[SliceMetricRow] = []
    for key_vals, members in groups.items():
        slice_key = SliceKey(parts=tuple(zip(group_by, key_vals, strict=False)))
        recall_vals = [aq.record.per_query_recall_at_k.get(10) for aq in members]
        ndcg_vals = [aq.record.per_query_ndcg_at_k.get(10) for aq in members]
        metrics: dict[str, float | None] = {
            "recall@10": sum(v for v in recall_vals if v is not None) / len(recall_vals) if recall_vals else None,
            "ndcg@10":   sum(v for v in ndcg_vals if v is not None) / len(ndcg_vals) if ndcg_vals else None,
            "size":      float(len(members)),
        }
        rows.append(SliceMetricRow(key=slice_key, size=len(members), metrics=metrics))

    return SliceMetricTable(group_by=tuple(group_by), rows=tuple(rows))
