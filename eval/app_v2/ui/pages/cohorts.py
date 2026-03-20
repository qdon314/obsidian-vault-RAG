# eval/app_v2/ui/pages/cohorts.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.derived.slices import build_slice_table
from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.engine.services.forensics import list_queries_by_slice
from eval.app_v2.ui.widgets.diagnostic_card import render_diagnostic_card

_AVAILABLE_DIMS = ["query_type", "difficulty", "requires_synthesis", "is_unanswerable"]
_DEFAULT_DIMS = ["query_type", "difficulty"]
_WORST_N = 3
_QUERIES_PER_COHORT = 3


def render(bundle: RunBundle | None) -> None:
    st.header("Cohort Analysis")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    st.markdown(
        "Group queries by shared attributes and compare their aggregate retrieval performance. "
        "Use this to identify which query segments consistently underperform."
    )

    selected_dims = st.multiselect(
        "Group by dimensions",
        _AVAILABLE_DIMS,
        default=_DEFAULT_DIMS,
    )

    if not selected_dims:
        st.warning("Select at least one grouping dimension.")
        return

    table = build_slice_table(bundle.queries, group_by=selected_dims)

    if not table.rows:
        st.info("No data to group — run has no queries.")
        return

    # ── Full cohort table ──────────────────────────────────────────────────────
    st.subheader(f"All cohorts: {' x '.join(selected_dims)}")
    import pandas as pd

    rows = []
    for row in table.rows:
        label = " | ".join(f"{k}={v}" for k, v in row.key.parts)
        recall = row.metrics.get("recall@10")
        ndcg = row.metrics.get("ndcg@10")
        rows.append(
            {
                "Cohort": label,
                "Size": row.size,
                "recall@10": f"{recall:.3f}" if recall is not None else "—",
                "ndcg@10": f"{ndcg:.3f}" if ndcg is not None else "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Worst cohorts drill-down ───────────────────────────────────────────────
    st.subheader(f"Worst {_WORST_N} cohorts (by recall@10)")

    def _recall_sort_key(row: object) -> float:
        recall = getattr(row, "metrics", {}).get("recall@10")
        return recall if recall is not None else 1.0

    worst_rows = sorted(table.rows, key=_recall_sort_key)[:_WORST_N]

    for row in worst_rows:
        label = " | ".join(f"{k}={v}" for k, v in row.key.parts)
        recall = row.metrics.get("recall@10")
        header = f"**{label}** — n={row.size}"
        if recall is not None:
            header += f", recall@10={recall:.3f}"
        st.markdown(header)

        cohort_queries = list_queries_by_slice(bundle, row.key)
        if not cohort_queries:
            st.caption("No matching queries found.")
            continue
        for aq in list(cohort_queries)[:_QUERIES_PER_COHORT]:
            render_diagnostic_card(aq, show_forensics_link=True)
        st.divider()
