from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import Severity
from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.engine.services.filter import apply_facet_filters
from eval.app_v2.ui.widgets.chunk_stats_panel import render_chunk_stats_panel
from eval.app_v2.ui.widgets.diagnostic_card import render_diagnostic_card
from eval.app_v2.ui.widgets.facet_panel import render_facet_panel
from eval.app_v2.ui.widgets.metric_cards import (
    render_dominant_failure_banner,
    render_kpi_cards,
    render_severity_bar,
)

_TOP_N = 10


def _render_retrieval_metrics_table(bundle: RunBundle) -> None:
    """Render a compact retrieval metrics table matching the V1 'Metrics' tab."""
    import pandas as pd

    overall = bundle.aggregates.overall
    k_values = sorted(overall.recall_at_k.keys())
    if not k_values:
        st.info("No retrieval metrics available.")
        return

    metrics = ["Recall", "Precision", "Hit Rate", "NDCG"]
    include_tiered = bool(
        overall.critical_recall_at_k
        or overall.weighted_recall_at_k
        or overall.critical_hit_rate_at_k
    )
    if include_tiered:
        metrics.extend(["Critical Recall", "Weighted Recall", "Critical Hit Rate"])

    data: dict[str, list[str]] = {"Metric": metrics}
    for k in k_values:
        col_values = [
            f"{overall.recall_at_k.get(k, 0):.3f}",
            f"{overall.precision_at_k.get(k, 0):.3f}",
            f"{overall.hit_rate_at_k.get(k, 0):.3f}",
            f"{overall.ndcg_at_k.get(k, 0):.3f}",
        ]
        if include_tiered:
            col_values.extend(
                [
                    f"{overall.critical_recall_at_k.get(k, 0):.3f}",
                    f"{overall.weighted_recall_at_k.get(k, 0):.3f}",
                    f"{overall.critical_hit_rate_at_k.get(k, 0):.3f}",
                ]
            )
        data[f"@{k}"] = col_values

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("MRR", f"{overall.mrr:.3f}")
    c2.metric("MAP", f"{overall.map:.3f}")
    c3.metric("Avg Retrieved", f"{overall.avg_retrieved:.1f}")


def render(bundle: RunBundle | None) -> None:
    st.header("Triage")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    h = bundle.health

    # KPI cards (now up to 4 rows depending on data available)
    render_kpi_cards(h)

    # Full retrieval metrics table (collapsible)
    with st.expander("Full retrieval metrics table", expanded=False):
        _render_retrieval_metrics_table(bundle)

    st.divider()

    # Severity bar
    st.subheader("Severity breakdown")
    render_severity_bar(h)
    st.divider()

    # Dominant failure mode
    st.subheader("Dominant failure mode")
    render_dominant_failure_banner(h)
    st.divider()

    # Verdict badge
    if bundle.verdict is not None:
        v = bundle.verdict
        if v.decision == "SHIP":
            st.success("**Verdict: SHIP** ✅")
        else:
            st.error(f"**Verdict: BLOCK** 🚫 — Failed: {', '.join(v.failed_check_names)}")
    st.divider()

    # Worst slice
    if h.worst_slice:
        st.subheader("Worst slice")
        parts = dict(h.worst_slice.parts)
        st.markdown(" | ".join(f"**{k}**: `{v}`" for k, v in parts.items()))
    st.divider()

    # Facet filters
    with st.sidebar:
        selections = render_facet_panel(list(bundle.queries))
    filtered_queries = apply_facet_filters(bundle.queries, selections)

    # Top-N critical/moderate queries
    critical = [
        aq
        for aq in filtered_queries
        if aq.diagnostic.severity in (Severity.CRITICAL, Severity.MODERATE)
    ]
    critical_sorted = sorted(
        critical, key=lambda aq: 0 if aq.diagnostic.severity == Severity.CRITICAL else 1
    )

    st.subheader(f"Top {_TOP_N} queries needing attention")
    if not critical_sorted:
        st.success("No critical or moderate queries.")
    for aq in critical_sorted[:_TOP_N]:
        render_diagnostic_card(aq, show_forensics_link=True)

    st.divider()
    render_chunk_stats_panel(bundle)
