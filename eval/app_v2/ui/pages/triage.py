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


def render(bundle: RunBundle | None) -> None:
    st.header("Triage")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    h = bundle.health

    # KPI cards
    render_kpi_cards(h)
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
        aq for aq in filtered_queries
        if aq.diagnostic.severity in (Severity.CRITICAL, Severity.MODERATE)
    ]
    critical_sorted = sorted(critical, key=lambda aq: (
        0 if aq.diagnostic.severity == Severity.CRITICAL else 1
    ))

    st.subheader(f"Top {_TOP_N} queries needing attention")
    if not critical_sorted:
        st.success("No critical or moderate queries.")
    for aq in critical_sorted[:_TOP_N]:
        render_diagnostic_card(aq, show_forensics_link=True)

    st.divider()
    render_chunk_stats_panel(bundle)
