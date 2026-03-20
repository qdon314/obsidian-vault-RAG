# eval/app_v2/ui/pages/compare.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import ComparisonClassification
from eval.app_v2.engine.domain.models import ComparisonBundle, RunBundle
from eval.app_v2.engine.services.comparison import build_comparison

_CLASS_COLORS = {
    ComparisonClassification.IMPROVED: "🟢",
    ComparisonClassification.REGRESSED: "🔴",
    ComparisonClassification.MIXED: "🟡",
    ComparisonClassification.UNCHANGED: "⚪",
    ComparisonClassification.INSUFFICIENT_DATA: "❓",
}


def _render_aggregate_deltas(cb: ComparisonBundle) -> None:
    st.subheader("Aggregate deltas (B - A)")
    cols = st.columns(len(cb.aggregate_deltas) or 1)
    for col, (metric, delta) in zip(cols, cb.aggregate_deltas.items(), strict=False):
        if delta is None:
            col.metric(metric, "—")
        else:
            col.metric(metric, f"{delta:+.1%}")


def _render_compared_queries(
    cb: ComparisonBundle, filter_class: ComparisonClassification | None
) -> None:
    queries = cb.compared_queries
    if filter_class:
        queries = tuple(q for q in queries if q.classification == filter_class)

    st.markdown(f"**{len(queries)} queries** matching filter")
    for cq in queries[:50]:  # cap display at 50
        badge = _CLASS_COLORS.get(cq.classification, "")
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([0.05, 0.5, 0.15, 0.15, 0.15])
            c1.markdown(badge)
            c2.markdown(f"`{cq.qid}` — {cq.query[:60]}")
            c3.caption(
                f"Recall Δ: {cq.retrieval_delta:+.2f}"
                if cq.retrieval_delta is not None
                else "Recall Δ: —"
            )
            c4.caption(
                f"NDCG Δ: {cq.ndcg_delta:+.2f}" if cq.ndcg_delta is not None else "NDCG Δ: —"
            )
            c5.caption(
                f"Lat Δ: {cq.latency_delta_ms:+.0f}ms"
                if cq.latency_delta_ms is not None
                else "Lat Δ: —"
            )


def render(bundle_a: RunBundle | None, bundle_b: RunBundle | None = None) -> None:
    st.header("Compare")

    if bundle_a is None or bundle_b is None:
        st.info("Select two runs (A and B) from the sidebar to compare.")
        return

    cb = build_comparison(bundle_a, bundle_b)

    st.markdown(f"**A:** `{bundle_a.display_name}` → **B:** `{bundle_b.display_name}`")
    _render_aggregate_deltas(cb)
    st.divider()

    filter_opts = ["All"] + [c.value for c in ComparisonClassification]
    choice = st.selectbox("Filter by classification", filter_opts)
    filter_class = None if choice == "All" else ComparisonClassification(choice)
    _render_compared_queries(cb, filter_class)
