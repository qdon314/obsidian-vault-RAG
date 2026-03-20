from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.models import AnalyzedQuery, RunBundle
from eval.app_v2.ui.widgets.chunk_viewer import render_retrieved_chunks
from eval.app_v2.ui.widgets.diagnostic_card import render_diagnostic_detail


def _find_query(bundle: RunBundle, qid: str) -> AnalyzedQuery | None:
    for aq in bundle.queries:
        if aq.record.qid == qid:
            return aq
    return None


def _render_compression_context(trace_raw_data: dict, *, key_prefix: str = "") -> None:
    """Render before/after context text areas when compression metadata is present."""
    compression = (trace_raw_data.get("metadata") or {}).get("compression") or {}
    context_before: str | None = compression.get("context_before")
    context_after: str | None = compression.get("context_after")

    if not context_before and not context_after:
        return

    with st.expander("Context window (compression)", expanded=False):
        tab_before, tab_after = st.tabs(["Before", "After"])
        with tab_before:
            st.text_area(
                "",
                value=context_before or "— not captured —",
                height=300,
                disabled=True,
                key=f"{key_prefix}_ctx_before",
                label_visibility="collapsed",
            )
        with tab_after:
            st.text_area(
                "",
                value=context_after or "— not captured —",
                height=300,
                disabled=True,
                key=f"{key_prefix}_ctx_after",
                label_visibility="collapsed",
            )


def render(bundle: RunBundle | None) -> None:
    st.header("Forensics")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    # qid selection — prefer session_state push from Triage
    all_qids = [aq.record.qid for aq in bundle.queries]
    default_idx = 0
    pushed_qid = st.session_state.get("forensics_qid")
    if pushed_qid and pushed_qid in all_qids:
        default_idx = all_qids.index(pushed_qid)

    qid = st.selectbox("Query ID", all_qids, index=default_idx)
    aq = _find_query(bundle, qid)

    if aq is None:
        st.error(f"Query `{qid}` not found in bundle.")
        return

    # Query header
    r = aq.record
    with st.container(border=True):
        st.markdown(f"**Query:** {r.query}")
        cols = st.columns(4)
        cols[0].caption(f"Type: `{r.query_type or '—'}`")
        cols[1].caption(f"Difficulty: `{r.difficulty or '—'}`")
        cols[2].caption(f"Unanswerable: `{r.is_unanswerable}`")
        cols[3].caption(f"Trace: `{'✓' if r.trace else '✗'}`")
        if r.tags:
            st.caption(f"Tags: {', '.join(r.tags)}")

    st.divider()

    # Diagnostic detail
    render_diagnostic_detail(aq)

    # Chunk content
    render_retrieved_chunks(r, r.trace, key_prefix=qid)
    if r.trace is not None:
        _render_compression_context(r.trace.raw_data, key_prefix=qid)

    # Answer section
    if r.answer_text:
        st.divider()
        with st.expander("Generated answer", expanded=False):
            st.markdown(r.answer_text)
            if r.answer_metrics:
                st.caption(f"Answer metrics: {r.answer_metrics}")

    # Per-query metrics
    st.divider()
    with st.expander("Per-query retrieval metrics", expanded=False):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Recall@10", f"{r.per_query_recall_at_k.get(10, 0):.1%}")
        mc2.metric("Precision@10", f"{r.per_query_precision_at_k.get(10, 0):.1%}")
        mc3.metric("NDCG@10", f"{r.per_query_ndcg_at_k.get(10, 0):.1%}")
        mc4.metric("Latency", f"{r.latency_ms} ms" if r.latency_ms else "—")
