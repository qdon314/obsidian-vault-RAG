from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import Severity
from eval.app_v2.engine.domain.models import AnalyzedQuery

_SEV_BADGE = {
    Severity.OK: "🟢",
    Severity.MINOR: "🟡",
    Severity.MODERATE: "🟠",
    Severity.CRITICAL: "🔴",
}


def render_diagnostic_card(aq: AnalyzedQuery, *, show_forensics_link: bool = False) -> None:
    """Render a compact card for a single AnalyzedQuery."""
    d = aq.diagnostic
    r = aq.record
    badge = _SEV_BADGE.get(d.severity, "⚪")

    with st.container(border=True):
        cols = st.columns([0.05, 0.7, 0.25])
        cols[0].markdown(badge)
        cols[1].markdown(f"**`{r.qid}`** — {r.query[:80]}{'…' if len(r.query) > 80 else ''}")
        cols[2].markdown(f"`{d.diagnostic_code}`")

        with st.expander("Details", expanded=False):
            st.markdown(f"**Root cause:** {d.root_cause_summary}")
            if d.suggested_next_check:
                st.markdown(f"**Next check:** {d.suggested_next_check}")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.caption(f"Retrieval: `{d.retrieval_status}`")
            sc2.caption(f"Rerank: `{d.rerank_status}`")
            sc3.caption(f"Packing: `{d.packing_status}`")
            sc4.caption(f"Generation: `{d.generation_status}`")

        if show_forensics_link and st.button("Inspect in Forensics →", key=f"forensics_{r.qid}"):
            st.session_state["forensics_qid"] = r.qid


def render_diagnostic_detail(aq: AnalyzedQuery) -> None:
    """Full diagnostic detail panel for the Forensics page."""
    d = aq.diagnostic
    r = aq.record
    badge = _SEV_BADGE.get(d.severity, "⚪")

    st.markdown(f"## {badge} `{d.diagnostic_code}` — {d.severity.upper()}")
    st.markdown(f"**Root cause:** {d.root_cause_summary}")
    if d.suggested_next_check:
        st.info(f"Suggested next check: {d.suggested_next_check}")

    with st.expander("Stage status breakdown", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval", d.retrieval_status)
        c2.metric("Rerank", d.rerank_status)
        c3.metric("Packing", d.packing_status)
        c4.metric("Generation", d.generation_status)

    with st.expander("Retrieval sets", expanded=False):
        retrieved_set = frozenset(r.retrieved_chunk_ids)
        matched = r.relevant_chunk_ids & retrieved_set
        missed = r.relevant_chunk_ids - retrieved_set
        extra = retrieved_set - r.relevant_chunk_ids
        st.markdown(f"- **Relevant:** {sorted(r.relevant_chunk_ids)}")
        st.markdown(f"- **Retrieved:** {list(r.retrieved_chunk_ids[:10])}")
        st.markdown(f"- **Matched:** {sorted(matched)}")
        st.markdown(f"- **Missed:** {sorted(missed)}")
        st.markdown(f"- **Extra retrieved:** {sorted(extra)}")

    if r.trace:
        with st.expander("Trace — pipeline drill-down", expanded=False):
            import json

            st.json(json.dumps(r.trace.raw_data, indent=2, default=str))
