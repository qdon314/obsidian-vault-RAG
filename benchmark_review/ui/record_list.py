from __future__ import annotations

import streamlit as st

from benchmark_review.engine.models import ReviewRecord, ReviewStatus

_STATUS_ICON = {
    ReviewStatus.PENDING: "⬜",
    ReviewStatus.APPROVED: "✅",
    ReviewStatus.REJECTED: "❌",
    ReviewStatus.NEEDS_REVISION: "🔶",
}

_FILTER_OPTIONS = ["All", "Pending", "Approved", "Rejected", "Needs revision"]
_FILTER_MAP = {
    "All": None,
    "Pending": ReviewStatus.PENDING,
    "Approved": ReviewStatus.APPROVED,
    "Rejected": ReviewStatus.REJECTED,
    "Needs revision": ReviewStatus.NEEDS_REVISION,
}


def render(records: list[ReviewRecord]) -> str | None:
    """Render the left panel. Returns the selected candidate_id or None."""
    st.subheader("Candidates")

    filter_choice = st.selectbox("Filter", _FILTER_OPTIONS, key="filter_status")
    search = st.text_input("Search", placeholder="query text or citation", key="search_query")

    status_filter = _FILTER_MAP[filter_choice]
    filtered = [
        r
        for r in records
        if (status_filter is None or r.review_status == status_filter)
        and (
            not search
            or search.lower() in r.query.lower()
            or any(search.lower() in c.lower() for c in r.source_citations)
        )
    ]

    if not filtered:
        st.caption("No records match filter.")
        return st.session_state.get("selected_id")

    selected_id: str | None = st.session_state.get("selected_id")
    # Auto-select first pending if nothing selected
    if selected_id is None or not any(r.candidate_id == selected_id for r in filtered):
        pending = [r for r in filtered if r.review_status == ReviewStatus.PENDING]
        selected_id = (pending or filtered)[0].candidate_id

    for rec in filtered:
        icon = _STATUS_ICON[rec.review_status]
        label = f"{icon} {rec.candidate_id}"
        is_selected = rec.candidate_id == selected_id
        if st.button(
            label,
            key=f"btn_{rec.candidate_id}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            selected_id = rec.candidate_id

    return selected_id
