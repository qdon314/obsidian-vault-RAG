from __future__ import annotations

from collections import Counter

import streamlit as st

from benchmark_review.engine.models import ReviewRecord, ReviewStatus

_STATUS_COLOURS = {
    ReviewStatus.PENDING: "gray",
    ReviewStatus.APPROVED: "green",
    ReviewStatus.REJECTED: "red",
    ReviewStatus.NEEDS_REVISION: "orange",
}

_STATUS_LABELS = {
    ReviewStatus.PENDING: "Pending",
    ReviewStatus.APPROVED: "Approved",
    ReviewStatus.REJECTED: "Rejected",
    ReviewStatus.NEEDS_REVISION: "Needs revision",
}


def render(records: list[ReviewRecord]) -> None:
    counts = Counter(r.review_status for r in records)
    total = len(records)
    reviewed = total - counts[ReviewStatus.PENDING]

    cols = st.columns([3, 1, 1, 1, 1])
    with cols[0]:
        st.progress(reviewed / total if total else 0, text=f"{reviewed} / {total} reviewed")
    for col, status in zip(cols[1:], [ReviewStatus.PENDING, ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.NEEDS_REVISION]):
        colour = _STATUS_COLOURS[status]
        label = _STATUS_LABELS[status]
        with col:
            st.markdown(f":{colour}[**{counts[status]}** {label}]")
