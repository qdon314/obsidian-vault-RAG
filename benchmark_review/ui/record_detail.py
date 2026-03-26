from __future__ import annotations

from pathlib import Path

import streamlit as st

from benchmark_review.engine.models import ReviewRecord, ReviewStatus
from benchmark_review.engine.writer import save_decision


def render(rec: ReviewRecord, run_dir: Path, records: list[ReviewRecord]) -> None:
    """Render the right panel detail view for a single record."""
    st.subheader(rec.candidate_id)

    # Status badge
    _status_badge(rec.review_status)
    st.caption(f"`{rec.query_class}` · `{rec.difficulty}`")

    st.divider()

    # Query
    st.markdown("**Query**")
    st.info(rec.query)

    # Source citations
    st.markdown("**Source citations**")
    for c in rec.source_citations:
        st.code(c, language=None)

    # Validation flags
    if rec.validation_flags:
        st.warning("Validation flags: " + ", ".join(f"`{f}`" for f in rec.validation_flags))

    # Evidence
    _render_evidence(rec)

    # Semantic duplicate hint
    _render_duplicate_hint(rec, records)

    st.divider()

    # Review actions
    _render_actions(rec, run_dir, records)


def _status_badge(status: ReviewStatus) -> None:
    colour_map = {
        ReviewStatus.PENDING: "gray",
        ReviewStatus.APPROVED: "green",
        ReviewStatus.REJECTED: "red",
        ReviewStatus.NEEDS_REVISION: "orange",
    }
    colour = colour_map[status]
    st.markdown(f":{colour}[**{status.value.upper()}**]")


def _render_evidence(rec: ReviewRecord) -> None:
    if rec.is_unanswerable:
        st.markdown("**Unanswerable reason**")
        st.markdown(rec.unanswerable_reason or "_none provided_")
        if rec.critical_evidence:
            st.error("Pipeline bug: unanswerable record has non-empty critical evidence.")
        return

    for tier_label, spans in [
        ("Critical evidence", rec.critical_evidence),
        ("Supporting evidence", rec.supporting_evidence),
        ("Contextual evidence", rec.contextual_evidence),
    ]:
        if not spans:
            continue
        with st.expander(f"{tier_label} ({len(spans)} span{'s' if len(spans) != 1 else ''})", expanded=(tier_label == "Critical evidence")):
            for span in spans:
                st.markdown(f"**{span.citation}** · `{span.span_id}`")
                st.markdown(f"> {span.text}")


def _render_duplicate_hint(rec: ReviewRecord, all_records: list[ReviewRecord]) -> None:
    same_unit = [r for r in all_records if r.unit_id == rec.unit_id and r.candidate_id != rec.candidate_id]
    if not same_unit:
        return
    with st.expander(f"Similar queries — same unit ({len(same_unit)})", expanded=False):
        for other in same_unit:
            _status_badge(other.review_status)
            st.caption(other.candidate_id)
            st.markdown(other.query)
            if st.button("View", key=f"view_{other.candidate_id}_from_{rec.candidate_id}"):
                st.session_state["selected_id"] = other.candidate_id
                st.rerun()


def _render_actions(rec: ReviewRecord, run_dir: Path, all_records: list[ReviewRecord]) -> None:
    reviewer_id = st.session_state.get("reviewer_id", "")
    if not reviewer_id:
        st.warning("Enter your reviewer ID at the top of the page before making decisions.")
        return

    col1, col2, col3 = st.columns(3)
    approve = col1.button("✅ Approve", key=f"approve_{rec.candidate_id}", use_container_width=True)
    reject = col2.button("❌ Reject", key=f"reject_{rec.candidate_id}", use_container_width=True)
    revise = col3.button("🔶 Needs revision", key=f"revise_{rec.candidate_id}", use_container_width=True)

    if revise:
        st.session_state[f"pending_action_{rec.candidate_id}"] = "needs_revision"
    if reject:
        st.session_state[f"pending_action_{rec.candidate_id}"] = "rejected"

    pending_action = st.session_state.get(f"pending_action_{rec.candidate_id}")

    if pending_action in ("needs_revision", "rejected"):
        note_text = st.text_area(
            "Note (required)",
            key=f"note_{rec.candidate_id}",
            placeholder="Describe what needs fixing or why rejected",
        )
        if st.button("Save", key=f"save_note_{rec.candidate_id}", disabled=not note_text):
            status = ReviewStatus.NEEDS_REVISION if pending_action == "needs_revision" else ReviewStatus.REJECTED
            save_decision(
                run_dir=run_dir,
                candidate_id=rec.candidate_id,
                status=status,
                reviewed_by=reviewer_id,
                revision_notes=note_text if status == ReviewStatus.NEEDS_REVISION else None,
                rejection_note=note_text if status == ReviewStatus.REJECTED else None,
            )
            del st.session_state[f"pending_action_{rec.candidate_id}"]
            _advance_to_next_pending(rec.candidate_id, all_records)
            st.rerun()
        return

    if approve:
        save_decision(
            run_dir=run_dir,
            candidate_id=rec.candidate_id,
            status=ReviewStatus.APPROVED,
            reviewed_by=reviewer_id,
            revision_notes=None,
            rejection_note=None,
        )
        _advance_to_next_pending(rec.candidate_id, all_records)
        st.rerun()


def _advance_to_next_pending(current_id: str, records: list[ReviewRecord]) -> None:
    ids = [r.candidate_id for r in records]
    current_idx = ids.index(current_id) if current_id in ids else -1
    pending = [r for r in records[current_idx + 1:] if r.review_status == ReviewStatus.PENDING]
    if pending:
        st.session_state["selected_id"] = pending[0].candidate_id
