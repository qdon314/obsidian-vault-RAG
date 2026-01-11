"""
Query edit form component for modifying existing queries.
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from experiments.ui.state import get_state, set_editing_query, update_query
from rag.eval.schema import Difficulty, EvalQuery, QueryType


def render_query_edit_form() -> bool:
    """
    Render the edit form for an existing query.

    Returns True if the query was updated, False otherwise.
    """
    state = get_state()
    query = state.editing_query

    if not query:
        return False

    st.subheader(f"Editing: {query.qid}")

    # Cancel button at top
    if st.button("← Back to list", key="cancel_edit_top"):
        set_editing_query(None)
        st.rerun()
        return False

    with st.form("query_edit_form"):
        # Query text
        query_text = st.text_area(
            "Query text *",
            value=query.query,
            height=100,
            help="The question that will be asked to the RAG system.",
        )

        # Expected answer
        expected_answer = st.text_area(
            "Expected answer",
            value=query.expected_answer or "",
            height=150,
            help="The expected answer helps evaluate answer quality.",
        )

        # Query metadata
        col1, col2 = st.columns(2)

        with col1:
            query_type = st.selectbox(
                "Query type",
                options=list(QueryType),
                index=list(QueryType).index(query.query_type),
                format_func=lambda x: x.value,
                help="The type of query (factual, comparison, etc.)",
            )

            difficulty = st.selectbox(
                "Difficulty",
                options=list(Difficulty),
                index=list(Difficulty).index(query.difficulty),
                format_func=lambda x: x.value,
                help="How difficult is this query to answer?",
            )

        with col2:
            requires_synthesis = st.checkbox(
                "Requires synthesis",
                value=query.requires_synthesis,
                help="Does the answer require combining multiple pieces of information?",
            )

            is_unanswerable = st.checkbox(
                "Unanswerable",
                value=query.is_unanswerable,
                help="Check if this query cannot be answered from the corpus.",
            )

        # Tags
        tags_input = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(query.tags),
            help="Tags to categorize this query.",
        )

        # Notes
        notes = st.text_area(
            "Notes",
            value=query.notes or "",
            height=80,
            help="Free-form notes for evaluators.",
        )

        # Unanswerable reason (conditional)
        unanswerable_reason = query.unanswerable_reason
        if is_unanswerable:
            reason_options = ["not_in_corpus", "ambiguous", "other"]
            current_idx = (
                reason_options.index(unanswerable_reason)
                if unanswerable_reason in reason_options
                else 0
            )
            unanswerable_reason = st.selectbox(
                "Unanswerable reason",
                options=reason_options,
                index=current_idx,
                help="Why is this query unanswerable?",
            )

        st.divider()

        # Chunk IDs (editable as text)
        st.markdown(f"**Relevant chunk IDs ({len(query.relevant_chunk_ids)}):**")
        chunk_ids_text = st.text_area(
            "Chunk IDs (one per line)",
            value="\n".join(sorted(query.relevant_chunk_ids)),
            height=100,
            help="Edit chunk IDs directly. One ID per line.",
        )

        st.divider()

        # Submit buttons
        col1, col2 = st.columns(2)

        with col1:
            save_clicked = st.form_submit_button(
                "Save Changes",
                type="primary",
            )

        with col2:
            cancel_clicked = st.form_submit_button("Cancel")

    # Handle cancel
    if cancel_clicked:
        set_editing_query(None)
        st.rerun()
        return False

    # Handle save
    if save_clicked:
        # Validate
        if not query_text.strip():
            st.error("Query text is required.")
            return False

        # Parse tags
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]

        # Parse chunk IDs
        chunk_ids = {
            line.strip() for line in chunk_ids_text.split("\n") if line.strip()
        }

        # If unanswerable, clear chunk IDs
        if is_unanswerable:
            chunk_ids = set()

        # Create updated query (preserving qid and metadata)
        updated_metadata = dict(query.metadata)
        updated_metadata["last_modified"] = datetime.now(UTC).isoformat()

        updated_query = EvalQuery(
            qid=query.qid,
            query=query_text.strip(),
            relevant_chunk_ids=chunk_ids,
            expected_answer=expected_answer.strip() if expected_answer.strip() else None,
            expected_answer_alternatives=query.expected_answer_alternatives,
            query_type=query_type,
            difficulty=difficulty,
            requires_synthesis=requires_synthesis,
            notes=notes.strip() if notes.strip() else None,
            tags=tags,
            created_at=query.created_at,
            created_by=query.created_by,
            is_unanswerable=is_unanswerable,
            unanswerable_reason=unanswerable_reason if is_unanswerable else None,
            metadata=updated_metadata,
        )

        # Save
        update_query(updated_query)
        st.success(f"Updated query `{query.qid}`")
        return True

    return False
