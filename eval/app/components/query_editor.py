"""
Query editor form component.
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from eval.app.components.filter_builder import render_filter_builder
from eval.app.state import get_eval_store, get_state, increment_counter, reset_selection
from rag.domain.models import Chunk
from rag.eval.schema import Difficulty, EvalQuery, QuerySuggestion, QueryType


def _generate_qid(output_path) -> str:
    """Generate a unique query ID using timestamp to avoid race conditions."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return f"curated_{timestamp}"


def render_query_editor(
    chunks: list[Chunk],
    suggestion: QuerySuggestion | None = None,
) -> bool:
    """
    Render the query editor form.

    Args:
        chunks: List of chunks to use as ground truth (one or more)
        suggestion: Optional query suggestion to pre-fill form

    Returns True if a query was saved, False otherwise.
    """
    state = get_state()
    store = get_eval_store()

    if not chunks:
        st.warning("No chunks selected.")
        return False

    st.subheader("Edit Query")

    # Pre-fill from suggestion if available
    default_query = suggestion.query if suggestion else ""
    default_type = suggestion.query_type if suggestion else QueryType.FACTUAL
    default_difficulty = suggestion.difficulty if suggestion else Difficulty.EASY
    default_synthesis = suggestion.requires_synthesis if suggestion else False
    default_notes = suggestion.notes if suggestion else ""

    # Auto-set requires_synthesis if multiple chunks
    if len(chunks) > 1 and not default_synthesis:
        default_synthesis = True

    # Use session state for query text to persist edits
    if state.query_text:
        default_query = state.query_text

    # Filter builder (outside form for dynamic buttons)
    parsed_filter = render_filter_builder("query_editor", existing_filter=None)

    with st.form("query_editor_form"):
        # Query text
        query_text = st.text_area(
            "Query text *",
            value=default_query,
            height=100,
            placeholder="Enter the question to ask...",
            help="The question that will be asked to the RAG system.",
        )

        # Expected answer
        expected_answer = st.text_area(
            "Expected answer",
            value=state.expected_answer,
            height=150,
            placeholder="Enter the expected answer (optional but recommended)...",
            help="The expected answer helps evaluate answer quality.",
        )

        # Query metadata
        col1, col2 = st.columns(2)

        with col1:
            query_type = st.selectbox(
                "Query type",
                options=list(QueryType),
                index=list(QueryType).index(default_type),
                format_func=lambda x: x.value,
                help="The type of query (factual, comparison, etc.)",
            )

            difficulty = st.selectbox(
                "Difficulty",
                options=list(Difficulty),
                index=list(Difficulty).index(default_difficulty),
                format_func=lambda x: x.value,
                help="How difficult is this query to answer?",
            )

        with col2:
            requires_synthesis = st.checkbox(
                "Requires synthesis",
                value=default_synthesis,
                help="Does the answer require combining multiple pieces of information?",
            )

            is_unanswerable = st.checkbox(
                "Unanswerable",
                value=False,
                help="Check if this query cannot be answered from the corpus.",
            )

        # Tags
        tags_input = st.text_input(
            "Tags (comma-separated)",
            value="curated",
            placeholder="tag1, tag2, tag3",
            help="Tags to categorize this query.",
        )

        # Notes
        notes = st.text_area(
            "Notes",
            value=default_notes,
            height=80,
            placeholder="Any notes about this query...",
            help="Free-form notes for evaluators.",
        )

        # Unanswerable reason (conditional)
        unanswerable_reason = None
        if is_unanswerable:
            unanswerable_reason = st.selectbox(
                "Unanswerable reason",
                options=["not_in_corpus", "ambiguous", "other"],
                help="Why is this query unanswerable?",
            )

        st.divider()

        # Chunk info (read-only) - show all selected chunks
        st.markdown(f"**Ground truth chunks ({len(chunks)}):**")
        for chunk in chunks:
            st.code(chunk.chunk_id, language=None)

        # Submit buttons
        col1, col2 = st.columns(2)

        with col1:
            save_clicked = st.form_submit_button(
                "Save Query",
                type="primary",
            )

        with col2:
            skip_clicked = st.form_submit_button("Skip (don't save)")

    # Handle skip
    if skip_clicked:
        reset_selection()
        st.info("Skipped. Select chunks for the next query.")
        return False

    # Handle save
    if save_clicked:
        # Validate
        if not query_text.strip():
            st.error("Query text is required.")
            return False

        # Parse tags
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]

        # Create EvalQuery with all chunk IDs
        qid = _generate_qid(state.output_path)
        relevant_chunk_ids = set() if is_unanswerable else {c.chunk_id for c in chunks}

        # Store all source chunk IDs in metadata
        source_chunk_ids = [c.chunk_id for c in chunks]

        # Build metadata dict
        metadata: dict = {"source_chunk_ids": source_chunk_ids}
        if parsed_filter is not None:
            metadata["filter"] = parsed_filter

        query = EvalQuery(
            qid=qid,
            query=query_text.strip(),
            relevant_chunk_ids=relevant_chunk_ids,
            expected_answer=expected_answer.strip() if expected_answer.strip() else None,
            query_type=query_type,
            difficulty=difficulty,
            requires_synthesis=requires_synthesis,
            notes=notes.strip() if notes and notes.strip() else None,
            tags=tags,
            created_at=datetime.now(UTC).isoformat(),
            created_by="curated",
            is_unanswerable=is_unanswerable,
            unanswerable_reason=unanswerable_reason,
            metadata=metadata,
        )

        # Save
        store.append_query(query, state.output_path)
        increment_counter()

        st.success(f"Saved query `{qid}` to `{state.output_path}`")

        # Reset for next query
        reset_selection()

        return True

    return False
