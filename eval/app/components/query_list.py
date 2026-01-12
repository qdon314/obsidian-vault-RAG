"""
Query list component for viewing and selecting existing queries.
"""

from __future__ import annotations

import streamlit as st

from eval.app.state import (
    delete_query,
    get_state,
    load_existing_queries,
    set_editing_query,
)
from rag.eval.schema import Difficulty, EvalQuery, QueryType


def render_query_list() -> None:
    """
    Render the list of existing queries with filtering and actions.

    Allows viewing, editing, and deleting queries from the output file.
    """
    state = get_state()

    st.subheader("Existing Queries")

    # Load/refresh button
    col1, _ = st.columns([1, 4])
    with col1:
        if st.button("Refresh", key="refresh_queries"):
            load_existing_queries()
            st.rerun()

    # Load on first render
    if not state.existing_queries:
        load_existing_queries()

    if not state.existing_queries:
        st.info(f"No queries found in `{state.output_path}`")
        return

    # Filters
    with st.expander("Filters", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            filter_type = st.selectbox(
                "Query type",
                options=[None, *list(QueryType)],
                format_func=lambda x: "All" if x is None else x.value,
                key="filter_query_type",
            )

        with filter_col2:
            filter_difficulty = st.selectbox(
                "Difficulty",
                options=[None, *list(Difficulty)],
                format_func=lambda x: "All" if x is None else x.value,
                key="filter_difficulty",
            )

        with filter_col3:
            search_text = st.text_input(
                "Search query text",
                key="search_query_text",
                placeholder="Search...",
            )

    # Apply filters
    filtered = state.existing_queries
    if filter_type:
        filtered = [q for q in filtered if q.query_type == filter_type]
    if filter_difficulty:
        filtered = [q for q in filtered if q.difficulty == filter_difficulty]
    if search_text:
        search_lower = search_text.lower()
        filtered = [q for q in filtered if search_lower in q.query.lower()]

    st.caption(f"Showing {len(filtered)} of {len(state.existing_queries)} queries")

    # Query list
    for query in filtered:
        _render_query_card(query)


def _render_query_card(query: EvalQuery) -> None:
    """Render a single query card with actions."""
    with st.container():
        # Header row
        col1, col2, col3, col4 = st.columns([4, 1, 1, 1])

        with col1:
            st.markdown(f"**{query.query}**")

        with col2:
            st.caption(query.query_type.value)

        with col3:
            st.caption(query.difficulty.value)

        with col4:
            # Action buttons
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Edit", key=f"edit_{query.qid}"):
                    set_editing_query(query)
                    st.rerun()
            with btn_col2:
                if st.button("Del", key=f"del_{query.qid}"):
                    st.session_state[f"confirm_delete_{query.qid}"] = True
                    st.rerun()

        # Delete confirmation
        if st.session_state.get(f"confirm_delete_{query.qid}"):
            st.warning(f"Delete query `{query.qid}`?")
            confirm_col1, confirm_col2 = st.columns(2)
            with confirm_col1:
                if st.button("Yes, delete", key=f"confirm_yes_{query.qid}"):
                    delete_query(query.qid)
                    st.session_state[f"confirm_delete_{query.qid}"] = False
                    st.success("Deleted!")
                    st.rerun()
            with confirm_col2:
                if st.button("Cancel", key=f"confirm_no_{query.qid}"):
                    st.session_state[f"confirm_delete_{query.qid}"] = False
                    st.rerun()

        # Details expander
        with st.expander("Details", expanded=False):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown(f"**QID:** `{query.qid}`")
                st.markdown(f"**Requires synthesis:** {query.requires_synthesis}")
                st.markdown(f"**Unanswerable:** {query.is_unanswerable}")
                if query.tags:
                    st.markdown(f"**Tags:** {', '.join(query.tags)}")

            with detail_col2:
                st.markdown(f"**Chunks:** {len(query.relevant_chunk_ids)}")
                if query.created_at:
                    st.markdown(f"**Created:** {query.created_at[:10]}")
                if query.created_by:
                    st.markdown(f"**By:** {query.created_by}")

            if query.expected_answer:
                st.markdown("**Expected answer:**")
                st.text(query.expected_answer)

            if query.relevant_chunk_ids:
                st.markdown("**Relevant chunks:**")
                for chunk_id in sorted(query.relevant_chunk_ids):
                    st.code(chunk_id, language=None)

            if query.notes:
                st.markdown(f"**Notes:** {query.notes}")

        st.divider()
