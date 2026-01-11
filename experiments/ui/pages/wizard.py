"""
Wizard-style page for query curation.

Modes:
- Create: Select chunks, generate suggestions, edit and save new queries
- Review: View, edit, and delete existing queries

Steps (Create mode):
1. Select chunks (can add multiple)
2. Generate query suggestions (optional)
3. Edit and save the query
"""

from __future__ import annotations

import streamlit as st

from experiments.ui.components import (
    render_chunk_browser,
    render_query_edit_form,
    render_query_editor,
    render_query_generator,
    render_query_list,
)
from experiments.ui.state import (
    add_chunk_to_selection,
    get_state,
    remove_chunk_from_selection,
)


def _render_selected_chunks() -> None:
    """Render the list of currently selected chunks with remove buttons."""
    state = get_state()

    if not state.selected_chunks:
        st.info("No chunks added yet. Browse and add chunks below.")
        return

    st.markdown(f"**{len(state.selected_chunks)} chunk(s) selected:**")

    for chunk in state.selected_chunks:
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            # Show chunk preview
            preview = chunk.text[:80].replace("\n", " ") + "..."
            st.markdown(f"**{chunk.section_heading or 'No heading'}**")
            st.caption(preview)

        with col2:
            # Show source
            uri = chunk.metadata.get("uri", "")
            if uri:
                from pathlib import Path

                st.caption(Path(uri).name)

        with col3:
            if st.button("Remove", key=f"remove_{chunk.chunk_id}"):
                remove_chunk_from_selection(chunk.chunk_id)
                st.rerun()


def _render_create_tab() -> None:
    """Render the Create New Query tab content."""
    state = get_state()

    # Progress indicator
    st.markdown(f"**Queries created this session:** {state.query_counter}")
    st.divider()

    # Step 1: Chunk selection
    st.header("Step 1: Select Chunks")

    # Show currently selected chunks
    _render_selected_chunks()

    st.divider()

    # Chunk browser for adding more
    with st.expander("Browse & Add Chunks", expanded=len(state.selected_chunks) == 0):
        selected_chunk = render_chunk_browser()

        if selected_chunk:
            # Check if already added
            already_added = selected_chunk.chunk_id in {
                c.chunk_id for c in state.selected_chunks
            }

            if already_added:
                st.success("This chunk is already in your selection.")
            else:
                if st.button("Add this chunk", type="primary", key="add_chunk_btn"):
                    add_chunk_to_selection(selected_chunk)
                    st.rerun()

    # Can't proceed without at least one chunk
    if not state.selected_chunks:
        st.warning("Add at least one chunk to continue.")
        return

    st.divider()

    # Step 2: Query generation (optional)
    st.header("Step 2: Generate Query (Optional)")

    # Use first chunk for generation (or could combine later)
    primary_chunk = state.selected_chunks[0]
    if len(state.selected_chunks) > 1:
        st.caption(
            f"Query suggestions will be based on the first chunk: "
            f"'{primary_chunk.section_heading or 'No heading'}'"
        )

    suggestion = render_query_generator(primary_chunk)

    st.divider()

    # Step 3: Edit and save
    st.header("Step 3: Edit & Save")

    # Show all selected chunks for reference
    with st.expander(
        f"Show selected chunks ({len(state.selected_chunks)})", expanded=True
    ):
        for i, chunk in enumerate(state.selected_chunks):
            st.markdown(f"**Chunk {i + 1}: {chunk.section_heading or 'No heading'}**")
            st.text_area(
                "Content",
                value=chunk.text,
                height=120,
                disabled=True,
                key=f"chunk_ref_{i}",
            )
            if i < len(state.selected_chunks) - 1:
                st.divider()

    saved = render_query_editor(state.selected_chunks, suggestion)

    if saved:
        st.balloons()
        st.info("Query saved! Select chunks for the next query, or close the app.")


def _render_review_tab() -> None:
    """Render the Review/Edit Existing Queries tab content."""
    state = get_state()

    # If editing a specific query, show the edit form
    if state.editing_query:
        updated = render_query_edit_form()
        if updated:
            st.rerun()
        return

    # Otherwise show the query list
    render_query_list()


def render_wizard_page() -> None:
    """Render the query curation wizard with tabs for create/review modes."""
    # Tab selection
    tab_create, tab_review = st.tabs(["Create New Query", "Review Existing"])

    with tab_create:
        _render_create_tab()

    with tab_review:
        _render_review_tab()
