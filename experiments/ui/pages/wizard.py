"""
Wizard-style page for query curation.

Steps:
1. Select a chunk from the browser
2. Generate query suggestions (optional)
3. Edit and save the query
"""

from __future__ import annotations

import streamlit as st

from experiments.ui.components import (
    render_chunk_browser,
    render_query_editor,
    render_query_generator,
)
from experiments.ui.state import get_state


def render_wizard_page() -> None:
    """Render the query curation wizard."""
    state = get_state()

    # Progress indicator
    st.markdown(f"**Queries created this session:** {state.query_counter}")
    st.divider()

    # Step 1: Chunk selection
    st.header("Step 1: Select a Chunk")
    selected_chunk = render_chunk_browser()

    if not selected_chunk:
        st.info("Select a chunk to continue.")
        return

    st.divider()

    # Step 2: Query generation (optional)
    st.header("Step 2: Generate Query (Optional)")
    suggestion = render_query_generator(selected_chunk)

    st.divider()

    # Step 3: Edit and save
    st.header("Step 3: Edit & Save")

    # Show the chunk content again for reference
    with st.expander("Show chunk content for reference", expanded=True):
        st.text_area(
            "Chunk text",
            value=selected_chunk.text,
            height=150,
            disabled=True,
            key="chunk_reference",
        )

    saved = render_query_editor(selected_chunk, suggestion)

    if saved:
        st.balloons()
        st.info("Query saved! Select another chunk to continue, or close the app.")
