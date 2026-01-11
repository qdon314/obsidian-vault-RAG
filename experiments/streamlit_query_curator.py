#!/usr/bin/env python3
"""
Streamlit app for curating evaluation queries.

This tool provides a wizard-style UI for:
1. Browsing and selecting chunks from your indexed vault
2. Generating query suggestions using an LLM
3. Editing all EvalQuery fields
4. Saving queries to a JSONL file

Usage:
    streamlit run experiments/streamlit_query_curator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st  # noqa: E402

from experiments.ui.pages import render_wizard_page  # noqa: E402
from experiments.ui.state import get_eval_store, get_state, init_services, init_state  # noqa: E402


def render_sidebar() -> None:
    """Render the sidebar with configuration options."""
    state = get_state()
    store = get_eval_store()

    st.sidebar.title("Configuration")

    # Chunks path
    chunks_path_str = st.sidebar.text_input(
        "Chunks path",
        value=str(state.chunks_path),
        help="Path to the JSONL file containing indexed chunks.",
    )
    state.chunks_path = Path(chunks_path_str)

    # Output path
    output_path_str = st.sidebar.text_input(
        "Output path",
        value=str(state.output_path),
        help="Path to save curated queries (JSONL format).",
    )
    state.output_path = Path(output_path_str)

    st.sidebar.divider()

    # Stats
    st.sidebar.subheader("Stats")

    # Chunk count
    if state.chunks:
        st.sidebar.metric("Loaded chunks", len(state.chunks))
        st.sidebar.metric("Documents", len(state.chunks_by_doc))
    else:
        st.sidebar.info("Chunks not loaded yet.")

    # Query count
    if state.output_path.exists():
        query_count = store.count_queries(state.output_path)
        st.sidebar.metric("Queries in output file", query_count)
    else:
        st.sidebar.info("Output file will be created on first save.")

    st.sidebar.metric("Queries this session", state.query_counter)

    st.sidebar.divider()

    # Help
    st.sidebar.subheader("Help")
    st.sidebar.markdown("""
    **Workflow:**
    1. Browse documents and select a chunk
    2. Optionally generate query suggestions
    3. Edit the query form
    4. Save and continue

    **Tips:**
    - Use search to filter chunks
    - Generated queries are just suggestions
    - Fill in expected answers for better evaluation
    """)


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Query Curator",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize state and services
    init_state()
    init_services()

    # Header
    st.title("Query Curation Tool")
    st.markdown(
        "Create high-quality evaluation queries by browsing your indexed chunks."
    )

    # Sidebar
    render_sidebar()

    # Main content
    render_wizard_page()


if __name__ == "__main__":
    main()
