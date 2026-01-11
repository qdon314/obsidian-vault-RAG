"""
Query generator component using LLM suggestions.
"""

from __future__ import annotations

import streamlit as st

from experiments.ui.state import get_query_suggester, get_state
from rag.domain.models import Chunk
from rag.eval.schema import QuerySuggestion


def render_query_generator(chunk: Chunk) -> QuerySuggestion | None:
    """
    Render the query generator widget.

    Shows LLM-generated suggestions and allows selection or manual entry.
    Returns the selected/entered suggestion, or None if nothing selected.
    """
    state = get_state()
    suggester = get_query_suggester()

    st.subheader("Generate Query")

    if not suggester:
        st.warning(
            "No OpenAI API key configured. You can still write queries manually below."
        )
        return None

    # Generation controls
    col1, col2 = st.columns([3, 1])

    with col1:
        num_suggestions = st.slider(
            "Number of suggestions",
            min_value=1,
            max_value=5,
            value=3,
            key="num_suggestions",
        )

    with col2:
        generate_clicked = st.button(
            "Generate",
            type="primary",
            key="generate_btn",
        )

    # Generate suggestions
    if generate_clicked:
        with st.spinner("Generating query suggestions..."):
            state.suggestions = suggester.suggest_queries(
                chunk=chunk,
                num_suggestions=num_suggestions,
            )
            state.selected_suggestion_idx = None

        if state.suggestions:
            st.success(f"Generated {len(state.suggestions)} suggestions!")
        else:
            st.error("Failed to generate suggestions. Check logs for details.")

    # Display suggestions
    if state.suggestions:
        st.markdown("**Suggestions:**")

        for i, suggestion in enumerate(state.suggestions):
            col1, col2, col3 = st.columns([4, 1, 1])

            with col1:
                st.markdown(f"**{i+1}.** {suggestion.query}")

            with col2:
                st.caption(f"{suggestion.query_type.value}")

            with col3:
                if st.button("Use", key=f"use_suggestion_{i}"):
                    state.selected_suggestion_idx = i
                    state.query_text = suggestion.query
                    st.rerun()

        # Show notes for suggestions
        with st.expander("Suggestion details"):
            for i, suggestion in enumerate(state.suggestions):
                st.markdown(f"**{i+1}. {suggestion.query}**")
                st.markdown(f"- Type: {suggestion.query_type.value}")
                st.markdown(f"- Difficulty: {suggestion.difficulty.value}")
                st.markdown(f"- Requires synthesis: {suggestion.requires_synthesis}")
                if suggestion.notes:
                    st.markdown(f"- Notes: {suggestion.notes}")
                st.divider()

    # Return selected suggestion with bounds check
    if (
        state.selected_suggestion_idx is not None
        and 0 <= state.selected_suggestion_idx < len(state.suggestions)
    ):
        return state.suggestions[state.selected_suggestion_idx]

    return None
