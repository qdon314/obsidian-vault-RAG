"""
Chunk browser component with document tree and searchable list.
"""

from __future__ import annotations

import streamlit as st

from experiments.ui.state import get_state, load_chunks
from rag.domain.models import Chunk


def _get_doc_display_name(doc_id: str, chunks: list[Chunk]) -> str:
    """Get a display name for a document from its chunks."""
    if not chunks:
        return doc_id[:16] + "..."

    # Try to get URI from first chunk's metadata
    first_chunk = chunks[0]
    uri = first_chunk.metadata.get("uri", "")
    if uri:
        # Extract filename from path
        from pathlib import Path

        return Path(uri).name

    title = first_chunk.metadata.get("title", "")
    if title:
        return title

    return doc_id[:16] + "..."


def _get_chunk_preview(chunk: Chunk, max_len: int = 100) -> str:
    """Get a preview of chunk content."""
    text = chunk.text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def render_chunk_browser() -> Chunk | None:
    """
    Render the chunk browser with document tree and searchable list.

    Returns the selected chunk, or None if nothing selected.
    """
    state = get_state()

    # Load chunks if not loaded
    if not state.chunks:
        with st.spinner("Loading chunks..."):
            load_chunks()

    if not state.chunks:
        st.warning("No chunks loaded. Check your chunks path.")
        return None

    # Create two columns: document tree on left, chunk list on right
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Documents")

        # Build document options
        doc_options = []
        for doc_id, chunks in state.chunks_by_doc.items():
            display_name = _get_doc_display_name(doc_id, chunks)
            doc_options.append((doc_id, f"{display_name} ({len(chunks)} chunks)"))

        # Sort by display name
        doc_options.sort(key=lambda x: x[1].lower())

        # Document selector
        doc_labels = [opt[1] for opt in doc_options]
        doc_ids = [opt[0] for opt in doc_options]

        selected_idx = st.selectbox(
            "Select document",
            range(len(doc_labels)),
            format_func=lambda i: doc_labels[i],
            key="doc_selector",
        )

        if selected_idx is not None:
            state.selected_doc_id = doc_ids[selected_idx]

    with col2:
        st.subheader("Chunks")

        if state.selected_doc_id:
            doc_chunks = state.chunks_by_doc.get(state.selected_doc_id, [])

            # Search filter
            search_query = st.text_input(
                "Search chunks",
                placeholder="Filter by text content...",
                key="chunk_search",
            )

            # Filter chunks
            if search_query:
                filtered_chunks = [
                    c for c in doc_chunks if search_query.lower() in c.text.lower()
                ]
            else:
                filtered_chunks = doc_chunks

            if not filtered_chunks:
                st.info("No chunks match your search.")
                return None

            # Chunk selector
            chunk_options = []
            for chunk in filtered_chunks:
                heading = chunk.section_heading or "No heading"
                preview = _get_chunk_preview(chunk, max_len=60)
                label = f"[{chunk.chunk_index}] {heading}: {preview}"
                chunk_options.append((chunk, label))

            chunk_labels = [opt[1] for opt in chunk_options]
            chunk_objs = [opt[0] for opt in chunk_options]

            selected_chunk_idx = st.selectbox(
                f"Select chunk ({len(filtered_chunks)} available)",
                range(len(chunk_labels)),
                format_func=lambda i: chunk_labels[i],
                key="chunk_selector",
            )

            if selected_chunk_idx is not None:
                state.selected_chunk = chunk_objs[selected_chunk_idx]

    # Show selected chunk preview
    if state.selected_chunk:
        st.divider()
        st.subheader("Selected Chunk")

        chunk = state.selected_chunk
        st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
        st.markdown(f"**Section:** {chunk.section_heading or 'N/A'}")
        st.markdown(f"**Path:** {chunk.section_path or 'N/A'}")

        st.text_area(
            "Content",
            value=chunk.text,
            height=200,
            disabled=True,
            key=f"chunk_preview_{chunk.chunk_id}",
        )

        return chunk

    return None
