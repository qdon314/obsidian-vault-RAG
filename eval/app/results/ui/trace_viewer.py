"""
Trace viewer component for displaying detailed pipeline execution traces.

Shows retrieved candidates, reranked results, context packing, and generation details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from eval.app.results.domain.models import LoadedRun, QueryTrace


def render_trace_viewer(
    loaded_run: LoadedRun,
    trace_id: str | None = None,
) -> None:
    """Render trace viewer for a loaded run.

    Args:
        loaded_run: The run containing traces.
        trace_id: Optional specific trace ID to display.
    """
    if not loaded_run.traces:
        st.info("No traces available for this run")
        return

    st.subheader("Pipeline Traces")
    st.caption(f"{len(loaded_run.traces)} traces available")

    # Trace selector
    trace_ids = list(loaded_run.traces.keys())

    default_index = trace_ids.index(trace_id) if trace_id and trace_id in loaded_run.traces else 0

    selected_trace_id = st.selectbox(
        "Select trace",
        options=trace_ids,
        index=default_index,
        format_func=lambda tid: _format_trace_option(loaded_run.traces[tid]),
        key="trace_selector",
    )

    if selected_trace_id:
        trace = loaded_run.traces[selected_trace_id]
        _render_trace_detail(trace)


def _format_trace_option(trace: QueryTrace) -> str:
    """Format trace for display in selector."""
    query_preview = trace.query[:50] + "..." if len(trace.query) > 50 else trace.query
    return f"{trace.trace_id[:8]}... | {query_preview}"


def _render_trace_detail(trace: QueryTrace) -> None:
    """Render detailed view of a single trace."""
    # Header with basic info
    st.markdown(f"**Trace ID:** `{trace.trace_id}`")
    st.markdown(f"**Query:** {trace.query}")

    if trace.created_at:
        st.caption(f"Created: {trace.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    # Pipeline stages as tabs
    tabs = st.tabs(
        [
            "Retrieval",
            "Reranking",
            "Context",
            "Generation",
            "Raw Data",
        ]
    )

    with tabs[0]:
        _render_retrieval_stage(trace)

    with tabs[1]:
        _render_reranking_stage(trace)

    with tabs[2]:
        _render_context_stage(trace)

    with tabs[3]:
        _render_generation_stage(trace)

    with tabs[4]:
        _render_raw_trace(trace)


def _render_retrieval_stage(trace: QueryTrace) -> None:
    """Render retrieval stage details."""
    st.markdown("### Retrieval Stage")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Top K", trace.top_k)
    with col2:
        st.metric("Retrieved", len(trace.retrieved_candidates))

    if not trace.retrieved_candidates:
        st.info("No candidates retrieved")
        return

    st.markdown("**Retrieved Candidates:**")

    for i, cand in enumerate(trace.retrieved_candidates, 1):
        with st.expander(f"Rank {i}: {_get_chunk_id(cand)}", expanded=(i <= 3)):
            _render_candidate(cand)


def _render_reranking_stage(trace: QueryTrace) -> None:
    """Render reranking stage details."""
    st.markdown("### Reranking Stage")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reranker", trace.reranker or "None")
    with col2:
        st.metric("Keep K", trace.keep_k or "All")
    with col3:
        if trace.reranked_candidates:
            st.metric("Reranked", len(trace.reranked_candidates))
        else:
            st.metric("Reranked", "N/A")

    if not trace.reranked_candidates:
        st.info("No reranking performed or data not available")
        return

    st.markdown("**Reranked Candidates:**")

    for i, cand in enumerate(trace.reranked_candidates, 1):
        with st.expander(f"Rank {i}: {_get_chunk_id(cand)}", expanded=(i <= 3)):
            _render_candidate(cand, show_rerank_score=True)


def _render_context_stage(trace: QueryTrace) -> None:
    """Render context building stage details."""
    st.markdown("### Context Building Stage")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Token Budget", trace.token_budget or "N/A")
    with col2:
        if trace.packed_chunk_ids:
            st.metric("Packed Chunks", len(trace.packed_chunk_ids))
        else:
            st.metric("Packed Chunks", "N/A")

    if trace.packed_chunk_ids:
        st.markdown("**Packed Chunk IDs:**")
        for i, chunk_id in enumerate(trace.packed_chunk_ids, 1):
            st.code(f"{i}. {chunk_id}", language=None)
    else:
        st.info("Context packing data not available")


def _render_generation_stage(trace: QueryTrace) -> None:
    """Render generation stage details."""
    st.markdown("### Generation Stage")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", trace.model or "N/A")
    with col2:
        st.metric("Latency", f"{trace.latency_ms} ms" if trace.latency_ms else "N/A")

    if trace.answer_text:
        st.markdown("**Generated Answer:**")
        st.text_area(
            "Answer",
            value=trace.answer_text,
            height=200,
            disabled=True,
            key=f"trace_answer_{trace.trace_id}",
        )

        if trace.citations:
            st.markdown(f"**Citations ({len(trace.citations)}):**")
            for i, cit in enumerate(trace.citations, 1):
                chunk_id = cit.get("chunk_id", "unknown")
                quote = cit.get("quote", "")
                st.markdown(f"**[{i}]** `{chunk_id}`")
                if quote:
                    st.caption(f'"{quote[:200]}..."' if len(quote) > 200 else f'"{quote}"')
    else:
        st.info("No answer generated")


def _render_raw_trace(trace: QueryTrace) -> None:
    """Render raw trace data as JSON."""
    st.markdown("### Raw Trace Data")
    st.json(trace.raw_data)


def _get_chunk_id(cand: dict) -> str:
    """Extract chunk ID from candidate dict."""
    if "chunk" in cand:
        return cand["chunk"].get("chunk_id", "unknown")[:16] + "..."
    return cand.get("chunk_id", "unknown")[:16] + "..."


def _render_candidate(cand: dict, show_rerank_score: bool = False) -> None:
    """Render a single candidate with chunk details."""
    chunk = cand.get("chunk", {})

    # Scores
    col1, col2 = st.columns(2)
    with col1:
        score = cand.get("score")
        if score is not None:
            st.metric("Retrieval Score", f"{score:.4f}")
    with col2:
        if show_rerank_score:
            rerank_score = cand.get("rerank_score")
            if rerank_score is not None:
                st.metric("Rerank Score", f"{rerank_score:.4f}")

    # Chunk info
    if chunk:
        st.markdown(f"**Chunk ID:** `{chunk.get('chunk_id', 'N/A')}`")
        st.markdown(f"**Doc ID:** `{chunk.get('doc_id', 'N/A')[:16]}...`")

        if chunk.get("section_heading"):
            st.markdown(f"**Section:** {chunk['section_heading']}")

        # Chunk text
        text = chunk.get("text", "")
        if text:
            st.text_area(
                "Chunk Text",
                value=text,
                height=150,
                disabled=True,
                key=f"chunk_{chunk.get('chunk_id', 'unknown')}_{id(cand)}",
            )

        # Metadata
        metadata = chunk.get("metadata", {})
        if metadata:
            with st.expander("Chunk Metadata"):
                st.json(metadata)
