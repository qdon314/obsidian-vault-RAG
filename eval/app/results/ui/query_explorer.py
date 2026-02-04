"""
Query explorer component for browsing and filtering individual query results.

Provides interactive filtering, sorting, and detailed views of query results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from eval.app.results.services.filter_service import FilterService
from rag.eval.schema import Difficulty, QueryType

if TYPE_CHECKING:
    from eval.app.results.domain.models import LoadedRun, QueryTrace
    from rag.eval.models import EvalResult


def render_query_explorer(
    loaded_run: LoadedRun,
    filter_service: FilterService,
) -> None:
    """Render interactive query explorer with filters.

    Args:
        loaded_run: The run containing results to explore.
        filter_service: Service for filtering results.
    """
    st.subheader("Query Explorer")

    results = list(loaded_run.results)

    if not results:
        st.info("No query results in this run")
        return

    # Filter controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Query type filter
        available_types = sorted({r.query_type.value for r in results if r.query_type is not None})
        selected_types = st.multiselect(
            "Query Type",
            options=available_types,
            default=None,
            key="explorer_query_types",
        )

    with col2:
        # Difficulty filter
        available_diffs = sorted({r.difficulty.value for r in results if r.difficulty is not None})
        selected_diffs = st.multiselect(
            "Difficulty",
            options=available_diffs,
            default=None,
            key="explorer_difficulties",
        )

    with col3:
        # Pass/fail filter
        pass_fail = st.selectbox(
            "Pass/Fail",
            options=["All", "Pass (recall >= 0.5)", "Fail (recall < 0.5)"],
            key="explorer_pass_fail",
        )

    with col4:
        # Text search
        query_search = st.text_input(
            "Search queries",
            placeholder="Search text...",
            key="explorer_search",
        )

    # Apply filters
    query_types = {QueryType(t) for t in selected_types} if selected_types else None
    difficulties = {Difficulty(d) for d in selected_diffs} if selected_diffs else None
    pass_only = "Pass" in pass_fail
    fail_only = "Fail" in pass_fail

    filtered = filter_service.filter_results(
        results,
        query_types=query_types,
        difficulties=difficulties,
        pass_only=pass_only,
        fail_only=fail_only,
        query_search=query_search if query_search else None,
    )

    st.write(f"Showing {len(filtered)} / {len(results)} queries")

    if not filtered:
        st.info("No queries match the current filters")
        return

    # Build results dataframe
    import pandas as pd

    data = []
    for r in filtered:
        recall = filter_service.compute_recall(r, k=10)
        status = "Pass" if recall >= 0.5 else "Fail"

        row = {
            "Status": status,
            "QID": r.qid,
            "Query": r.query[:80] + "..." if len(r.query) > 80 else r.query,
            "Type": r.query_type.value if r.query_type else "N/A",
            "Difficulty": r.difficulty.value if r.difficulty else "N/A",
            "Recall@10": f"{recall:.2f}",
            "Latency": f"{r.latency_ms}ms" if r.latency_ms else "N/A",
        }
        data.append(row)

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detail view
    st.divider()
    st.subheader("Query Details")

    # Select query to view details
    qid_options = [r.qid for r in filtered]
    selected_qid = st.selectbox(
        "Select query to view details",
        options=qid_options,
        format_func=lambda qid: next(
            (f"{qid}: {r.query[:50]}..." for r in filtered if r.qid == qid),
            f"{qid}: (query not found)",
        ),
        key="explorer_selected_query",
    )

    if selected_qid:
        result = next((r for r in filtered if r.qid == selected_qid), None)
        if result is None:
            st.error(f"Query {selected_qid} not found")
            return

        # Get trace if available
        trace = None
        if result.trace_id and loaded_run.traces:
            trace = loaded_run.traces.get(result.trace_id)

        _render_query_detail(result, filter_service, trace)


def _render_query_detail(
    result: EvalResult,
    filter_service: FilterService,
    trace: QueryTrace | None = None,
) -> None:
    """Render detailed view of a single query result."""
    # Query info
    st.markdown(f"**Query:** {result.query}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Type:** {result.query_type.value if result.query_type else 'N/A'}")
    with col2:
        st.markdown(f"**Difficulty:** {result.difficulty.value if result.difficulty else 'N/A'}")
    with col3:
        st.markdown(f"**Unanswerable:** {'Yes' if result.is_unanswerable else 'No'}")

    st.divider()

    # Retrieval results
    st.markdown("### Retrieval Results")

    retrieved = set(result.retrieval_result.retrieved_chunk_ids)
    relevant = result.retrieval_result.relevant_chunk_ids
    matched = filter_service.get_matched_chunks(result)
    missed = filter_service.get_missed_chunks(result)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Retrieved", len(retrieved))
    with col2:
        st.metric("Relevant", len(relevant))
    with col3:
        recall = filter_service.compute_recall(result, k=10)
        st.metric("Recall@10", f"{recall:.2f}")

    if matched:
        with st.expander(f"Matched Chunks ({len(matched)})", expanded=True):
            for chunk_id in sorted(matched):
                rank = list(result.retrieval_result.retrieved_chunk_ids).index(chunk_id) + 1
                st.success(f"Rank {rank}: `{chunk_id}`")

    if missed:
        with st.expander(f"Missed Chunks ({len(missed)})", expanded=True):
            for chunk_id in sorted(missed):
                st.error(f"`{chunk_id}`")

    # Extra retrieved (not relevant)
    extra = retrieved - relevant
    if extra:
        with st.expander(f"Extra Retrieved ({len(extra)})", expanded=False):
            for chunk_id in list(result.retrieval_result.retrieved_chunk_ids):
                if chunk_id in extra:
                    rank = list(result.retrieval_result.retrieved_chunk_ids).index(chunk_id) + 1
                    st.warning(f"Rank {rank}: `{chunk_id}`")

    # Answer (if available)
    if result.answer:
        st.divider()
        st.markdown("### Generated Answer")

        st.text_area(
            "Answer text",
            value=result.answer.text,
            height=150,
            disabled=True,
            key=f"answer_text_{result.qid}",
        )

        if result.answer.citations:
            with st.expander(f"Citations ({len(result.answer.citations)})", expanded=False):
                for i, cit in enumerate(result.answer.citations, 1):
                    st.markdown(f"**[{i}]** `{cit.chunk_id}`")
                    if cit.quote:
                        st.caption(
                            f'"{cit.quote[:200]}..."'
                            if len(cit.quote or "") > 200
                            else f'"{cit.quote}"'
                        )

    # Answer metrics (if available)
    if result.answer_metrics:
        st.divider()
        st.markdown("### Answer Quality Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if result.answer_metrics.quality_score is not None:
                st.metric("Quality Score", f"{result.answer_metrics.quality_score:.2f}")
            if result.answer_metrics.correctness is not None:
                st.metric("Correctness (0-5)", f"{result.answer_metrics.correctness:.1f}")

        with col2:
            if result.answer_metrics.citation_coverage is not None:
                st.metric("Citation Coverage", f"{result.answer_metrics.citation_coverage:.2f}")
            if result.answer_metrics.completeness is not None:
                st.metric("Completeness (0-5)", f"{result.answer_metrics.completeness:.1f}")

        with col3:
            if result.answer_metrics.hallucination_severity is not None:
                st.metric(
                    "Hallucination Severity (0-5)",
                    f"{result.answer_metrics.hallucination_severity:.1f}",
                )
            if result.answer_metrics.semantic_similarity is not None:
                st.metric("Semantic Similarity", f"{result.answer_metrics.semantic_similarity:.2f}")
        with col4:
            if result.answer_metrics.relevance is not None:
                st.metric("Relevance (0-5)", f"{result.answer_metrics.relevance:.1f}")
        with col5:
            if result.outcome_label is not None:
                st.metric("Outcome Label", result.outcome_label.name)

    # Claims (if available)
    if result.groundedness_result and result.groundedness_result.claims:
        st.divider()
        st.markdown("### Groundedness Claims")

        # Summary stats
        supported = sum(1 for c in result.groundedness_result.claims if c.supported)
        unsupported = len(result.groundedness_result.claims) - supported

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Claims", len(result.groundedness_result.claims))
        with col2:
            st.metric("Supported", supported)
        with col3:
            st.metric("Unsupported", unsupported)

        # Individual claims
        for claim in result.groundedness_result.claims:
            icon = "✓" if claim.supported else "✗"
            status_color = "green" if claim.supported else "red"
            role_badge = "🔑" if claim.role == "core" else "📎"
            header = (
                f"{icon} {role_badge} {claim.claim[:80]}{'...' if len(claim.claim) > 80 else ''}"
            )

            with st.expander(header, expanded=not claim.supported):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Supported:** :{status_color}[{claim.supported}]")
                with col2:
                    role_color = "blue" if claim.role == "core" else "gray"
                    st.markdown(f"**Role:** :{role_color}[{claim.role.upper()}]")
                if claim.chunk_id:
                    st.markdown(f"**Chunk ID:** `{claim.chunk_id}`")
                if claim.supported and claim.quote:
                    st.markdown(f"**Quote:** _{claim.quote}_")
                if claim.claim:
                    st.markdown(f"**Claim:** {claim.claim}")
                if claim.note:
                    st.markdown(f"**Note:** {claim.note}")

    # Trace details
    if trace:
        st.divider()
        st.markdown("### Pipeline Trace")
        st.caption(f"Trace ID: `{result.trace_id}`")

        # Show trace in expandable sections
        with st.expander("Retrieved Candidates", expanded=False):
            if trace.retrieved_candidates:
                for i, cand in enumerate(trace.retrieved_candidates, 1):
                    chunk = cand.get("chunk", {})
                    score = cand.get("score", 0)
                    chunk_id = chunk.get("chunk_id", "unknown")
                    text_preview = chunk.get("text", "")[:100] + "..."

                    st.markdown(f"**{i}. `{chunk_id}`** (score: {score:.4f})")
                    st.caption(text_preview)
            else:
                st.info("No retrieved candidates data")

        if trace.reranked_candidates:
            with st.expander("Reranked Candidates", expanded=False):
                for i, cand in enumerate(trace.reranked_candidates, 1):
                    chunk = cand.get("chunk", {})
                    score = cand.get("score", 0)
                    rerank_score = cand.get("rerank_score")
                    chunk_id = chunk.get("chunk_id", "unknown")

                    score_str = f"score: {score:.4f}"
                    if rerank_score is not None:
                        score_str += f", rerank: {rerank_score:.4f}"

                    st.markdown(f"**{i}. `{chunk_id}`** ({score_str})")

        if trace.packed_chunk_ids:
            with st.expander("Packed Context", expanded=False):
                st.markdown(f"**Token Budget:** {trace.token_budget}")
                st.markdown(f"**Chunks Packed:** {len(trace.packed_chunk_ids)}")
                for i, chunk_id in enumerate(trace.packed_chunk_ids, 1):
                    st.code(f"{i}. {chunk_id}", language=None)

        with st.expander("Raw Trace Data", expanded=False):
            st.json(trace.raw_data)

    elif result.trace_id:
        st.divider()
        st.caption(f"Trace ID: `{result.trace_id}` (trace data not loaded)")
