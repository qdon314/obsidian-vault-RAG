"""
Metrics table component for displaying evaluation metrics.

Renders retrieval metrics, answer quality, and latency in tabular format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from eval.app.results.domain.models import LoadedRun
    from rag.eval.models import RetrievalSummary


def render_metrics_table(
    loaded_run: LoadedRun,
    *,
    show_by_type: bool = True,
    show_by_difficulty: bool = True,
) -> None:
    """Render comprehensive metrics tables for a loaded run.

    Args:
        loaded_run: The run to display metrics for.
        show_by_type: Whether to show breakdown by query type.
        show_by_difficulty: Whether to show breakdown by difficulty.
    """
    st.subheader("Overall Retrieval Metrics")
    _render_retrieval_table(loaded_run.aggregates.overall, key_prefix="overall")

    # By query type
    if show_by_type and loaded_run.aggregates.by_type:
        st.subheader("Metrics by Query Type")

        for qtype, summary in sorted(loaded_run.aggregates.by_type.items()):
            with st.expander(
                f"{qtype.upper()} ({int(summary.num_queries)} queries)",
                expanded=False,
            ):
                _render_retrieval_table(summary, key_prefix=f"type_{qtype}")

    # By difficulty
    if show_by_difficulty and loaded_run.aggregates.by_difficulty:
        st.subheader("Metrics by Difficulty")

        for diff, summary in sorted(loaded_run.aggregates.by_difficulty.items()):
            with st.expander(
                f"{diff.upper()} ({int(summary.num_queries)} queries)",
                expanded=False,
            ):
                _render_retrieval_table(summary, key_prefix=f"diff_{diff}")

    # Answer quality
    if loaded_run.aggregates.answer_quality:
        st.subheader("Answer Quality Metrics")
        _render_answer_quality_table(loaded_run.aggregates.answer_quality)

    # Latency
    if loaded_run.aggregates.latency_ms:
        st.subheader("Latency Statistics")
        _render_latency_table(loaded_run.aggregates.latency_ms)


def _render_retrieval_table(
    summary: RetrievalSummary,
    key_prefix: str,
) -> None:
    """Render a retrieval metrics table."""
    # Determine available k values
    k_values = sorted(summary.recall_at_k.keys())

    if not k_values:
        st.info("No retrieval metrics available")
        return

    # Build table data
    metrics = ["Recall", "Precision", "Hit Rate", "NDCG"]
    data: dict[str, list[str]] = {"Metric": metrics}

    for k in k_values:
        col_values = [
            f"{summary.recall_at_k.get(k, 0):.3f}",
            f"{summary.precision_at_k.get(k, 0):.3f}",
            f"{summary.hit_rate_at_k.get(k, 0):.3f}",
            f"{summary.ndcg_at_k.get(k, 0):.3f}",
        ]
        data[f"@{k}"] = col_values

    # Display as dataframe
    import pandas as pd

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Global metrics below
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MRR", f"{summary.mrr:.3f}")
    with col2:
        st.metric("MAP", f"{summary.map:.3f}")
    with col3:
        st.metric("Avg Retrieved", f"{summary.avg_retrieved:.1f}")


def _render_answer_quality_table(answer_quality: dict[str, float]) -> None:
    """Render answer quality metrics."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if "avg_quality_score" in answer_quality:
            st.metric(
                "Avg Quality Score",
                f"{answer_quality['avg_quality_score']:.3f}",
            )
        if "median_quality_score" in answer_quality:
            st.metric(
                "Median Quality Score",
                f"{answer_quality['median_quality_score']:.3f}",
            )

    with col2:
        if "avg_correctness_0_5" in answer_quality:
            st.metric(
                "Avg Correctness (0-5)",
                f"{answer_quality['avg_correctness_0_5']:.2f}",
            )
        if "avg_citation_coverage" in answer_quality:
            st.metric(
                "Avg Citation Coverage",
                f"{answer_quality['avg_citation_coverage']:.3f}",
            )

    with col3:
        if "avg_hallucination_severity_0_5" in answer_quality:
            st.metric(
                "Avg Hallucination Severity (0-5)",
                f"{answer_quality['avg_hallucination_severity_0_5']:.2f}",
            )
        if "evidence_bounded_rate" in answer_quality:
            st.metric(
                "Evidence Bounded Rate",
                f"{answer_quality['evidence_bounded_rate']:.1%}",
                help="Percentage of answers where all claims are supported by context",
            )
        if "hallucinated_on_unanswerable_rate" in answer_quality:
            st.metric(
                "Hallucinated on Unanswerable",
                f"{answer_quality['hallucinated_on_unanswerable_rate']:.1%}",
                help="Of queries where context was insufficient, how often did the model hallucinate instead of abstaining",
            )


def _render_latency_table(latency_ms: dict[str, float]) -> None:
    """Render latency statistics."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if "avg" in latency_ms:
            st.metric("Average", f"{latency_ms['avg']:.0f} ms")

    with col2:
        if "p50" in latency_ms:
            st.metric("Median (P50)", f"{latency_ms['p50']:.0f} ms")

    with col3:
        if "p95" in latency_ms:
            st.metric("P95", f"{latency_ms['p95']:.0f} ms")
