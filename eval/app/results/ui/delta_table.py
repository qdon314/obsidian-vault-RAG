"""
Delta table component for showing metric differences between runs.

Displays metrics with visual indicators for improvements/regressions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from eval.app.results.domain.models import RunComparison


def render_delta_table(comparison: RunComparison) -> None:
    """Render table showing metric deltas between two runs.

    Args:
        comparison: RunComparison with computed deltas.
    """
    import pandas as pd

    st.subheader("Metric Deltas")

    data = []

    agg_a = comparison.run_a.aggregates.overall
    agg_b = comparison.run_b.aggregates.overall

    # Recall deltas
    for k in sorted(comparison.recall_delta.keys()):
        delta = comparison.recall_delta[k]
        a_val = agg_a.recall_at_k.get(k, 0)
        b_val = agg_b.recall_at_k.get(k, 0)
        data.append(_build_row(f"Recall@{k}", a_val, b_val, delta))

    # Tiered recall deltas
    for k in sorted(comparison.critical_recall_delta.keys()):
        delta = comparison.critical_recall_delta[k]
        a_val = agg_a.critical_recall_at_k.get(k, 0)
        b_val = agg_b.critical_recall_at_k.get(k, 0)
        data.append(_build_row(f"Critical Recall@{k}", a_val, b_val, delta))

    for k in sorted(comparison.weighted_recall_delta.keys()):
        delta = comparison.weighted_recall_delta[k]
        a_val = agg_a.weighted_recall_at_k.get(k, 0)
        b_val = agg_b.weighted_recall_at_k.get(k, 0)
        data.append(_build_row(f"Weighted Recall@{k}", a_val, b_val, delta))

    for k in sorted(comparison.critical_hit_rate_delta.keys()):
        delta = comparison.critical_hit_rate_delta[k]
        a_val = agg_a.critical_hit_rate_at_k.get(k, 0)
        b_val = agg_b.critical_hit_rate_at_k.get(k, 0)
        data.append(_build_row(f"Critical Hit Rate@{k}", a_val, b_val, delta))

    # Precision deltas
    for k in sorted(comparison.precision_delta.keys()):
        delta = comparison.precision_delta[k]
        a_val = agg_a.precision_at_k.get(k, 0)
        b_val = agg_b.precision_at_k.get(k, 0)
        data.append(_build_row(f"Precision@{k}", a_val, b_val, delta))

    # NDCG deltas
    for k in sorted(comparison.ndcg_delta.keys()):
        delta = comparison.ndcg_delta[k]
        a_val = agg_a.ndcg_at_k.get(k, 0)
        b_val = agg_b.ndcg_at_k.get(k, 0)
        data.append(_build_row(f"NDCG@{k}", a_val, b_val, delta))

    # Global metrics
    data.append(_build_row("MRR", agg_a.mrr, agg_b.mrr, comparison.mrr_delta))
    data.append(_build_row("MAP", agg_a.map, agg_b.map, comparison.map_delta))

    # Quality delta
    if comparison.quality_delta is not None:
        qa_a = comparison.run_a.aggregates.answer_quality or {}
        qa_b = comparison.run_b.aggregates.answer_quality or {}
        data.append(
            _build_row(
                "Quality Score",
                qa_a.get("avg_quality_score", 0),
                qa_b.get("avg_quality_score", 0),
                comparison.quality_delta,
            )
        )

    # Latency delta (note: lower is better for latency)
    if comparison.latency_delta is not None:
        lat_a = comparison.run_a.aggregates.latency_ms or {}
        lat_b = comparison.run_b.aggregates.latency_ms or {}
        data.append(
            _build_row(
                "Avg Latency (ms)",
                lat_a.get("avg", 0),
                lat_b.get("avg", 0),
                comparison.latency_delta,
                lower_is_better=True,
                format_ms=True,
            )
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _build_row(
    metric: str,
    a_val: float,
    b_val: float,
    delta: float,
    lower_is_better: bool = False,
    format_ms: bool = False,
) -> dict:
    """Build a row for the delta table."""
    # Determine change indicator
    threshold = 0.01 if not format_ms else 10  # 1% change for metrics, 10ms for latency

    if lower_is_better:
        # For latency, negative delta is good
        if delta < -threshold:
            change = "Better"
        elif delta > threshold:
            change = "Worse"
        else:
            change = "Same"
    else:
        # For other metrics, positive delta is good
        if delta > threshold:
            change = "Better"
        elif delta < -threshold:
            change = "Worse"
        else:
            change = "Same"

    # Format values
    if format_ms:
        a_str = f"{a_val:.0f}"
        b_str = f"{b_val:.0f}"
        delta_str = f"{delta:+.0f}"
    else:
        a_str = f"{a_val:.3f}"
        b_str = f"{b_val:.3f}"
        delta_str = f"{delta:+.3f}"

    return {
        "Metric": metric,
        "Run A": a_str,
        "Run B": b_str,
        "Delta": delta_str,
        "Change": change,
    }


def render_summary_metrics(comparison: RunComparison) -> None:
    """Render high-level summary metrics with delta indicators.

    Shows the most important metrics with st.metric delta display.
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    agg_b = comparison.run_b.aggregates.overall

    with col1:
        recall_10 = agg_b.recall_at_k.get(10, 0)
        delta = comparison.recall_delta.get(10, 0)
        st.metric(
            "Recall@10",
            f"{recall_10:.3f}",
            delta=f"{delta:+.3f}",
            delta_color="normal",
        )

    with col2:
        critical_recall_10 = agg_b.critical_recall_at_k.get(10)
        if critical_recall_10 is not None:
            delta = comparison.critical_recall_delta.get(10, 0)
            st.metric(
                "Critical Recall@10",
                f"{critical_recall_10:.3f}",
                delta=f"{delta:+.3f}",
                delta_color="normal",
            )
        else:
            st.metric("Critical Recall@10", "N/A")

    with col3:
        ndcg_10 = agg_b.ndcg_at_k.get(10, 0)
        delta = comparison.ndcg_delta.get(10, 0)
        st.metric(
            "NDCG@10",
            f"{ndcg_10:.3f}",
            delta=f"{delta:+.3f}",
            delta_color="normal",
        )

    with col4:
        st.metric(
            "MRR",
            f"{agg_b.mrr:.3f}",
            delta=f"{comparison.mrr_delta:+.3f}",
            delta_color="normal",
        )

    with col5:
        if comparison.quality_delta is not None:
            qa_b = comparison.run_b.aggregates.answer_quality or {}
            quality = qa_b.get("avg_quality_score", 0)
            st.metric(
                "Quality Score",
                f"{quality:.3f}",
                delta=f"{comparison.quality_delta:+.3f}",
                delta_color="normal",
            )
        else:
            st.metric("Quality Score", "N/A")


def render_query_changes_summary(comparison: RunComparison) -> None:
    """Render summary of query-level changes."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Improved Queries",
            len(comparison.improved_queries),
            delta=None,
        )

    with col2:
        st.metric(
            "Regressed Queries",
            len(comparison.regressed_queries),
            delta=None,
        )

    with col3:
        st.metric(
            "Unchanged Queries",
            len(comparison.unchanged_queries),
            delta=None,
        )
