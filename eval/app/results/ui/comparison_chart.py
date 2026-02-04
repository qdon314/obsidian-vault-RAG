"""
Comparison chart component for visualizing differences between runs.

Uses Plotly for interactive bar charts comparing metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from eval.app.results.domain.models import RunComparison


def render_comparison_chart(
    comparison: RunComparison,
    metric: str = "recall",
) -> None:
    """Render side-by-side comparison bar chart.

    Args:
        comparison: RunComparison with two runs to compare.
        metric: Metric to visualize ("recall", "precision", "ndcg").
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    agg_a = comparison.run_a.aggregates.overall
    agg_b = comparison.run_b.aggregates.overall

    # Get metric data
    if metric == "recall":
        a_data = agg_a.recall_at_k
        b_data = agg_b.recall_at_k
        title = "Recall@K Comparison"
        y_label = "Recall"
    elif metric == "precision":
        a_data = agg_a.precision_at_k
        b_data = agg_b.precision_at_k
        title = "Precision@K Comparison"
        y_label = "Precision"
    elif metric == "ndcg":
        a_data = agg_a.ndcg_at_k
        b_data = agg_b.ndcg_at_k
        title = "NDCG@K Comparison"
        y_label = "NDCG"
    elif metric == "hit_rate":
        a_data = agg_a.hit_rate_at_k
        b_data = agg_b.hit_rate_at_k
        title = "Hit Rate@K Comparison"
        y_label = "Hit Rate"
    else:
        st.error(f"Unknown metric: {metric}")
        return

    k_values = sorted(set(a_data.keys()) | set(b_data.keys()))

    fig = go.Figure()

    # Run A bars
    fig.add_trace(
        go.Bar(
            x=[f"@{k}" for k in k_values],
            y=[a_data.get(k, 0) for k in k_values],
            name=comparison.run_a.summary.display_name,
            text=[f"{a_data.get(k, 0):.3f}" for k in k_values],
            textposition="auto",
        )
    )

    # Run B bars
    fig.add_trace(
        go.Bar(
            x=[f"@{k}" for k in k_values],
            y=[b_data.get(k, 0) for k in k_values],
            name=comparison.run_b.summary.display_name,
            text=[f"{b_data.get(k, 0):.3f}" for k in k_values],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="K",
        yaxis_title=y_label,
        barmode="group",
        height=400,
        yaxis={"range": [0, 1.05]},  # Metrics are 0-1
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    st.plotly_chart(fig, use_container_width=True)


def render_global_metrics_comparison(comparison: RunComparison) -> None:
    """Render comparison of global metrics (MRR, MAP)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    agg_a = comparison.run_a.aggregates.overall
    agg_b = comparison.run_b.aggregates.overall

    metrics = ["MRR", "MAP"]
    a_values = [agg_a.mrr, agg_a.map]
    b_values = [agg_b.mrr, agg_b.map]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=a_values,
            name=comparison.run_a.summary.display_name,
            text=[f"{v:.3f}" for v in a_values],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=b_values,
            name=comparison.run_b.summary.display_name,
            text=[f"{v:.3f}" for v in b_values],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Global Metrics Comparison",
        yaxis_title="Score",
        barmode="group",
        height=350,
        yaxis={"range": [0, 1.05]},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    st.plotly_chart(fig, use_container_width=True)
