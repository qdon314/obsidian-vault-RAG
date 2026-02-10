"""
Trend chart component for visualizing metrics over time.

Uses Plotly for interactive line charts showing metric trends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from eval.app.results.domain.models import TrendAnalysis


def render_trend_chart(
    trend: TrendAnalysis,
    metric: str = "recall",
    k: int = 10,
) -> None:
    """Render multi-run trend line chart.

    Args:
        trend: TrendAnalysis with time series data.
        metric: Metric to plot ("recall", "critical_recall", "weighted_recall",
                "critical_hit_rate", "precision", "ndcg", "mrr", "map",
                "quality", "latency").
        k: K value for @k metrics.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    # Format timestamps for x-axis
    x_labels = [ts.strftime("%m/%d %H:%M") for ts in trend.timestamps]

    # Get y values based on metric
    if metric == "recall":
        y_values = trend.recall_series.get(k, ())
        title = f"Recall@{k} Trend"
        y_label = "Recall"
    elif metric == "critical_recall":
        y_values = trend.critical_recall_series.get(k, ())
        title = f"Critical Recall@{k} Trend"
        y_label = "Critical Recall"
    elif metric == "weighted_recall":
        y_values = trend.weighted_recall_series.get(k, ())
        title = f"Weighted Recall@{k} Trend"
        y_label = "Weighted Recall"
    elif metric == "critical_hit_rate":
        y_values = trend.critical_hit_rate_series.get(k, ())
        title = f"Critical Hit Rate@{k} Trend"
        y_label = "Critical Hit Rate"
    elif metric == "precision":
        y_values = trend.precision_series.get(k, ())
        title = f"Precision@{k} Trend"
        y_label = "Precision"
    elif metric == "ndcg":
        y_values = trend.ndcg_series.get(k, ())
        title = f"NDCG@{k} Trend"
        y_label = "NDCG"
    elif metric == "mrr":
        y_values = trend.mrr_series
        title = "MRR Trend"
        y_label = "MRR"
    elif metric == "map":
        y_values = trend.map_series
        title = "MAP Trend"
        y_label = "MAP"
    elif metric == "quality":
        y_values = trend.quality_series
        title = "Answer Quality Score Trend"
        y_label = "Quality Score"
    elif metric == "latency":
        y_values = trend.latency_series
        title = "Average Latency Trend"
        y_label = "Latency (ms)"
    else:
        st.error(f"Unknown metric: {metric}")
        return

    if not y_values:
        st.info(f"No data available for {metric}")
        return

    # Filter out None values for quality/latency
    valid_points = [(x, y) for x, y in zip(x_labels, y_values, strict=False) if y is not None]

    if not valid_points:
        st.info(f"No valid data points for {metric}")
        return

    x_valid = [p[0] for p in valid_points]
    y_valid = [p[1] for p in valid_points]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_valid,
            y=y_valid,
            mode="lines+markers",
            name=metric.title(),
            line={"width": 2},
            marker={"size": 8},
            text=[f"{v:.3f}" if metric != "latency" else f"{v:.0f}" for v in y_valid],
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        )
    )

    y_range = None if metric == "latency" else [0, 1.05]

    fig.update_layout(
        title=title,
        xaxis_title="Run Timestamp",
        yaxis_title=y_label,
        height=400,
        yaxis={"range": y_range} if y_range else {},
        xaxis={"tickangle": -45},
    )

    st.plotly_chart(fig, use_container_width=True)


def render_multi_metric_trend(
    trend: TrendAnalysis,
    metrics: list[str] | None = None,
    k: int = 10,
) -> None:
    """Render multiple metrics on the same trend chart.

    Args:
        trend: TrendAnalysis with time series data.
        metrics: List of metrics to plot. Defaults to recall, precision, ndcg.
        k: K value for @k metrics.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    if metrics is None:
        metrics = ["recall", "precision", "ndcg"]

    x_labels = [ts.strftime("%m/%d %H:%M") for ts in trend.timestamps]

    fig = go.Figure()

    for metric in metrics:
        if metric == "recall":
            y_values = trend.recall_series.get(k, ())
            name = f"Recall@{k}"
        elif metric == "critical_recall":
            y_values = trend.critical_recall_series.get(k, ())
            name = f"Critical Recall@{k}"
        elif metric == "weighted_recall":
            y_values = trend.weighted_recall_series.get(k, ())
            name = f"Weighted Recall@{k}"
        elif metric == "critical_hit_rate":
            y_values = trend.critical_hit_rate_series.get(k, ())
            name = f"Critical Hit Rate@{k}"
        elif metric == "precision":
            y_values = trend.precision_series.get(k, ())
            name = f"Precision@{k}"
        elif metric == "ndcg":
            y_values = trend.ndcg_series.get(k, ())
            name = f"NDCG@{k}"
        elif metric == "mrr":
            y_values = trend.mrr_series
            name = "MRR"
        elif metric == "map":
            y_values = trend.map_series
            name = "MAP"
        else:
            continue

        if not y_values:
            continue

        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y_values,
                mode="lines+markers",
                name=name,
                line={"width": 2},
                marker={"size": 6},
            )
        )

    fig.update_layout(
        title=f"Metrics Trend (k={k})",
        xaxis_title="Run Timestamp",
        yaxis_title="Score",
        height=450,
        yaxis={"range": [0, 1.05]},
        xaxis={"tickangle": -45},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    st.plotly_chart(fig, use_container_width=True)
