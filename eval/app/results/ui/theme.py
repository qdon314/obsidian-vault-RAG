"""
Theme utilities for Streamlit app.

Provides consistent theming across all UI components.
"""

from __future__ import annotations

import streamlit as st


def get_current_theme() -> str:
    """Get the current theme from session state."""
    return st.session_state.get("theme", "dark")


def is_dark_theme() -> bool:
    """Check if dark theme is active."""
    return get_current_theme() == "dark"


def get_plotly_layout() -> dict:
    """Get Plotly layout configuration for current theme."""
    if is_dark_theme():
        return {
            "template": "plotly_dark",
            "paper_bgcolor": "rgba(14, 17, 23, 0)",
            "plot_bgcolor": "rgba(14, 17, 23, 0)",
            "font": {"color": "#fafafa"},
        }
    else:
        return {
            "template": "plotly_white",
            "paper_bgcolor": "rgba(255, 255, 255, 0)",
            "plot_bgcolor": "rgba(255, 255, 255, 0)",
            "font": {"color": "#1a1c23"},
        }


# Theme-aware color palette
CHART_COLORS = {
    "primary": "#4a9eff",    # Blue
    "secondary": "#ff6b6b",  # Coral/Red
    "tertiary": "#51cf66",   # Green
    "quaternary": "#9775fa", # Purple
    "quinary": "#ffa94d",    # Orange
}


def get_chart_colors() -> dict[str, str]:
    """Get the chart color palette.

    Returns consistent colors that work in both themes.
    """
    return CHART_COLORS


def get_metric_color(metric: str) -> str:
    """Get color for a specific metric type."""
    metric_colors = {
        "recall": CHART_COLORS["primary"],
        "precision": CHART_COLORS["secondary"],
        "ndcg": CHART_COLORS["tertiary"],
        "mrr": CHART_COLORS["quaternary"],
        "map": CHART_COLORS["quinary"],
        "quality": CHART_COLORS["tertiary"],
        "latency": CHART_COLORS["secondary"],
    }
    return metric_colors.get(metric.lower(), CHART_COLORS["primary"])
