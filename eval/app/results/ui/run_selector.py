"""
Run selection component for the results analyzer.

Provides single and multi-select run selection widgets.
"""

from __future__ import annotations

import streamlit as st

from eval.app.results.domain.models import RunSummary


def render_run_selector(
    runs: list[RunSummary],
    *,
    key: str,
    multi: bool = False,
    label: str | None = None,
) -> list[str] | str | None:
    """Render run selection widget.

    Args:
        runs: List of available run summaries.
        key: Unique key for the widget.
        multi: If True, allow multiple selections.
        label: Optional custom label for the selector.

    Returns:
        For single select: Selected run_id or None.
        For multi select: List of selected run_ids (may be empty).
    """
    if not runs:
        st.warning("No runs available")
        return [] if multi else None

    # Build display options
    def format_run(run: RunSummary) -> str:
        """Format run for display."""
        parts = [run.display_name]

        if run.overall_recall_at_10 is not None:
            parts.append(f"R@10: {run.overall_recall_at_10:.2f}")

        if run.generator_model:
            # Shorten model name
            model = run.generator_model
            if "/" in model:
                model = model.split("/")[-1]
            if len(model) > 20:
                model = model[:17] + "..."
            parts.append(model)

        return " | ".join(parts)

    options = {run.run_id: format_run(run) for run in runs}

    if multi:
        default_label = "Select runs to compare"
        selected = st.multiselect(
            label or default_label,
            options=list(options.keys()),
            format_func=lambda rid: options[rid],
            key=key,
        )
        return selected if selected else []
    else:
        default_label = "Select a run"
        # Add None option for single select
        selected = st.selectbox(
            label or default_label,
            options=[None] + list(options.keys()),
            format_func=lambda rid: "-- Select a run --" if rid is None else options[rid],
            key=key,
        )
        return selected


def render_run_info_card(run: RunSummary) -> None:
    """Render a compact info card for a run summary."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Timestamp", run.timestamp.strftime("%Y-%m-%d %H:%M"))
        if run.generator_model:
            st.caption(f"Generator: {run.generator_model}")

    with col2:
        st.metric("Queries", run.num_queries)
        if run.embedder_model:
            st.caption(f"Embedder: {run.embedder_model}")

    with col3:
        if run.overall_recall_at_10 is not None:
            st.metric("Recall@10", f"{run.overall_recall_at_10:.3f}")
        if run.reranker_name:
            st.caption(f"Reranker: {run.reranker_name}")
