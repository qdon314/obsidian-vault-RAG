from __future__ import annotations

from pathlib import Path

import streamlit as st

_BENCHMARK_RUNS_DIR = Path("benchmark_runs")


def render() -> Path | None:
    """Render run + reviewer selector. Returns selected run_dir or None."""
    run_dirs = (
        sorted(
            [d for d in _BENCHMARK_RUNS_DIR.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if _BENCHMARK_RUNS_DIR.exists()
        else []
    )

    if not run_dirs:
        st.error(f"No benchmark runs found in `{_BENCHMARK_RUNS_DIR}/`.")
        return None

    col1, col2 = st.columns([2, 1])
    with col1:
        run_name = st.selectbox(
            "Benchmark run",
            [d.name for d in run_dirs],
            key="selected_run_name",
        )
    with col2:
        reviewer_id = st.text_input(
            "Reviewer ID",
            value=st.session_state.get("reviewer_id", ""),
            key="reviewer_id_input",
            placeholder="e.g. jsmith",
        )
        st.session_state["reviewer_id"] = reviewer_id

    return _BENCHMARK_RUNS_DIR / run_name
