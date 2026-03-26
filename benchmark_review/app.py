"""Benchmark review app.

Launch with:
    make benchmark-review
or:
    ./scripts/py -m streamlit run benchmark_review/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from benchmark_review.engine.loader import load_run
from benchmark_review.ui import progress_bar, record_detail, record_list, run_selector

st.set_page_config(
    page_title="Benchmark Review",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Benchmark Review")

run_dir: Path | None = run_selector.render()
if run_dir is None:
    st.stop()

st.divider()

# Load records fresh on every rerun (fast at JSONL scale; picks up sidecar changes)
records = load_run(run_dir)

progress_bar.render(records)
st.divider()

left, right = st.columns([1, 2])

with left:
    selected_id = record_list.render(records)
    if selected_id:
        st.session_state["selected_id"] = selected_id

with right:
    selected_id = st.session_state.get("selected_id")
    if selected_id:
        selected_rec = next((r for r in records if r.candidate_id == selected_id), None)
        if selected_rec:
            record_detail.render(selected_rec, run_dir, records)
    else:
        st.info("Select a candidate from the list to begin reviewing.")
