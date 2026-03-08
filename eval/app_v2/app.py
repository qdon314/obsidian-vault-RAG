"""Entry point: streamlit run eval/app_v2/app.py"""
from __future__ import annotations

import streamlit as st

from eval.app_v2.ui.app import DEFAULT_RUNS_DIR, discover_runs, load_bundle, run_selector_widget
from eval.app_v2.ui.pages.artifacts import render as artifacts_page
from eval.app_v2.ui.pages.forensics import render as forensics_page
from eval.app_v2.ui.pages.triage import render as triage_page

PAGES = {
    "Triage": triage_page,
    "Forensics": forensics_page,
    "Artifacts": artifacts_page,
}


def main() -> None:
    st.set_page_config(page_title="Results Analyzer v2", layout="wide")
    runs = discover_runs(DEFAULT_RUNS_DIR)

    with st.sidebar:
        st.title("Results Analyzer v2")
        page = st.radio("Page", list(PAGES.keys()))
        selected = run_selector_widget(runs)

    bundle = None
    if selected:
        name, run_dir = selected
        bundle = load_bundle(name, str(run_dir))

    PAGES[page](bundle)


if __name__ == "__main__":
    main()
