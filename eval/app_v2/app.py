"""Entry point: streamlit run eval/app_v2/app.py"""
from __future__ import annotations

import streamlit as st

from eval.app_v2.ui.app import DEFAULT_RUNS_DIR, discover_runs, load_bundle, run_selector_widget
from eval.app_v2.ui.pages.artifacts import render as artifacts_page
from eval.app_v2.ui.pages.compare import render as compare_page
from eval.app_v2.ui.pages.forensics import render as forensics_page
from eval.app_v2.ui.pages.triage import render as triage_page
from eval.app_v2.ui.pages.verdicts import render as verdicts_page

PAGES = {
    "Triage": triage_page,
    "Forensics": forensics_page,
    "Artifacts": artifacts_page,
    "Compare": compare_page,
    "Verdict": verdicts_page,
}


def main() -> None:
    st.set_page_config(page_title="Results Analyzer v2", layout="wide")
    runs = discover_runs(DEFAULT_RUNS_DIR)

    with st.sidebar:
        st.title("Results Analyzer v2")
        page = st.radio("Page", list(PAGES.keys()))
        selected = run_selector_widget(runs, key="run_a", label="Run (A)")
        selected_b = run_selector_widget(runs, key="run_b", label="Compare to (B)")

    bundle = None
    if selected:
        name, run_dir = selected
        bundle = load_bundle(name, str(run_dir))

    bundle_b = None
    if selected_b:
        name_b, run_dir_b = selected_b
        bundle_b = load_bundle(name_b, str(run_dir_b))

    if page == "Compare":
        compare_page(bundle, bundle_b)
    else:
        PAGES[page](bundle)


if __name__ == "__main__":
    main()
