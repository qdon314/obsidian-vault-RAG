"""Entry point: streamlit run eval/app_v2/app.py

S3 mode: set EVAL_S3_BUCKET and EVAL_S3_PREFIX env vars to load runs from S3.
Runs are cached locally in EVAL_S3_CACHE_DIR (default: ~/.cache/eval-runs).
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.ui.app import (
    DEFAULT_RUNS_DIR,
    discover_runs,
    discover_runs_s3,
    load_bundle,
    load_bundle_from_s3,
    run_selector_widget,
)
from eval.app_v2.ui.pages.artifacts import render as artifacts_page
from eval.app_v2.ui.pages.cohorts import render as cohorts_page
from eval.app_v2.ui.pages.compare import render as compare_page
from eval.app_v2.ui.pages.forensics import render as forensics_page
from eval.app_v2.ui.pages.trends import render as trends_page
from eval.app_v2.ui.pages.triage import render as triage_page
from eval.app_v2.ui.pages.verdicts import render as verdicts_page

SINGLE_RUN_PAGES = {
    "Triage": triage_page,
    "Forensics": forensics_page,
    "Artifacts": artifacts_page,
    "Cohorts": cohorts_page,
    "Verdict": verdicts_page,
}

ALL_PAGE_NAMES = [*list(SINGLE_RUN_PAGES), "Compare", "Trends"]

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "eval-runs"


def _s3_config() -> tuple[str, str, Path] | None:
    """Return (bucket, prefix, cache_dir) if S3 env vars are set, else None."""
    bucket = os.getenv("EVAL_S3_BUCKET", "obsidian-rag-corpus")
    prefix = os.getenv("EVAL_S3_PREFIX", "eval/runs")
    if not bucket:
        return None
    cache_dir = Path(os.getenv("EVAL_S3_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))
    return bucket, prefix, cache_dir


def _load(selected: tuple[str, Path] | None, s3: tuple[str, str, Path] | None) -> RunBundle | None:
    if not selected:
        return None
    name, local_path = selected
    if s3:
        bucket, prefix, cache_dir = s3
        s3_prefix = f"{prefix.rstrip('/')}/{name}"
        return load_bundle_from_s3(name, bucket, s3_prefix, str(cache_dir))
    return load_bundle(name, str(local_path))


def main() -> None:
    st.set_page_config(page_title="Results Analyzer v2", layout="wide")

    s3 = _s3_config()
    if s3:
        import boto3  # type: ignore[import-untyped]
        bucket, prefix, cache_dir = s3
        client = boto3.client("s3")
        runs = discover_runs_s3(client, bucket, prefix, cache_dir)
    else:
        runs = discover_runs(DEFAULT_RUNS_DIR)

    with st.sidebar:
        st.title("Results Analyzer v2")
        if s3:
            st.caption(f"Source: s3://{s3[0]}/{s3[1]}")
        page = st.radio("Page", ALL_PAGE_NAMES)
        selected = run_selector_widget(runs) if page != "Trends" else None
        selected_b = run_selector_widget(runs, key="run_b", label="Compare to (B)") if page == "Compare" else None

    if page == "Trends":
        trends_page(runs)
    elif page == "Compare":
        compare_page(_load(selected, s3), _load(selected_b, s3))
    else:
        SINGLE_RUN_PAGES[page](_load(selected, s3))


if __name__ == "__main__":
    main()
