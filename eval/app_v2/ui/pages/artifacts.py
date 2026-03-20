from __future__ import annotations

import json

import streamlit as st

from eval.app_v2.engine.domain.models import RunBundle


def render(bundle: RunBundle | None) -> None:
    st.header("Artifacts")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    # Loader warnings
    if bundle.warnings:
        st.subheader(f"Loader warnings ({len(bundle.warnings)})")
        for w in bundle.warnings:
            st.warning(
                f"`{w.code}` — {w.message}" + (f" [{w.artifact_name}]" if w.artifact_name else "")
            )
    else:
        st.success("No loader warnings.")

    st.divider()

    # Raw artifact viewers
    st.subheader("Raw artifacts")
    for name, payload in bundle.raw_artifacts.items():
        with st.expander(f"`{name}`", expanded=False):
            try:
                st.code(json.dumps(payload, indent=2, default=str)[:5000], language="json")
            except Exception:
                st.text(str(payload)[:2000])
