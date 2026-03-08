# eval/app_v2/ui/pages/verdicts.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.derived.contributors import contributor_queries_for_code
from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.ui.widgets.diagnostic_card import render_diagnostic_card


def render(bundle: RunBundle | None) -> None:
    st.header("Verdict")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    v = bundle.verdict
    if v is None:
        st.warning("No verdict file found for this run. Run `make verdict` to generate one.")
        return

    # SHIP/BLOCK badge
    if v.decision == "SHIP":
        st.success("## ✅ SHIP")
    else:
        st.error("## 🚫 BLOCK")

    st.divider()

    # Failed checks
    raw = v.raw
    st.subheader(f"Threshold checks ({len(raw.checks)} total)")
    for check in raw.checks:
        icon = "✅" if check.passed else "❌"
        delta = f" (baseline: {check.baseline:.3f})" if check.baseline is not None else ""
        st.markdown(
            f"{icon} **{check.name}** — current: `{check.current:.3f}` / threshold: `{check.threshold:.3f}`{delta}"
        )

    st.divider()

    # Contributor queries for each failed check
    failed_names = set(v.failed_check_names)
    if failed_names:
        st.subheader("Contributing queries (worst per failure mode)")
        # Map failed check names to DiagnosticCodes heuristically
        # Exact mapping depends on check name conventions in verdict.py
        # For now, show worst queries regardless of code
        from eval.app_v2.engine.derived.contributors import worst_queries
        worst = worst_queries(bundle.queries, limit=10)
        for aq in worst:
            render_diagnostic_card(aq, show_forensics_link=True)
