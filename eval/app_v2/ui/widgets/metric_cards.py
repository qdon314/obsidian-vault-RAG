from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import Severity
from eval.app_v2.engine.domain.models import RunHealthSummary

_SEVERITY_COLORS = {
    Severity.OK:       "#2ecc71",
    Severity.MINOR:    "#f39c12",
    Severity.MODERATE: "#e67e22",
    Severity.CRITICAL: "#e74c3c",
}


def render_kpi_cards(health: RunHealthSummary) -> None:
    """Render headline KPI metric cards from a RunHealthSummary."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@10", f"{health.headline_recall_at_10:.1%}")
    c2.metric("NDCG@10",   f"{health.headline_ndcg_at_10:.1%}")
    c3.metric(
        "Avg Quality",
        f"{health.avg_quality_score:.2f}" if health.avg_quality_score is not None else "—",
    )
    c4.metric(
        "Avg Latency",
        f"{health.avg_latency_ms:.0f} ms" if health.avg_latency_ms is not None else "—",
    )


def render_severity_bar(health: RunHealthSummary) -> None:
    """Horizontal breakdown: OK | MINOR | MODERATE | CRITICAL counts."""
    total = sum(health.severity_counts.values()) or 1
    cols = st.columns(4)
    for col, sev in zip(cols, [Severity.OK, Severity.MINOR, Severity.MODERATE, Severity.CRITICAL], strict=True):
        n = health.severity_counts.get(sev, 0)
        col.markdown(
            f"<div style='background:{_SEVERITY_COLORS[sev]};padding:8px;border-radius:4px;"
            f"text-align:center'><b>{sev.upper()}</b><br>{n} ({n/total:.0%})</div>",
            unsafe_allow_html=True,
        )


def render_dominant_failure_banner(health: RunHealthSummary) -> None:
    """Show the dominant failure mode as a colored banner."""
    if health.dominant_failure_mode is None:
        st.success("No dominant failure mode — run looks healthy.")
        return
    st.error(
        f"**Dominant failure:** `{health.dominant_failure_mode}` — "
        f"{health.dominant_failure_summary or ''} "
        f"({health.diagnostic_counts.get(health.dominant_failure_mode, 0)} queries)"
    )
