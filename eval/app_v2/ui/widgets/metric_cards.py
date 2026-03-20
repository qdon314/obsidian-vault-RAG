from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import Severity
from eval.app_v2.engine.domain.models import RunHealthSummary

_SEVERITY_COLORS = {
    Severity.OK: "#2ecc71",
    Severity.MINOR: "#f39c12",
    Severity.MODERATE: "#e67e22",
    Severity.CRITICAL: "#e74c3c",
}

_FMT_PCT = lambda v: f"{v:.1%}" if v is not None else "—"  # noqa: E731
_FMT_F2 = lambda v: f"{v:.2f}" if v is not None else "—"  # noqa: E731
_FMT_F3 = lambda v: f"{v:.3f}" if v is not None else "—"  # noqa: E731
_FMT_MS = lambda v: f"{v:.0f} ms" if v is not None else "—"  # noqa: E731
_FMT_0_5 = lambda v: f"{v:.2f} / 5" if v is not None else "—"  # noqa: E731
_FMT_RATIO = lambda v: f"{v:.2f} x " if v is not None else "N/A"  # noqa: E731


def render_kpi_cards(health: RunHealthSummary) -> None:
    """Render headline KPI metric cards from a RunHealthSummary.

    Displays up to five rows of metrics, each shown conditionally:
      Row 1 — Core retrieval (always)
      Row 2 — Extended retrieval (when tiered metrics present)
      Row 3 — Answer quality (when answer quality present)
      Row 4 — Safety & latency (when present)
      Row 5 — Compression efficiency (when compression adapter was used)
    """
    # ── Row 1: Core retrieval (always shown) ─────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@10", _FMT_PCT(health.headline_recall_at_10))
    c2.metric("NDCG@10", _FMT_PCT(health.headline_ndcg_at_10))
    c3.metric("MRR", _FMT_F3(health.headline_mrr))
    c4.metric("MAP", _FMT_F3(health.headline_map))

    # ── Row 2: Extended retrieval (shown when at least one value was computed) ──
    has_extended = any(
        [
            health.headline_hit_rate_at_10 is not None,
            health.headline_precision_at_10 is not None,
            health.headline_critical_recall_at_10 is not None,
            health.headline_weighted_recall_at_10 is not None,
        ]
    )
    if has_extended:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(
            "Hit Rate@10",
            _FMT_PCT(health.headline_hit_rate_at_10),
            help="Fraction of queries where at least one relevant chunk was retrieved in top-10",
        )
        d2.metric("Precision@10", _FMT_PCT(health.headline_precision_at_10))
        d3.metric(
            "Critical Recall@10",
            _FMT_PCT(health.headline_critical_recall_at_10),
            help="Recall computed only against critical-tier relevant chunks",
        )
        d4.metric(
            "Weighted Recall@10",
            _FMT_PCT(health.headline_weighted_recall_at_10),
            help="Tiered recall: critical x 1.0, supporting x 0.5, context x 0.2",
        )

    # ── Row 3: Answer quality (shown when answer quality data present) ────────
    has_quality = any(
        [
            health.avg_quality_score is not None,
            health.median_quality_score is not None,
            health.avg_correctness is not None,
            health.avg_hallucination_severity is not None,
        ]
    )
    if has_quality:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(
            "Avg Quality",
            _FMT_F2(health.avg_quality_score),
            help="Composite quality score (0 - 1): correctness x 0.45 + hallucination x 0.30 + citation_coverage x 0.15 + …",
        )
        e2.metric("Median Quality", _FMT_F2(health.median_quality_score))
        e3.metric(
            "Avg Correctness",
            _FMT_0_5(health.avg_correctness),
            help="LLM judge correctness score (0 - 5)",
        )
        e4.metric(
            "Avg Hallucination",
            _FMT_0_5(health.avg_hallucination_severity),
            help="Hallucination severity score (0=none, 5=fatal). Lower is better.",
        )

    # ── Row 4: Safety & latency (shown when either is present) ───────────────
    has_safety = any(
        [
            health.evidence_bounded_rate is not None,
            health.hallucinated_on_unanswerable_rate is not None,
            health.avg_citation_coverage is not None,
        ]
    )
    has_latency = any(
        [
            health.p50_latency_ms is not None,
            health.p95_latency_ms is not None,
        ]
    )
    if has_safety or has_latency:
        f1, f2, f3, f4 = st.columns(4)
        f1.metric(
            "Evidence Bounded",
            _FMT_PCT(health.evidence_bounded_rate),
            help="Fraction of answers where every claim is supported by retrieved context",
        )
        f2.metric(
            "Halluc. on Unanswerable",
            _FMT_PCT(health.hallucinated_on_unanswerable_rate),
            help="Of queries where context was insufficient, how often the model hallucinated instead of abstaining",
        )
        f3.metric(
            "P50 Latency",
            _FMT_MS(health.p50_latency_ms),
        )
        f4.metric(
            "P95 Latency",
            _FMT_MS(health.p95_latency_ms),
        )

    # ── Row 5: Compression efficiency (shown when compression adapter was used) ──
    if health.avg_compression_ratio is not None:
        g1, _g2, _g3, _g4 = st.columns(4)
        g1.metric(
            "Avg Compression Ratio",
            _FMT_RATIO(health.avg_compression_ratio),
            help=(
                "Average ratio of compressed tokens to original tokens (lower = more compressed). "
                "N/A when no compression adapter was used."
            ),
        )


def render_severity_bar(health: RunHealthSummary) -> None:
    """Horizontal breakdown: OK | MINOR | MODERATE | CRITICAL counts."""
    total = sum(health.severity_counts.values()) or 1
    cols = st.columns(4)
    for col, sev in zip(
        cols, [Severity.OK, Severity.MINOR, Severity.MODERATE, Severity.CRITICAL], strict=True
    ):
        n = health.severity_counts.get(sev, 0)
        col.markdown(
            f"<div style='background:{_SEVERITY_COLORS[sev]};padding:8px;border-radius:4px;"
            f"text-align:center'><b>{sev.upper()}</b><br>{n} ({n / total:.0%})</div>",
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
